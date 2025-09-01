# app/search.py
from __future__ import annotations

import os
import logging
from typing import List, Optional, Sequence, Dict, Any, Set
import psycopg2
import psycopg2.extras

from app.rerankers import Candidate, CrossEncoderReranker, LLMJudgeReranker

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DB_DSN_DEFAULT = "postgresql://rag:ragpwd@pg:5432/ragdb"


# ----------------------------- Helpers ---------------------------------------

def _connect(dsn: Optional[str]):
    return psycopg2.connect(dsn or os.getenv("DB_DSN", DB_DSN_DEFAULT))

def _table_cols(conn, table: str) -> Set[str]:
    q = """
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(q, (table,))
        return {r[0] for r in cur.fetchall()}

def _has_table(conn, table: str) -> bool:
    q = """
      SELECT 1
      FROM information_schema.tables
      WHERE table_schema='public' AND table_name=%s
      LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(q, (table,))
        return cur.fetchone() is not None


# ----------------------------- Reranking -------------------------------------

def _to_candidates(rows: List[Dict[str, Any]]) -> List[Candidate]:
    # Kept for convenience if you want to call rerankers from here.
    return [
        Candidate(
            chunk_id=r.get("id"),
            repo=r.get("repo_name"),
            path=r.get("path"),
            start_line=r.get("start_line"),
            end_line=r.get("end_line"),
            content=r.get("content"),
            symbol=r.get("symbol_name"),
            score=float(r.get("hybrid_score") or r.get("score") or 0.0),
        )
        for r in rows
    ]

def rerank_candidates(
    query: str,
    rows: List[Dict[str, Any]],
    rerank: bool,
    reranker_kind: str,
    top_rerank: int,
) -> List[Candidate]:
    cands = _to_candidates(rows)
    if not cands or not rerank:
        return cands[:top_rerank]
    rr = LLMJudgeReranker() if reranker_kind == "judge" else CrossEncoderReranker()
    return rr.rerank(query, cands, top_k=top_rerank)


# ----------------------------- Hybrid Search ---------------------------------

def search_hybrid(
    query: str,
    *,
    language: Optional[str] = None,   # back-compat with your api.py
    topn: int = 20,                   # back-compat; aliased to LIMIT
    w_vec: float = 0.6,
    w_lex: float = 0.3,
    w_sym: float = 0.1,
    repo: Optional[str] = None,       # optional filter
    path_glob: Optional[str] = None,  # optional filter
    query_embedding: Optional[Sequence[float]] = None,  # optional vector
    conn: Optional[psycopg2.extensions.connection] = None,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Schema-aware hybrid search that returns columns expected by api.py::_to_candidates:
      id, repo_name, path, start_line, end_line, symbol_name, content, hybrid_score

    Adapts to:
      - repo_name/repo on code_files or code_chunks (or neither)
      - path on code_files or code_chunks
      - symbol_simple/symbol_name/symbol
      - embedding present/absent
    """
    lang = (language or "english").lower()
    if lang not in {"english", "simple"}:
        lang = "english"
    top_k = int(topn)

    created_conn = None
    try:
        if conn is None:
            created_conn = _connect(dsn)
            conn = created_conn

        has_files = _has_table(conn, "code_files")
        cols_c = _table_cols(conn, "code_chunks")
        cols_f = _table_cols(conn, "code_files") if has_files else set()

        # Column existence
        has_c_file_id = "file_id" in cols_c
        has_f_id = "id" in cols_f
        do_join_files = has_files and has_f_id and has_c_file_id

        # repo_name choices (prefer repo_name, fallback repo, else NULL)
        if do_join_files and "repo_name" in cols_f:
            repo_expr = "f.repo_name"
        elif do_join_files and "repo" in cols_f:
            repo_expr = "f.repo"
        elif "repo_name" in cols_c:
            repo_expr = "c.repo_name"
        elif "repo" in cols_c:
            repo_expr = "c.repo"
        else:
            repo_expr = "NULL::text"

        # path choices (prefer path, fallback file_path, else NULL)
        if do_join_files and "path" in cols_f:
            path_expr = "f.path"
        elif do_join_files and "file_path" in cols_f:
            path_expr = "f.file_path"
        elif "path" in cols_c:
            path_expr = "c.path"
        elif "file_path" in cols_c:
            path_expr = "c.file_path"
        else:
            path_expr = "NULL::text"

        # symbol choices (prefer symbol_simple, then symbol_name, then symbol, else NULL)
        if "symbol_simple" in cols_c and "symbol_name" in cols_c:
            symbol_expr = "COALESCE(c.symbol_simple, c.symbol_name)"
        elif "symbol_simple" in cols_c and "symbol" in cols_c:
            symbol_expr = "COALESCE(c.symbol_simple, c.symbol)"
        elif "symbol_name" in cols_c:
            symbol_expr = "c.symbol_name"
        elif "symbol" in cols_c:
            symbol_expr = "c.symbol"
        else:
            symbol_expr = "NULL::text"

        has_c_embed = "embedding" in cols_c  # default column name for pgvector

        # WHERE filters (only apply if column exists)
        where_clauses = []
        params: Dict[str, Any] = {
            "q": query,
            "top_k": top_k,
            "w_vec": w_vec,
            "w_lex": w_lex,
            "w_sym": w_sym,
        }

        if repo:
            if do_join_files and ("repo_name" in cols_f or "repo" in cols_f):
                col = "repo_name" if "repo_name" in cols_f else "repo"
                where_clauses.append(f"f.{col} = %(repo)s")
                params["repo"] = repo
            elif ("repo_name" in cols_c or "repo" in cols_c):
                col = "repo_name" if "repo_name" in cols_c else "repo"
                where_clauses.append(f"c.{col} = %(repo)s")
                params["repo"] = repo
            else:
                log.info("search_hybrid: repo filter ignored (no repo column)")

        if path_glob:
            if do_join_files and ("path" in cols_f or "file_path" in cols_f):
                col = "path" if "path" in cols_f else "file_path"
                where_clauses.append(f"f.{col} LIKE %(path_glob)s")
                params["path_glob"] = path_glob
            elif ("path" in cols_c or "file_path" in cols_c):
                col = "path" if "path" in cols_c else "file_path"
                where_clauses.append(f"c.{col} LIKE %(path_glob)s")
                params["path_glob"] = path_glob
            else:
                log.info("search_hybrid: path_glob filter ignored (no path column)")

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Lexical + symbol component
        lex_sym = f"""
            %(w_lex)s * COALESCE(
                ts_rank(
                    to_tsvector('{lang}', c.content),
                    plainto_tsquery('{lang}', %(q)s)
                ),
                0
            )
            + %(w_sym)s * CASE
                WHEN LOWER({symbol_expr}) LIKE LOWER('%%' || %(q)s || '%%')
                THEN 1.0 ELSE 0.0
              END
        """

        # Optional vector term
        vec_term = ""
        if has_c_embed and query_embedding is not None:
            vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
            params["embed"] = vector_literal
            vec_term = " + %(w_vec)s * (1 - (c.embedding <=> %(embed)s::vector))"

        # Build FROM/JOIN
        from_sql = "FROM code_chunks c"
        if do_join_files:
            from_sql += " JOIN code_files f ON f.id = c.file_id"

        sql = f"""
            SELECT
              c.id AS id,
              {repo_expr} AS repo_name,
              {path_expr} AS path,
              c.start_line,
              c.end_line,
              c.content,
              {symbol_expr} AS symbol_name,
              ({lex_sym}{vec_term}) AS hybrid_score
            {from_sql}
            {where_sql}
            ORDER BY hybrid_score DESC
            LIMIT %(top_k)s
        """

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            log.info(
                "search_hybrid: rows=%d join_files=%s embed=%s",
                len(rows), do_join_files, bool(vec_term)
            )
            return list(rows)

    finally:
        if created_conn is not None:
            created_conn.close()
