# app/search.py
from __future__ import annotations

import os
import re
import sys
import logging
from typing import List, Optional, Sequence, Dict, Any, Set, Tuple

import psycopg2
import psycopg2.extras
from pathlib import Path

# --- make local modules importable (so we can `from search import search_hybrid`)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


from rerankers import Candidate, CrossEncoderReranker, LLMJudgeReranker

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ------------------------------ DB helpers -----------------------------------

def _connect(dsn: Optional[str]) -> psycopg2.extensions.connection:
    dsn = dsn or os.getenv("DB_DSN") or os.getenv("DB_DSN_DEFAULT") or "postgresql://rag:ragpwd@pg:5432/ragdb"
    return psycopg2.connect(dsn)

def _table_cols(conn: psycopg2.extensions.connection, table: str) -> Set[str]:
    sql = """
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        return {r[0] for r in cur.fetchall()}

def _has_table(conn: psycopg2.extensions.connection, table: str) -> bool:
    sql = """
      SELECT 1
      FROM information_schema.tables
      WHERE table_schema='public' AND table_name=%s
      LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        return cur.fetchone() is not None

# ------------------------------ Soft-bonus knobs ------------------------------
# Added to the existing hybrid score; they never hard-filter, only nudge ordering.
W_FN   = 1.20  # filename (basename) bonus — strong enough to beat vector-only hits
W_EXT  = 0.40  # extension affinity bonus (js<->jsx, ts<->tsx, etc.)
W_TOK  = 0.25  # tiny content token bonus

# Extension affinity groups (expand freely)
EXT_AFFINITY: Dict[str, List[str]] = {
    "js":  ["js", "jsx", "mjs", "cjs"],
    "jsx": ["jsx", "js"],
    "ts":  ["ts", "tsx"],
    "tsx": ["tsx", "ts"],
    "py":  ["py"],
    "java":["java"],
    "rb":  ["rb"],
    "go":  ["go"],
    "kt":  ["kt", "kts"],
    "cs":  ["cs"],
    "cpp": ["cpp","cc","cxx","hpp","hh","hxx"],
    "c":   ["c","h"],
    "rs":  ["rs"],
    "php": ["php"],
    "scala":["scala","sc"],
}

LANG_DEFAULT = {"english", "simple"}

def _split_filename(path_like: str) -> Tuple[str, Optional[str]]:
    """Return (basename_without_ext, ext_without_dot) for 'dir/Foo.tsx'."""
    base = path_like.rsplit("/", 1)[-1]
    if "." in base:
        stem, ext = base.rsplit(".", 1)
        return stem, ext.lower()
    return base, None

def _extract_filename_hint(q: str) -> Optional[str]:
    """
    Extract a filename-like token from the query. Handles names with dots/slashes.
    Examples: 'summarize routes.tsx', 'open src/app.py', 'Person.java'
    """
    m = re.search(r'([A-Za-z0-9._/\-]+?\.[A-Za-z0-9]+)\b', q)
    return m.group(1) if m else None

def _preferred_exts(ext: Optional[str]) -> List[str]:
    if not ext:
        return []
    return EXT_AFFINITY.get(ext.lower(), [ext.lower()])

# Optional: basic language→extensions hint (used if api passes code_language)
CODE_LANG_HINTS: Dict[str, List[str]] = {
    "java": ["java"],
    "python": ["py"],
    "py": ["py"],
    "javascript": ["js","jsx","mjs","cjs"],
    "js": ["js","jsx","mjs","cjs"],
    "typescript": ["ts","tsx"],
    "ts": ["ts","tsx"],
    "go": ["go"], "golang": ["go"],
    "ruby": ["rb"],
    "rust": ["rs"],
    "csharp": ["cs"], "c#": ["cs"],
    "c++": ["cpp","cc","cxx","hpp","hh","hxx"], "cpp": ["cpp","cc","cxx","hpp","hh","hxx"],
    "c": ["c","h"],
    "kotlin": ["kt","kts"],
    "scala": ["scala","sc"],
    "php": ["php"],
}

# ------------------------------ Candidate utils -------------------------------

def _to_candidates(rows: List[Dict[str, Any]]) -> List[Candidate]:
    out: List[Candidate] = []
    for r in rows:
        out.append(
            Candidate(
                chunk_id=r["id"],
                repo=r.get("repo_name"),  # may be NULL in your schema
                path=r["path"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                content=r["content"],
                symbol=r.get("symbol_name"),
                score=float(r.get("hybrid_score") or r.get("score") or 0.0),
            )
        )
    return out

def rerank_candidates(
    q: str,
    rows: List[Dict[str, Any]],
    enabled: bool,
    kind: str = "cross",
    top_k: int = 20,
) -> List[Candidate]:
    if not enabled:
        return _to_candidates(rows)
    cands = _to_candidates(rows)
    rer = LLMJudgeReranker() if kind == "judge" else CrossEncoderReranker()
    return rer.rerank(q, cands, top_k=top_k)

# ------------------------------ Hybrid search ---------------------------------

def search_hybrid(
    query: str,
    *,
    language: Optional[str] = None,          # tsvector dict: english|simple
    code_language: Optional[str] = None,     # e.g., 'java', 'ts', 'py'
    topn: int = 20,
    w_vec: float = 0.6,
    w_lex: float = 0.3,
    w_sym: float = 0.1,
    repo: Optional[str] = None,              # optional filter if present in schema
    path_glob: Optional[str] = None,         # optional LIKE filter provided by caller
    query_embedding: Optional[Sequence[float]] = None,
    conn: Optional[psycopg2.extensions.connection] = None,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: vector + lexical + symbol, plus generic transparent soft-bonuses
    (filename + extension affinity + tiny content token) and pre-sort flags so that
    filename/extension hits bubble above unrelated chunks without hard filtering.

    Returns rows with: id, repo_name (nullable), path, start_line, end_line, content,
    symbol_name, hybrid_score, __fn_hit, __ext_hit.
    """
    lang = (language or "english").lower()
    if lang not in LANG_DEFAULT:
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

        # repo expression (tolerant if your schema lacks it)
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

        # path expression
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

        # symbol expression
        if "symbol_name" in cols_c:
            symbol_expr = "c.symbol_name"
        elif "symbol_simple" in cols_c:
            symbol_expr = "c.symbol_simple"
        elif "symbol" in cols_c:
            symbol_expr = "c.symbol"
        else:
            symbol_expr = "NULL::text"

        # embedding presence
        has_c_embed = "embedding" in cols_c

        # WHERE assembly
        where_clauses: List[str] = []
        params: Dict[str, Any] = {
            "q": query,
            "top_k": top_k,
            "w_vec": w_vec,
            "w_lex": w_lex,
            "w_sym": w_sym,
        }

        # repo filter (if schema supports)
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

        # optional LIKE filter from caller
        if path_glob:
            if do_join_files and ("path" in cols_f or "file_path" in cols_f):
                col = "path" if "path" in cols_f else "file_path"
                where_clauses.append(f"f.{col} LIKE %(path_glob)s")
                params["path_glob"] = path_glob
            elif ("path" in cols_c or "file_path" in cols_c):
                col = "path" if "path" in cols_c else "file_path"
                where_clauses.append(f"c.{col} LIKE %(path_glob)s")
                params["path_glob"] = path_glob

        # optional code_language hint → mild narrowing by extension (still not hard-only)
        if code_language:
            exts = CODE_LANG_HINTS.get(code_language.lower(), [])
            if exts:
                params["__lang_ext_regex"] = r'\.(' + "|".join(re.escape(e) for e in exts) + r')$'
                if do_join_files and ("path" in cols_f or "file_path" in cols_f):
                    pcol = "path" if "path" in cols_f else "file_path"
                    where_clauses.append(f"LOWER(f.{pcol}) ~ LOWER(%(__lang_ext_regex)s)")
                elif ("path" in cols_c or "file_path" in cols_c):
                    pcol = "path" if "path" in cols_c else "file_path"
                    where_clauses.append(f"LOWER(c.{pcol}) ~ LOWER(%(__lang_ext_regex)s)")

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

        # ----------------------- Generic soft bonuses + flags ------------------
        filename_hint = _extract_filename_hint(query)
        fn_bonus_sql = ""
        ext_bonus_sql = ""
        tok_bonus_sql = ""
        fn_flag_sql = "0"
        ext_flag_sql = "0"

        if filename_hint:
            basename, ext = _split_filename(filename_hint)
            exts = _preferred_exts(ext)
            params["__basename"] = basename

            # filename (basename) bonus + flag
            if do_join_files and ("path" in cols_f or "file_path" in cols_f):
                pcol = "path" if "path" in cols_f else "file_path"
                cond = f"(LOWER(f.{pcol}) LIKE LOWER('%%' || %(__basename)s || '%%') OR LOWER({symbol_expr}) LIKE LOWER('%%' || %(__basename)s || '%%'))"
                fn_bonus_sql = f" + {W_FN} * CASE WHEN {cond} THEN 1 ELSE 0 END"
                fn_flag_sql = f"CASE WHEN {cond} THEN 1 ELSE 0 END"
            elif ("path" in cols_c or "file_path" in cols_c):
                pcol = "path" if "path" in cols_c else "file_path"
                cond = f"(LOWER(c.{pcol}) LIKE LOWER('%%' || %(__basename)s || '%%') OR LOWER({symbol_expr}) LIKE LOWER('%%' || %(__basename)s || '%%'))"
                fn_bonus_sql = f" + {W_FN} * CASE WHEN {cond} THEN 1 ELSE 0 END"
                fn_flag_sql = f"CASE WHEN {cond} THEN 1 ELSE 0 END"

            # extension affinity bonus + flag
            if exts:
                params["__ext_regex"] = r'\.(' + "|".join(re.escape(e) for e in exts) + r')$'
                if do_join_files and ("path" in cols_f or "file_path" in cols_f):
                    pcol = "path" if "path" in cols_f else "file_path"
                    ext_bonus_sql = f" + {W_EXT} * CASE WHEN LOWER(f.{pcol}) ~ LOWER(%(__ext_regex)s) THEN 1 ELSE 0 END"
                    ext_flag_sql  = f"CASE WHEN LOWER(f.{pcol}) ~ LOWER(%(__ext_regex)s) THEN 1 ELSE 0 END"
                elif ("path" in cols_c or "file_path" in cols_c):
                    pcol = "path" if "path" in cols_c else "file_path"
                    ext_bonus_sql = f" + {W_EXT} * CASE WHEN LOWER(c.{pcol}) ~ LOWER(%(__ext_regex)s) THEN 1 ELSE 0 END"
                    ext_flag_sql  = f"CASE WHEN LOWER(c.{pcol}) ~ LOWER(%(__ext_regex)s) THEN 1 ELSE 0 END"

            # tiny content token bonus (very light)
            tok_bonus_sql = (
                f" + {W_TOK} * CASE WHEN to_tsvector('{lang}', c.content) "
                f"@@ plainto_tsquery('{lang}', %(__basename)s) THEN 1 ELSE 0 END"
            )

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
              ({lex_sym}{vec_term}{fn_bonus_sql}{ext_bonus_sql}{tok_bonus_sql}) AS hybrid_score,
              {fn_flag_sql} AS __fn_hit,
              {ext_flag_sql} AS __ext_hit
            {from_sql}
            {where_sql}
            ORDER BY
              __fn_hit DESC,     -- filename match first if present
              __ext_hit DESC,    -- then extension affinity
              hybrid_score DESC  -- then overall hybrid score
            LIMIT %(top_k)s
        """

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

            # Optional: uncomment to inspect ordering
            # for i, r in enumerate(rows[:5], 1):
            #     log.info("hit#%d fn=%s ext=%s s=%.4f %s:%s-%s",
            #              i, r.get("__fn_hit"), r.get("__ext_hit"), r["hybrid_score"],
            #              r.get("path"), r.get("start_line"), r.get("end_line"))

            log.info("search_hybrid: rows=%d join_files=%s embed=%s",
                     len(rows), do_join_files, bool(vec_term))
            return list(rows)

    finally:
        if created_conn is not None:
            created_conn.close()
