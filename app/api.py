# app/api.py
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any, Set, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel


# ------------------------------ Logging ---------------------------------------

log = logging.getLogger("api")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
log.addHandler(_handler)
log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# ------------------------------ Config ----------------------------------------

DB_DSN_DEFAULT = "postgresql://rag:ragpwd@pg:5432/ragdb"

# Ollama embedding config (optional)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://ollama:11434"
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
DISABLE_EMBEDDING = os.getenv("DISABLE_EMBEDDING", "").lower() in {"1", "true", "yes"}

# -------------------------- Reranker imports (soft) ----------------------------

try:
    from app.rerankers import Candidate as RRQtyCandidate
    from app.rerankers import CrossEncoderReranker, LLMJudgeReranker

    @dataclass
    class Candidate(RRQtyCandidate):  # type: ignore[misc]
        pass
except Exception:
    @dataclass
    class Candidate:
        chunk_id: int
        repo: Optional[str]
        path: Optional[str]
        start_line: Optional[int]
        end_line: Optional[int]
        content: str
        symbol: Optional[str]
        score: float

    class CrossEncoderReranker:
        def rerank(self, query: str, cands: List[Candidate], top_k: int = 20) -> List[Candidate]:
            return cands[:top_k]

    class LLMJudgeReranker(CrossEncoderReranker):
        pass

# ------------------------------ DB Helpers ------------------------------------

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

# ------------------------------ Embeddings ------------------------------------

def _ollama_embed(text: str) -> Optional[List[float]]:
    """Return embedding via Ollama /api/embeddings, or None if unavailable/empty."""
    if DISABLE_EMBEDDING:
        return None
    payload = {"model": OLLAMA_EMBED_MODEL, "input": text}
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read()
            obj = json.loads(body.decode("utf-8"))
            # Expect {"embedding":[...]} or {"data":[{"embedding":[...]}]}
            emb = None
            if isinstance(obj.get("embedding"), list):
                emb = obj["embedding"]
            elif isinstance(obj.get("data"), list) and obj["data"] and isinstance(obj["data"][0].get("embedding"), list):
                emb = obj["data"][0]["embedding"]
            # Treat empty vector as unavailable
            if not emb or len(emb) == 0:
                log.warning("Ollama embeddings empty; skipping vector component")
                return None
            return emb
    except (URLError, HTTPError) as e:
        log.warning("Ollama embeddings unavailable: %s", e)
        return None
    except Exception as e:
        log.warning("Ollama embeddings error: %s", e)
        return None

# ------------------------------ Reranking glue --------------------------------

def _to_candidates(rows: List[Dict[str, Any]]) -> List[Candidate]:
    return [
        Candidate(
            chunk_id=int(r.get("id") or r.get("chunk_id")),
            repo=r.get("repo_name"),
            path=r.get("path"),
            start_line=r.get("start_line"),
            end_line=r.get("end_line"),
            content=r.get("content") or "",
            symbol=r.get("symbol_name"),
            score=float(r.get("hybrid_score") or r.get("score") or 0.0),
        )
        for r in rows
    ]

def _rerank_candidates(
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

# ------------------------------ Language filter --------------------------------

def _build_language_filter(
    code_language: Optional[str],
    do_join_files: bool,
    cols_f: Set[str],
    cols_c: Set[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Programming language filter. Prefers f.language; falls back to path extension.
    """
    if not code_language:
        return "", {}

    cl = code_language.strip().lower()
    params: Dict[str, Any] = {}

    if do_join_files and "language" in cols_f:
        return " AND LOWER(f.language) = %(code_language)s", {"code_language": cl}

    # Fallback: infer via extension
    ext_map: Dict[str, List[str]] = {
        "java": ["%.java"],
        "javascript": ["%.js", "%.mjs", "%.cjs"],
        "typescript": ["%.ts", "%.tsx"],
        "python": ["%.py"],
        "go": ["%.go"],
        "ruby": ["%.rb"],
        "csharp": ["%.cs"],
    }
    pats = ext_map.get(cl)
    if not pats:
        return "", {}

    path_col = None
    if do_join_files and ("path" in cols_f or "file_path" in cols_f):
        path_col = "f.path" if "path" in cols_f else "f.file_path"
    elif ("path" in cols_c or "file_path" in cols_c):
        path_col = "c.path" if "path" in cols_c else "c.file_path"

    if not path_col:
        return "", {}

    likes = []
    for i, pat in enumerate(pats):
        key = f"code_ext_{i}"
        likes.append(f"{path_col} ILIKE %({key})s")
        params[key] = pat

    return " AND (" + " OR ".join(likes) + ")", params

# ------------------------------ Hybrid Search ---------------------------------

def search_hybrid(
    query: str,
    *,
    language: Optional[str] = None,        # text-search dict: english|simple
    code_language: Optional[str] = None,   # programming language filter e.g. 'java'
    topn: int = 20,
    w_vec: float = 0.6,
    w_lex: float = 0.3,
    w_sym: float = 0.1,
    repo: Optional[str] = None,
    path_glob: Optional[str] = None,
    query_embedding: Optional[Sequence[float]] = None,
    conn: Optional[psycopg2.extensions.connection] = None,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns rows with: id, repo_name, path, start_line, end_line, content, symbol_name, hybrid_score, score
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

        # Join decision
        has_c_file_id = "file_id" in cols_c
        has_f_id = "id" in cols_f
        do_join_files = has_files and has_f_id and has_c_file_id

        # repo expression
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

        has_c_embed = "embedding" in cols_c

        # WHERE filters
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

        if path_glob:
            if do_join_files and ("path" in cols_f or "file_path" in cols_f):
                col = "path" if "path" in cols_f else "file_path"
                where_clauses.append(f"f.{col} LIKE %(path_glob)s")
                params["path_glob"] = path_glob
            elif ("path" in cols_c or "file_path" in cols_c):
                col = "path" if "path" in cols_c else "file_path"
                where_clauses.append(f"c.{col} LIKE %(path_glob)s")
                params["path_glob"] = path_glob

        # Programming-language restriction
        code_lang_sql, code_lang_params = _build_language_filter(
            code_language, do_join_files, cols_f, cols_c
        )
        if code_lang_sql:
            where_clauses.append("TRUE" + code_lang_sql)  # append with AND
            params.update(code_lang_params)

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

        # Optional vector term (only with non-empty embedding)
        vec_term = ""
        use_vec = bool(has_c_embed and query_embedding and len(query_embedding) > 0)
        if use_vec:
            vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
            params["embed"] = vector_literal
            vec_term = " + %(w_vec)s * (1 - (c.embedding <=> %(embed)s::vector))"

        # FROM/JOIN
        from_sql = "FROM code_chunks c"
        if do_join_files:
            from_sql += " JOIN code_files f ON f.id = c.file_id"

        # Build SQL with and without vector component
        sql_base = f"""
            SELECT
              c.id AS id,
              {repo_expr} AS repo_name,
              {path_expr} AS path,
              c.start_line,
              c.end_line,
              c.content,
              {symbol_expr} AS symbol_name,
              ({{score_expr}}) AS hybrid_score,
              ({{score_expr}}) AS score
            {from_sql}
            {where_sql}
            ORDER BY hybrid_score DESC
            LIMIT %(top_k)s
        """
        sql_with_vec = sql_base.format(score_expr=f"{lex_sym}{vec_term}")
        sql_no_vec  = sql_base.format(score_expr=f"{lex_sym}")

        # Execute, with safe fallback if pgvector errors
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                if use_vec:
                    cur.execute(sql_with_vec, params)
                else:
                    cur.execute(sql_no_vec, params)
            except Exception as e:
                # Fall back if vector type/dim issues arise
                msg = str(e).lower()
                if "vector" in msg or "<=>" in msg:
                    log.warning("Vector scoring failed (%s); retrying without vector", e)
                    cur.execute(sql_no_vec, params)
                else:
                    raise
            rows = cur.fetchall()
            log.info(
                "search_hybrid: rows=%d join_files=%s embed_component=%s",
                len(rows), do_join_files, use_vec
            )
            return list(rows)

    finally:
        if created_conn is not None:
            created_conn.close()

# ------------------------------ FastAPI App -----------------------------------

app = FastAPI(title="RAG Search API", version="1.0.1")

def _maybe_compute_query_embedding(
    conn, want_vector: bool, query: str
) -> Optional[List[float]]:
    """If code_chunks has 'embedding' and vector term is desired, try to embed the query."""
    if not want_vector:
        return None
    if "embedding" not in _table_cols(conn, "code_chunks"):
        return None
    emb = _ollama_embed(query)
    # Guard against empty or invalid
    if not emb or not isinstance(emb, list) or len(emb) == 0:
        log.warning("[embedding] unavailable or empty; skipping vector component")
        return None
    log.info("[embedding] model=%s len=%d", OLLAMA_EMBED_MODEL, len(emb))
    return emb

def _stitch_context(rows: List[Dict[str, Any]]) -> str:
    """Human-friendly stitched context (useful for debugging)."""
    lines: List[str] = []
    for r in rows:
        repo = r.get("repo_name") or "None"
        path = r.get("path") or "None"
        start = r.get("start_line")
        end = r.get("end_line")
        sym = r.get("symbol_name")
        hdr = f"// {repo}:{path}:{start}-{end}" + (f" [{sym}]" if sym else "")
        lines.append(hdr)
        lines.append(r.get("content") or "")
        lines.append("")
    return "\n".join(lines).rstrip()

@app.get("/search")
def search_endpoint(
    q: str = Query(..., description="Natural language/code query"),
    language: Optional[str] = Query(None, description="TS dictionary: english|simple"),
    code_language: Optional[str] = Query(None, description="Programming language filter (e.g., java)"),
    topn: int = Query(20, ge=1, le=200),
    repo: Optional[str] = Query(None),
    path_glob: Optional[str] = Query(None, description="SQL LIKE pattern, e.g. %.java"),
    use_vector: bool = Query(True, description="If true, try vector component (needs pgvector + Ollama)"),
    rerank: bool = Query(False, description="Enable reranking (if available)"),
    reranker_kind: str = Query("cross", description="cross|judge"),
    top_rerank: int = Query(20, ge=1, le=200),
):
    """Hybrid search with optional programming-language filter and reranking."""
    conn = None
    try:
        conn = _connect(None)

        query_embedding = _maybe_compute_query_embedding(conn, use_vector, q)

        rows = search_hybrid(
            q,
            language=language,
            code_language=code_language,
            topn=topn,
            repo=repo,
            path_glob=path_glob,
            query_embedding=query_embedding,
            conn=conn,
        )

        # Optional rerank
        cands = _rerank_candidates(q, rows, rerank, reranker_kind, top_rerank)

        if rerank:
            results = [
                {
                    "chunk_id": c.chunk_id,
                    "repo": c.repo,
                    "path": c.path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "content": c.content,
                    "symbol": c.symbol,
                    "score": c.score,
                }
                for c in cands
            ]
            stitched = _stitch_context([
                {
                    "repo_name": r["repo"],
                    "path": r["path"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "symbol_name": r["symbol"],
                    "content": r["content"],
                } for r in results
            ])
        else:
            results = [
                {
                    "chunk_id": int(r.get("id")),
                    "repo": r.get("repo_name"),
                    "path": r.get("path"),
                    "start_line": r.get("start_line"),
                    "end_line": r.get("end_line"),
                    "content": r.get("content"),
                    "symbol": r.get("symbol_name"),
                    "score": float(r.get("hybrid_score") or r.get("score") or 0.0),
                }
                for r in rows
            ]
            stitched = _stitch_context(rows)

        return JSONResponse(
            {
                "query": q,
                "results": results,
                "stitched_context": stitched,
            }
        )

    except Exception as e:
        log.exception("search failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn is not None:
            conn.close()

@app.get("/healthz")
def healthz():
    return {"ok": True}

class AskBody(BaseModel):
    question: str

# ----- Strands -----
try:
    import inspect, anyio
    from app.strands_agent import run_with_rate_limits as run_strands
    STRANDS_OK, STRANDS_ERR = True, None
except Exception as e:
    STRANDS_OK, STRANDS_ERR = False, e

@app.post("/ask/strands")
async def ask_strands(body: AskBody):
    if not STRANDS_OK:
        raise HTTPException(status_code=503, detail=f"Strands disabled: {STRANDS_ERR}")
    # If run_strands is async, await it; if sync, run it on a thread.
    if inspect.iscoroutinefunction(run_strands):
        ans = await run_strands(body.question)
    else:
    # Run sync function without blocking the event loop
        ans = await run_in_threadpool(run_strands, body.question)
    return {"engine": "strands", "answer": ans}

# ----- LangGraph -----
try:
    from app.agent_graph import ask_once as run_langgraph
    LG_OK, LG_ERR = True, None
except Exception as e:
    LG_OK, LG_ERR = False, e

@app.post("/ask/langgraph")
async def ask_langgraph(body: AskBody):
    if not LG_OK:
        raise HTTPException(status_code=503, detail=f"LangGraph disabled: {LG_ERR}")
    return {"engine": "langgraph", "answer": run_langgraph(body.question)}

# ----- Graph endpoints -----
try:
    from app.search import graph_relations, graph_impact
    G_OK, G_ERR = True, None
except Exception as e:
    G_OK, G_ERR = False, e

@app.get("/graph/relations")
def graph_rel(repo: str, file: str, direction: str = "both", depth: int = 2):
    if not G_OK:
        raise HTTPException(status_code=503, detail=f"Graph disabled: {G_ERR}")
    return graph_relations(repo, file, direction, depth)

@app.get("/graph/impact")
def graph_imp(repo: str, file: str, depth: int = 2):
    if not G_OK:
        raise HTTPException(status_code=503, detail=f"Graph disabled: {G_ERR}")
    return graph_impact(repo, file, depth)

