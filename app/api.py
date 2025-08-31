import os
from typing import Literal, Optional, Dict, Any, List, TypedDict

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# --- DB config ---
DB_DSN = os.getenv("DB_DSN", "postgresql://rag:ragpwd@pg:5432/ragdb")

# --- LangGraph imports (keep the original demo alive) ---
from langgraph.graph import StateGraph, START, END
import ollama  # used by langgraph path

# --- Strands agent wrapper (existing demo) ---
from app.strands_agent import run_with_rate_limits

# --- New retrieval/reranking/answering modules ---
from app.search import search_hybrid
from app.rerankers import Candidate, CrossEncoderReranker, LLMJudgeReranker
from app.answering import synthesize_answer

app = FastAPI(title="RAG PGVector API", version="2.1.0")


# ------------------------------------------------------------------------------
# Basics
# ------------------------------------------------------------------------------

def _conn():
    return psycopg2.connect(DB_DSN)


def _get_one(cur, sql: str, args: tuple):
    cur.execute(sql, args)
    return cur.fetchone()


def _all(cur, sql: str, args: tuple):
    cur.execute(sql, args)
    return cur.fetchall()


def _resolve_file(cur, repo_name: str, file_hint: str):
    """
    Resolve a file inside a repo by either:
      - exact relpath match (case-sensitive)
      - or suffix/basename match (case-insensitive)

    Returns (repo_id, repo_name, file_id, file_path).
    Raises HTTPException 404/400 for not found / ambiguous.
    """
    repo = _get_one(
        cur,
        "SELECT id, name FROM repositories WHERE name ILIKE %s LIMIT 1",
        (repo_name,),
    )
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repo not found: {repo_name}")
    repo_id, repo_name_db = repo[0], repo[1]

    # Try exact path first
    exact = _get_one(
        cur,
        "SELECT id, path FROM code_files WHERE repository_id = %s AND path = %s LIMIT 1",
        (repo_id, file_hint),
    )
    if exact:
        return repo_id, repo_name_db, exact[0], exact[1]

    # Otherwise, suffix/basename match
    like = f"%/{file_hint.lstrip('/')}" if "/" in file_hint else f"%/{file_hint}"
    matches = _all(
        cur,
        "SELECT id, path FROM code_files "
        "WHERE repository_id = %s AND path ILIKE %s "
        "ORDER BY LENGTH(path) ASC LIMIT 25",
        (repo_id, like),
    )
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"File not found in repo '{repo_name}': {file_hint}",
        )

    base = file_hint.split("/")[-1]
    exact_base = [m for m in matches if m[1].endswith("/" + base) or m[1] == base]
    pick = exact_base[0] if exact_base else matches[0]

    same_base = [m for m in matches if m[1].endswith("/" + base) or m[1] == base]
    if len(same_base) > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Ambiguous file name '{file_hint}' in repo '{repo_name}'",
                "candidates": [p for (_id, p) in same_base[:25]],
            },
        )

    return repo_id, repo_name_db, pick[0], pick[1]


# ------------------------------------------------------------------------------
# Graph traversal helpers (new-schema SQL)
# ------------------------------------------------------------------------------

def _edges_query(direction: Literal["in", "out"]) -> str:
    """
    direction='out'  : edges starting from chunks in the target file (fan-out)
    direction='in'   : edges pointing into chunks in the target file (fan-in)
    """
    if direction == "out":
        return """
        WITH RECURSIVE start AS (
          SELECT id FROM code_chunks
          WHERE file_id = %s
        ),
        edges AS (
          SELECT e.src_chunk_id, e.dst_chunk_id, 1 AS depth, e.edge_type
          FROM chunk_edges e
          WHERE e.src_chunk_id IN (SELECT id FROM start)
          UNION ALL
          SELECT e.src_chunk_id, e.dst_chunk_id, x.depth + 1, e.edge_type
          FROM chunk_edges e
          JOIN edges x ON e.src_chunk_id = x.dst_chunk_id
          WHERE x.depth < %s
        )
        SELECT
          ed.depth,
          ed.edge_type AS relation,
          s.id          AS src_chunk_id,
          sf.path       AS src_path,
          s.symbol_name AS src_symbol,
          s.start_line  AS src_start,
          s.end_line    AS src_end,
          d.id          AS dst_chunk_id,
          df.path       AS dst_path,
          d.symbol_name AS dst_symbol,
          df.language   AS dst_language
        FROM edges ed
        JOIN code_chunks s ON s.id = ed.src_chunk_id
        JOIN code_files  sf ON sf.id = s.file_id
        JOIN code_chunks d ON d.id = ed.dst_chunk_id
        JOIN code_files  df ON df.id = d.file_id
        WHERE sf.repository_id = %s AND df.repository_id = %s
        ORDER BY ed.depth, sf.path, s.start_line, df.path;
        """
    else:
        return """
        WITH RECURSIVE start AS (
          SELECT id FROM code_chunks
          WHERE file_id = %s
        ),
        edges AS (
          SELECT e.src_chunk_id, e.dst_chunk_id, 1 AS depth, e.edge_type
          FROM chunk_edges e
          WHERE e.dst_chunk_id IN (SELECT id FROM start)
          UNION ALL
          SELECT e.src_chunk_id, e.dst_chunk_id, x.depth + 1, e.edge_type
          FROM chunk_edges e
          JOIN edges x ON e.dst_chunk_id = x.src_chunk_id
          WHERE x.depth < %s
        )
        SELECT
          ed.depth,
          ed.edge_type AS relation,
          s.id          AS src_chunk_id,
          sf.path       AS src_path,
          s.symbol_name AS src_symbol,
          s.start_line  AS src_start,
          s.end_line    AS src_end,
          d.id          AS dst_chunk_id,
          df.path       AS dst_path,
          d.symbol_name AS dst_symbol,
          df.language   AS dst_language
        FROM edges ed
        JOIN code_chunks s ON s.id = ed.src_chunk_id
        JOIN code_files  sf ON sf.id = s.file_id
        JOIN code_chunks d ON d.id = ed.dst_chunk_id
        JOIN code_files  df ON df.id = d.file_id
        WHERE sf.repository_id = %s AND df.repository_id = %s
        ORDER BY ed.depth, sf.path, s.start_line, df.path;
        """


# ------------------------------------------------------------------------------
# Graph endpoints
# ------------------------------------------------------------------------------

@app.get("/graph/relations")
def graph_relations(
    repo: str = Query(..., description="Repository name (e.g., 'spring-petclinic')"),
    file: str = Query(..., description="File path or basename (e.g., 'Person.java' or 'src/main/.../Person.java')"),
    direction: Literal["in", "out", "both"] = Query("both"),
    depth: int = Query(2, ge=1, le=6, description="Traversal depth (hops)"),
    limit_edges: int = Query(5000, ge=1, le=20000, description="Safety cap on returned edges"),
):
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        repo_id, repo_name, file_id, file_path = _resolve_file(cur, repo, file)

        def run_dir(dirflag: Literal["in", "out"]) -> List[Dict[str, Any]]:
            sql = _edges_query(dirflag)
            rows = _all(cur, sql, (file_id, depth, repo_id, repo_id))
            return rows[:limit_edges]

        results: List[Dict[str, Any]] = []
        if direction in ("out", "both"):
            results.extend(run_dir("out"))
        if direction in ("in", "both"):
            results.extend(run_dir("in"))

        files_summary: Dict[str, int] = {}
        counts_by_relation: Dict[str, int] = {}
        for r in results:
            counts_by_relation[r["relation"]] = counts_by_relation.get(r["relation"], 0) + 1
            for path in (r["src_path"], r["dst_path"]):
                if path == file_path:
                    continue
                files_summary[path] = min(files_summary.get(path, 999), r["depth"])

        affected = [{"path": p, "min_depth": d} for p, d in sorted(files_summary.items(), key=lambda x: (x[1], x[0]))]

        return {
            "repo": repo_name,
            "target": {"file_id": file_id, "path": file_path},
            "direction": direction,
            "depth": depth,
            "edge_count": len(results),
            "counts_by_relation": counts_by_relation,
            "affected_files": affected,
            "edges": results,
        }


def _impact_sql() -> str:
    # imports is TEXT[]; match via unnest. content_tsv is TSVECTOR ('english').
    return """
    WITH RECURSIVE
    start AS (
      SELECT id FROM code_chunks
      WHERE file_id = %s
    ),
    direct_dep AS (
      SELECT DISTINCT e.src_chunk_id, 1 AS depth, e.edge_type
      FROM chunk_edges e
      WHERE e.dst_chunk_id IN (SELECT id FROM start)
    ),
    dep AS (
      SELECT src_chunk_id, 1 AS depth
      FROM direct_dep
      UNION ALL
      SELECT e.src_chunk_id, d.depth + 1
      FROM chunk_edges e
      JOIN dep d ON e.dst_chunk_id = d.src_chunk_id
      WHERE d.depth < %s
    ),
    dep_files AS (
      SELECT DISTINCT f.path, MIN(d.depth) AS min_depth
      FROM dep d
      JOIN code_chunks s ON s.id = d.src_chunk_id
      JOIN code_files f   ON f.id = s.file_id
      WHERE f.repository_id = %s
      GROUP BY f.path
    ),
    import_refs AS (
      SELECT DISTINCT f.path
      FROM code_chunks c
      JOIN code_files f ON f.id = c.file_id
      WHERE f.repository_id = %s
        AND EXISTS (
          SELECT 1
          FROM unnest(c.imports) AS imp(txt)
          WHERE txt ILIKE %s
        )
    ),
    lex_refs AS (
      SELECT DISTINCT f.path
      FROM code_chunks c
      JOIN code_files f ON f.id = c.file_id
      WHERE f.repository_id = %s
        AND c.content_tsv @@ plainto_tsquery('english', %s)
    ),
    unioned AS (
      SELECT df.path,
             CASE WHEN df.min_depth = 1 THEN 'edge:direct' ELSE 'edge:transitive' END AS reason,
             df.min_depth
      FROM dep_files df
      UNION ALL
      SELECT ir.path, 'imports:' || %s AS reason, NULL::int AS min_depth
      FROM import_refs ir
      UNION ALL
      SELECT lr.path, 'lexical:' || %s AS reason, NULL::int AS min_depth
      FROM lex_refs lr
    ),
    clean AS (
      SELECT * FROM unioned u
      WHERE u.path <> (SELECT path FROM code_files WHERE id = %s)
    )
    SELECT path,
           MIN(min_depth) AS min_depth,
           ARRAY_AGG(DISTINCT reason) AS reasons
    FROM clean
    GROUP BY path
    ORDER BY (MIN(CASE WHEN reason LIKE 'edge:%' THEN 0 ELSE 1 END)),
             COALESCE(MIN(min_depth), 999),
             path
    LIMIT %s;
    """


@app.get("/graph/impact")
def graph_impact(
    repo: str = Query(..., description="Repository name (e.g., 'spring-petclinic')"),
    file: str = Query(..., description="Target file (e.g., 'Person.java' or 'src/.../Person.java')"),
    depth: int = Query(2, ge=1, le=6, description="Traversal depth for dependents"),
    token: Optional[str] = Query(
        None,
        description="Optional lexical/import token to match (defaults to file basename without extension, e.g., 'Person').",
    ),
    limit_files: int = Query(5000, ge=1, le=20000, description="Max affected files to return"),
    include_edges: bool = Query(True, description="Also include raw 'in' edges used for impact?"),
    limit_edges: int = Query(5000, ge=1, le=20000, description="Safety cap on returned edges"),
):
    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        repo_id, repo_name, file_id, file_path = _resolve_file(cur, repo, file)
        if not token:
            base = os.path.basename(file_path)
            token = base.split(".", 1)[0] if "." in base else base
        import_like = f"%{token}%"

        sql = _impact_sql()
        rows = _all(
            cur,
            sql,
            (
                file_id,                 # start.file_id
                depth,
                repo_id,                 # dep_files repo
                repo_id,                 # import_refs repo
                import_like,             # import match
                repo_id,                 # lex_refs repo
                token,                   # lex token
                token,                   # reason string (imports)
                token,                   # reason string (lexical)
                file_id,                 # clean: exclude self
                limit_files,
            ),
        )

        result: Dict[str, Any] = {
            "repo": repo_name,
            "target": {"file_id": file_id, "path": file_path},
            "depth": depth,
            "token": token,
            "affected_files": rows,
        }

        if include_edges:
            e_sql = _edges_query("in")
            edges = _all(cur, e_sql, (file_id, depth, repo_id, repo_id))
            result["edges"] = edges[:limit_edges]

            counts: Dict[str, int] = {}
            for e in result["edges"]:
                counts[e["relation"]] = counts.get(e["relation"], 0) + 1
            result["counts_by_relation"] = counts

        return result


# ------------------------------------------------------------------------------
# Search + (optional) reranking
# ------------------------------------------------------------------------------

def _to_candidates(rows: List[Dict[str, Any]]) -> List[Candidate]:
    # Expecting keys from search_hybrid: id, repo_name, path, start_line, end_line,
    # symbol_name, content, hybrid_score (or score)
    out: List[Candidate] = []
    for r in rows:
        out.append(
            Candidate(
                chunk_id=r["id"],
                repo=r["repo_name"],
                path=r["path"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                content=r["content"],
                symbol=r.get("symbol_name"),
                score=float(r.get("hybrid_score") or r.get("score") or 0.0),
            )
        )
    return out


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    language: Optional[str] = Query(None, description="Filter by language (java|typescript|javascript|python)"),
    topn: int = Query(50, ge=1, le=200),
    k: int = Query(10, ge=1, le=50),
    rerank: bool = Query(False),
    reranker: Literal["cross", "judge"] = Query("cross"),
    w_vec: float = Query(0.7, ge=0.0, le=1.0),
    w_lex: float = Query(0.3, ge=0.0, le=1.0),
):
    rows = search_hybrid(q, language=language, topn=topn, w_vec=w_vec, w_lex=w_lex)
    cands = _to_candidates(rows)

    if rerank and cands:
        rr = CrossEncoderReranker() if reranker == "cross" else LLMJudgeReranker()
        cands = rr.rerank(q, cands, top_k=k)
    else:
        cands = cands[:k]

    stitched = "\n\n".join(
        f"// {c.repo}:{c.path}:{c.start_line}-{c.end_line} [{c.symbol or ''}]\n{c.content}"
        for c in cands
    )
    return {
        "query": q,
        "results": [c.__dict__ for c in cands],
        "stitched_context": stitched,
    }


# ------------------------------------------------------------------------------
# Ask endpoints: Strands vs. LangGraph vs. direct (new)
# ------------------------------------------------------------------------------

class AskIn(BaseModel):
    question: str


@app.post("/ask/strands")
def ask_strands(payload: AskIn):
    """
    Uses Strands Agent + Postgres retriever tool.
    Honors RL_* env vars for basic rate limiting.
    """
    user_prompt = (
        "Answer the user's question using repository context.\n"
        "If needed, call the `retrieve_context` tool with the question text to gather relevant code.\n"
        f"User question:\n{payload.question}"
    )
    answer = run_with_rate_limits(user_prompt)
    return {"engine": "strands", "answer": answer}


class SimpleAskIn(BaseModel):
    question: str
    language: Optional[str] = None
    context_k: int = 6
    rerank: bool = True
    reranker: Literal["cross", "judge"] = "cross"


@app.post("/ask")
def ask_simple(payload: SimpleAskIn):
    """
    Direct ask path that uses the new search + rerank + provenance-first synthesis.
    """
    rows = search_hybrid(payload.question, language=payload.language, topn=50, w_vec=0.7, w_lex=0.3)
    cands = _to_candidates(rows)
    if payload.rerank and cands:
        rr = CrossEncoderReranker() if payload.reranker == "cross" else LLMJudgeReranker()
        cands = rr.rerank(payload.question, cands, top_k=payload.context_k)
    else:
        cands = cands[:payload.context_k]
    return {"engine": "direct", **synthesize_answer(payload.question, cands)}


# ------- LangGraph adapter (minimal): reuse your original retrieve/generate ----

class LGState(TypedDict):
    question: str
    context: str
    answer: str


EMB_MODEL = os.getenv("EMB_MODEL", "mxbai-embed-large:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
OLLAMA_BASE = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
_ollama = ollama.Client(host=OLLAMA_BASE)


def _norm_model(s: str) -> str:
    s = (s or "").strip()
    return s if ":" in s else f"{s}:latest"


def _as_vector_literal(vec):
    # pgvector-style literal for parameterized queries
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


SYS = (
    "You are a senior code assistant. Use the provided repository context strictly.\n"
    "Cite file/line provenance already included in the context comments. If context is insufficient, say so briefly."
)


def _retrieve(state: LGState) -> LGState:
    """
    Direct hybrid retriever over the new schema (no rag.* dependencies).
    Produces stitched, provenance-tagged context snippets.
    """
    q = state["question"]
    # Embed the question via Ollama
    res = _ollama.embeddings(model=_norm_model(EMB_MODEL), prompt=q)
    q_emb = res["embedding"]
    q_emb_lit = _as_vector_literal(q_emb)

    with psycopg2.connect(DB_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Hybrid: lexical (tsvector) + vector similarity on code_chunks
        sql = """
            WITH q AS (
              SELECT %s::vector AS emb,
                     plainto_tsquery('english', %s) AS tsq
            ),
            scored AS (
              SELECT
                c.id,
                r.name  AS repo_name,
                f.path  AS path,
                c.language,
                c.symbol_name,
                c.symbol_kind,
                c.start_line,
                c.end_line,
                c.content,
                CASE WHEN c.embedding IS NOT NULL
                     THEN 1 - (c.embedding <=> (SELECT emb FROM q))
                     ELSE 0.0
                END AS vec_score,
                ts_rank_cd(c.content_tsv, (SELECT tsq FROM q)) AS lex_score
              FROM code_chunks c
              JOIN code_files f ON f.id = c.file_id
              JOIN repositories r ON r.id = f.repository_id
              WHERE ( (SELECT tsq FROM q) @@ c.content_tsv ) OR c.embedding IS NOT NULL
            )
            SELECT *,
                   (0.7*vec_score + 0.3*lex_score) AS s
            FROM scored
            ORDER BY s DESC
            LIMIT 12;
        """
        cur.execute(sql, (q_emb_lit, q))
        rows = cur.fetchall()

    if not rows:
        stitched = "// No results found for the question."
    else:
        # Include provenance line as a comment on each snippet
        stitched = "\n\n".join(
            f"// {r['repo_name']}:{r['path']}:{r['start_line']}-{r['end_line']} "
            f"[{(r['symbol_name'] or '')}]"
            f"\n{r['content']}"
            for r in rows
        )
    state["context"] = stitched
    return state


def _generate_answer(state: LGState) -> LGState:
    ctx = state["context"]
    q = state["question"]
    prompt = f"{SYS}\n\nContext:\n{ctx}\n\nUser question:\n{q}\n\nAnswer:"
    content_parts = []
    try:
        stream = _ollama.chat(
            model=_norm_model(LLM_MODEL),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                content_parts.append(token)
    except Exception as e:
        content_parts.append(f"LLM failure: {e}")
    state["answer"] = "".join(content_parts).strip()
    return state


def _build_langgraph_app():
    g = StateGraph(LGState)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate_answer", _generate_answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate_answer")
    g.add_edge("generate_answer", END)
    return g.compile()


_langgraph_app = None


def get_langgraph_app():
    global _langgraph_app
    if _langgraph_app is None:
        _langgraph_app = _build_langgraph_app()
    return _langgraph_app


@app.post("/ask/langgraph")
def ask_langgraph(payload: AskIn):
    """
    Uses the original LangGraph-style pipeline over the new schema.
    """
    app_ = get_langgraph_app()
    out = app_.invoke({"question": payload.question, "context": "", "answer": ""})
    return {"engine": "langgraph", "answer": (out.get("answer") or "").strip()}
