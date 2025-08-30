import os
from typing import Optional, List

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Query
from pydantic import BaseModel

import ollama

DB_DSN = os.getenv("DB_DSN", "postgresql://rag:ragpwd@localhost:5432/ragdb")

# Ollama (embeddings)
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMB_MODEL = os.getenv("EMB_MODEL", "mxbai-embed-large")  # 1024-dim
_ollama = ollama.Client(host=OLLAMA_BASE)

app = FastAPI(title="Code RAG Retrieval API (Ollama)")

def as_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class SearchResult(BaseModel):
    id: int
    repo_name: str
    path: str
    language: str
    symbol_name: Optional[str]
    symbol_kind: Optional[str]
    symbol_signature: Optional[str]
    start_line: int
    end_line: int
    score: float
    content: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    stitched_context: str


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Natural language or symbol query"),
    repo: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = 12,
    w_vec: float = 0.65,
    w_lex: float = 0.25,
    w_sym: float = 0.10,
):
    # embed query with Ollama
    q_emb = _ollama.embeddings(model=EMB_MODEL, prompt=q)["embedding"]
    q_emb_lit = as_vector_literal(q_emb)

    conn = psycopg2.connect(DB_DSN)
    conn.set_session(readonly=True, autocommit=True)

    with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH q AS (
              SELECT %s::vector AS emb,
                     plainto_tsquery('simple', unaccent(%s)) AS tsq,
                     %s AS qraw
            ),
            scored AS (
              SELECT
                v.id, v.repo_name, v.path, v.language,
                v.symbol_name, v.symbol_kind, v.symbol_signature,
                v.start_line, v.end_line, v.content,
                1 - (v.embedding <=> (SELECT emb FROM q))           AS vec_score,
                ts_rank_cd(v.content_tsv, (SELECT tsq FROM q))      AS lex_score,
                CASE
                  WHEN v.symbol_name IS NOT NULL AND lower(v.symbol_name) = lower((SELECT qraw FROM q)) THEN 1.0
                  WHEN v.symbol_signature ILIKE ('%' || (SELECT qraw FROM q) || '%') THEN 0.5
                  ELSE 0.0
                END AS sym_boost
              FROM rag.v_chunk_search v
              WHERE (%s IS NULL OR v.repo_name = %s)
                AND (%s::rag.language IS NULL OR v.language = %s::rag.language)
                AND ( (SELECT tsq FROM q) @@ v.content_tsv OR (SELECT emb FROM q) IS NOT NULL )
            )
            SELECT id, repo_name, path, language, symbol_name, symbol_kind, symbol_signature,
                   start_line, end_line, content,
                   (%s*vec_score) + (%s*lex_score) + (%s*sym_boost) AS hybrid_score
            FROM scored
            ORDER BY hybrid_score DESC
            LIMIT %s;
            """,
            (
                q_emb_lit,
                q,
                q,
                repo,
                repo,
                language,
                language,
                w_vec,
                w_lex,
                w_sym,
                top_k,
            ),
        )
        rows = cur.fetchall()

    results = [
        SearchResult(
            id=r["id"],
            repo_name=r["repo_name"],
            path=r["path"],
            language=r["language"],
            symbol_name=r["symbol_name"],
            symbol_kind=r["symbol_kind"],
            symbol_signature=r["symbol_signature"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            score=float(r["hybrid_score"]),
            content=r["content"],
        )
        for r in rows
    ]

    stitched = "\n\n".join(
        f"// {r.repo_name}:{r.path}:{r.start_line}-{r.end_line}  [{r.symbol_signature or r.symbol_name or ''}]\n{r.content}".strip()
        for r in results
    )
    return SearchResponse(results=results, stitched_context=stitched)
