import os
from typing import TypedDict

import psycopg2
import psycopg2.extras
from langgraph.graph import StateGraph, START, END
import ollama

DB_DSN = os.getenv("DB_DSN", "postgresql://rag:ragpwd@localhost:5432/ragdb")

# Ollama config
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMB_MODEL = os.getenv("EMB_MODEL", "mxbai-embed-large")  # 1024-dim
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
_ollama = ollama.Client(host=OLLAMA_BASE)


def as_vector_literal(vec):
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class State(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: State) -> State:
    q = state["question"]
    q_emb = _ollama.embeddings(model=EMB_MODEL, prompt=q)["embedding"]
    q_emb_lit = as_vector_literal(q_emb)

    with psycopg2.connect(DB_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
                1 - (v.embedding <=> (SELECT emb FROM q)) AS vec_score,
                ts_rank_cd(v.content_tsv, (SELECT tsq FROM q)) AS lex_score
              FROM rag.v_chunk_search v
              WHERE ( (SELECT tsq FROM q) @@ v.content_tsv OR (SELECT emb FROM q) IS NOT NULL )
            )
            SELECT *, (0.7*vec_score + 0.3*lex_score) AS s
            FROM scored
            ORDER BY s DESC
            LIMIT 12;
            """,
            (q_emb_lit, q, q),
        )
        rows = cur.fetchall()

    stitched = "\n\n".join(
        f"// {r['repo_name']}:{r['path']}:{r['start_line']}-{r['end_line']} [{r['symbol_signature'] or r['symbol_name'] or ''}]\n{r['content']}"
        for r in rows
    )
    state["context"] = stitched
    return state


SYS = """You are a senior code assistant. Use the provided repository context strictly.
Cite file/line provenance already included in the context comments. If context is insufficient, say so briefly."""


def answer(state: State) -> State:
    ctx = state["context"]
    q = state["question"]
    prompt = f"{SYS}\n\nContext:\n{ctx}\n\nUser question:\n{q}\n\nAnswer:"

    resp = _ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    # python client returns {'message': {'role': 'assistant', 'content': '...'}, ...}
    content = resp.get("message", {}).get("content") or ""
    state["answer"] = content
    return state


def main():
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    app = graph.compile()

    # simple CLI demo
    try:
        q = input("Ask about your codebase: ").strip()
    except EOFError:
        q = "Where is JWT verification implemented?"
    out = app.invoke({"question": q, "context": "", "answer": ""})
    print("\n=== ANSWER ===\n", out["answer"])


if __name__ == "__main__":
    main()
