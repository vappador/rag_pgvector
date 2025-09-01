# app/agent_graph.py
"""
Verbose, step-by-step logging for the RAG agent, now using search_hybrid().
"""

import os
import sys
import json
import time
import uuid
import logging
from typing import TypedDict
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from langgraph.graph import StateGraph, START, END
import ollama

# --- make local modules importable (so we can `from search import search_hybrid`)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from search import search_hybrid  # <-- same-folder import

# ------------------------------- Logging --------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("agent")
RUN_ID = os.getenv("RUN_ID", uuid.uuid4().hex[:8])

def preview(txt: str, limit: int = 240) -> str:
    if not txt:
        return ""
    txt = str(txt).replace("\n", "\\n")
    return txt if len(txt) <= limit else txt[:limit] + f"... (+{len(txt)-limit} more)"

@contextmanager
def span(name: str):
    t0 = time.perf_counter()
    log.info(f"[run={RUN_ID}] ▶ {name} …")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info(f"[run={RUN_ID}] ✔ {name} done in {dt:.3f}s")

# ------------------------------ Config ----------------------------------------

DB_DSN = os.getenv("DB_DSN", "postgresql://rag:ragpwd@pg:5432/ragdb")
OLLAMA_BASE = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMB_MODEL = os.getenv("EMB_MODEL", "mxbai-embed-large:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
SHOW_PROMPT = os.getenv("SHOW_PROMPT", "1").lower() in ("1", "true", "yes")
SAVE_PROMPT_PATH = os.getenv("SAVE_PROMPT_PATH", "/workspace/debug/last_prompt.md")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))
ROWS_LIMIT = int(os.getenv("ROWS_LIMIT", "12"))
VEC_WEIGHT = float(os.getenv("VEC_WEIGHT", "0.7"))
LEX_WEIGHT = float(os.getenv("LEX_WEIGHT", "0.3"))
REASONING_BULLETS = int(os.getenv("REASONING_BULLETS", "4"))

_ollama = ollama.Client(host=OLLAMA_BASE)

# ------------------------------- State ----------------------------------------

class State(TypedDict):
    question: str
    context: str
    answer: str

# ------------------------- Retrieval & Generation -----------------------------

def _truncate_center(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n// … (context truncated) …\n" + text[-keep:]

def retrieve(state: State) -> State:
    q = state["question"]
    log.info(f"[run={RUN_ID}] [embedding] question preview: \"{preview(q)}\"")

    # Embed query with Ollama
    with span("Embedding query"):
        try:
            res = _ollama.embeddings(model=EMB_MODEL, prompt=q)
            q_emb = res.get("embedding") or []
        except Exception as e:
            log.warning(f"[run={RUN_ID}] [embedding] failed: {e}")
            q_emb = []
    log.info(f"[run={RUN_ID}] [embedding] len={len(q_emb)}")

    # Run hybrid search (uses filename/extension soft bonuses internally)
    with span("DB search (hybrid via search_hybrid)"):
        with psycopg2.connect(DB_DSN) as conn:
            rows = search_hybrid(
                q,
                language="english",
                topn=ROWS_LIMIT,
                w_vec=VEC_WEIGHT,
                w_lex=LEX_WEIGHT,
                w_sym=0.1,
                query_embedding=q_emb if q_emb else None,
                conn=conn,
            )

        log.info(f"[run={RUN_ID}] [db] rows={len(rows)} (top {min(3, len(rows))} follows)")
        for i, r in enumerate(rows[:3], 1):
            log.info(
                "[run=%s] [db] hit#%d score=%.4f fn=%s ext=%s %s:%s-%s [%s]",
                RUN_ID, i,
                float(r.get("hybrid_score") or 0.0),
                r.get("__fn_hit"), r.get("__ext_hit"),
                r.get("path"), r.get("start_line"), r.get("end_line"),
                (r.get("symbol_name") or "").strip()
            )

    stitched = "// No results found" if not rows else "\n\n".join(
        f"// {r.get('repo_name')}:{r.get('path')}:{r['start_line']}-{r['end_line']} "
        f"[{r.get('symbol_name') or ''}]\n{r['content']}"
        for r in rows
    )
    stitched = _truncate_center(stitched, MAX_CONTEXT_CHARS)

    state["context"] = stitched
    log.info(f"[run={RUN_ID}] [retrieve] stitched context preview: \"{preview(stitched)}\"")
    return state

# ------------------------------- Prompting ------------------------------------

SYS = (
    "You are a senior code assistant working over a code repository. Use the provided context strictly.\n"
    "Cite file/line provenance that is included in the context comments. If context is insufficient, say so briefly.\n\n"
    "Respond in this exact format:\n"
    "### Reasoning (brief)\n"
    f"- 3 to {REASONING_BULLETS} concise bullets about the factors that matter. No step-by-step derivations.\n"
    "### Answer\n"
    "Provide the final answer with citations to the provided context when relevant."
)

def build_prompt(question: str, context: str) -> str:
    return (
        f"{SYS}\n\n"
        "### Context (citations are present in the comments)\n"
        f"{context}\n\n"
        "### User question\n"
        f"{question}\n\n"
        "### Follow the required response format above."
    )

def generate_answer(state: State) -> State:
    ctx, q = state["context"], state["question"]
    prompt = build_prompt(q, ctx)
    if SHOW_PROMPT:
        print("\n────────────\n RAG PROMPT\n────────────\n")
        print(prompt)
    try:
        os.makedirs(os.path.dirname(SAVE_PROMPT_PATH), exist_ok=True)
        with open(SAVE_PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt)
    except Exception:
        pass

    log.info(f"[run={RUN_ID}] [llm] model={LLM_MODEL}")
    full = ""
    with span("LLM generation (stream)"):
        try:
            stream = _ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                stream=True,
            )
            for chunk in stream:
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    print(piece, end="", flush=True)
                    full += piece
        except Exception as e:
            log.error(f"[run={RUN_ID}] [llm] call failed: {e}")
            full = "I couldn't generate an answer because the local model was unavailable."
    print()
    state["answer"] = full.strip()
    return state

# --------------------------------- Main ---------------------------------------

def main():
    log.info(f"[run={RUN_ID}] Starting agent with:")
    log.info(f"[run={RUN_ID}]   DB_DSN={DB_DSN}")
    log.info(f"[run={RUN_ID}]   OLLAMA_BASE={OLLAMA_BASE}")
    log.info(f"[run={RUN_ID}]   EMB_MODEL={EMB_MODEL} LLM_MODEL={LLM_MODEL}")
    log.info(f"[run={RUN_ID}]   ROWS_LIMIT={ROWS_LIMIT}  VEC_WEIGHT={VEC_WEIGHT}  LEX_WEIGHT={LEX_WEIGHT}")

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    app = graph.compile()

    try:
        q = input("Ask about your codebase: ").strip()
    except EOFError:
        q = "Write a summary of Person.java"
    log.info(f"[run={RUN_ID}] Question: \"{preview(q)}\"")

    out = app.invoke({"question": q, "context": "", "answer": ""})

    print("\n────────────\n ANSWER (final)\n────────────\n")
    print(out.get("answer", "").strip())

if __name__ == "__main__":
    main()
