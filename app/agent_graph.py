# app/agent_graph.py
"""
Verbose, step-by-step logging for the RAG agent.

ENV knobs:
- LOG_LEVEL: DEBUG | INFO | WARNING | ERROR (default: INFO)
- LOG_PREVIEW_CHARS: int, max chars to show when previewing question/context (default: 240)
- OLLAMA_DEBUG: 1/true to print preflight required/present sets
- OLLAMA_AUTOPULL: 1/true to auto-pull missing models (default: 0)
- OLLAMA_HOST / OLLAMA_BASE_URL: Ollama base URL (default: http://ollama:11434)

Run example:
  LOG_LEVEL=DEBUG docker compose exec app python /workspace/app/agent_graph.py
"""

import os
import sys
import json
import time
import uuid
import logging
from typing import TypedDict
from contextlib import contextmanager
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import psycopg2
import psycopg2.extras
from langgraph.graph import StateGraph, START, END
import ollama

# ------------------------------- Logging --------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "240"))
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("agent")
RUN_ID = os.getenv("RUN_ID", uuid.uuid4().hex[:8])

def preview(txt: str, limit: int = LOG_PREVIEW_CHARS) -> str:
    if txt is None:
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
EMB_MODEL = os.getenv("EMB_MODEL", "mxbai-embed-large:latest")  # 1024-dim
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
AUTO_PULL = os.getenv("OLLAMA_AUTOPULL", "0").lower() in ("1", "true", "yes")
DEBUG_PRE = os.getenv("OLLAMA_DEBUG", "0").lower() in ("1", "true", "yes")

_ollama = ollama.Client(host=OLLAMA_BASE)

# ------------------------------ Helpers ---------------------------------------

def as_vector_literal(vec):
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def _norm_model(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if ":" in s else f"{s}:latest"

def _http_tags() -> dict:
    url = OLLAMA_BASE.rstrip("/") + "/api/tags"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=5) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))

def _client_tags_anyshape():
    return _ollama.list()

def _get_ollama_tags():
    # Prefer HTTP (matches your curl), fallback to client.
    try:
        return _http_tags()
    except Exception as http_err:
        try:
            return _client_tags_anyshape()
        except Exception as client_err:
            raise RuntimeError(
                f"Failed to list Ollama models via HTTP and client. "
                f"http_error={type(http_err).__name__}: {http_err}; "
                f"client_error={type(client_err).__name__}: {client_err}"
            )

def _iter_model_dicts(tags_obj):
    if tags_obj is None:
        return
    if isinstance(tags_obj, dict):
        models = tags_obj.get("models")
        if isinstance(models, (list, tuple)):
            for m in models:
                if isinstance(m, dict):
                    yield m
            return
        if any(k in tags_obj for k in ("name", "model", "tag")):
            yield tags_obj
            return
        return
    if isinstance(tags_obj, (list, tuple)):
        for el in tags_obj:
            if isinstance(el, dict):
                if "models" in el and isinstance(el["models"], (list, tuple)):
                    for m in el["models"]:
                        if isinstance(m, dict):
                            yield m
                elif any(k in el for k in ("name", "model", "tag")):
                    yield el
            elif isinstance(el, (list, tuple)):
                for sub in el:
                    if isinstance(sub, dict) and any(k in sub for k in ("name", "model", "tag")):
                        yield sub

def _present_models_from_tags(tags_obj) -> set[str]:
    present: set[str] = set()
    for m in _iter_model_dicts(tags_obj) or []:
        name = (m.get("name") or "").strip()
        tag = (m.get("tag") or "").strip()
        model = (m.get("model") or "").strip()
        if model:
            present.add(_norm_model(model))
        if name and tag and ":" not in name:
            present.add(_norm_model(f"{name}:{tag}"))
        if name:
            present.add(_norm_model(name))
    return present

# ------------------------------- State ----------------------------------------

class State(TypedDict):
    question: str
    context: str
    answer: str

# ----------------------------- Preflight --------------------------------------

def _ensure_models():
    with span("Preflight: list models"):
        try:
            tags_raw = _get_ollama_tags()
            present = _present_models_from_tags(tags_raw)
        except Exception as e:
            log.error(
                f"[run={RUN_ID}] Could not obtain model list from Ollama at {OLLAMA_BASE}: "
                f"{type(e).__name__}: {e}"
            )
            sys.exit(2)

        required = {_norm_model(EMB_MODEL), _norm_model(LLM_MODEL)}
        if DEBUG_PRE or log.isEnabledFor(logging.DEBUG):
            log.debug(f"[run={RUN_ID}] required={sorted(required)}")
            log.debug(f"[run={RUN_ID}] present={sorted(present)}")

        missing = sorted(required - present)
        if missing:
            if not AUTO_PULL:
                log.error(f"[run={RUN_ID}] Missing models: {', '.join(missing)}")
                log.info("           Run:  docker compose exec -T ollama ollama pull <model>")
                log.info("           Or:   make pull-models")
                sys.exit(2)

            for m in missing:
                with span(f"Auto-pull model: {m}"):
                    try:
                        for chunk in _ollama.pull(model=m, stream=True):
                            status = chunk.get("status") or ""
                            digest = chunk.get("digest") or ""
                            if status:
                                log.info(f"[run={RUN_ID}] pull: {status} {digest}")
                    except Exception as e:
                        log.error(f"[run={RUN_ID}] Failed to pull {m}: {e}")
                        sys.exit(2)

# ------------------------- Retrieval & Generation -----------------------------

def retrieve(state: State) -> State:
    q = state["question"]
    log.info(f"[run={RUN_ID}] [embedding] question preview: \"{preview(q)}\"")

    with span("Embedding query"):
        res = _ollama.embeddings(model=_norm_model(EMB_MODEL), prompt=q)
        q_emb = res["embedding"]
        emb_len = len(q_emb)
        # quick stats to prove we got something sensible
        head = ", ".join(f"{x:.6f}" for x in q_emb[:6])
        tail = ", ".join(f"{x:.6f}" for x in q_emb[-6:]) if emb_len > 6 else ""
        log.info(f"[run={RUN_ID}] [embedding] model={_norm_model(EMB_MODEL)} len={emb_len} "
                 f"head=[{head}] tail=[{tail}]")

    q_emb_lit = as_vector_literal(q_emb)

    log.info(f"[run={RUN_ID}] [db] connecting to Postgres: {DB_DSN}")
    with span("DB search (vector + lexical)"):
        with psycopg2.connect(DB_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = '15000ms';")
            try:
                used_unaccent = True
                sql = """
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
                """
                cur.execute(sql, (q_emb_lit, q, q))
            except psycopg2.Error:
                used_unaccent = False
                sql = """
                    WITH q AS (
                      SELECT %s::vector AS emb,
                             plainto_tsquery('simple', %s) AS tsq,
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
                """
                cur.execute(sql, (q_emb_lit, q, q))

            rows = cur.fetchall()
            log.info(f"[run={RUN_ID}] [db] used_unaccent={used_unaccent} rows={len(rows)}")

            # log top-3 hits with scores and provenance
            for i, r in enumerate(rows[:3], start=1):
                log.info(
                    "[run=%s] [db] hit#%d s=%.4f vec=%.4f lex=%.4f %s:%s:%s-%s [%s]",
                    RUN_ID, i, r.get("s", 0.0), r.get("vec_score", 0.0), r.get("lex_score", 0.0),
                    r.get("repo_name", "?"), r.get("path", "?"), r.get("start_line", "?"), r.get("end_line", "?"),
                    (r.get("symbol_signature") or r.get("symbol_name") or "").strip()
                )

    if not rows:
        stitched = "// No results found for the question."
    else:
        stitched = "\n\n".join(
            f"// {r['repo_name']}:{r['path']}:{r['start_line']}-{r['end_line']} "
            f"[{r['symbol_signature'] or r['symbol_name'] or ''}]\n{r['content']}"
            for r in rows
        )

    state["context"] = stitched
    log.info(f"[run={RUN_ID}] [retrieve] stitched context preview: \"{preview(stitched)}\"")
    return state

SYS = (
    "You are a senior code assistant. Use the provided repository context strictly.\n"
    "Cite file/line provenance already included in the context comments. If context is insufficient, say so briefly."
)

def generate_answer(state: State) -> State:
    ctx = state["context"]
    q = state["question"]
    prompt = f"{SYS}\n\nContext:\n{ctx}\n\nUser question:\n{q}\n\nAnswer:"

    log.info(f"[run={RUN_ID}] [llm] model={_norm_model(LLM_MODEL)}")
    log.info(f"[run={RUN_ID}] [llm] prompt sizes: context_chars={len(ctx)} question_chars={len(q)} sys_chars={len(SYS)}")
    log.debug(f"[run={RUN_ID}] [llm] prompt preview: \"{preview(prompt)}\"")

    tokens = 0
    chars = 0
    with span("LLM generation (stream)"):
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
                    # Real-time output to console
                    print(token, end="", flush=True)
                    content_parts.append(token)
                    tokens += 1
                    chars += len(token)
            print()  # newline after stream
        except Exception as e:
            log.error(f"[run={RUN_ID}] [llm] call failed: {e}")
            content_parts.append("I couldn't generate an answer because the local model was unavailable.")

    log.info(f"[run={RUN_ID}] [llm] streamed tokens={tokens} approx_chars={chars}")
    state["answer"] = "".join(content_parts).strip()
    log.info(f"[run={RUN_ID}] [llm] final answer preview: \"{preview(state['answer'])}\"")
    return state

# --------------------------------- Main ---------------------------------------

def main():
    log.info(f"[run={RUN_ID}] Starting agent with:")
    log.info(f"[run={RUN_ID}]   DB_DSN={DB_DSN}")
    log.info(f"[run={RUN_ID}]   OLLAMA_BASE={OLLAMA_BASE}")
    log.info(f"[run={RUN_ID}]   EMB_MODEL={_norm_model(EMB_MODEL)} LLM_MODEL={_norm_model(LLM_MODEL)}")
    _ensure_models()

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    app = graph.compile()

    # simple CLI demo
    try:
        q = input("Ask about your codebase: ").strip()
    except EOFError:
        q = "Where is JWT verification implemented?"
    log.info(f"[run={RUN_ID}] Question: \"{preview(q)}\"")
    out = app.invoke({"question": q, "context": "", "answer": ""})
    print("\n=== ANSWER ===\n", out.get("answer", "").strip())

if __name__ == "__main__":
    main()
