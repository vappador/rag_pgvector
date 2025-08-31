"""
Strands RAG agent (Postgres + pgvector) with Ollama, logging, and rate limiting.

Env:
- DB_DSN: postgresql://rag:ragpwd@pg:5432/ragdb
- OLLAMA_BASE_URL or OLLAMA_HOST: http://ollama:11434 (use host.docker.internal in Docker)
- EMB_MODEL: mxbai-embed-large:latest
- LLM_MODEL: llama3.1:8b
- STRANDS_TEMPERATURE: default 0.2
- STRANDS_MAX_TOKENS: default 2048
- RL_MAX_CONCURRENCY / RL_RPM / RL_TPM: local rate-limit knobs
- LOG_LEVEL: INFO (default), DEBUG etc.

This module exposes:
- run_with_rate_limits(prompt: str) -> str
- retrieve_context(question: str, top_k: int = 12) -> str (tool)
"""

import os
import sys
import time
import uuid
import json
import logging
import threading
from contextlib import contextmanager
from typing import List, Dict, Any

import psycopg2
import psycopg2.extras
import requests

from strands import Agent
from strands.models import OllamaModel
from strands.tools import tool
from strands.observability import get_metrics

# Optional: auto-load .env when running locally (already handled by docker-compose)
try:
    from dotenv import load_dotenv  # python-dotenv
    load_dotenv()  # loads variables from a .env file into os.environ if present
except Exception:
    pass


# ------------------------------- Logging --------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("strands-agent")
RUN_ID = os.getenv("RUN_ID", uuid.uuid4().hex[:8])
LOG_PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "240"))

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
EMB_MODEL = os.getenv("EMB_MODEL", "mxbai-embed-large:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

TEMPERATURE = float(os.getenv("STRANDS_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("STRANDS_MAX_TOKENS", "2048"))

MAX_CONCURRENCY = int(os.getenv("RL_MAX_CONCURRENCY", "2"))
REQS_PER_MIN = int(os.getenv("RL_RPM", "60"))
TOKENS_PER_MIN = int(os.getenv("RL_TPM", "100000"))

# ----------------------------- Helpers ----------------------------------------

def as_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def _norm_model(s: str) -> str:
    s = (s or "").strip()
    return s if ":" in s else f"{s}:latest"

# --------------------------- Model & Tool -------------------------------------

ollama_model = OllamaModel(
    model_name=_norm_model(LLM_MODEL),
    base_url=OLLAMA_BASE,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

@tool(name="retrieve_context", description="Retrieve top relevant code/context from Postgres RAG index for a natural-language question.")
def retrieve_context(question: str, top_k: int = 12) -> str:
    log.info(f"[run={RUN_ID}] [embedding] question: \"{preview(question)}\"")
    emb_url = OLLAMA_BASE.rstrip("/") + "/api/embeddings"
    with span("Embedding query"):
        r = requests.post(emb_url, json={"model": _norm_model(EMB_MODEL), "prompt": question}, timeout=30)
        r.raise_for_status()
        q_emb = r.json().get("embedding") or []
        log.info(f"[run={RUN_ID}] [embedding] len={len(q_emb)}")

    q_emb_lit = as_vector_literal(q_emb)

    with span("DB search (vector + lexical)"):
        with psycopg2.connect(DB_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = '15000ms';")
            try:
                used_unaccent = True
                sql = f"""
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
                    LIMIT {int(top_k)};
                """
                cur.execute(sql, (q_emb_lit, question, question))
            except psycopg2.Error:
                used_unaccent = False
                sql = f"""
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
                    LIMIT {int(top_k)};
                """
                cur.execute(sql, (q_emb_lit, question, question))

            rows = cur.fetchall()
            log.info(f"[run={RUN_ID}] [db] rows={len(rows)} used_unaccent={used_unaccent}")

    if not rows:
        return "// No results found for the question."

    stitched = "\n\n".join(
        f"// {r['repo_name']}:{r['path']}:{r['start_line']}-{r['end_line']} "
        f"[{r['symbol_signature'] or r['symbol_name'] or ''}]\n{r['content']}"
        for r in rows
    )
    return stitched

SYS = (
    "You are a senior code assistant. Use the provided repository context strictly.\n"
    "Cite file/line provenance already included in the context comments. "
    "If context is insufficient, say so briefly."
)

agent = Agent(
    model=ollama_model,
    tools=[retrieve_context],
    system_prompt=SYS,
    stream=False,
)

# ------------------------- Simple Rate Limiter --------------------------------

class TokenBucket:
    def __init__(self, capacity_per_min: int):
        self.capacity = max(1, capacity_per_min)
        self.tokens = float(self.capacity)
        self.rate_per_sec = self.capacity / 60.0
        self.lock = threading.Lock()
        self.last = time.time()

    def take(self, n: int) -> None:
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
                self.last = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
            time.sleep(0.02)

req_bucket = TokenBucket(REQS_PER_MIN)
tok_bucket = TokenBucket(TOKENS_PER_MIN)
_concurrency = threading.Semaphore(MAX_CONCURRENCY)

def _estimate_tokens(s: str) -> int:
    # crude estimate (~4 chars/token)
    return max(1, len(s) // 4)

def run_with_rate_limits(prompt: str) -> str:
    est_in = _estimate_tokens(prompt)
    req_bucket.take(1)
    tok_bucket.take(est_in)
    with _concurrency:
        with span("Agent.run"):
            result = agent(prompt)
    metrics = get_metrics()
    usage = getattr(metrics, "get_last_usage", lambda: None)()
    if usage:
        log.info(f"[run={RUN_ID}] [metrics] input_tokens={usage.input_tokens} output_tokens={usage.output_tokens} total={usage.total_tokens}")
    return result.text if hasattr(result, "text") else str(result)
