# app/strands_agent.py
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

Logging / Observability:
- LOG_LEVEL: INFO (default), DEBUG etc.
- LOG_PROMPTS: 1/0 — log system/user/answer previews + injection warnings
- LOG_OLLAMA_HTTP: 1/0 — log HTTP requests/responses to Ollama
- LOG_BODY_PREVIEW: int — max chars of bodies to log (default 1200)
- LOG_HEADERS: 1/0 — log HTTP headers (auth redacted)
- LOG_FORMAT=pretty|json — pretty emoji console vs JSON lines (see app/obs.py)
- LOG_COLOR=1|0 — ANSI color (pretty mode)
- LOG_PREVIEW_CHARS — preview truncation (default 240)

Retrieval knobs:
- FORCE_PRE_RETRIEVE: 1/0 — if 1, always fetch DB context and inject into the prompt before LLM
- PRE_RETRIEVE_TOPK: int — how many chunks to fetch when FORCE_PRE_RETRIEVE=1 (default 12)
- CONTEXT_MAX_CHARS: int — cap injected context length (default 24000)
- CONTEXT_HEADER / CONTEXT_FOOTER: optional strings wrapped around injected context

Exports:
- run_with_rate_limits(prompt: str) -> str
- retrieve_context(question: str, top_k: int = 12) -> str   (Strands tool)
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import logging
import threading
from typing import List, Dict, Any, Tuple
import re

import psycopg2
import psycopg2.extras
import requests

from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools import tool

# Local observability
from app import obs

# Optional: auto-load .env when running locally (already handled by docker-compose)
try:
    from dotenv import load_dotenv  # python-dotenv
    load_dotenv()
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
LOG_PROMPTS = os.getenv("LOG_PROMPTS", "1").lower() in ("1", "true", "yes")
LOG_OLLAMA_HTTP = os.getenv("LOG_OLLAMA_HTTP", "1").lower() in ("1", "true", "yes")
LOG_BODY_PREVIEW = int(os.getenv("LOG_BODY_PREVIEW", "1200"))
LOG_HEADERS = os.getenv("LOG_HEADERS", "0").lower() in ("1", "true", "yes")


def preview(txt: str, limit: int = LOG_PREVIEW_CHARS) -> str:
    if txt is None:
        return ""
    txt = str(txt).replace("\n", "\\n")
    return txt if len(txt) <= limit else txt[:limit] + f"... (+{len(txt)-limit} more)"


# ------------------------------ Security --------------------------------------

# Lightweight heuristic for prompt injection / policy breaks. Tunable.
_INJECT_PATTERNS = [
    r"(?i)\bignore (?:all|previous|above) (?:instructions|rules)\b",
    r"(?i)\boverride (?:system|developer) (?:prompt|message)\b",
    r"(?i)\breveal (?:system|developer) prompt\b",
    r"(?i)\bact as (?:sys|developer|admin)\b",
    r"(?i)\bdeveloper mode\b",
    r"(?i)\bexfiltrat(?:e|ion)\b",
    r"(?i)\bmake a (?:http|https|ftp) request\b",
    r"(?i)\bwrite files? to\b",
    r"(?i)\brun (?:bash|sh|powershell|cmd)\b",
]


def scan_for_injection(text: str) -> Tuple[bool, List[str]]:
    hits = []
    for pat in _INJECT_PATTERNS:
        if re.search(pat, text or ""):
            hits.append(pat)
    return (len(hits) > 0, hits)


# ------------------------------ HTTP Logging ----------------------------------

def _redact_headers(h: Dict[str, str]) -> Dict[str, str]:
    if not h:
        return {}
    masked = dict(h)
    for k in list(masked.keys()):
        if k.lower() in ("authorization", "x-api-key", "cookie", "set-cookie"):
            masked[k] = "***"
    return masked


def install_ollama_http_logger(base_url: str):
    """Monkey-patch requests to log all calls to OLLAMA_BASE (request+response), and forward to obs."""
    if not LOG_OLLAMA_HTTP:
        return
    base = (base_url or "").rstrip("/")
    if not base:
        return

    real_request = requests.sessions.Session.request

    def logged(self, method, url, **kwargs):
        t0 = time.perf_counter()
        try:
            if url.startswith(base):
                body = None
                if "json" in kwargs and kwargs["json"] is not None:
                    try:
                        body = json.dumps(kwargs["json"])[:LOG_BODY_PREVIEW]
                    except Exception:
                        body = str(kwargs["json"])[:LOG_BODY_PREVIEW]
                elif "data" in kwargs and kwargs["data"] is not None:
                    body = str(kwargs["data"])[:LOG_BODY_PREVIEW]
                h = _redact_headers(kwargs.get("headers", {}))
                if LOG_HEADERS:
                    log.info(f"[run={RUN_ID}] [HTTP→OLLAMA] {method} {url} headers={h} body={preview(body, LOG_BODY_PREVIEW)}")
                else:
                    log.info(f"[run={RUN_ID}] [HTTP→OLLAMA] {method} {url} body={preview(body, LOG_BODY_PREVIEW)}")
            resp = real_request(self, method, url, **kwargs)
            if url.startswith(base):
                dt = (time.perf_counter() - t0) * 1000
                try:
                    text_preview = resp.text[:LOG_BODY_PREVIEW]
                except Exception:
                    text_preview = "<non-text response>"
                log.info(f"[run={RUN_ID}] [HTTP←OLLAMA] {resp.status_code} in {dt:.1f}ms body={preview(text_preview, LOG_BODY_PREVIEW)}")
                # Forward to obs for pretty/JSON logs
                obs.log_http(url=url, method=method, status=resp.status_code, duration_ms=int(dt), body_preview=text_preview)
            return resp
        except Exception as e:
            if url.startswith(base):
                log.exception(f"[run={RUN_ID}] [HTTP OLLAMA ERROR] {method} {url}: {e}")
            raise

    # Idempotent patch (only once)
    if getattr(requests.sessions.Session.request, "__wrapped__", None) is None:
        logged.__wrapped__ = real_request  # type: ignore[attr-defined]
        requests.sessions.Session.request = logged  # type: ignore[assignment]


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

FORCE_PRE_RETRIEVE = os.getenv("FORCE_PRE_RETRIEVE", "1").lower() in ("1", "true", "yes")
PRE_RETRIEVE_TOPK = int(os.getenv("PRE_RETRIEVE_TOPK", "12"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "24000"))
CONTEXT_HEADER = os.getenv("CONTEXT_HEADER", "/* Repository context (top matches) */")
CONTEXT_FOOTER = os.getenv("CONTEXT_FOOTER", "/* End of repository context */")


# ----------------------------- Helpers ----------------------------------------

def as_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _norm_model(s: str) -> str:
    s = (s or "").strip()
    return s if ":" in s else f"{s}:latest"


# --------------------------- Retrieval (shared) --------------------------------

def _retrieve_context_impl(question: str, top_k: int = 12) -> str:
    # --- Embedding step (with logging + injection scan) ---
    if LOG_PROMPTS:
        inj, pats = scan_for_injection(question)
        if inj:
            log.warning(f"[run={RUN_ID}] [injection?] patterns={pats} on question=\"{preview(question)}\"")
        log.info(f"[run={RUN_ID}] [embedding] model={_norm_model(EMB_MODEL)} question_len={len(question)} preview=\"{preview(question)}\"")

    emb_url = OLLAMA_BASE.rstrip("/") + "/api/embeddings"
    with obs.span("Embedding query"):
        payload = {"model": _norm_model(EMB_MODEL), "prompt": question}
        log.debug(f"[run={RUN_ID}] [embedding] POST {emb_url} json_keys={list(payload.keys())}")
        r = requests.post(emb_url, json=payload, timeout=30)
        r.raise_for_status()
        q_emb = r.json().get("embedding") or []
        head = ", ".join(f"{x:.3f}" for x in q_emb[:6])
        tail = ", ".join(f"{x:.3f}" for x in q_emb[-6:])
        log.info(f"[run={RUN_ID}] [embedding] dim={len(q_emb)} head=[{head}] tail=[{tail}]")

    q_emb_lit = as_vector_literal(q_emb)

    # --- DB search (try with unaccent; on error, rollback and retry without it) ---
    with obs.span("DB search (vector + lexical)"):
        with psycopg2.connect(DB_DSN) as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
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
                        v.chunk_id AS id, v.repo_name, v.path, v.language,
                        v.symbol_name, v.symbol_kind, v.symbol_signature,
                        v.start_line, v.end_line, v.content,
                        1 - (v.embedding <=> (SELECT emb FROM q)) AS vec_score,
                        ts_rank_cd(v.content_tsv, (SELECT tsq FROM q)) AS lex_score
                      FROM v_chunk_search v
                      WHERE ( (SELECT tsq FROM q) @@ v.content_tsv OR (SELECT emb FROM q) IS NOT NULL )
                    )
                    SELECT *, (0.7*vec_score + 0.3*lex_score) AS s
                    FROM scored
                    ORDER BY s DESC
                    LIMIT {int(top_k)};
                """
                t0 = time.perf_counter()
                cur.execute(sql, (q_emb_lit, question, question))
                dt_ms = int((time.perf_counter() - t0) * 1000)
                rows = cur.fetchall()
                obs.log_db(sql, duration_ms=dt_ms, rows=len(rows))
            except psycopg2.errors.UndefinedFunction as e:
                # Likely unaccent() not available. Roll back and retry without it.
                used_unaccent = False
                log.warning(f"[run={RUN_ID}] [db] unaccent() missing; retrying without it ({e})")
                try:
                    conn.rollback()
                except Exception:
                    pass
                try:
                    cur.close()
                except Exception:
                    pass
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SET LOCAL statement_timeout = '15000ms';")
                sql = f"""
                    WITH q AS (
                      SELECT %s::vector AS emb,
                             plainto_tsquery('simple', %s) AS tsq,
                             %s AS qraw
                    ),
                    scored AS (
                      SELECT
                        v.chunk_id AS id, v.repo_name, v.path, v.language,
                        v.symbol_name, v.symbol_kind, v.symbol_signature,
                        v.start_line, v.end_line, v.content,
                        1 - (v.embedding <=> (SELECT emb FROM q)) AS vec_score,
                        ts_rank_cd(v.content_tsv, (SELECT tsq FROM q)) AS lex_score
                      FROM v_chunk_search v
                      WHERE ( (SELECT tsq FROM q) @@ v.content_tsv OR (SELECT emb FROM q) IS NOT NULL )
                    )
                    SELECT *, (0.7*vec_score + 0.3*lex_score) AS s
                    FROM scored
                    ORDER BY s DESC
                    LIMIT {int(top_k)};
                """
                t0 = time.perf_counter()
                cur.execute(sql, (q_emb_lit, question, question))
                dt_ms = int((time.perf_counter() - t0) * 1000)
                rows = cur.fetchall()
                obs.log_db(sql, duration_ms=dt_ms, rows=len(rows))
            except psycopg2.errors.UndefinedColumn as e:
                # Surface real column errors immediately (e.g., selecting v.id from view exposing chunk_id)
                log.error(f"[run={RUN_ID}] [db] column error: {e}")
                raise
            except psycopg2.Error as e:
                # Generic DB error path (still log + re-raise)
                log.error(f"[run={RUN_ID}] [db] unexpected error: {e}")
                raise

            # Compact top-5 preview with safe float formatting
            def _fnum(x: Any) -> float:
                try:
                    return float(x)
                except Exception:
                    return 0.0

            def _row_line(r):
                return (
                    f"{r.get('repo_name')}:{r.get('path')}:{r.get('start_line')}-{r.get('end_line')} "
                    f"s={_fnum(r.get('s')):.3f} vec={_fnum(r.get('vec_score')):.3f} lex={_fnum(r.get('lex_score')):.3f}"
                )

            preview_rows = "; ".join(_row_line(r) for r in rows[:5]) if rows else ""
            log.info(f"[run={RUN_ID}] [db] rows={len(rows)} used_unaccent={used_unaccent} top5=[{preview_rows}]")

    if not rows:
        return "// No results found for the question."

    stitched = "\n\n".join(
        f"// {r['repo_name']}:{r['path']}:{r['start_line']}-{r['end_line']} "
        f"[{r['symbol_signature'] or r['symbol_name'] or ''}]\n{r['content']}"
        for r in rows
    )
    log.debug(f"[run={RUN_ID}] [retrieve_context_impl] stitched_len={len(stitched)}")
    return stitched


# ------------------------------ Tool wrapper ----------------------------------

@tool(name="retrieve_context", description="Retrieve top relevant code/context from Postgres RAG index for a natural-language question.")
@obs.observe_tool("retrieve_context")
def retrieve_context(question: str, top_k: int = 12) -> str:
    return _retrieve_context_impl(question, top_k)


# ------------------------ Context augmentation --------------------------------

def _augment_with_context(user_prompt: str, top_k: int) -> Tuple[str, str]:
    """Fetch context and inject into the prompt. Returns (augmented_prompt, context_used)."""
    ctx = _retrieve_context_impl(user_prompt, top_k=top_k) or ""
    if ctx.startswith("// No results"):
        log.info(f"[run={RUN_ID}] [augment] no context found; sending raw user prompt")
        return user_prompt, ""
    # Cap context size
    if len(ctx) > CONTEXT_MAX_CHARS:
        ctx = ctx[:CONTEXT_MAX_CHARS] + f"\n// ...(truncated {len(ctx) - CONTEXT_MAX_CHARS} chars)"
    augmented = (
        f"{CONTEXT_HEADER}\n{ctx}\n{CONTEXT_FOOTER}\n\n"
        f"Question:\n{user_prompt}"
    )
    log.info(f"[run={RUN_ID}] [augment] injected_context_len={len(ctx)} augmented_len={len(augmented)}")
    if LOG_PROMPTS:
        log.debug(f"[run={RUN_ID}] [augment] augmented_preview=\"{preview(augmented)}\"")
    return augmented, ctx


# -------------------------------- System prompt --------------------------------

SYS = (
    "You are a senior code assistant. Use the provided repository context strictly.\n"
    "Cite file/line provenance already included in the context comments. "
    "If context is insufficient, say so briefly."
)


# ------------------------------- Agent setup -----------------------------------

# Install HTTP logger before any client calls
install_ollama_http_logger(OLLAMA_BASE)

ollama_model = OllamaModel(
    host=OLLAMA_BASE,
    model_id=_norm_model(LLM_MODEL),
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

agent = Agent(
    model=ollama_model,
    tools=[retrieve_context],
    system_prompt=SYS,
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


# --------------------------------- Run loop ------------------------------------

def run_with_rate_limits(prompt: str) -> str:
    # Pre-retrieve context if configured, so it's definitely present in the prompt.
    if FORCE_PRE_RETRIEVE:
        augmented_prompt, _ctx = _augment_with_context(prompt, PRE_RETRIEVE_TOPK)
        user_for_model = augmented_prompt
    else:
        user_for_model = prompt

    est_in = _estimate_tokens(user_for_model)
    req_bucket.take(1)
    tok_bucket.take(est_in)

    if LOG_PROMPTS:
        inj, pats = scan_for_injection(user_for_model)
        if inj:
            log.warning(f"[run={RUN_ID}] [injection?] patterns={pats} on user_prompt=\"{preview(user_for_model)}\"")
        log.info(
            f"[run={RUN_ID}] [agent] system_len={len(SYS)} user_len={len(user_for_model)} "
            f"system_preview=\"{preview(SYS)}\" user_preview=\"{preview(user_for_model)}\""
        )

    # Agent.run with concurrency guard + rich observability
    with _concurrency:
        obs.agent_start(SYS, user_for_model)
        with obs.span("Agent.run"):
            t0 = time.perf_counter()
            result = agent(user_for_model)
            agent_ms = int((time.perf_counter() - t0) * 1000)

    # Strands SDK v1.6+: metrics are on the AgentResult
    usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None) or {}
    if usage:
        log.info(
            f"[run={RUN_ID}] [metrics] input_tokens={usage.get('inputTokens')} "
            f"output_tokens={usage.get('outputTokens')} total={usage.get('totalTokens')}"
        )

    # "Behind Agent.run" probe
    out = result.text if hasattr(result, "text") else str(result)
    obs.log_model_call(
        model_id=_norm_model(LLM_MODEL),
        prompt=user_for_model,
        response_text=out,
        tokens_in=usage.get("inputTokens") if usage else None,
        tokens_out=usage.get("outputTokens") if usage else None,
        duration_ms=None  # you can pass agent_ms here if desired
    )

    if LOG_PROMPTS:
        log.info(f"[run={RUN_ID}] [agent→user] answer_len={len(out)} preview=\"{preview(out)}\"")

    obs.agent_end(out, tokens_in=usage.get("inputTokens") if usage else None,
                       tokens_out=usage.get("outputTokens") if usage else None)
    return out
