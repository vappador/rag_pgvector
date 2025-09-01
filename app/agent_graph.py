# app/agent_graph.py
"""
Verbose, step-by-step logging for the RAG agent, with prompt + brief reasoning display.

ENV knobs:
- LOG_LEVEL: DEBUG | INFO | WARNING | ERROR (default: INFO)
- LOG_PREVIEW_CHARS: int, max chars to show in previews (default: 240)
- OLLAMA_DEBUG: 1/true to debug preflight sets
- OLLAMA_AUTOPULL: 1/true to auto-pull missing models (default: 0)
- OLLAMA_HOST / OLLAMA_BASE_URL: Ollama base URL (default: http://ollama:11434)
- DB_DSN: postgres DSN (default: postgresql://rag:ragpwd@pg:5432/ragdb)

New/updated:
- SHOW_PROMPT: 1/true to print full prompt (default: 1)
- SAVE_PROMPT_PATH: where to save last prompt (default: /workspace/debug/last_prompt.md)
- MAX_CONTEXT_CHARS: max chars of stitched context (default: 20000)
- ROWS_LIMIT: retrieval limit (default: 12)
- VEC_WEIGHT / LEX_WEIGHT: retrieval score weights (default: 0.7 / 0.3)
- REASONING_BULLETS: max brief reasoning bullets (default: 4)

Run example:
  LOG_LEVEL=DEBUG docker compose exec app python /workspace/app/agent_graph.py
"""

import os
import sys
import json
import time
import uuid
import logging
from typing import TypedDict, Set
from contextlib import contextmanager
from urllib.request import urlopen, Request

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

SHOW_PROMPT = os.getenv("SHOW_PROMPT", "1").lower() in ("1", "true", "yes")
SAVE_PROMPT_PATH = os.getenv("SAVE_PROMPT_PATH", "/workspace/debug/last_prompt.md")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))
ROWS_LIMIT = int(os.getenv("ROWS_LIMIT", "12"))
VEC_WEIGHT = float(os.getenv("VEC_WEIGHT", "0.7"))
LEX_WEIGHT = float(os.getenv("LEX_WEIGHT", "0.3"))
REASONING_BULLETS = int(os.getenv("REASONING_BULLETS", "4"))

_ollama = ollama.Client(host=OLLAMA_BASE)

# ------------------------------ Tiny ANSI UI ----------------------------------

def _color(s: str, code: str) -> str:
    if os.getenv("NO_COLOR"):
        return s
    return f"\x1b[{code}m{s}\x1b[0m"

def banner(title: str, char: str = "─", code="36;1"):
    line = char * max(8, len(title) + 2)
    print(_color(f"\n{line}\n {title}\n{line}\n", code))

# ------------------------------ DB helpers ------------------------------------

def _table_cols(conn, table: str) -> Set[str]:
    sql = """
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema='public' AND table_name=%s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        return {r[0] for r in cur.fetchall()}

def _has_table(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name=%s LIMIT 1",
            (table,),
        )
        return cur.fetchone() is not None

def _has_unaccent(conn) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname='unaccent' LIMIT 1;")
            return cur.fetchone() is not None
    except Exception:
        return False

# ------------------------------ Ollama helpers --------------------------------

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

def _present_models_from_tags(tags_obj) -> set[str]:
    present: set[str] = set()
    if tags_obj is None:
        return present
    def _iter(d):
        if isinstance(d, dict):
            models = d.get("models")
            if isinstance(models, (list, tuple)):
                for m in models:
                    if isinstance(m, dict):
                        yield m
            elif any(k in d for k in ("name","model","tag")):
                yield d
        elif isinstance(d, (list, tuple)):
            for el in d:
                yield from _iter(el)
    for m in _iter(tags_obj):
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

def _truncate_center(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n// … (context truncated) …\n" + text[-keep:]

def retrieve(state: State) -> State:
    q = state["question"]
    log.info(f"[run={RUN_ID}] [embedding] question preview: \"{preview(q)}\"")

    # ----- Embed (robust) -----
    with span("Embedding query"):
        q_emb = []
        try:
            res = _ollama.embeddings(model=_norm_model(EMB_MODEL), prompt=q)
            q_emb = res.get("embedding") or []
        except Exception as e:
            log.warning(f"[run={RUN_ID}] [embedding] failed: {e}; continuing without vector")
            q_emb = []
        emb_len = len(q_emb)
        head = ", ".join(f"{x:.6f}" for x in q_emb[:6]) if emb_len else ""
        tail = ", ".join(f"{x:.6f}" for x in q_emb[-6:]) if emb_len > 6 else ""
        log.info(f"[run={RUN_ID}] [embedding] model={_norm_model(EMB_MODEL)} len={emb_len} "
                 f"head=[{head}] tail=[{tail}]")
    use_vec = emb_len > 0
    q_emb_lit = as_vector_literal(q_emb) if use_vec else None

    log.info(f"[run={RUN_ID}] [db] connecting to Postgres: {DB_DSN}")
    with span("DB search (vector + lexical)"):
        with psycopg2.connect(DB_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = '15000ms';")

            # Schema introspection
            unaccent_ok = _has_unaccent(conn)
            has_chunks = _has_table(conn, "code_chunks")
            if not has_chunks:
                raise RuntimeError("code_chunks table not found")

            cols_c = _table_cols(conn, "code_chunks")
            has_files = _has_table(conn, "code_files")
            cols_f = _table_cols(conn, "code_files") if has_files else set()
            do_join_files = has_files and "id" in cols_f and "file_id" in cols_c

            # Pick expressions
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

            if do_join_files and "language" in cols_f:
                lang_expr = "f.language"
            elif "language" in cols_c:
                lang_expr = "c.language"
            else:
                lang_expr = "NULL::text"

            if "symbol_simple" in cols_c and "symbol_name" in cols_c:
                symbol_name_expr = "COALESCE(c.symbol_simple, c.symbol_name)"
            elif "symbol_simple" in cols_c and "symbol" in cols_c:
                symbol_name_expr = "COALESCE(c.symbol_simple, c.symbol)"
            elif "symbol_name" in cols_c:
                symbol_name_expr = "c.symbol_name"
            elif "symbol" in cols_c:
                symbol_name_expr = "c.symbol"
            else:
                symbol_name_expr = "NULL::text"

            symbol_kind_expr = "c.symbol_kind" if "symbol_kind" in cols_c else "NULL::text"
            symbol_sig_expr  = "c.symbol_signature" if "symbol_signature" in cols_c else "NULL::text"

            # tsvector for content
            if "content_tsv" in cols_c:
                content_tsv_expr = "c.content_tsv"
            else:
                if unaccent_ok:
                    content_tsv_expr = "to_tsvector('simple', unaccent(c.content))"
                else:
                    content_tsv_expr = "to_tsvector('simple', c.content)"

            has_embedding = "embedding" in cols_c and use_vec

            # FROM/JOIN
            from_sql = "FROM code_chunks c"
            if do_join_files:
                from_sql += " JOIN code_files f ON f.id = c.file_id"

            # tsquery expr
            if unaccent_ok:
                tsq_expr = "plainto_tsquery('simple', unaccent(%s::text))"
            else:
                tsq_expr = "plainto_tsquery('simple', %s::text)"

            # CTE and score pieces
            if has_embedding:
                q_cte = f"SELECT %s::vector AS emb, {tsq_expr} AS tsq, %s::text AS qraw"
                params = (q_emb_lit, q, q)
                vec_score = "1 - (c.embedding <=> (SELECT emb FROM q))"
            else:
                q_cte = f"SELECT NULL::vector AS emb, {tsq_expr} AS tsq, %s::text AS qraw"
                params = (q, q)
                vec_score = "0.0"

            lex_score = f"ts_rank_cd({content_tsv_expr}, (SELECT tsq FROM q))"

            sql = f"""
                WITH q AS ( {q_cte} )
                SELECT
                  c.id,
                  {repo_expr}      AS repo_name,
                  {path_expr}      AS path,
                  {lang_expr}      AS language,
                  {symbol_name_expr} AS symbol_name,
                  {symbol_kind_expr} AS symbol_kind,
                  {symbol_sig_expr}  AS symbol_signature,
                  c.start_line, c.end_line,
                  c.content,
                  {vec_score} AS vec_score,
                  {lex_score} AS lex_score,
                  ({VEC_WEIGHT}*({vec_score}) + {LEX_WEIGHT}*({lex_score})) AS s
                {from_sql}
                WHERE ((SELECT tsq FROM q) @@ {content_tsv_expr}) OR (SELECT emb FROM q) IS NOT NULL
                ORDER BY s DESC
                LIMIT {ROWS_LIMIT};
            """

            rows = []
            try:
                cur.execute(sql, params)
            except psycopg2.Error as e:
                log.warning(f"[run={RUN_ID}] [db] first query failed: {type(e).__name__}: {e}")
                conn.rollback()

                tsq_expr_fb = "plainto_tsquery('simple', %s::text)"
                q_cte_fb = f"SELECT NULL::vector AS emb, {tsq_expr_fb} AS tsq, %s::text AS qraw"
                sql_fb = f"""
                    WITH q AS ( {q_cte_fb} )
                    SELECT
                      c.id,
                      {repo_expr}      AS repo_name,
                      {path_expr}      AS path,
                      {lang_expr}      AS language,
                      {symbol_name_expr} AS symbol_name,
                      {symbol_kind_expr} AS symbol_kind,
                      {symbol_sig_expr}  AS symbol_signature,
                      c.start_line, c.end_line,
                      c.content,
                      0.0 AS vec_score,
                      ts_rank_cd({content_tsv_expr}, (SELECT tsq FROM q)) AS lex_score,
                      ({LEX_WEIGHT}*ts_rank_cd({content_tsv_expr}, (SELECT tsq FROM q))) AS s
                    {from_sql}
                    WHERE (SELECT tsq FROM q) @@ {content_tsv_expr}
                    ORDER BY s DESC
                    LIMIT {ROWS_LIMIT};
                """
                cur.execute(sql_fb, (q, q))
                rows = cur.fetchall()
            else:
                rows = cur.fetchall()

            log.info(f"[run={RUN_ID}] [db] rows={len(rows)} (top {min(3, len(rows))} follows)")
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
            f"[{r.get('symbol_signature') or r.get('symbol_name') or ''}]\n{r['content']}"
            for r in rows
        )

    # Safety net: bound total context size to keep prompts manageable
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

def save_text(path: str, text: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        log.warning(f"[run={RUN_ID}] Could not save prompt to {path}: {e}")

def generate_answer(state: State) -> State:
    ctx = state["context"]
    q = state["question"]
    prompt = build_prompt(q, ctx)

    # Show & optionally save the *full* RAG prompt
    if SHOW_PROMPT:
        banner("RAG PROMPT")
        print(prompt)
    save_text(SAVE_PROMPT_PATH, prompt)

    log.info(f"[run={RUN_ID}] [llm] model={_norm_model(LLM_MODEL)}")
    log.info(f"[run={RUN_ID}] [llm] prompt sizes: context_chars={len(ctx)} question_chars={len(q)} sys_chars={len(SYS)}")
    log.debug(f"[run={RUN_ID}] [llm] prompt preview: \"{preview(prompt)}\"")

    tokens = 0
    chars = 0
    usage = {}
    with span("LLM generation (stream)"):
        content_parts = []
        try:
            banner("MODEL OUTPUT (stream)")
            stream = _ollama.chat(
                model=_norm_model(LLM_MODEL),
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                stream=True,
            )
            for chunk in stream:
                # Streamed tokens
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    # Light formatting: color headings if they appear
                    if "### Reasoning" in piece:
                        print(_color(piece, "33;1"), end="", flush=True)  # yellow
                    elif "### Answer" in piece:
                        print(_color(piece, "32;1"), end="", flush=True)  # green
                    else:
                        print(piece, end="", flush=True)
                    content_parts.append(piece)
                    tokens += 1
                    chars += len(piece)

                # Final usage stats are usually on the last chunk
                if chunk.get("done"):
                    # Typical keys: prompt_eval_count, eval_count, total_duration, load_duration
                    usage = {k: v for k, v in chunk.items() if k not in ("message", "done")}
            print()
        except Exception as e:
            log.error(f"[run={RUN_ID}] [llm] call failed: {e}")
            content_parts.append("I couldn't generate an answer because the local model was unavailable.")

    log.info(f"[run={RUN_ID}] [llm] streamed tokens={tokens} approx_chars={chars}")
    if usage:
        log.info(f"[run={RUN_ID}] [llm] usage={usage}")

    full = "".join(content_parts).strip()
    state["answer"] = full
    log.info(f"[run={RUN_ID}] [llm] final answer preview: \"{preview(state['answer'])}\"")
    return state

# --------------------------------- Main ---------------------------------------

def main():
    log.info(f"[run={RUN_ID}] Starting agent with:")
    log.info(f"[run={RUN_ID}]   DB_DSN={DB_DSN}")
    log.info(f"[run={RUN_ID}]   OLLAMA_BASE={OLLAMA_BASE}")
    log.info(f"[run={RUN_ID}]   EMB_MODEL={_norm_model(EMB_MODEL)} LLM_MODEL={_norm_model(LLM_MODEL)}")
    log.info(f"[run={RUN_ID}]   ROWS_LIMIT={ROWS_LIMIT}  VEC_WEIGHT={VEC_WEIGHT}  LEX_WEIGHT={LEX_WEIGHT}")
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

    banner("ANSWER (final)")
    print(out.get("answer", "").strip())

if __name__ == "__main__":
    main()
