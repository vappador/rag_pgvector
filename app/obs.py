# app/obs.py
"""
Observability helpers for Strands RAG agent.

Provides:
- Pretty or JSON logs (toggle with LOG_FORMAT=pretty|json).
- Emoji console output for fun, structured logs.
- Span context manager for timing operations.
- Tool decorator to auto-log start/end, duration, previews.
- Emitters for DB, HTTP, model, agent lifecycle events.
- Safe previews + header redaction.

Env vars:
- LOG_FORMAT=pretty|json
- LOG_COLOR=1|0
- LOG_BODY_PREVIEW=1200
- LOG_PREVIEW_CHARS=240
"""

import json
import os
import sys
import time
import uuid
import threading
from contextlib import contextmanager
from typing import Any, Dict, Callable, List

# ---------------------------- Config ----------------------------
LOG_FORMAT = os.getenv("LOG_FORMAT", "pretty").lower()   # pretty | json
LOG_BODY_PREVIEW = int(os.getenv("LOG_BODY_PREVIEW", "1200"))
RUN_ID = os.getenv("RUN_ID", uuid.uuid4().hex[:8])
PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "240"))
COLOR = os.getenv("LOG_COLOR", "1").lower() in ("1","true","yes")

RESET = "\x1b[0m" if COLOR else ""
BOLD  = "\x1b[1m" if COLOR else ""
DIM   = "\x1b[2m" if COLOR else ""
CYAN  = "\x1b[36m" if COLOR else ""
GREEN = "\x1b[32m" if COLOR else ""
YELL  = "\x1b[33m" if COLOR else ""
RED   = "\x1b[31m" if COLOR else ""
MAG   = "\x1b[35m" if COLOR else ""
BLUE  = "\x1b[34m" if COLOR else ""

def _now_ms() -> int: return int(time.time()*1000)

def _preview(s: Any, n: int = PREVIEW_CHARS) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", "\\n")
    return s if len(s)<=n else s[:n] + f"...(+{len(s)-n} more)"

def _redact_headers(h: Dict[str,str]) -> Dict[str,str]:
    if not h: return {}
    out = dict(h)
    for k in list(out.keys()):
        if k.lower() in ("authorization","x-api-key","cookie","set-cookie"):
            out[k] = "***"
    return out

# ---------------------------- Emitter/Bus ----------------------------
_subscribers: Dict[str, List[Callable[[Dict[str,Any]], None]]] = {}
_lock = threading.Lock()

def on(event: str, handler: Callable[[Dict[str,Any]], None]) -> None:
    with _lock:
        _subscribers.setdefault(event, []).append(handler)

def emit(event: str, payload: Dict[str,Any]) -> None:
    payload = {"event": event, "run": RUN_ID, "ts": _now_ms(), **payload}
    _console(payload)
    for h in _subscribers.get(event, []):
        try: h(payload)
        except Exception: pass

def _console(d: Dict[str,Any]) -> None:
    if LOG_FORMAT == "json":
        print(json.dumps(d, ensure_ascii=False), flush=True)
        return
    # pretty
    ev = d.get("event","evt").upper()
    icon = {
        "agent_start":"🤖","agent_end":"✅","tool_start":"🧰","tool_end":"🧪",
        "model_call":"🛰️","db_query":"🗄️","http_call":"🌐","warn":"⚠️","error":"💥",
        "trace":"📍"
    }.get(d.get("event",""), "•")
    line = f"{DIM}{d['ts']}{RESET} {BOLD}{icon} {ev}{RESET} {CYAN}[run={RUN_ID}]{RESET}"
    msg = d.get("msg")
    if msg: line += f" {msg}"
    for k in ("tool","sql","url","status","duration_ms","len_in","len_out","tokens_in","tokens_out"):
        if k in d: line += f" {DIM}{k}={d[k]}{RESET}"
    if "preview" in d: line += f" {MAG}preview={_preview(d['preview'])}{RESET}"
    print(line, flush=True)

# ---------------------------- Spans ----------------------------
@contextmanager
def span(name: str, **fields):
    t0 = _now_ms()
    emit("trace", {"msg": f"▶ {name} …", **fields})
    try:
        yield
    finally:
        dt = _now_ms()-t0
        emit("trace", {"msg": f"✔ {name} done", "duration_ms": dt, **fields})

# ---------------------------- Decorators ----------------------------
def _unwrap_tool_call(_args, _kwargs):
    """
    Normalize how different runtimes pass tool arguments.
    Returns (pos_args, kw_args) ready for calling the underlying function.
    """
    # Common wrappers used by tool frameworks
    for key in ("args", "tool_input", "arguments", "input"):
        if key in _kwargs:
            val = _kwargs[key]
            if isinstance(val, dict):
                return (), dict(val)  # all kwargs
            if isinstance(val, (list, tuple)):
                return list(val), {}  # all args
            # fallback: pass through as a single positional arg
            return (val,), {}
    # If nothing special, pass through as-is
    return _args, _kwargs

def observe_tool(tool_name: str):
    def deco(fn):
        def wrapper(*args, **kwargs):
            # For preview: show whatever we got (before unwrapping)
            emit("tool_start", {"tool": tool_name, "preview": kwargs or args})
            t0 = _now_ms()
            try:
                call_args, call_kwargs = _unwrap_tool_call(args, kwargs)
                out = fn(*call_args, **call_kwargs)
                emit("tool_end", {
                    "tool": tool_name,
                    "duration_ms": _now_ms() - t0,
                    "len_out": len(str(out)) if out is not None else 0,
                    "preview": out,
                })
                return out
            except Exception as e:
                emit("error", {"tool": tool_name, "msg": f"tool error: {e}"})
                raise
        return wrapper
    return deco

# ---------------------------- Probes ----------------------------
def log_model_call(model_id: str, prompt: str, response_text: str = "", **metrics):
    emit("model_call", {
        "model": model_id,
        "len_in": len(prompt or ""),
        "len_out": len(response_text or ""),
        "preview": response_text or "",
        **metrics
    })

def log_db(sql: str, duration_ms: int, rows: int = 0):
    emit("db_query", {"sql": sql.strip().splitlines()[0][:140]+" …", "duration_ms": duration_ms, "rows": rows})

def log_http(url: str, method: str, status: int, duration_ms: int, body_preview: str = ""):
    emit("http_call", {"url": url, "status": status, "duration_ms": duration_ms,
                       "preview": body_preview, "method": method})

def agent_start(system: str, user: str):
    emit("agent_start", {"preview": f"sys={_preview(system)} || user={_preview(user)}"})

def agent_end(answer: str, tokens_in: int = None, tokens_out: int = None):
    emit("agent_end", {"preview": answer, "tokens_in": tokens_in, "tokens_out": tokens_out})
