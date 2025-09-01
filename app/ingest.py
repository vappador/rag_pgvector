# app/ingest.py
"""
Repo ingestion for code-focused RAG (Postgres + pgvector).

What this version does:
- Early chunking (AST-first; windowed fallback) to avoid server-side embedding truncation.
- Rich logs, including when fallback goes into windowed mode.
- Embeddings via Ollama with explicit host (OLLAMA_HOST/OLLAMA_BASE_URL).
- **Edges generation**:
  - Intra-file call edges after chunk upserts.
  - Cross-file call edges via a DB stored procedure (build_cross_file_call_edges).
- New metadata & indices to improve cross-file resolution:
  - code_files.module, code_chunks.symbol_simple, code_chunks.symbol_signature
  - file_imports (normalized import surface)
  - code_symbols (export/definition surface)
- Flags:
  - --build-edges / --no-build-edges (default: on). Env: BUILD_EDGES=1/0.
  - --embed to compute embeddings.
  - --report (or REPORT=1) to print a read-only ingestion quality report (key metrics).

NOTE: Cross-file edge logic is materialized in SQL and invoked once at end of run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import psycopg2
import psycopg2.extras

# ------------------------------ Logging ---------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ingest")

# ------------------------------ Optional deps ---------------------------------

# Tree-sitter (optional) with robust fallback:
# 1) Try loading your compiled my-languages.so (TS_LANG_SO / build/my-languages.so)
# 2) If incompatible (e.g., "Language version 15" vs runtime 13-14), fall back to
#    prebuilt grammars from `tree_sitter_languages`.
TS_AVAILABLE = False
PARSER_BY_LANG: Dict[str, "Parser"] = {}
LANG_SO_DEFAULT = str(Path("build") / "my-languages.so")
TS_LANG_SO = os.environ.get("TS_LANG_SO", LANG_SO_DEFAULT)

def _load_ts_parsers() -> Dict[str, "Parser"]:
    parsers: Dict[str, "Parser"] = {}
    try:
        from tree_sitter import Language, Parser  # type: ignore
    except Exception as e:
        log.info(f"Tree-sitter unavailable; using fallback chunking. Reason: {e}")
        return parsers

    # 1) Try custom shared object first (if present)
    so_path = Path(TS_LANG_SO) if TS_LANG_SO else None
    tried_custom = False
    if so_path and so_path.exists():
        tried_custom = True
        try:
            langs = ("java", "typescript", "javascript", "python")
            for name in langs:
                ts_lang = Language(str(so_path), name)
                p = Parser()
                p.set_language(ts_lang)
                parsers[name] = p
            log.info(f"Tree-sitter enabled via {so_path} for: {', '.join(parsers.keys())}")
            return parsers
        except Exception as e:
            # Typical error: "Incompatible Language version 15. Must be between 13 and 14"
            log.warning(
                f"Failed to load {so_path} ({e}). Falling back to prebuilt grammars."
            )

    # 2) Fallback to prebuilt grammars that match the installed python binding
    try:
        from tree_sitter import Parser  # re-import safe
        from tree_sitter_languages import get_language  # type: ignore

        mapping = {
            "java": "java",
            "typescript": "typescript",
            "javascript": "javascript",
            "python": "python",
        }
        for name, lang_id in mapping.items():
            ts_lang = get_language(lang_id)
            p = Parser()
            p.set_language(ts_lang)
            parsers[name] = p

        log.info(f"Tree-sitter prebuilt grammars enabled for: {', '.join(parsers.keys())}")
        return parsers
    except Exception as e:
        # If we tried the custom .so and prebuilt also failed, report the reason
        reason = f"custom={tried_custom}, error={e}"
        log.info(f"Tree-sitter unavailable; using fallback chunking. Reason: {reason}")
        return {}

# initialize
PARSER_BY_LANG = _load_ts_parsers()
TS_AVAILABLE = len(PARSER_BY_LANG) > 0


# Ollama (optional, only used when --embed)
OLLAMA_AVAILABLE = False
OLLAMA_CLIENT = None
try:
    import ollama  # type: ignore

    OLLAMA_AVAILABLE = True
    _OLLAMA_HOST = (
        os.environ.get("OLLAMA_HOST")
        or os.environ.get("OLLAMA_BASE_URL")
        or "http://ollama:11434"
    )
    try:
        OLLAMA_CLIENT = ollama.Client(host=_OLLAMA_HOST)
        log.info(f"Ollama client host = {_OLLAMA_HOST}")
    except Exception as e:
        log.warning(f"Failed to init Ollama client at {_OLLAMA_HOST}: {e}")
        OLLAMA_CLIENT = None
except Exception as e:
    log.info(f"Ollama python package not available: {e}")
    OLLAMA_AVAILABLE = False

# ------------------------------ Helpers ---------------------------------------

LANG_EXTS = {
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
}

def detect_language(path: Path) -> Optional[str]:
    return LANG_EXTS.get(path.suffix.lower())

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def git_root(path: Path) -> Optional[Path]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=str(path)
        ).decode().strip()
        return Path(out)
    except Exception:
        return None

def git_head_sha(repo_dir: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir))
            .decode()
            .strip()
        )
    except Exception:
        return None

def git_changed_files(repo_dir: Path, base_ref: str) -> List[Path]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", base_ref, "HEAD"], cwd=str(repo_dir)
        ).decode()
        return [repo_dir / p for p in out.splitlines() if p.strip()]
    except Exception:
        return []

# ------------------------------ AST chunking ----------------------------------

@dataclass
class Chunk:
    language: str
    symbol_kind: str          # class|method|function|module|module_part
    symbol_name: str          # e.g., Person#updateEmail or FQN for Java
    symbol_signature: Optional[str]  # e.g., Person#updateEmail(email: str)
    start_line: int
    end_line: int
    content: str
    imports: List[str]
    calls: List[str]
    ast: Dict[str, Any]

def _node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

def _walk(node) -> Iterable:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for i in reversed(range(n.child_count)):
            stack.append(n.child(i))

# ------------ Signature helpers (language-agnostic, header-based) ------------

_WS_RE = re.compile(r"\s+")

def _extract_params_from_header(body: str) -> str:
    """
    Heuristic: take text up to the first block/colon, then grab first (...) group.
    Works for Java/TS/JS (stops at '{') and Python (stops at ':').
    Falls back to empty string if none found.
    """
    if not body:
        return ""
    header = body
    for stop in ("{", ":\n", ":\r", ":\r\n", ":\t", ":\f", ":\v", ": "):
        if stop in header:
            header = header.split(stop, 1)[0]
            break
    m = re.search(r"\((.*?)\)", header, flags=re.S)
    if not m:
        return ""
    # normalize whitespace inside params
    params = _WS_RE.sub(" ", m.group(1).strip())
    # trim trailing commas etc.
    params = params.strip().strip(",")
    return params

def _build_signature(symbol_name: str, content: str) -> str:
    params = _extract_params_from_header(content)
    return f"{symbol_name}({params})" if params else f"{symbol_name}()"

# -------- Enhanced call extraction helpers --------

def _callee_name_from_call_expr_ts(src: bytes, node) -> Optional[str]:
    callee = node.child_by_field_name("function") or node.child_by_field_name("callee")
    if callee is None:
        return None
    if callee.type == "member_expression":
        obj = callee.child_by_field_name("object")
        prop = callee.child_by_field_name("property")
        if obj is not None and prop is not None:
            return _node_text(src, obj) + "." + _node_text(src, prop)
    return _node_text(src, callee)

def _py_call_name(src: bytes, call_node) -> Optional[str]:
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "attribute":
        parts = []
        cur = fn
        while cur is not None:
            if cur.type == "attribute":
                attr = cur.child_by_field_name("attribute")
                obj = cur.child_by_field_name("object")
                if attr is not None:
                    parts.append(_node_text(src, attr))
                cur = obj
            else:
                parts.append(_node_text(src, cur))
                break
        parts = list(reversed(parts))
        return ".".join(parts)
    return _node_text(src, fn)

# ------------------------------ Language chunkers ------------------------------

def chunk_file_ast(path: Path, lang: str) -> List[Chunk]:
    """
    AST-aware chunking per language. Gracefully returns [] if parser missing.
    """
    if not TS_AVAILABLE or lang not in PARSER_BY_LANG:
        return []

    parser = PARSER_BY_LANG[lang]
    source = path.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node

    chunks: List[Chunk] = []

    if lang == "java":
        # package
        java_package = None
        for node in _walk(root):
            if node.type == "package_declaration":
                java_package = (
                    _node_text(source, node)
                    .strip()
                    .replace("package", "")
                    .rstrip(";")
                    .strip()
                )
                break

        # imports
        imports = set()
        for n in _walk(root):
            if n.type == "import_declaration":
                text = _node_text(source, n).strip().rstrip(";")
                if text.startswith("import"):
                    imports.add(text.replace("import", "", 1).strip())

        # classes -> methods/constructors
        for class_node in _walk(root):
            if class_node.type in ("class_declaration", "interface_declaration"):
                class_name = None
                for ch in _walk(class_node):
                    if ch.type == "identifier":
                        class_name = _node_text(source, ch)
                        break
                if not class_name:
                    continue

                for n in _walk(class_node):
                    if n.type in ("method_declaration", "constructor_declaration"):
                        meth_name = None
                        for ch in _walk(n):
                            if ch.type == "identifier":
                                meth_name = _node_text(source, ch)
                                break
                        if not meth_name:
                            continue

                        calls = set()

                        def add_call(name: Optional[str]):
                            if not name:
                                return
                            name = name.strip()
                            if not name:
                                return
                            calls.add(name)
                            if "." in name:
                                calls.add(name.split(".")[-1])

                        for ch in _walk(n):
                            t = ch.type
                            if t == "method_invocation":
                                sel = ch.child_by_field_name("object")
                                m = ch.child_by_field_name("name")
                                if sel is not None and m is not None:
                                    add_call(_node_text(source, sel) + "." + _node_text(source, m))
                                elif m is not None:
                                    add_call(_node_text(source, m))
                            if t == "field_access":
                                fld = ch.child_by_field_name("field")
                                if fld is not None:
                                    add_call(_node_text(source, fld))
                            if t == "object_creation_expression":
                                typ = ch.child_by_field_name("type")
                                if typ is not None:
                                    add_call(_node_text(source, typ))  # constructor Type

                        start_line = n.start_point[0] + 1
                        end_line = n.end_point[0] + 1
                        content = _node_text(source, n)
                        fq_sym = f"{java_package}.{class_name}#{meth_name}" if java_package else f"{class_name}#{meth_name}"
                        signature = _build_signature(fq_sym, content)
                        ast_meta = {"class": class_name, "method": meth_name}
                        if java_package:
                            ast_meta["package"] = java_package
                        chunks.append(
                            Chunk(
                                language=lang,
                                symbol_kind="method",
                                symbol_name=fq_sym,
                                symbol_signature=signature,
                                start_line=start_line,
                                end_line=end_line,
                                content=content,
                                imports=sorted(imports),
                                calls=sorted(calls),
                                ast=ast_meta,
                            )
                        )

    elif lang in ("typescript", "javascript"):
        imports = set()
        for n in _walk(root):
            if n.type == "import_declaration":
                imports.add(_node_text(source, n).strip())

        # naive export capture
        exports = set()
        for n in _walk(root):
            if n.type in ("export_statement", "export_clause", "export_specifier", "lexical_declaration"):
                text = _node_text(source, n)
                if "export" in text:
                    exports.add(text)

        # function declarations
        for n in _walk(root):
            if n.type == "function_declaration":
                name = None
                for ch in _walk(n):
                    if ch.type == "identifier":
                        name = _node_text(source, ch)
                        break
                if name:
                    calls = set()
                    for ch in _walk(n):
                        if ch.type == "call_expression":
                            nm = _callee_name_from_call_expr_ts(source, ch)
                            if nm:
                                calls.add(nm)
                                if "." in nm:
                                    calls.add(nm.split(".")[-1])
                    signature = _build_signature(name, _node_text(source, n))
                    ast_meta = {"function": name, "exports": sorted(list(exports))}
                    chunks.append(
                        Chunk(
                            language=lang,
                            symbol_kind="function",
                            symbol_name=name,
                            symbol_signature=signature,
                            start_line=n.start_point[0] + 1,
                            end_line=n.end_point[0] + 1,
                            content=_node_text(source, n),
                            imports=sorted(imports),
                            calls=sorted(calls),
                            ast=ast_meta,
                        )
                    )

        # class + methods
        for n in _walk(root):
            if n.type == "class_declaration":
                class_name = None
                for ch in _walk(n):
                    if ch.type == "identifier":
                        class_name = _node_text(source, ch)
                        break
                if not class_name:
                    continue
                for m in _walk(n):
                    if m.type == "method_definition":
                        meth_name = None
                        for idn in _walk(m):
                            if idn.type in ("property_identifier", "identifier"):
                                meth_name = _node_text(source, idn)
                                break
                        if not meth_name:
                            continue
                        calls = set()
                        for ch in _walk(m):
                            if ch.type == "call_expression":
                                nm = _callee_name_from_call_expr_ts(source, ch)
                                if nm:
                                    calls.add(nm)
                                    if "." in nm:
                                        calls.add(nm.split(".")[-1])
                        sym_name = f"{class_name}#{meth_name}"
                        signature = _build_signature(sym_name, _node_text(source, m))
                        ast_meta = {"class": class_name, "method": meth_name, "exports": sorted(list(exports))}
                        chunks.append(
                            Chunk(
                                language=lang,
                                symbol_kind="method",
                                symbol_name=sym_name,
                                symbol_signature=signature,
                                start_line=m.start_point[0] + 1,
                                end_line=m.end_point[0] + 1,
                                content=_node_text(source, m),
                                imports=sorted(imports),
                                calls=sorted(calls),
                                ast=ast_meta,
                            )
                        )

    elif lang == "python":
        imports = set()
        for n in _walk(root):
            if n.type in ("import_statement", "import_from_statement"):
                imports.add(_node_text(source, n).strip())

        # top-level functions
        for n in _walk(root):
            if n.type == "function_definition":
                name = None
                for ch in _walk(n):
                    if ch.type == "identifier":
                        name = _node_text(source, ch)
                        break
                if name:
                    calls = set()
                    for ch in _walk(n):
                        if ch.type == "call":
                            nm = _py_call_name(source, ch)
                            if nm:
                                calls.add(nm)
                                if "." in nm:
                                    calls.add(nm.split(".")[-1])
                    signature = _build_signature(name, _node_text(source, n))
                    chunks.append(
                        Chunk(
                            language=lang,
                            symbol_kind="function",
                            symbol_name=name,
                            symbol_signature=signature,
                            start_line=n.start_point[0] + 1,
                            end_line=n.end_point[0] + 1,
                            content=_node_text(source, n),
                            imports=sorted(imports),
                            calls=sorted(calls),
                            ast={"function": name},
                        )
                    )

        # classes + methods
        for n in _walk(root):
            if n.type == "class_definition":
                class_name = None
                for ch in _walk(n):
                    if ch.type == "identifier":
                        class_name = _node_text(source, ch)
                        break
                if not class_name:
                    continue
                for m in _walk(n):
                    if m.type == "function_definition":
                        meth_name = None
                        for idn in _walk(m):
                            if idn.type == "identifier":
                                meth_name = _node_text(source, idn)
                                break
                        if not meth_name:
                            continue
                        calls = set()
                        for ch in _walk(m):
                            if ch.type == "call":
                                nm = _py_call_name(source, ch)
                                if nm:
                                    calls.add(nm)
                                    if "." in nm:
                                        calls.add(nm.split(".")[-1])
                        sym_name = f"{class_name}#{meth_name}"
                        signature = _build_signature(sym_name, _node_text(source, m))
                        chunks.append(
                            Chunk(
                                language=lang,
                                symbol_kind="method",
                                symbol_name=sym_name,
                                symbol_signature=signature,
                                start_line=m.start_point[0] + 1,
                                end_line=m.end_point[0] + 1,
                                content=_node_text(source, m),
                                imports=sorted(imports),
                                calls=sorted(calls),
                                ast={"class": class_name, "method": meth_name},
                            )
                        )

    return chunks

# ------------------------------ Fallback chunking ------------------------------

def _windowed_segments(text: str, max_chars: int = 1800, overlap_chars: int = 300) -> List[Tuple[int, int, str]]:
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    segs: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        segs.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return segs

def chunk_file_fallback(path: Path, lang: str, max_chars: int, overlap_chars: int, verbose_filename: str) -> List[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines() or [""]
    segs = _windowed_segments(text, max_chars=max_chars, overlap_chars=overlap_chars)
    if len(segs) == 1:
        log.debug(f"[fallback] {verbose_filename}: single-window (<= {max_chars} chars)")
        start_idx, end_idx, seg = segs[0]
        return [
            Chunk(
                language=lang or "unknown",
                symbol_kind="module",
                symbol_name=path.name,
                symbol_signature=f"{path.name}()",
                start_line=1,
                end_line=len(lines),
                content=seg,
                imports=[],
                calls=[],
                ast={"file": str(path), "language": lang or "unknown"},
            )
        ]
    log.info(
        f"[windowed] {verbose_filename}: size={len(text)} chars -> "
        f"{len(segs)} segments (max={max_chars}, overlap={overlap_chars})"
    )
    chunks: List[Chunk] = []
    for idx, (start_idx, end_idx, seg) in enumerate(segs, start=1):
        seg_lines = seg.splitlines()
        chunks.append(
            Chunk(
                language=lang or "unknown",
                symbol_kind="module_part",
                symbol_name=f"{path.name}#part{idx}",
                symbol_signature=f"{path.name}#part{idx}()",
                start_line=1,
                end_line=len(seg_lines),
                content=seg,
                imports=[],
                calls=[],
                ast={"file": str(path), "language": lang or "unknown", "part": idx, "offset_range": [start_idx, end_idx]},
            )
        )
    return chunks

def make_chunks(path: Path, max_chars: int = 1800, overlap_chars: int = 300) -> List[Chunk]:
    lang = detect_language(path)
    if not lang:
        return []
    chunks = chunk_file_ast(path, lang)
    if chunks:
        return chunks
    return chunk_file_fallback(path=path, lang=lang, max_chars=max_chars, overlap_chars=overlap_chars, verbose_filename=str(path))

# ------------------------------ DB helpers ------------------------------------

def get_or_create_repository(cur, name: str, url: str) -> int:
    cur.execute(
        """
        INSERT INTO repositories (name, url)
        VALUES (%s, %s)
        ON CONFLICT (name) DO UPDATE SET url = EXCLUDED.url
        RETURNING id
        """,
        (name, url),
    )
    return cur.fetchone()[0]

def update_repository_sha(cur, repo_id: int, sha: Optional[str]) -> None:
    if not sha:
        return
    cur.execute("UPDATE repositories SET last_ingested_sha = %s WHERE id = %s", (sha, repo_id))

def upsert_code_file(cur, repository_id: int, rel_path: str, language: Optional[str], size_bytes: int, checksum: Optional[str], module: Optional[str]) -> int:
    cur.execute(
        """
        INSERT INTO code_files (repository_id, path, language, size_bytes, checksum, module)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (repository_id, path) DO UPDATE
          SET language = EXCLUDED.language,
              size_bytes = EXCLUDED.size_bytes,
              checksum = EXCLUDED.checksum,
              module = EXCLUDED.module,
              updated_at = now()
        RETURNING id
        """,
        (repository_id, rel_path, language, size_bytes, checksum, module),
    )
    return cur.fetchone()[0]

def ensure_chunk_edges_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_edges (
          id SERIAL PRIMARY KEY,
          src_chunk_id INTEGER NOT NULL,
          dst_chunk_id INTEGER NOT NULL,
          edge_type TEXT NOT NULL,
          weight REAL DEFAULT 1.0,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunk_edges_src_dst_type
        ON chunk_edges (src_chunk_id, dst_chunk_id, edge_type);
        """
    )

def insert_edge_if_absent(cur, src_id: int, dst_id: int, edge_type: str, weight: float = 1.0):
    cur.execute(
        """
        INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, edge_type, weight)
        SELECT %s, %s, %s, %s
        WHERE NOT EXISTS (
          SELECT 1 FROM chunk_edges
          WHERE src_chunk_id = %s AND dst_chunk_id = %s AND edge_type = %s
        )
        """,
        (src_id, dst_id, edge_type, weight, src_id, dst_id, edge_type),
    )

def simple_symbol_name(symbol_name: str) -> str:
    if "#" in symbol_name:
        return symbol_name.split("#", 1)[1] or symbol_name
    return symbol_name

def upsert_code_chunk(cur, file_id: int, c: Chunk, embed: bool, embed_model: Optional[str], embed_max_chars: int) -> int:
    content_hash = sha256_text(f"{c.symbol_kind}|{c.symbol_name}|{c.start_line}|{c.end_line}|{c.content}")

    embedding = None
    embedding_model = None
    embedding_hash = None

    if embed:
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Embedding requested but python 'ollama' package not available.")
        if OLLAMA_CLIENT is None:
            raise RuntimeError("Embedding requested but Ollama client failed to initialize (check OLLAMA_HOST/OLLAMA_BASE_URL).")

        model = embed_model or "mxbai-embed-large"
        text_for_embed = c.content if len(c.content) <= embed_max_chars else c.content[:embed_max_chars]
        if len(c.content) > embed_max_chars:
            log.debug(f"[embed-truncate] {c.symbol_name}: {len(c.content)} -> {embed_max_chars} chars")

        try:
            resp = OLLAMA_CLIENT.embeddings(model=model, prompt=text_for_embed)
            vec = resp.get("embedding")
            if vec and isinstance(vec, list):
                embedding = vec
                embedding_model = model
                embedding_hash = sha256_text(text_for_embed)
        except Exception as e:
            log.warning(f"[embed-warn] {c.symbol_name}: {e}")

    symbol_simple = simple_symbol_name(c.symbol_name)

    cur.execute(
        """
        INSERT INTO code_chunks
          (file_id, start_line, end_line, content, content_tsv,
           language, symbol_kind, symbol_name, symbol_simple, symbol_signature,
           imports, calls, ast,
           embedding, embedding_model, embedding_hash, content_hash)
        VALUES
          (%s, %s, %s, %s, to_tsvector('simple', unaccent(%s)),
           %s, %s, %s, %s, %s,
           %s, %s, %s,
           %s, %s, %s, %s)
        ON CONFLICT (file_id, start_line, end_line) DO UPDATE SET
           content = EXCLUDED.content,
           content_tsv = EXCLUDED.content_tsv,
           language = EXCLUDED.language,
           symbol_kind = EXCLUDED.symbol_kind,
           symbol_name = EXCLUDED.symbol_name,
           symbol_simple = EXCLUDED.symbol_simple,
           symbol_signature = EXCLUDED.symbol_signature,
           imports = EXCLUDED.imports,
           calls = EXCLUDED.calls,
           ast = EXCLUDED.ast,
           content_hash = EXCLUDED.content_hash,
           embedding = COALESCE(EXCLUDED.embedding, code_chunks.embedding),
           embedding_model = COALESCE(EXCLUDED.embedding_model, code_chunks.embedding_model),
           embedding_hash = COALESCE(EXCLUDED.embedding_hash, code_chunks.embedding_hash),
           updated_at = now()
        RETURNING id
        """,
        (
            file_id,
            c.start_line,
            c.end_line,
            c.content,
            c.content,
            c.language,
            c.symbol_kind,
            c.symbol_name,
            symbol_simple,
            (c.symbol_signature or f"{c.symbol_name}()"),
            c.imports,
            c.calls,
            psycopg2.extras.Json(c.ast),
            embedding,
            embedding_model,
            embedding_hash,
            content_hash,
        ),
    )
    return cur.fetchone()[0]

# ------------------------------ New: imports & symbols -------------------------

def parse_import(raw: str, lang: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (module, symbol) best-effort.
    """
    if not raw:
        return (None, None)
    try:
        text = raw.strip()
        if lang == "java":
            t = text.replace("import", "").replace("static", "").strip().rstrip(";")
            if "." in t:
                mod, sym = t.rsplit(".", 1)
                return (mod.strip(), sym.strip())
            else:
                return (t, None)

        if lang in ("typescript", "javascript"):
            mod = None
            sym = None
            if " from " in text:
                after_from = text.split(" from ", 1)[1].strip()
                quote = "'" if "'" in after_from else '"'
                if quote in after_from:
                    mod = after_from.split(quote)[1]
            if "import {" in text:
                inside = text.split("import {",1)[1].split("}",1)[0]
                first = inside.split(",")[0].strip()
                if " as " in first:
                    first = first.split(" as ",1)[0].strip()
                sym = first or None
            elif "import * as" in text:
                sym = None
            elif text.startswith("import "):
                rest = text[len("import "):].strip()
                if rest and not rest.startswith("{") and not rest.startswith("*") and " from " in rest:
                    default_sym = rest.split(" from ",1)[0].strip()
                    if default_sym and default_sym not in ("", ";"):
                        sym = None
            return (mod, sym)

        if lang == "python":
            if text.startswith("from "):
                parts = text.split()
                if len(parts) >= 4 and parts[0] == "from" and parts[2] == "import":
                    mod = parts[1].strip()
                    sym = parts[3].split(",")[0].strip()
                    return (mod, sym)
            if text.startswith("import "):
                mod = text.split()[1].split(",")[0].strip()
                return (mod, None)
    except Exception:
        pass
    return (None, None)

def upsert_file_imports(cur, file_id: int, imports: List[str], lang: Optional[str]) -> None:
    for raw in imports or []:
        module, symbol = parse_import(raw, lang)
        cur.execute(
            "INSERT INTO file_imports(file_id, raw_import, module, symbol) VALUES (%s, %s, %s, %s)",
            (file_id, raw, module, symbol),
        )

def _is_exported(c: Chunk) -> bool:
    try:
        ex = c.ast.get("exports")
        if isinstance(ex, list):
            sname = simple_symbol_name(c.symbol_name)
            return any(sname in e for e in ex)
    except Exception:
        pass
    return c.symbol_kind == "function" and c.language == "python"

def upsert_code_symbol(cur, repo_id: int, file_id: int, chunk_id: int, c: Chunk, module: Optional[str]) -> None:
    class_name = None
    if isinstance(c.ast, dict):
        class_name = c.ast.get("class")
    symbol_full = c.symbol_name
    cur.execute(
        """
        INSERT INTO code_symbols(repository_id, file_id, chunk_id, language, module, class_name, symbol_simple, symbol_full, exported)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (chunk_id) DO UPDATE SET
           language = EXCLUDED.language,
           module = EXCLUDED.module,
           class_name = EXCLUDED.class_name,
           symbol_simple = EXCLUDED.symbol_simple,
           symbol_full = EXCLUDED.symbol_full,
           exported = EXCLUDED.exported
        """,
        (
            repo_id, file_id, chunk_id, c.language, module, class_name,
            simple_symbol_name(c.symbol_name), symbol_full, _is_exported(c)
        )
    )

# ------------------------------ Module derivation ------------------------------

def derive_module(repo_root: Optional[Path], file_path: Path, lang: Optional[str], java_package: Optional[str]) -> Optional[str]:
    try:
        if lang == "java":
            if java_package:
                return java_package
            p = str(file_path).replace("\\", "/")
            if "/java/" in p:
                after = p.split("/java/",1)[1]
                if "/" in after:
                    parts = after.split("/")
                    return ".".join(parts[:-1])
            if repo_root and file_path.is_relative_to(repo_root):
                rel = file_path.relative_to(repo_root)
                parts = list(rel.parts)
                if parts:
                    return ".".join(parts[:-1])
            return None
        if lang == "python":
            if repo_root and file_path.is_relative_to(repo_root):
                rel = file_path.relative_to(repo_root)
                parts = list(rel.parts)
                if parts and parts[-1].endswith(".py"):
                    parts[-1] = parts[-1][:-3]
                return ".".join(parts[:-1]) if len(parts) > 1 else parts[0].replace(".py","")
            return file_path.stem
        if lang in ("typescript","javascript"):
            if repo_root and file_path.is_relative_to(repo_root):
                rel = file_path.relative_to(repo_root)
                p = str(rel)
            else:
                p = str(file_path)
            for ext in (".ts",".tsx",".js",".jsx"):
                if p.endswith(ext):
                    p = p[: -len(ext)]
                    break
            return p.replace("\\","/")
        return None
    except Exception:
        return None

# ------------------------------ Reporting (read-only) --------------------------

QUICK_REPORT_SQL = {
    "overall": """
        SELECT
          (SELECT count(*) FROM repositories) AS repos,
          (SELECT count(*) FROM code_files)   AS files,
          (SELECT count(*) FROM code_chunks)  AS chunks,
          (SELECT count(*) FROM chunk_edges)  AS edges;
    """,
    "embedding": """
        SELECT
          count(*) FILTER (WHERE embedding IS NOT NULL) AS with_emb,
          count(*) FILTER (WHERE embedding IS NULL)     AS without_emb,
          round(100.0 * count(*) FILTER (WHERE embedding IS NOT NULL) / NULLIF(count(*),0),1) AS pct_with_emb
        FROM code_chunks;
    """,
    "raw_calls": """
        WITH calls AS (
          SELECT cardinality(calls) AS n_calls FROM code_chunks
        )
        SELECT
          sum(n_calls) AS total_calls_recorded,
          (SELECT count(*) FROM chunk_edges WHERE edge_type='calls') AS call_edges_created,
          round(100.0 * (SELECT count(*) FROM chunk_edges WHERE edge_type='calls') / NULLIF(sum(n_calls),0), 1) AS pct_resolved_intra_file_raw
        FROM calls;
    """,
    "eligible_calls": """
        -- helpers are inline to avoid requiring CREATE FUNCTION permissions
        WITH
        norm_full AS (
          SELECT 1
        ),
        defs AS (
          SELECT
            c.file_id,
            array_agg(DISTINCT regexp_replace(regexp_replace(split_part(c.symbol_name,'#',2), '\\(.*\\)$', ''), '<[^>]+>', '', 'g'))
              FILTER (WHERE c.symbol_name IS NOT NULL) AS defs_full
          FROM code_chunks c
          GROUP BY c.file_id
        ),
        calls_expanded AS (
          SELECT
            c.id AS src_chunk_id,
            c.file_id,
            regexp_replace(regexp_replace(split_part(x,'#',2), '\\(.*\\)$', ''), '<[^>]+>', '', 'g') AS call_full
          FROM code_chunks c
          LEFT JOIN LATERAL unnest(c.calls) AS x ON TRUE
          WHERE x IS NOT NULL AND x <> ''
        ),
        eligible AS (
          SELECT DISTINCT ce.src_chunk_id
          FROM calls_expanded ce
          JOIN defs d USING (file_id)
          WHERE ce.call_full IS NOT NULL AND ce.call_full = ANY(d.defs_full)
        ),
        resolved_intra AS (
          SELECT DISTINCT e.src_chunk_id
          FROM chunk_edges e
          JOIN code_chunks s ON s.id = e.src_chunk_id
          JOIN code_chunks t ON t.id = e.dst_chunk_id
          WHERE e.edge_type='calls' AND s.file_id = t.file_id
        )
        SELECT
          (SELECT count(*) FROM eligible)       AS eligible_calls_intra_file,
          (SELECT count(*) FROM resolved_intra) AS resolved_call_edges_intra_file,
          ROUND(100.0 * (SELECT count(*) FROM resolved_intra) / NULLIF((SELECT count(*) FROM eligible),0), 1) AS pct_resolved_of_eligible_intra;
    """
}

def _print_rowdicts(title: str, rows: List[dict]) -> None:
    if not rows:
        log.info(f"[report] {title}: <no rows>")
        return
    # pretty single-line JSON objects for logs
    for r in rows:
        log.info(f"[report] {title}: {json.dumps(r, default=str)}")

def _fetch_all_dicts(cur, sql: str) -> List[dict]:
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    out = []
    for row in cur.fetchall():
        out.append({cols[i]: row[i] for i in range(len(cols))})
    return out

def run_quick_report(cur) -> None:
    log.info("=== Ingestion Report (quick) ===")
    for key in ("overall", "embedding", "raw_calls", "eligible_calls"):
        try:
            rows = _fetch_all_dicts(cur, QUICK_REPORT_SQL[key])
            _print_rowdicts(key, rows)
        except Exception as e:
            log.warning(f"[report:{key}] failed: {e}")

def run_sql_file_report(cur, sql_path: Path) -> bool:
    """
    Executes *read-only* statements (SELECT / WITH) from useful_sql.sql.
    Returns True if at least one statement ran; otherwise False.
    """
    if not sql_path.exists():
        return False
    text = sql_path.read_text(encoding="utf-8", errors="ignore")
    # naive split: keep only SELECT/WITH statements to stay read-only
    stmts: List[str] = []
    for part in text.split(";"):
        s = part.strip()
        if not s:
            continue
        head = s.split(None, 1)[0].upper()
        if head in ("SELECT", "WITH"):
            stmts.append(s)
    ran_any = False
    if not stmts:
        return False
    log.info(f"=== Ingestion Report ({sql_path}) ===")
    for s in stmts:
        try:
            rows = _fetch_all_dicts(cur, s)
            # derive a short tag from the first keyword/line
            tag = s.splitlines()[0][:60].replace("\n", " ")
            _print_rowdicts(tag, rows)
            ran_any = True
        except Exception as e:
            log.warning(f"[report:file] skipped one statement: {e}")
    return ran_any

# ------------------------------ Main ingest -----------------------------------

def iter_candidate_files(
    repo_dir: Path,
    only_changed: bool,
    base_ref: Optional[str],
    follow_symlinks: bool,
    max_file_size_bytes: Optional[int],
) -> List[Path]:
    if only_changed and base_ref:
        candidates = git_changed_files(repo_dir, base_ref)
    else:
        candidates = []
        for p in repo_dir.rglob("*"):
            try:
                if p.is_symlink() and not follow_symlinks:
                    continue
                if p.is_file():
                    candidates.append(p)
            except Exception:
                continue

    out: List[Path] = []
    for p in candidates:
        if detect_language(p) is None:
            continue
        try:
            if max_file_size_bytes is not None and p.stat().st_size > max_file_size_bytes:
                continue
        except Exception:
            pass
        out.append(p)
    return out

def main():
    ap = argparse.ArgumentParser()

    # Required
    ap.add_argument("--repo-dir", required=True, help="Path to local git working tree")
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--repo-url", default="")

    # DSN: prefer DB_DSN if present, else PG_DSN, else default
    _dsn_default = os.environ.get("DB_DSN") or os.environ.get("PG_DSN") or "dbname=rag user=postgres host=db password=postgres"
    ap.add_argument("--dsn", default=_dsn_default)

    # Changed-only (both spellings)
    ap.add_argument("--changed-only", dest="changed_only", action="store_true", help="Ingest only files changed since base ref / last ingested SHA")
    ap.add_argument("--changed_only",  dest="changed_only", action="store_true", help=argparse.SUPPRESS)

    ap.add_argument("--base-ref", default=None, help="Git base ref for --changed-only (e.g., origin/main). If absent, uses repositories.last_ingested_sha when available.")
    ap.add_argument("--follow-symlinks", action="store_true")
    ap.add_argument("--max-file-size", type=int, default=2_000_000, help="Skip files larger than this many bytes (default 2MB)")

    # Chunking/window config (chars ~= 4x tokens)
    ap.add_argument("--window-max-chars", type=int, default=int(os.environ.get("WINDOW_MAX_CHARS", "1800")), help="Max chars per fallback window (default 1800 ≈ 450 tokens)")
    ap.add_argument("--window-overlap-chars", type=int, default=int(os.environ.get("WINDOW_OVERLAP_CHARS", "300")), help="Overlap chars between fallback windows (default 300)")

    # Embeddings
    ap.add_argument("--embed", action="store_true", help="Compute and store embeddings via Ollama")
    ap.add_argument("--embed-model", default=os.environ.get("EMB_MODEL", "mxbai-embed-large"), help="Ollama embedding model (default from EMB_MODEL or mxbai-embed-large)")
    ap.add_argument("--embed-max-chars", type=int, default=int(os.environ.get("EMB_MAX_CHARS", "2000")), help="Safety cap on text length sent to embed API (default 2000 chars)")

    # Edges
    build_edges_default = os.environ.get("BUILD_EDGES", "1").strip() not in ("0", "false", "False", "")
    ap.add_argument("--build-edges", dest="build_edges", action="store_true", default=build_edges_default, help="Build intra-file call edges (default: on) and run cross-file materializer")
    ap.add_argument("--no-build-edges", dest="build_edges", action="store_false", help="Disable edges building")

    # Report
    report_default = os.environ.get("REPORT", "0").strip() in ("1", "true", "True")
    ap.add_argument("--report", dest="report", action="store_true", default=report_default, help="Print read-only ingestion quality report (uses db/usefulSQL/useful_sql.sql if present)")

    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        log.error(f"repo_dir not found: {repo_dir}")
        sys.exit(1)

    # Git SHA & base ref logic
    head_sha = git_head_sha(repo_dir)
    base_ref = args.base_ref
    root = git_root(repo_dir)
    if args.changed_only and not base_ref and root is None:
        log.warning("--changed-only requested but not a git repo; ignoring.")

    log.info(
        f"Starting ingest: repo_name={args.repo_name} dir={repo_dir} "
        f"changed_only={bool(args.changed_only)} base_ref={base_ref or '<auto/none>'} "
        f"embed={bool(args.embed)} model={args.embed_model} build_edges={args.build_edges} report={bool(args.report)}"
    )
    if not TS_AVAILABLE:
        log.info("AST chunking disabled (no tree-sitter). Using windowed fallback as needed.")

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        repo_id = get_or_create_repository(cur, args.repo_name, args.repo_url)
        log.info(f"Repository id={repo_id}")

        if args.build_edges:
            ensure_chunk_edges_table(cur)

        # If --changed-only and no explicit base-ref, use repositories.last_ingested_sha
        if args.changed_only and not base_ref:
            cur.execute("SELECT last_ingested_sha FROM repositories WHERE id = %s", (repo_id,))
            row = cur.fetchone()
            base_ref = row[0] if row and row[0] else None
            log.info(f"Auto base_ref from DB: {base_ref}")

        files = iter_candidate_files(
            repo_dir=repo_dir,
            only_changed=args.changed_only,
            base_ref=base_ref,
            follow_symlinks=args.follow_symlinks,
            max_file_size_bytes=args.max_file_size,
        )

        total_files = 0
        total_chunks = 0
        total_edges = 0

        for fp in files:
            total_files += 1
            rel_path = str(fp.relative_to(repo_dir)) if root else str(fp)
            lang = detect_language(fp)
            try:
                size_bytes = fp.stat().st_size
            except Exception:
                size_bytes = 0

            file_text_for_checksum = None
            try:
                file_text_for_checksum = fp.read_text(encoding="utf-8", errors="ignore")
                checksum = sha256_text(file_text_for_checksum)
            except Exception:
                checksum = None

            # derive module (for Java we may refine after chunks)
            module_guess = derive_module(root, fp, lang, None)

            file_id = upsert_code_file(cur, repository_id=repo_id, rel_path=rel_path, language=lang, size_bytes=size_bytes, checksum=checksum, module=module_guess)

            # Chunk
            if TS_AVAILABLE and lang in PARSER_BY_LANG:
                log.debug(f"[AST] {rel_path}")
            else:
                log.debug(f"[fallback] {rel_path}")

            chunks = make_chunks(fp, max_chars=args.window_max_chars, overlap_chars=args.window_overlap_chars)

            if not chunks:
                log.debug(f"[skip] {rel_path}: no chunks (unknown language or empty file)")
                continue

            # If Java chunks exposed a package, update file module with it
            if lang == "java":
                java_pkg = None
                for c in chunks:
                    if isinstance(c.ast, dict) and c.ast.get("package"):
                        java_pkg = c.ast.get("package")
                        break
                if java_pkg and java_pkg != module_guess:
                    cur.execute("UPDATE code_files SET module = %s, updated_at = now() WHERE id = %s", (java_pkg, file_id))
                    module_guess = java_pkg

            # Upsert chunks and collect info for edge building
            name_to_chunks: Dict[str, List[Tuple[int, str, str]]] = {}
            chunk_ids_and_calls: List[Tuple[int, List[str], str]] = []

            file_level_imports = set()
            for c in chunks:
                for imp in c.imports or []:
                    file_level_imports.add(imp)

            for c in chunks:
                cid = upsert_code_chunk(cur, file_id=file_id, c=c, embed=args.embed, embed_model=args.embed_model, embed_max_chars=args.embed_max_chars)
                sname = simple_symbol_name(c.symbol_name)
                name_to_chunks.setdefault(sname, []).append((cid, c.symbol_name, c.symbol_kind))
                chunk_ids_and_calls.append((cid, c.calls or [], c.symbol_name))
                total_chunks += 1

                # symbol surface
                upsert_code_symbol(cur, repo_id, file_id, cid, c, module_guess)

            # file imports surface
            upsert_file_imports(cur, file_id, sorted(list(file_level_imports)), lang)

            # Intra-file call edges
            new_edges = 0
            if args.build_edges:
                for src_id, calls, src_sym in chunk_ids_and_calls:
                    for callee in calls:
                        callee = callee.strip()
                        if not callee:
                            continue
                        targets = name_to_chunks.get(callee)
                        if not targets:
                            continue
                        for dst_id, dst_sym, _dst_kind in targets:
                            if dst_id == src_id:
                                continue
                            insert_edge_if_absent(cur, src_id, dst_id, edge_type="calls", weight=1.0)
                            new_edges += 1
                total_edges += new_edges

            log.info(f"[file] {rel_path}: chunks={len(chunks)} edges+={new_edges} bytes={size_bytes} lang={lang or 'unknown'}")

            # commit periodically
            if total_files % 50 == 0:
                conn.commit()
                log.info(f"[progress] files={total_files} chunks={total_chunks} edges={total_edges} (committed)")

        # record last ingested sha if we have one
        update_repository_sha(cur, repo_id, head_sha)

        conn.commit()

        # Cross-file materializer
        if args.build_edges:
            try:
                cur.execute("SELECT build_cross_file_call_edges(3, 0.62)")
                created = cur.fetchone()[0]
                conn.commit()
                log.info(f"[cross-file] edges+={created}")
            except Exception as e:
                conn.rollback()
                log.warning(f"[cross-file] materialization failed: {e}")

        # ------------------ Success Report (read-only) ------------------
        if args.report:
            # Prefer your repo file if it exists; else run the embedded quick report
            sql_path = Path("db/usefulSQL/useful_sql.sql")
            try:
                used_file = run_sql_file_report(cur, sql_path)
                if not used_file:
                    run_quick_report(cur)
            except Exception as e:
                log.warning(f"[report] falling back to quick report: {e}")
                run_quick_report(cur)
        # ---------------------------------------------------------------

        log.info(f"[done] files={total_files} chunks={total_chunks} edges={total_edges} sha={head_sha or 'n/a'}")
    except Exception as e:
        conn.rollback()
        log.exception(f"[error] {e}")
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

if __name__ == "__main__":
    main()