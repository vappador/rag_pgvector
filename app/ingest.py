# app/ingest.py
"""
Repo ingestion for code-focused RAG (Postgres + pgvector).

What this version does:
- Early chunking (AST-first; windowed fallback) to avoid server-side embedding truncation.
- Rich logs, including when fallback goes into windowed mode.
- Embeddings via Ollama with explicit host (OLLAMA_HOST/OLLAMA_BASE_URL).
- **Edges generation**: intra-file call edges after chunk upserts.
  - Creates table chunk_edges if it does not exist.
  - Logs edge counts per file and totals.
  - Flag: --build-edges / --no-build-edges (default: on). Env: BUILD_EDGES=1/0.

NOTE: Cross-file edges are not generated here (see comment at bottom to extend).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import logging
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

# Tree-sitter (optional)
TS_AVAILABLE = False
PARSER_BY_LANG: Dict[str, "Parser"] = {}
LANG_SO_DEFAULT = str(Path("build") / "my-languages.so")
TS_LANG_SO = os.environ.get("TS_LANG_SO", LANG_SO_DEFAULT)

try:
    from tree_sitter import Language, Parser  # type: ignore

    if Path(TS_LANG_SO).exists():
        _LANGS = {}
        for lang_name in ("java", "typescript", "javascript", "python"):
            try:
                _LANGS[lang_name] = Language(TS_LANG_SO, lang_name)
            except Exception as e:
                log.debug(f"Tree-sitter: skipping '{lang_name}': {e}")
        for name, ts_lang in _LANGS.items():
            p = Parser()
            p.set_language(ts_lang)
            PARSER_BY_LANG[name] = p
        TS_AVAILABLE = len(PARSER_BY_LANG) > 0
        if TS_AVAILABLE:
            log.info(f"Tree-sitter enabled for: {', '.join(PARSER_BY_LANG.keys())}")
    else:
        log.info("Tree-sitter bundle not found; using fallback chunking.")
except Exception as e:
    log.info(f"Tree-sitter unavailable; using fallback chunking. Reason: {e}")
    TS_AVAILABLE = False

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
    symbol_name: str          # e.g., Person#updateEmail or function name
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
        # imports
        imports = set()
        for n in _walk(root):
            if n.type == "import_declaration":
                text = _node_text(source, n).strip().rstrip(";")
                if text.startswith("import"):
                    imports.add(text.replace("import", "", 1).strip())

        # classes -> methods
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
                        for ch in _walk(n):
                            if ch.type == "method_invocation":
                                for idn in _walk(ch):
                                    if idn.type == "identifier":
                                        calls.add(_node_text(source, idn))
                                        break

                        start_line = n.start_point[0] + 1
                        end_line = n.end_point[0] + 1
                        content = _node_text(source, n)
                        symbol = f"{class_name}#{meth_name}"
                        chunks.append(
                            Chunk(
                                language=lang,
                                symbol_kind="method",
                                symbol_name=symbol,
                                start_line=start_line,
                                end_line=end_line,
                                content=content,
                                imports=sorted(imports),
                                calls=sorted(calls),
                                ast={"class": class_name, "method": meth_name},
                            )
                        )

    elif lang in ("typescript", "javascript"):
        imports = set()
        for n in _walk(root):
            if n.type == "import_declaration":
                imports.add(_node_text(source, n).strip())

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
                            for idn in _walk(ch):
                                if idn.type == "identifier":
                                    calls.add(_node_text(source, idn))
                                    break
                    chunks.append(
                        Chunk(
                            language=lang,
                            symbol_kind="function",
                            symbol_name=name,
                            start_line=n.start_point[0] + 1,
                            end_line=n.end_point[0] + 1,
                            content=_node_text(source, n),
                            imports=sorted(imports),
                            calls=sorted(calls),
                            ast={"function": name},
                        )
                    )

        # class methods
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
                                for idn in _walk(ch):
                                    if idn.type == "identifier":
                                        calls.add(_node_text(source, idn))
                                        break
                        chunks.append(
                            Chunk(
                                language=lang,
                                symbol_kind="method",
                                symbol_name=f"{class_name}#{meth_name}",
                                start_line=m.start_point[0] + 1,
                                end_line=m.end_point[0] + 1,
                                content=_node_text(source, m),
                                imports=sorted(imports),
                                calls=sorted(calls),
                                ast={"class": class_name, "method": meth_name},
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
                            fn = ch.child_by_field_name("function")
                            if fn is not None:
                                target = _node_text(source, fn)
                                if target:
                                    calls.add(target.split("(")[0].strip())
                    chunks.append(
                        Chunk(
                            language=lang,
                            symbol_kind="function",
                            symbol_name=name,
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
                                fn = ch.child_by_field_name("function")
                                if fn is not None:
                                    target = _node_text(source, fn)
                                    if target:
                                        calls.add(target.split("(")[0].strip())
                        chunks.append(
                            Chunk(
                                language=lang,
                                symbol_kind="method",
                                symbol_name=f"{class_name}#{meth_name}",
                                start_line=m.start_point[0] + 1,
                                end_line=m.end_point[0] + 1,
                                content=_node_text(source, m),
                                imports=sorted(imports),
                                calls=sorted(calls),
                                ast={"class": class_name, "method": meth_name},
                            )
                        )

    return chunks

# ------------------------------ Fallback chunking (windowed) -------------------

def _windowed_segments(
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 300,
) -> List[Tuple[int, int, str]]:
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

def chunk_file_fallback(
    path: Path,
    lang: str,
    max_chars: int,
    overlap_chars: int,
    verbose_filename: str,
) -> List[Chunk]:
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
                start_line=1,
                end_line=len(seg_lines),
                content=seg,
                imports=[],
                calls=[],
                ast={
                    "file": str(path),
                    "language": lang or "unknown",
                    "part": idx,
                    "offset_range": [start_idx, end_idx],
                },
            )
        )
    return chunks

def make_chunks(
    path: Path,
    max_chars: int = 1800,
    overlap_chars: int = 300,
) -> List[Chunk]:
    lang = detect_language(path)
    if not lang:
        return []
    chunks = chunk_file_ast(path, lang)
    if chunks:
        return chunks
    return chunk_file_fallback(
        path=path,
        lang=lang,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        verbose_filename=str(path),
    )

# ------------------------------ DB upserts ------------------------------------

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
    cur.execute(
        "UPDATE repositories SET last_ingested_sha = %s WHERE id = %s",
        (sha, repo_id),
    )

def upsert_code_file(
    cur,
    repository_id: int,
    rel_path: str,
    language: Optional[str],
    size_bytes: int,
    checksum: Optional[str],
) -> int:
    cur.execute(
        """
        INSERT INTO code_files (repository_id, path, language, size_bytes, checksum)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (repository_id, path) DO UPDATE
          SET language = EXCLUDED.language,
              size_bytes = EXCLUDED.size_bytes,
              checksum = EXCLUDED.checksum,
              updated_at = now()
        RETURNING id
        """,
        (repository_id, rel_path, language, size_bytes, checksum),
    )
    return cur.fetchone()[0]

def ensure_chunk_edges_table(cur) -> None:
    """
    Creates chunk_edges if missing with minimal schema:
    (src_chunk_id, dst_chunk_id, edge_type, weight, created_at).
    """
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
    # Helpful index for uniqueness (if you add a unique constraint yourself,
    # ON CONFLICT DO NOTHING can be used; here we do a WHERE NOT EXISTS insert).
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
    """
    For 'Class#method' -> 'method'; for 'foo' -> 'foo'.
    """
    if "#" in symbol_name:
        return symbol_name.split("#", 1)[1] or symbol_name
    return symbol_name

def upsert_code_chunk(
    cur,
    file_id: int,
    c: Chunk,
    embed: bool,
    embed_model: Optional[str],
    embed_max_chars: int,
) -> int:
    """
    Upsert a chunk by (file_id, start_line, end_line), compute content_hash,
    optionally compute embedding via Ollama.
    """
    content_hash = sha256_text(
        f"{c.symbol_kind}|{c.symbol_name}|{c.start_line}|{c.end_line}|{c.content}"
    )

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

    cur.execute(
        """
        INSERT INTO code_chunks
          (file_id, start_line, end_line, content, language,
           symbol_kind, symbol_name, imports, calls, ast,
           embedding, embedding_model, embedding_hash, content_hash)
        VALUES
          (%s, %s, %s, %s, %s,
           %s, %s, %s, %s, %s,
           %s, %s, %s, %s)
        ON CONFLICT (file_id, start_line, end_line) DO UPDATE SET
           content = EXCLUDED.content,
           language = EXCLUDED.language,
           symbol_kind = EXCLUDED.symbol_kind,
           symbol_name = EXCLUDED.symbol_name,
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
            c.language,
            c.symbol_kind,
            c.symbol_name,
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
    _dsn_default = (
        os.environ.get("DB_DSN")
        or os.environ.get("PG_DSN")
        or "dbname=rag user=postgres host=db password=postgres"
    )
    ap.add_argument("--dsn", default=_dsn_default)

    # Changed-only: accept both spellings
    ap.add_argument("--changed-only", dest="changed_only", action="store_true",
                    help="Ingest only files changed since base ref / last ingested SHA")
    ap.add_argument("--changed_only", dest="changed_only", action="store_true",
                    help=argparse.SUPPRESS)

    ap.add_argument("--base-ref", default=None,
                    help="Git base ref for --changed-only (e.g., origin/main). If absent, uses repositories.last_ingested_sha when available.")
    ap.add_argument("--follow-symlinks", action="store_true")
    ap.add_argument("--max-file-size", type=int, default=2_000_000,
                    help="Skip files larger than this many bytes (default 2MB)")

    # Chunking/window config (chars ~= 4x tokens)
    ap.add_argument("--window-max-chars", type=int, default=int(os.environ.get("WINDOW_MAX_CHARS", "1800")),
                    help="Max chars per fallback window (default 1800 ≈ 450 tokens)")
    ap.add_argument("--window-overlap-chars", type=int, default=int(os.environ.get("WINDOW_OVERLAP_CHARS", "300")),
                    help="Overlap chars between fallback windows (default 300)")

    # Embeddings
    ap.add_argument("--embed", action="store_true", help="Compute and store embeddings via Ollama")
    ap.add_argument("--embed-model", default=os.environ.get("EMB_MODEL", "mxbai-embed-large"),
                    help="Ollama embedding model (default from EMB_MODEL or mxbai-embed-large)")
    ap.add_argument("--embed-max-chars", type=int, default=int(os.environ.get("EMB_MAX_CHARS", "2000")),
                    help="Safety cap on text length sent to embed API (default 2000 chars)")

    # Edges
    build_edges_default = os.environ.get("BUILD_EDGES", "1").strip() not in ("0", "false", "False", "")
    ap.add_argument("--build-edges", dest="build_edges", action="store_true", default=build_edges_default,
                    help="Build intra-file call edges (default: on)")
    ap.add_argument("--no-build-edges", dest="build_edges", action="store_false",
                    help="Disable edges building")

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
        f"embed={bool(args.embed)} model={args.embed_model} build_edges={args.build_edges}"
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

            try:
                checksum = sha256_text(fp.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                checksum = None

            file_id = upsert_code_file(
                cur,
                repository_id=repo_id,
                rel_path=rel_path,
                language=lang,
                size_bytes=size_bytes,
                checksum=checksum,
            )

            # Chunk
            if TS_AVAILABLE and lang in PARSER_BY_LANG:
                log.debug(f"[AST] {rel_path}")
            else:
                log.debug(f"[fallback] {rel_path}")

            chunks = make_chunks(
                fp,
                max_chars=args.window_max_chars,
                overlap_chars=args.window_overlap_chars,
            )

            if not chunks:
                log.debug(f"[skip] {rel_path}: no chunks (unknown language or empty file)")
                continue

            # Upsert chunks and collect info for edge building
            # Map: simple_name -> list of (chunk_id, full_symbol, symbol_kind)
            name_to_chunks: Dict[str, List[Tuple[int, str, str]]] = {}
            # Keep source chunk info with its calls list
            chunk_ids_and_calls: List[Tuple[int, List[str], str]] = []

            for c in chunks:
                cid = upsert_code_chunk(
                    cur,
                    file_id=file_id,
                    c=c,
                    embed=args.embed,
                    embed_model=args.embed_model,
                    embed_max_chars=args.embed_max_chars,
                )
                sname = simple_symbol_name(c.symbol_name)
                name_to_chunks.setdefault(sname, []).append((cid, c.symbol_name, c.symbol_kind))
                chunk_ids_and_calls.append((cid, c.calls or [], c.symbol_name))
                total_chunks += 1

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

                # (Optional) very coarse import edges (within same file only)
                # If you prefer not to add these, comment this block out.
                # for src_id, _calls, _src_sym in chunk_ids_and_calls:
                #     for c in chunks:
                #         for imp in c.imports or []:
                #             # simple heuristic: connect to all chunks in same file
                #             for dst_id, _dst_sym, _dst_kind in [x for lst in name_to_chunks.values() for x in lst]:
                #                 if dst_id != src_id:
                #                     insert_edge_if_absent(cur, src_id, dst_id, edge_type="imports", weight=0.2)
                #                     new_edges += 1

                total_edges += new_edges

            log.info(f"[file] {rel_path}: chunks={len(chunks)} edges+={new_edges} bytes={size_bytes} lang={lang or 'unknown'}")

            # commit periodically to keep memory in check
            if total_files % 50 == 0:
                conn.commit()
                log.info(f"[progress] files={total_files} chunks={total_chunks} edges={total_edges} (committed)")

        # record last ingested sha if we have one
        update_repository_sha(cur, repo_id, head_sha)

        conn.commit()
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
