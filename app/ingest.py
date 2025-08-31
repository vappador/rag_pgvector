# app/ingest.py
"""
Repo ingestion for code-focused RAG (Postgres + pgvector).

Features:
- Inserts/updates repositories, code_files, code_chunks per the new schema.
- AST-aware chunking with Tree-sitter (java/ts/js/py), graceful fallback to whole-file chunk.
- Extracts symbol_name/symbol_kind, imports, calls, ast json, language, start_line/end_line.
- Computes content_hash for incremental/idem-potent updates.
- Optional incremental ingest via git diff or last_ingested_sha.
- Optional on-the-fly embeddings via Ollama (--embed).

ENV / CLI:
  PG_DSN              : psycopg2 DSN (default: dbname=rag user=postgres host=db password=postgres)
  OLLAMA_HOST         : base URL (default: http://ollama:11434)
  TS_LANG_SO          : path to tree-sitter language bundle (default: build/my-languages.so)

Example:
  docker compose exec app python /workspace/app/ingest.py \
    --repo-dir /workspace/repo --repo-name rag_pgvector \
    --repo-url https://github.com/vappador/rag_pgvector --embed --embed-model mxbai-embed-large
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import psycopg2
import psycopg2.extras

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
        # Add/remove languages here as you like; ensure your .so was built with these grammars.
        for lang_name in ("java", "typescript", "javascript", "python"):
            try:
                _LANGS[lang_name] = Language(TS_LANG_SO, lang_name)
            except Exception:
                pass  # ignore missing grammars in the bundle

        for name, ts_lang in _LANGS.items():
            p = Parser()
            p.set_language(ts_lang)
            PARSER_BY_LANG[name] = p

        TS_AVAILABLE = len(PARSER_BY_LANG) > 0
except Exception:
    TS_AVAILABLE = False

# Ollama (optional, only used when --embed)
OLLAMA_AVAILABLE = False
try:
    import ollama  # type: ignore

    OLLAMA_AVAILABLE = True
except Exception:
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
    symbol_kind: str          # class|method|function|module
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
                # class name is an identifier descendant
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
                            # target of call stored in field 'function'
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

def chunk_file_fallback(path: Path, lang: str) -> List[Chunk]:
    """
    Fallback when AST not available: one module-level chunk = entire file.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    return [
        Chunk(
            language=lang or "unknown",
            symbol_kind="module",
            symbol_name=path.name,
            start_line=1,
            end_line=len(lines) if lines else 1,
            content=text,
            imports=[],
            calls=[],
            ast={"file": str(path), "language": lang or "unknown"},
        )
    ]

def make_chunks(path: Path) -> List[Chunk]:
    lang = detect_language(path)
    if not lang:
        return []
    chunks = chunk_file_ast(path, lang)
    if not chunks:
        chunks = chunk_file_fallback(path, lang)
    # enrich ast with file name & lang
    for c in chunks:
        c.ast = dict(c.ast or {})
        c.ast.setdefault("file", str(path))
        c.ast.setdefault("language", lang)
        c.language = lang
    return chunks

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

def upsert_code_chunk(
    cur,
    file_id: int,
    c: Chunk,
    embed: bool,
    embed_model: Optional[str],
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
        model = embed_model or "mxbai-embed-large"
        try:
            resp = ollama.embeddings(model=model, prompt=c.content)
            vec = resp.get("embedding")
            if vec and isinstance(vec, list):
                embedding = vec
                embedding_model = model
                embedding_hash = sha256_text(c.content)  # hash of the input to embedding
        except Exception as e:
            # Don't fail ingestion if embedding fails; just proceed without it
            print(f"[warn] embedding failed for {c.symbol_name}: {e}", file=sys.stderr)

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
           -- only overwrite embedding when we provided one this time
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

    # Filter by language/extensions, size
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
    ap.add_argument("--repo-dir", required=True, help="Path to local git working tree")
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--dsn", default=os.environ.get("PG_DSN", "dbname=rag user=postgres host=db password=postgres"))

    ap.add_argument("--changed-only", action="store_true", help="Ingest only files changed since base ref / last ingested SHA")
    ap.add_argument("--base-ref", default=None, help="Git base ref for --changed-only (e.g., origin/main). If absent, uses repositories.last_ingested_sha when available.")
    ap.add_argument("--follow-symlinks", action="store_true")
    ap.add_argument("--max-file-size", type=int, default=2_000_000, help="Skip files larger than this many bytes (default 2MB)")

    ap.add_argument("--embed", action="store_true", help="Compute and store embeddings via Ollama")
    ap.add_argument("--embed-model", default="mxbai-embed-large", help="Ollama embedding model (default: mxbai-embed-large)")

    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        print(f"[error] repo_dir not found: {repo_dir}", file=sys.stderr)
        sys.exit(1)

    # Git SHA & base ref logic
    head_sha = git_head_sha(repo_dir)
    base_ref = args.base_ref
    root = git_root(repo_dir)
    if args.changed-only and not base_ref and root is None:
        print("[warn] --changed-only requested but not a git repo; ignoring.", file=sys.stderr)

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        repo_id = get_or_create_repository(cur, args.repo_name, args.repo_url)

        # If --changed-only and no explicit base-ref, use repositories.last_ingested_sha
        if args.changed-only and not base_ref:
            cur.execute("SELECT last_ingested_sha FROM repositories WHERE id = %s", (repo_id,))
            row = cur.fetchone()
            base_ref = row[0] if row and row[0] else None

        files = iter_candidate_files(
            repo_dir=repo_dir,
            only_changed=args.changed-only,
            base_ref=base_ref,
            follow_symlinks=args.follow_symlinks,
            max_file_size_bytes=args.max_file_size,
        )

        total_files = 0
        total_chunks = 0

        for fp in files:
            total_files += 1
            rel_path = str(fp.relative_to(repo_dir)) if root else str(fp)
            lang = detect_language(fp)
            try:
                size_bytes = fp.stat().st_size
            except Exception:
                size_bytes = 0

            # file checksum for optional audits (entire file content)
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

            chunks = make_chunks(fp)
            for c in chunks:
                _ = upsert_code_chunk(
                    cur,
                    file_id=file_id,
                    c=c,
                    embed=args.embed,
                    embed_model=args.embed_model,
                )
                total_chunks += 1

            # commit periodically to keep memory in check
            if total_files % 50 == 0:
                conn.commit()
                print(f"[info] processed {total_files} files, {total_chunks} chunks...")

        # record last ingested sha if we have one
        update_repository_sha(cur, repo_id, head_sha)

        conn.commit()
        print(f"[done] files={total_files} chunks={total_chunks} sha={head_sha or 'n/a'}")
    except Exception as e:
        conn.rollback()
        print(f"[error] {e}", file=sys.stderr)
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

if __name__ == "__main__":
    main()
