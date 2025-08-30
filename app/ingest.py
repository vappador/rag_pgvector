# app/ingest.py
import os
import re
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import psycopg2
import psycopg2.extras
from git import Repo
from tqdm import tqdm

# Tree-sitter: use get_parser only (avoid ABI issues with get_language)
from tree_sitter_languages import get_parser

# Ollama: local embeddings
import ollama

# -----------------------
# Config & logging
# -----------------------
DB_DSN = os.getenv("DB_DSN", "postgresql://rag:ragpwd@localhost:5432/ragdb")

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMB_MODEL = os.getenv("EMB_MODEL", "mxbai-embed-large")  # 1024-d
EMB_DIM = int(os.getenv("EMB_DIM", "1024"))

_ollama = ollama.Client(host=OLLAMA_BASE)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ingest")

# -----------------------
# Language detection & filters
# -----------------------
LANG_MAP = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
}
TEST_HINTS = re.compile(r"(^|/)(test|tests|spec|__tests__|__spec__)/", re.I)
SKIP_DIRS = {"node_modules", "dist", "build", "target", ".git", ".idea", ".vscode"}
SKIP_FILE_SUBSTR = {".min.", ".map."}

# Calls stoplist (reduce noise from control keywords / builtins)
CALLS_STOP = {
    "if", "for", "while", "switch", "catch", "return", "class", "function", "new",
    "throw", "await", "super", "this", "try", "typeof", "console", "log"
}

# -----------------------
# Helpers
# -----------------------
def is_code(p: Path) -> bool:
    if p.suffix.lower() not in LANG_MAP:
        return False
    if any(part.lower() in SKIP_DIRS for part in p.parts):
        return False
    name = p.name.lower()
    if any(sub in name for sub in SKIP_FILE_SUBSTR):
        return False
    return True


def language_for(p: Path) -> str:
    return LANG_MAP.get(p.suffix.lower(), "other")


def to_text(src: bytes, start: int, end: int) -> str:
    return src[start:end].decode("utf-8", errors="ignore")


def as_vector_literal(vec: List[float]) -> str:
    """Format a Python list of floats as pgvector's textual literal: [0.1,0.2,...]"""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def zeros_vec(dim: int) -> List[float]:
    return [0.0] * dim


def safe_dim(vec: List[float], dim: int) -> List[float]:
    """Pad or truncate to expected dimension."""
    if len(vec) == dim:
        return vec
    if len(vec) > dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def embed_one(text: str, retries: int = 2) -> List[float]:
    """Embed a single text with retries; return zeros on failure to avoid NULLs."""
    for i in range(retries + 1):
        try:
            r = _ollama.embeddings(model=EMB_MODEL, prompt=text)
            emb = r.get("embedding")
            if not emb:
                raise RuntimeError("No 'embedding' in Ollama response")
            return safe_dim(emb, EMB_DIM)
        except Exception as e:
            if i < retries:
                log.warning("Embedding retry %d/%d: %s", i + 1, retries, e)
            else:
                log.error("Embedding failed; using zeros. Error: %s", e)
                log.debug("Traceback:\n%s", traceback.format_exc())
                return zeros_vec(EMB_DIM)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Per-text embed to isolate failures; still fairly quick for typical chunk counts."""
    return [embed_one(t) for t in texts]


def extract_ts(js: bool = False):
    """Return ready-to-use parser; avoid get_language to sidestep ABI issues."""
    return get_parser("javascript" if js else "typescript")


def extract_java():
    return get_parser("java")


def fallback_window_chunks(
    code: str, win: int = 120, overlap: int = 20
) -> List[Tuple[int, int, str, Optional[str], Optional[str], Optional[str], Optional[str], list, list]]:
    """Fallback windowed chunking when AST parsing fails."""
    lines = code.splitlines()
    if not lines:
        return []
    i = 0
    chunks = []
    while i < len(lines):
        s, e = i, min(i + win, len(lines))
        txt = "\n".join(lines[s:e])
        chunks.append((s + 1, e, txt, None, None, None, None, [], []))
        if e == len(lines):
            break
        i = e - overlap
    return chunks


def _guess_symbol(header: str, langname: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Try hard to extract (name, kind, signature) from the first line of the chunk.
    Handles common TS/JS and Java patterns.
    """
    sig = header.strip()

    # Classes / interfaces (TS/JS/Java)
    m = re.search(r"\bclass\s+([A-Za-z_]\w*)", header)
    if m:
        return m.group(1), "class", sig
    m = re.search(r"\binterface\s+([A-Za-z_]\w*)", header)
    if m:
        return m.group(1), "class", sig  # treat interfaces as class-kind

    # Java method: modifiers then name(...)
    m = re.search(r"(?:public|private|protected|static|final|\s)+([A-Za-z_]\w*)\s*\(", header)
    if m and langname == "java":
        return m.group(1), "function", sig

    # JS/TS: function declaration / export default function
    m = re.search(r"\bfunction\s+([A-Za-z_]\w*)", header)
    if m:
        return m.group(1), "function", sig
    m = re.search(r"\bexport\s+default\s+function\s+([A-Za-z_]\w*)", header)
    if m:
        return m.group(1), "function", sig

    # JS/TS: var-assigned fn or arrow fn: const/let/var name = ( or function
    m = re.search(r"\b(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?(?:function|\()", header)
    if m:
        return m.group(1), "function", sig

    # Methods (TS/JS inside classes): modifiers then name(
    m = re.search(r"(?:public|private|protected|static|readonly|async|\s)+([A-Za-z_]\w*)\s*\(", header)
    if m:
        return m.group(1), "function", sig

    # Last resort: any identifier followed by '('
    m = re.search(r"\b([A-Za-z_]\w*)\s*\(", header)
    if m and m.group(1) not in CALLS_STOP:
        return m.group(1), "function", sig

    return None, None, sig


def ts_java_symbols(
    p: Path, code: str, langname: str
) -> List[Tuple[int, int, str, Optional[str], Optional[str], Optional[str], Optional[str], list, list]]:
    """
    Extract symbol-level chunks using Tree-sitter. On parser errors,
    gracefully falls back to windowed chunking.

    Returns list of:
      (start_line, end_line, text, symbol_name, symbol_kind, signature, doc, imports, calls)
    """
    src = code.encode("utf-8", errors="ignore")

    # Choose parser (safe)
    try:
        if langname == "java":
            parser = extract_java()
            import_kind = "import_declaration"
            fnkinds = ("method_declaration",)
            clskinds = ("class_declaration", "interface_declaration", "enum_declaration")
        else:
            parser = extract_ts(js=(langname == "javascript"))
            import_kind = "import_statement"
            fnkinds = ("function_declaration", "method_definition")
            clskinds = ("class_declaration",)
    except Exception as e:
        log.warning("Parser init failed for %s: %s; falling back to windowing", p, e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return fallback_window_chunks(code)

    # Parse (safe)
    try:
        tree = parser.parse(src)
    except Exception as e:
        log.warning("Parser.parse() failed for %s: %s; falling back to windowing", p, e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return fallback_window_chunks(code)

    if not tree or not getattr(tree, "root_node", None):
        log.warning("No parse tree for %s; falling back to windowing", p)
        return fallback_window_chunks(code)

    root = tree.root_node

    def lines_of(node):
        return node.start_point[0] + 1, node.end_point[0] + 1

    chunks = []

    # Top-level imports (best-effort)
    imports = []
    try:
        for n in root.children:
            if getattr(n, "type", None) == import_kind:
                imports.append(to_text(src, n.start_byte, n.end_byte).strip())
    except Exception:
        pass  # not fatal

    # DFS for classes & functions
    stack = [root]
    while stack:
        n = stack.pop()
        try:
            stack.extend(n.children)
        except Exception:
            continue

        try:
            ntype = getattr(n, "type", "")
            if ntype in clskinds or ntype in fnkinds:
                start_l, end_l = lines_of(n)
                text = to_text(src, n.start_byte, n.end_byte)

                # Signature & symbol heuristics from header only
                header = text.split("\n", 1)[0][:300]
                name, kind, sig = _guess_symbol(header, langname)
                if ntype in clskinds:
                    kind = "class"

                # Doc: previous sibling comment if present
                doc = None
                prev = getattr(n, "prev_sibling", None)
                if prev is not None and getattr(prev, "type", "") in ("comment", "block_comment"):
                    doc = to_text(src, prev.start_byte, prev.end_byte).strip()

                # Light call extraction: foo( … ), filter stop words
                calls = sorted(
                    {m for m in re.findall(r"\b([A-Za-z_]\w*)\s*\(", text) if m not in CALLS_STOP}
                )[:50]

                chunks.append((start_l, end_l, text, name, kind, sig, doc, imports[:50], calls))
        except Exception:
            log.debug("Node processing failed on %s; continuing.\n%s", p, traceback.format_exc())
            continue

    if not chunks:
        return fallback_window_chunks(code)

    return chunks


def upsert(conn, sql, params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def build_edges_for_repo(conn, repo_id: int):
    """
    Populate rag.chunk_edges for a repo:
      - 'calls' edges: src.calls[] -> dst.symbol_name match within same repo
      - 'imports' edges (js/ts): module token -> matching target files (incl. index.*)
      - 'imports' edges (java): FQCN -> ClassName.java
    """
    with conn.cursor() as cur:
        # Remove existing edges for this repo to avoid duplicates
        cur.execute(
            """
            DELETE FROM rag.chunk_edges e
            USING rag.code_chunks s
            JOIN rag.files sf ON sf.id = s.file_id
            WHERE e.src_chunk_id = s.id AND sf.repo_id = %s
            """,
            (repo_id,),
        )

        # ---- calls edges (symbol-name match) ----
        cur.execute(
            """
            INSERT INTO rag.chunk_edges (src_chunk_id, dst_chunk_id, relation)
            SELECT DISTINCT
              src.id  AS src_chunk_id,
              dst.id  AS dst_chunk_id,
              'calls' AS relation
            FROM rag.code_chunks src
            JOIN rag.files sf ON sf.id = src.file_id
            JOIN LATERAL jsonb_array_elements_text(src.calls) AS c(name) ON TRUE
            JOIN rag.code_chunks dst
              ON lower(dst.symbol_name) = lower(c.name)
            JOIN rag.files df ON df.id = dst.file_id
            WHERE src.valid_to IS NULL
              AND dst.valid_to IS NULL
              AND sf.repo_id = %s
              AND df.repo_id = %s
            """,
            (repo_id, repo_id),
        )

        # ---- imports edges (JS/TS) ----
        cur.execute(
            """
            WITH src AS (
              SELECT cc.id AS src_chunk_id, f.id AS src_file_id, f.repo_id, f.path AS src_path,
                     regexp_matches(imp, $$from\\s+['"]([^'"]+)['"]|require\\(\\s*['"]([^'"]+)['"]\\s*\\)$$, 'i') AS m
              FROM rag.code_chunks cc
              JOIN rag.files f ON f.id = cc.file_id
              CROSS JOIN LATERAL jsonb_array_elements_text(cc.imports) AS imp
              WHERE f.language IN ('javascript','typescript') AND cc.valid_to IS NULL AND f.repo_id = %s
            ),
            mods AS (
              SELECT src_chunk_id, src_file_id, repo_id, src_path,
                     COALESCE(NULLIF(m[1],''), NULLIF(m[2],'')) AS modname
              FROM src
            ),
            targets AS (
              SELECT DISTINCT
                s.src_chunk_id, s.src_file_id, s.repo_id, s.src_path, s.modname,
                tf.id AS dst_file_id
              FROM mods s
              JOIN rag.files tf ON tf.repo_id = s.repo_id
              WHERE s.modname IS NOT NULL
                AND (
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '.ts'  OR
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '.tsx' OR
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '.js'  OR
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '.jsx' OR
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '/index.ts'  OR
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '/index.tsx' OR
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '/index.js'  OR
                  tf.path ILIKE '%%/' || split_part(s.modname, '/', array_length(string_to_array(s.modname,'/'),1)) || '/index.jsx'
                )
            )
            INSERT INTO rag.chunk_edges (src_chunk_id, dst_chunk_id, relation)
            SELECT DISTINCT
              t.src_chunk_id,
              dc.id AS dst_chunk_id,
              'imports'
            FROM targets t
            JOIN rag.code_chunks dc ON dc.file_id = t.dst_file_id AND dc.valid_to IS NULL
            """,
            (repo_id,),
        )

        # ---- imports edges (Java) ----
        cur.execute(
            """
            WITH java_imps AS (
              SELECT cc.id AS src_chunk_id, f.id AS src_file_id, f.repo_id,
                     (regexp_matches(imp, $$import\\s+([A-Za-z0-9_.]+);$$, 'i'))[1] AS fqcn
              FROM rag.code_chunks cc
              JOIN rag.files f ON f.id = cc.file_id
              CROSS JOIN LATERAL jsonb_array_elements_text(cc.imports) AS imp
              WHERE f.language = 'java' AND cc.valid_to IS NULL AND f.repo_id = %s
            ),
            cls AS (
              SELECT src_chunk_id, src_file_id, repo_id,
                     split_part(fqcn, '.', array_length(string_to_array(fqcn, '.'),1)) AS cls_name
              FROM java_imps WHERE fqcn IS NOT NULL
            ),
            tgt AS (
              SELECT DISTINCT j.src_chunk_id, j.repo_id, jf.id AS dst_file_id
              FROM cls j
              JOIN rag.files jf ON jf.repo_id = j.repo_id
              WHERE jf.path ILIKE '%%/' || j.cls_name || '.java'
            )
            INSERT INTO rag.chunk_edges (src_chunk_id, dst_chunk_id, relation)
            SELECT DISTINCT
              t.src_chunk_id,
              dc.id AS dst_chunk_id,
              'imports'
            FROM tgt t
            JOIN rag.code_chunks dc ON dc.file_id = t.dst_file_id AND dc.valid_to IS NULL
            """,
            (repo_id,),
        )

    conn.commit()
    log.info("chunk_edges backfilled for repo_id=%s", repo_id)


# -----------------------
# Main ingestion routine
# -----------------------
def ingest_repo(repo_url: str, dest: Path, repo_name: Optional[str] = None):
    # Clone or pull
    try:
        if dest.exists():
            repo = Repo(str(dest))
            repo.remote().pull()
            log.info("Pulled latest for %s", dest)
        else:
            repo = Repo.clone_from(repo_url, str(dest))
            log.info("Cloned %s into %s", repo_url, dest)
    except Exception as e:
        log.error("Git operation failed for %s: %s", repo_url, e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return

    if not repo_name:
        repo_name = Path(repo_url.rstrip("/").split("/")[-1]).stem

    try:
        default_branch = repo.active_branch.name
    except Exception:
        default_branch = "main"

    with psycopg2.connect(DB_DSN) as conn:
        conn.autocommit = False

        # repositories row
        rid = upsert(
            conn,
            """
            INSERT INTO rag.repositories(name,url,default_branch)
            VALUES(%s,%s,%s)
            ON CONFLICT (name) DO UPDATE
              SET url=EXCLUDED.url, default_branch=EXCLUDED.default_branch
            RETURNING id
            """,
            (repo_name, repo_url, default_branch),
        )

        head = repo.head.commit
        committed_dt = datetime.fromtimestamp(head.committed_date)

        # commits row
        cid = upsert(
            conn,
            """
            INSERT INTO rag.commits(repo_id,commit_hash,author,committed_at,message)
            VALUES(%s,%s,%s,%s,%s)
            ON CONFLICT (repo_id,commit_hash) DO UPDATE
              SET author=EXCLUDED.author, committed_at=EXCLUDED.committed_at, message=EXCLUDED.message
            RETURNING id
            """,
            (rid, head.hexsha, getattr(head.author, "name", None), committed_dt, head.message),
        )

        files = [p for p in dest.rglob("*") if p.is_file() and is_code(p)]
        log.info("Found %d code files in %s", len(files), repo_name)

        for p in tqdm(files, desc=f"Ingest {repo_name}"):
            try:
                rel = p.relative_to(dest).as_posix()
                lang = language_for(p)
                is_test = bool(TEST_HINTS.search(rel))
                size = p.stat().st_size
                text = p.read_text(encoding="utf-8", errors="ignore")

                # Extract chunks (AST → fallback window)
                chunks = ts_java_symbols(p, text, lang)
                if not chunks:
                    log.debug("No chunks produced for %s; skipping file.", rel)
                    continue

                with conn.cursor() as cur:
                    # files row
                    cur.execute(
                        """
                        INSERT INTO rag.files(repo_id,path,language,is_test,size_bytes,latest_commit_id)
                        VALUES(%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (repo_id,path) DO UPDATE SET
                          language=EXCLUDED.language,
                          is_test=EXCLUDED.is_test,
                          size_bytes=EXCLUDED.size_bytes,
                          latest_commit_id=EXCLUDED.latest_commit_id
                        RETURNING id
                        """,
                        (rid, rel, lang, is_test, size, cid),
                    )
                    file_id = cur.fetchone()[0]

                    # Embeddings (safe, per text)
                    texts = [c[2] for c in chunks]
                    embs = embed_batch(texts)
                    emb_strs = [as_vector_literal(e) for e in embs]

                    # Prepare batch rows
                    rows = []
                    for idx, ((s, e, txt, name, kind, sig, doc, imports, calls), emb_str) in enumerate(
                        zip(chunks, emb_strs)
                    ):
                        rows.append(
                            (
                                file_id,
                                idx,
                                s,
                                e,
                                name,
                                kind,
                                sig,
                                doc,
                                psycopg2.extras.Json(imports),
                                psycopg2.extras.Json(calls),
                                txt,
                                emb_str,
                                len(txt.split()),
                                committed_dt,
                            )
                        )

                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO rag.code_chunks
                          (file_id,chunk_index,start_line,end_line,symbol_name,symbol_kind,symbol_signature,doc_comment,imports,calls,
                           content,embedding,token_count,committed_at)
                        VALUES %s
                        ON CONFLICT (file_id,chunk_index) DO UPDATE SET
                          start_line=EXCLUDED.start_line,
                          end_line=EXCLUDED.end_line,
                          symbol_name=EXCLUDED.symbol_name,
                          symbol_kind=EXCLUDED.symbol_kind,
                          symbol_signature=EXCLUDED.symbol_signature,
                          doc_comment=EXCLUDED.doc_comment,
                          imports=EXCLUDED.imports,
                          calls=EXCLUDED.calls,
                          content=EXCLUDED.content,
                          embedding=EXCLUDED.embedding,
                          token_count=EXCLUDED.token_count,
                          committed_at=EXCLUDED.committed_at,
                          valid_to=NULL
                        """,
                        rows,
                    )

                conn.commit()
            except Exception as e:
                # Skip file but keep ingesting the rest
                conn.rollback()
                log.error("Failed to ingest file %s: %s", p, e)
                log.debug("Traceback:\n%s", traceback.format_exc())
                continue

        # Analyze for better ANN/BM25 performance
        try:
            with conn.cursor() as cur:
                cur.execute("ANALYZE rag.code_chunks;")
            conn.commit()
        except Exception as e:
            log.warning("ANALYZE failed (non-fatal): %s", e)

        # Build edges for this repo (calls/imports)
        try:
            build_edges_for_repo(conn, rid)
        except Exception as e:
            log.warning("Edge backfill failed (non-fatal): %s", e)
            log.debug("Traceback:\n%s", traceback.format_exc())


# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    repo_urls = [u.strip() for u in os.getenv("REPO_URLS", "").split(",") if u.strip()]
    if not repo_urls:
        log.error('No REPO_URLS provided. Example: REPO_URLS="https://github.com/expressjs/express.git,https://github.com/spring-projects/spring-petclinic.git"')
        raise SystemExit(2)

    base = Path("/workspace/repos")
    base.mkdir(parents=True, exist_ok=True)

    for url in repo_urls:
        name = Path(url.rstrip("/").split("/")[-1]).stem
        dest = base / name
        log.info("Ingesting repo: %s -> %s", url, dest)
        ingest_repo(url, dest, repo_name=name)

    log.info("Ingestion complete.")
