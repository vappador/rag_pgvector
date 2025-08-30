# app/api.py
import os
from typing import Literal, Optional, Dict, Any, List

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query

DB_DSN = os.getenv("DB_DSN", "postgresql://rag:ragpwd@pg:5432/ragdb")

app = FastAPI(title="RAG PGVector API", version="1.1.0")


def _conn():
    return psycopg2.connect(DB_DSN)


def _get_one(cur, sql: str, args: tuple):
    cur.execute(sql, args)
    return cur.fetchone()


def _all(cur, sql: str, args: tuple):
    cur.execute(sql, args)
    return cur.fetchall()


def _resolve_file(cur, repo_name: str, file_hint: str):
    """
    Resolve a file inside a repo by either:
      - exact relpath match (case-sensitive)
      - or suffix/basename match (case-insensitive)

    Returns (repo_id, repo_name, file_id, file_path).
    Raises HTTPException 404/400 for not found / ambiguous.
    """
    repo = _get_one(
        cur,
        "SELECT id, name FROM rag.repositories WHERE name ILIKE %s LIMIT 1",
        (repo_name,),
    )
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repo not found: {repo_name}")
    repo_id, repo_name_db = repo[0], repo[1]

    # Try exact first
    exact = _get_one(
        cur,
        "SELECT id, path FROM rag.files WHERE repo_id = %s AND path = %s LIMIT 1",
        (repo_id, file_hint),
    )
    if exact:
        return repo_id, repo_name_db, exact[0], exact[1]

    # If the hint includes a slash, treat it as suffix; else treat as basename
    like = f"%/{file_hint.lstrip('/')}" if "/" in file_hint else f"%/{file_hint}"
    matches = _all(
        cur,
        "SELECT id, path FROM rag.files WHERE repo_id = %s AND path ILIKE %s "
        "ORDER BY LENGTH(path) ASC LIMIT 25",
        (repo_id, like),
    )
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"File not found in repo '{repo_name}': {file_hint}",
        )

    base = file_hint.split("/")[-1]
    exact_base = [m for m in matches if m[1].endswith("/" + base)]
    pick = exact_base[0] if exact_base else matches[0]

    same_base = [m for m in matches if m[1].endswith("/" + base)]
    if len(same_base) > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Ambiguous file name '{file_hint}' in repo '{repo_name}'",
                "candidates": [p for (_id, p) in same_base[:25]],
            },
        )

    return repo_id, repo_name_db, pick[0], pick[1]


def _edges_query(direction: Literal["in", "out"]) -> str:
    """
    Returns a parameterized SQL for directional traversal up to N hops.
    Params expected at execution time (in order):
      file_id, depth, repo_id, repo_id
    """
    if direction == "out":
        # Start from chunks in the target file as sources; walk forward
        return """
        WITH RECURSIVE start AS (
          SELECT id FROM rag.code_chunks
          WHERE file_id = %s AND valid_to IS NULL
        ),
        edges AS (
          SELECT e.src_chunk_id, e.dst_chunk_id, 1 AS depth, e.relation
          FROM rag.chunk_edges e
          WHERE e.src_chunk_id IN (SELECT id FROM start)
          UNION ALL
          SELECT e.src_chunk_id, e.dst_chunk_id, x.depth + 1, e.relation
          FROM rag.chunk_edges e
          JOIN edges x ON e.src_chunk_id = x.dst_chunk_id
          WHERE x.depth < %s
        )
        SELECT
          ed.depth,
          ed.relation,
          s.id          AS src_chunk_id,
          sf.path       AS src_path,
          s.symbol_name AS src_symbol,
          s.start_line  AS src_start,
          s.end_line    AS src_end,
          d.id          AS dst_chunk_id,
          df.path       AS dst_path,
          d.symbol_name AS dst_symbol,
          df.language   AS dst_language
        FROM edges ed
        JOIN rag.code_chunks s ON s.id = ed.src_chunk_id AND s.valid_to IS NULL
        JOIN rag.files sf      ON sf.id = s.file_id
        JOIN rag.code_chunks d ON d.id = ed.dst_chunk_id AND d.valid_to IS NULL
        JOIN rag.files df      ON df.id = d.file_id
        WHERE sf.repo_id = %s AND df.repo_id = %s
        ORDER BY ed.depth, sf.path, s.start_line, df.path;
        """
    else:
        # "in": start from target as destinations; walk reverse (dependents)
        return """
        WITH RECURSIVE start AS (
          SELECT id FROM rag.code_chunks
          WHERE file_id = %s AND valid_to IS NULL
        ),
        edges AS (
          SELECT e.src_chunk_id, e.dst_chunk_id, 1 AS depth, e.relation
          FROM rag.chunk_edges e
          WHERE e.dst_chunk_id IN (SELECT id FROM start)
          UNION ALL
          SELECT e.src_chunk_id, e.dst_chunk_id, x.depth + 1, e.relation
          FROM rag.chunk_edges e
          JOIN edges x ON e.dst_chunk_id = x.src_chunk_id
          WHERE x.depth < %s
        )
        SELECT
          ed.depth,
          ed.relation,
          s.id          AS src_chunk_id,
          sf.path       AS src_path,
          s.symbol_name AS src_symbol,
          s.start_line  AS src_start,
          s.end_line    AS src_end,
          d.id          AS dst_chunk_id,
          df.path       AS dst_path,
          d.symbol_name AS dst_symbol,
          df.language   AS dst_language
        FROM edges ed
        JOIN rag.code_chunks s ON s.id = ed.src_chunk_id AND s.valid_to IS NULL
        JOIN rag.files sf      ON sf.id = s.file_id
        JOIN rag.code_chunks d ON d.id = ed.dst_chunk_id AND d.valid_to IS NULL
        JOIN rag.files df      ON df.id = d.file_id
        WHERE sf.repo_id = %s AND df.repo_id = %s
        ORDER BY ed.depth, sf.path, s.start_line, df.path;
        """


@app.get("/graph/relations")
def graph_relations(
    repo: str = Query(..., description="Repository name (e.g., 'spring-petclinic')"),
    file: str = Query(..., description="File path or basename (e.g., 'Person.java' or 'src/main/.../Person.java')"),
    direction: Literal["in", "out", "both"] = Query("both"),
    depth: int = Query(2, ge=1, le=6, description="Traversal depth (hops)"),
    limit_edges: int = Query(5000, ge=1, le=20000, description="Safety cap on returned edges"),
):
    """
    Relationship graph around a file, with edges and a per-file summary.
    - direction=out: dependencies (edges from file -> others)
    - direction=in: dependents (edges to file <- others)
    - direction=both: union of the two
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            repo_id, repo_name, file_id, file_path = _resolve_file(cur, repo, file)

            def run_dir(dirflag: Literal["in", "out"]) -> List[Dict[str, Any]]:
                sql = _edges_query(dirflag)
                rows = _all(cur, sql, (file_id, depth, repo_id, repo_id))
                return rows[:limit_edges]

            results: List[Dict[str, Any]] = []
            if direction in ("out", "both"):
                results.extend(run_dir("out"))
            if direction in ("in", "both"):
                results.extend(run_dir("in"))

            # Summaries
            files_summary: Dict[str, int] = {}
            counts_by_relation: Dict[str, int] = {}
            for r in results:
                counts_by_relation[r["relation"]] = counts_by_relation.get(r["relation"], 0) + 1
                for path in (r["src_path"], r["dst_path"]):
                    if path == file_path:
                        continue
                    files_summary[path] = min(files_summary.get(path, 999), r["depth"])

            affected = [{"path": p, "min_depth": d} for p, d in sorted(files_summary.items(), key=lambda x: (x[1], x[0]))]

            return {
                "repo": repo_name,
                "target": {"file_id": file_id, "path": file_path},
                "direction": direction,
                "depth": depth,
                "edge_count": len(results),
                "counts_by_relation": counts_by_relation,
                "affected_files": affected,
                "edges": results,  # each has: depth, relation, src/dst chunk+file+symbol
            }


def _impact_sql() -> str:
    """
    Parameterized SQL for "impact" view (dependents + imports + lexical).
    Params expected at execution (in order):
      file_id, depth, repo_id, repo_id, import_like, repo_id, lexical_token, token_label, token_label, file_id, limit_files
    """
    return """
    WITH RECURSIVE
    start AS (
      SELECT id FROM rag.code_chunks
      WHERE file_id = %s AND valid_to IS NULL
    ),
    -- chunks that point to the target file's chunks
    direct_dep AS (
      SELECT DISTINCT e.src_chunk_id, 1 AS depth, e.relation
      FROM rag.chunk_edges e
      WHERE e.dst_chunk_id IN (SELECT id FROM start)
    ),
    dep AS (
      SELECT src_chunk_id, 1 AS depth
      FROM direct_dep
      UNION ALL
      SELECT e.src_chunk_id, d.depth + 1
      FROM rag.chunk_edges e
      JOIN dep d ON e.dst_chunk_id = d.src_chunk_id
      WHERE d.depth < %s
    ),
    dep_files AS (
      SELECT DISTINCT f.path, MIN(d.depth) AS min_depth
      FROM dep d
      JOIN rag.code_chunks s ON s.id = d.src_chunk_id AND s.valid_to IS NULL
      JOIN rag.files f       ON f.id = s.file_id
      WHERE f.repo_id = %s
      GROUP BY f.path
    ),
    import_refs AS (
      SELECT DISTINCT f.path
      FROM rag.code_chunks c
      JOIN rag.files f ON f.id = c.file_id
      WHERE f.repo_id = %s
        AND c.valid_to IS NULL
        AND EXISTS (
          SELECT 1 FROM jsonb_array_elements_text(c.imports) imp(txt)
          WHERE txt ILIKE %s
        )
    ),
    lex_refs AS (
      SELECT DISTINCT f.path
      FROM rag.code_chunks c
      JOIN rag.files f ON f.id = c.file_id
      WHERE f.repo_id = %s
        AND c.valid_to IS NULL
        AND c.content_tsv @@ plainto_tsquery('simple', %s)
    ),
    unioned AS (
      SELECT df.path,
             CASE WHEN df.min_depth = 1 THEN 'edge:direct' ELSE 'edge:transitive' END AS reason,
             df.min_depth
      FROM dep_files df
      UNION ALL
      SELECT ir.path, 'imports:' || %s AS reason, NULL::int AS min_depth
      FROM import_refs ir
      UNION ALL
      SELECT lr.path, 'lexical:' || %s AS reason, NULL::int AS min_depth
      FROM lex_refs lr
    ),
    clean AS (
      SELECT * FROM unioned u
      WHERE u.path <> (SELECT path FROM rag.files WHERE id = %s)
    )
    SELECT path,
           MIN(min_depth) AS min_depth,
           ARRAY_AGG(DISTINCT reason) AS reasons
    FROM clean
    GROUP BY path
    ORDER BY (MIN(CASE WHEN reason LIKE 'edge:%' THEN 0 ELSE 1 END)),
             COALESCE(MIN(min_depth), 999),
             path
    LIMIT %s;
    """


@app.get("/graph/impact")
def graph_impact(
    repo: str = Query(..., description="Repository name (e.g., 'spring-petclinic')"),
    file: str = Query(..., description="Target file (e.g., 'Person.java' or 'src/.../Person.java')"),
    depth: int = Query(2, ge=1, le=6, description="Traversal depth for dependents"),
    token: Optional[str] = Query(
        None,
        description="Optional lexical/import token to match (defaults to file basename without extension, e.g., 'Person').",
    ),
    limit_files: int = Query(5000, ge=1, le=20000, description="Max affected files to return"),
    include_edges: bool = Query(True, description="Also include raw 'in' edges used for impact?"),
    limit_edges: int = Query(5000, ge=1, le=20000, description="Safety cap on returned edges"),
):
    """
    Impact view: given a file, list files likely affected if it changes.
    Combines:
      - graph dependents (1..depth hops)
      - import refs mentioning the token
      - lexical references to the token (tsvector)
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            repo_id, repo_name, file_id, file_path = _resolve_file(cur, repo, file)

            # Default token from basename without extension, e.g., "Person" for "Person.java"
            if not token:
                base = os.path.basename(file_path)
                token = base.split(".", 1)[0] if "." in base else base

            # Compose params
            import_like = f"%{token}%"

            sql = _impact_sql()
            rows = _all(
                cur,
                sql,
                (
                    file_id,              # start.file_id
                    depth,                # dep depth
                    repo_id,              # dep_files filter
                    repo_id,              # import_refs repo
                    import_like,          # import text LIKE
                    repo_id,              # lex_refs repo
                    token,                # lexical token
                    token,                # label for imports
                    token,                # label for lexical
                    file_id,              # exclude target file
                    limit_files,          # LIMIT
                ),
            )

            result: Dict[str, Any] = {
                "repo": repo_name,
                "target": {"file_id": file_id, "path": file_path},
                "depth": depth,
                "token": token,
                "affected_files": rows,
            }

            if include_edges:
                # Provide raw 'in' direction edges used for dependents
                e_sql = _edges_query("in")
                edges = _all(cur, e_sql, (file_id, depth, repo_id, repo_id))
                result["edges"] = edges[:limit_edges]

                # Quick counts by relation
                counts: Dict[str, int] = {}
                for e in result["edges"]:
                    counts[e["relation"]] = counts.get(e["relation"], 0) + 1
                result["counts_by_relation"] = counts

            return result
