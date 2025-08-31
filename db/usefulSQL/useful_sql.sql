/* =============================================================================
   RAG Ingestion Healthcheck Toolkit (PostgreSQL)
   -----------------------------------------------------------------------------
   Purpose:
     Quick, copy-pastable queries to assess how your code ingestion ran and
     whether the database is “good enough” for a code-RAG workflow.

   Usage:
     - Paste into psql or your SQL client and run sections you need.
     - If your tables live in a custom schema (e.g., rag.*), set search_path
       accordingly (see “Schema” section below).

   Interpreting Results (rules of thumb):
     ✅ More chunks than files (several per file) → AST chunking worked.
     ✅ High embedding coverage (>90%) → vector search ready.
     ✅ Low windowing rate (<20–30%) → fewer fallback chunks.
     ✅ Some edges per chunk (depends on stack) → better graph navigation.
     ⚠️ Many orphan chunks or unresolved callees → consider improving call
        extraction and/or adding cross-file edges.

   Postgres version:
     These queries use standard Postgres + pgvector features present in 14+.

   ============================================================================= */


/* -----------------------------------------------------------------------------
   Schema (pick ONE of the following lines and uncomment it)
   ----------------------------------------------------------------------------- */

-- SET search_path TO public;             -- tables live in "public" (default)
-- SET search_path TO rag, public;        -- if you created tables in schema "rag"


/* -----------------------------------------------------------------------------
   Optional: pgvector runtime tuning for ANN queries in this session
   ----------------------------------------------------------------------------- */

-- SET ivfflat.probes = 10;  -- increase for better recall (slower); 5–20 is common


/* =============================================================================
   1) Overall ingestion summary
   What “good” looks like:
     - repos ≥ 1, files ≥ 10s–1000s depending on source
     - chunks >> files (AST chunking emits multiple chunks per file)
     - edges present if you enabled edge building
   ============================================================================= */

SELECT
  (SELECT count(*) FROM repositories) AS repos,
  (SELECT count(*) FROM code_files)   AS files,
  (SELECT count(*) FROM code_chunks)  AS chunks,
  (SELECT count(*) FROM chunk_edges)  AS edges;


/* =============================================================================
   2) Per-repo summary (files, chunks, edges, last SHA)
   Good:
     - Non-zero chunks per repo
     - last_ingested_sha populated for incremental ingest
   ============================================================================= */

SELECT r.name, r.last_ingested_sha,
       count(DISTINCT f.id) AS files,
       count(c.id)          AS chunks,
       count(e.id)          AS edges
FROM repositories r
LEFT JOIN code_files  f ON f.repository_id = r.id
LEFT JOIN code_chunks c ON c.file_id       = f.id
LEFT JOIN chunk_edges e ON e.src_chunk_id  = c.id
GROUP BY r.id
ORDER BY chunks DESC, files DESC;


/* =============================================================================
   3) Language distribution & average chunk size
   Good:
     - Languages you expect are present
     - Reasonable avg chunk length (e.g., 150–800 chars for code)
   ============================================================================= */

SELECT COALESCE(c.language,'?') AS lang,
       count(*)                                       AS chunks,
       round(avg(char_length(c.content)))             AS avg_chars,
       round(avg(c.end_line - c.start_line + 1))      AS avg_lines
FROM code_chunks c
GROUP BY 1
ORDER BY chunks DESC;


/* =============================================================================
   4) Fallback windowing vs. AST chunks
   Good:
     - Higher ast_chunks and lower pct_windowed indicate better AST coverage.
   Notes:
     symbol_kind heuristics:
       - 'function'/'method'/'class'  → AST-based chunks
       - 'module'                     → whole-file fallback (no window)
       - 'module_part'                → windowed fallback
   ============================================================================= */

SELECT
  sum( (c.symbol_kind = 'module_part')::int )                    AS windowed_chunks,
  sum( (c.symbol_kind = 'module')::int )                         AS whole_file_fallback,
  sum( (c.symbol_kind IN ('function','method','class'))::int )   AS ast_chunks,
  count(*)                                                       AS total_chunks,
  round(100.0 * sum( (c.symbol_kind = 'module_part')::int ) / NULLIF(count(*),0), 1) AS pct_windowed
FROM code_chunks c;


/* =============================================================================
   5) Files with NO chunks (missed by language filter, empty, or errors)
   Good:
     - Ideally empty result. Investigate any important files listed here.
   ============================================================================= */

SELECT r.name AS repo, f.path
FROM code_files f
JOIN repositories r ON r.id = f.repository_id
LEFT JOIN code_chunks c ON c.file_id = f.id
WHERE c.id IS NULL
ORDER BY 1,2
LIMIT 100;


/* =============================================================================
   6) Embedding coverage
   Good:
     - pct_with_emb >= 90% for vector search heavy flows.
   ============================================================================= */

SELECT
  count(*) FILTER (WHERE embedding IS NOT NULL) AS with_emb,
  count(*) FILTER (WHERE embedding IS NULL)     AS without_emb,
  round(100.0 * count(*) FILTER (WHERE embedding IS NOT NULL) / NULLIF(count(*),0),1) AS pct_with_emb
FROM code_chunks;


/* =============================================================================
   7) Calls recorded vs. call edges created (intra-file resolution effectiveness)
   Good:
     - pct_resolved_intra_file > 20–50% (varies a lot by codebase & extractor).
   Notes:
     - If low, consider improving call extraction and/or adding cross-file edges.
   ============================================================================= */

WITH calls AS (
  SELECT id AS chunk_id, cardinality(calls) AS n_calls
  FROM code_chunks
)
SELECT
  sum(n_calls) AS total_calls_recorded,
  (SELECT count(*) FROM chunk_edges WHERE edge_type='calls') AS call_edges_created,
  round(
    100.0 * (SELECT count(*) FROM chunk_edges WHERE edge_type='calls')
    / NULLIF(sum(n_calls),0),
    1
  ) AS pct_resolved_intra_file
FROM calls;


/* =============================================================================
   8) Orphan chunks (no incoming AND no outgoing edges)
   Good:
     - Some orphans are normal; many may indicate weak linking.
   ============================================================================= */

SELECT count(*) AS orphan_chunks
FROM code_chunks c
LEFT JOIN chunk_edges e_out ON e_out.src_chunk_id = c.id
LEFT JOIN chunk_edges e_in  ON e_in.dst_chunk_id  = c.id
WHERE e_out.id IS NULL AND e_in.id IS NULL;


/* =============================================================================
   9) Top files by edge density (edges per chunk)
   Good:
     - Higher edges_per_chunk often improves graph-walk retrieval.
   ============================================================================= */

SELECT r.name AS repo, f.path,
       count(c.id) AS chunks,
       count(e.id) AS edges,
       round(1.0 * count(e.id) / NULLIF(count(c.id),0), 2) AS edges_per_chunk
FROM code_files f
JOIN repositories r ON r.id = f.repository_id
LEFT JOIN code_chunks c ON c.file_id = f.id
LEFT JOIN chunk_edges e ON e.src_chunk_id = c.id
GROUP BY r.name, f.path
ORDER BY edges_per_chunk DESC NULLS LAST, chunks DESC
LIMIT 25;


/* =============================================================================
   10) Unresolved callees (calls that didn’t match any local symbol in file)
   Good:
     - Use this to refine the extractor (e.g., member calls router.get → get).
   ============================================================================= */

WITH symbols AS (
  SELECT file_id, array_agg(DISTINCT split_part(symbol_name,'#',2)) AS sym
  FROM code_chunks
  GROUP BY file_id
),
calls AS (
  SELECT file_id, unnest(calls) AS callee
  FROM code_chunks
)
SELECT f.path, c.callee, count(*) AS occurrences
FROM calls c
JOIN symbols s ON s.file_id = c.file_id
JOIN code_files f ON f.id = c.file_id
WHERE c.callee IS NOT NULL AND c.callee <> ''
  AND NOT (c.callee = ANY(s.sym))
GROUP BY f.path, c.callee
ORDER BY occurrences DESC
LIMIT 50;


/* =============================================================================
   11) Duplicate chunk content across files (possible copies to dedupe)
   Good:
     - Some duplication is normal; a lot may suggest vendor code or generated code.
   ============================================================================= */

SELECT content_hash, count(*) AS n,
       array_agg(DISTINCT r.name || ':' || f.path) AS locations
FROM code_chunks c
JOIN code_files f ON f.id = c.file_id
JOIN repositories r ON r.id = f.repository_id
GROUP BY content_hash
HAVING count(*) > 1
ORDER BY n DESC
LIMIT 25;


/* =============================================================================
   12) Retrieval probes (text & vector)
   Notes:
     - Replace the sample text or vector with your target query.
     - ANN quality will depend on embedding model & chunking quality.
   ============================================================================= */

-- 12a) TEXT (full-text)
SELECT r.name, f.path, c.symbol_name,
       ts_rank(c.content_tsv, plainto_tsquery('english','router get')) AS rank
FROM code_chunks c
JOIN code_files f ON f.id = c.file_id
JOIN repositories r ON r.id = f.repository_id
WHERE c.content_tsv @@ plainto_tsquery('english','router get')
ORDER BY rank DESC
LIMIT 10;

-- 12b) VECTOR (ANN): supply a 1024-d query vector (or bind param).
-- Example skeleton only (replace [...] with your vector):
-- SELECT r.name, f.path, c.symbol_name
-- FROM code_chunks c
-- JOIN code_files f ON f.id = c.file_id
-- JOIN repositories r ON r.id = f.repository_id
-- WHERE c.embedding IS NOT NULL
-- ORDER BY c.embedding <-> '[ /* 1024 floats here */ ]'::vector
-- LIMIT 10;

/* =============================================================================
   13) How many calls did we capture vs edges built
   ============================================================================= */
-- total call tokens recorded
SELECT SUM(array_length(calls,1)) AS total_calls
FROM code_chunks;

-- edges proportion
SELECT
  (SELECT COUNT(*) FROM chunk_edges WHERE edge_type='calls') AS call_edges,
  (SELECT SUM(array_length(calls,1)) FROM code_chunks)       AS recorded_calls,
  (SELECT COUNT(*)::float / NULLIF(SUM(array_length(calls,1)),0)
     FROM code_chunks, LATERAL (SELECT 1) s)                 AS dummy_ratio; -- just placeholder

-- per-file effectiveness
SELECT r.name AS repo, f.path,
       SUM(array_length(c.calls,1)) AS calls_recorded,
       SUM(CE.edge_count) AS call_edges
FROM code_files f
JOIN repositories r ON r.id=f.repository_id
JOIN code_chunks c ON c.file_id=f.id
LEFT JOIN LATERAL (
  SELECT COUNT(*) AS edge_count
  FROM chunk_edges e WHERE e.src_chunk_id = c.id AND e.edge_type='calls'
) CE ON TRUE
GROUP BY 1,2
ORDER BY calls_recorded DESC NULLS LAST;

/* =============================================================================
   14) find callers that still don't resolve
   ============================================================================= */
WITH caller_calls AS (
  SELECT c.id AS src_chunk_id, unnest(c.calls) AS call FROM code_chunks c
),
resolved AS (
  SELECT DISTINCT e.src_chunk_id FROM chunk_edges e WHERE e.edge_type='calls'
)
SELECT cc.call, COUNT(*) AS misses
    FROM caller_calls cc
LEFT JOIN resolved r ON r.src_chunk_id = cc.src_chunk_id
WHERE r.src_chunk_id IS NULL
GROUP BY cc.call
ORDER BY misses DESC
LIMIT 50;


/* =============================================================================
   (Optional) Maintenance tips after bulk ingest
   ============================================================================= */

-- VACUUM (ANALYZE, VERBOSE) code_chunks;
-- ANALYZE code_chunks;
-- REINDEX INDEX idx_chunks_embedding_ivf;   -- occasionally for pgvector IVFFLAT