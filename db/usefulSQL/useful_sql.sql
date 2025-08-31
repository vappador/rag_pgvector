/*
  useful_sql.sql

  Purpose:
    Self-contained SQL healthcheck & evaluation toolkit for your RAG ingestion in PostgreSQL (using pgvector).

  Run in psql to validate ingestion quality before integrating into ingestion.py.

  Key enhancements:
    - Measures both raw and eligible intra-file resolution %
    - Adds normalization and member-aware matching
    - Includes diagnostics and optional backfill suggestions
*/

-- EDIT AS NEEDED:
// SET search_path TO public; -- or your schema
-- SET ivfflat.probes = 10;   -- tune for vector recall if using ANN

-- 1) Define normalization helpers
CREATE OR REPLACE FUNCTION norm_full(t text)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT NULLIF(
    regexp_replace(
      regexp_replace(split_part($1,'#',2), '\(.*\)$', ''),  -- strip trailing ()
      '<[^>]+>', '', 'g'                                   -- strip generics like <T>
    ), ''
  );
$$;

CREATE OR REPLACE FUNCTION norm_tail(t text)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
  WITH b AS (SELECT norm_full($1) AS s)
  SELECT NULLIF(
    split_part((SELECT s FROM b), '.', array_length(string_to_array((SELECT s FROM b),'.'),1)),
    ''
  );
$$;

CREATE OR REPLACE FUNCTION split_class_method(sym text)
RETURNS TABLE(cls text, meth text) LANGUAGE sql IMMUTABLE AS $$
  SELECT
    NULLIF(split_part(split_part(norm_full(sym),'#',2),'.',1), '') AS cls,
    NULLIF(split_part(split_part(norm_full(sym),'#',2),'.',2), '') AS meth;
$$;

-- 2) RAW resolution metric (original, straightforward)
WITH calls_data AS (
  SELECT cardinality(calls) AS n_calls
  FROM code_chunks
)
SELECT
  'raw' AS metric,
  SUM(n_calls) AS total_calls_recorded,
  (SELECT count(*) FROM chunk_edges WHERE edge_type='calls') AS call_edges_created,
  ROUND(
    100.0 * (SELECT count(*) FROM chunk_edges WHERE edge_type='calls')
    / NULLIF(SUM(n_calls),0),
    1
  ) AS pct_resolved_intra_file_raw
FROM calls_data;

-- 3) ELIGIBLE-only resolution (fair normalization-aware)
WITH
 defs AS (
   SELECT
     file_id,
     array_agg(DISTINCT norm_full(symbol_name)) FILTER (WHERE symbol_name IS NOT NULL) AS defs_full,
     array_agg(DISTINCT norm_tail(symbol_name)) FILTER (WHERE symbol_name IS NOT NULL) AS defs_tail,
     array_agg(DISTINCT (split_class_method(symbol_name)).meth) FILTER (WHERE symbol_name IS NOT NULL) AS defs_meth
   FROM code_chunks
   GROUP BY file_id
 ),
 calls_expanded AS (
   SELECT
     id AS src_chunk_id,
     file_id,
     norm_full(unnest(calls)) AS call_full,
     norm_tail(unnest(calls)) AS call_tail
   FROM code_chunks
 ),
 eligible AS (
   SELECT DISTINCT ce.src_chunk_id
   FROM calls_expanded ce
   JOIN defs d USING (file_id)
   WHERE
     (ce.call_full IS NOT NULL AND ce.call_full = ANY(d.defs_full))
     OR (ce.call_tail IS NOT NULL AND ce.call_tail = ANY(d.defs_tail))
     OR (ce.call_tail IS NOT NULL AND ce.call_tail = ANY(d.defs_meth))
 ),
 resolved_intra AS (
   SELECT DISTINCT e.src_chunk_id
   FROM chunk_edges e
   JOIN code_chunks s ON s.id = e.src_chunk_id
   JOIN code_chunks t ON t.id = e.dst_chunk_id
   WHERE e.edge_type='calls' AND s.file_id = t.file_id
 )
SELECT
  'eligible' AS metric,
  (SELECT count(*) FROM eligible) AS eligible_calls_intra_file,
  (SELECT count(*) FROM resolved_intra) AS resolved_call_edges_intra_file,
  ROUND(
    100.0 * (SELECT count(*) FROM resolved_intra)
      / NULLIF((SELECT count(*) FROM eligible),0),
    1
  ) AS pct_resolved_of_eligible_intra;

-- 4) Diagnostics: unmatched calls list (top offenders)
WITH
 defs_for_uf AS (
   SELECT file_id,
          array_agg(DISTINCT norm_full(symbol_name)) FILTER (WHERE symbol_name IS NOT NULL) AS defs_full
   FROM code_chunks GROUP BY file_id
 ),
 calls_exp AS (
   SELECT file_id, norm_full(unnest(calls)) AS call_norm
   FROM code_chunks
 ),
 unmatched AS (
   SELECT ce.file_id, ce.call_norm
   FROM calls_exp ce
   LEFT JOIN defs_for_uf d ON d.file_id = ce.file_id
   WHERE ce.call_norm IS NOT NULL AND NOT ce.call_norm = ANY(d.defs_full)
 )
SELECT cf.path, u.call_norm, COUNT(*) AS occurrences
FROM unmatched u
JOIN code_files cf ON cf.id = u.file_id
GROUP BY cf.path, u.call_norm
ORDER BY occurrences DESC
LIMIT 50;

-- 5) Optional: suggestions for backfilling (comment out unless needed)
-- BEGIN;
-- WITH defs_backfill AS (
--   SELECT file_id, id AS dst_chunk_id, norm_full(symbol_name) AS def_norm
--   FROM code_chunks WHERE symbol_name IS NOT NULL
-- ), calls_backfill AS (
--   SELECT id AS src_chunk_id, file_id, norm_full(unnest(calls)) AS call_norm
--   FROM code_chunks
-- ), to_insert AS (
--   SELECT cb.src_chunk_id, d.dst_chunk_id
--   FROM calls_backfill cb
--   JOIN defs_backfill d ON cb.file_id = d.file_id AND cb.call_norm = d.def_norm
--   LEFT JOIN chunk_edges ce ON ce.edge_type='calls' AND ce.src_chunk_id = cb.src_chunk_id AND ce.dst_chunk_id = d.dst_chunk_id
--   WHERE ce.id IS NULL
-- )
-- INSERT INTO chunk_edges(edge_type, src_chunk_id, dst_chunk_id)
-- SELECT 'calls', src_chunk_id, dst_chunk_id FROM to_insert;
-- COMMIT;

-- End of file
