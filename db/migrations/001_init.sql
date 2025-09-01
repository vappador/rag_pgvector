-- 001_init.sql (updated)
-- Fresh schema for code-focused RAG on Postgres + pgvector
-- Safe to run on a brand-new database.

BEGIN;

-- Extensions ------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Tables ----------------------------------------------------------------------

-- A source control repository (e.g., one Git repo)
CREATE TABLE IF NOT EXISTS repositories (
  id                BIGSERIAL PRIMARY KEY,
  name              TEXT NOT NULL UNIQUE,
  url               TEXT,
  default_branch    TEXT DEFAULT 'main',
  last_ingested_sha TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Files within a repository (logical path, language)
CREATE TABLE IF NOT EXISTS code_files (
  id             BIGSERIAL PRIMARY KEY,
  repository_id  BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
  path           TEXT   NOT NULL,        -- repo-relative path
  language       TEXT,                   -- detected language (java, typescript, etc.)
  size_bytes     INTEGER,
  checksum       TEXT,                   -- optional (e.g., sha256 of file content)
  module         TEXT,                   -- NEW: derived module/package identifier
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (repository_id, path)
);

-- Chunks: function/method/class-level units (AST-aware)
CREATE TABLE IF NOT EXISTS code_chunks (
  id              BIGSERIAL PRIMARY KEY,
  file_id         BIGINT NOT NULL REFERENCES code_files(id) ON DELETE CASCADE,

  -- Positioning within file
  start_line      INTEGER NOT NULL,
  end_line        INTEGER NOT NULL,

  -- Content and hybrid search columns
  content         TEXT    NOT NULL,
  content_tsv     TSVECTOR,

  -- Language + symbol metadata
  language        TEXT,
  symbol_kind     TEXT,                           -- class | method | function | module | ...
  symbol_name     TEXT,                           -- e.g. Person#updateEmail or updateEmail
  symbol_simple   TEXT,                           -- NEW: normalized callee key ('updateEmail')

  -- AST-derived metadata
  imports         TEXT[] NOT NULL DEFAULT '{}',
  calls           TEXT[] NOT NULL DEFAULT '{}',
  ast             JSONB,

  -- Embeddings (fixed dim; match your embedding model, default 1024 for mxbai-embed-large)
  embedding       VECTOR(1024),
  embedding_model TEXT,
  embedding_hash  TEXT,
  content_hash    TEXT,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (file_id, start_line, end_line)
);

ALTER TABLE code_chunks
  ADD COLUMN IF NOT EXISTS symbol_signature TEXT;

-- (Optional) seed it
UPDATE code_chunks
SET symbol_signature = symbol_name
WHERE symbol_signature IS NULL;

-- (Optional) index
CREATE INDEX IF NOT EXISTS idx_chunks_symbol_signature
  ON code_chunks(symbol_signature);

-- Then your view can use c.symbol_signature directly (no NULL stub).


-- Graph edges between chunks (code-aware navigation)
CREATE TABLE IF NOT EXISTS chunk_edges (
  id            BIGSERIAL PRIMARY KEY,
  src_chunk_id  BIGINT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
  dst_chunk_id  BIGINT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
  edge_type     TEXT   NOT NULL CHECK (edge_type IN ('calls','imports','references')),
  weight        REAL   NOT NULL DEFAULT 1.0,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (src_chunk_id, dst_chunk_id, edge_type)
);

-- NEW: normalized import surface per file
CREATE TABLE IF NOT EXISTS file_imports (
  id          BIGSERIAL PRIMARY KEY,
  file_id     BIGINT NOT NULL REFERENCES code_files(id) ON DELETE CASCADE,
  raw_import  TEXT NOT NULL,
  module      TEXT,         -- e.g., 'com.acme.util' or 'lodash/get' or 'pkg.mod'
  symbol      TEXT,         -- e.g., 'ClassName' or '{fn}'
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- NEW: symbol/definition surface per chunk
CREATE TABLE IF NOT EXISTS code_symbols (
  id            BIGSERIAL PRIMARY KEY,
  repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
  file_id       BIGINT NOT NULL REFERENCES code_files(id) ON DELETE CASCADE,
  chunk_id      BIGINT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
  language      TEXT,
  module        TEXT,
  class_name    TEXT,
  symbol_simple TEXT,   -- e.g. 'updateEmail'
  symbol_full   TEXT,   -- e.g. 'Person#updateEmail' or 'pkg.mod.fn'
  exported      BOOLEAN DEFAULT FALSE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (chunk_id)
);

-- Triggers & helper functions --------------------------------------------------

-- Keep content_tsv in sync with content (also set directly by ingest)
CREATE OR REPLACE FUNCTION trg_update_content_tsv() RETURNS trigger AS $$
BEGIN
  NEW.content_tsv := to_tsvector('simple', unaccent(coalesce(NEW.content,'')));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_chunks_tsv ON code_chunks;
CREATE TRIGGER trig_chunks_tsv
BEFORE INSERT OR UPDATE OF content ON code_chunks
FOR EACH ROW
EXECUTE FUNCTION trg_update_content_tsv();

-- Updated_at timestamps
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_repos_updated_at ON repositories;
CREATE TRIGGER trig_repos_updated_at
BEFORE UPDATE ON repositories
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trig_files_updated_at ON code_files;
CREATE TRIGGER trig_files_updated_at
BEFORE UPDATE ON code_files
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trig_chunks_updated_at ON code_chunks;
CREATE TRIGGER trig_chunks_updated_at
BEFORE UPDATE ON code_chunks
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Indexes ---------------------------------------------------------------------

-- Hybrid search: textual
CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin        ON code_chunks USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_language       ON code_chunks (language);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol_btree   ON code_chunks (symbol_name);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol_simple  ON code_chunks (symbol_simple);
CREATE INDEX IF NOT EXISTS idx_files_path_btree      ON code_files (path);
CREATE INDEX IF NOT EXISTS idx_files_module          ON code_files (module);

-- Arrays & JSONB for AST metadata
CREATE INDEX IF NOT EXISTS idx_chunks_imports_gin    ON code_chunks USING GIN (imports);
CREATE INDEX IF NOT EXISTS idx_chunks_calls_gin      ON code_chunks USING GIN (calls);
CREATE INDEX IF NOT EXISTS idx_chunks_ast_gin        ON code_chunks USING GIN (ast jsonb_path_ops);

-- Vector index (choose one — IVFFLAT is broadly available)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivf ON code_chunks
USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

-- Edges convenience indexes
CREATE INDEX IF NOT EXISTS idx_edges_src             ON chunk_edges (src_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst             ON chunk_edges (dst_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_type            ON chunk_edges (edge_type);

-- file_imports / code_symbols indexes
CREATE INDEX IF NOT EXISTS idx_file_imports_file     ON file_imports (file_id);
CREATE INDEX IF NOT EXISTS idx_file_imports_module   ON file_imports (module);
CREATE INDEX IF NOT EXISTS idx_code_symbols_repo     ON code_symbols (repository_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_file     ON code_symbols (file_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_chunk    ON code_symbols (chunk_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_ss       ON code_symbols (symbol_simple);
CREATE INDEX IF NOT EXISTS idx_code_symbols_exported ON code_symbols (exported);

-- Views -----------------------------------------------------------------------
-- Public search view used by the retriever
CREATE OR REPLACE VIEW v_chunk_search AS
SELECT
  c.id AS chunk_id,
  r.name AS repo_name,
  f.path,
  f.language,
  c.symbol_name,
  c.symbol_kind,
  c.symbol_signature,
  c.start_line,
  c.end_line,
  c.content,
  c.embedding,
  c.content_tsv
FROM code_chunks AS c
JOIN code_files  AS f ON f.id = c.file_id
JOIN repositories AS r ON r.id = f.repository_id;

-- Handy view for provenance strings like repo:path:start-end
CREATE OR REPLACE VIEW v_chunk_citation AS
SELECT
  c.id AS chunk_id,
  r.name AS repo,
  f.path,
  c.start_line,
  c.end_line,
  COALESCE(c.symbol_name, '') AS symbol,
  (r.name || ':' || f.path || ':' || c.start_line || '-' || c.end_line) AS citation
FROM code_chunks c
JOIN code_files f ON f.id = c.file_id
JOIN repositories r ON r.id = f.repository_id;

-- Functions -------------------------------------------------------------------

-- Cross-file call edge builder (fixed GET DIAGNOSTICS usage)
CREATE OR REPLACE FUNCTION build_cross_file_call_edges(
  max_per_src    integer DEFAULT 3,
  sim_threshold  real    DEFAULT 0.62   -- kept for API compatibility; not used in this matcher
)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
  created_edges integer := 0;
  last_rows     integer := 0;
BEGIN
  WITH exploded AS (
    SELECT
      src.id         AS src_chunk_id,
      src.file_id    AS src_file_id,
      lower(trim(call_name)) AS call_name
    FROM code_chunks AS src
    CROSS JOIN LATERAL unnest(src.calls) AS call_name
  ),
  candidates AS (
    SELECT
      e.src_chunk_id,
      dst.id              AS dst_chunk_id,
      dst.file_id         AS dst_file_id,
      cs.symbol_simple,
      row_number() OVER (PARTITION BY e.src_chunk_id ORDER BY cs.exported DESC, cs.symbol_full) AS rn
    FROM exploded e
    JOIN code_symbols cs
      ON lower(cs.symbol_simple) = e.call_name
    JOIN code_chunks dst
      ON dst.id = cs.chunk_id
    WHERE e.src_file_id <> dst.file_id
  )
  INSERT INTO chunk_edges (src_chunk_id, dst_chunk_id, edge_type, weight)
  SELECT DISTINCT c.src_chunk_id, c.dst_chunk_id, 'calls', 1.0
  FROM candidates c
  WHERE c.rn <= max_per_src
  ON CONFLICT (src_chunk_id, dst_chunk_id, edge_type) DO NOTHING;

  GET DIAGNOSTICS last_rows = ROW_COUNT;
  created_edges := created_edges + last_rows;

  RETURN created_edges;
END;
$$;

COMMIT;

-- Post-load tips:
-- * Run ANALYZE after bulk ingest:   VACUUM (ANALYZE, VERBOSE) code_chunks;
-- * For IVFFLAT, also set probes at query time: SET ivfflat.probes = 10;
-- * If you change embedding dims, you must ALTER the column and rebuild indexes:
--     ALTER TABLE code_chunks ALTER COLUMN embedding TYPE vector(<new_dim>);
--     DROP INDEX IF EXISTS idx_chunks_embedding_ivf;
--     CREATE INDEX idx_chunks_embedding_ivf ON code_chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
