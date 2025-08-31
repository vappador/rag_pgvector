-- 001_init.sql
-- Fresh schema for code-focused RAG on Postgres + pgvector
-- Safe to run on a brand-new database.

BEGIN;

-- Extensions ------------------------------------------------------------------
-- pgvector extension is named "vector"
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- optional, helpful for fuzzy text
CREATE EXTENSION IF NOT EXISTS btree_gin;        -- optional, for mixed GIN use

-- Optional: declare a domain so changing embedding dimension is one-line
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_type t
    JOIN pg_namespace n ON n.oid = t.typnamespace
    WHERE t.typname = 'embedding_vec' AND n.nspname = 'public'
  ) THEN
    -- Default to 1024 dims (Ollama mxbai-embed-large). Change here if you switch models.
    EXECUTE 'CREATE DOMAIN embedding_vec AS vector(1024)';
  END IF;
END$$;

-- Tables ----------------------------------------------------------------------

-- A source control repository (e.g., one Git repo)
CREATE TABLE IF NOT EXISTS repositories (
  id                BIGSERIAL PRIMARY KEY,
  name              TEXT NOT NULL UNIQUE,
  url               TEXT,
  default_branch    TEXT DEFAULT 'main',
  last_ingested_sha TEXT,                 -- for incremental ingest
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
  language        TEXT,                           -- redundant but handy for filtering
  symbol_kind     TEXT,                           -- class | method | function | module | ...
  symbol_name     TEXT,                           -- e.g. Person#updateEmail or updateEmail

  -- AST-derived metadata
  imports         TEXT[] NOT NULL DEFAULT '{}',
  calls           TEXT[] NOT NULL DEFAULT '{}',
  ast             JSONB,

  -- Embeddings
  embedding       embedding_vec,                  -- vector(n); n defined by domain above
  embedding_model TEXT,                           -- e.g., "mxbai-embed-large"
  embedding_hash  TEXT,                           -- hash of embedding input to avoid re-embed
  content_hash    TEXT,                           -- hash of chunk content for idempotence

  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- A chunk is uniquely identified by its file + line span
  UNIQUE (file_id, start_line, end_line)
);

-- Graph edges between chunks (code-aware navigation)
CREATE TABLE IF NOT EXISTS chunk_edges (
  id            BIGSERIAL PRIMARY KEY,
  src_chunk_id  BIGINT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
  dst_chunk_id  BIGINT NOT NULL REFERENCES code_chunks(id) ON DELETE CASCADE,
  edge_type     TEXT   NOT NULL CHECK (edge_type IN ('calls','imports','references')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (src_chunk_id, dst_chunk_id, edge_type)
);

-- Triggers & helper functions --------------------------------------------------

-- Keep content_tsv in sync with content
CREATE OR REPLACE FUNCTION trg_update_content_tsv() RETURNS trigger AS $$
BEGIN
  NEW.content_tsv :=
    to_tsvector('english', coalesce(NEW.content,''));
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
CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin       ON code_chunks USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_language      ON code_chunks (language);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol_btree  ON code_chunks (symbol_name);
CREATE INDEX IF NOT EXISTS idx_files_path_btree     ON code_files (path);

-- Arrays & JSONB for AST metadata
CREATE INDEX IF NOT EXISTS idx_chunks_imports_gin   ON code_chunks USING GIN (imports);
CREATE INDEX IF NOT EXISTS idx_chunks_calls_gin     ON code_chunks USING GIN (calls);
CREATE INDEX IF NOT EXISTS idx_chunks_ast_gin       ON code_chunks USING GIN (ast jsonb_path_ops);

-- Vector index (choose one — IVFFLAT is broadly available)
-- Note: IVFFLAT requires setting `SET enable_seqscan = off;` during benchmarking,
-- and you should ANALYZE after bulk loads. Tune `lists` per dataset size.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivf ON code_chunks
USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

-- If your pgvector version supports HNSW and you prefer it, use this instead:
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON code_chunks
-- USING hnsw (embedding vector_l2_ops);

-- Edges convenience indexes
CREATE INDEX IF NOT EXISTS idx_edges_src             ON chunk_edges (src_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst             ON chunk_edges (dst_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_type            ON chunk_edges (edge_type);

-- Views -----------------------------------------------------------------------

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

COMMIT;

-- Post-load tips:
-- * Run ANALYZE after bulk ingest:   VACUUM (ANALYZE, VERBOSE) code_chunks;
-- * For IVFFLAT, also set probes at query time: SET ivfflat.probes = 10;
-- * If you change embedding dims, ALTER the domain: DROP DOMAIN embedding_vec; CREATE DOMAIN embedding_vec AS vector(<new_dim>); then ALTER TABLE code_chunks ALTER COLUMN embedding TYPE embedding_vec;
