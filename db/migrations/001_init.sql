CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.repositories (
  id BIGSERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,         -- org/repo or local name
  url  TEXT NOT NULL,
  default_branch TEXT NOT NULL DEFAULT 'main',
  last_commit TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag.commits (
  id BIGSERIAL PRIMARY KEY,
  repo_id BIGINT NOT NULL REFERENCES rag.repositories(id) ON DELETE CASCADE,
  commit_hash TEXT NOT NULL,
  author TEXT,
  committed_at TIMESTAMPTZ,
  message TEXT,
  UNIQUE (repo_id, commit_hash)
);

DO $$ BEGIN
  CREATE TYPE rag.language AS ENUM ('typescript','javascript','java','other');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS rag.files (
  id BIGSERIAL PRIMARY KEY,
  repo_id BIGINT NOT NULL REFERENCES rag.repositories(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  language rag.language NOT NULL,
  is_test BOOLEAN NOT NULL DEFAULT FALSE,
  size_bytes INTEGER,
  latest_commit_id BIGINT REFERENCES rag.commits(id) ON DELETE SET NULL,
  UNIQUE (repo_id, path)
);

CREATE TABLE IF NOT EXISTS rag.code_chunks (
  id BIGSERIAL PRIMARY KEY,
  file_id BIGINT NOT NULL REFERENCES rag.files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  symbol_name TEXT,
  symbol_kind TEXT,
  symbol_signature TEXT,
  doc_comment TEXT,
  imports JSONB,
  calls JSONB,
  content TEXT NOT NULL,
  content_tsv tsvector,
  embedding vector(1024),
  token_count INTEGER,
  committed_at TIMESTAMPTZ,
  valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
  valid_to TIMESTAMPTZ,
  CONSTRAINT uq_file_chunk UNIQUE (file_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS rag.chunk_edges (
  src_chunk_id BIGINT NOT NULL REFERENCES rag.code_chunks(id) ON DELETE CASCADE,
  dst_chunk_id BIGINT NOT NULL REFERENCES rag.code_chunks(id) ON DELETE CASCADE,
  relation TEXT NOT NULL,  -- calls/imports/implements/tests
  PRIMARY KEY (src_chunk_id, dst_chunk_id, relation)
);

CREATE INDEX IF NOT EXISTS code_chunks_tsv_gin ON rag.code_chunks USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS code_chunks_symbol_name_idx ON rag.code_chunks(symbol_name);
CREATE INDEX IF NOT EXISTS code_chunks_symbol_sig_idx ON rag.code_chunks(symbol_signature);
CREATE INDEX IF NOT EXISTS code_chunks_imports_gin ON rag.code_chunks USING GIN (imports);
CREATE INDEX IF NOT EXISTS code_chunks_calls_gin   ON rag.code_chunks USING GIN (calls);

-- ANN index (IVFFLAT)
DO $$ BEGIN
  CREATE INDEX code_chunks_embedding_ivfflat
    ON rag.code_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

CREATE OR REPLACE FUNCTION rag.update_tsv() RETURNS trigger AS $$
BEGIN
  NEW.content_tsv := to_tsvector('simple', unaccent(NEW.content));
  RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS code_chunks_tsv_trg ON rag.code_chunks;
CREATE TRIGGER code_chunks_tsv_trg
BEFORE INSERT OR UPDATE OF content ON rag.code_chunks
FOR EACH ROW EXECUTE FUNCTION rag.update_tsv();

CREATE OR REPLACE VIEW rag.v_chunk_search AS
SELECT
  cc.id, f.id AS file_id, r.id AS repo_id,
  r.name AS repo_name, f.path, f.language, cc.symbol_name, cc.symbol_kind, cc.symbol_signature,
  cc.start_line, cc.end_line, cc.content, cc.embedding, cc.content_tsv, cc.token_count, cc.committed_at
FROM rag.code_chunks cc
JOIN rag.files f ON f.id = cc.file_id
JOIN rag.repositories r ON r.id = f.repo_id
WHERE cc.valid_to IS NULL;
