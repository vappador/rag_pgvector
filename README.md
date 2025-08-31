# RAG Codebase on Postgres + pgvector (Ollama-powered)

Local, production-ready Retrieval-Augmented Generation (RAG) for **code repositories** (TypeScript, Java, Node.js) using:

- **Postgres 16 + pgvector** for hybrid retrieval (vector + lexical + symbol boosts)  
- **Tree-sitter** for AST-aware chunking (function/class-level)  
- **Ollama** (local) for **embeddings** and **chat**  
- **FastAPI** retrieval service with:
  - a **LangGraph**-based agent  
  - a modern **Strands**-based agent (with rate limiting, observability, and tool APIs)  

---

## What you get

- **Clean schema** for repos → files → code chunks (+ AST metadata)  
- **Hybrid search**: vector similarity + BM25/tsvector + symbol/name boosts  
- **Local-only models**: embeddings via Ollama’s `mxbai-embed-large` (1024-dim), chat via `llama3.1:8b`  
- **Simple ingestion**: clone/parse/chunk/embed/index  
- **Agents**:  
  - **LangGraph Agent**: minimal graph that retrieves & answers with stitched, provenance-rich context  
  - **Strands Agent**: production-ready with **rate limits**, **tooling**, and **observability**  

---

## Repository layout

```
rag-pgvector/
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ db/
│  └─ migrations/001_init.sql
├─ app/
│  ├─ requirements.txt
│  ├─ db.py
│  ├─ ingest.py
│  ├─ api.py              # unified API (search, graph, ask endpoints)
│  ├─ agent_graph.py      # LangGraph demo agent
│  └─ strands_agent.py    # Strands demo agent
└─ Makefile
```

---

## Architecture (high-level)

```mermaid
flowchart LR
  subgraph Git
    A[Repo A]:::git --> C
    B[Repo B]:::git --> C
  end

  subgraph Ingestion
    C[Clone / Pull] --> D["Tree-sitter AST chunking<br/>(symbols, doc, imports, calls)"]
    D --> E["Ollama Embeddings<br/>(mxbai-embed-large 1024d)"]
    E --> F[(Postgres + pgvector)]
  end

  subgraph Retrieval API
    G[/HTTP GET /search/] --> H["Ollama Embed Query"]
    H --> F
    F --> I["Hybrid Rank<br/>(vec + BM25 + symbol boost)"]
    I --> J[Context Stitch]
  end

  subgraph Agents
    K[LangGraph Agent] -->|calls| G
    J --> K
    K --> L["Ollama Chat (llama3.1:8b)"]

    M[Strands Agent] -->|Tool: retrieve_context| F
    F --> M
    M --> N["Ollama Chat (llama3.1:8b)"]
  end

classDef git fill:#f0f8ff,stroke:#7aa,stroke-width:1px;
```

---

## LangGraph vs Strands (at a glance)

```mermaid
flowchart TB
  subgraph LangGraph
    LG1([StateGraph])
    LG2[retrieve node]
    LG3[generate_answer node]
    LG1 --> LG2 --> LG3
    LG3 --> LG4[(Ollama Chat)]
    LG2 --> LG5[(Postgres + pgvector)]
  end

  subgraph Strands
    ST1([Agent Loop])
    ST2{{Tool: retrieve_context}}
    ST1 -->|decides tool use| ST2
    ST2 --> ST5[(Postgres + pgvector)]
    ST1 --> ST4[(Ollama Chat)]
    ST1 --> ST6[(Observability + Rate Limits)]
  end
```

**Key differences**  
- **Control flow**: LangGraph uses explicit nodes/edges; Strands uses an agent loop that calls tools as needed.  
- **Retrieval**: In LangGraph, retrieval is a node; in Strands, retrieval is a **Tool** the model can invoke.  
- **Ops features**: Strands sample includes **rate limits** and **metrics** hooks.  
- **Both** share the same Postgres+pgvector hybrid query and Ollama models.  

---

## ⚙️ Configuration

All runtime knobs are environment-driven.

```bash
cp .env.example .env
```

Key variables in `.env`:

```env
# Models
EMB_MODEL=mxbai-embed-large:latest
LLM_MODEL=llama3.1:8b
EMB_DIM=1024

# Strands Agent
STRANDS_TEMPERATURE=0.2
STRANDS_MAX_TOKENS=2048

# Strands Rate Limits
RL_MAX_CONCURRENCY=2
RL_RPM=60
RL_TPM=100000

# Logging
LOG_LEVEL=INFO
LOG_PREVIEW_CHARS=240

# Postgres DSN
DB_DSN=postgresql://rag:ragpwd@pg:5432/ragdb

# Ollama endpoint
OLLAMA_HOST=http://ollama:11434
OLLAMA_BASE_URL=http://ollama:11434
```

> **Note on dimensions:** The DB schema sets `embedding vector(1024)`. If you use a different embedding model (e.g., 768-dim), update both the env and DB column type accordingly (see “Changing embedding dimension” below).

---

## Quickstart

### 1) Bring up core services

```bash
docker compose up -d pg ollama
docker compose --profile init up migrate
docker compose up -d app
```

### 2) Pull models into Ollama

```bash
docker exec -it ollama ollama pull mxbai-embed-large
docker exec -it ollama ollama pull llama3.1:8b
```

### 3) Ingest repositories

```bash
REPO_URLS="https://github.com/expressjs/express.git,https://github.com/spring-projects/spring-petclinic.git" docker compose --profile ingest up --exit-code-from ingest ingest
```

### 4) Try the retrieval API

```bash
curl "http://localhost:8000/search?q=jwt%20verification&language=java"
```

### 5) Run the agents

- **LangGraph CLI demo**

```bash
docker compose exec app python /workspace/app/agent_graph.py
```

- **Strands API demo**

```bash
curl -X POST http://localhost:8000/ask/strands   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'
```

- **LangGraph API demo**

```bash
curl -X POST http://localhost:8000/ask/langgraph   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'
```

---

## 🔌 API Reference

### API Endpoints Table

| Method | Path              | Purpose                                  |
|--------|------------------|------------------------------------------|
| GET    | `/search`        | Hybrid retrieval (vector + lexical)      |
| POST   | `/ask/strands`   | Ask a question via **Strands Agent**     |
| POST   | `/ask/langgraph` | Ask a question via **LangGraph Agent**   |
| GET    | `/graph/relations` | Show dependency relations for a file   |
| GET    | `/graph/impact`    | Show impact analysis of a file change  |

---

### `/search` (retrieval only)

**Example**

```bash
curl "http://localhost:8000/search?q=jwt%20verification&language=java"
```

**Response (abridged)**
```json
{
  "results": [
    {
      "id": 123,
      "repo_name": "spring-petclinic",
      "path": "src/main/java/.../JwtFilter.java",
      "language": "java",
      "symbol_name": "doFilter",
      "symbol_kind": "function",
      "symbol_signature": "public void doFilter(...)",
      "start_line": 12,
      "end_line": 90,
      "score": 1.234,
      "content": "..."
    }
  ],
  "stitched_context": "// spring-petclinic:src/...:12-90 [public void doFilter(...)]\n..."
}
```

---

### `/ask/strands`

**POST body**
```json
{ "question": "Where is JWT verification implemented?" }
```

**curl**
```bash
curl -X POST http://localhost:8000/ask/strands   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'
```

**Response (example)**
```json
{
  "engine": "strands",
  "answer": "JWT verification is implemented in ... [repo:path:line-range signature]"
}
```

---

### `/ask/langgraph`

**POST body**
```json
{ "question": "Where is JWT verification implemented?" }
```

**curl**
```bash
curl -X POST http://localhost:8000/ask/langgraph   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'
```

**Response (example)**
```json
{
  "engine": "langgraph",
  "answer": "JWT verification is implemented in ... [repo:path:line-range signature]"
}
```

---

### `/graph/relations`

Explores dependency edges for a given file.

Example:
```bash
curl "http://localhost:8000/graph/relations?repo=spring-petclinic&file=Person.java&direction=both&depth=2"
```

---

### `/graph/impact`

Shows files impacted by a file change.

Example:
```bash
curl "http://localhost:8000/graph/impact?repo=spring-petclinic&file=Person.java&depth=2"
```

---

## 🧠 Strands Features

- **Rate limiting** (RPM, TPM, concurrency) via `.env`  
- **Observability** via `strands.observability.get_metrics()`  
- **Tools**: retrieval is registered as a Strands tool (`retrieve_context`)  
- **System prompt** ensures provenance is cited inline  

---

## Schema Overview

**Tables**
- `rag.repositories` ⇒ one per repo  
- `rag.commits` ⇒ commit metadata  
- `rag.files` ⇒ file metadata (language, size, test flag)  
- `rag.code_chunks` ⇒ **the core**: function/class or windowed chunks  
  - content, `embedding vector(1024)`, `content_tsv` (lexical), `symbol_name`, `symbol_kind`, `symbol_signature`, `doc_comment`, `imports JSONB`, `calls JSONB`, `committed_at`, `valid_from`, `valid_to`  
- `rag.chunk_edges` ⇒ optional relations (`calls`, `imports`, `implements`, `tests`)  
- `rag.v_chunk_search` ⇒ view joining chunks ↔ files ↔ repos  

**Indexes**
- `GIN(content_tsv)` for lexical/BM25  
- `IVFFLAT` on `embedding` (cosine) for ANN  
- btree on symbol fields and GIN on JSONB `imports`/`calls`  

---

## Retrieval Strategy (Hybrid)

- **Vector**: cosine similarity on pgvector (query embedded via Ollama)  
- **Lexical**: `ts_rank_cd` over a `simple` + `unaccent` tsvector of code text  
- **Symbol boost**: direct hits on `symbol_name` or `symbol_signature`  
- **Ordering**: `hybrid_score = w_vec*vec + w_lex*lex + w_sym*sym_boost`  

Tuning knobs are exposed via query params (`w_vec`, `w_lex`, `w_sym`, `top_k`).

---

## Changing embedding dimension (if you use another model)

Current schema assumes **1024-dim** (`mxbai-embed-large`). To switch:

1. Choose a new Ollama embedding model (e.g., `nomic-embed-text` at 768-dim).  
2. Update `.env`:
   ```env
   EMB_MODEL=nomic-embed-text
   EMB_DIM=768
   ```
3. Adjust DB schema:
   ```sql
   -- Clear data if re-ingesting:
   TRUNCATE rag.code_chunks;
   DROP INDEX IF EXISTS code_chunks_embedding_ivfflat;
   ALTER TABLE rag.code_chunks ALTER COLUMN embedding TYPE vector(768);
   CREATE INDEX code_chunks_embedding_ivfflat
     ON rag.code_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
   ANALYZE rag.code_chunks;
   ```
4. Re-ingest repositories.  

If you want to preserve data, you must recompute embeddings to the new dimension.

---

## Production Hardening

- Put **auth** (API keys/JWT) in front of `/search` and `/ask/*`  
- Use **read-only DB roles** for the API  
- Enable **SSL** for Postgres (or run inside a trusted network)  
- Add **webhook/cron ingestion**, **idempotent staging**, **soft-deletes**, and **partitioning** for very large repos  
- Consider **HNSW** indexes if your pgvector build supports it (great online insert performance)  

---

## Troubleshooting

- **“dimension mismatch” / `cannot convert 1024-d to 768-d`**  
  Your embedding model dimension and `vector(N)` must match. See **Changing embedding dimension**.

- **Ollama healthcheck failing**  
  Ensure the `ollama` container has pulled the models:
  ```bash
  docker logs -f ollama
  docker exec -it ollama ollama list
  docker exec -it ollama ollama pull mxbai-embed-large
  ```

- **Slow queries after big ingests**  
  Re-ANALYZE:
  ```sql
  ANALYZE rag.code_chunks;
  ```
  Increase IVFFLAT `lists` (e.g., 100 → 200) and reindex.

- **No results for symbol queries**  
  Try including exact method/class names; symbol boosts rely on `symbol_name`/`symbol_signature`.

---

## License

MIT (suggested)

---

## Commands Reference

```bash
# Build app
docker compose build app

# Start DB + Ollama
docker compose up -d pg ollama
docker compose --profile init run --rm ollama-init

# Run migration
docker compose --profile init run --rm migrate

# Start API
docker compose up -d app

# Pull models into Ollama
docker exec -it ollama ollama pull mxbai-embed-large
docker exec -it ollama ollama pull llama3.1:8b

# Ingest repos
REPO_URLS="https://github.com/expressjs/express.git" docker compose --profile ingest up --exit-code-from ingest ingest

# Test search
curl "http://localhost:8000/search?q=jwt%20verification&language=java"

# Test Strands Agent
curl -X POST http://localhost:8000/ask/strands   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'

# Test LangGraph Agent
curl -X POST http://localhost:8000/ask/langgraph   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'
```
