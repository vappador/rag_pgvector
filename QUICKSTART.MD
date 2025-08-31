# RAG + Agents Quickstart

This is a short guide to get **RAG with Postgres + pgvector** and both **LangGraph** and **Strands** agents running locally.

---

## Prerequisites

- Docker & Docker Compose installed  
- Ports available: `5432` (Postgres), `11434` (Ollama), `8000` (API)  

---

## Setup

### 1. Copy environment config

```bash
cp .env.example .env
```

Edit `.env` to adjust models or rate limits. Defaults:

```env
EMB_MODEL=mxbai-embed-large:latest
LLM_MODEL=llama3.1:8b
DB_DSN=postgresql://rag:ragpwd@pg:5432/ragdb
```

### 2. Start services

```bash
docker compose up -d pg ollama
docker compose --profile init up migrate
docker compose up -d app
```

### 3. Pull models into Ollama

```bash
docker exec -it ollama ollama pull mxbai-embed-large
docker exec -it ollama ollama pull llama3.1:8b
```

### 4. Ingest repositories

```bash
REPO_URLS="https://github.com/expressjs/express.git" docker compose --profile ingest up --exit-code-from ingest ingest
```

---

## Usage

### Test search

```bash
curl "http://localhost:8000/search?q=jwt%20verification&language=java"
```

### Ask via Strands Agent

```bash
curl -X POST http://localhost:8000/ask/strands   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'
```

### Ask via LangGraph Agent

```bash
curl -X POST http://localhost:8000/ask/langgraph   -H "Content-Type: application/json"   -d '{"question":"Where is JWT verification implemented?"}'
```

---

## Stopping services

```bash
docker compose down
```

---

That’s it — you now have a **RAG API** and two agent flavors (LangGraph + Strands) running locally!
