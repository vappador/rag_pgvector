# Makefile for rag_pgvector
SHELL := /bin/bash
COMPOSE ?= docker compose

# Config (can also live in .env)
EMB_MODEL ?= mxbai-embed-large
LLM_MODEL ?= llama3.1:8b
REPO_URLS ?=
APP_PORT  ?= 8000
PG_PORT   ?= 5432

# Down flags (all optional, passed via CLI)
#   VOL=1           -> remove volumes (-v)
#   RMIMAGES=local  -> remove images built by compose (--rmi local)
#   RMIMAGES=all    -> remove all service images  (--rmi all)
#   ORPHANS=1       -> remove orphans (--remove-orphans)
VOL       ?= 0
RMIMAGES  ?= none
ORPHANS   ?= 0

# Compose down dynamic flags
DOWN_FLAGS :=
DOWN_FLAGS += $(if $(filter 1 true yes on,$(VOL)),-v,)
DOWN_FLAGS += $(if $(filter local all,$(RMIMAGES)),--rmi $(RMIMAGES),)
DOWN_FLAGS += $(if $(filter 1 true yes on,$(ORPHANS)),--remove-orphans,)

.PHONY: help \
	build up-core up down stop restart ps compose-config \
	migrate pull-models init \
	app-shell logs logs-app logs-pg logs-ollama \
	psql ingest agent \
	db-truncate db-reset truncate reset \
	health ollama-tags nuke

help:
	@echo "Targets:"
	@echo "  build            Build app/ingest images"
	@echo "  up-core          Start Postgres + Ollama only"
	@echo "  pull-models      Wait for Ollama, then pull EMB_MODEL/LLM_MODEL inside the ollama container"
	@echo "  migrate          One-shot: run DB migration"
	@echo "  init             up-core + migrate + pull-models"
	@echo "  up               Build & start API (depends on pg+ollama)"
	@echo "  down             Stop containers (safe by default). Flags: VOL=1 RMIMAGES=local|all ORPHANS=1"
	@echo "  nuke             Aggressive down (VOL=1 ORPHANS=1 RMIMAGES=local)"
	@echo "  stop             Stop containers (keep volumes)"
	@echo "  restart          Restart app"
	@echo "  psql             Open psql into Postgres"
	@echo "  ingest           Run one-shot ingestion (REPO_URLS=...)"
	@echo "  agent            Run local agent inside app container"
	@echo "  db-truncate      Truncate RAG data (keep schema)"
	@echo "  db-reset         Drop & recreate schema (runs migration)"
	@echo "  health           Check service health (pg, ollama)"
	@echo "  logs[-app|-pg|-ollama] Tail logs"
	@echo "  compose-config   Show effective compose config"
	@echo "  ollama-tags      List models in Ollama"

# ---------------------------------------------
# Build & core services
# ---------------------------------------------
build:
	$(COMPOSE) build app ingest

up-core:
	$(COMPOSE) up -d pg ollama

up: build up-core
	$(COMPOSE) up -d app
	@echo "API: http://localhost:$(APP_PORT)"

# SAFE by default: no volume/image deletion unless you pass flags.
down:
	$(COMPOSE) down $(DOWN_FLAGS)

# Handy alias for a thorough cleanup:
#   make nuke   (equivalent to: make down VOL=1 ORPHANS=1 RMIMAGES=local)
nuke:
	$(MAKE) down VOL=1 ORPHANS=1 RMIMAGES=local

stop:
	$(COMPOSE) stop

restart:
	$(COMPOSE) restart app

ps:
	$(COMPOSE) ps

compose-config:
	$(COMPOSE) config

# ---------------------------------------------
# One-shot init steps
# ---------------------------------------------
migrate: up-core
	$(COMPOSE) --profile init run --rm migrate

# Wait using ollama CLI (image always has it). Timeout + helpful logs on failure.
pull-models: up-core
	@echo "Waiting for Ollama daemon in container..."
	@TIMEOUT=120; i=0; \
	while ! $(COMPOSE) exec -T ollama ollama list >/dev/null 2>&1; do \
	  sleep 2; i=$$((i+2)); \
	  if [ $$i -ge $$TIMEOUT ]; then \
	    echo "Timed out waiting for Ollama. Recent logs:"; \
	    $(COMPOSE) logs --no-color --tail=200 ollama || true; \
	    exit 1; \
	  fi; \
	done; \
	echo "Ollama is ready."
	@echo "Pulling embedding model: $(EMB_MODEL)"; \
	$(COMPOSE) exec -T ollama ollama pull "$(EMB_MODEL)"
	@echo "Pulling chat model: $(LLM_MODEL)"; \
	$(COMPOSE) exec -T ollama ollama pull "$(LLM_MODEL)"
	@echo "Installed models:"; \
	$(COMPOSE) exec -T ollama ollama list

# Convenience: everything needed before 'up'
init: up-core migrate pull-models

# ---------------------------------------------
# Admin DB utilities (profile: admin)
# ---------------------------------------------
db-truncate:
	$(COMPOSE) --profile admin run --rm db-truncate

truncate: db-truncate

db-reset:
	$(COMPOSE) --profile admin run --rm db-reset

reset: db-reset

# ---------------------------------------------
# Dev tools
# ---------------------------------------------
psql:
	$(COMPOSE) exec pg psql -U rag -d ragdb

logs:
	$(COMPOSE) logs -f

logs-app:
	$(COMPOSE) logs -f app

logs-pg:
	$(COMPOSE) logs -f pg

logs-ollama:
	$(COMPOSE) logs -f ollama

app-shell:
	$(COMPOSE) exec app sh

ollama-tags:
	$(COMPOSE) exec -T ollama ollama list

health:
	@echo "Postgres health:" && $(COMPOSE) ps pg || true
	@echo
	@echo "Ollama version:" && $(COMPOSE) exec -T ollama ollama --version || true
	@echo
	@echo "Ollama models:" && $(COMPOSE) exec -T ollama ollama list || true

# ---------------------------------------------
# Workflows
# ---------------------------------------------
# Example:
#   make ingest REPO_URLS="https://github.com/expressjs/express.git,https://github.com/spring-projects/spring-petclinic.git"
ingest: build up-core
ifneq ($(strip $(REPO_URLS)),)
	REPO_URLS="$(REPO_URLS)" $(COMPOSE) --profile ingest up --build --exit-code-from ingest ingest
else
	@echo "ERROR: REPO_URLS is empty. Example usage:"; \
	echo 'make ingest REPO_URLS="https://github.com/expressjs/express.git,https://github.com/spring-projects/spring-petclinic.git"'; \
	exit 2
endif

agent:
	$(COMPOSE) exec app python /workspace/app/agent_graph.py
