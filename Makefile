.PHONY: up down migrate ingest agent

up:
	docker compose up -d --build
	# wait for Postgres then run migration
	sleep 3 && docker compose exec pg psql -U rag -d ragdb -f /workspace/db/migrations/001_init.sql

down:
	docker compose down -v

migrate:
	docker compose exec pg psql -U rag -d ragdb -f /workspace/db/migrations/001_init.sql

ingest:
# pass repo URLs as env, comma-separated
# example: REPO_URLS="https://github.com/expressjs/express.git,https://github.com/spring-projects/spring-petclinic.git"
	docker compose exec -e REPO_URLS="$(REPO_URLS)" app python /workspace/app/ingest.py

agent:
	tdocker compose exec app python /workspace/app/agent_graph.py
