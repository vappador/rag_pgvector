# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 TZ=UTC

# Basic tools (curl/jq)
RUN set -eux; \
  if command -v apt-get >/dev/null 2>&1; then \
    apt-get update && apt-get install -y --no-install-recommends curl jq && \
    rm -rf /var/lib/apt/lists/*; \
  elif command -v apk >/dev/null 2>&1; then \
    apk add --no-cache curl jq; \
  elif command -v microdnf >/dev/null 2>&1; then \
    microdnf -y update && microdnf -y install curl jq && microdnf clean all; \
  elif command -v dnf >/dev/null 2>&1; then \
    dnf -y install curl jq && dnf clean all; \
  elif command -v yum >/dev/null 2>&1; then \
    yum -y install curl jq && yum clean all; \
  else \
    echo "No supported package manager found to install curl/jq" >&2; exit 1; \
  fi

# Build deps for psycopg2 et al.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps
COPY app/requirements.txt /workspace/app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install -r /workspace/app/requirements.txt

# --- Build Tree-sitter bundle into NON-mounted path (/opt/ts) ---
# Clone grammars (as you already do) ...
RUN mkdir -p /opt/ts/vendor && \
    git clone --depth 1 https://github.com/tree-sitter/tree-sitter-java /opt/ts/vendor/tree-sitter-java && \
    git clone --depth 1 https://github.com/tree-sitter/tree-sitter-typescript /opt/ts/vendor/tree-sitter-typescript && \
    git clone --depth 1 https://github.com/tree-sitter/tree-sitter-javascript /opt/ts/vendor/tree-sitter-javascript && \
    git clone --depth 1 https://github.com/tree-sitter/tree-sitter-python /opt/ts/vendor/tree-sitter-python

# Write the Python builder script using a heredoc
RUN cat >/opt/ts/build_ts.py <<'PY'
from tree_sitter import Language
Language.build_library(
    '/opt/ts/my-languages.so',
    [
        '/opt/ts/vendor/tree-sitter-java',
        '/opt/ts/vendor/tree-sitter-typescript/typescript',
        '/opt/ts/vendor/tree-sitter-typescript/tsx',
        '/opt/ts/vendor/tree-sitter-javascript',
        '/opt/ts/vendor/tree-sitter-python',
    ],
)
print("Built /opt/ts/my-languages.so")
PY

# Run it, then clean up
RUN python /opt/ts/build_ts.py && rm -f /opt/ts/build_ts.py && ls -l /opt/ts

# App code
COPY . /workspace

# Add ingest entrypoint script
COPY scripts/ingest_entrypoint.sh /workspace/scripts/ingest_entrypoint.sh
RUN chmod +x /workspace/scripts/ingest_entrypoint.sh

# Non-root
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /workspace
USER appuser

# Serve unified API (Strands + LangGraph demos + graph endpoints)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

