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

# App code
COPY . /workspace

# Non-root
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /workspace
USER appuser

# Serve unified API (Strands + LangGraph demos + graph endpoints)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
