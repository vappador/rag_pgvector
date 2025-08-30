# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 TZ=UTC

# System deps (git for GitPython, TLS certs)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps (cache wheel/metadata)
COPY app/requirements.txt /workspace/app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install -r /workspace/app/requirements.txt

# Copy the rest
COPY . /workspace

# (optional) non-root
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /workspace
USER appuser

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
