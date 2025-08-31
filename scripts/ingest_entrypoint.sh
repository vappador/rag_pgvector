#!/usr/bin/env bash
set -euo pipefail

echo ">>> ingest_entrypoint.sh starting"
: "${DB_DSN:?DB_DSN env var required}"
: "${EMB_MODEL:=mxbai-embed-large:latest}"
: "${EMB_DIM:=1024}"
: "${REPO_URLS:=}"

if [[ -z "${REPO_URLS}" ]]; then
  echo "ERROR: REPO_URLS is empty. Provide a comma/newline/space-separated list of git URLs."
  exit 2
fi

# Optional knobs
: "${BASE_REF:=}"
: "${CHANGED_ONLY:=}"
: "${FOLLOW_SYMLINKS:=}"
: "${MAX_FILE_SIZE:=}"

# Normalize REPO_URLS into lines
mapfile -t repos < <(printf "%s\n" "${REPO_URLS}" | tr ', ' '\n' | sed '/^\s*$/d')

echo "Found ${#repos[@]} repo URL(s)"
mkdir -p /workspace/repos

for url in "${repos[@]}"; do
  url="${url//$'\r'/}"   # strip CR if any
  url="$(echo "$url" | xargs)"  # trim
  [[ -z "$url" ]] && continue

  name="$(basename "${url%%.git}" )"
  repo_dir="/workspace/repos/${name}"

  echo ">>> Preparing repo: $name ($url)"
  if [[ -d "$repo_dir/.git" ]]; then
    echo "Repo exists. Fetching updates..."
    git -C "$repo_dir" remote set-url origin "$url" || true
    git -C "$repo_dir" fetch --all --prune || true
  else
    echo "Cloning $url -> $repo_dir"
    git clone --depth=1 "$url" "$repo_dir"
  fi

  # Build argument list for ingest.py
  args=(
    --repo-dir "$repo_dir"
    --repo-name "$name"
    --repo-url "$url"
    --dsn "$DB_DSN"
    --embed
    --embed-model "$EMB_MODEL"
  )

  if [[ -n "${FOLLOW_SYMLINKS}" ]]; then
    args+=(--follow-symlinks)
  fi
  if [[ -n "${MAX_FILE_SIZE}" ]]; then
    args+=(--max-file-size "${MAX_FILE_SIZE}")
  fi
  if [[ -n "${CHANGED_ONLY}" ]]; then
    args+=(--changed-only)
  fi
  if [[ -n "${BASE_REF}" ]]; then
    args+=(--base-ref "${BASE_REF}")
  fi

  echo ">>> Running ingest.py for $name"
  python /workspace/app/ingest.py "${args[@]}"
  echo ">>> Ingest completed for $name"
done

echo ">>> All ingestions complete."
