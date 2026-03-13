#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
load_env "${1:-$DEFAULT_ENV_FILE}"

ensure_command git
ensure_command python3
ensure_state_dirs

mkdir -p "$AUTORESEARCH_UPSTREAM_ROOT"

if [[ ! -d "$AUTORESEARCH_UPSTREAM_DIR/.git" ]]; then
  log "cloning upstream autoresearch into $AUTORESEARCH_UPSTREAM_DIR"
  git clone "$AUTORESEARCH_UPSTREAM_URL" "$AUTORESEARCH_UPSTREAM_DIR" >/dev/null
else
  log "updating upstream autoresearch checkout"
  git -C "$AUTORESEARCH_UPSTREAM_DIR" fetch --tags origin >/dev/null
fi

git -C "$AUTORESEARCH_UPSTREAM_DIR" checkout --quiet "$AUTORESEARCH_UPSTREAM_COMMIT"

platform="$(uname -s)"
machine="$(uname -m)"
sync_supported=0
sync_status="skipped"
note="source checkout only"

if [[ "$platform" == "Linux" ]] && command -v uv >/dev/null 2>&1; then
  sync_supported=1
  log "running uv sync inside upstream autoresearch checkout"
  if (cd "$AUTORESEARCH_UPSTREAM_DIR" && uv sync); then
    sync_status="ok"
    note="full upstream dependency sync completed"
  else
    sync_status="failed"
    note="uv sync failed"
  fi
else
  note="upstream pins torch==2.9.1+cu128 and cannot be fully synced on this host"
fi

status_file="$AUTORESEARCH_UPSTREAM_ROOT/install-status.json"
PLATFORM="$platform" \
MACHINE="$machine" \
SYNC_STATUS="$sync_status" \
SYNC_SUPPORTED="$sync_supported" \
NOTE="$note" \
python3 - "$status_file" <<'PY'
import json
import os
import sys

payload = {
    "commit": os.environ["AUTORESEARCH_UPSTREAM_COMMIT"],
    "machine": os.environ["MACHINE"],
    "platform": os.environ["PLATFORM"],
    "sync_status": os.environ["SYNC_STATUS"],
    "sync_supported": os.environ["SYNC_SUPPORTED"] == "1",
    "upstream_dir": os.environ["AUTORESEARCH_UPSTREAM_DIR"],
    "upstream_url": os.environ["AUTORESEARCH_UPSTREAM_URL"],
    "note": os.environ["NOTE"],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [[ "$sync_status" == "failed" ]]; then
  die "upstream autoresearch checkout succeeded but uv sync failed"
fi

log "upstream install metadata written to $status_file"
