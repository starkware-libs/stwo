#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
env_file="${1:-$DEFAULT_ENV_FILE}"
load_env "$env_file"

"$EXPERIMENT_ROOT/scripts/create_run.sh" "$env_file"

stop_file="$AUTORESEARCH_RUN_DIR/STOP"
iteration=0

while true; do
  if [[ -f "$stop_file" ]]; then
    log "stop file detected: $stop_file"
    exit 0
  fi

  iteration=$((iteration + 1))
  log "starting iteration $iteration"
  "$EXPERIMENT_ROOT/scripts/run_iteration.sh" "$env_file"

  if [[ "$AUTORESEARCH_AGENT_MAX_ITERATIONS" != "0" ]] && [[ "$iteration" -ge "$AUTORESEARCH_AGENT_MAX_ITERATIONS" ]]; then
    log "reached AUTORESEARCH_AGENT_MAX_ITERATIONS=$AUTORESEARCH_AGENT_MAX_ITERATIONS"
    exit 0
  fi
done
