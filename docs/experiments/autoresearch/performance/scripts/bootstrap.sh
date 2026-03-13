#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
env_file="${1:-$DEFAULT_ENV_FILE}"
load_env "$env_file"

ensure_command git
ensure_command cargo
ensure_command python3
ensure_state_dirs

"$EXPERIMENT_ROOT/scripts/install_upstream_autoresearch.sh" "$env_file"

if ! cargo criterion --version >/dev/null 2>&1; then
  log "installing cargo-criterion"
  cargo install cargo-criterion
fi

case "$AUTORESEARCH_AGENT" in
  codex)
    ensure_command codex
    ;;
  claude)
    ensure_command claude
    ;;
  noop)
    ;;
  *)
    die "unsupported AUTORESEARCH_AGENT value: $AUTORESEARCH_AGENT"
    ;;
esac

log "bootstrap complete"
log "repo root: $REPO_ROOT"
log "worktree root: $AUTORESEARCH_WORKTREE_ROOT"
log "run root: $AUTORESEARCH_RUNS_ROOT"
log "agent: $AUTORESEARCH_AGENT"

