#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
env_file="${1:-$DEFAULT_ENV_FILE}"
load_env "$env_file"

ensure_command git
ensure_state_dirs

mkdir -p "$AUTORESEARCH_RUN_DIR"

if [[ ! -e "$AUTORESEARCH_WORKTREE_PATH/.git" ]]; then
  log "creating worktree $AUTORESEARCH_WORKTREE_PATH on branch autoresearch/$AUTORESEARCH_RUN_TAG"
  git -C "$REPO_ROOT" rev-parse --verify "$AUTORESEARCH_BASE_BRANCH" >/dev/null 2>&1 \
    || die "base branch does not exist locally: $AUTORESEARCH_BASE_BRANCH"
  git -C "$REPO_ROOT" worktree add -B "autoresearch/$AUTORESEARCH_RUN_TAG" "$AUTORESEARCH_WORKTREE_PATH" "$AUTORESEARCH_BASE_BRANCH" >/dev/null
fi

if ! git_is_clean "$AUTORESEARCH_WORKTREE_PATH"; then
  die "worktree is not clean: $AUTORESEARCH_WORKTREE_PATH"
fi

results_tsv="$AUTORESEARCH_RUN_DIR/results.tsv"
if [[ ! -f "$results_tsv" ]]; then
  printf 'timestamp\tbase_commit\tcandidate_commit\tprofile\tcomposite_ratio\timprovement_pct\tstatus\tdescription\n' >"$results_tsv"
fi

if [[ ! -f "$AUTORESEARCH_RUN_DIR/run.json" ]]; then
  python3 - "$AUTORESEARCH_RUN_DIR/run.json" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

starting_commit = subprocess.run(
    ["git", "-C", os.environ["AUTORESEARCH_WORKTREE_PATH"], "rev-parse", "--short", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

payload = {
    "base_branch": os.environ["AUTORESEARCH_BASE_BRANCH"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "repo_root": os.environ["REPO_ROOT"],
    "run_tag": os.environ["AUTORESEARCH_RUN_TAG"],
    "starting_commit": starting_commit,
    "worktree": os.environ["AUTORESEARCH_WORKTREE_PATH"],
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
fi

if [[ ! -f "$AUTORESEARCH_RUN_DIR/baseline.fast.json" ]]; then
  "$EXPERIMENT_ROOT/scripts/benchmark_contract.sh" "$env_file" "$AUTORESEARCH_WORKTREE_PATH" "$AUTORESEARCH_FAST_PROFILE" "$AUTORESEARCH_RUN_DIR/baseline.fast.json"
  cp "$AUTORESEARCH_RUN_DIR/baseline.fast.json" "$AUTORESEARCH_RUN_DIR/best.fast.json"
fi

if [[ "$AUTORESEARCH_ENABLE_PROMOTION" == "1" ]] && [[ ! -f "$AUTORESEARCH_RUN_DIR/baseline.promotion.json" ]]; then
  "$EXPERIMENT_ROOT/scripts/benchmark_contract.sh" "$env_file" "$AUTORESEARCH_WORKTREE_PATH" "$AUTORESEARCH_PROMOTION_PROFILE" "$AUTORESEARCH_RUN_DIR/baseline.promotion.json"
  cp "$AUTORESEARCH_RUN_DIR/baseline.promotion.json" "$AUTORESEARCH_RUN_DIR/best.promotion.json"
fi

touch "$AUTORESEARCH_RUN_DIR/results.jsonl"
log "run ready: $AUTORESEARCH_RUN_DIR"
