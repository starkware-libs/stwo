#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
env_file="${1:-$DEFAULT_ENV_FILE}"
load_env "$env_file"

ensure_command git
ensure_command python3

"$EXPERIMENT_ROOT/scripts/create_run.sh" "$env_file"

if ! git_is_clean "$AUTORESEARCH_WORKTREE_PATH"; then
  die "worktree must be clean before starting a new iteration"
fi

attempt_id="$(date -u +%Y%m%dT%H%M%SZ)"
attempt_prefix="$AUTORESEARCH_RUN_DIR/$attempt_id"
mkdir -p "$AUTORESEARCH_RUN_DIR"

base_commit="$(git_current_commit "$AUTORESEARCH_WORKTREE_PATH")"

best_summary=$(
  python3 - "$AUTORESEARCH_RUN_DIR/best.fast.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(f'best fast-profile commit: {payload["commit"]}')
print(f'benchmark count: {payload["benchmark_count"]}')
print(f'geomean_estimate_ns: {payload["geomean_estimate_ns"]:.3f}')
PY
)

recent_results="$(tail -n 6 "$AUTORESEARCH_RUN_DIR/results.tsv" 2>/dev/null || true)"

prompt_file="$attempt_prefix.prompt.md"
{
  printf '# Iteration Context\n\n'
  printf 'You are working inside a dedicated experimental git worktree rooted at `%s`.\n\n' "$AUTORESEARCH_WORKTREE_PATH"
  printf 'Base commit for this iteration: `%s`\n\n' "$base_commit"
  printf '## Current best summary\n%s\n\n' "$best_summary"
  printf '## Recent result rows\n```\n%s\n```\n\n' "$recent_results"
  printf '## Required behavior\n'
  printf -- '- Make exactly one coherent performance experiment.\n'
  printf -- '- Stay inside the allowlist in `config/policy.json`.\n'
  printf -- '- Do not run git state management commands.\n'
  printf -- '- Do not benchmark; the outer loop handles that.\n'
  printf -- '- Return JSON matching `AGENT_RESPONSE.schema.json`.\n\n'
  printf '## Program\n\n'
  cat "$AUTORESEARCH_PROGRAM_FILE"
  printf '\n'
} >"$prompt_file"

agent_response_file="$attempt_prefix.agent-response.json"

append_jsonl_record() {
  local status="$1"
  local description="$2"
  local comparison_file="${3:-}"
  python3 - "$AUTORESEARCH_RUN_DIR/results.jsonl" "$status" "$description" "$agent_response_file" "$comparison_file" "$attempt_id" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

results_path = Path(sys.argv[1])
status = sys.argv[2]
description = sys.argv[3]
agent_response_path = Path(sys.argv[4])
comparison_path = sys.argv[5]
attempt_id = sys.argv[6]

payload = {
    "attempt_completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "attempt_id": attempt_id,
    "description": description,
    "status": status,
}
if agent_response_path.exists():
    payload["agent_response"] = json.loads(agent_response_path.read_text(encoding="utf-8"))
if comparison_path:
    payload["comparison"] = json.loads(Path(comparison_path).read_text(encoding="utf-8"))

with results_path.open("a", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
}

run_agent() {
  case "$AUTORESEARCH_AGENT" in
    codex)
      command=(codex exec --cd "$AUTORESEARCH_WORKTREE_PATH" --full-auto --sandbox "$AUTORESEARCH_AGENT_SANDBOX" --output-schema "$AUTORESEARCH_SCHEMA_FILE" -o "$agent_response_file")
      if [[ -n "$AUTORESEARCH_AGENT_MODEL" ]]; then
        command+=(-m "$AUTORESEARCH_AGENT_MODEL")
      fi
      "${command[@]}" - <"$prompt_file"
      ;;
    claude)
      schema_inline="$(tr -d '\n' <"$AUTORESEARCH_SCHEMA_FILE")"
      command=(claude -p --permission-mode auto --output-format json --json-schema "$schema_inline")
      if [[ -n "$AUTORESEARCH_AGENT_MODEL" ]]; then
        command+=(--model "$AUTORESEARCH_AGENT_MODEL")
      fi
      "${command[@]}" "$(cat "$prompt_file")" >"$agent_response_file"
      ;;
    noop)
      cat >"$agent_response_file" <<'EOF'
{"description":"noop agent run","hypothesis":"no code changes requested","risk_level":"low","blocked":true,"blocked_reason":"AUTORESEARCH_AGENT=noop"}
EOF
      ;;
    *)
      die "unsupported AUTORESEARCH_AGENT value: $AUTORESEARCH_AGENT"
      ;;
  esac
}

run_agent

if git_is_clean "$AUTORESEARCH_WORKTREE_PATH"; then
  timestamp="$attempt_id"
  printf '%s\t%s\t-\tfast\t1.000000\t0.000000\tnoop\tno code changes\n' "$timestamp" "$base_commit" >>"$AUTORESEARCH_RUN_DIR/results.tsv"
  append_jsonl_record "noop" "no code changes"
  log "iteration produced no diff"
  exit 0
fi

if ! "$EXPERIMENT_ROOT/scripts/guard_paths.sh" "$env_file" "$AUTORESEARCH_WORKTREE_PATH" >"$attempt_prefix.guard.log" 2>&1; then
  reset_worktree "$AUTORESEARCH_WORKTREE_PATH"
  printf '%s\t%s\t-\tfast\t1.000000\t0.000000\tblocked\tpath or unsafe guard failed\n' "$attempt_id" "$base_commit" >>"$AUTORESEARCH_RUN_DIR/results.tsv"
  append_jsonl_record "blocked" "path or unsafe guard failed"
  log "guard rejected the diff"
  exit 0
fi

if ! "$EXPERIMENT_ROOT/scripts/test_gate.sh" "$env_file" "$AUTORESEARCH_WORKTREE_PATH" "$attempt_prefix.tests.log"; then
  reset_worktree "$AUTORESEARCH_WORKTREE_PATH"
  printf '%s\t%s\t-\tfast\t1.000000\t0.000000\tgate_fail\ttargeted tests failed\n' "$attempt_id" "$base_commit" >>"$AUTORESEARCH_RUN_DIR/results.tsv"
  append_jsonl_record "gate_fail" "targeted tests failed"
  log "test gate failed"
  exit 0
fi

"$EXPERIMENT_ROOT/scripts/benchmark_contract.sh" "$env_file" "$AUTORESEARCH_WORKTREE_PATH" "$AUTORESEARCH_FAST_PROFILE" "$attempt_prefix.fast.json"
python3 "$EXPERIMENT_ROOT/scripts/compare_metrics.py" \
  "$AUTORESEARCH_RUN_DIR/best.fast.json" \
  "$attempt_prefix.fast.json" \
  >"$attempt_prefix.fast.compare.json"

fast_ratio=$(
  python3 - "$attempt_prefix.fast.compare.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["composite_ratio"])
PY
)

keep_fast=1
if ! python3 - "$fast_ratio" "$AUTORESEARCH_FAST_THRESHOLD" <<'PY'
import sys

ratio = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if ratio <= threshold else 1)
PY
then
  keep_fast=0
fi

if [[ "$keep_fast" != "1" ]]; then
  improvement_pct=$(
    python3 - "$attempt_prefix.fast.compare.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["improvement_pct"])
PY
  )
  reset_worktree "$AUTORESEARCH_WORKTREE_PATH"
  printf '%s\t%s\t-\tfast\t%s\t%s\tdiscard\tfast profile did not improve enough\n' \
    "$attempt_id" "$base_commit" "$fast_ratio" "$improvement_pct" >>"$AUTORESEARCH_RUN_DIR/results.tsv"
  append_jsonl_record "discard" "fast profile did not improve enough" "$attempt_prefix.fast.compare.json"
  log "candidate discarded after fast-profile comparison"
  exit 0
fi

if [[ "$AUTORESEARCH_ENABLE_PROMOTION" == "1" ]]; then
  if ! "$EXPERIMENT_ROOT/scripts/promotion_gate.sh" "$env_file" "$AUTORESEARCH_WORKTREE_PATH" "$AUTORESEARCH_RUN_DIR" "$attempt_id"; then
    reset_worktree "$AUTORESEARCH_WORKTREE_PATH"
    printf '%s\t%s\t-\tpromotion\t1.000000\t0.000000\tgate_fail\tpromotion gate failed\n' "$attempt_id" "$base_commit" >>"$AUTORESEARCH_RUN_DIR/results.tsv"
    append_jsonl_record "gate_fail" "promotion gate failed"
    log "promotion gate failed"
    exit 0
  fi
fi

description=$(
  python3 - "$agent_response_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["description"].replace("\n", " ").strip())
PY
)
hypothesis=$(
  python3 - "$agent_response_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["hypothesis"].replace("\n", " ").strip())
PY
)

git -C "$AUTORESEARCH_WORKTREE_PATH" add -A
git -C "$AUTORESEARCH_WORKTREE_PATH" commit -m "autoresearch: $description" -m "$hypothesis" >/dev/null

candidate_commit="$(git_current_commit "$AUTORESEARCH_WORKTREE_PATH")"
cp "$attempt_prefix.fast.json" "$AUTORESEARCH_RUN_DIR/best.fast.json"
if [[ "$AUTORESEARCH_ENABLE_PROMOTION" == "1" ]]; then
  cp "$AUTORESEARCH_RUN_DIR/${attempt_id}.promotion.json" "$AUTORESEARCH_RUN_DIR/best.promotion.json"
fi

improvement_pct=$(
  python3 - "$attempt_prefix.fast.compare.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["improvement_pct"])
PY
)

printf '%s\t%s\t%s\tfast\t%s\t%s\tkeep\t%s\n' \
  "$attempt_id" "$base_commit" "$candidate_commit" "$fast_ratio" "$improvement_pct" "$description" \
  >>"$AUTORESEARCH_RUN_DIR/results.tsv"

append_jsonl_record "keep" "$description" "$attempt_prefix.fast.compare.json"

log "accepted experiment at commit $candidate_commit"
