#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

if [[ $# -ne 4 ]]; then
  die "usage: promotion_gate.sh <env-file> <worktree> <run-dir> <attempt-prefix>"
fi

env_file="$1"
worktree="$2"
run_dir="$3"
attempt_prefix="$4"

load_env "$env_file"
ensure_command python3

mkdir -p "$run_dir"

"$EXPERIMENT_ROOT/scripts/test_gate.sh" "$env_file" "$worktree" "$run_dir/${attempt_prefix}.promotion.tests.log"
"$EXPERIMENT_ROOT/scripts/benchmark_contract.sh" "$env_file" "$worktree" "$AUTORESEARCH_PROMOTION_PROFILE" "$run_dir/${attempt_prefix}.promotion.json"

python3 "$EXPERIMENT_ROOT/scripts/compare_metrics.py" \
  "$run_dir/best.promotion.json" \
  "$run_dir/${attempt_prefix}.promotion.json" \
  >"$run_dir/${attempt_prefix}.promotion.compare.json"

promotion_ratio=$(
  python3 - "$run_dir/${attempt_prefix}.promotion.compare.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["composite_ratio"])
PY
)

python3 - "$promotion_ratio" "$AUTORESEARCH_PROMOTION_THRESHOLD" <<'PY'
import sys

ratio = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if ratio <= threshold else 1)
PY

if [[ "$AUTORESEARCH_ENABLE_POSEIDON_PROMOTION" == "1" ]]; then
  poseidon_stdout="$run_dir/${attempt_prefix}.poseidon.raw.jsonl"
  poseidon_stderr="$run_dir/${attempt_prefix}.poseidon.stderr.log"
  (
    cd "$worktree"
    RUSTFLAGS="$AUTORESEARCH_RUSTFLAGS" \
      cargo criterion --message-format=json --plotting-backend disabled \
      -p stwo-examples --bench poseidon --features parallel \
      >"$poseidon_stdout" 2>"$poseidon_stderr"
  )
fi

log "promotion gate passed"

