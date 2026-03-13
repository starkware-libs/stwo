#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

if [[ $# -ne 4 ]]; then
  die "usage: benchmark_contract.sh <env-file> <worktree> <profile.json> <output.json>"
fi

env_file="$1"
worktree="$2"
profile_json="$3"
output_json="$4"

load_env "$env_file"
ensure_command cargo
ensure_command python3

mkdir -p "$(dirname "$output_json")"

profile_data=()
while IFS= read -r line; do
  profile_data+=("$line")
done < <(python3 - "$profile_json" <<'PY'
import json
import sys

profile = json.load(open(sys.argv[1], encoding="utf-8"))
print(profile["name"])
print(profile["package"])
print(profile["features"])
for bench in profile["criterion_benches"]:
    print(bench)
PY
)

profile_name="${profile_data[0]}"
package_name="${profile_data[1]}"
features="${profile_data[2]}"
benches=("${profile_data[@]:3}")

raw_jsonl="${output_json%.json}.raw.jsonl"
stderr_log="${output_json%.json}.stderr.log"

log "running benchmark profile $profile_name in $worktree"
: >"$raw_jsonl"
: >"$stderr_log"

for bench in "${benches[@]}"; do
  log "benchmark target: $bench"
  (
    cd "$worktree"
    RUSTFLAGS="$AUTORESEARCH_RUSTFLAGS" \
      cargo criterion --message-format=json --plotting-backend disabled \
      -p "$package_name" --features "$features" --bench "$bench" \
      >>"$raw_jsonl" 2>>"$stderr_log"
  )
done

python3 "$EXPERIMENT_ROOT/scripts/parse_criterion_json.py" "$raw_jsonl" "$profile_name" "$worktree" >"$output_json"
log "benchmark metrics written to $output_json"
