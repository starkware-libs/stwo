#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

if [[ $# -ne 2 ]]; then
  die "usage: guard_paths.sh <env-file> <worktree>"
fi

env_file="$1"
worktree="$2"

load_env "$env_file"
ensure_command git
ensure_command python3

tracked_files=()
untracked_files=()

while IFS= read -r line; do
  [[ -n "$line" ]] && tracked_files+=("$line")
done < <(git -C "$worktree" diff --name-only --relative HEAD)

while IFS= read -r line; do
  [[ -n "$line" ]] && untracked_files+=("$line")
done < <(git -C "$worktree" ls-files --others --exclude-standard)

changed_files=()
if [[ ${#tracked_files[@]} -gt 0 ]]; then
  changed_files+=("${tracked_files[@]}")
fi
if [[ ${#untracked_files[@]} -gt 0 ]]; then
  changed_files+=("${untracked_files[@]}")
fi

if [[ ${#changed_files[@]} -eq 0 ]]; then
  log "guard_paths: no changed files"
  exit 0
fi

python3 - "$AUTORESEARCH_POLICY_FILE" "${changed_files[@]}" <<'PY'
import json
import sys

policy = json.load(open(sys.argv[1], encoding="utf-8"))
changed_files = sys.argv[2:]

allowed = policy["allowed_edit_paths"]
forbidden = policy["forbidden_edit_paths"]

def matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix)

errors = []
for path in changed_files:
    if any(matches(path, prefix) for prefix in forbidden):
        errors.append(f"forbidden path touched: {path}")
        continue
    if not any(matches(path, prefix) for prefix in allowed):
        errors.append(f"path outside allowlist: {path}")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)
PY

combined_diff=""
if [[ ${#tracked_files[@]} -gt 0 ]]; then
  combined_diff+=$(git -C "$worktree" diff --no-ext-diff --unified=0 HEAD -- "${tracked_files[@]}" || true)
fi

for file in "${untracked_files[@]}"; do
  combined_diff+=$'\n'
  combined_diff+=$(git -C "$worktree" diff --no-ext-diff --no-index --unified=0 /dev/null "$worktree/$file" || true)
done

if printf '%s\n' "$combined_diff" | grep -E '^\+[^+].*\bunsafe\b' >/dev/null; then
  die "guard_paths: diff introduces or edits unsafe"
fi

log "guard_paths: allowlist checks passed"
