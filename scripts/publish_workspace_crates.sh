#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Publish workspace crates in dependency order.

Usage:
  scripts/publish_workspace_crates.sh --order-only
  scripts/publish_workspace_crates.sh --dry-run
  scripts/publish_workspace_crates.sh --skip-first 3
  scripts/publish_workspace_crates.sh -- --registry crates-io
EOF
}

dry_run=0
order_only=0
skip_first=0
publish_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --order-only)
      order_only=1
      shift
      ;;
    --skip-first|-s)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --skip-first requires a numeric argument." >&2
        exit 1
      fi
      skip_first="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      publish_args+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}/.."

# Hard-coded publish order. Keep in sync with workspace crates.
# NOTE: crates/std-shims is intentionally versioned separately (not part of the
# workspace version), so it is excluded from the publish list.
CRATES_TO_PUBLISH=(
  stwo
  stwo-constraint-framework
  stwo-air-utils-derive
  stwo-air-utils
)

echo "Publish order:"
for name in "${CRATES_TO_PUBLISH[@]}"; do
  echo "- ${name}"
done

if [[ "${order_only}" -eq 1 ]]; then
  exit 0
fi

for name in "${CRATES_TO_PUBLISH[@]:${skip_first}}"; do
  echo ""
  echo "==> Publishing ${name}"
  cmd=(cargo publish -p "${name}")
  if [[ "${dry_run}" -eq 1 ]]; then
    cmd+=(--dry-run)
  fi
  if [[ "${#publish_args[@]}" -gt 0 ]]; then
    cmd+=("${publish_args[@]}")
  fi
  (cd "${WORKSPACE_ROOT}" && "${cmd[@]}")
done
