#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

if [[ $# -ne 3 ]]; then
  die "usage: test_gate.sh <env-file> <worktree> <log-file>"
fi

env_file="$1"
worktree="$2"
log_file="$3"

load_env "$env_file"
ensure_command cargo
mkdir -p "$(dirname "$log_file")"

tracked_files=()
untracked_files=()
while IFS= read -r line; do
  [[ -n "$line" ]] && tracked_files+=("$line")
done < <(git -C "$worktree" diff --name-only --relative HEAD)
while IFS= read -r line; do
  [[ -n "$line" ]] && untracked_files+=("$line")
done < <(git -C "$worktree" ls-files --others --exclude-standard)

declare -A selected_filters=()

add_filter() {
  selected_filters["$1"]=1
}

all_changed_files=()
if [[ ${#tracked_files[@]} -gt 0 ]]; then
  all_changed_files+=("${tracked_files[@]}")
fi
if [[ ${#untracked_files[@]} -gt 0 ]]; then
  all_changed_files+=("${untracked_files[@]}")
fi

for file in "${all_changed_files[@]}"; do
  case "$file" in
    crates/stwo/src/prover/backend/simd/bit_reverse.rs)
      add_filter "prover::backend::simd::bit_reverse::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/blake2s.rs)
      add_filter "prover::backend::simd::blake2s::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/blake2s_lifted.rs)
      add_filter "prover::backend::simd::blake2s_lifted::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/cm31.rs)
      add_filter "prover::backend::simd::cm31::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/column.rs)
      add_filter "prover::backend::simd::column::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/conversion.rs)
      add_filter "prover::backend::simd::conversion::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/domain.rs)
      add_filter "prover::backend::simd::domain::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/fft/*)
      add_filter "prover::backend::simd::fft::ifft::tests::"
      add_filter "prover::backend::simd::fft::rfft::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/m31.rs)
      add_filter "prover::backend::simd::m31::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/prefix_sum.rs)
      add_filter "prover::backend::simd::prefix_sum::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/qm31.rs)
      add_filter "prover::backend::simd::qm31::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/poseidon252_lifted.rs)
      add_filter "prover::backend::simd::poseidon252_lifted::tests::"
      ;;
    crates/stwo/src/prover/backend/simd/very_packed_m31.rs)
      add_filter "prover::backend::simd::m31::tests::"
      ;;
    crates/stwo/src/prover/mempool.rs)
      add_filter "prover::backend::simd::column::tests::"
      ;;
  esac
done

if [[ ${#selected_filters[@]} -eq 0 ]]; then
  add_filter "prover::backend::simd::bit_reverse::tests::"
  add_filter "prover::backend::simd::blake2s_lifted::tests::"
  add_filter "prover::backend::simd::cm31::tests::"
  add_filter "prover::backend::simd::column::tests::"
  add_filter "prover::backend::simd::conversion::tests::"
  add_filter "prover::backend::simd::domain::tests::"
  add_filter "prover::backend::simd::fft::ifft::tests::"
  add_filter "prover::backend::simd::fft::rfft::tests::"
  add_filter "prover::backend::simd::m31::tests::"
  add_filter "prover::backend::simd::prefix_sum::tests::"
  add_filter "prover::backend::simd::qm31::tests::"
  add_filter "prover::backend::simd::poseidon252_lifted::tests::"
fi

{
  printf 'Running targeted test gate\n'
  printf 'Worktree: %s\n' "$worktree"
} >"$log_file"

for filter in "${!selected_filters[@]}"; do
  {
    printf '\n==> %s\n' "$filter"
    if [[ "$AUTORESEARCH_USE_AVX_WRAPPER" == "1" ]]; then
      (
        cd "$worktree"
        "$REPO_ROOT/scripts/test_avx.sh" -p stwo --features prover "$filter" -- --nocapture
      )
    else
      (
        cd "$worktree"
        RUSTFLAGS="$AUTORESEARCH_RUSTFLAGS" cargo test -p stwo --features prover "$filter" -- --nocapture
      )
    fi
  } >>"$log_file" 2>&1
done

log "test gate passed"
