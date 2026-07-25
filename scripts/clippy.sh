#!/bin/bash

# Instruct bash to immediately exit if any command in the pipeline fails, has a non-zero exit
# status, or there's a reference to an undefined variable.
set -eou pipefail

# Clippy is check-only and never links libstwo_cuda, so the nvcc/CMake native build in build.rs is
# unnecessary here; skip it so `--all-features` (which enables `cuda`) can lint the CUDA Rust on
# hosts without a CUDA toolchain. Override by exporting STWO_CUDA_SKIP_BUILD=0 before invoking.
export STWO_CUDA_SKIP_BUILD="${STWO_CUDA_SKIP_BUILD:-1}"

cargo +nightly-2026-01-15 clippy --workspace "$@" --all-targets --all-features -- -D warnings \
    -D future-incompatible -D nonstandard-style -D rust-2018-idioms -D unused

# Extract all crate names from the workspace metadata
crates=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort -u)

# Run clippy on each crate individually for catching issues that might not be detected when running
# clippy on the entire workspace
for crate in $crates; do
  echo "Clippy on crate: $crate"
  cargo +nightly-2026-01-15 clippy -p "$crate" --all-targets --all-features -- -D warnings \
    -D future-incompatible -D nonstandard-style -D rust-2018-idioms -D unused
done
