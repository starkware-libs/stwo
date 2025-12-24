#!/bin/bash

# Instruct bash to immediately exit if any command in the pipeline fails, has a non-zero exit
# status, or there's a reference to an undefined variable.
set -eou pipefail

cargo +nightly-2025-07-14 clippy --workspace "$@" --all-targets --all-features -- -D warnings \
    -D future-incompatible -D nonstandard-style -D rust-2018-idioms -D unused

# Extract all crate names from the workspace metadata
crates=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort -u)

# Run clippy on each crate individually for catching issues that might not be detected when running
# clippy on the entire workspace
for crate in $crates; do
  echo "Clippy on crate: $crate"
  cargo +nightly-2025-07-14 clippy -p "$crate" --all-targets --all-features -- -D warnings \
    -D future-incompatible -D nonstandard-style -D rust-2018-idioms -D unused
done
