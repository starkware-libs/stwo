#!/bin/bash
# Can be used as a drop in replacement for `cargo test` with avx512f flag on.
# For example, `./scripts/test_avx.sh` will run all tests(not only avx).
RUSTFLAGS="-Awarnings -C target-cpu=native -C target-feature=+avx512f -C opt-level=2" cargo +nightly-2025-01-02 test "$@"
