#!/bin/bash
# Can be used as a drop in replacement for `cargo bench`.
# For example, `./scripts/bench.sh` will run all benchmarks.
# or `./scripts/bench.sh M31` will run only the M31 benchmarks.
RUSTFLAGS="-Awarnings -C target-cpu=native -C opt-level=3" cargo criterion --output-format bencher $@
