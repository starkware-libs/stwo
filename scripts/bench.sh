#!/bin/bash

RUSTFLAGS="-Awarnings -C target-cpu=native -C target-feature=+avx512f -C opt-level=2" cargo bench "$@"
