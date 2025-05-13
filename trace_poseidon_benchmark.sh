LOG_N_INSTANCES=18  \
    RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo test trace_simd_poseidon_prove --features parallel,tracing -- --nocapture
