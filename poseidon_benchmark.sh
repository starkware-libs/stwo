LOG_N_INSTANCES=18 RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info \
    RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo test test_simd_poseidon_prove --features parallel -- --nocapture
