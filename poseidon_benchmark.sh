LOG_N_INSTANCES=18 RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info \
    cargo test --release test_simd_poseidon_prove -- --nocapture
