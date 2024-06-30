RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 \
    RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f -C opt-level=3" \
    cargo test test_simd_poseidon_prove -- --nocapture
