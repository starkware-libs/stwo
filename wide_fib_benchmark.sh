LOG_N_INSTANCES=23 RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info \
    RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo test test_wide_fib_prove_with_blake --features parallel -- --nocapture
