RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 \
    RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f -C opt-level=2" \
    cargo test test_avx_wide_fib_prove -- --nocapture
