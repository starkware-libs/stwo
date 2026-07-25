//! Soundness-NEUTRAL prove_ex sub-phase instrumentation (timers only).
//!
//! Enabled at RUN time via `PROVE_EX_TIMERS=1` — a runtime env check, NOT a compile-time feature,
//! so the timers are available on the production-fast binary (the numbers reflect production).
//! Emits `[prove_ex] <name> <secs>s` lines. Does NOT change any prover logic or proof output
//! (byte-identity is guarded by the proof fingerprint); each read is ~free when off.
//!
//! CUDA kernel launches are async on stream 0; reading the clock without a device sync would
//! mis-attribute a phase's GPU time to whatever later forces a sync. [`prove_ex_sync`] calls the
//! CUDA-runtime `cudaDeviceSynchronize` (linked via `cargo:rustc-link-lib=cudart`) so each timer
//! read reflects real completed GPU work. On non-cuda builds it is a no-op.
//!
//! This module carries the timer helpers the `prove_ex` / PCS hook sites call; it exists as its own
//! module (rather than inline in `prover::mod`) so the observation code is separated from the prove
//! logic. The `diag` crate feature is declared for parity with downstream diagnostic tooling, but
//! the timers themselves stay runtime-env (they compile in every `prover` build).

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaDeviceSynchronize() -> i32;
}

/// Force a full device sync (real GPU completion) before reading a timer. No-op without cuda.
// Not const: under the `cuda` feature the body calls the CUDA FFI, which is not const.
#[allow(clippy::missing_const_for_fn)]
#[inline]
pub fn prove_ex_sync() {
    #[cfg(feature = "cuda")]
    // SAFETY: FFI to the CUDA runtime; blocks until all device work completes. No state change.
    unsafe {
        let _ = cudaDeviceSynchronize();
    }
}

/// Whether prove_ex sub-phase timers are enabled (`PROVE_EX_TIMERS=1`).
#[inline]
pub fn prove_ex_timers_on() -> bool {
    std::env::var("PROVE_EX_TIMERS").is_ok()
}

/// Emit one sub-phase timer line iff timers are enabled. Syncs the device first (so a GPU phase's
/// time is attributed correctly), then prints `[prove_ex] <name> <elapsed>s`. `enabled` is the
/// once-per-prove `prove_ex_timers_on()` read the caller already holds (avoids a re-read per
/// phase).
#[inline]
pub fn phase(enabled: bool, name: &str, elapsed: std::time::Duration) {
    if enabled {
        prove_ex_sync();
        eprintln!("[prove_ex] {name} {:.3}s", elapsed.as_secs_f64());
    }
}
