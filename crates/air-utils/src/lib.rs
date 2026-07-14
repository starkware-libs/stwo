#![feature(exact_size_is_empty, raw_slice_split, portable_simd)]
// `unsafe fn` bodies here (e.g. `ComponentTrace::uninitialized`) are unsafe by
// design; opt back into pre-2024 behavior instead of wrapping each op in an
// `unsafe {}` block.
#![allow(unsafe_op_in_unsafe_fn)]
pub mod lookup_data;
pub mod trace;

/// Ensures a usable rayon global thread pool exists before running parallel code in tests.
///
/// On threadless wasm targets (e.g. `wasm32-wasip1`) rayon cannot spawn worker threads, so its
/// default lazily-built global pool aborts. Build a single-threaded global pool that runs work on
/// the calling thread instead. Idempotent and a no-op off wasm, where the default pool is fine.
#[cfg(test)]
pub(crate) fn ensure_rayon_pool() {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(1).use_current_thread().build_global();
    }
}
