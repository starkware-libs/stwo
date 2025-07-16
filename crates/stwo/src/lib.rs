#![allow(incomplete_features)]
#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![feature(
    array_chunks,
    array_try_from_fn,
    array_windows,
    exact_size_is_empty,
    int_roundings,
    iter_array_chunks,
    portable_simd,
    slice_ptr_get
)]
pub mod core;

#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "tracing")]
pub mod tracing;
