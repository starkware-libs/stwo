#![allow(incomplete_features)]
#![cfg_attr(
    all(feature = "simd", target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    feature = "simd",
    feature(array_chunks, iter_array_chunks, portable_simd, slice_ptr_get)
)]
pub mod core;

#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "tracing")]
pub mod tracing;
