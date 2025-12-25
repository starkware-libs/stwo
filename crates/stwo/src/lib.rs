#![allow(incomplete_features)]
#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    feature = "prover",
    feature(array_chunks, iter_array_chunks, portable_simd, slice_ptr_get)
)]
pub mod core;

#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "tracing")]
pub mod tracing;

#[cfg(not(target_env = "msvc"))] // jemalloc is not supported on MSVC targets
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;