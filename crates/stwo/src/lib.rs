#![allow(incomplete_features)]
#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    feature = "prover",
    feature(iter_array_chunks, portable_simd, slice_ptr_get)
)]
pub mod core;

// NitrooZK device-resident CUDA backend FFI module. Public so a downstream crate can `#include`
// the C++ headers / call the FFI bindings for its own circuit-specific GPU constraint kernel.
#[cfg(feature = "cuda")]
pub mod stwo_cuda;

#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "tracing")]
pub mod tracing;
