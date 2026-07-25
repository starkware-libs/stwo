#![allow(incomplete_features)]
// The SIMD/FFT kernels rely on entire `unsafe fn` bodies being unsafe by design.
// Rust 2024 makes `unsafe_op_in_unsafe_fn` deny-by-default; opt back into the
// pre-2024 behavior rather than wrapping every operation in an `unsafe {}` block.
#![allow(unsafe_op_in_unsafe_fn)]
#![cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), feature(stdarch_x86_avx512))]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "prover", feature(iter_array_chunks, portable_simd, slice_ptr_get))]
pub mod core;

// NitrooZK device-resident CUDA backend FFI module. Public so a downstream crate can `#include`
// the C++ headers / call the FFI bindings for its own circuit-specific GPU constraint kernel.
#[cfg(feature = "cuda")]
pub mod stwo_cuda;

#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "tracing")]
pub mod tracing;
