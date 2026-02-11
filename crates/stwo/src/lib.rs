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
pub use core::fields::{m31, qm31};

#[cfg(feature = "prover")]
pub mod prover;
#[cfg(feature = "tracing")]
pub mod tracing;

#[cfg(test)]
#[macro_export]
macro_rules! m31 {
    ($m:expr) => {
        $crate::core::fields::m31::M31::from_u32_unchecked($m)
    };
}

#[cfg(test)]
#[macro_export]
macro_rules! qm31 {
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        $crate::core::fields::qm31::QM31::from_u32_unchecked($a, $b, $c, $d)
    };
}
