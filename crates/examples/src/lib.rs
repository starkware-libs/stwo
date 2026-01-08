#![cfg_attr(feature = "simd", feature(portable_simd, iter_array_chunks, array_chunks))]

#[cfg(feature = "simd")]
pub mod blake;
#[cfg(feature = "simd")]
pub mod plonk;
#[cfg(feature = "simd")]
pub mod poseidon;
#[cfg(feature = "simd")]
pub mod state_machine;
#[cfg(feature = "simd")]
pub mod wide_fibonacci;
#[cfg(feature = "simd")]
pub mod xor;
