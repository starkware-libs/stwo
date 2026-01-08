#![cfg_attr(
    feature = "simd",
    feature(exact_size_is_empty, raw_slice_split, portable_simd, array_chunks)
)]

#[cfg(feature = "simd")]
pub mod lookup_data;
#[cfg(feature = "simd")]
pub mod trace;
