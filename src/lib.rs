#![feature(
    stdsimd,
    new_uninit,
    is_sorted,
    array_chunks,
    slice_group_by,
    exact_size_is_empty
)]
pub mod commitment_scheme;
pub mod core;
pub mod fibonacci;
pub mod hash_functions;
pub mod math;
pub mod platform;
