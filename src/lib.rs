#![feature(
    array_chunks,
    exact_size_is_empty,
    is_sorted,
    new_uninit,
    slice_group_by,
    stdsimd,
    get_many_mut,
    option_get_or_insert_default
)]
pub mod commitment_scheme;
pub mod core;
pub mod fibonacci;
pub mod hash_functions;
pub mod math;
pub mod platform;
