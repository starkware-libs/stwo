#![feature(
    array_chunks,
    iter_array_chunks,
    exact_size_is_empty,
    is_sorted,
    new_uninit,
    slice_group_by,
    stdsimd,
    get_many_mut,
    int_roundings,
    slice_flatten,
    assert_matches,
    portable_simd
)]
pub mod core;
pub mod examples;
pub mod hash_functions;
pub mod math;
pub mod trace_generation;
