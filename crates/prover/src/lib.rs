#![allow(incomplete_features)]
#![feature(
    array_chunks,
    array_try_from_fn,
    array_windows,
    assert_matches,
    exact_size_is_empty,
    generic_const_exprs,
    get_many_mut,
    int_roundings,
    iter_array_chunks,
    portable_simd,
    slice_ptr_get,
    stdarch_x86_avx512,
    trait_upcasting,
)]
pub mod constraint_framework;
pub mod core;
pub mod examples;
pub mod math;
