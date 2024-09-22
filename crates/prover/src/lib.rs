#![allow(incomplete_features)]
#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]
#![feature(
    array_chunks,
    array_try_from_fn,
    array_windows,
    assert_matches,
    exact_size_is_empty,
    get_many_mut,
    int_roundings,
    iter_array_chunks,
    portable_simd,
    slice_ptr_get,
    trait_upcasting
)]
pub mod constraint_framework;
pub mod core;
pub mod examples;
pub mod math;
