#![allow(incomplete_features)]
#![feature(
    array_methods,
    array_chunks,
    assert_matches,
    exact_size_is_empty,
    generic_const_exprs,
    get_many_mut,
    int_roundings,
    is_sorted,
    iter_array_chunks,
    new_uninit,
    portable_simd,
    slice_first_last_chunk,
    slice_flatten,
    slice_group_by,
    stdsimd
)]
pub mod constraint_framework;
pub mod core;
pub mod examples;
pub mod math;
// TODO: Add back once InteractionElements and LookupValues get refactored out. InteractionElements
// removed in favour of storing interaction elements the components directly with LookupElements.
// LookupValues removed in favour of storing lookup values on a claim struct.
// pub mod trace_generation;
