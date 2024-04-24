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
    assert_matches
)]
pub mod core;
pub mod examples;
pub mod hash_functions;
pub mod math;
pub mod platform;

#[cfg(test)]
pub use core::fields::cm31::cm31;
#[cfg(test)]
pub use core::fields::m31::m31;
#[cfg(test)]
pub use core::fields::qm31::qm31;
