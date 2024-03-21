#![feature(
    array_chunks,
    exact_size_is_empty,
    get_many_mut,
    int_roundings,
    is_sorted,
    iter_array_chunks,
    new_uninit,
    stdarch_x86_avx512
)]
pub mod commitment_scheme;
pub mod core;
pub mod fibonacci;
pub mod hash_functions;
pub mod math;
pub mod platform;
