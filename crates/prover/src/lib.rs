#![allow(incomplete_features)]
#![cfg_attr(not(feature = "std"), no_std)]
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

// Re-export common alloc types and macros for no_std
#[cfg(not(feature = "std"))]
extern crate alloc;

mod prelude {
    #[cfg(not(feature = "std"))]
    pub use alloc::{
        borrow::ToOwned, boxed::Box, format, string::String, string::ToString, vec, vec::Vec,
    };
    #[cfg(feature = "std")]
    pub use std::{
        borrow::ToOwned, boxed::Box, format, string::String, string::ToString, vec, vec::Vec,
    };
}

pub mod collections {
    #[cfg(not(feature = "std"))]
    pub use alloc::collections::btree_map;
    #[cfg(not(feature = "std"))]
    pub use alloc::collections::btree_map::BTreeMap;
    #[cfg(not(feature = "std"))]
    pub use alloc::collections::btree_set;
    #[cfg(not(feature = "std"))]
    pub use alloc::collections::btree_set::BTreeSet;
    #[cfg(feature = "std")]
    pub use std::collections::*;

    #[cfg(not(feature = "std"))]
    pub use hashbrown::hash_map;
    #[cfg(not(feature = "std"))]
    pub use hashbrown::hash_map::HashMap;
    #[cfg(not(feature = "std"))]
    pub use hashbrown::hash_set;
    #[cfg(not(feature = "std"))]
    pub use hashbrown::hash_set::HashSet;
}

pub use prelude::*;

pub mod constraint_framework;
pub mod core;
pub mod examples;
pub mod math;
