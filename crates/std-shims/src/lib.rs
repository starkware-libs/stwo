#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
pub use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
#[cfg(feature = "std")]
pub use std::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    string::{String, ToString},
    vec,
    vec::Vec,
};
