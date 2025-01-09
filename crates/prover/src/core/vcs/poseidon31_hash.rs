use std::fmt;

use serde::{Deserialize, Serialize};

use crate::core::fields::m31::M31;
use crate::core::vcs::hash::Hash;

// Wrapper for the Poseidon31 hash type.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Default, Eq, Deserialize, Serialize)]
pub struct Poseidon31Hash(pub [M31; 8]);

impl fmt::Display for Poseidon31Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Debug>::fmt(self, f)
    }
}

impl Hash for Poseidon31Hash {}
