use serde::{Deserialize, Serialize};
use std_shims::Vec;

use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Default)]
pub struct MerkleDecommitmentLifted<H: MerkleHasherLifted> {
    /// Hash values that the verifier needs but cannot deduce from previous computations, in the
    /// order they are needed.
    pub hash_witness: Vec<H::Hash>,
}
