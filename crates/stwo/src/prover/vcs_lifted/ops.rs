use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::prover::backend::{Col, ColumnOps};

/// Trait for performing Merkle operations on a commitment scheme.
pub trait MerkleOpsLifted<H: MerkleHasherLifted>:
    ColumnOps<BaseField> + ColumnOps<H::Hash> + for<'de> Deserialize<'de> + Serialize
{
    /// Computes the leaves of the lifted Merkle commitment.
    fn build_leaves(columns: &[&Col<Self, BaseField>]) -> Col<Self, H::Hash>;

    /// Given a layer of hashes as input, computes a new layer by hashing pairs
    /// of adjacent elements of the input, as in a standard Merkle tree.
    fn build_next_layer(prev_layer: &Col<Self, H::Hash>) -> Col<Self, H::Hash>;
}
