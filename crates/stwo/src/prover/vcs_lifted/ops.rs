use serde::{Deserialize, Serialize};

use crate::core::fields::m31::BaseField;
use crate::core::vcs::MerkleHasher;
use crate::prover::backend::{Col, ColumnOps};

/// Trait for performing Merkle operations on a commitment scheme.
pub trait MerkleOpsLifted<H: MerkleHasher>:
    ColumnOps<BaseField> + ColumnOps<H::Hash> + for<'de> Deserialize<'de> + Serialize
{
    /// Main changes: 1. no columns, 2. always a prev_layer.
    fn commit_on_layer(log_size: u32, prev_layer: &Col<Self, H::Hash>) -> Col<Self, H::Hash>;

    /// Here is the main logic of the lifted FRI.
    fn commit_on_first_layer(
        log_size: u32,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, H::Hash>;
}
