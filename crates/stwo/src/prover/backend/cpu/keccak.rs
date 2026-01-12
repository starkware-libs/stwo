//! CPU implementation for Keccak256 backend operations

use itertools::Itertools;

use super::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::keccak_hash::KeccakHash;
use crate::core::vcs::keccak_merkle::KeccakMerkleHasher;
use crate::core::vcs::MerkleHasher;
use crate::prover::backend::{Col, Column};
use crate::prover::vcs::ops::MerkleOps;

// ColumnOps<KeccakHash> is automatically provided by the generic implementation in mod.rs

impl MerkleOps<KeccakMerkleHasher> for CpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<KeccakHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<KeccakHash> {
        // Simple sequential implementation
        (0..1 << log_size)
            .map(|i| {
                KeccakMerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.at(i)).collect_vec(),
                )
            })
            .collect()
    }
}