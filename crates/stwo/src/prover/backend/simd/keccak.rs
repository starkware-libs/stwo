//! Simple stub implementation for Keccak256 backend operations
//! This provides basic functionality for testing without full SIMD optimization

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::keccak_hash::KeccakHash;
use crate::core::vcs::keccak_merkle::KeccakMerkleHasher;
use crate::core::vcs::MerkleHasher;
use crate::parallel_iter;
use crate::prover::backend::{Col, Column, ColumnOps};
use crate::prover::vcs::ops::MerkleOps;

impl ColumnOps<KeccakHash> for SimdBackend {
    type Column = Vec<KeccakHash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        // TODO: Implement bit reversal if needed
        unimplemented!("Keccak bit_reverse_column not implemented")
    }
}

impl MerkleOps<KeccakMerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<KeccakHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<KeccakHash> {
        // Simple sequential implementation - can be optimized later with SIMD
        parallel_iter!(0..1 << log_size)
            .map(|i| {
                KeccakMerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.at(i)).collect_vec(),
                )
            })
            .collect()
    }
}