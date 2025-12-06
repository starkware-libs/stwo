use itertools::Itertools;

use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::keccak_hash::Keccak256Hash;
use crate::core::vcs::keccak_merkle::{Keccak256M31MerkleHasher, Keccak256MerkleHasher};
use crate::core::vcs::MerkleHasher;
use crate::parallel_iter;
use crate::prover::backend::{Col, Column, ColumnOps};
use crate::prover::vcs::ops::MerkleOps;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl ColumnOps<Keccak256Hash> for SimdBackend {
    type Column = Vec<Keccak256Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Keccak256MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Keccak256Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Keccak256Hash> {
        let n_elements = 1 << log_size;
        
        // Parallel iterator if feature enabled, otherwise sequential
        parallel_iter!(0..n_elements)
            .map(|i| {
                // Scalar implementation iterating over what would be SIMD lanes
                Keccak256MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.at(i)).collect_vec(),
                )
            })
            .collect()
    }
}

impl MerkleOps<Keccak256M31MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Keccak256Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Keccak256Hash> {
        // Reuse the non-M31 implementation since the difference is handled by the hasher
        let n_elements = 1 << log_size;
        
        parallel_iter!(0..n_elements)
            .map(|i| {
                Keccak256M31MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.at(i)).collect_vec(),
                )
            })
            .collect()
    }
}

