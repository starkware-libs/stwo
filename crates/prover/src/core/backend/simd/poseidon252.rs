use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use starknet_ff::FieldElement as FieldElement252;

use super::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::m31::BaseField;
#[cfg(not(target_arch = "wasm32"))]
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::ops::MerkleOps;
use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher;
use crate::parallel_iter;

impl ColumnOps<FieldElement252> for SimdBackend {
    type Column = Vec<FieldElement252>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Poseidon252MerkleHasher> for SimdBackend {
    // TODO(ShaharS): replace with SIMD implementation.
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<FieldElement252>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<FieldElement252> {
        let iter = parallel_iter!(0..(1 << log_size));
        iter.map(|i| {
            Poseidon252MerkleHasher::hash_node(
                prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                &columns.iter().map(|column| column.at(i)).collect_vec(),
            )
        })
        .collect()
    }
}
