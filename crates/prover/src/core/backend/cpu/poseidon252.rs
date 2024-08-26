use itertools::Itertools;
use starknet_ff::FieldElement as FieldElement252;

use super::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher;

impl MerkleOps<Poseidon252MerkleHasher> for CpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<FieldElement252>>,
        columns: &[&Vec<BaseField>],
    ) -> Vec<FieldElement252> {
        (0..(1 << log_size))
            .map(|i| {
                Poseidon252MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column[i]).collect_vec(),
                )
            })
            .collect()
    }
}
