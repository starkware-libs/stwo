use itertools::Itertools;

use crate::core::backend::{Col, CpuBackend};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::vcs::poseidon31_hash::Poseidon31Hash;
use crate::core::vcs::poseidon31_merkle::Poseidon31MerkleHasher;

impl MerkleOps<Poseidon31MerkleHasher> for CpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Poseidon31Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Poseidon31Hash> {
        (0..(1 << log_size))
            .map(|i| {
                Poseidon31MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column[i]).collect_vec(),
                )
            })
            .collect()
    }
}
