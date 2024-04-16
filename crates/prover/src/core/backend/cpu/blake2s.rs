use itertools::Itertools;

use crate::commitment_scheme::blake2_hash::Blake2sHash;
use crate::commitment_scheme::blake2_merkle::Blake2sMerkleHasher;
use crate::commitment_scheme::ops::{MerkleHasher, MerkleOps};
use crate::core::backend::CPUBackend;
use crate::core::fields::m31::BaseField;

impl MerkleOps<Blake2sMerkleHasher> for CPUBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Vec<BaseField>],
    ) -> Vec<Blake2sHash> {
        (0..(1 << log_size))
            .map(|i| {
                Blake2sMerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column[i]).collect_vec(),
                )
            })
            .collect()
    }
}
