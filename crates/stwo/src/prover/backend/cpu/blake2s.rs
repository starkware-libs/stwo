use itertools::Itertools;

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasherGeneric;
use crate::core::vcs::MerkleHasher;
use crate::prover::backend::CpuBackend;
use crate::prover::vcs::ops::MerkleOps;

impl<const IS_M31_OUTPUT: bool> MerkleOps<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>
    for CpuBackend
{
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Blake2sHash>>,
        columns: &[&Vec<BaseField>],
    ) -> Vec<Blake2sHash> {
        (0..(1 << log_size))
            .map(|i| {
                Blake2sMerkleHasherGeneric::<IS_M31_OUTPUT>::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column[i]).collect_vec(),
                )
            })
            .collect()
    }
}
