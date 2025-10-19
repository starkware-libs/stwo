use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::MerkleHasher;
use crate::prover::backend::CpuBackend;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl MerkleOpsLifted<Blake2sMerkleHasher> for CpuBackend {
    /// Receives the columns in ascending order!!!
    fn commit_on_first_layer(_log_size: u32, columns: &[&Vec<BaseField>]) -> Vec<Blake2sHash> {
        let mut prev_layer: Vec<Blake2sHasher> = vec![Blake2sHasher::default()];
        for col in columns.iter() {
            prev_layer = col
                .iter()
                .enumerate()
                .map(|(idx, felt)| {
                    let mut hasher = prev_layer[idx % prev_layer.len()].clone();
                    hasher.update(felt.0.to_le_bytes().as_slice());
                    hasher
                })
                .collect();
        }
        prev_layer.into_iter().map(|x| x.finalize()).collect()
    }

    fn commit_on_layer(log_size: u32, prev_layer: &Vec<Blake2sHash>) -> Vec<Blake2sHash> {
        (0..(1 << log_size))
            .map(|i| {
                // TODO: hash_node will probably be deleted in lifted vcs.
                Blake2sMerkleHasher::hash_node(
                    Some((prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &[],
                )
            })
            .collect()
    }
}
