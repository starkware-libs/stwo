use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::prover::backend::CpuBackend;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl MerkleOpsLifted<Blake2sMerkleHasher> for CpuBackend {
    /// Receives the columns in ascending order!!!
    fn commit_on_first_layer(
        _log_size: u32,
        columns: &[&Vec<BaseField>],
    ) -> Vec<<Blake2sMerkleHasher as MerkleHasherLifted>::Hash> {
        let mut prev_layer: Vec<Blake2sMerkleHasher> = vec![Blake2sMerkleHasher::default()];
        for col in columns.iter() {
            prev_layer = col
                .iter()
                .enumerate()
                .map(|(idx, felt)| {
                    let mut hasher = prev_layer[idx % prev_layer.len()].clone();
                    hasher.update_leaf(*felt);
                    hasher
                })
                .collect();
        }
        prev_layer.into_iter().map(|x| x.finalize()).collect()
    }

    fn commit_on_layer(
        log_size: u32,
        prev_layer: &Vec<<Blake2sMerkleHasher as MerkleHasherLifted>::Hash>,
    ) -> Vec<<Blake2sMerkleHasher as MerkleHasherLifted>::Hash> {
        (0..(1 << log_size))
            .map(|i| {
                <Blake2sMerkleHasher as MerkleHasherLifted>::hash_children((
                    prev_layer[2 * i],
                    prev_layer[2 * i + 1],
                ))
            })
            .collect()
    }
}
