use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::prover::backend::CpuBackend;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl<H: MerkleHasherLifted> MerkleOpsLifted<H> for CpuBackend {
    /// TODO(Leo): document. Assumption on order of cols.
    fn commit_on_first_layer(columns: &[&Vec<BaseField>]) -> Vec<H::Hash> {
        let hasher = H::default_with_prefix();
        let mut prev_layer: Vec<H> = vec![hasher];
        for col in columns.iter() {
            prev_layer = col
                .iter()
                .enumerate()
                .map(|(idx, felt)| {
                    let mut hasher = prev_layer[idx % prev_layer.len()].clone();
                    hasher.update_leaves(&[*felt]);
                    hasher
                })
                .collect();
        }
        prev_layer.into_iter().map(|x| x.finalize()).collect()
    }

    fn commit_on_inner_layer(prev_layer: &Vec<H::Hash>) -> Vec<H::Hash> {
        assert!(prev_layer.len().is_power_of_two());
        let log_size = prev_layer.len().ilog2() as usize - 1;
        (0..(1 << log_size))
            .map(|i| H::hash_children((prev_layer[2 * i], prev_layer[2 * i + 1])))
            .collect()
    }
}
