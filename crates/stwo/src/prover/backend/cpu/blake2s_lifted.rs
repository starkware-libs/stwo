use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::parallel_iter;
use crate::prover::backend::CpuBackend;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl<H: MerkleHasherLifted> MerkleOpsLifted<H> for CpuBackend {
    /// Computes the leaves of the Merkle tree. This is the core logic of the lifted Merkle
    /// commitment. The input columns are assumed to be in increasing order of length.
    ///
    /// The columns are interpreted as evaluations of polynomials in bit reversed order.
    /// For example, consider a polynomial that on the canonical circle domain of size 8 has
    /// evaluations (in natural order and bit reversed respectively):
    ///     a   a
    ///     b   e
    ///     c   c
    ///     d   g
    ///     e   b
    ///     f   f
    ///     g   d
    ///     h   h
    /// Then the evaluations of its lifted polynomial on the canonical circle domain of size 16 are
    /// (in natural and bit reversed order respectively):
    ///     a   a
    ///     b   e
    ///     c   a
    ///     d   e
    ///     a   c
    ///     b   g
    ///     c   c
    ///     d   g
    ///     e   b
    ///     f   f
    ///     g   b
    ///     h   f
    ///     e   d
    ///     f   h
    ///     g   d
    ///     h   h
    fn build_leaves(columns: &[&Vec<BaseField>]) -> Vec<H::Hash> {
        let hasher = H::default_with_initial_state();
        if columns.is_empty() {
            return vec![hasher.finalize()];
        }
        if columns[0].len() == 1 {
            panic!("A column must be of length >= 2.")
        }
        let mut prev_layer: Vec<H> = vec![hasher; 2];
        let mut prev_layer_log_size: u32 = 1;
        for (log_size, group) in columns.iter().group_by(|c| c.len().ilog2()).into_iter() {
            let log_ratio = log_size - prev_layer_log_size;
            prev_layer = (0..1 << log_size)
                // We only clone when starting a column chunk of different size.
                .map(|idx| prev_layer[(idx >> (log_ratio + 1) << 1) + (idx & 1)].clone())
                .collect();

            // We chunk by 16 because it's the amount of M31 elements needed to trigger a
            // hash permutation, both in blake and in poseidon.
            for chunk in &group.into_iter().chunks(16) {
                let vec = chunk.into_iter().collect_vec();
                prev_layer.iter_mut().enumerate().for_each(|(i, hasher)| {
                    hasher.update_leaf(&vec.iter().map(|v| v[i]).collect_vec());
                })
            }
            prev_layer_log_size = log_size;
        }
        prev_layer.into_iter().map(|x| x.finalize()).collect()
    }

    fn build_next_layer(prev_layer: &Vec<H::Hash>) -> Vec<H::Hash> {
        let log_size: u32 = prev_layer.len().ilog2() - 1;
        parallel_iter!(0..(1 << log_size))
            .map(|i| H::hash_children((prev_layer[2 * i], prev_layer[2 * i + 1])))
            .collect()
    }
}
