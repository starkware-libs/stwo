use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
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
        for col in columns.iter() {
            // TODO(Leo): the clone in the map can be avoided when `prev_layer`
            // has the same size of `col`. It can also be avoided by not using
            // hashers and manipulating the underlying hash state directly, as
            // is done in the SIMD implementation.
            let curr_layer_log_size = col.len().ilog2();
            let shift = curr_layer_log_size - prev_layer_log_size;
            prev_layer = col
                .iter()
                .enumerate()
                .map(|(idx, felt)| {
                    let mut hasher = prev_layer[(idx >> (shift + 1) << 1) + (idx & 1)].clone();
                    hasher.update_leaf(&[*felt]);
                    hasher
                })
                .collect();
            prev_layer_log_size = curr_layer_log_size;
        }
        prev_layer.into_iter().map(|x| x.finalize()).collect()
    }

    fn build_next_layer(prev_layer: &Vec<H::Hash>) -> Vec<H::Hash> {
        let log_size = prev_layer.len().ilog2() as usize - 1;
        (0..(1 << log_size))
            .map(|i| H::hash_children((prev_layer[2 * i], prev_layer[2 * i + 1])))
            .collect()
    }
}
