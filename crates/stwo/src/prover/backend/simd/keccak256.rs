//! Naive SIMD-backend bindings for the Keccak256 merkle path.
//!
//! These delegate to the CPU implementation (or to per-element scalar code) and
//! exist to satisfy the trait bounds of `BackendForChannel<Keccak256MerkleChannel>`
//! for `SimdBackend`. A future "Optimized" stage will replace these with parallel-
//! permutation (`keccak::parallel`, `f1600x4`/`x8`) implementations.

use itertools::Itertools;

use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::keccak256_hash::Keccak256Hash;
use crate::core::vcs_lifted::keccak256_merkle::Keccak256MerkleHasher;
use crate::prover::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl ColumnOps<Keccak256Hash> for SimdBackend {
    type Column = Vec<Keccak256Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

/// Naive `MerkleOpsLifted` for `SimdBackend`: copies columns to CPU and dispatches to the generic
/// `CpuBackend` lifted impl. Correctness-first; the optimized stage will replace this with a
/// parallel-permutation implementation.
impl MerkleOpsLifted<Keccak256MerkleHasher> for SimdBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Keccak256Hash> {
        let cpu_cols = columns.iter().map(|column| column.to_cpu()).collect_vec();
        <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_leaves(
            &cpu_cols.iter().collect_vec(),
            lifting_log_size,
        )
    }

    fn build_next_layer(prev_layer: &Vec<Keccak256Hash>) -> Vec<Keccak256Hash> {
        <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasher>>::build_next_layer(prev_layer)
    }
}
