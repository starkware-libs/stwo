//! Naive SIMD-backend bindings for the Keccak256 channel/merkle path.
//!
//! These delegate to the CPU implementation (or to per-element scalar code) and
//! exist to satisfy the trait bounds of `BackendForChannel<Keccak256MerkleChannel>`
//! for `SimdBackend`. A future "Optimized" stage will replace these with parallel-
//! permutation (`keccak::parallel`, `f1600x4`/`x8`) implementations.

use itertools::Itertools;

use super::SimdBackend;
use crate::core::channel::{Channel, Keccak256Channel};
use crate::core::fields::m31::BaseField;
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::keccak256_hash::Keccak256Hash;
use crate::core::vcs::keccak256_merkle::Keccak256MerkleHasher;
use crate::core::vcs::MerkleHasher;
use crate::core::vcs_lifted::keccak256_merkle::Keccak256MerkleHasher as Keccak256MerkleHasherLifted;
use crate::prover::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::prover::vcs::ops::MerkleOps;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl ColumnOps<Keccak256Hash> for SimdBackend {
    type Column = Vec<Keccak256Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl GrindOps<Keccak256Channel> for SimdBackend {
    fn grind(channel: &Keccak256Channel, pow_bits: u32) -> u64 {
        let mut nonce = 0u64;
        loop {
            if channel.verify_pow_nonce(pow_bits, nonce) {
                return nonce;
            }
            nonce += 1;
        }
    }
}

// TODO: replace with an optimized SIMD implementation using parallel keccak permutations
// (e.g. `keccak::parallel` `f1600x4`/`x8`).
impl MerkleOps<Keccak256MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Keccak256Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Keccak256Hash> {
        (0..(1usize << log_size))
            .map(|i| {
                Keccak256MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.at(i)).collect_vec(),
                )
            })
            .collect()
    }
}

/// Naive `MerkleOpsLifted` for `SimdBackend`: copies columns to CPU and dispatches to the generic
/// `CpuBackend` lifted impl. Correctness-first; the optimized stage will replace this with a
/// parallel-permutation implementation.
impl MerkleOpsLifted<Keccak256MerkleHasherLifted> for SimdBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Keccak256Hash> {
        let cpu_cols = columns.iter().map(|column| column.to_cpu()).collect_vec();
        <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasherLifted>>::build_leaves(
            &cpu_cols.iter().collect_vec(),
            lifting_log_size,
        )
    }

    fn build_next_layer(prev_layer: &Vec<Keccak256Hash>) -> Vec<Keccak256Hash> {
        <CpuBackend as MerkleOpsLifted<Keccak256MerkleHasherLifted>>::build_next_layer(prev_layer)
    }
}
