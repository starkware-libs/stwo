//! SIMD-backend bindings for the Keccak256 merkle path.
//!
//! `MerkleOpsLifted` lives in [`super::keccak256_lifted`] and uses SIMD-parallel
//! Keccak-f\[1600\] (8 sponges per call). This module supplies the trait scaffolding
//! (`ColumnOps<Keccak256Hash>`) needed for `BackendForChannel<Keccak256MerkleChannel>`.

use super::SimdBackend;
use crate::core::vcs::keccak256_hash::Keccak256Hash;
use crate::prover::backend::ColumnOps;

impl ColumnOps<Keccak256Hash> for SimdBackend {
    type Column = Vec<Keccak256Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}
