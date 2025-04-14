use starknet_ff::FieldElement as FieldElement252;

use super::utils::transmute_col_refs;
use super::WebBackend;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::ops::MerkleOps;
use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher;

impl ColumnOps<FieldElement252> for WebBackend {
    type Column = Vec<FieldElement252>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Poseidon252MerkleHasher> for WebBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<FieldElement252>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<FieldElement252> {
        <SimdBackend as MerkleOps<Poseidon252MerkleHasher>>::commit_on_layer(
            log_size,
            prev_layer,
            transmute_col_refs(columns),
        )
    }
}
