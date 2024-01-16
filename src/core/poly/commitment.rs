use std::ops::Deref;

use super::circle::CircleEvaluation;
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::commitment_scheme::utils::ColumnArray;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, IntoSlice};
use crate::core::utils::bit_reverse_vec;

/// Polynomial commitment interface. Used to internalize commitment logic (e.g. reordering of
/// evaluations)
pub struct PolynomialCommitmentScheme<F: ExtensionOf<BaseField>, H: Hasher> {
    merkle_tree: MerkleTree<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> PolynomialCommitmentScheme<F, H> {
    pub fn commit(polynomials: Vec<&CircleEvaluation<F>>) -> Self
    where
        F: IntoSlice<H::NativeType>,
    {
        let bit_reversed_polynomials: ColumnArray<F> = polynomials
            .iter()
            .map(|p| bit_reverse_vec(&p.values, p.domain.log_size()))
            .collect();
        let merkle_tree = MerkleTree::<F, H>::commit(bit_reversed_polynomials);
        Self { merkle_tree }
    }
}

impl<F: ExtensionOf<BaseField>, H: Hasher> Deref for PolynomialCommitmentScheme<F, H> {
    type Target = MerkleTree<F, H>;

    fn deref(&self) -> &Self::Target {
        &self.merkle_tree
    }
}
