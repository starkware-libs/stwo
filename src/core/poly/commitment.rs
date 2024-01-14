use std::collections::BTreeSet;
use std::ops::Deref;

use super::circle::CircleEvaluation;
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::commitment_scheme::utils::ColumnArray;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{ExtensionOf, IntoSlice};
use crate::core::utils::{bit_reverse_index, bit_reverse_vec};

/// Polynomial commitment interface. Used to internalize commitment logic (e.g. reordering of
/// evaluations)
pub struct PolynomialCommitment<F: ExtensionOf<BaseField>, H: Hasher> {
    merkle_tree: MerkleTree<F, H>,
    domain_log_size: u32,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> PolynomialCommitment<F, H> {
    pub fn commit(polynomials: Vec<&CircleEvaluation<F>>) -> Self
    where
        F: IntoSlice<H::NativeType>,
    {
        let bit_reversed_polynomials: ColumnArray<F> = polynomials
            .iter()
            .map(|p| bit_reverse_vec(&p.values, p.domain.log_size()))
            .collect();
        let merkle_tree = MerkleTree::<F, H>::commit(bit_reversed_polynomials);
        Self {
            merkle_tree,
            domain_log_size: polynomials[0].domain.log_size(),
        }
    }

    pub fn decommit(&self, queries: &BTreeSet<usize>) -> PolynomialDecommitment<F, H>
    where
        F: IntoSlice<H::NativeType>,
    {
        let bit_reversed_queries: BTreeSet<usize> = queries
            .iter()
            .map(|query| bit_reverse_index(*query as u32, self.domain_log_size) as usize)
            .collect();
        let merkle_decommitment = self.merkle_tree.generate_decommitment(bit_reversed_queries);
        PolynomialDecommitment {
            merkle_decommitment,
            domain_log_size: self.domain_log_size,
        }
    }
}

impl<F: ExtensionOf<BaseField>, H: Hasher> Deref for PolynomialCommitment<F, H> {
    type Target = MerkleTree<F, H>;

    fn deref(&self) -> &Self::Target {
        &self.merkle_tree
    }
}

pub struct PolynomialDecommitment<F: ExtensionOf<BaseField>, H: Hasher> {
    merkle_decommitment: MerkleDecommitment<F, H>,
    domain_log_size: u32,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> PolynomialDecommitment<F, H> {
    pub fn verify(&self, root: H::Hash, queries: &BTreeSet<usize>) -> bool
    where
        F: IntoSlice<H::NativeType>,
    {
        let bit_reversed_queries: BTreeSet<usize> = queries
            .iter()
            .map(|query| bit_reverse_index(*query as u32, self.domain_log_size) as usize)
            .collect();
        self.merkle_decommitment.verify(root, bit_reversed_queries)
    }
}
