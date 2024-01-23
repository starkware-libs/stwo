use std::ops::Deref;

use super::circle::CircleEvaluation;
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
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

    pub fn decommit(&self, positions: &DecommitmentPositions) -> PolynomialDecommitment<F, H>
    where
        F: IntoSlice<H::NativeType>,
    {
        let merkle_decommitment = self
            .merkle_tree
            .generate_decommitment(positions.deref().clone());
        PolynomialDecommitment {
            merkle_decommitment,
        }
    }
}

impl<F: ExtensionOf<BaseField>, H: Hasher> Deref for PolynomialCommitmentScheme<F, H> {
    type Target = MerkleTree<F, H>;

    fn deref(&self) -> &Self::Target {
        &self.merkle_tree
    }
}

/// An ordered set of decommitment positions over a bit reversed `CircleDomain`.
pub struct DecommitmentPositions(pub Vec<usize>);

impl Deref for DecommitmentPositions {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct PolynomialDecommitment<F: ExtensionOf<BaseField>, H: Hasher> {
    merkle_decommitment: MerkleDecommitment<F, H>,
}

impl<F: ExtensionOf<BaseField>, H: Hasher> PolynomialDecommitment<F, H> {
    pub fn verify(&self, root: H::Hash, positions: &DecommitmentPositions) -> bool
    where
        F: IntoSlice<H::NativeType>,
    {
        self.merkle_decommitment.verify(root, positions)
    }
}

pub struct CommitmentProof<F: ExtensionOf<BaseField>, H: Hasher> {
    pub decommitment: PolynomialDecommitment<F, H>,
    pub commitment: H::Hash,
}

impl<F: ExtensionOf<BaseField> + IntoSlice<H::NativeType>, H: Hasher> CommitmentProof<F, H> {
    pub fn verify(&self, positions: &DecommitmentPositions) -> bool {
        self.decommitment.verify(self.commitment, positions)
    }
}

#[cfg(test)]
mod tests {
    use super::PolynomialCommitmentScheme;
    use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::commitment_scheme::utils::tests::generate_test_queries;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::M31;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::queries::Queries;
    use crate::m31;

    #[test]
    fn test_polynomial_commitment_scheme() {
        let log_size = 7;
        let size = 1 << log_size;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let values = (0..size).map(|x| m31!(x)).collect();
        let polynomial = CircleEvaluation::new(domain, values);
        let queries = Queries(generate_test_queries((size / 2) as usize, size as usize));
        let positions = queries
            .to_sparse_sub_circle_domain(1)
            .to_decommitment_positions();

        let commitment_scheme =
            PolynomialCommitmentScheme::<M31, Blake2sHasher>::commit(vec![&polynomial]);
        let decommitment = commitment_scheme.decommit(&positions);

        assert!(decommitment.verify(commitment_scheme.root(), &positions));
    }

    #[test]
    pub fn test_decommitment_positions() {
        let channel = &mut Blake2sChannel::new(Blake2sHash::default());
        let log_domain_size = 31;
        let n_queries = 100;
        let fri_step_size = 3;

        let queries = Queries::generate(channel, log_domain_size, n_queries);
        let queries_with_added_positions = queries
            .to_sparse_sub_circle_domain(fri_step_size)
            .to_decommitment_positions();

        assert!(queries_with_added_positions.is_sorted());
        assert_eq!(
            queries_with_added_positions.len(),
            n_queries * (1 << fri_step_size)
        );
    }

    #[test]
    pub fn test_dedup_decommitment_positions() {
        let log_domain_size = 7;

        // Generate all possible queries.
        let queries = Queries((0..1 << log_domain_size).collect());
        let queries_with_conjugates = queries
            .to_sparse_sub_circle_domain(log_domain_size - 2)
            .to_decommitment_positions();

        assert_eq!(queries.0, queries_with_conjugates.0);
    }
}
