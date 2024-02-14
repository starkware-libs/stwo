use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::ExtensionOf;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use super::poly::BitReversedOrder;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

pub struct CommitmentSchemeProver<F: ExtensionOf<BaseField>> {
    pub polynomials: Vec<CirclePoly<F>>,
    pub evaluations: Vec<CircleEvaluation<F, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MerkleTree<F, Blake2sHasher>,
}

impl<F: ExtensionOf<BaseField>> CommitmentSchemeProver<F> {
    pub fn new(
        polynomials: Vec<CirclePoly<F>>,
        domains: Vec<CanonicCoset>,
        channel: &mut Blake2sChannel,
    ) -> Self {
        assert_eq!(polynomials.len(), domains.len(),);
        let evaluations = zip(&polynomials, domains)
            .map(|(poly, domain)| poly.evaluate(domain.circle_domain()).bit_reverse())
            .collect_vec();
        let commitment = MerkleTree::<F, Blake2sHasher>::commit(
            evaluations
                .iter()
                .map(|eval| eval.values.clone())
                .collect_vec(),
        );
        channel.mix_digest(commitment.root());

        CommitmentSchemeProver {
            polynomials,
            evaluations,
            commitment,
        }
    }
}

impl<F: ExtensionOf<BaseField>> Deref for CommitmentSchemeProver<F> {
    type Target = MerkleTree<F, Blake2sHasher>;

    fn deref(&self) -> &Self::Target {
        &self.commitment
    }
}

pub struct Commitments(pub Vec<Blake2sHash>);

impl Deref for Commitments {
    type Target = Vec<Blake2sHash>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Decommitments {
    pub trace_decommitment: MerkleDecommitment<BaseField, Blake2sHasher>,
    // TODO(AlonH): Consider committing only on base field elements.
    pub composition_polynomial_decommitment: MerkleDecommitment<SecureField, Blake2sHasher>,
}
