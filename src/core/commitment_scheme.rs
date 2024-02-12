use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::fields::ExtensionOf;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use super::poly::BitReversedOrder;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

pub struct CommitmentScheme<F: ExtensionOf<BaseField>> {
    pub polynomials: Vec<CirclePoly<F>>,
    pub evaluations: Vec<CircleEvaluation<F, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MerkleTree<F, Blake2sHasher>,
}

impl<F: ExtensionOf<BaseField>> CommitmentScheme<F> {
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

        CommitmentScheme {
            polynomials,
            evaluations,
            commitment,
        }
    }
}

impl<F: ExtensionOf<BaseField>> Deref for CommitmentScheme<F> {
    type Target = MerkleTree<F, Blake2sHasher>;

    fn deref(&self) -> &Self::Target {
        &self.commitment
    }
}
