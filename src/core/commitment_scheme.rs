use std::iter::zip;

use itertools::Itertools;

use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use super::poly::BitReversedOrder;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

pub struct TraceCommitmentScheme {
    pub polynomials: Vec<CirclePoly<BaseField>>,
    pub evaluations: Vec<CircleEvaluation<BaseField, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle.
    pub commitment: MerkleTree<BaseField, Blake2sHasher>,
}

impl TraceCommitmentScheme {
    pub fn new(
        polynomials: Vec<CirclePoly<BaseField>>,
        domains: Vec<CanonicCoset>,
        channel: &mut Blake2sChannel,
    ) -> Self {
        assert_eq!(polynomials.len(), domains.len(),);
        let evaluations = zip(&polynomials, domains)
            .map(|(poly, domain)| poly.evaluate(domain.circle_domain()).bit_reverse())
            .collect_vec();
        let commitment = MerkleTree::<BaseField, Blake2sHasher>::commit(
            evaluations
                .iter()
                .map(|eval| eval.values.clone())
                .collect_vec(),
        );
        channel.mix_digest(commitment.root());

        TraceCommitmentScheme {
            polynomials,
            evaluations,
            commitment,
        }
    }
}
