use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::backend::CPUBackend;
use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::fields::ExtensionOf;
use super::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use super::poly::BitReversedOrder;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

type B = CPUBackend;

pub struct CommitmentSchemeProver<F: ExtensionOf<BaseField>> {
    pub polynomials: Vec<CirclePoly<B, F>>,
    pub evaluations: Vec<CircleEvaluation<B, F, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MerkleTree<F, Blake2sHasher>,
}

impl<F: ExtensionOf<BaseField>> CommitmentSchemeProver<F> {
    pub fn new(
        polynomials: Vec<CirclePoly<B, F>>,
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
