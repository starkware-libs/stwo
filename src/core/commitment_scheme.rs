use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::fields::ExtensionOf;
use super::poly::circle::CanonicCoset;
use super::poly::BitReversedOrder;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

pub struct CommitmentSchemeProver<F: ExtensionOf<BaseField>> {
    pub polynomials: Vec<CPUCirclePoly<F>>,
    pub evaluations: Vec<CPUCircleEvaluation<F, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MerkleTree<F, Blake2sHasher>,
}

impl<F: ExtensionOf<BaseField>> CommitmentSchemeProver<F> {
    pub fn new(
        polynomials: Vec<CPUCirclePoly<F>>,
        domains: Vec<CanonicCoset>,
        channel: &mut Blake2sChannel,
    ) -> Self {
        assert_eq!(polynomials.len(), domains.len(),);
        let evaluations = zip(&polynomials, domains)
            .map(|(poly, domain)| poly.evaluate(domain.circle_domain()))
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

pub struct CommitmentSchemeVerifier {
    pub commitment: Blake2sHash,
}

impl CommitmentSchemeVerifier {
    pub fn new(commitment: Blake2sHash, channel: &mut Blake2sChannel) -> Self {
        channel.mix_digest(commitment);
        CommitmentSchemeVerifier { commitment }
    }

    pub fn verify<F: ExtensionOf<BaseField>>(
        &self,
        decommitment: &MerkleDecommitment<F, Blake2sHasher>,
        positions: &[usize],
    ) -> bool {
        decommitment.verify(self.commitment, positions)
    }
}
