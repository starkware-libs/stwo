use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::poly::circle::CanonicCoset;
use super::poly::BitReversedOrder;
use super::ColumnVec;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

// TODO(AlonH): Add CommitmentScheme structs to contain multiple CommitmentTree instances.
pub struct CommitmentTreeProver {
    pub polynomials: ColumnVec<CPUCirclePoly<BaseField>>,
    pub evaluations: ColumnVec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MerkleTree<BaseField, Blake2sHasher>,
}

impl CommitmentTreeProver {
    pub fn new(
        polynomials: Vec<CPUCirclePoly<BaseField>>,
        log_blowup_factor: u32,
        channel: &mut Blake2sChannel,
    ) -> Self {
        let domains = polynomials
            .iter()
            .map(|poly| CanonicCoset::new(poly.log_size() + log_blowup_factor))
            .collect_vec();
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

        CommitmentTreeProver {
            polynomials,
            evaluations,
            commitment,
        }
    }

    // TODO(AlonH): change interface after integrating mixed degree merkle.
    pub fn decommit(
        &self,
        positions: Vec<usize>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        MerkleDecommitment<BaseField, Blake2sHasher>,
    ) {
        let values = self
            .evaluations
            .iter()
            .map(|c| positions.iter().map(|p| c[*p]).collect())
            .collect();
        let decommitment = self.commitment.generate_decommitment(positions);
        (values, decommitment)
    }
}

impl Deref for CommitmentTreeProver {
    type Target = MerkleTree<BaseField, Blake2sHasher>;

    fn deref(&self) -> &Self::Target {
        &self.commitment
    }
}

pub struct CommitmentTreeVerifier {
    pub commitment: Blake2sHash,
}

impl CommitmentTreeVerifier {
    pub fn new(commitment: Blake2sHash, channel: &mut Blake2sChannel) -> Self {
        channel.mix_digest(commitment);
        CommitmentTreeVerifier { commitment }
    }

    pub fn verify(
        &self,
        decommitment: &MerkleDecommitment<BaseField, Blake2sHasher>,
        positions: &[usize],
    ) -> bool {
        decommitment.verify(self.commitment, positions)
    }
}
