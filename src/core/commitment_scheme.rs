use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::poly::circle::CanonicCoset;
use super::poly::BitReversedOrder;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

pub struct CommitmentSchemeProver {
    pub polynomials: Vec<CPUCirclePoly<BaseField>>,
    pub evaluations: Vec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MerkleTree<BaseField, Blake2sHasher>,
}

impl CommitmentSchemeProver {
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

        CommitmentSchemeProver {
            polynomials,
            evaluations,
            commitment,
        }
    }

    pub fn secure_field_evaluation(&self) -> CPUCircleEvaluation<SecureField, BitReversedOrder> {
        assert_eq!(self.evaluations.len(), 4);
        let domain = self.evaluations[0].domain;
        let mut values = Vec::new();
        for index in 0..domain.size() {
            values.push(SecureField::from_m31_array(std::array::from_fn(|i| {
                self.evaluations[i].values[index]
            })));
        }
        CPUCircleEvaluation::new(domain, values)
    }
}

impl Deref for CommitmentSchemeProver {
    type Target = MerkleTree<BaseField, Blake2sHasher>;

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

    pub fn verify(
        &self,
        decommitment: &MerkleDecommitment<BaseField, Blake2sHasher>,
        positions: &[usize],
    ) -> bool {
        decommitment.verify(self.commitment, positions)
    }
}
