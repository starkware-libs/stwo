use std::collections::BTreeMap;
use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use super::channel::Blake2sChannel;
use super::fields::m31::BaseField;
use super::poly::circle::CanonicCoset;
use super::poly::BitReversedOrder;
use super::queries::SparseSubCircleDomain;
use super::ColumnVec;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

pub type OpenedValues = Vec<Vec<Vec<BaseField>>>;
pub type Decommitments = Vec<MerkleDecommitment<BaseField, Blake2sHasher>>;

pub struct CommitmentSchemeProver {
    pub trees: Vec<CommitmentTreeProver>,
    pub log_blowup_factor: u32,
}

impl CommitmentSchemeProver {
    pub fn new(log_blowup_factor: u32) -> Self {
        CommitmentSchemeProver {
            trees: Vec::new(),
            log_blowup_factor,
        }
    }

    pub fn commit(
        &mut self,
        polynomials: ColumnVec<CPUCirclePoly<BaseField>>,
        channel: &mut Blake2sChannel,
    ) {
        let tree = CommitmentTreeProver::new(polynomials, self.log_blowup_factor, channel);
        self.trees.push(tree);
    }

    pub fn roots(&self) -> Vec<Blake2sHash> {
        self.trees.iter().map(|tree| tree.root()).collect()
    }

    pub fn decommit(
        &self,
        positions: BTreeMap<u32, SparseSubCircleDomain>,
    ) -> (OpenedValues, Decommitments) {
        self.trees
            .iter()
            .map(|tree| {
                tree.decommit(
                    positions[&(tree.polynomials[0].log_size() + self.log_blowup_factor)].flatten(),
                )
            })
            .unzip()
    }
}

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
            .map(|(poly, domain)| poly.evaluate(domain.circle_domain()))
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

#[derive(Default)]
pub struct CommitmentSchemeVerifier {
    pub commitments: Vec<CommitmentTreeVerifier>,
}

impl CommitmentSchemeVerifier {
    pub fn new() -> Self {
        CommitmentSchemeVerifier {
            commitments: Vec::new(),
        }
    }

    pub fn commit(&mut self, commitment: Blake2sHash, channel: &mut Blake2sChannel) {
        let verifier = CommitmentTreeVerifier::new(commitment, channel);
        self.commitments.push(verifier);
    }

    pub fn verify(
        &self,
        decommitments: &[MerkleDecommitment<BaseField, Blake2sHasher>],
        positions: &[SparseSubCircleDomain],
    ) -> bool {
        self.commitments
            .iter()
            .zip(decommitments)
            .zip(positions)
            .all(|((commitment, decommitment), positions)| {
                commitment.verify(decommitment, &positions.flatten())
            })
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
