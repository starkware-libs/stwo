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
use crate::commitment_scheme::merkle_input::{MerkleTreeInput, MerkleTreeStructure};
use crate::commitment_scheme::mixed_degree_decommitment::MixedDecommitment;
use crate::commitment_scheme::mixed_degree_merkle_tree::MixedDegreeMerkleTree;
use crate::core::channel::Channel;

pub mod utils;

/// Holds a vector for each tree, which holds a vector for each column, which holds its respective
/// opened values.
#[derive(Debug)]
pub struct OpenedValues(pub Vec<ColumnVec<Vec<BaseField>>>);

impl Deref for OpenedValues {
    type Target = Vec<Vec<Vec<BaseField>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct Decommitments(Vec<MixedDecommitment<BaseField, Blake2sHasher>>);

impl Deref for Decommitments {
    type Target = Vec<MixedDecommitment<BaseField, Blake2sHasher>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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

    pub fn commit(&mut self, polynomials: ColumnVec<CPUCirclePoly>, channel: &mut Blake2sChannel) {
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
        let (values, decommitments) = self
            .trees
            .iter()
            .map(|tree| {
                tree.decommit(
                    positions[&(tree.polynomials[0].log_size() + self.log_blowup_factor)].flatten(),
                )
            })
            .unzip();
        (OpenedValues(values), Decommitments(decommitments))
    }
}

pub struct CommitmentTreeProver {
    pub polynomials: ColumnVec<CPUCirclePoly>,
    pub evaluations: ColumnVec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    pub commitment: MixedDegreeMerkleTree<BaseField, Blake2sHasher>,
    pub structure: MerkleTreeStructure,
}

impl CommitmentTreeProver {
    pub fn new(
        polynomials: Vec<CPUCirclePoly>,
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
        let mut merkle_input = MerkleTreeInput::new();
        const LOG_N_BASEFIELD_ELEMENTS_IN_SACK: u32 = 4;
        for column in evaluations.iter().map(|eval| &eval.values) {
            let inject_depth = column.len().ilog2() - (LOG_N_BASEFIELD_ELEMENTS_IN_SACK - 1);
            merkle_input.insert_column(inject_depth as usize, column);
        }
        let (tree, root) = MixedDegreeMerkleTree::<BaseField, Blake2sHasher>::commit(&merkle_input);
        channel.mix_digest(root);

        let structure = merkle_input.structure();

        CommitmentTreeProver {
            polynomials,
            evaluations,
            commitment: tree,
            structure,
        }
    }

    // TODO(AlonH): change interface after integrating mixed degree merkle.
    pub fn decommit(
        &self,
        positions: Vec<usize>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        MixedDecommitment<BaseField, Blake2sHasher>,
    ) {
        let values = self
            .evaluations
            .iter()
            .map(|c| positions.iter().map(|p| c[*p]).collect())
            .collect();
        // Assuming rectangle trace, queries should be similar for all columns.
        let queries = std::iter::repeat(positions.to_vec())
            .take(self.evaluations.len())
            .collect_vec();

        // Rebuild the merkle input for now. TODO(Ohad): change after tree refactor.
        let eval_vec = self
            .evaluations
            .iter()
            .map(|eval| &eval.values[..])
            .collect_vec();
        let input = self.structure.build_input(&eval_vec);
        let decommitment = self.commitment.decommit(&input, &queries);
        (values, decommitment)
    }
}

impl Deref for CommitmentTreeProver {
    type Target = MixedDegreeMerkleTree<BaseField, Blake2sHasher>;

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
        decommitments: &[MixedDecommitment<BaseField, Blake2sHasher>],
        positions: &[SparseSubCircleDomain],
    ) -> bool {
        self.commitments
            .iter()
            .zip(decommitments)
            .zip(positions)
            .all(|((commitment, decommitment), positions)| {
                let positions = std::iter::repeat(positions.flatten())
                    .take(decommitment.queried_values.len())
                    .collect_vec();
                commitment.verify(decommitment, &positions)
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
        decommitment: &MixedDecommitment<BaseField, Blake2sHasher>,
        queries: &[Vec<usize>],
    ) -> bool {
        decommitment.verify(
            self.commitment,
            queries,
            decommitment.queried_values.iter().copied(),
        )
    }
}
