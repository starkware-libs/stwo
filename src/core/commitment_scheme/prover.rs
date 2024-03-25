use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use super::super::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
use super::super::backend::CPUBackend;
use super::super::channel::Blake2sChannel;
use super::super::circle::CirclePoint;
use super::super::fields::m31::BaseField;
use super::super::fields::qm31::SecureField;
use super::super::fri::{FriConfig, FriProof, FriProver};
use super::super::oods::get_pair_oods_quotient;
use super::super::poly::circle::CanonicCoset;
use super::super::poly::BitReversedOrder;
use super::super::proof_of_work::{ProofOfWork, ProofOfWorkProof};
use super::super::prover::{
    LOG_BLOWUP_FACTOR, LOG_LAST_LAYER_DEGREE_BOUND, N_QUERIES, PROOF_OF_WORK_BITS,
};
use super::super::ColumnVec;
use super::utils::TreeVec;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_input::{MerkleTreeColumnLayout, MerkleTreeInput};
use crate::commitment_scheme::mixed_degree_decommitment::MixedDecommitment;
use crate::commitment_scheme::mixed_degree_merkle_tree::MixedDegreeMerkleTree;
use crate::core::channel::Channel;

type MerkleHasher = Blake2sHasher;
type ProofChannel = Blake2sChannel;

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver {
    pub trees: TreeVec<CommitmentTreeProver>,
    pub log_blowup_factor: u32,
}

impl CommitmentSchemeProver {
    pub fn new(log_blowup_factor: u32) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::<CommitmentTreeProver>::default(),
            log_blowup_factor,
        }
    }

    pub fn commit(&mut self, polynomials: ColumnVec<CPUCirclePoly>, channel: &mut ProofChannel) {
        let tree = CommitmentTreeProver::new(polynomials, self.log_blowup_factor, channel);
        self.trees.push(tree);
    }

    pub fn roots(&self) -> TreeVec<Blake2sHash> {
        self.trees.as_ref().map(|tree| tree.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&CPUCirclePoly>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    fn evaluations(&self) -> TreeVec<ColumnVec<&CPUCircleEvaluation<BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.evaluations.iter().collect())
    }

    pub fn prove_values(
        &self,
        prove_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut ProofChannel,
    ) -> CommitmentSchemeProof {
        // Evaluate polynomials on open points.
        let proved_values =
            self.polynomials()
                .zip_cols(&prove_points)
                .map_cols(|(poly, points)| {
                    points
                        .iter()
                        .map(|point| poly.eval_at_point(*point))
                        .collect_vec()
                });
        channel.mix_felts(&proved_values.clone().flatten_cols());

        // Compute oods quotients for boundary constraints on prove_points.
        let quotients = self
            .evaluations()
            .zip_cols(&proved_values)
            .zip_cols(&prove_points)
            .map_cols(|((evaluation, values), points)| {
                zip(points, values)
                    .map(|(&point, &value)| {
                        get_pair_oods_quotient(point, value, evaluation).bit_reverse()
                    })
                    .collect_vec()
            });

        // Run FRI commitment phase on the oods quotients.
        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
        // TODO(spapini): Remove rev() when we start accumulating by size.
        //   This is only done because fri demands descending sizes.
        let fri_prover = FriProver::<CPUBackend, MerkleHasher>::commit(
            channel,
            fri_config,
            &quotients.flatten_cols_rev(),
        );

        // Proof of work.
        let proof_of_work = ProofOfWork::new(PROOF_OF_WORK_BITS).prove(channel);

        // FRI decommitment phase.
        let (fri_proof, fri_query_domains) = fri_prover.decommit(channel);

        // Decommit the FRI queries on the merkle trees.
        let decommitment_results = self.trees.as_ref().map(|tree| {
            let queries = tree
                .polynomials
                .iter()
                .map(|poly| {
                    fri_query_domains[&(poly.log_size() + self.log_blowup_factor)].flatten()
                })
                .collect();
            tree.decommit(queries)
        });

        let queried_values = decommitment_results.as_ref().map(|(v, _)| v.clone());
        let decommitments = decommitment_results.map(|(_, d)| d);

        CommitmentSchemeProof {
            proved_values,
            decommitments,
            queried_values,
            proof_of_work,
            fri_proof,
        }
    }
}

#[derive(Debug)]
pub struct CommitmentSchemeProof {
    pub proved_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    pub decommitments: TreeVec<MixedDecommitment<BaseField, MerkleHasher>>,
    pub queried_values: TreeVec<ColumnVec<Vec<BaseField>>>,
    pub proof_of_work: ProofOfWorkProof,
    pub fri_proof: FriProof<MerkleHasher>,
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver {
    pub polynomials: ColumnVec<CPUCirclePoly>,
    pub evaluations: ColumnVec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
    pub commitment: MixedDegreeMerkleTree<BaseField, Blake2sHasher>,
    column_layout: MerkleTreeColumnLayout,
}

impl CommitmentTreeProver {
    fn new(
        polynomials: ColumnVec<CPUCirclePoly>,
        log_blowup_factor: u32,
        channel: &mut ProofChannel,
    ) -> Self {
        let evaluations = polynomials
            .iter()
            .map(|poly| {
                poly.evaluate(
                    CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain(),
                )
            })
            .collect_vec();

        let mut merkle_input = MerkleTreeInput::new();
        const LOG_N_BASE_FIELD_ELEMENTS_IN_SACK: u32 = 4;

        for eval in evaluations.iter() {
            // The desired depth for a column of log_length n is such that Blake2s hashes are
            // filled(64B). Explicitly: There are 2^(d-1) hash 'sacks' at depth d,
            // hence, with elements of 4 bytes, 2^(d-1) = 2^n / 16, => d = n-3.
            let inject_depth = std::cmp::max::<i32>(
                eval.domain.log_size() as i32 - (LOG_N_BASE_FIELD_ELEMENTS_IN_SACK as i32 - 1),
                1,
            );
            merkle_input.insert_column(inject_depth as usize, &eval.values);
        }

        let (tree, root) =
            MixedDegreeMerkleTree::<BaseField, Blake2sHasher>::commit_default(&merkle_input);
        channel.mix_digest(root);

        let column_layout = merkle_input.column_layout();

        CommitmentTreeProver {
            polynomials,
            evaluations,
            commitment: tree,
            column_layout,
        }
    }

    /// Decommits the merkle tree on the given query positions.
    fn decommit(
        &self,
        queries: ColumnVec<Vec<usize>>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        MixedDecommitment<BaseField, Blake2sHasher>,
    ) {
        let values = zip(&self.evaluations, &queries)
            .map(|(column, column_queries)| column_queries.iter().map(|q| column[*q]).collect())
            .collect();

        // Rebuild the merkle input for now.
        // TODO(Ohad): change after tree refactor. Consider removing the input struct and have the
        // decommitment take queries and columns only.
        let eval_vec = self
            .evaluations
            .iter()
            .map(|eval| &eval.values[..])
            .collect_vec();
        let input = self.column_layout.build_input(&eval_vec);
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
