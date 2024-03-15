//! Implements a FRI polynomial commitment scheme.
//! This is a protocol where the prover can commit on a set of polynomials and then prove their
//! opening on a set of points.
//! Note: This implementation is not really a polynomial commitment scheme, because we are not in
//! the unique decoding regime. This is enough for a STARK proof though, where we onyl want to imply
//! the existence of such polynomials, and re ok with having a small decoding list.

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
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::commitment_scheme::merkle_tree::MerkleTree;
use crate::core::channel::Channel;

type MerkleHasher = Blake2sHasher;
type ProofChannel = Blake2sChannel;

/// The prover side of a FRI polynomial commitment scheme. See [self].
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
            tree.decommit(
                fri_query_domains[&(tree.polynomials[0].log_size() + self.log_blowup_factor)]
                    .flatten(),
            )
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
    pub decommitments: TreeVec<MerkleDecommitment<BaseField, MerkleHasher>>,
    pub queried_values: TreeVec<ColumnVec<Vec<BaseField>>>,
    pub proof_of_work: ProofOfWorkProof,
    pub fri_proof: FriProof<MerkleHasher>,
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver {
    pub polynomials: ColumnVec<CPUCirclePoly>,
    pub evaluations: ColumnVec<CPUCircleEvaluation<BaseField, BitReversedOrder>>,
    // TODO(AlonH): Change to mixed degree merkle and remove values clone.
    commitment: MerkleTree<BaseField, MerkleHasher>,
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
        let commitment = MerkleTree::<BaseField, MerkleHasher>::commit(
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
    /// Decommits the merkle tree on the given query positions.
    fn decommit(
        &self,
        queries: Vec<usize>,
    ) -> (
        ColumnVec<Vec<BaseField>>,
        MerkleDecommitment<BaseField, MerkleHasher>,
    ) {
        let values = self
            .evaluations
            .iter()
            .map(|c| queries.iter().map(|p| c[*p]).collect())
            .collect();
        let decommitment = self.commitment.generate_decommitment(queries);
        (values, decommitment)
    }
}

impl Deref for CommitmentTreeProver {
    type Target = MerkleTree<BaseField, MerkleHasher>;

    fn deref(&self) -> &Self::Target {
        &self.commitment
    }
}
