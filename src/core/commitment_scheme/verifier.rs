use std::iter::zip;

use itertools::Itertools;

use super::super::channel::Blake2sChannel;
use super::super::circle::CirclePoint;
use super::super::fields::m31::BaseField;
use super::super::fields::qm31::SecureField;
use super::super::fri::{CirclePolyDegreeBound, FriConfig, FriVerifier};
use super::super::proof_of_work::ProofOfWork;
use super::super::prover::{
    LOG_BLOWUP_FACTOR, LOG_LAST_LAYER_DEGREE_BOUND, N_QUERIES, PROOF_OF_WORK_BITS,
};
use super::quotients::{fri_answers, PointOpening};
use super::utils::TreeVec;
use super::CommitmentSchemeProof;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::core::channel::Channel;
use crate::core::ColumnVec;

type MerkleHasher = Blake2sHasher;
type ProofChannel = Blake2sChannel;

/// The verifier side of a FRI polynomial commitment scheme. See [self].
#[derive(Default)]
pub struct CommitmentSchemeVerifier {
    pub trees: TreeVec<CommitmentTreeVerifier>,
}

impl CommitmentSchemeVerifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// A [TreeColumns] of the log sizes of each column in each commitment tree.
    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        self.trees.as_ref().map(|tree| tree.log_sizes.to_vec())
    }

    /// Reads a commitment from the prover.
    pub fn commit(
        &mut self,
        commitment: Blake2sHash,
        log_sizes: Vec<u32>,
        channel: &mut ProofChannel,
    ) {
        let verifier = CommitmentTreeVerifier::new(commitment, log_sizes, channel);
        self.trees.push(verifier);
    }

    pub fn verify_opening(
        &self,
        open_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof,
        channel: &mut ProofChannel,
    ) -> bool {
        let random_coeff = channel.draw_felt();

        // Compute degree bounds for oods quotients without looking at the proof.
        let bounds = self
            .column_log_sizes()
            .zip_cols(&open_points)
            .map_cols(|(log_size, open_points)| {
                vec![CirclePolyDegreeBound::new(log_size); open_points.len()]
            })
            .flatten_all()
            .into_iter()
            .sorted()
            .rev()
            .dedup()
            .collect_vec();

        // FRI commitment phase on oods quotients.
        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
        let mut fri_verifier =
            FriVerifier::commit(channel, fri_config, proof.fri_proof, bounds).unwrap();

        // Verify proof of work.
        ProofOfWork::new(PROOF_OF_WORK_BITS).verify(channel, &proof.proof_of_work);

        // Get FRI query domains.
        let fri_query_domains = fri_verifier.column_opening_positions(channel);

        // Verify merkle decommitments.
        if !self
            .trees
            .as_ref()
            .zip(&proof.decommitments)
            .map(|(tree, decommitment)| {
                // TODO(spapini): Also very opened_value here.
                tree.verify(
                    decommitment,
                    &fri_query_domains[&(tree.log_sizes[0] + LOG_BLOWUP_FACTOR)].flatten(),
                )
            })
            .0
            .iter()
            .all(|x| *x)
        {
            return false;
        }

        // Answer FRI queries.
        let openings = open_points
            .zip_cols(proof.opened_values)
            .map_cols(|(open_points, opened_values)| {
                zip(open_points, opened_values)
                    .map(|(point, value)| PointOpening { point, value })
                    .collect_vec()
            })
            .flatten();

        // TODO(spapini): Properly defined column log size and dinstinguish between poly and
        // commitment.
        let fri_answers = fri_answers(
            self.column_log_sizes()
                .flatten()
                .into_iter()
                .map(|x| x + LOG_BLOWUP_FACTOR)
                .collect(),
            &openings,
            random_coeff,
            fri_query_domains,
            &proof.queried_values.flatten(),
        );

        fri_verifier.decommit(fri_answers).is_ok()
    }
}

/// Verifier data for a single commitment tree in a commitment scheme.
pub struct CommitmentTreeVerifier {
    pub commitment: Blake2sHash,
    pub log_sizes: Vec<u32>,
}

impl CommitmentTreeVerifier {
    pub fn new(commitment: Blake2sHash, log_sizes: Vec<u32>, channel: &mut ProofChannel) -> Self {
        channel.mix_digest(commitment);
        CommitmentTreeVerifier {
            commitment,
            log_sizes,
        }
    }

    pub fn verify(
        &self,
        decommitment: &MerkleDecommitment<BaseField, MerkleHasher>,
        positions: &[usize],
    ) -> bool {
        decommitment.verify(self.commitment, positions)
    }
}
