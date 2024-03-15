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
use super::quotients::{fri_answers, PointSample};
use super::utils::TreeVec;
use super::CommitmentSchemeProof;
use crate::commitment_scheme::blake2_hash::{Blake2sHash, Blake2sHasher};
use crate::commitment_scheme::mixed_degree_decommitment::MixedDecommitment;
use crate::core::channel::Channel;
use crate::core::prover::VerificationError;
use crate::core::ColumnVec;

type ProofChannel = Blake2sChannel;

/// The verifier side of a FRI polynomial commitment scheme. See [super].
#[derive(Default)]
pub struct CommitmentSchemeVerifier {
    pub trees: TreeVec<CommitmentTreeVerifier>,
}

impl CommitmentSchemeVerifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// A [TreeVec<ColumnVec>] of the log sizes of each column in each commitment tree.
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

    pub fn verify_values(
        &self,
        prove_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof,
        channel: &mut ProofChannel,
    ) -> Result<(), VerificationError> {
        channel.mix_felts(&proof.proved_values.clone().flatten_cols());
        let random_coeff = channel.draw_felt();

        let bounds = self
            .column_log_sizes()
            .zip_cols(&prove_points)
            .map_cols(|(log_size, prove_points)| {
                vec![CirclePolyDegreeBound::new(log_size); prove_points.len()]
            })
            .flatten_cols()
            .into_iter()
            .sorted()
            .rev()
            .dedup()
            .collect_vec();

        // FRI commitment phase on OODS quotients.
        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
        let mut fri_verifier =
            FriVerifier::commit(channel, fri_config, proof.fri_proof, bounds).unwrap();

        // Verify proof of work.
        ProofOfWork::new(PROOF_OF_WORK_BITS).verify(channel, &proof.proof_of_work)?;

        // Get FRI query domains.
        let fri_query_domains = fri_verifier.column_opening_positions(channel);

        // Verify merkle decommitments.
        if !self
            .trees
            .as_ref()
            .zip(&proof.decommitments)
            .map(|(tree, decommitment)| {
                // TODO(spapini): Also verify proved_values here.
                // Assuming columns are of equal lengths, replicate queries for all columns.
                // TOOD(AlonH): remove this assumption.
                tree.verify(
                    decommitment,
                    &std::iter::repeat(
                        fri_query_domains[&(tree.log_sizes[0] + LOG_BLOWUP_FACTOR)]
                            .flatten()
                            .clone(),
                    )
                    .take(tree.log_sizes.len())
                    .collect_vec(),
                )
            })
            .iter()
            .all(|x| *x)
        {
            return Err(VerificationError::MerkleVerificationFailed);
        }

        // Answer FRI queries.
        let openings = prove_points
            .zip_cols(proof.proved_values)
            .map_cols(|(prove_points, proved_values)| {
                zip(prove_points, proved_values)
                    .map(|(point, value)| PointSample { point, value })
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
        )?;

        fri_verifier.decommit(fri_answers)?;
        Ok(())
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
