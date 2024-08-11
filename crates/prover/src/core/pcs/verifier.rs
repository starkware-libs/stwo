use std::iter::zip;

use itertools::Itertools;

use super::super::circle::CirclePoint;
use super::super::fields::qm31::SecureField;
use super::super::fri::{CirclePolyDegreeBound, FriConfig, FriVerifier};
use super::super::prover::{
    LOG_BLOWUP_FACTOR, LOG_LAST_LAYER_DEGREE_BOUND, N_QUERIES, PROOF_OF_WORK_BITS,
};
use super::quotients::{fri_answers, PointSample};
use super::utils::TreeVec;
use super::CommitmentSchemeProof;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::prover::VerificationError;
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::verifier::MerkleVerifier;
use crate::core::ColumnVec;

/// The verifier side of a FRI polynomial commitment scheme. See [super].
#[derive(Default)]
pub struct CommitmentSchemeVerifier<MC: MerkleChannel> {
    pub trees: TreeVec<MerkleVerifier<MC::H>>,
}

impl<MC: MerkleChannel> CommitmentSchemeVerifier<MC> {
    pub fn new() -> Self {
        Self::default()
    }

    /// A [TreeVec<ColumnVec>] of the log sizes of each column in each commitment tree.
    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        self.trees
            .as_ref()
            .map(|tree| tree.column_log_sizes.clone())
    }

    /// Reads a commitment from the prover.
    pub fn commit(
        &mut self,
        commitment: <MC::H as MerkleHasher>::Hash,
        log_sizes: &[u32],
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, commitment);
        let extended_log_sizes = log_sizes
            .iter()
            .map(|&log_size| log_size + LOG_BLOWUP_FACTOR)
            .collect();
        let verifier = MerkleVerifier::new(commitment, extended_log_sizes);
        self.trees.push(verifier);
    }

    pub fn verify_values(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof<MC::H>,
        channel: &mut MC::C,
    ) -> Result<(), VerificationError> {
        channel.mix_felts(&proof.sampled_values.clone().flatten_cols());
        let random_coeff = channel.draw_felt();

        let bounds = self
            .column_log_sizes()
            .zip_cols(&sampled_points)
            .map_cols(|(log_size, sampled_points)| {
                vec![CirclePolyDegreeBound::new(log_size - LOG_BLOWUP_FACTOR); sampled_points.len()]
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
            FriVerifier::<MC>::commit(channel, fri_config, proof.fri_proof, bounds)?;

        // Verify proof of work.
        channel.mix_nonce(proof.proof_of_work);
        if channel.leading_zeros() < PROOF_OF_WORK_BITS {
            return Err(VerificationError::ProofOfWork);
        }

        // Get FRI query domains.
        let fri_query_domains = fri_verifier.column_query_positions(channel);

        // Verify merkle decommitments.
        self.trees
            .as_ref()
            .zip_eq(proof.decommitments)
            .zip_eq(proof.queried_values.clone())
            .map(|((tree, decommitment), queried_values)| {
                let queries = fri_query_domains
                    .iter()
                    .map(|(&log_size, domain)| (log_size, domain.flatten()))
                    .collect();
                tree.verify(queries, queried_values, decommitment)
            })
            .0
            .into_iter()
            .collect::<Result<_, _>>()?;

        // Answer FRI queries.
        let samples = sampled_points
            .zip_cols(proof.sampled_values)
            .map_cols(|(sampled_points, sampled_values)| {
                zip(sampled_points, sampled_values)
                    .map(|(point, value)| PointSample { point, value })
                    .collect_vec()
            })
            .flatten();

        // TODO(spapini): Properly defined column log size and dinstinguish between poly and
        // commitment.
        let fri_answers = fri_answers(
            self.column_log_sizes().flatten().into_iter().collect(),
            &samples,
            random_coeff,
            fri_query_domains,
            &proof.queried_values.flatten(),
        )?;

        fri_verifier.decommit(fri_answers)?;
        Ok(())
    }
}
