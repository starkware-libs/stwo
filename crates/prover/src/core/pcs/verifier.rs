use std::iter::zip;

use itertools::Itertools;

use super::super::circle::CirclePoint;
use super::super::fields::qm31::SecureField;
use super::super::fri::{CirclePolyDegreeBound, FriVerifier};
use super::super::proof_of_work::ProofOfWork;
use super::quotients::{fri_answers, PointSample};
use super::utils::TreeVec;
use super::{CommitmentSchemeProof, PcsConfig};
use crate::core::channel::Channel;
use crate::core::prover::VerificationError;
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::verifier::MerkleVerifier;
use crate::core::ColumnVec;

/// The verifier side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeVerifier<H: MerkleHasher> {
    pub trees: TreeVec<MerkleVerifier<H>>,
    pub config: PcsConfig,
}

impl<H: MerkleHasher> CommitmentSchemeVerifier<H> {
    pub fn new(config: PcsConfig) -> Self {
        Self {
            trees: TreeVec::default(),
            config,
        }
    }

    /// A [TreeVec<ColumnVec>] of the log sizes of each column in each commitment tree.
    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        self.trees
            .as_ref()
            .map(|tree| tree.column_log_sizes.clone())
    }

    /// Reads a commitment from the prover.
    pub fn commit<C>(&mut self, commitment: C::Digest, log_sizes: &[u32], channel: &mut C)
    where
        C: Channel,
        H: MerkleHasher<Hash = C::Digest>,
    {
        channel.mix_digest(commitment);
        let extended_log_sizes = log_sizes
            .iter()
            .map(|&log_size| log_size + self.config.fri_config.log_blowup_factor)
            .collect();
        let verifier = MerkleVerifier::new(commitment, extended_log_sizes);
        self.trees.push(verifier);
    }

    pub fn verify_values<C>(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof<H>,
        channel: &mut C,
    ) -> Result<(), VerificationError>
    where
        C: Channel,
        H: MerkleHasher<Hash = C::Digest>,
    {
        channel.mix_felts(&proof.sampled_values.clone().flatten_cols());
        let random_coeff = channel.draw_felt();

        let bounds = self
            .column_log_sizes()
            .zip_cols(&sampled_points)
            .map_cols(|(log_size, sampled_points)| {
                vec![
                    CirclePolyDegreeBound::new(log_size - self.config.fri_config.log_blowup_factor);
                    sampled_points.len()
                ]
            })
            .flatten_cols()
            .into_iter()
            .sorted()
            .rev()
            .dedup()
            .collect_vec();

        // FRI commitment phase on OODS quotients.
        let mut fri_verifier =
            FriVerifier::commit(channel, self.config.fri_config, proof.fri_proof, bounds)?;

        // Verify proof of work.
        ProofOfWork::new(self.config.pow_bits).verify(channel, &proof.proof_of_work)?;

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
