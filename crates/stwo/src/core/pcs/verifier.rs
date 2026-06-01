use core::iter::zip;

use itertools::Itertools;
use std_shims::{String, Vec};

use super::super::circle::CirclePoint;
use super::super::fields::qm31::SecureField;
use super::super::fri::{CirclePolyDegreeBound, FriVerifier};
use super::quotients::{fri_answers, PointSample};
use super::utils::TreeVec;
use super::PcsConfig;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::pcs::quotients::CommitmentSchemeProof;
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleVerifierLifted;
use crate::core::verifier::VerificationError;
use crate::core::zk::{
    mix_zk_public_metadata, validate_zk_public_metadata_against_verifier_config,
    validate_zk_public_only_metadata, validate_zk_sampled_values_shape,
    validate_zk_witness_metadata, zk_fri_batch_mask_fri_config, zk_fri_batch_mask_query_positions,
    ZkColumnDegreeBound, ZkCommitmentSchemeProof, ZkVerificationConfig,
};
use crate::core::ColumnVec;

/// The verifier side of a FRI polynomial commitment scheme. See [super].
#[derive(Default)]
pub struct CommitmentSchemeVerifier<MC: MerkleChannel> {
    pub trees: TreeVec<MerkleVerifierLifted<MC::H>>,
    pub config: PcsConfig,
}

impl<MC: MerkleChannel> CommitmentSchemeVerifier<MC> {
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

    fn validate_zk_private_column_degree_bounds(
        &self,
        bounds: &[ZkColumnDegreeBound],
    ) -> Result<(), VerificationError> {
        for bound in bounds {
            let Some(tree) = self.trees.0.get(bound.range.tree_index) else {
                return Err(VerificationError::InvalidStructure(String::from(
                    "ZK private column degree bound references a missing commitment tree",
                )));
            };
            if bound.range.column_start >= bound.range.column_end
                || bound.range.column_end > tree.column_log_sizes.len()
            {
                return Err(VerificationError::InvalidStructure(String::from(
                    "ZK private column degree bound references missing commitment columns",
                )));
            }
            for column_index in bound.range.column_start..bound.range.column_end {
                let Some(committed_log_degree) = tree.column_log_sizes[column_index]
                    .checked_sub(self.config.fri_config.log_blowup_factor)
                else {
                    return Err(VerificationError::InvalidStructure(String::from(
                        "ZK private column commitment log size is below PCS blowup",
                    )));
                };
                if committed_log_degree != bound.log_degree_bound {
                    return Err(VerificationError::InvalidStructure(String::from(
                        "ZK private column degree bound does not match committed PCS column",
                    )));
                }
            }
        }

        Ok(())
    }

    /// Reads a commitment from the prover.
    pub fn commit(
        &mut self,
        commitment: <MC::H as MerkleHasherLifted>::Hash,
        log_sizes: &[u32],
        channel: &mut MC::C,
    ) {
        MC::mix_root(channel, commitment);
        let extended_log_sizes = log_sizes
            .iter()
            .map(|&log_size| log_size + self.config.fri_config.log_blowup_factor)
            .collect();
        let verifier =
            MerkleVerifierLifted::new(commitment, extended_log_sizes, self.config.lifting_log_size);
        self.trees.push(verifier);
    }

    pub fn verify_values(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: CommitmentSchemeProof<MC::H>,
        channel: &mut MC::C,
    ) -> Result<(), VerificationError> {
        channel.mix_felts(&proof.sampled_values.clone().flatten_cols());
        let random_coeff = channel.draw_secure_felt();
        let lifting_log_size = self.trees.last().unwrap().height;
        let bound =
            CirclePolyDegreeBound::new(lifting_log_size - self.config.fri_config.log_blowup_factor);

        // FRI commitment phase on OODS quotients.
        let mut fri_verifier =
            FriVerifier::<MC>::commit(channel, self.config.fri_config, proof.fri_proof, bound)?;

        // Verify proof of work.
        if !channel.verify_pow_nonce(self.config.pow_bits, proof.proof_of_work) {
            return Err(VerificationError::ProofOfWork);
        }
        channel.mix_u64(proof.proof_of_work);
        // Get FRI query positions.
        let query_positions = fri_verifier.sample_query_positions(channel);
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].height,
        );

        // Build the query positions tree: the preprocessed tree needs a different treatment than
        // the other trees.
        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        // Verify decommitments.
        self.trees
            .as_ref()
            .zip_eq(proof.decommitments)
            .zip_eq(proof.queried_values.clone())
            .zip_eq(query_positions_tree)
            .map(
                |(((tree, decommitment), queried_values), query_positions)| {
                    tree.verify(query_positions, queried_values, decommitment)
                },
            )
            .0
            .into_iter()
            .collect::<Result<(), _>>()?;
        // Answer FRI queries.
        let samples = sampled_points.zip_cols(proof.sampled_values).map_cols(
            |(sampled_points, sampled_values)| {
                zip(sampled_points, sampled_values)
                    .map(|(point, value)| PointSample { point, value })
                    .collect_vec()
            },
        );

        let fri_answers = fri_answers(
            self.column_log_sizes(),
            samples,
            random_coeff,
            &query_positions,
            proof.queried_values,
            lifting_log_size,
        )?;

        fri_verifier.decommit(fri_answers)?;

        Ok(())
    }

    pub fn verify_values_zk(
        &self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        proof: ZkCommitmentSchemeProof<MC::H>,
        zk_config: &ZkVerificationConfig,
        channel: &mut MC::C,
    ) -> Result<(), VerificationError> {
        if proof.version != zk_config.metadata.version
            || proof.public_metadata != zk_config.metadata
        {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK public metadata does not match verifier configuration",
            )));
        }

        let lifting_log_size = self.trees.last().unwrap().height;
        validate_zk_public_metadata_against_verifier_config(&proof.public_metadata, zk_config)
            .map_err(|_| {
                VerificationError::InvalidStructure(String::from(
                    "Invalid ZK public metadata or degree bounds",
                ))
            })?;
        if zk_config
            .metadata
            .witness_randomization
            .private_column_degree_bounds
            .is_empty()
            && zk_config
                .metadata
                .quotient_integration
                .quotient_degree_bounds
                .is_empty()
        {
            validate_zk_public_only_metadata(
                &zk_config.metadata,
                lifting_log_size,
                self.config.fri_config.log_blowup_factor,
            )
            .map_err(|_| {
                VerificationError::InvalidStructure(String::from("Invalid ZK public metadata"))
            })?;
        } else {
            validate_zk_witness_metadata(
                &zk_config.metadata,
                lifting_log_size,
                self.config.fri_config.log_blowup_factor,
            )
            .map_err(|_| {
                VerificationError::InvalidStructure(String::from("Invalid ZK witness metadata"))
            })?;
        }
        if zk_config.metadata.degree_profile.fri_first_layer_log_size != lifting_log_size
            || zk_config
                .metadata
                .quotient_integration
                .fri_first_layer_log_size
                != lifting_log_size
        {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK FRI first-layer domain does not match PCS lifting domain",
            )));
        }
        self.validate_zk_private_column_degree_bounds(
            &zk_config
                .metadata
                .witness_randomization
                .private_column_degree_bounds,
        )?;
        if proof.randomized_pcs_proof.commitments.len() != self.trees.len()
            || proof
                .randomized_pcs_proof
                .commitments
                .iter()
                .zip(self.trees.iter())
                .any(|(commitment, tree)| commitment != &tree.root)
        {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK PCS proof commitments do not match verifier commitment state",
            )));
        }

        let ZkCommitmentSchemeProof {
            randomized_pcs_proof: proof,
            fri_batch_mask,
            ..
        } = proof;

        if fri_batch_mask.log_size != lifting_log_size {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK FRI batch mask domain does not match first FRI layer",
            )));
        }
        if fri_batch_mask.commitment != fri_batch_mask.fri_proof.first_layer.commitment {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK FRI batch mask opening commitment does not match its low-degree proof",
            )));
        }

        mix_zk_public_metadata(
            channel,
            &zk_config.metadata,
            &zk_config.column_degree_bounds,
        );
        channel.mix_felts(&proof.sampled_values.clone().flatten_cols());
        let bound =
            CirclePolyDegreeBound::new(lifting_log_size - self.config.fri_config.log_blowup_factor);
        let fri_batch_mask_fri_proof = fri_batch_mask.fri_proof.clone();
        let fri_batch_mask_fri_verifier = FriVerifier::<MC>::commit(
            channel,
            zk_fri_batch_mask_fri_config(self.config.fri_config),
            fri_batch_mask_fri_proof,
            bound,
        )?;
        let random_coeff = channel.draw_secure_felt();

        let mut fri_verifier =
            FriVerifier::<MC>::commit(channel, self.config.fri_config, proof.fri_proof, bound)?;

        if !channel.verify_pow_nonce(self.config.pow_bits, proof.proof_of_work) {
            return Err(VerificationError::ProofOfWork);
        }
        channel.mix_u64(proof.proof_of_work);
        let query_positions = fri_verifier.sample_query_positions(channel);
        let fri_batch_mask_query_positions = zk_fri_batch_mask_query_positions(
            channel,
            lifting_log_size,
            self.config.fri_config.n_queries,
            &query_positions,
            self.config.fri_config,
        )
        .map_err(|_| {
            VerificationError::InvalidStructure(String::from(
                "Insufficient ZK FRI batch mask query domain",
            ))
        })?;
        let preprocessed_query_positions = prepare_preprocessed_query_positions(
            &query_positions,
            lifting_log_size,
            self.trees[0].height,
        );
        if zk_config
            .metadata
            .witness_randomization
            .private_column_degree_bounds
            .iter()
            .any(|bound| bound.range.tree_index == 0)
            && preprocessed_query_positions != query_positions
        {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK private witness columns in tree 0 require matching query positions",
            )));
        }

        let query_positions_tree = TreeVec::new(
            self.trees
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i == 0 {
                        preprocessed_query_positions.as_slice()
                    } else {
                        query_positions.as_slice()
                    }
                })
                .collect::<Vec<_>>(),
        );
        if proof.decommitments.len() != self.trees.len()
            || proof.queried_values.len() != self.trees.len()
            || sampled_points.len() != self.trees.len()
            || proof.sampled_values.len() != self.trees.len()
        {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK PCS proof tree structure does not match verifier state",
            )));
        }
        for (tree_index, tree) in self.trees.iter().enumerate() {
            if sampled_points[tree_index].len() != tree.column_log_sizes.len()
                || proof.sampled_values[tree_index].len() != tree.column_log_sizes.len()
            {
                return Err(VerificationError::InvalidStructure(String::from(
                    "ZK PCS sampled-value column structure does not match verifier state",
                )));
            }
        }
        validate_zk_sampled_values_shape(&sampled_points, &proof.sampled_values).map_err(|_| {
            VerificationError::InvalidStructure(String::from(
                "ZK PCS sampled-value shape does not match sampled points",
            ))
        })?;

        self.trees
            .as_ref()
            .zip_eq(proof.decommitments)
            .zip_eq(proof.queried_values.clone())
            .zip_eq(query_positions_tree)
            .map(
                |(((tree, decommitment), queried_values), query_positions)| {
                    tree.verify(query_positions, queried_values, decommitment)
                },
            )
            .0
            .into_iter()
            .collect::<Result<(), _>>()?;

        let samples = sampled_points.zip_cols(proof.sampled_values).map_cols(
            |(sampled_points, sampled_values)| {
                zip(sampled_points, sampled_values)
                    .map(|(point, value)| PointSample { point, value })
                    .collect_vec()
            },
        );

        let mut fri_answers = fri_answers(
            self.column_log_sizes(),
            samples,
            random_coeff,
            &query_positions,
            proof.queried_values,
            lifting_log_size,
        )?;
        let fri_batch_mask_fri_values = fri_batch_mask.fri_queried_values.to_secure_values();
        if fri_batch_mask_fri_values.len() != fri_batch_mask_query_positions.len() {
            return Err(VerificationError::InvalidStructure(String::from(
                "Unexpected ZK FRI batch mask low-degree query count",
            )));
        }
        let fri_batch_mask_values = fri_batch_mask
            .verify_openings(&query_positions, lifting_log_size)
            .map_err(|_| {
                VerificationError::InvalidStructure(String::from(
                    "Invalid ZK FRI batch mask openings",
                ))
            })?;
        fri_batch_mask_fri_verifier.decommit_on_query_positions(
            &fri_batch_mask_query_positions,
            fri_batch_mask_fri_values,
        )?;
        if fri_batch_mask_values.len() != fri_answers.len() {
            return Err(VerificationError::InvalidStructure(String::from(
                "Unexpected ZK FRI batch mask query count",
            )));
        }
        for (answer, mask_value) in fri_answers.iter_mut().zip(fri_batch_mask_values) {
            *answer += mask_value;
        }

        fri_verifier.decommit(fri_answers)?;

        Ok(())
    }
}
