use std_shims::{vec, String};
use thiserror::Error;

use crate::core::air::{Component, Components};
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::fri::FriVerificationError;
use crate::core::pcs::utils::try_get_lifting_log_size;
use crate::core::pcs::CommitmentSchemeVerifier;
use crate::core::proof::StarkProof;
use crate::core::vcs_lifted::verifier::MerkleVerificationError;
use crate::core::zk::{
    derive_zk_stark_degree_bound_profile, draw_zk_oods_point,
    draw_zk_oods_sample_point_plan_with_semantic_sample_domains, mix_zk_public_metadata,
    mix_zk_quotient_split_mask_profile, validate_zk_committed_column_log_sizes,
    validate_zk_composition_column_log_sizes_against_bounds,
    validate_zk_public_metadata_against_verifier_config,
    validate_zk_quotient_split_mask_query_budget, validate_zk_sample_points_outside_exclusion_set,
    validate_zk_sampled_values_shape, zk_metadata_requires_private_stark_activation,
    zk_oods_exclusion_set, zk_trace_domain_log_size_from_column_bounds,
    ZkCommittedColumnLogSizeValidationError, ZkOodsExclusionSet, ZkOodsSamplePointPlanningError,
    ZkOodsSamplePointValidationError, ZkStarkDegreeBoundProfileError, ZkStarkProof,
    ZkVerificationConfig, ZkWitnessRandomizationVerifierAudit,
};
pub const PREPROCESSED_TRACE_IDX: usize = 0;

// TODO(Leo): remove this once the composition poly split can be dependant on a config instead of
// being hardcoded.
pub const COMPOSITION_LOG_SPLIT: u32 = 1;

pub fn verify<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: StarkProof<MC::H>,
) -> Result<(), VerificationError> {
    let include_all_preprocessed_columns = false;
    verify_ex(
        components,
        channel,
        commitment_scheme,
        proof,
        include_all_preprocessed_columns,
    )
}

pub fn verify_ex<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: StarkProof<MC::H>,
    include_all_preprocessed_columns: bool,
) -> Result<(), VerificationError> {
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .column_log_sizes
        .len();

    let components = Components {
        components: components.to_vec(),
        n_preprocessed_columns,
    };
    let split_composition_log_degree_bound =
        components.composition_log_degree_bound() - COMPOSITION_LOG_SPLIT;
    tracing::info!(
        "Split composition polynomial log degree bound: {}",
        split_composition_log_degree_bound
    );

    // If `self.config.lifting_log_size` is None, the lifting size is the length of the split
    // composition polynomials' domain.
    let lifting_log_size = try_get_lifting_log_size(
        &commitment_scheme.config,
        split_composition_log_degree_bound + commitment_scheme.config.fri_config.log_blowup_factor,
    )?;
    if include_all_preprocessed_columns {
        let preprocessed_trace_height = commitment_scheme.trees[PREPROCESSED_TRACE_IDX].height;
        if lifting_log_size < preprocessed_trace_height {
            Err(crate::core::pcs::utils::InvalidLiftingLogSizeError {
                lifting_log_size,
                min_log_size: preprocessed_trace_height,
            })?;
        }
    }

    // The max degree of a committed polynomial. If `lifting_log_size` is not set,
    // the largest degree is attained by the splits of the composition polynomial.
    let max_log_degree_bound = split_composition_log_degree_bound;

    let random_coeff = channel.draw_secure_felt();

    // Read composition polynomial commitment.
    commitment_scheme.commit(
        *proof.commitments.last().unwrap(),
        &[max_log_degree_bound; 2 * SECURE_EXTENSION_DEGREE],
        channel,
    );

    let committed_column_log_sizes = commitment_scheme
        .trees
        .as_ref()
        .map(|tree| tree.column_log_sizes.clone());
    let requires_lifted_sample_planning = committed_column_log_sizes
        .iter()
        .flatten()
        .any(|&column_log_size| column_log_size < lifting_log_size);
    let (oods_point, sample_points) = if requires_lifted_sample_planning {
        let mut mask_offsets = components
            .mask_offsets(include_all_preprocessed_columns)
            .ok_or_else(|| {
                VerificationError::InvalidStructure(String::from(
                    "Lifted component does not expose structured mask offsets",
                ))
            })?;
        mask_offsets.push(vec![vec![0]; 2 * SECURE_EXTENSION_DEGREE]);

        let (mut semantic_sample_step_log_sizes, mut semantic_sample_base_doublings) = components
            .semantic_sample_step_log_sizes_and_lifts(include_all_preprocessed_columns)
            .ok_or_else(|| {
                VerificationError::InvalidStructure(String::from(
                    "Lifted component does not expose semantic sample metadata",
                ))
            })?;
        semantic_sample_step_log_sizes.push(vec![
            vec![max_log_degree_bound];
            2 * SECURE_EXTENSION_DEGREE
        ]);
        semantic_sample_base_doublings.push(vec![vec![0]; 2 * SECURE_EXTENSION_DEGREE]);

        let sample_point_plan = draw_zk_oods_sample_point_plan_with_semantic_sample_domains(
            channel,
            &mask_offsets,
            &semantic_sample_step_log_sizes,
            &semantic_sample_base_doublings,
            &committed_column_log_sizes,
            lifting_log_size,
            &ZkOodsExclusionSet {
                forbidden_cosets: vec![],
                reject_line_degeneracy: false,
            },
            1,
        )
        .map_err(VerificationError::ZkOodsSamplePointPlan)?;

        (
            sample_point_plan.semantic_oods_point,
            sample_point_plan.pcs_sample_points,
        )
    } else {
        let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
        let mut sample_points = components.mask_points(
            oods_point,
            max_log_degree_bound,
            include_all_preprocessed_columns,
        );
        sample_points.push(vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE]);

        (oods_point, sample_points)
    };

    let sample_points_by_column = sample_points.as_cols_ref().flatten();
    tracing::info!("Sampling {} columns.", sample_points_by_column.len());
    tracing::info!(
        "Total sample points: {}.",
        sample_points_by_column.into_iter().flatten().count()
    );

    let composition_oods_eval = proof
        .extract_composition_oods_eval(oods_point, max_log_degree_bound)
        .ok_or(VerificationError::InvalidStructure(
            std_shims::ToString::to_string(&"Unexpected sampled_values structure"),
        ))?;

    if composition_oods_eval
        != components.eval_composition_polynomial_at_point(
            oods_point,
            &proof.sampled_values,
            random_coeff,
            max_log_degree_bound,
        )
    {
        return Err(VerificationError::OodsNotMatching);
    }
    commitment_scheme.verify_values(sample_points, proof.0, channel)
}

pub fn verify_zk<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: ZkStarkProof<MC::H>,
    zk_config: &ZkVerificationConfig,
) -> Result<(), VerificationError> {
    let include_all_preprocessed_columns = false;
    verify_zk_ex(
        components,
        channel,
        commitment_scheme,
        proof,
        zk_config,
        include_all_preprocessed_columns,
    )
}

pub fn verify_zk_ex<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: ZkStarkProof<MC::H>,
    zk_config: &ZkVerificationConfig,
    include_all_preprocessed_columns: bool,
) -> Result<(), VerificationError> {
    verify_zk_ex_with_optional_witness_randomization_audit(
        components,
        channel,
        commitment_scheme,
        proof,
        zk_config,
        None,
        include_all_preprocessed_columns,
    )
}

pub fn verify_zk_with_witness_randomization_audit<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: ZkStarkProof<MC::H>,
    zk_config: &ZkVerificationConfig,
    witness_randomization_audit: &ZkWitnessRandomizationVerifierAudit,
) -> Result<(), VerificationError> {
    let include_all_preprocessed_columns = false;
    verify_zk_ex_with_witness_randomization_audit(
        components,
        channel,
        commitment_scheme,
        proof,
        zk_config,
        witness_randomization_audit,
        include_all_preprocessed_columns,
    )
}

pub fn verify_zk_ex_with_witness_randomization_audit<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: ZkStarkProof<MC::H>,
    zk_config: &ZkVerificationConfig,
    witness_randomization_audit: &ZkWitnessRandomizationVerifierAudit,
    include_all_preprocessed_columns: bool,
) -> Result<(), VerificationError> {
    verify_zk_ex_with_optional_witness_randomization_audit(
        components,
        channel,
        commitment_scheme,
        proof,
        zk_config,
        Some(witness_randomization_audit),
        include_all_preprocessed_columns,
    )
}

fn verify_zk_ex_with_optional_witness_randomization_audit<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: ZkStarkProof<MC::H>,
    zk_config: &ZkVerificationConfig,
    witness_randomization_audit: Option<&ZkWitnessRandomizationVerifierAudit>,
    include_all_preprocessed_columns: bool,
) -> Result<(), VerificationError> {
    validate_zk_public_metadata_against_verifier_config(&zk_config.metadata, zk_config).map_err(
        |_| {
            VerificationError::InvalidStructure(String::from(
                "Invalid ZK verifier metadata or degree bounds",
            ))
        },
    )?;
    let requires_private_stark_activation =
        zk_metadata_requires_private_stark_activation(&zk_config.metadata);
    if requires_private_stark_activation {
        if commitment_scheme.config.lifting_log_size
            != Some(zk_config.metadata.degree_profile.fri_first_layer_log_size)
        {
            return Err(VerificationError::InvalidStructure(String::from(
                "ZK private witness verification requires PCS lifting at the ZK FRI first layer",
            )));
        }
        let profile = zk_config.quotient_split_mask_profile.ok_or_else(|| {
            VerificationError::InvalidStructure(String::from(
                "Missing ZK quotient split mask profile",
            ))
        })?;
        validate_zk_quotient_split_mask_query_budget(
            profile,
            commitment_scheme.config.fri_config.n_queries,
        )
        .map_err(|_| {
            VerificationError::InvalidStructure(String::from(
                "ZK quotient split mask query budget is too small",
            ))
        })?;
    }

    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .column_log_sizes
        .len();

    let components = Components {
        components: components.to_vec(),
        n_preprocessed_columns,
    };
    let base_column_log_degree_bounds = components.column_log_sizes();
    let zk_degree_profile = derive_zk_stark_degree_bound_profile(
        base_column_log_degree_bounds.clone(),
        components.composition_log_degree_bound(),
        COMPOSITION_LOG_SPLIT,
        zk_config,
        zk_config.metadata.degree_profile.fri_first_layer_log_size,
        commitment_scheme.config.fri_config.log_blowup_factor,
    )
    .map_err(VerificationError::ZkDegreeProfile)?;
    let committed_column_log_sizes = commitment_scheme
        .trees
        .as_ref()
        .map(|tree| tree.column_log_sizes.clone());
    validate_zk_committed_column_log_sizes(
        &committed_column_log_sizes,
        &zk_degree_profile.column_log_degree_bounds,
        commitment_scheme.config.fri_config.log_blowup_factor,
    )
    .map_err(VerificationError::ZkCommittedColumnLogSizes)?;
    let split_composition_log_degree_bound = zk_degree_profile
        .split_composition_log_degree_bounds
        .iter()
        .copied()
        .max()
        .unwrap_or(zk_degree_profile.split_composition_log_degree_bound);
    tracing::info!(
        "ZK split composition polynomial log degree bound: {}",
        split_composition_log_degree_bound
    );

    let lifting_log_size = try_get_lifting_log_size(
        &commitment_scheme.config,
        split_composition_log_degree_bound + commitment_scheme.config.fri_config.log_blowup_factor,
    )?;
    if lifting_log_size != zk_degree_profile.fri_first_layer_log_size {
        return Err(VerificationError::InvalidStructure(String::from(
            "ZK degree profile does not match PCS lifting domain",
        )));
    }
    if include_all_preprocessed_columns {
        let preprocessed_trace_height = commitment_scheme.trees[PREPROCESSED_TRACE_IDX].height;
        if lifting_log_size < preprocessed_trace_height {
            Err(crate::core::pcs::utils::InvalidLiftingLogSizeError {
                lifting_log_size,
                min_log_size: preprocessed_trace_height,
            })?;
        }
    }

    let max_log_degree_bound =
        lifting_log_size - commitment_scheme.config.fri_config.log_blowup_factor;

    let actual_trace_domain_log_size =
        zk_trace_domain_log_size_from_column_bounds(&base_column_log_degree_bounds).ok_or_else(
            || VerificationError::InvalidStructure(String::from("Invalid ZK trace geometry")),
        )?;
    if zk_config.metadata.degree_profile.trace_domain_log_size != actual_trace_domain_log_size {
        return Err(VerificationError::InvalidStructure(String::from(
            "ZK trace domain metadata does not match component trace geometry",
        )));
    }

    mix_zk_public_metadata(
        channel,
        &zk_config.metadata,
        &zk_config.column_degree_bounds,
    );
    if requires_private_stark_activation {
        let profile = zk_config.quotient_split_mask_profile.ok_or_else(|| {
            VerificationError::InvalidStructure(String::from(
                "Missing ZK quotient split mask profile",
            ))
        })?;
        mix_zk_quotient_split_mask_profile(channel, profile).map_err(|_| {
            VerificationError::InvalidStructure(String::from(
                "Invalid ZK quotient split mask profile",
            ))
        })?;
    }
    let random_coeff = channel.draw_secure_felt();

    let Some(&composition_commitment) = proof.0.randomized_pcs_proof.commitments.last() else {
        return Err(VerificationError::InvalidStructure(String::from(
            "ZK proof is missing composition commitment",
        )));
    };
    commitment_scheme.commit(
        composition_commitment,
        &zk_degree_profile.split_composition_log_degree_bounds,
        channel,
    );
    let composition_column_log_sizes = &commitment_scheme.trees.last().unwrap().column_log_sizes;
    validate_zk_composition_column_log_sizes_against_bounds(
        composition_column_log_sizes,
        &zk_degree_profile.split_composition_log_degree_bounds,
        commitment_scheme.config.fri_config.log_blowup_factor,
    )
    .map_err(|_| {
        VerificationError::InvalidStructure(String::from("Invalid ZK composition column log sizes"))
    })?;

    let oods_exclusion_set = zk_oods_exclusion_set(actual_trace_domain_log_size, lifting_log_size)?;
    let zk_committed_column_log_sizes = commitment_scheme
        .trees
        .as_ref()
        .map(|tree| tree.column_log_sizes.clone());
    let requires_lifted_sample_planning = zk_committed_column_log_sizes
        .iter()
        .flatten()
        .any(|&column_log_size| column_log_size < lifting_log_size);
    let (oods_point, sample_points) = if requires_private_stark_activation
        || requires_lifted_sample_planning
    {
        let mut mask_offsets = components
            .mask_offsets(include_all_preprocessed_columns)
            .ok_or_else(|| {
                VerificationError::InvalidStructure(String::from(
                    "Private ZK component does not expose structured mask offsets",
                ))
            })?;
        mask_offsets.push(vec![vec![0]; 2 * SECURE_EXTENSION_DEGREE]);

        let (mut semantic_sample_step_log_sizes, mut semantic_sample_base_doublings) = components
            .semantic_sample_step_log_sizes_and_lifts(include_all_preprocessed_columns)
            .ok_or_else(|| {
                VerificationError::InvalidStructure(String::from(
                    "Private ZK component does not expose semantic sample metadata",
                ))
            })?;
        semantic_sample_step_log_sizes.push(vec![
            vec![max_log_degree_bound];
            2 * SECURE_EXTENSION_DEGREE
        ]);
        semantic_sample_base_doublings.push(vec![vec![0]; 2 * SECURE_EXTENSION_DEGREE]);

        let sample_point_plan = draw_zk_oods_sample_point_plan_with_semantic_sample_domains(
            channel,
            &mask_offsets,
            &semantic_sample_step_log_sizes,
            &semantic_sample_base_doublings,
            &zk_committed_column_log_sizes,
            lifting_log_size,
            &oods_exclusion_set,
            64,
        )
        .map_err(VerificationError::ZkOodsSamplePointPlan)?;

        (
            sample_point_plan.semantic_oods_point,
            sample_point_plan.pcs_sample_points,
        )
    } else {
        let oods_point = draw_zk_oods_point(channel, &oods_exclusion_set, 64).map_err(|_| {
            VerificationError::InvalidStructure(String::from(
                "Could not sample a valid ZK OODS point",
            ))
        })?;
        let mut sample_points = components.mask_points(
            oods_point,
            max_log_degree_bound,
            include_all_preprocessed_columns,
        );
        let composition_sample_points = vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE];
        sample_points.push(composition_sample_points);

        (oods_point, sample_points)
    };

    let sample_points_by_column = sample_points.as_cols_ref().flatten();
    tracing::info!("Sampling {} ZK columns.", sample_points_by_column.len());
    tracing::info!(
        "Total ZK sample points: {}.",
        sample_points_by_column.into_iter().flatten().count()
    );

    validate_zk_sample_points_outside_exclusion_set(&sample_points, &oods_exclusion_set)
        .map_err(VerificationError::ZkOodsSamplePoint)?;

    validate_zk_sampled_values_shape(&sample_points, &proof.0.randomized_pcs_proof.sampled_values)
        .map_err(|_| {
            VerificationError::InvalidStructure(String::from(
                "Unexpected ZK sampled_values structure",
            ))
        })?;

    let composition_oods_eval = proof
        .extract_composition_oods_eval(oods_point, zk_degree_profile.composition_log_degree_bound)
        .ok_or(VerificationError::InvalidStructure(
            std_shims::ToString::to_string(&"Unexpected ZK sampled_values structure"),
        ))?;

    let component_composition_eval = if requires_private_stark_activation {
        components.eval_zk_composition_polynomial_at_point(
            oods_point,
            &proof.0.randomized_pcs_proof.sampled_values,
            random_coeff,
            max_log_degree_bound,
        )
    } else {
        components.eval_composition_polynomial_at_point(
            oods_point,
            &proof.0.randomized_pcs_proof.sampled_values,
            random_coeff,
            max_log_degree_bound,
        )
    };
    if composition_oods_eval != component_composition_eval {
        return Err(VerificationError::OodsNotMatching);
    }

    if let Some(witness_randomization_audit) = witness_randomization_audit {
        commitment_scheme.verify_values_zk_with_witness_randomization_audit(
            sample_points,
            proof.0,
            zk_config,
            witness_randomization_audit,
            channel,
        )
    } else {
        commitment_scheme.verify_values_zk(sample_points, proof.0, zk_config, channel)
    }
}

#[derive(Clone, Debug, Error)]
pub enum VerificationError {
    #[error("Proof has invalid structure: {0}.")]
    InvalidStructure(String),
    #[error(transparent)]
    Merkle(#[from] MerkleVerificationError),
    #[error(
        "The composition polynomial OODS value does not match the trace OODS values
    (DEEP-ALI failure)."
    )]
    OodsNotMatching,
    #[error("Invalid ZK STARK degree profile: {0:?}.")]
    ZkDegreeProfile(ZkStarkDegreeBoundProfileError),
    #[error("Invalid ZK committed column log sizes: {0:?}.")]
    ZkCommittedColumnLogSizes(ZkCommittedColumnLogSizeValidationError),
    #[error("Invalid ZK OODS sample point: {0:?}.")]
    ZkOodsSamplePoint(ZkOodsSamplePointValidationError),
    #[error("Invalid ZK OODS sample point plan: {0:?}.")]
    ZkOodsSamplePointPlan(ZkOodsSamplePointPlanningError),
    #[error(transparent)]
    Fri(#[from] FriVerificationError),
    #[error("Proof of work verification failed.")]
    ProofOfWork,
    #[error(transparent)]
    InvalidLiftingLogSize(#[from] crate::core::pcs::utils::InvalidLiftingLogSizeError),
    #[error(transparent)]
    InvalidCanonicCosetLogSize(#[from] crate::core::poly::circle::InvalidCanonicCosetLogSize),
}
