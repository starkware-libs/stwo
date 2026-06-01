use rand::{CryptoRng, RngCore};
use thiserror::Error;
use tracing::{info, instrument, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::pcs::utils::{try_get_lifting_log_size, InvalidLiftingLogSizeError};
use crate::core::proof::{ExtendedStarkProof, StarkProof};
use crate::core::verifier::{COMPOSITION_LOG_SPLIT, PREPROCESSED_TRACE_IDX};
use crate::core::zk::{
    derive_zk_stark_degree_bound_profile, draw_zk_oods_point, mix_zk_public_metadata,
    validate_zk_committed_column_log_sizes, validate_zk_composition_column_log_sizes,
    validate_zk_sample_points_outside_exclusion_set, zk_metadata_requires_private_stark_activation,
    zk_oods_exclusion_set, ExtendedZkStarkProof, ZkCommittedColumnLogSizeValidationError,
    ZkOodsSamplePointValidationError, ZkOodsSamplingError, ZkStarkDegreeBoundProfileError,
    ZkStarkProof, ZkVerificationConfig,
};
use crate::prover::backend::BackendForChannel;
use crate::prover::zk::{ZkProvingConfig, ZkProvingConfigError};

mod air;
pub use air::component_prover::{ComponentProver, ComponentProvers, Poly, Trace};
pub use air::{AccumulationOps, ColumnAccumulator, DomainEvaluationAccumulator, EvaluationMode};
pub mod pcs;
pub use pcs::quotient_ops::QuotientOps;
pub use pcs::{CommitmentSchemeProver, CommitmentTreeProver, TreeBuilder};
pub mod backend;
pub mod channel;
pub mod fri;
pub mod line;
pub mod lookups;
pub mod mempool;
pub mod poly;
pub mod secure_column;
pub mod vcs;
pub mod vcs_lifted;
pub mod zk;

pub fn prove<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
) -> Result<StarkProof<MC::H>, ProvingError> {
    Ok(prove_ex(components, channel, commitment_scheme, false)?.proof)
}

#[instrument(skip_all)]
pub fn prove_ex<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    mut commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
    include_all_preprocessed_columns: bool,
) -> Result<ExtendedStarkProof<MC::H>, ProvingError> {
    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let component_provers = ComponentProvers {
        components: components.to_vec(),
        n_preprocessed_columns,
    };
    let trace = commitment_scheme.trace();

    // Evaluate and commit on composition polynomial.
    let random_coeff = channel.draw_secure_felt();

    let span = span!(Level::INFO, "Composition", class = "Composition").entered();
    let span1 = span!(
        Level::INFO,
        "Generation",
        class = "CompositionPolynomialGeneration"
    )
    .entered();

    let composition_poly = component_provers.compute_composition_polynomial(
        random_coeff,
        &trace,
        commitment_scheme.twiddles,
        commitment_scheme.config.fri_config.log_blowup_factor,
    );
    span1.exit();

    // Commit on the Composition Polynomial by splitting its coeffs to two polynomialsof degree
    // half the size of the original polynomial, and commit on each half separately.
    let mut tree_builder = commitment_scheme.tree_builder();
    let (left_comp_poly_half, right_comp_poly_half) = composition_poly.split_at_mid();

    tree_builder.extend_polys(left_comp_poly_half.into_coordinate_polys());
    tree_builder.extend_polys(right_comp_poly_half.into_coordinate_polys());
    tree_builder.commit(channel);
    span.exit();

    // Draw OODS point.
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);

    let split_composition_log_size = commitment_scheme
        .trees
        .last()
        .unwrap()
        .commitment
        .layers
        .len() as u32
        - 1;

    // If `self.config.lifting_log_size` is None, the lifting size is the length of the split
    // composition polynomials' domain.
    let lifting_log_size =
        try_get_lifting_log_size(&commitment_scheme.config, split_composition_log_size)?;
    if include_all_preprocessed_columns {
        // If all the preprocessed columns are included, the lifting log size must be greater than
        // or equal to the preprocessed log size.
        let preprocessed_log_size = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
            .commitment
            .layers
            .len() as u32
            - 1;
        if lifting_log_size < preprocessed_log_size {
            Err(InvalidLiftingLogSizeError {
                lifting_log_size,
                min_log_size: preprocessed_log_size,
            })?;
        }
    }
    let max_log_degree_bound =
        lifting_log_size - commitment_scheme.config.fri_config.log_blowup_factor;

    // Get mask sample points relative to oods point.
    let mut sample_points = component_provers.components().mask_points(
        oods_point,
        max_log_degree_bound,
        include_all_preprocessed_columns,
    );

    // Add the composition polynomial mask points.
    sample_points.push(vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE]);

    // Prove the trace and composition OODS values, and retrieve them.
    let commitment_scheme_proof = commitment_scheme.prove_values(sample_points, channel);
    let proof = StarkProof(commitment_scheme_proof.proof);
    info!(proof_size_estimate = proof.size_estimate());

    // Evaluate composition polynomial at OODS point and check that it matches the trace OODS
    // values. This is a sanity check.
    if proof
        .extract_composition_oods_eval(oods_point, max_log_degree_bound)
        .unwrap()
        != component_provers
            .components()
            .eval_composition_polynomial_at_point(
                oods_point,
                &proof.sampled_values,
                random_coeff,
                max_log_degree_bound,
            )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(ExtendedStarkProof {
        proof,
        aux: commitment_scheme_proof.aux,
    })
}

/// Convenience wrapper for [`prove_zk_ex`] using the default preprocessed-column
/// sampling policy.
///
/// This is an explicit ZK API. It does not change [`prove`]. At this stage,
/// STARK-level private witness mode is fail-closed until the reviewed Phase 3
/// degree/OODS integration is wired; public-only FRI batch masking remains
/// available through this API.
pub fn prove_zk<B, MC, R>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
    zk_config: &ZkProvingConfig,
    rng: &mut R,
) -> Result<ZkStarkProof<MC::H>, ProvingError>
where
    B: BackendForChannel<MC>,
    MC: MerkleChannel,
    R: RngCore + CryptoRng + ?Sized,
{
    Ok(prove_zk_ex(
        components,
        channel,
        commitment_scheme,
        zk_config,
        rng,
        false,
    )?
    .proof)
}

/// Produces an explicit ZK STARK proof using the paper-aligned PCS ZK path.
///
/// This API does not alter the default [`prove_ex`] transcript or proof format.
/// It currently rejects private witness metadata until the Phase 3
/// STARK-level degree and OODS derivation is implemented.
#[instrument(skip_all)]
pub fn prove_zk_ex<B, MC, R>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    mut commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
    zk_config: &ZkProvingConfig,
    rng: &mut R,
    include_all_preprocessed_columns: bool,
) -> Result<ExtendedZkStarkProof<MC::H>, ProvingError>
where
    B: BackendForChannel<MC>,
    MC: MerkleChannel,
    R: RngCore + CryptoRng + ?Sized,
{
    if !zk_config.privacy_map.private_columns.is_empty()
        || zk_metadata_requires_private_stark_activation(&zk_config.metadata)
    {
        zk_config
            .validate_witness_and_quotient_pre_activation()
            .map_err(ProvingError::ZkConfig)?;
        return Err(ProvingError::ZkConfig(
            ZkProvingConfigError::Phase2And3ActivationBlocked,
        ));
    }

    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let component_provers = ComponentProvers {
        components: components.to_vec(),
        n_preprocessed_columns,
    };

    let actual_trace_domain_log_size = commitment_scheme
        .trees
        .iter()
        .flat_map(|tree| tree.polynomials.iter())
        .map(|poly| {
            poly.evals
                .domain
                .log_size()
                .checked_sub(commitment_scheme.config.fri_config.log_blowup_factor)
        })
        .collect::<Option<Vec<_>>>()
        .and_then(|log_sizes| log_sizes.into_iter().max())
        .ok_or(ProvingError::InvalidZkTraceGeometry)?;
    if zk_config.metadata.degree_profile.trace_domain_log_size != actual_trace_domain_log_size {
        return Err(ProvingError::InvalidZkTraceGeometry);
    }
    let zk_verification_config = ZkVerificationConfig {
        metadata: zk_config.metadata.clone(),
        column_degree_bounds: zk_config.column_degree_bounds.clone(),
    };
    let zk_degree_profile = derive_zk_stark_degree_bound_profile(
        component_provers.components().column_log_sizes(),
        component_provers
            .components()
            .composition_log_degree_bound(),
        COMPOSITION_LOG_SPLIT,
        &zk_verification_config,
        zk_config.metadata.degree_profile.fri_first_layer_log_size,
        commitment_scheme.config.fri_config.log_blowup_factor,
    )
    .map_err(ProvingError::ZkDegreeProfile)?;
    let committed_column_log_sizes = commitment_scheme.trees.as_ref().map(|tree| {
        tree.polynomials
            .iter()
            .map(|poly| poly.evals.domain.log_size())
            .collect()
    });
    validate_zk_committed_column_log_sizes(
        &committed_column_log_sizes,
        &zk_degree_profile.column_log_degree_bounds,
        commitment_scheme.config.fri_config.log_blowup_factor,
    )
    .map_err(ProvingError::ZkCommittedColumnLogSizes)?;

    let trace = commitment_scheme.trace();
    zk_config
        .validate_for_fri_batch_mask_only(
            zk_config.metadata.degree_profile.fri_first_layer_log_size,
            commitment_scheme.config.fri_config.log_blowup_factor,
        )
        .map_err(ProvingError::ZkConfig)?;
    mix_zk_public_metadata(
        channel,
        &zk_config.metadata,
        &zk_config.column_degree_bounds,
    );

    // Evaluate and commit on composition polynomial. In the explicit ZK path,
    // `commitment_scheme.trace()` reflects any prior ZK-randomized private
    // witness tree commits.
    let random_coeff = channel.draw_secure_felt();

    let span = span!(Level::INFO, "ZK Composition", class = "Composition").entered();
    let span1 = span!(
        Level::INFO,
        "ZK Generation",
        class = "CompositionPolynomialGeneration"
    )
    .entered();

    let composition_poly = component_provers.compute_composition_polynomial_with_log_degree_bound(
        random_coeff,
        &trace,
        commitment_scheme.twiddles,
        commitment_scheme.config.fri_config.log_blowup_factor,
        zk_degree_profile.trace_log_degree_bound,
        &zk_degree_profile.column_log_degree_bounds,
        zk_degree_profile.composition_log_degree_bound,
    )?;
    span1.exit();

    let mut tree_builder = commitment_scheme.tree_builder();
    let (left_comp_poly_half, right_comp_poly_half) = composition_poly.split_at_mid();

    tree_builder.extend_polys(left_comp_poly_half.into_coordinate_polys());
    tree_builder.extend_polys(right_comp_poly_half.into_coordinate_polys());
    tree_builder.commit(channel);
    let composition_column_log_sizes = commitment_scheme
        .trees
        .last()
        .unwrap()
        .polynomials
        .iter()
        .map(|poly| poly.evals.domain.log_size())
        .collect::<Vec<_>>();
    validate_zk_composition_column_log_sizes(
        &composition_column_log_sizes,
        2 * SECURE_EXTENSION_DEGREE,
        zk_degree_profile.split_composition_log_degree_bound,
        commitment_scheme.config.fri_config.log_blowup_factor,
    )
    .map_err(|_| ProvingError::InvalidZkDegreeGeometry)?;
    span.exit();

    let split_composition_log_size = commitment_scheme
        .trees
        .last()
        .unwrap()
        .commitment
        .layers
        .len() as u32
        - 1;

    let lifting_log_size =
        try_get_lifting_log_size(&commitment_scheme.config, split_composition_log_size)?;
    if lifting_log_size != zk_degree_profile.fri_first_layer_log_size {
        return Err(ProvingError::InvalidZkDegreeGeometry);
    }
    if include_all_preprocessed_columns {
        let preprocessed_log_size = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
            .commitment
            .layers
            .len() as u32
            - 1;
        if lifting_log_size < preprocessed_log_size {
            Err(InvalidLiftingLogSizeError {
                lifting_log_size,
                min_log_size: preprocessed_log_size,
            })?;
        }
    }
    let max_log_degree_bound =
        lifting_log_size - commitment_scheme.config.fri_config.log_blowup_factor;

    zk_config
        .validate_for_fri_batch_mask_only(
            lifting_log_size,
            commitment_scheme.config.fri_config.log_blowup_factor,
        )
        .map_err(ProvingError::ZkConfig)?;
    let oods_exclusion_set = zk_oods_exclusion_set(actual_trace_domain_log_size, lifting_log_size)?;
    let oods_point = draw_zk_oods_point(channel, &oods_exclusion_set, 64)
        .map_err(ProvingError::ZkOodsSampling)?;

    let mut sample_points = component_provers.components().mask_points(
        oods_point,
        max_log_degree_bound,
        include_all_preprocessed_columns,
    );
    sample_points.push(vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE]);
    validate_zk_sample_points_outside_exclusion_set(&sample_points, &oods_exclusion_set)
        .map_err(ProvingError::ZkOodsSamplePoint)?;

    let commitment_scheme_proof = commitment_scheme
        .prove_values_zk(sample_points, zk_config, rng, channel)
        .map_err(ProvingError::ZkConfig)?;
    let proof = ZkStarkProof(commitment_scheme_proof.proof);

    if proof
        .extract_composition_oods_eval(oods_point, max_log_degree_bound)
        .unwrap()
        != component_provers
            .components()
            .eval_composition_polynomial_at_point(
                oods_point,
                &proof.0.randomized_pcs_proof.sampled_values,
                random_coeff,
                max_log_degree_bound,
            )
    {
        return Err(ProvingError::ConstraintsNotSatisfied);
    }

    Ok(ExtendedZkStarkProof {
        proof,
        aux: commitment_scheme_proof.aux,
    })
}

#[derive(Clone, Copy, Debug, Error)]
pub enum ProvingError {
    #[error("Constraints not satisfied.")]
    ConstraintsNotSatisfied,
    #[error("Invalid ZK proving config: {0:?}.")]
    ZkConfig(ZkProvingConfigError),
    #[error("Could not sample a valid ZK OODS point: {0:?}.")]
    ZkOodsSampling(ZkOodsSamplingError),
    #[error("Invalid ZK OODS sample point: {0:?}.")]
    ZkOodsSamplePoint(ZkOodsSamplePointValidationError),
    #[error("Invalid ZK STARK degree profile: {0:?}.")]
    ZkDegreeProfile(ZkStarkDegreeBoundProfileError),
    #[error("Invalid ZK committed column log sizes: {0:?}.")]
    ZkCommittedColumnLogSizes(ZkCommittedColumnLogSizeValidationError),
    #[error("Invalid ZK trace geometry.")]
    InvalidZkTraceGeometry,
    #[error("Invalid ZK degree geometry.")]
    InvalidZkDegreeGeometry,
    #[error(transparent)]
    InvalidLiftingLogSize(#[from] crate::core::pcs::utils::InvalidLiftingLogSizeError),
    #[error(transparent)]
    InvalidCanonicCosetLogSize(#[from] crate::core::poly::circle::InvalidCanonicCosetLogSize),
}
