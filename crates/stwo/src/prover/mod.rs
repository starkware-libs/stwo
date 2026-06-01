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
    derive_zk_stark_degree_bound_profile, draw_zk_oods_point, draw_zk_oods_point_with_preimage,
    mix_zk_public_metadata, mix_zk_quotient_split_mask_profile,
    validate_zk_committed_column_log_sizes,
    validate_zk_composition_column_log_sizes_against_bounds,
    validate_zk_sample_points_outside_exclusion_set, validate_zk_witness_metadata,
    zk_metadata_requires_private_stark_activation, zk_oods_exclusion_set,
    zk_trace_domain_log_size_from_column_bounds, ExtendedZkStarkProof,
    ZkCommittedColumnLogSizeValidationError, ZkOodsSamplePointValidationError, ZkOodsSamplingError,
    ZkStarkDegreeBoundProfileError, ZkStarkProof, ZkVerificationConfig,
};
use crate::prover::backend::BackendForChannel;
use crate::prover::poly::circle::{CircleCoefficients, PolyOps, SecureCirclePoly};
use crate::prover::zk::{
    mask_composition_split_pair_from_prover_rng, ZkProvingConfig, ZkProvingConfigError,
    ZkQuotientSplitMaskError,
};

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

type SecureCoordinatePolys<B> = [CircleCoefficients<B>; SECURE_EXTENSION_DEGREE];

fn zk_composition_split_coordinate_polys<B, R>(
    left_comp_poly_half: SecureCirclePoly<B>,
    right_comp_poly_half: SecureCirclePoly<B>,
    zk_config: &ZkProvingConfig,
    rng: &mut R,
) -> Result<(SecureCoordinatePolys<B>, SecureCoordinatePolys<B>), ProvingError>
where
    B: PolyOps,
    R: RngCore + CryptoRng + ?Sized,
{
    let requires_private_witness_integration = !zk_config.privacy_map.private_columns.is_empty()
        || zk_metadata_requires_private_stark_activation(&zk_config.metadata);
    if requires_private_witness_integration {
        let profile = zk_config
            .quotient_split_mask_profile
            .ok_or(ProvingError::ZkConfig(
                ZkProvingConfigError::MissingQuotientSplitMaskProfile,
            ))?;
        let masked = mask_composition_split_pair_from_prover_rng(
            left_comp_poly_half,
            right_comp_poly_half,
            profile,
            rng,
        )
        .map_err(ProvingError::ZkQuotientSplitMask)?;

        return Ok((
            masked.left_hat.into_coordinate_polys(),
            masked.right_hat.into_coordinate_polys(),
        ));
    }

    Ok((
        left_comp_poly_half.into_coordinate_polys(),
        right_comp_poly_half.into_coordinate_polys(),
    ))
}

fn validate_zk_stark_proving_config(
    zk_config: &ZkProvingConfig,
    lifting_log_size: u32,
    log_blowup_factor: u32,
    n_queries: usize,
) -> Result<(), ZkProvingConfigError> {
    let requires_private_witness_integration = !zk_config.privacy_map.private_columns.is_empty()
        || zk_metadata_requires_private_stark_activation(&zk_config.metadata);
    if requires_private_witness_integration {
        zk_config.validate_witness_and_quotient_static_config()?;
        zk_config.validate_quotient_split_mask_query_budget(n_queries)?;
        validate_zk_witness_metadata(&zk_config.metadata, lifting_log_size, log_blowup_factor)
            .map_err(ZkProvingConfigError::Metadata)
    } else {
        zk_config.validate_for_fri_batch_mask_only(lifting_log_size, log_blowup_factor)
    }
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
/// private witness mode requires verifier-owned metadata, randomized witness
/// commitments, quotient split masking, and a verifier audit. Public-only FRI
/// batch masking remains available through the same explicit ZK API.
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
/// Private witness metadata requires randomized witness commitments and the
/// reviewed quotient split masking path.
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
    let requires_private_witness_integration = !zk_config.privacy_map.private_columns.is_empty()
        || zk_metadata_requires_private_stark_activation(&zk_config.metadata);
    if requires_private_witness_integration {
        zk_config
            .validate_witness_and_quotient_static_config()
            .map_err(ProvingError::ZkConfig)?;
        if commitment_scheme.config.lifting_log_size
            != Some(zk_config.metadata.degree_profile.fri_first_layer_log_size)
        {
            return Err(ProvingError::InvalidZkDegreeGeometry);
        }
    }

    let n_preprocessed_columns = commitment_scheme.trees[PREPROCESSED_TRACE_IDX]
        .polynomials
        .len();
    let component_provers = ComponentProvers {
        components: components.to_vec(),
        n_preprocessed_columns,
    };

    let base_column_log_degree_bounds = component_provers.components().column_log_sizes();
    let actual_trace_domain_log_size =
        zk_trace_domain_log_size_from_column_bounds(&base_column_log_degree_bounds)
            .ok_or(ProvingError::InvalidZkTraceGeometry)?;
    if zk_config.metadata.degree_profile.trace_domain_log_size != actual_trace_domain_log_size {
        return Err(ProvingError::InvalidZkTraceGeometry);
    }
    let zk_verification_config = ZkVerificationConfig {
        metadata: zk_config.metadata.clone(),
        column_degree_bounds: zk_config.column_degree_bounds.clone(),
        quotient_split_mask_profile: zk_config.quotient_split_mask_profile,
    };
    let zk_degree_profile = derive_zk_stark_degree_bound_profile(
        base_column_log_degree_bounds,
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
    validate_zk_stark_proving_config(
        zk_config,
        zk_config.metadata.degree_profile.fri_first_layer_log_size,
        commitment_scheme.config.fri_config.log_blowup_factor,
        commitment_scheme.config.fri_config.n_queries,
    )
    .map_err(ProvingError::ZkConfig)?;
    mix_zk_public_metadata(
        channel,
        &zk_config.metadata,
        &zk_config.column_degree_bounds,
    );
    if requires_private_witness_integration {
        mix_zk_quotient_split_mask_profile(
            channel,
            zk_config
                .quotient_split_mask_profile
                .ok_or(ProvingError::ZkConfig(
                    ZkProvingConfigError::MissingQuotientSplitMaskProfile,
                ))?,
        )
        .map_err(|err| {
            ProvingError::ZkConfig(ZkProvingConfigError::QuotientSplitMaskProfile(err))
        })?;
    }

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
    let (left_comp_poly_half, right_comp_poly_half) = zk_composition_split_coordinate_polys(
        left_comp_poly_half,
        right_comp_poly_half,
        zk_config,
        rng,
    )?;

    tree_builder.extend_polys(left_comp_poly_half);
    tree_builder.extend_polys(right_comp_poly_half);
    tree_builder.commit(channel);
    let composition_column_log_sizes = commitment_scheme
        .trees
        .last()
        .unwrap()
        .polynomials
        .iter()
        .map(|poly| poly.evals.domain.log_size())
        .collect::<Vec<_>>();
    validate_zk_composition_column_log_sizes_against_bounds(
        &composition_column_log_sizes,
        &zk_degree_profile.split_composition_log_degree_bounds,
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

    validate_zk_stark_proving_config(
        zk_config,
        lifting_log_size,
        commitment_scheme.config.fri_config.log_blowup_factor,
        commitment_scheme.config.fri_config.n_queries,
    )
    .map_err(ProvingError::ZkConfig)?;
    let oods_exclusion_set = zk_oods_exclusion_set(actual_trace_domain_log_size, lifting_log_size)?;
    let (oods_point, composition_right_sample_point) = if requires_private_witness_integration {
        draw_zk_oods_point_with_preimage(channel, &oods_exclusion_set, 64)
            .map_err(ProvingError::ZkOodsSampling)?
    } else {
        let point = draw_zk_oods_point(channel, &oods_exclusion_set, 64)
            .map_err(ProvingError::ZkOodsSampling)?;
        (point, point)
    };

    let mut sample_points = component_provers.components().mask_points(
        oods_point,
        max_log_degree_bound,
        include_all_preprocessed_columns,
    );
    let mut composition_sample_points = vec![vec![oods_point]; 2 * SECURE_EXTENSION_DEGREE];
    if requires_private_witness_integration {
        for column_points in &mut composition_sample_points[SECURE_EXTENSION_DEGREE..] {
            *column_points = vec![composition_right_sample_point];
        }
    }
    sample_points.push(composition_sample_points);
    validate_zk_sample_points_outside_exclusion_set(&sample_points, &oods_exclusion_set)
        .map_err(ProvingError::ZkOodsSamplePoint)?;

    let commitment_scheme_proof = commitment_scheme
        .prove_values_zk(sample_points, zk_config, rng, channel)
        .map_err(ProvingError::ZkConfig)?;
    let proof = ZkStarkProof(commitment_scheme_proof.proof);

    if proof
        .extract_composition_oods_eval(oods_point, zk_degree_profile.composition_log_degree_bound)
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
    #[error("Invalid ZK quotient split mask: {0:?}.")]
    ZkQuotientSplitMask(ZkQuotientSplitMaskError),
    #[error("Invalid ZK trace geometry.")]
    InvalidZkTraceGeometry,
    #[error("Invalid ZK degree geometry.")]
    InvalidZkDegreeGeometry,
    #[error(transparent)]
    InvalidLiftingLogSize(#[from] crate::core::pcs::utils::InvalidLiftingLogSizeError),
    #[error(transparent)]
    InvalidCanonicCosetLogSize(#[from] crate::core::poly::circle::InvalidCanonicCosetLogSize),
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::core::air::accumulation::PointEvaluationAccumulator;
    use crate::core::air::Component;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::M31;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::zk::{
        canonical_zk_private_column_scope_hash, canonical_zk_randomizer_space_hash,
        canonical_zk_split_derivation_hash, expected_zk_fri_batch_degree_bound,
        zk_trace_domain_half_coset, ZkCircleCosetEncoding, ZkColumnDegreeBound, ZkColumnRange,
        ZkDegreeProfile, ZkPrivacyMap, ZkPrivacyMapHash, ZkPrivateColumnScope,
        ZkPrivateColumnScopeEntry, ZkPrivateColumnUsage, ZkProofVersion, ZkPublicMetadata,
        ZkPublicStatementHash, ZkQuotientIntegrationProfile, ZkRandomizerSpaceEntry,
        ZkVerificationConfig, ZkWitnessRandomizationProfile, ZkWitnessRandomizationVerifierAudit,
    };
    use crate::core::ColumnVec;
    use crate::prover::backend::cpu::CpuBackend;
    use crate::prover::poly::circle::{CircleCoefficients, PolyOps};
    use crate::prover::zk::{ZkDerivationGate, ZkDerivationReview};

    const TEST_TRACE_LOG_SIZE: u32 = 5;
    const TEST_RANDOMIZED_LOG_DEGREE: u32 = TEST_TRACE_LOG_SIZE + 1;
    const TEST_FRI_FIRST_LAYER_LOG_SIZE: u32 = TEST_RANDOMIZED_LOG_DEGREE + 2;

    struct NoConstraintPrivateComponent;

    impl Component for NoConstraintPrivateComponent {
        fn n_constraints(&self) -> usize {
            0
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            TEST_RANDOMIZED_LOG_DEGREE + 1
        }

        fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
            TreeVec(vec![vec![TEST_TRACE_LOG_SIZE], vec![TEST_TRACE_LOG_SIZE]])
        }

        fn mask_points(
            &self,
            point: CirclePoint<SecureField>,
            _max_log_degree_bound: u32,
        ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
            TreeVec(vec![vec![vec![point]], vec![vec![point]]])
        }

        fn preprocessed_column_indices(&self) -> ColumnVec<usize> {
            vec![0]
        }

        fn evaluate_constraint_quotients_at_point(
            &self,
            _point: CirclePoint<SecureField>,
            _mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
            _evaluation_accumulator: &mut PointEvaluationAccumulator,
            _max_log_degree_bound: u32,
        ) {
        }
    }

    impl ComponentProver<CpuBackend> for NoConstraintPrivateComponent {
        fn evaluate_constraint_quotients_on_domain(
            &self,
            _trace: &Trace<'_, CpuBackend>,
            _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        ) {
        }

        fn evaluate_constraint_quotients_on_domain_with_log_degree_bound(
            &self,
            _trace: &Trace<'_, CpuBackend>,
            _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
            _max_constraint_log_degree_bound: u32,
        ) -> Result<(), ProvingError> {
            Ok(())
        }
    }

    fn test_hash(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn test_derivation_reviews() -> Vec<ZkDerivationReview> {
        [
            ZkDerivationGate::StwoSplitQueryExpansion,
            ZkDerivationGate::CircleRandomizerSpace,
            ZkDerivationGate::OodsDomainExclusion,
            ZkDerivationGate::ZkAwareDegreeMetadata,
            ZkDerivationGate::FriBatchMaskDegree,
            ZkDerivationGate::PrivateLookupPermutationExclusion,
            ZkDerivationGate::ProofDataSecrecy,
            ZkDerivationGate::ZkPerformanceControls,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, gate)| ZkDerivationReview {
            gate,
            review_hash: test_hash(index as u8 + 10),
        })
        .collect()
    }

    fn private_stark_test_configs() -> (
        ZkProvingConfig,
        ZkVerificationConfig,
        ZkWitnessRandomizationVerifierAudit,
    ) {
        let private_range = ZkColumnRange::new(1, 0, 1);
        let quotient_range = ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE);
        let privacy_map_hash = ZkPrivacyMapHash(test_hash(1));
        let mut private_column_scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: [0; 32],
            entries: vec![ZkPrivateColumnScopeEntry {
                range: private_range,
                usage: ZkPrivateColumnUsage::OrdinaryWitness,
            }],
        };
        let private_column_scope_hash =
            canonical_zk_private_column_scope_hash(&private_column_scope);
        private_column_scope.hash = private_column_scope_hash;
        let trace_domain = CanonicCoset::new(TEST_TRACE_LOG_SIZE).coset;
        let h_witness = 1u64 << TEST_TRACE_LOG_SIZE;
        let randomizer_space_hash = canonical_zk_randomizer_space_hash(
            private_column_scope_hash,
            &[ZkRandomizerSpaceEntry {
                range: private_range,
                trace_domain: ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(trace_domain)),
                randomized_log_degree: TEST_RANDOMIZED_LOG_DEGREE,
                randomizer_dimension: h_witness,
            }],
        );
        let private_degree_bound = ZkColumnDegreeBound {
            range: private_range,
            log_degree_bound: TEST_RANDOMIZED_LOG_DEGREE,
        };
        let quotient_degree_bound = ZkColumnDegreeBound {
            range: quotient_range,
            log_degree_bound: TEST_FRI_FIRST_LAYER_LOG_SIZE - 2,
        };
        let h_batch = expected_zk_fri_batch_degree_bound(TEST_FRI_FIRST_LAYER_LOG_SIZE, 1)
            .expect("test FRI layer has a valid batch-mask degree");
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash,
            public_statement_hash: ZkPublicStatementHash(test_hash(2)),
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: TEST_TRACE_LOG_SIZE,
                h_witness,
                h_batch,
                fri_first_layer_log_size: TEST_FRI_FIRST_LAYER_LOG_SIZE,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness,
                randomizer_space_hash,
                private_column_scope_hash,
                private_column_degree_bounds: vec![private_degree_bound],
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch,
                fri_first_layer_log_size: TEST_FRI_FIRST_LAYER_LOG_SIZE,
                split_derivation_hash: canonical_zk_split_derivation_hash(COMPOSITION_LOG_SPLIT),
                quotient_degree_bounds: vec![quotient_degree_bound],
            },
        };
        let column_degree_bounds = vec![private_degree_bound, quotient_degree_bound];
        let quotient_split_mask_profile =
            crate::core::zk::stwo_composition_quotient_split_mask_profile(
                quotient_degree_bound.range.tree_index,
                quotient_degree_bound.log_degree_bound + 1,
                quotient_degree_bound.log_degree_bound,
                1u64 << quotient_degree_bound.log_degree_bound,
                quotient_degree_bound.log_degree_bound + 1,
                quotient_degree_bound.log_degree_bound,
            )
            .expect("test quotient split mask profile must be valid");
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![private_range],
            hash: privacy_map_hash,
        };
        let prover_config = ZkProvingConfig {
            metadata: metadata.clone(),
            privacy_map: privacy_map.clone(),
            private_column_scope: Some(private_column_scope.clone()),
            quotient_split_mask_profile: Some(quotient_split_mask_profile),
            query_closure: None,
            randomizer_rank_profile: None,
            derived_randomizer_metadata: None,
            column_degree_bounds: column_degree_bounds.clone(),
            derivation_reviews: test_derivation_reviews(),
        };
        let verifier_config = ZkVerificationConfig {
            metadata,
            column_degree_bounds,
            quotient_split_mask_profile: Some(quotient_split_mask_profile),
        };
        let verifier_audit = ZkWitnessRandomizationVerifierAudit {
            privacy_map,
            private_column_scope,
        };

        (prover_config, verifier_config, verifier_audit)
    }

    fn private_stark_test_pcs_config() -> PcsConfig {
        PcsConfig {
            fri_config: crate::core::fri::FriConfig::new(1, 1, 3, 1),
            lifting_log_size: Some(TEST_FRI_FIRST_LAYER_LOG_SIZE),
            ..PcsConfig::default()
        }
    }

    fn private_stark_test_preprocessed_column() -> CircleCoefficients<CpuBackend> {
        CircleCoefficients::new(
            (0..1 << TEST_TRACE_LOG_SIZE)
                .map(|value| M31::from(7_000 + value as u32))
                .collect(),
        )
    }

    fn private_stark_test_witness_column() -> CircleCoefficients<CpuBackend> {
        CircleCoefficients::new(
            (0..1 << TEST_TRACE_LOG_SIZE)
                .map(|value| M31::from(value as u32))
                .collect(),
        )
    }

    fn commit_private_stark_test_inputs(
        commitment_scheme: &mut CommitmentSchemeProver<'_, CpuBackend, Blake2sMerkleChannel>,
        prover_channel: &mut Blake2sChannel,
        zk_prover_config: &crate::prover::zk::ZkProvingConfig,
        witness_rng_seed: u64,
    ) {
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(vec![private_stark_test_preprocessed_column()]);
        tree_builder.commit(prover_channel);

        let mut witness_rng = StdRng::seed_from_u64(witness_rng_seed);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(vec![private_stark_test_witness_column()]);
        tree_builder
            .commit_zk_witness_randomized(zk_prover_config, &mut witness_rng, prover_channel)
            .unwrap();
    }

    fn verifier_for_private_stark_test_proof(
        config: PcsConfig,
        proof: &ZkStarkProof<<Blake2sMerkleChannel as crate::core::channel::MerkleChannel>::H>,
    ) -> (
        Blake2sChannel,
        CommitmentSchemeVerifier<Blake2sMerkleChannel>,
    ) {
        let mut verifier_channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(
            proof.0.randomized_pcs_proof.commitments[0],
            &[TEST_TRACE_LOG_SIZE],
            &mut verifier_channel,
        );
        verifier.commit(
            proof.0.randomized_pcs_proof.commitments[1],
            &[TEST_RANDOMIZED_LOG_DEGREE],
            &mut verifier_channel,
        );

        (verifier_channel, verifier)
    }

    fn prove_private_stark_test_proof(
        witness_rng_seed: u64,
        proof_rng_seed: u64,
    ) -> (
        PcsConfig,
        ZkVerificationConfig,
        ZkWitnessRandomizationVerifierAudit,
        ZkStarkProof<<Blake2sMerkleChannel as crate::core::channel::MerkleChannel>::H>,
    ) {
        let config = private_stark_test_pcs_config();
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(TEST_FRI_FIRST_LAYER_LOG_SIZE).half_coset(),
        );
        let mut prover_channel = Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();

        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) =
            private_stark_test_configs();
        commit_private_stark_test_inputs(
            &mut commitment_scheme,
            &mut prover_channel,
            &zk_prover_config,
            witness_rng_seed,
        );

        let component = NoConstraintPrivateComponent;
        let mut proof_rng = StdRng::seed_from_u64(proof_rng_seed);
        let proof = prove_zk::<CpuBackend, Blake2sMerkleChannel, _>(
            &[&component],
            &mut prover_channel,
            commitment_scheme,
            &zk_prover_config,
            &mut proof_rng,
        )
        .unwrap();

        (config, zk_verifier_config, zk_verifier_audit, proof)
    }

    fn verify_private_stark_test_proof(
        config: PcsConfig,
        proof: ZkStarkProof<<Blake2sMerkleChannel as crate::core::channel::MerkleChannel>::H>,
        verifier_config: &ZkVerificationConfig,
        verifier_audit: &ZkWitnessRandomizationVerifierAudit,
    ) -> Result<(), crate::core::verifier::VerificationError> {
        let component = NoConstraintPrivateComponent;
        let (mut verifier_channel, mut verifier) =
            verifier_for_private_stark_test_proof(config, &proof);

        crate::core::verifier::verify_zk_with_witness_randomization_audit::<Blake2sMerkleChannel>(
            &[&component],
            &mut verifier_channel,
            &mut verifier,
            proof,
            verifier_config,
            verifier_audit,
        )
    }

    #[test]
    fn private_witness_zk_stark_top_level_proves_and_verifies_with_audit() {
        let config = private_stark_test_pcs_config();
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(TEST_FRI_FIRST_LAYER_LOG_SIZE).half_coset(),
        );
        let mut prover_channel = Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();

        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) =
            private_stark_test_configs();
        commit_private_stark_test_inputs(
            &mut commitment_scheme,
            &mut prover_channel,
            &zk_prover_config,
            1,
        );

        let component = NoConstraintPrivateComponent;
        let mut proof_rng = StdRng::seed_from_u64(2);
        let proof = prove_zk_ex::<CpuBackend, Blake2sMerkleChannel, _>(
            &[&component],
            &mut prover_channel,
            commitment_scheme,
            &zk_prover_config,
            &mut proof_rng,
            false,
        )
        .unwrap();

        let (mut verifier_channel, mut verifier) =
            verifier_for_private_stark_test_proof(config, &proof.proof);
        let (mut unaudited_verifier_channel, mut unaudited_verifier) =
            verifier_for_private_stark_test_proof(config, &proof.proof);
        let unaudited_error = crate::core::verifier::verify_zk_ex::<Blake2sMerkleChannel>(
            &[&component],
            &mut unaudited_verifier_channel,
            &mut unaudited_verifier,
            proof.proof.clone(),
            &zk_verifier_config,
            false,
        )
        .unwrap_err();
        assert!(matches!(
            unaudited_error,
            crate::core::verifier::VerificationError::InvalidStructure(message)
                if message == "Invalid ZK witness-randomization verifier audit"
        ));

        crate::core::verifier::verify_zk_ex_with_witness_randomization_audit::<
            Blake2sMerkleChannel,
        >(
            &[&component],
            &mut verifier_channel,
            &mut verifier,
            proof.proof,
            &zk_verifier_config,
            &zk_verifier_audit,
            false,
        )
        .unwrap();
    }

    #[test]
    fn private_witness_zk_stark_convenience_wrappers_prove_and_verify() {
        let config = private_stark_test_pcs_config();
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(TEST_FRI_FIRST_LAYER_LOG_SIZE).half_coset(),
        );
        let mut prover_channel = Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();

        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) =
            private_stark_test_configs();
        commit_private_stark_test_inputs(
            &mut commitment_scheme,
            &mut prover_channel,
            &zk_prover_config,
            11,
        );

        let component = NoConstraintPrivateComponent;
        let mut proof_rng = StdRng::seed_from_u64(12);
        let proof = prove_zk::<CpuBackend, Blake2sMerkleChannel, _>(
            &[&component],
            &mut prover_channel,
            commitment_scheme,
            &zk_prover_config,
            &mut proof_rng,
        )
        .unwrap();

        let (mut verifier_channel, mut verifier) =
            verifier_for_private_stark_test_proof(config, &proof);
        crate::core::verifier::verify_zk_with_witness_randomization_audit::<Blake2sMerkleChannel>(
            &[&component],
            &mut verifier_channel,
            &mut verifier,
            proof,
            &zk_verifier_config,
            &zk_verifier_audit,
        )
        .unwrap();
    }

    #[test]
    fn private_witness_zk_stark_repeated_proofs_change_private_material() {
        let config = private_stark_test_pcs_config();
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(TEST_FRI_FIRST_LAYER_LOG_SIZE).half_coset(),
        );
        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) =
            private_stark_test_configs();
        let component = NoConstraintPrivateComponent;
        let prove_once = |witness_rng_seed: u64, proof_rng_seed: u64| {
            let mut prover_channel = Blake2sChannel::default();
            let mut commitment_scheme =
                CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new(config, &twiddles);
            commitment_scheme.set_store_polynomials_coefficients();
            commit_private_stark_test_inputs(
                &mut commitment_scheme,
                &mut prover_channel,
                &zk_prover_config,
                witness_rng_seed,
            );

            let mut proof_rng = StdRng::seed_from_u64(proof_rng_seed);
            prove_zk::<CpuBackend, Blake2sMerkleChannel, _>(
                &[&component],
                &mut prover_channel,
                commitment_scheme,
                &zk_prover_config,
                &mut proof_rng,
            )
            .unwrap()
        };
        let proof0 = prove_once(21, 22);
        let proof1 = prove_once(31, 32);

        assert_eq!(
            proof0.0.randomized_pcs_proof.commitments[0],
            proof1.0.randomized_pcs_proof.commitments[0]
        );
        assert_ne!(
            proof0.0.randomized_pcs_proof.commitments[1],
            proof1.0.randomized_pcs_proof.commitments[1]
        );
        assert_ne!(
            proof0.0.randomized_pcs_proof.commitments[2],
            proof1.0.randomized_pcs_proof.commitments[2]
        );
        assert_ne!(
            proof0.0.fri_batch_mask.commitment,
            proof1.0.fri_batch_mask.commitment
        );
        assert_ne!(
            proof0
                .0
                .randomized_pcs_proof
                .sampled_values
                .clone()
                .flatten_cols(),
            proof1
                .0
                .randomized_pcs_proof
                .sampled_values
                .clone()
                .flatten_cols()
        );

        for proof in [proof0, proof1] {
            let (mut verifier_channel, mut verifier) =
                verifier_for_private_stark_test_proof(config, &proof);
            crate::core::verifier::verify_zk_with_witness_randomization_audit::<
                Blake2sMerkleChannel,
            >(
                &[&component],
                &mut verifier_channel,
                &mut verifier,
                proof,
                &zk_verifier_config,
                &zk_verifier_audit,
            )
            .unwrap();
        }
    }

    #[test]
    fn private_witness_zk_stark_rejects_mismatched_public_metadata() {
        let (config, mut verifier_config, verifier_audit, proof) =
            prove_private_stark_test_proof(41, 42);
        verifier_config.metadata.public_statement_hash = ZkPublicStatementHash(test_hash(91));

        assert!(
            verify_private_stark_test_proof(config, proof, &verifier_config, &verifier_audit)
                .is_err()
        );
    }

    #[test]
    fn private_witness_zk_stark_rejects_mismatched_privacy_map_audit() {
        let (config, verifier_config, mut verifier_audit, proof) =
            prove_private_stark_test_proof(51, 52);
        verifier_audit.privacy_map.hash = ZkPrivacyMapHash(test_hash(92));

        assert!(
            verify_private_stark_test_proof(config, proof, &verifier_config, &verifier_audit)
                .is_err()
        );
    }

    #[test]
    fn private_witness_zk_stark_rejects_wrong_committed_private_degree() {
        let (config, verifier_config, verifier_audit, proof) =
            prove_private_stark_test_proof(61, 62);
        let component = NoConstraintPrivateComponent;
        let mut verifier_channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(
            proof.0.randomized_pcs_proof.commitments[0],
            &[TEST_TRACE_LOG_SIZE],
            &mut verifier_channel,
        );
        verifier.commit(
            proof.0.randomized_pcs_proof.commitments[1],
            &[TEST_TRACE_LOG_SIZE],
            &mut verifier_channel,
        );

        assert!(
            crate::core::verifier::verify_zk_with_witness_randomization_audit::<
                Blake2sMerkleChannel,
            >(
                &[&component],
                &mut verifier_channel,
                &mut verifier,
                proof,
                &verifier_config,
                &verifier_audit,
            )
            .is_err()
        );
    }

    #[test]
    fn private_composition_split_helper_masks_opened_halves() {
        let (zk_prover_config, ..) = private_stark_test_configs();
        let profile = zk_prover_config
            .quotient_split_mask_profile
            .expect("private ZK test config must include split profile");
        let composition_poly = SecureCirclePoly(std::array::from_fn(|coordinate| {
            crate::prover::backend::cpu::CpuCirclePoly::new(
                (0..1 << profile.split_identity_log_degree_bound)
                    .map(|index| M31::from(coordinate as u32 * 10_000 + index as u32))
                    .collect(),
            )
        }));
        let point = CirclePoint::get_point(37_913);
        let original_eval = composition_poly.eval_at_point(point);
        let (left, right) = SecureCirclePoly(composition_poly.clone()).split_at_mid();
        let raw_left_eval = left.eval_at_point(point);
        let raw_right_eval = right.eval_at_point(point);
        let split_factor = point
            .repeated_double(profile.split_identity_log_degree_bound - 2)
            .x;
        let mut rng = StdRng::seed_from_u64(51);

        let (masked_left, masked_right) =
            zk_composition_split_coordinate_polys(left, right, &zk_prover_config, &mut rng)
                .unwrap();
        let masked_left = SecureCirclePoly(masked_left);
        let masked_right = SecureCirclePoly(masked_right);

        assert_eq!(masked_left.log_size(), profile.left_masked_log_degree_bound);
        assert_eq!(
            masked_right.log_size(),
            profile.right_masked_log_degree_bound
        );
        assert_ne!(masked_left.eval_at_point(point), raw_left_eval);
        assert_ne!(masked_right.eval_at_point(point), raw_right_eval);
        assert_eq!(
            masked_left.eval_at_point(point) + split_factor * masked_right.eval_at_point(point),
            original_eval
        );
    }
}
