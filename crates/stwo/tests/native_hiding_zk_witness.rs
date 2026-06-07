#![cfg(feature = "prover")]

use rand::rngs::StdRng;
use rand::SeedableRng;
use stwo::core::air::accumulation::PointEvaluationAccumulator;
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sChannel, MerkleChannel};
use stwo::core::circle::CirclePoint;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::fri::FriConfig;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, PcsHidingConfig, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::verifier::{
    verify_zk_ex, verify_zk_ex_with_witness_randomization_audit, COMPOSITION_LOG_SPLIT,
};
use stwo::core::zk::{
    canonical_zk_private_column_scope_hash, canonical_zk_randomizer_space_hash,
    canonical_zk_split_derivation_hash, expected_zk_fri_batch_degree_bound,
    stwo_composition_quotient_split_mask_profile, zk_trace_domain_half_coset,
    ZkCircleCosetEncoding, ZkColumnDegreeBound, ZkColumnRange, ZkDegreeProfile, ZkPrivacyMap,
    ZkPrivacyMapHash, ZkPrivateColumnScope, ZkPrivateColumnScopeEntry, ZkPrivateColumnUsage,
    ZkProofVersion, ZkPublicMetadata, ZkPublicStatementHash, ZkQuotientIntegrationProfile,
    ZkRandomizerSpaceEntry, ZkStarkProof, ZkVerificationConfig, ZkWitnessRandomizationProfile,
    ZkWitnessRandomizationVerifierAudit,
};
use stwo::prover::backend::cpu::CpuBackend;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};
use stwo::prover::zk::{ZkDerivationGate, ZkDerivationReview, ZkProvingConfig};
use stwo::prover::{
    prove_zk_ex, CommitmentSchemeProver, ComponentProver, DomainEvaluationAccumulator,
    ProvingError, Trace,
};

const TRACE_LOG_SIZE: u32 = 5;
const RANDOMIZED_LOG_DEGREE: u32 = TRACE_LOG_SIZE + 1;
const FRI_FIRST_LAYER_LOG_SIZE: u32 = RANDOMIZED_LOG_DEGREE + 2;

struct PrivateWitnessNoopComponent;

impl Component for PrivateWitnessNoopComponent {
    fn n_constraints(&self) -> usize {
        0
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        RANDOMIZED_LOG_DEGREE + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<Vec<u32>> {
        TreeVec(vec![vec![TRACE_LOG_SIZE], vec![TRACE_LOG_SIZE]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
        _max_log_degree_bound: u32,
    ) -> TreeVec<Vec<Vec<CirclePoint<SecureField>>>> {
        TreeVec(vec![vec![vec![point]], vec![vec![point]]])
    }

    fn mask_offsets(&self) -> Option<TreeVec<Vec<Vec<isize>>>> {
        Some(TreeVec(vec![vec![vec![0]], vec![vec![0]]]))
    }

    fn preprocessed_column_indices(&self) -> Vec<usize> {
        vec![0]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &TreeVec<Vec<Vec<SecureField>>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
        _max_log_degree_bound: u32,
    ) {
    }
}

impl ComponentProver<CpuBackend> for PrivateWitnessNoopComponent {
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

fn derivation_reviews() -> Vec<ZkDerivationReview> {
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

fn zk_configs() -> (
    ZkProvingConfig,
    ZkVerificationConfig,
    ZkWitnessRandomizationVerifierAudit,
) {
    let private_range = ZkColumnRange::new(1, 0, 1);
    let quotient_range = ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE);

    let mut private_column_scope = ZkPrivateColumnScope {
        version: ZkProofVersion::V1,
        hash: [0; 32],
        entries: vec![ZkPrivateColumnScopeEntry {
            range: private_range,
            usage: ZkPrivateColumnUsage::OrdinaryWitness,
            trace_domain_log_size: TRACE_LOG_SIZE,
            semantic_trace_domain_log_sizes: vec![TRACE_LOG_SIZE],
        }],
    };
    private_column_scope.hash = canonical_zk_private_column_scope_hash(&private_column_scope);

    let trace_domain = CanonicCoset::new(TRACE_LOG_SIZE).coset;
    let h_witness = 1u64 << TRACE_LOG_SIZE;
    let randomizer_space_hash = canonical_zk_randomizer_space_hash(
        private_column_scope.hash,
        &[ZkRandomizerSpaceEntry {
            range: private_range,
            trace_domain: ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(trace_domain)),
            semantic_trace_domains: vec![ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(
                trace_domain,
            ))],
            randomized_log_degree: RANDOMIZED_LOG_DEGREE,
            randomizer_dimension: h_witness,
        }],
    );

    let private_degree_bound = ZkColumnDegreeBound {
        range: private_range,
        log_degree_bound: RANDOMIZED_LOG_DEGREE,
    };
    let quotient_degree_bound = ZkColumnDegreeBound {
        range: quotient_range,
        log_degree_bound: FRI_FIRST_LAYER_LOG_SIZE - 2,
    };
    let h_batch = expected_zk_fri_batch_degree_bound(FRI_FIRST_LAYER_LOG_SIZE, 1).unwrap();
    let privacy_map_hash = ZkPrivacyMapHash(test_hash(1));
    let metadata = ZkPublicMetadata {
        version: ZkProofVersion::V1,
        privacy_map_hash,
        public_statement_hash: ZkPublicStatementHash(test_hash(2)),
        logup_statistical_security_budget_hash: [0x5a; 32],
        degree_profile: ZkDegreeProfile {
            trace_domain_log_size: TRACE_LOG_SIZE,
            h_witness,
            h_batch,
            fri_first_layer_log_size: FRI_FIRST_LAYER_LOG_SIZE,
        },
        witness_randomization: ZkWitnessRandomizationProfile {
            h_witness,
            randomizer_space_hash,
            private_column_scope_hash: private_column_scope.hash,
            private_column_degree_bounds: vec![private_degree_bound],
        },
        quotient_integration: ZkQuotientIntegrationProfile {
            h_batch,
            fri_first_layer_log_size: FRI_FIRST_LAYER_LOG_SIZE,
            split_derivation_hash: canonical_zk_split_derivation_hash(COMPOSITION_LOG_SPLIT),
            quotient_degree_bounds: vec![quotient_degree_bound],
        },
    };
    let column_degree_bounds = vec![private_degree_bound, quotient_degree_bound];
    let quotient_split_mask_profile = stwo_composition_quotient_split_mask_profile(
        quotient_degree_bound.range.tree_index,
        quotient_degree_bound.log_degree_bound + 1,
        quotient_degree_bound.log_degree_bound,
        1u64 << quotient_degree_bound.log_degree_bound,
        quotient_degree_bound.log_degree_bound + 1,
        quotient_degree_bound.log_degree_bound,
    )
    .unwrap();
    let privacy_map = ZkPrivacyMap {
        version: ZkProofVersion::V1,
        private_columns: vec![private_range],
        hash: privacy_map_hash,
    };

    (
        ZkProvingConfig {
            metadata: metadata.clone(),
            privacy_map: privacy_map.clone(),
            private_column_scope: Some(private_column_scope.clone()),
            quotient_split_mask_profile: Some(quotient_split_mask_profile),
            logup_statistical_security_budgets: Vec::new(),
            query_closure: None,
            randomizer_rank_profile: None,
            derived_randomizer_metadata: None,
            column_degree_bounds: column_degree_bounds.clone(),
            derivation_reviews: derivation_reviews(),
        },
        ZkVerificationConfig {
            metadata,
            column_degree_bounds,
            quotient_split_mask_profile: Some(quotient_split_mask_profile),
            logup_statistical_security_budgets: Vec::new(),
        },
        ZkWitnessRandomizationVerifierAudit {
            privacy_map,
            private_column_scope,
        },
    )
}

fn hidden_pcs_config() -> PcsConfig {
    PcsConfig {
        fri_config: FriConfig::new(1, 1, 3, 1),
        lifting_log_size: Some(FRI_FIRST_LAYER_LOG_SIZE),
        hiding: Some(PcsHidingConfig::new(4)),
        ..PcsConfig::default()
    }
}

fn preprocessed_column() -> CircleCoefficients<CpuBackend> {
    CircleCoefficients::new(
        (0..1 << TRACE_LOG_SIZE)
            .map(|value| M31::from(7_000 + value as u32))
            .collect(),
    )
}

fn private_witness_column() -> CircleCoefficients<CpuBackend> {
    CircleCoefficients::new(
        (0..1 << TRACE_LOG_SIZE)
            .map(|value| M31::from(value as u32))
            .collect(),
    )
}

fn prove_with_native_hiding(
    witness_randomization_seed: u64,
    pcs_hiding_seed: u64,
    proof_masking_seed: u64,
) -> (
    ZkStarkProof<<Blake2sMerkleChannel as MerkleChannel>::H>,
    ZkVerificationConfig,
    ZkWitnessRandomizationVerifierAudit,
) {
    let config = hidden_pcs_config();
    let twiddles =
        CpuBackend::precompute_twiddles(CanonicCoset::new(FRI_FIRST_LAYER_LOG_SIZE).half_coset());
    let mut prover_channel = Blake2sChannel::default();
    let mut pcs_hiding_rng = StdRng::seed_from_u64(pcs_hiding_seed);
    let mut commitment_scheme =
        CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new_hiding(
            config,
            &twiddles,
            &mut pcs_hiding_rng,
        );
    commitment_scheme.set_store_polynomials_coefficients();

    let (zk_prover_config, zk_verifier_config, zk_verifier_audit) = zk_configs();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(vec![preprocessed_column()]);
    tree_builder.commit(&mut prover_channel);

    let mut witness_randomization_rng = StdRng::seed_from_u64(witness_randomization_seed);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_polys(vec![private_witness_column()]);
    tree_builder
        .commit_zk_witness_randomized(
            &zk_prover_config,
            &mut witness_randomization_rng,
            &mut prover_channel,
        )
        .unwrap();

    let component = PrivateWitnessNoopComponent;
    let mut proof_masking_rng = StdRng::seed_from_u64(proof_masking_seed);
    let proof = prove_zk_ex::<CpuBackend, Blake2sMerkleChannel, _>(
        &[&component],
        &mut prover_channel,
        commitment_scheme,
        &zk_prover_config,
        &mut proof_masking_rng,
        false,
    )
    .unwrap()
    .proof;

    (proof, zk_verifier_config, zk_verifier_audit)
}

fn verifier_for(
    config: PcsConfig,
    proof: &ZkStarkProof<<Blake2sMerkleChannel as MerkleChannel>::H>,
) -> (
    Blake2sChannel,
    CommitmentSchemeVerifier<Blake2sMerkleChannel>,
) {
    let mut channel = Blake2sChannel::default();
    let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
    verifier.commit(
        proof.0.randomized_pcs_proof.commitments[0],
        &[TRACE_LOG_SIZE],
        &mut channel,
    );
    verifier.commit(
        proof.0.randomized_pcs_proof.commitments[1],
        &[RANDOMIZED_LOG_DEGREE],
        &mut channel,
    );

    (channel, verifier)
}

#[test]
fn native_hiding_zk_path_hides_private_witness_material() {
    let (proof0, verifier_config, verifier_audit) = prove_with_native_hiding(11, 12, 13);
    let (proof1, ..) = prove_with_native_hiding(21, 22, 23);

    assert_ne!(
        proof0.0.randomized_pcs_proof.commitments[1],
        proof1.0.randomized_pcs_proof.commitments[1]
    );
    assert_ne!(
        proof0.0.randomized_pcs_proof.commitments[2],
        proof1.0.randomized_pcs_proof.commitments[2]
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

    let component = PrivateWitnessNoopComponent;
    let (mut verifier_channel, mut verifier) = verifier_for(hidden_pcs_config(), &proof0);
    verify_zk_ex_with_witness_randomization_audit::<Blake2sMerkleChannel>(
        &[&component],
        &mut verifier_channel,
        &mut verifier,
        proof0.clone(),
        &verifier_config,
        &verifier_audit,
        false,
    )
    .unwrap();

    let (mut transparent_channel, mut transparent_verifier) =
        verifier_for(PcsConfig::default(), &proof0);
    assert!(
        verify_zk_ex_with_witness_randomization_audit::<Blake2sMerkleChannel>(
            &[&component],
            &mut transparent_channel,
            &mut transparent_verifier,
            proof0.clone(),
            &verifier_config,
            &verifier_audit,
            false,
        )
        .is_err()
    );

    let (mut unaudited_channel, mut unaudited_verifier) =
        verifier_for(hidden_pcs_config(), &proof0);
    assert!(verify_zk_ex::<Blake2sMerkleChannel>(
        &[&component],
        &mut unaudited_channel,
        &mut unaudited_verifier,
        proof0,
        &verifier_config,
        false,
    )
    .is_err());
}
