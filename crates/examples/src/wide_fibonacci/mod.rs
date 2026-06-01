use itertools::Itertools;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::PackedBaseField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Backend, Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

pub type WideFibonacciComponent<const N: usize> = FrameworkComponent<WideFibonacciEval<N>>;

mod fib_with_preprocessed;

pub struct FibInput {
    pub a: BaseField,
    pub b: BaseField,
}

pub struct FibInputSimd {
    a: PackedBaseField,
    b: PackedBaseField,
}

pub fn generate_trace<const N: usize, B: Backend>(
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    assert!(inputs.len().is_power_of_two());
    let log_size = inputs.len().ilog2();
    let mut trace = (0..N)
        .map(|_| Col::<B, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].set(vec_index, a);
        trace[1].set(vec_index, b);
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.set(vec_index, b);
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<B, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// Same as [`generate_trace`] but optimized for simd.
pub fn generate_trace_simd<const N: usize>(
    log_size: u32,
    inputs: &[FibInputSimd],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a separate Fibonacci sequence of length `N`.
#[derive(Clone)]
pub struct WideFibonacciEval<const N: usize> {
    pub log_n_rows: u32,
}
impl<const N: usize> FrameworkEval for WideFibonacciEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sM31Channel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::channel::Poseidon252Channel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleChannel;
    use stwo::core::verifier::{verify, verify_zk_with_witness_randomization_audit};
    use stwo::core::zk::{
        canonical_zk_private_column_scope_hash, canonical_zk_randomizer_space_hash,
        canonical_zk_split_derivation_hash, expected_zk_fri_batch_degree_bound,
        stwo_composition_quotient_split_mask_profile, zk_trace_domain_half_coset,
        ZkCircleCosetEncoding, ZkColumnDegreeBound, ZkColumnRange, ZkDegreeProfile, ZkPrivacyMap,
        ZkPrivacyMapHash, ZkPrivateColumnScope, ZkPrivateColumnScopeEntry, ZkPrivateColumnUsage,
        ZkProofVersion, ZkPublicMetadata, ZkPublicStatementHash, ZkQuotientIntegrationProfile,
        ZkRandomizerSpaceEntry, ZkVerificationConfig, ZkWitnessRandomizationProfile,
        ZkWitnessRandomizationVerifierAudit,
    };
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::{Column, CpuBackend};
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::zk::{ZkDerivationGate, ZkDerivationReview, ZkProvingConfig};
    use stwo::prover::{prove, prove_zk, CommitmentSchemeProver};
    use stwo_constraint_framework::{
        assert_constraints_on_polys, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
    };

    use super::WideFibonacciEval;
    use crate::wide_fibonacci::{generate_trace, FibInput, WideFibonacciComponent};

    const FIB_SEQUENCE_LENGTH: usize = 100;
    const ZK_FIB_SEQUENCE_LENGTH: usize = 3;
    const ZK_FIB_LOG_N_INSTANCES: u32 = 5;
    const ZK_FIB_RANDOMIZED_LOG_DEGREE: u32 = ZK_FIB_LOG_N_INSTANCES + 1;
    const ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE: u32 = ZK_FIB_RANDOMIZED_LOG_DEGREE + 3;

    fn generate_test_inputs(log_n_instances: u32) -> Vec<FibInput> {
        (0..1 << log_n_instances)
            .map(|i| FibInput {
                a: BaseField::one(),
                b: BaseField::from_u32_unchecked(i as u32),
            })
            .collect_vec()
    }

    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { log_n_rows: N }.evaluate(eval);
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
            review_hash: test_hash(index as u8 + 30),
        })
        .collect()
    }

    fn private_wide_fibonacci_zk_configs() -> (
        ZkProvingConfig,
        ZkVerificationConfig,
        ZkWitnessRandomizationVerifierAudit,
    ) {
        let private_ranges = (0..ZK_FIB_SEQUENCE_LENGTH)
            .map(|column| ZkColumnRange::new(1, column, column + 1))
            .collect_vec();
        let quotient_range = ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE);
        let privacy_map_hash = ZkPrivacyMapHash(test_hash(1));
        let mut private_column_scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: [0; 32],
            entries: private_ranges
                .iter()
                .copied()
                .map(|range| ZkPrivateColumnScopeEntry {
                    range,
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                })
                .collect(),
        };
        private_column_scope.hash = canonical_zk_private_column_scope_hash(&private_column_scope);

        let trace_domain = CanonicCoset::new(ZK_FIB_LOG_N_INSTANCES).coset;
        let h_witness = 1u64 << ZK_FIB_LOG_N_INSTANCES;
        let randomizer_space_entries = private_ranges
            .iter()
            .copied()
            .map(|range| ZkRandomizerSpaceEntry {
                range,
                trace_domain: ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(trace_domain)),
                randomized_log_degree: ZK_FIB_RANDOMIZED_LOG_DEGREE,
                randomizer_dimension: h_witness,
            })
            .collect_vec();
        let randomizer_space_hash = canonical_zk_randomizer_space_hash(
            private_column_scope.hash,
            &randomizer_space_entries,
        );
        let private_degree_bounds = private_ranges
            .iter()
            .copied()
            .map(|range| ZkColumnDegreeBound {
                range,
                log_degree_bound: ZK_FIB_RANDOMIZED_LOG_DEGREE,
            })
            .collect_vec();
        let quotient_degree_bound = ZkColumnDegreeBound {
            range: quotient_range,
            log_degree_bound: ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE - 2,
        };
        let h_batch = expected_zk_fri_batch_degree_bound(ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE, 1)
            .expect("test FRI layer has a valid batch-mask degree");
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash,
            public_statement_hash: ZkPublicStatementHash(test_hash(2)),
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: ZK_FIB_LOG_N_INSTANCES,
                h_witness,
                h_batch,
                fri_first_layer_log_size: ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness,
                randomizer_space_hash,
                private_column_scope_hash: private_column_scope.hash,
                private_column_degree_bounds: private_degree_bounds.clone(),
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch,
                fri_first_layer_log_size: ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE,
                split_derivation_hash: canonical_zk_split_derivation_hash(
                    stwo::core::verifier::COMPOSITION_LOG_SPLIT,
                ),
                quotient_degree_bounds: vec![quotient_degree_bound],
            },
        };
        let mut column_degree_bounds = private_degree_bounds;
        column_degree_bounds.push(quotient_degree_bound);
        let quotient_split_mask_profile = stwo_composition_quotient_split_mask_profile(
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
            private_columns: private_ranges,
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

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let traces = TreeVec::new(vec![
            vec![],
            generate_trace::<FIB_SEQUENCE_LENGTH, SimdBackend>(&generate_test_inputs(
                LOG_N_INSTANCES,
            )),
        ]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test]
    #[should_panic]
    fn test_wide_fibonacci_constraints_fails() {
        const LOG_N_INSTANCES: u32 = 6;

        let mut trace = generate_trace::<FIB_SEQUENCE_LENGTH, SimdBackend>(&generate_test_inputs(
            LOG_N_INSTANCES,
        ));
        // Modify the trace such that a constraint fail.
        trace[17].values.set(2, BaseField::one());
        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
    }

    #[test_log::test]
    fn test_wide_fib_prove_with_blake() {
        for log_n_instances in 4..=8 {
            let config = PcsConfig::default();
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test_log::test]
    fn test_wide_fib_zk_private_witness_prove_with_blake() {
        let config = PcsConfig {
            fri_config: FriConfig::new(1, 1, 3, 1),
            lifting_log_size: Some(ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE),
            ..PcsConfig::default()
        };
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE).half_coset(),
        );
        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();

        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) =
            private_wide_fibonacci_zk_configs();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        let trace = generate_trace::<ZK_FIB_SEQUENCE_LENGTH, CpuBackend>(&generate_test_inputs(
            ZK_FIB_LOG_N_INSTANCES,
        ));
        let mut witness_rng = StdRng::seed_from_u64(101);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder
            .commit_zk_witness_randomized(&zk_prover_config, &mut witness_rng, prover_channel)
            .unwrap();

        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<ZK_FIB_SEQUENCE_LENGTH> {
                log_n_rows: ZK_FIB_LOG_N_INSTANCES,
            },
            SecureField::zero(),
        );
        let mut proof_rng = StdRng::seed_from_u64(102);
        let proof = prove_zk::<CpuBackend, Blake2sM31MerkleChannel, _>(
            &[&component],
            prover_channel,
            commitment_scheme,
            &zk_prover_config,
            &mut proof_rng,
        )
        .unwrap();

        let verifier_channel = &mut Blake2sM31Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
        commitment_scheme.commit(
            proof.0.randomized_pcs_proof.commitments[0],
            &[],
            verifier_channel,
        );
        commitment_scheme.commit(
            proof.0.randomized_pcs_proof.commitments[1],
            &vec![ZK_FIB_RANDOMIZED_LOG_DEGREE; ZK_FIB_SEQUENCE_LENGTH],
            verifier_channel,
        );

        verify_zk_with_witness_randomization_audit(
            &[&component],
            verifier_channel,
            commitment_scheme,
            proof,
            &zk_verifier_config,
            &zk_verifier_audit,
        )
        .unwrap();
    }

    #[test]
    fn test_wide_fib_zk_private_witness_repeated_proofs_hide_private_material() {
        let config = PcsConfig {
            fri_config: FriConfig::new(1, 1, 3, 1),
            lifting_log_size: Some(ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE),
            ..PcsConfig::default()
        };
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(ZK_FIB_FRI_FIRST_LAYER_LOG_SIZE).half_coset(),
        );
        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) =
            private_wide_fibonacci_zk_configs();
        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<ZK_FIB_SEQUENCE_LENGTH> {
                log_n_rows: ZK_FIB_LOG_N_INSTANCES,
            },
            SecureField::zero(),
        );

        let prove_once = |witness_rng_seed: u64, proof_rng_seed: u64| {
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme =
                CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(
                    config, &twiddles,
                );
            commitment_scheme.set_store_polynomials_coefficients();

            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            let trace = generate_trace::<ZK_FIB_SEQUENCE_LENGTH, CpuBackend>(
                &generate_test_inputs(ZK_FIB_LOG_N_INSTANCES),
            );
            let mut witness_rng = StdRng::seed_from_u64(witness_rng_seed);
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder
                .commit_zk_witness_randomized(&zk_prover_config, &mut witness_rng, prover_channel)
                .unwrap();

            let mut proof_rng = StdRng::seed_from_u64(proof_rng_seed);
            prove_zk::<CpuBackend, Blake2sM31MerkleChannel, _>(
                &[&component],
                prover_channel,
                commitment_scheme,
                &zk_prover_config,
                &mut proof_rng,
            )
            .unwrap()
        };

        let proof_a = prove_once(201, 202);
        let proof_b = prove_once(301, 302);

        assert_eq!(proof_a.0.public_metadata, proof_b.0.public_metadata);
        assert_eq!(
            proof_a.0.randomized_pcs_proof.commitments[0],
            proof_b.0.randomized_pcs_proof.commitments[0]
        );
        assert_ne!(
            proof_a.0.randomized_pcs_proof.commitments[1],
            proof_b.0.randomized_pcs_proof.commitments[1]
        );
        assert_ne!(
            proof_a.0.randomized_pcs_proof.commitments[2],
            proof_b.0.randomized_pcs_proof.commitments[2]
        );
        assert_ne!(
            proof_a.0.randomized_pcs_proof.sampled_values[1],
            proof_b.0.randomized_pcs_proof.sampled_values[1]
        );
        assert_ne!(
            proof_a.0.fri_batch_mask.commitment,
            proof_b.0.fri_batch_mask.commitment
        );

        for proof in [proof_a, proof_b] {
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
            commitment_scheme.commit(
                proof.0.randomized_pcs_proof.commitments[0],
                &[],
                verifier_channel,
            );
            commitment_scheme.commit(
                proof.0.randomized_pcs_proof.commitments[1],
                &vec![ZK_FIB_RANDOMIZED_LOG_DEGREE; ZK_FIB_SEQUENCE_LENGTH],
                verifier_channel,
            );

            verify_zk_with_witness_randomization_audit(
                &[&component],
                verifier_channel,
                commitment_scheme,
                proof,
                &zk_verifier_config,
                &zk_verifier_audit,
            )
            .unwrap();
        }
    }

    /// Tests the subdomain evaluation path (log_expansion > 0) by using log_blowup_factor = 2
    /// with constraint degree 1, so the committed domain is larger than the eval domain.
    #[test_log::test]
    fn test_wide_fib_prove_with_larger_blowup() {
        for log_n_instances in 4..=7 {
            let config = PcsConfig {
                pow_bits: 10,
                fri_config: FriConfig::new(0, 2, 3, 1),
                lifting_log_size: None,
            };
            // Precompute twiddles for the larger committed domain.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace.
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    /// Same as [test_wide_fib_prove_with_blake] but with FRI fold step > 1.
    #[test]
    fn test_wide_fib_prove_with_blake_with_fri_jumps() {
        for log_n_instances in 4..=8 {
            let mut config = PcsConfig::default();
            // Test different steps.
            config.fri_config.fold_step = if (4..6).contains(&log_n_instances) {
                2
            } else {
                3
            };
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sM31Channel::default();
            let mut commitment_scheme = CommitmentSchemeProver::<
                SimdBackend,
                Blake2sM31MerkleChannel,
            >::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(vec![]);
            tree_builder.commit(prover_channel);

            // Trace.
            let trace =
                generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(log_n_instances));
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                },
                SecureField::zero(),
            );

            let proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sM31Channel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wide_fib_prove_with_poseidon() {
        const LOG_N_INSTANCES: u32 = 6;
        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Poseidon252Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

        // TODO(ilya): remove the following once preprocessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace =
            generate_trace::<FIB_SEQUENCE_LENGTH, _>(&generate_test_inputs(LOG_N_INSTANCES));
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: LOG_N_INSTANCES,
            },
            SecureField::zero(),
        );
        let proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Poseidon252Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }

    #[test]
    fn test_e2e_lifted_fib_prove() {
        const LOG_SIZE_SHORT: u32 = 3;
        const LOG_SIZE_LONG: u32 = 9;

        const N_COLS_LONG_COMPONENT: usize = 4;
        const N_COLS_SHORT_COMPONENT: usize = 5;

        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(LOG_SIZE_LONG + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Blake2sM31Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        // Trace.
        let trace = [
            generate_trace::<N_COLS_LONG_COMPONENT, _>(&generate_test_inputs(LOG_SIZE_LONG)),
            generate_trace::<N_COLS_SHORT_COMPONENT, _>(&generate_test_inputs(LOG_SIZE_SHORT)),
        ]
        .concat();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Generate components.
        let mut trace_alloc = TraceLocationAllocator::default();
        let component0 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_COLS_LONG_COMPONENT> {
                log_n_rows: LOG_SIZE_LONG,
            },
            SecureField::zero(),
        );
        let component1 = WideFibonacciComponent::new(
            &mut trace_alloc,
            WideFibonacciEval::<N_COLS_SHORT_COMPONENT> {
                log_n_rows: LOG_SIZE_SHORT,
            },
            SecureField::zero(),
        );

        // Prove.
        let proof = prove::<CpuBackend, Blake2sM31MerkleChannel>(
            &[&component0, &component1],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sM31Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);

        let trace_sizes = [
            vec![LOG_SIZE_LONG; N_COLS_LONG_COMPONENT],
            vec![LOG_SIZE_SHORT; N_COLS_SHORT_COMPONENT],
        ]
        .concat();
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = TreeVec::new(vec![vec![], trace_sizes]);
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);

        assert!(verify(
            &[&component0, &component1],
            verifier_channel,
            commitment_scheme,
            proof,
        )
        .is_ok());
    }
}
