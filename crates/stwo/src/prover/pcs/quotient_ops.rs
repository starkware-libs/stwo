use std::iter::zip;

use itertools::Itertools;
use tracing::{span, Level};

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    build_samples_with_randomness_and_periodicity, ColumnSampleBatch, PointSample,
};
use crate::core::pcs::TreeVec;
use crate::prover::backend::ColumnOps;
use crate::prover::poly::circle::{CircleEvaluation, PolyOps, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

pub trait QuotientOps: PolyOps {
    /// Receives a non-empty set of columns of the *same* size, and populates the vector
    /// `accumulated_numerators_vec` with their FRI numerators accumulations, across
    /// `sample_batches`.
    ///
    /// For each sample batch in `sample_batches`, accumulates the numerators of the columns
    /// involved in this batch over the evaluation subdomain (the first
    /// `column_size >> log_blowup_factor` rows in bit-reversed order) and pushes an
    /// `AccumulatedNumerators` object into `accumulated_numerators_vec`.
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    );

    /// Given a vector of `AccumulatedNumerators` (computed on evaluation subdomains), computes
    /// the full quotient on the subdomain of size `2^(lifting_log_size - log_blowup_factor)`,
    /// then interpolates and evaluates on the full domain of size `2^lifting_log_size`.
    ///
    /// For each subdomain point:
    /// * Computes the denominator inverse for each sample point.
    /// * Multiplies it by the accumulated numerator for that (subdomain point, sample point).
    /// * Sums across sample points.
    ///
    /// The result is then extended to the full evaluation domain via interpolation on the
    /// subdomain followed by evaluation on the full domain, using the provided `twiddles`.
    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureEvaluation<Self, BitReversedOrder>;
}

/// Helper struct that keeps track of the accumulation of the numerators involved in the FRI
/// quotients.
///
/// Note: `Clone` is derived only for benchmarking purposes.
#[derive(Clone)]
pub struct AccumulatedNumerators<B: ColumnOps<BaseField>> {
    /// One of the sample points received by the pcs.
    pub sample_point: CirclePoint<SecureField>,
    /// Stores a circle evaluation of the form:
    ///     p -> ∑ α^{k_i} * (cᵢ * f̃ᵢ(p) - bᵢ)
    /// where
    /// * p ∈ canonic coset of log size = l (where l is the log size of the column).
    /// * i runs over some column indices of the trace.
    /// * α is the random coefficient for the accumulation.
    /// * k_i is the randomness exponent for column i and sample point `sample_point`.
    /// * f̃ᵢ is the lift of the trace poly fᵢ to log size l.
    /// * (bᵢ, cᵢ) are the `b` and `c` line coefficients for column i and sample point
    ///   `sample_point`.
    pub partial_numerators_acc: SecureColumnByCoords<B>,
    /// Stores an accumulation of the form
    ///      ∑ α^{k_i} * aᵢ
    /// where
    /// * i runs over some column indices of the trace.
    /// * α and k_i are as in the previous docstring for `partial_numerators_acc`.
    /// * aᵢ is the `a` line coefficient for column i and sample point `sample_point`.
    ///
    /// The index set of the summation is equal to the index set of the summation referred in the
    /// previous docstring for `partial_numerators_acc`.
    pub first_linear_term_acc: SecureField,
}

pub fn compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &TreeVec<Vec<&CircleEvaluation<B, BaseField, BitReversedOrder>>>,
    samples: &TreeVec<Vec<Vec<PointSample>>>,
    random_coeff: SecureField,
    lifting_log_size: u32,
    twiddles: &TwiddleTree<B>,
    log_blowup_factor: u32,
) -> SecureEvaluation<B, BitReversedOrder> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    let mut accumulated_numerators_vec: Vec<AccumulatedNumerators<B>> = vec![];
    let samples_with_randomness = build_samples_with_randomness_and_periodicity(
        samples,
        columns
            .0
            .iter()
            .map(|x| x.iter().map(|c| c.domain.log_size()))
            .collect(),
        lifting_log_size,
        random_coeff,
    );

    // Populate `accumulated_numerators_vec`, per (log_size, sample_point). After this iteration,
    // `accumulated_numerators_vec` will have length equal to
    //
    //   ∑_k (# of distinct sample points per log size k).
    //
    zip(
        columns.iter().flatten(),
        samples_with_randomness.iter().flatten(),
    )
    .sorted_by_key(|(c, _)| c.domain.log_size())
    .group_by(|(c, _)| c.domain.log_size())
    .into_iter()
    .for_each(|(_, tuples)| {
        let (columns, samples_with_randomness): (Vec<_>, Vec<_>) = tuples.unzip();
        // TODO: slice.
        let sample_batches = ColumnSampleBatch::new_vec(&samples_with_randomness);
        B::accumulate_numerators(
            &columns,
            &sample_batches,
            &mut accumulated_numerators_vec,
            log_blowup_factor,
        )
    });

    // Group and accumulate the numerators per sample point: the accumulations (of different
    // lengths) get lifted and accumulated to a single vector. After this step, there is a single
    // accumulation per sample point.
    let accumulations_per_sample_point = accumulated_numerators_vec
        .into_iter()
        .sorted_by_key(|c| (c.sample_point.x, c.sample_point.y))
        .group_by(|c| c.sample_point)
        .into_iter()
        .map(|(sample_point, accumulations_per_log_size)| {
            let accumulations_per_log_size = accumulations_per_log_size.collect_vec();
            // Accumulate the `a` coefficients.
            let first_linear_term_acc: SecureField = accumulations_per_log_size
                .iter()
                .map(|x| x.first_linear_term_acc)
                .sum();
            // Lift and accumulate the partial numerators vectors.
            // `partial_numerators_acc` is already sorted increasingly by size as required by
            // `B::lift_and_accumulate`.
            let partial_numerators_acc = accumulations_per_log_size
                .into_iter()
                .map(|x| x.partial_numerators_acc)
                .collect_vec();
            let res = B::lift_and_accumulate(partial_numerators_acc).unwrap();

            AccumulatedNumerators {
                sample_point,
                partial_numerators_acc: res,
                first_linear_term_acc,
            }
        })
        .collect_vec();

    B::compute_quotients_and_combine(
        accumulations_per_sample_point,
        lifting_log_size,
        log_blowup_factor,
        twiddles,
    )
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{CryptoRng, Rng, RngCore, SeedableRng};

    use crate::core::channel::Blake2sChannel;
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::SECURE_EXTENSION_DEGREE;
    use crate::core::pcs::quotients::PointSample;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::verifier::VerificationError;
    use crate::core::zk::{
        canonical_zk_private_column_scope_hash, canonical_zk_randomizer_space_hash,
        canonical_zk_split_derivation_hash, expected_zk_fri_batch_degree_bound,
        zk_trace_domain_half_coset, ZkCircleCosetEncoding, ZkColumnDegreeBound, ZkColumnRange,
        ZkDegreeProfile, ZkPrivacyMap, ZkPrivacyMapHash, ZkPrivateColumnScope,
        ZkPrivateColumnScopeEntry, ZkPrivateColumnUsage, ZkProofVersion, ZkPublicMetadata,
        ZkPublicStatementHash, ZkQuotientIntegrationProfile, ZkRandomizerSpaceEntry,
        ZkVerificationConfig, ZkWitnessRandomizationProfile, ZkWitnessRandomizationVerifierAudit,
    };
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Backend, BackendForChannel, Column, CpuBackend};
    use crate::prover::pcs::quotient_ops::compute_fri_quotients;
    use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
    use crate::prover::zk::{ZkDerivationGate, ZkDerivationReview, ZkProvingConfig};
    use crate::prover::{CommitmentSchemeProver, SecureField};

    struct DeterministicTestCryptoRng(SmallRng);

    impl DeterministicTestCryptoRng {
        fn seed_from_u64(seed: u64) -> Self {
            Self(SmallRng::seed_from_u64(seed))
        }
    }

    impl RngCore for DeterministicTestCryptoRng {
        fn next_u32(&mut self) -> u32 {
            self.0.next_u32()
        }

        fn next_u64(&mut self) -> u64 {
            self.0.next_u64()
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            self.0.fill_bytes(dest);
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
            self.0.try_fill_bytes(dest)
        }
    }

    impl CryptoRng for DeterministicTestCryptoRng {}

    fn zk_public_only_configs(
        lifting_log_size: u32,
        log_blowup_factor: u32,
    ) -> (ZkProvingConfig, ZkVerificationConfig) {
        let h_batch =
            expected_zk_fri_batch_degree_bound(lifting_log_size, log_blowup_factor).unwrap();
        let privacy_map_hash = ZkPrivacyMapHash([1; 32]);
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash,
            public_statement_hash: ZkPublicStatementHash([2; 32]),
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: lifting_log_size - log_blowup_factor,
                h_witness: 0,
                h_batch,
                fri_first_layer_log_size: lifting_log_size,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness: 0,
                randomizer_space_hash: [0; 32],
                private_column_scope_hash: [0; 32],
                private_column_degree_bounds: Vec::new(),
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch,
                fri_first_layer_log_size: lifting_log_size,
                split_derivation_hash: [0; 32],
                quotient_degree_bounds: Vec::new(),
            },
        };
        let prover_config = ZkProvingConfig {
            metadata: metadata.clone(),
            privacy_map: ZkPrivacyMap {
                version: ZkProofVersion::V1,
                private_columns: Vec::new(),
                hash: privacy_map_hash,
            },
            private_column_scope: None,
            quotient_split_mask_profile: None,
            query_closure: None,
            randomizer_rank_profile: None,
            derived_randomizer_metadata: None,
            column_degree_bounds: Vec::new(),
            derivation_reviews: Vec::new(),
        };
        let verifier_config = ZkVerificationConfig {
            metadata,
            column_degree_bounds: Vec::new(),
        };

        (prover_config, verifier_config)
    }

    fn nonzero_hash(seed: u8) -> [u8; 32] {
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
            review_hash: nonzero_hash(index as u8 + 10),
        })
        .collect()
    }

    fn zk_private_witness_configs(
        private_tree_index: usize,
        trace_log_size: u32,
        randomized_log_degree: u32,
        fri_first_layer_log_size: u32,
        log_blowup_factor: u32,
    ) -> (
        ZkProvingConfig,
        ZkVerificationConfig,
        ZkWitnessRandomizationVerifierAudit,
    ) {
        let range = ZkColumnRange::new(private_tree_index, 0, 1);
        let quotient_range =
            ZkColumnRange::new(private_tree_index + 1, 0, 2 * SECURE_EXTENSION_DEGREE);
        let h_witness = 1u64 << trace_log_size;
        let h_batch =
            expected_zk_fri_batch_degree_bound(fri_first_layer_log_size, log_blowup_factor)
                .unwrap();
        let privacy_map_hash = ZkPrivacyMapHash(nonzero_hash(1));
        let mut private_column_scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: [0; 32],
            entries: vec![ZkPrivateColumnScopeEntry {
                range,
                usage: ZkPrivateColumnUsage::OrdinaryWitness,
            }],
        };
        let private_column_scope_hash =
            canonical_zk_private_column_scope_hash(&private_column_scope);
        private_column_scope.hash = private_column_scope_hash;
        let trace_domain = CanonicCoset::new(trace_log_size).coset;
        let randomizer_space_hash = canonical_zk_randomizer_space_hash(
            private_column_scope_hash,
            &[ZkRandomizerSpaceEntry {
                range,
                trace_domain: ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(trace_domain)),
                randomized_log_degree,
                randomizer_dimension: h_witness,
            }],
        );
        let private_degree_bound = ZkColumnDegreeBound {
            range,
            log_degree_bound: randomized_log_degree,
        };
        let quotient_degree_bound = ZkColumnDegreeBound {
            range: quotient_range,
            log_degree_bound: fri_first_layer_log_size - log_blowup_factor,
        };
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash,
            public_statement_hash: ZkPublicStatementHash(nonzero_hash(2)),
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: trace_log_size,
                h_witness,
                h_batch,
                fri_first_layer_log_size,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness,
                randomizer_space_hash,
                private_column_scope_hash,
                private_column_degree_bounds: vec![private_degree_bound],
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch,
                fri_first_layer_log_size,
                split_derivation_hash: canonical_zk_split_derivation_hash(1),
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
            private_columns: vec![range],
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
            derivation_reviews: derivation_reviews(),
        };
        let verifier_config = ZkVerificationConfig {
            metadata,
            column_degree_bounds,
        };
        let verifier_audit = ZkWitnessRandomizationVerifierAudit {
            privacy_map,
            private_column_scope,
        };

        (prover_config, verifier_config, verifier_audit)
    }

    #[test]
    fn test_quotients_are_low_degree() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 4;

        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(M31::from).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let eval = polynomial.evaluate(eval_domain);

        let sample_points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = sample_points
            .into_iter()
            .map(|x| PointSample {
                point: x,
                value: polynomial.eval_at_point(x),
            })
            .collect_vec();
        let rand_coeff =
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));
        let quot_eval = compute_fri_quotients(
            &TreeVec(vec![vec![&eval]]),
            &TreeVec(vec![vec![samples]]),
            rand_coeff,
            LOG_SIZE + LOG_BLOWUP_FACTOR,
            &CpuBackend::precompute_twiddles(eval_domain.half_coset),
            LOG_BLOWUP_FACTOR,
        );
        let mut coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.clone()).interpolate())
            .collect_vec();
        let zeros = coeffs[0].coeffs.split_off((1 << LOG_SIZE) - 1);

        assert!(zeros.iter().all(|c| c.is_zero()));
    }

    /// Generates a vector of random polynomials, such that the last one is of degree
    /// `LIFTING_LOG_SIZE - 1` and all the previous ones are of degree < `LIFTING_LOG_SIZE - 1`.
    fn prepare_polys<B: Backend, const N_COLS: usize, const LIFTING_LOG_SIZE: u32>(
    ) -> Vec<CircleCoefficients<B>> {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut polys: Vec<CircleCoefficients<B>> = (0..N_COLS - 1)
            .map(|_| {
                CircleCoefficients::new(
                    (0..1 << rng.gen_range(4..LIFTING_LOG_SIZE - 1))
                        .map(M31::from)
                        .collect(),
                )
            })
            .collect();
        polys.push(CircleCoefficients::new(
            (0..1 << LIFTING_LOG_SIZE).map(M31::from).collect(),
        ));
        polys
    }

    fn prove_and_verify_pcs<
        B: BackendForChannel<Blake2sMerkleChannel>,
        const STORE_COEFFS: bool,
    >() -> Result<(), VerificationError> {
        const N_COLS: usize = 10;
        const LIFTING_LOG_SIZE: u32 = 8;

        // Setup the prover side of the pcs.
        let mut channel = Blake2sChannel::default();
        let config = PcsConfig::default();
        let twiddles = B::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(config, &twiddles);
        if STORE_COEFFS {
            commitment_scheme.set_store_polynomials_coefficients();
        }
        let polys = prepare_polys::<B, N_COLS, LIFTING_LOG_SIZE>();
        let sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut channel);

        let mut rng = SmallRng::seed_from_u64(0);
        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let samples = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![(0..N_COLS)
            .zip(mask_structure.iter())
            .map(|(_, i)| samples.into_iter().take(*i).collect_vec())
            .collect_vec()];

        let proof = commitment_scheme.prove_values(TreeVec(sampled_points.clone()), &mut channel);

        // Verifier side of the pcs.
        let mut channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(proof.proof.commitments[0], &sizes, &mut channel);
        verifier.verify_values(TreeVec(sampled_points), proof.proof, &mut channel)
    }

    fn prove_and_verify_zk_pcs<
        B: BackendForChannel<Blake2sMerkleChannel>,
        const STORE_COEFFS: bool,
    >() -> Result<(), VerificationError> {
        const N_COLS: usize = 10;
        const LIFTING_LOG_SIZE: u32 = 8;

        let mut channel = Blake2sChannel::default();
        let config = PcsConfig::default();
        let twiddles = B::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(config, &twiddles);
        if STORE_COEFFS {
            commitment_scheme.set_store_polynomials_coefficients();
        }
        let polys = prepare_polys::<B, N_COLS, LIFTING_LOG_SIZE>();
        let sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut channel);

        let lifting_log_size = commitment_scheme
            .trees
            .last()
            .unwrap()
            .commitment
            .layers
            .len() as u32
            - 1;
        let (zk_prover_config, zk_verifier_config) =
            zk_public_only_configs(lifting_log_size, config.fri_config.log_blowup_factor);

        let mut rng = SmallRng::seed_from_u64(0);
        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let samples = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![(0..N_COLS)
            .zip(mask_structure.iter())
            .map(|(_, i)| samples.into_iter().take(*i).collect_vec())
            .collect_vec()];

        let mut zk_rng = DeterministicTestCryptoRng::seed_from_u64(1);
        let proof = commitment_scheme
            .prove_values_zk(
                TreeVec(sampled_points.clone()),
                &zk_prover_config,
                &mut zk_rng,
                &mut channel,
            )
            .map_err(|err| VerificationError::InvalidStructure(format!("{err:?}")))?;

        let mut channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(
            proof.proof.randomized_pcs_proof.commitments[0],
            &sizes,
            &mut channel,
        );
        verifier.verify_values_zk(
            TreeVec(sampled_points),
            proof.proof,
            &zk_verifier_config,
            &mut channel,
        )
    }

    fn prove_and_verify_zk_private_witness_pcs<
        B: BackendForChannel<Blake2sMerkleChannel>,
        const STORE_COEFFS: bool,
    >() -> Result<(), VerificationError> {
        const N_COLS: usize = 10;
        const TRACE_LOG_SIZE: u32 = 8;
        const RANDOMIZED_LOG_DEGREE: u32 = TRACE_LOG_SIZE + 1;
        const PRIVATE_TREE_INDEX: usize = 1;

        let mut channel = Blake2sChannel::default();
        let config = PcsConfig::default();
        let fri_first_layer_log_size = RANDOMIZED_LOG_DEGREE + config.fri_config.log_blowup_factor;
        let twiddles =
            B::precompute_twiddles(CanonicCoset::new(fri_first_layer_log_size).half_coset());
        let mut commitment_scheme =
            CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(config, &twiddles);
        if STORE_COEFFS {
            commitment_scheme.set_store_polynomials_coefficients();
        }
        let preprocessed_polys = prepare_polys::<B, 1, TRACE_LOG_SIZE>();
        let preprocessed_sizes = preprocessed_polys
            .iter()
            .map(|poly| poly.log_size())
            .collect_vec();
        let witness_polys = prepare_polys::<B, N_COLS, TRACE_LOG_SIZE>();
        let mut witness_sizes = witness_polys
            .iter()
            .map(|poly| poly.log_size())
            .collect_vec();
        witness_sizes[0] = RANDOMIZED_LOG_DEGREE;
        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) = zk_private_witness_configs(
            PRIVATE_TREE_INDEX,
            TRACE_LOG_SIZE,
            RANDOMIZED_LOG_DEGREE,
            fri_first_layer_log_size,
            config.fri_config.log_blowup_factor,
        );

        let mut witness_rng = DeterministicTestCryptoRng::seed_from_u64(3);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(preprocessed_polys);
        tree_builder.commit(&mut channel);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(witness_polys);
        tree_builder
            .commit_zk_witness_randomized(&zk_prover_config, &mut witness_rng, &mut channel)
            .map_err(|err| VerificationError::InvalidStructure(format!("{err:?}")))?;

        let mut rng = SmallRng::seed_from_u64(0);
        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let samples = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![
            vec![samples.into_iter().take(1).collect_vec()],
            (0..N_COLS)
                .zip(mask_structure.iter())
                .map(|(_, i)| samples.into_iter().take(*i).collect_vec())
                .collect_vec(),
        ];

        let mut zk_rng = DeterministicTestCryptoRng::seed_from_u64(5);
        let proof = commitment_scheme
            .prove_values_zk(
                TreeVec(sampled_points.clone()),
                &zk_prover_config,
                &mut zk_rng,
                &mut channel,
            )
            .map_err(|err| VerificationError::InvalidStructure(format!("{err:?}")))?;

        let mut channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(
            proof.proof.randomized_pcs_proof.commitments[0],
            &preprocessed_sizes,
            &mut channel,
        );
        verifier.commit(
            proof.proof.randomized_pcs_proof.commitments[1],
            &witness_sizes,
            &mut channel,
        );
        verifier.verify_values_zk_with_witness_randomization_audit(
            TreeVec(sampled_points),
            proof.proof,
            &zk_verifier_config,
            &zk_verifier_audit,
            &mut channel,
        )
    }

    #[test]
    fn test_pcs_prove_and_verify_cpu() {
        assert!(prove_and_verify_pcs::<CpuBackend, true>().is_ok());
    }
    #[test]
    fn test_pcs_prove_and_verify_simd() {
        assert!(prove_and_verify_pcs::<SimdBackend, true>().is_ok());
    }
    #[test]
    fn test_pcs_prove_and_verify_simd_with_barycentric() {
        assert!(prove_and_verify_pcs::<SimdBackend, false>().is_ok());
    }

    #[test]
    fn test_zk_pcs_prove_and_verify_cpu() {
        assert!(prove_and_verify_zk_pcs::<CpuBackend, true>().is_ok());
    }

    #[test]
    fn test_zk_pcs_prove_and_verify_simd() {
        assert!(prove_and_verify_zk_pcs::<SimdBackend, true>().is_ok());
    }

    #[test]
    fn test_zk_private_witness_pcs_prove_and_verify_cpu() {
        assert!(prove_and_verify_zk_private_witness_pcs::<CpuBackend, true>().is_ok());
    }

    /// Tests that SIMD quotient computation produces low-degree quotients even when the trace
    /// polynomial is very small (subdomain size < N_LANES).
    #[test]
    fn test_simd_quotients_are_low_degree_small_trace() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 4;

        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(M31::from).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let cpu_eval = polynomial.evaluate(eval_domain);
        let simd_eval = CircleEvaluation::new(eval_domain, BaseColumn::from_cpu(&cpu_eval.values));

        let sample_points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = sample_points
            .into_iter()
            .map(|x| PointSample {
                point: x,
                value: polynomial.eval_at_point(x),
            })
            .collect_vec();
        let rand_coeff =
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));
        let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);
        let quot_eval = compute_fri_quotients(
            &TreeVec(vec![vec![&simd_eval]]),
            &TreeVec(vec![vec![samples]]),
            rand_coeff,
            LOG_SIZE + LOG_BLOWUP_FACTOR,
            &twiddles,
            LOG_BLOWUP_FACTOR,
        );
        let mut coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.to_cpu()).interpolate())
            .collect_vec();
        let zeros = coeffs[0].coeffs.split_off((1 << LOG_SIZE) - 1);

        assert!(zeros.iter().all(|c| c.is_zero()));
    }
}
