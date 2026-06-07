//! AIR for Poseidon2 hash function from <https://eprint.iacr.org/2023/323.pdf>.

pub mod zk;

use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
use num_traits::One;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, Relation,
    RelationEntry, TraceLocationAllocator,
};
use tracing::{info, span, Level};

const N_LOG_INSTANCES_PER_ROW: usize = 3;
const N_INSTANCES_PER_ROW: usize = 1 << N_LOG_INSTANCES_PER_ROW;
const N_STATE: usize = 16;
const N_PARTIAL_ROUNDS: usize = 14;
const N_HALF_FULL_ROUNDS: usize = 4;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
const N_COLUMNS_PER_REP: usize = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const N_COLUMNS: usize = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const LOG_EXPAND: u32 = 2;
// TODO(shahars): Use poseidon's real constants.
const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 2 * N_HALF_FULL_ROUNDS] =
    [[BaseField::from_u32_unchecked(1234); N_STATE]; 2 * N_HALF_FULL_ROUNDS];
const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] =
    [BaseField::from_u32_unchecked(1234); N_PARTIAL_ROUNDS];

pub type PoseidonComponent = FrameworkComponent<PoseidonEval>;

relation!(PoseidonElements, N_STATE);

#[derive(Clone)]
pub struct PoseidonEval {
    pub log_n_rows: u32,
    pub lookup_elements: PoseidonElements,
    pub claimed_sum: SecureField,
}
impl FrameworkEval for PoseidonEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_EXPAND
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_poseidon_constraints(&mut eval, &self.lookup_elements);
        eval
    }
}

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t12.clone() + t12.clone() + t3.clone();
    let t5 = t02.clone() + t02.clone() + t2.clone();
    let t6 = t3.clone() + t5.clone();
    let t7 = t2.clone() + t4.clone();
    [t6, t5, t7, t4]
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i].clone(),
            state[4 * i + 1].clone(),
            state[4 * i + 2].clone(),
            state[4 * i + 3].clone(),
        ]);
    }
    for j in 0..4 {
        let s =
            state[j].clone() + state[j + 4].clone() + state[j + 8].clone() + state[j + 12].clone();
        for i in 0..4 {
            state[4 * i + j] += s.clone();
        }
    }
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // TODO(shahars): Check that these coefficients are good according to section  5.3 of Poseidon2
    // paper.
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);
    state.iter_mut().enumerate().for_each(|(i, s)| {
        // TODO(andrew): Change to rotations.
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

pub fn eval_poseidon_constraints<E: EvalAtRow>(eval: &mut E, lookup_elements: &PoseidonElements) {
    for _ in 0..N_INSTANCES_PER_ROW {
        let mut state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

        // Require state lookup.
        let initial_state = state.clone();

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            });
            apply_external_round_matrix(&mut state);
            // TODO(andrew) Apply round matrix after the pow5, as is the order in the paper.
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            state.iter_mut().for_each(|s| {
                let m = eval.next_trace_mask();
                eval.add_constraint(s.clone() - m.clone());
                *s = m;
            });
        });

        // Partial rounds.
        (0..N_PARTIAL_ROUNDS).for_each(|round| {
            state[0] += INTERNAL_ROUND_CONSTS[round];
            apply_internal_round_matrix(&mut state);
            state[0] = pow5(state[0].clone());
            let m = eval.next_trace_mask();
            eval.add_constraint(state[0].clone() - m.clone());
            state[0] = m;
        });

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i].clone()));
            state.iter_mut().for_each(|s| {
                let m = eval.next_trace_mask();
                eval.add_constraint(s.clone() - m.clone());
                *s = m;
            });
        });

        // Provide state lookups.
        eval.add_to_relation(RelationEntry::new(
            lookup_elements,
            E::EF::one(),
            &initial_state,
        ));
        eval.add_to_relation(RelationEntry::new(lookup_elements, -E::EF::one(), &state));
    }

    eval.finalize_logup_in_pairs();
}

pub struct LookupData {
    initial_state: [[BaseColumn; N_STATE]; N_INSTANCES_PER_ROW],
    final_state: [[BaseColumn; N_STATE]; N_INSTANCES_PER_ROW],
}
pub fn gen_trace(
    log_size: u32,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    LookupData,
) {
    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);
    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    let mut lookup_data = LookupData {
        initial_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseColumn::zeros(1 << log_size))
        }),
        final_state: std::array::from_fn(|_| {
            std::array::from_fn(|_| BaseColumn::zeros(1 << log_size))
        }),
    };

    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        // Initial state.
        let mut col_index = 0;
        for rep_i in 0..N_INSTANCES_PER_ROW {
            let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
                PackedBaseField::from_array(std::array::from_fn(|i| {
                    BaseField::from_u32_unchecked((vec_index * 16 + i + state_i + rep_i) as u32)
                }))
            });
            state.iter().copied().for_each(|s| {
                trace[col_index].data[vec_index] = s;
                col_index += 1;
            });
            lookup_data.initial_state[rep_i]
                .iter_mut()
                .zip(state)
                .for_each(|(res, state_i)| res.data[vec_index] = state_i);

            // 4 full rounds.
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round][i]);
                });
                apply_external_round_matrix(&mut state);
                state = std::array::from_fn(|i| pow5(state[i]));
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
            });

            // Partial rounds.
            (0..N_PARTIAL_ROUNDS).for_each(|round| {
                state[0] += PackedBaseField::broadcast(INTERNAL_ROUND_CONSTS[round]);
                apply_internal_round_matrix(&mut state);
                state[0] = pow5(state[0]);
                trace[col_index].data[vec_index] = state[0];
                col_index += 1;
            });

            // 4 full rounds.
            (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                (0..N_STATE).for_each(|i| {
                    state[i] += PackedBaseField::broadcast(
                        EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i],
                    );
                });
                apply_external_round_matrix(&mut state);
                state = std::array::from_fn(|i| pow5(state[i]));
                state.iter().copied().for_each(|s| {
                    trace[col_index].data[vec_index] = s;
                    col_index += 1;
                });
            });

            lookup_data.final_state[rep_i]
                .iter_mut()
                .zip(state)
                .for_each(|(res, state_i)| res.data[vec_index] = state_i);
        }
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    (trace, lookup_data)
}

pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = unsafe { LogupTraceGenerator::uninitialized(log_size) };

    for rep_i in 0..N_INSTANCES_PER_ROW {
        let frac_at_row = |vec_row: usize| {
            let denom0: PackedSecureField = lookup_elements.combine(
                &lookup_data.initial_state[rep_i]
                    .each_ref()
                    .map(|s| s.data[vec_row]),
            );
            let denom1: PackedSecureField = lookup_elements.combine(
                &lookup_data.final_state[rep_i]
                    .each_ref()
                    .map(|s| s.data[vec_row]),
            );
            (denom1 - denom0, denom0 * denom1)
        };
        let range = 0..1 << (log_size - LOG_N_LANES);

        #[cfg(not(feature = "parallel"))]
        logup_gen.col_from_iter(range.map(frac_at_row));

        #[cfg(feature = "parallel")]
        logup_gen.col_from_par_iter(range.into_par_iter().map(frac_at_row));
    }

    logup_gen.finalize_last()
}

pub fn prove_poseidon(
    log_n_instances: u32,
    config: PcsConfig,
) -> (PoseidonComponent, StarkProof<Blake2sMerkleHasher>) {
    assert!(log_n_instances >= N_LOG_INSTANCES_PER_ROW as u32);
    let log_n_rows = log_n_instances - N_LOG_INSTANCES_PER_ROW as u32;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_store_polynomials_coefficients();

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let constant_trace = vec![];
    tree_builder.extend_evals(constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let (trace, lookup_data) = gen_trace(log_n_rows);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup elements.
    let lookup_elements = PoseidonElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, claimed_sum) = gen_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            log_n_rows,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );
    info!("Poseidon component info:\n{}", component);
    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

#[cfg(test)]
mod tests {
    use std::{array, env};

    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fri::FriConfig;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
    use stwo::core::verifier::{
        verify, verify_zk_with_witness_randomization_audit, VerificationError,
    };
    use stwo::core::zk::{
        build_stwo_zk_air_metadata, canonical_zk_privacy_map_hash,
        derive_stwo_zk_air_degree_bounds, zk_trace_domain_log_size_from_column_bounds,
        ZkAirMetadataBuildError, ZkColumnRange, ZkLogupClaimManifestEntry,
        ZkLogupClaimMetadataCompleteness, ZkLogupClaimPolicy, ZkLogupClaimVisibility,
        ZkPrivateColumnScopeEntry, ZkPrivateColumnUsage, ZkStarkProof, ZkTraceTreeScope,
        ZkTraceTreeScopeBinding, ZkVerificationConfig, ZkWitnessRandomizationVerifierAudit,
    };
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::poly::circle::PolyOps;
    use stwo::prover::zk::{ZkDerivationGate, ZkDerivationReview, ZkProvingConfig};
    use stwo::prover::{prove_zk, CommitmentSchemeProver};
    use stwo_constraint_framework::{assert_constraints_on_polys, TraceLocationAllocator};

    use crate::poseidon::{
        apply_internal_round_matrix, apply_m4, eval_poseidon_constraints, gen_interaction_trace,
        gen_trace, prove_poseidon, PoseidonComponent, PoseidonElements, PoseidonEval, LOG_EXPAND,
        N_LOG_INSTANCES_PER_ROW,
    };

    const ZK_POSEIDON_LOG_N_INSTANCES: u32 = 8;
    const ZK_POSEIDON_LOG_N_ROWS: u32 =
        ZK_POSEIDON_LOG_N_INSTANCES - N_LOG_INSTANCES_PER_ROW as u32;
    const ZK_POSEIDON_RANDOMIZED_LOG_DEGREE: u32 = ZK_POSEIDON_LOG_N_ROWS + 1;
    const ZK_POSEIDON_FRI_LOG_BLOWUP_FACTOR: u32 = 1;

    fn poseidon_zk_private_constraint_log_expansion(log_n_rows: u32) -> u32 {
        let component = poseidon_zk_metadata_component(log_n_rows);
        super::zk::poseidon_zk_private_constraint_log_expansion(
            &component,
            log_n_rows,
            super::zk::poseidon_zk_randomized_log_degree(log_n_rows),
        )
        .unwrap()
    }

    fn zk_poseidon_degree_bounds() -> super::zk::PoseidonZkDegreeBounds {
        super::zk::derive_poseidon_zk_degree_bounds_from_log_expansion(
            ZK_POSEIDON_LOG_N_ROWS,
            ZK_POSEIDON_RANDOMIZED_LOG_DEGREE,
            poseidon_zk_private_constraint_log_expansion(ZK_POSEIDON_LOG_N_ROWS),
            ZK_POSEIDON_FRI_LOG_BLOWUP_FACTOR,
        )
    }

    fn zk_poseidon_fri_first_layer_log_size() -> u32 {
        zk_poseidon_degree_bounds().fri_first_layer_log_size
    }
    fn poseidon_zk_metadata_component(log_n_rows: u32) -> PoseidonComponent {
        super::zk::poseidon_zk_metadata_component(log_n_rows)
    }

    fn poseidon_zk_canonical_metadata_result(
        component: &PoseidonComponent,
    ) -> Result<super::zk::PoseidonZkCanonicalMetadata, super::zk::PoseidonZkMetadataError> {
        super::zk::poseidon_zk_canonical_metadata(component, ZK_POSEIDON_FRI_LOG_BLOWUP_FACTOR)
    }

    fn poseidon_zk_canonical_metadata(
        component: &PoseidonComponent,
    ) -> super::zk::PoseidonZkCanonicalMetadata {
        poseidon_zk_canonical_metadata_result(component).unwrap()
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
            review_hash: test_hash(index as u8 + 80),
        })
        .collect()
    }

    fn private_poseidon_zk_configs(
        component: &PoseidonComponent,
    ) -> (
        ZkProvingConfig,
        ZkVerificationConfig,
        ZkWitnessRandomizationVerifierAudit,
    ) {
        let (prover_config, verifier_config, verifier_audit, _) = super::zk::poseidon_zk_configs(
            component,
            ZK_POSEIDON_FRI_LOG_BLOWUP_FACTOR,
            test_derivation_reviews(),
        )
        .unwrap();

        (prover_config, verifier_config, verifier_audit)
    }

    fn test_semantically_public_logup_claim_policy(interaction_index: u32) -> ZkLogupClaimPolicy {
        ZkLogupClaimPolicy {
            interaction_index,
            claim_index: 0,
            visibility: ZkLogupClaimVisibility::SemanticallyPublic,
            semantic_domain: b"stwo.examples.poseidon.test-public-logup-claim.v1".to_vec(),
            semantic_statement: b"test fixture declares this LogUp scalar public".to_vec(),
        }
    }

    fn verify_poseidon_zk_private_witness_proof_result(
        config: PcsConfig,
        component: &PoseidonComponent,
        proof: ZkStarkProof<Blake2sMerkleHasher>,
        zk_verifier_config: &ZkVerificationConfig,
        zk_verifier_audit: &ZkWitnessRandomizationVerifierAudit,
        bind_metadata_before_lookup: bool,
    ) -> Result<(), VerificationError> {
        let canonical_metadata = poseidon_zk_canonical_metadata(component);
        let verifier_channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        if proof.0.randomized_pcs_proof.commitments.len() < 3 {
            return Err(VerificationError::InvalidStructure(
                "Poseidon ZK proof is missing trace commitments".to_string(),
            ));
        }
        commitment_scheme.commit(
            proof.0.randomized_pcs_proof.commitments[0],
            &canonical_metadata.component_column_log_sizes[0],
            verifier_channel,
        );
        commitment_scheme.commit(
            proof.0.randomized_pcs_proof.commitments[1],
            &vec![
                canonical_metadata.randomized_witness_log_degree;
                canonical_metadata.component_column_log_sizes[1].len()
            ],
            verifier_channel,
        );
        if bind_metadata_before_lookup {
            super::zk::mix_poseidon_zk_verifier_metadata_before_lookup(
                verifier_channel,
                zk_verifier_config,
            );
        }
        let lookup_elements = PoseidonElements::draw(verifier_channel);
        if lookup_elements != component.lookup_elements {
            return Err(VerificationError::InvalidStructure(
                "Poseidon lookup challenge mismatch".to_string(),
            ));
        }
        commitment_scheme.commit(
            proof.0.randomized_pcs_proof.commitments[2],
            &vec![
                canonical_metadata.randomized_witness_log_degree;
                canonical_metadata.component_column_log_sizes[2].len()
            ],
            verifier_channel,
        );

        verify_zk_with_witness_randomization_audit(
            &[component],
            verifier_channel,
            commitment_scheme,
            proof,
            zk_verifier_config,
            zk_verifier_audit,
        )
    }

    fn verify_poseidon_zk_private_witness_proof(
        config: PcsConfig,
        component: &PoseidonComponent,
        proof: ZkStarkProof<Blake2sMerkleHasher>,
        zk_verifier_config: &ZkVerificationConfig,
        zk_verifier_audit: &ZkWitnessRandomizationVerifierAudit,
    ) {
        verify_poseidon_zk_private_witness_proof_result(
            config,
            component,
            proof,
            zk_verifier_config,
            zk_verifier_audit,
            true,
        )
        .unwrap();
    }

    struct PoseidonZkPrivateProofFixture {
        config: PcsConfig,
        component: PoseidonComponent,
        proof: ZkStarkProof<Blake2sMerkleHasher>,
        zk_verifier_config: ZkVerificationConfig,
        zk_verifier_audit: ZkWitnessRandomizationVerifierAudit,
    }

    fn poseidon_zk_test_pcs_config() -> PcsConfig {
        PcsConfig {
            fri_config: FriConfig::new(ZK_POSEIDON_FRI_LOG_BLOWUP_FACTOR, 1, 3, 1),
            lifting_log_size: Some(zk_poseidon_fri_first_layer_log_size()),
            ..PcsConfig::default()
        }
    }

    fn prove_poseidon_zk_private_witness_for_test(
        witness_rng_seed: u64,
        proof_rng_seed: u64,
    ) -> PoseidonZkPrivateProofFixture {
        let config = poseidon_zk_test_pcs_config();
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(zk_poseidon_fri_first_layer_log_size()).half_coset(),
        );
        let metadata_component = poseidon_zk_metadata_component(ZK_POSEIDON_LOG_N_ROWS);
        let (zk_prover_config, zk_verifier_config, zk_verifier_audit) =
            private_poseidon_zk_configs(&metadata_component);
        let prover_channel = &mut Blake2sChannel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);
        commitment_scheme.set_store_polynomials_coefficients();

        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(vec![]);
        tree_builder.commit(prover_channel);

        let (trace, lookup_data) = gen_trace(ZK_POSEIDON_LOG_N_ROWS);
        let mut witness_rng = StdRng::seed_from_u64(witness_rng_seed);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder
            .commit_zk_witness_randomized(&zk_prover_config, &mut witness_rng, prover_channel)
            .unwrap();

        super::zk::mix_poseidon_zk_prover_metadata_before_lookup(prover_channel, &zk_prover_config);
        let lookup_elements = PoseidonElements::draw(prover_channel);
        let (trace, claimed_sum) =
            gen_interaction_trace(ZK_POSEIDON_LOG_N_ROWS, lookup_data, &lookup_elements);
        let mut interaction_rng = StdRng::seed_from_u64(witness_rng_seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder
            .commit_zk_witness_randomized(&zk_prover_config, &mut interaction_rng, prover_channel)
            .unwrap();

        let component = PoseidonComponent::new(
            &mut TraceLocationAllocator::default(),
            PoseidonEval {
                log_n_rows: ZK_POSEIDON_LOG_N_ROWS,
                lookup_elements,
                claimed_sum,
            },
            claimed_sum,
        );
        assert_eq!(
            &component.trace_log_degree_bounds().0,
            &metadata_component.trace_log_degree_bounds().0
        );
        let mut proof_rng = StdRng::seed_from_u64(proof_rng_seed);
        let proof = prove_zk::<SimdBackend, Blake2sMerkleChannel, _>(
            &[&component],
            prover_channel,
            commitment_scheme,
            &zk_prover_config,
            &mut proof_rng,
        )
        .unwrap();

        PoseidonZkPrivateProofFixture {
            config,
            component,
            proof,
            zk_verifier_config,
            zk_verifier_audit,
        }
    }

    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn test_poseidon_prove_wasm() {
        const LOG_N_INSTANCES: u32 = 10;
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64, 1),
            lifting_log_size: None,
            hiding: None,
        };

        // Prove.
        prove_poseidon(LOG_N_INSTANCES, config);
    }

    #[test]
    fn test_apply_m4() {
        let m4 = ndarray::arr2(&[
            [5, 7, 1, 3].map(M31),
            [4, 6, 1, 1].map(M31),
            [1, 3, 5, 7].map(M31),
            [1, 1, 4, 6].map(M31),
        ]);
        let state = [0, 1, 2, 3].map(M31);
        let expected_dot = m4.dot(&ndarray::arr2(&[state]).t());
        let expected_dot: [_; 4] = expected_dot.into_raw_vec_and_offset().0.try_into().unwrap();

        let actual_dot = apply_m4(state);

        assert_eq!(expected_dot, actual_dot);
    }

    #[test]
    fn test_apply_internal() {
        const W: usize = 16;
        let mut state = array::from_fn(|i| M31((i * 3 + 187) as u32));
        let mut internal_matrix = ndarray::arr2(&[[M31(1); W]; W]);
        for (i, elem) in internal_matrix.diag_mut().iter_mut().enumerate() {
            *elem += M31((1 << (i + 1)) as u32);
        }
        let expected_state = internal_matrix.dot(&ndarray::arr2(&[state]).t());
        let expected_state: [_; W] = expected_state
            .into_raw_vec_and_offset()
            .0
            .try_into()
            .unwrap();

        apply_internal_round_matrix(&mut state);

        assert_eq!(state, expected_state);
    }

    #[test]
    fn test_poseidon_constraints() {
        const LOG_N_ROWS: u32 = 8;

        // Trace.
        let (trace0, interaction_data) = gen_trace(LOG_N_ROWS);
        let lookup_elements = PoseidonElements::dummy();
        let (trace1, claimed_sum) =
            gen_interaction_trace(LOG_N_ROWS, interaction_data, &lookup_elements);

        let traces = TreeVec::new(vec![vec![], trace0, trace1]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());
        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_ROWS),
            |mut eval| {
                eval_poseidon_constraints(&mut eval, &lookup_elements);
            },
            claimed_sum,
        );
    }

    #[test_log::test]
    fn test_simd_poseidon_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_poseidon_prove -- --nocapture

        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64, 1),
            lifting_log_size: None,
            hiding: None,
        };

        // Prove.
        let (component, proof) = prove_poseidon(log_n_instances, config);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(proof.config);

        // Decommit.
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Trace columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup element.
        let lookup_elements = PoseidonElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(&[&component], channel, commitment_scheme, proof).unwrap();
    }

    #[test]
    fn test_poseidon_zk_degree_bounds_are_formula_driven() {
        let bounds = zk_poseidon_degree_bounds();

        assert_eq!(bounds.trace_log_degree, ZK_POSEIDON_LOG_N_ROWS);
        assert_eq!(
            bounds.randomized_private_column_log_degree,
            ZK_POSEIDON_RANDOMIZED_LOG_DEGREE
        );
        assert_eq!(bounds.public_air_constraint_log_expansion, LOG_EXPAND);
        assert_eq!(
            bounds.private_constraint_log_expansion,
            super::zk::poseidon_zk_masked_private_constraint_log_expansion(
                LOG_EXPAND,
                ZK_POSEIDON_LOG_N_ROWS,
                ZK_POSEIDON_RANDOMIZED_LOG_DEGREE,
            )
        );
        assert_eq!(
            bounds.full_composition_log_degree_bound,
            bounds.randomized_private_column_log_degree + bounds.private_constraint_log_expansion
        );
        assert_eq!(
            bounds.split_composition_log_degree_bound,
            bounds.full_composition_log_degree_bound - bounds.composition_log_split
        );
        assert_eq!(
            bounds.left_masked_split_log_degree_bound,
            bounds.full_composition_log_degree_bound
        );
        assert_eq!(
            bounds.right_masked_split_log_degree_bound,
            bounds.split_composition_log_degree_bound
        );
        assert_eq!(
            bounds.fri_first_layer_log_size,
            bounds.left_masked_split_log_degree_bound + bounds.fri_log_blowup_factor
        );
    }

    #[test]
    fn test_poseidon_zk_canonical_metadata_rejects_private_logup_claim_leakage() {
        let component = poseidon_zk_metadata_component(ZK_POSEIDON_LOG_N_ROWS);

        assert!(matches!(
            poseidon_zk_canonical_metadata_result(&component),
            Err(super::zk::PoseidonZkMetadataError::Air(
                ZkAirMetadataBuildError::IncompleteLogupClaimMetadata
            ))
        ));
    }

    #[test]
    fn test_poseidon_zk_reusable_api_rejects_private_logup_claim_leakage() {
        let component = super::zk::poseidon_zk_metadata_component(ZK_POSEIDON_LOG_N_ROWS);
        assert!(matches!(
            super::zk::poseidon_zk_configs(
                &component,
                ZK_POSEIDON_FRI_LOG_BLOWUP_FACTOR,
                test_derivation_reviews(),
            ),
            Err(super::zk::PoseidonZkMetadataError::Air(
                ZkAirMetadataBuildError::IncompleteLogupClaimMetadata
            ))
        ));
    }

    #[test]
    fn test_generic_zk_air_metadata_builder_covers_multiple_interactions_and_fails_closed() {
        let component_bounds = TreeVec::new(vec![vec![], vec![5, 5], vec![5], vec![5, 5]]);
        let trace_domain_log_size =
            zk_trace_domain_log_size_from_column_bounds(&component_bounds).unwrap();
        let randomized_log_degree = trace_domain_log_size + 1;
        let private_expansion = super::zk::poseidon_zk_masked_private_constraint_log_expansion(
            LOG_EXPAND,
            trace_domain_log_size,
            randomized_log_degree,
        );
        let degree_bounds = derive_stwo_zk_air_degree_bounds(
            trace_domain_log_size,
            randomized_log_degree,
            LOG_EXPAND,
            private_expansion,
            ZK_POSEIDON_FRI_LOG_BLOWUP_FACTOR,
            stwo::core::verifier::COMPOSITION_LOG_SPLIT,
        )
        .unwrap();
        let scope_bindings = vec![
            ZkTraceTreeScopeBinding {
                tree_index: 0,
                scope: ZkTraceTreeScope::Preprocessed,
                column_log_degree_bounds: component_bounds[0].clone(),
            },
            ZkTraceTreeScopeBinding {
                tree_index: 1,
                scope: ZkTraceTreeScope::OriginalTrace,
                column_log_degree_bounds: component_bounds[1].clone(),
            },
            ZkTraceTreeScopeBinding {
                tree_index: 2,
                scope: ZkTraceTreeScope::InteractionTrace {
                    interaction_index: 0,
                },
                column_log_degree_bounds: component_bounds[2].clone(),
            },
            ZkTraceTreeScopeBinding {
                tree_index: 3,
                scope: ZkTraceTreeScope::InteractionTrace {
                    interaction_index: 1,
                },
                column_log_degree_bounds: component_bounds[3].clone(),
            },
        ];
        let private_scope_entry =
            |range: ZkColumnRange, usage: ZkPrivateColumnUsage| ZkPrivateColumnScopeEntry {
                range,
                usage,
                trace_domain_log_size: component_bounds[range.tree_index][range.column_start],
                semantic_trace_domain_log_sizes: vec![
                    component_bounds[range.tree_index][range.column_start],
                ],
            };
        let private_scope_entries = vec![
            private_scope_entry(
                ZkColumnRange::new(1, 0, 1),
                ZkPrivateColumnUsage::OrdinaryWitness,
            ),
            private_scope_entry(
                ZkColumnRange::new(1, 1, 2),
                ZkPrivateColumnUsage::OrdinaryWitness,
            ),
            private_scope_entry(ZkColumnRange::new(2, 0, 1), ZkPrivateColumnUsage::LogUp),
            private_scope_entry(ZkColumnRange::new(3, 0, 1), ZkPrivateColumnUsage::LogUp),
            private_scope_entry(ZkColumnRange::new(3, 1, 2), ZkPrivateColumnUsage::LogUp),
        ];

        let metadata = build_stwo_zk_air_metadata(
            component_bounds.clone(),
            trace_domain_log_size,
            randomized_log_degree,
            degree_bounds,
            scope_bindings.clone(),
            vec![],
            private_scope_entries.clone(),
            ZkLogupClaimMetadataCompleteness::Complete,
            vec![
                ZkLogupClaimManifestEntry {
                    interaction_index: 0,
                    claim_count: 1,
                },
                ZkLogupClaimManifestEntry {
                    interaction_index: 1,
                    claim_count: 1,
                },
            ],
            vec![
                test_semantically_public_logup_claim_policy(0),
                test_semantically_public_logup_claim_policy(1),
            ],
            b"stwo.examples.test.zk-air-metadata.v1",
            b"two-logup-interaction-trees",
        )
        .unwrap();
        assert_eq!(metadata.private_ranges.len(), 5);
        assert_eq!(
            metadata.trace_tree_scope_bindings.len(),
            component_bounds.len() + 1
        );
        assert!(metadata
            .trace_tree_scope_bindings
            .iter()
            .any(|binding| binding.scope
                == ZkTraceTreeScope::InteractionTrace {
                    interaction_index: 1
                }));

        let omitted_interaction_range = ZkColumnRange::new(3, 1, 2);
        let mut omitted_private_scope_entries = private_scope_entries;
        omitted_private_scope_entries.retain(|entry| entry.range != omitted_interaction_range);
        assert_eq!(
            build_stwo_zk_air_metadata(
                component_bounds,
                trace_domain_log_size,
                randomized_log_degree,
                degree_bounds,
                scope_bindings,
                vec![],
                omitted_private_scope_entries,
                ZkLogupClaimMetadataCompleteness::Complete,
                vec![
                    ZkLogupClaimManifestEntry {
                        interaction_index: 0,
                        claim_count: 1,
                    },
                    ZkLogupClaimManifestEntry {
                        interaction_index: 1,
                        claim_count: 1,
                    },
                ],
                vec![
                    test_semantically_public_logup_claim_policy(0),
                    test_semantically_public_logup_claim_policy(1),
                ],
                b"stwo.examples.test.zk-air-metadata.v1",
                b"two-logup-interaction-trees",
            )
            .unwrap_err(),
            ZkAirMetadataBuildError::MissingPrivateInteractionColumn {
                range: omitted_interaction_range,
            }
        );
    }

    #[test]
    fn test_poseidon_zk_logup_interaction_columns_require_private_claim_protocol() {
        let component = poseidon_zk_metadata_component(ZK_POSEIDON_LOG_N_ROWS);
        assert!(ZkPrivateColumnUsage::LogUp.eligible_for_witness_randomization());
        assert!(matches!(
            poseidon_zk_canonical_metadata_result(&component),
            Err(super::zk::PoseidonZkMetadataError::Air(
                ZkAirMetadataBuildError::IncompleteLogupClaimMetadata
            ))
        ));
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_private_proof_surface_does_not_dump_full_private_trace() {
        let fixture = prove_poseidon_zk_private_witness_for_test(701, 702);
        let metadata = poseidon_zk_canonical_metadata(&fixture.component);
        let randomized_pcs_proof = &fixture.proof.0.randomized_pcs_proof;
        let trace_row_count = 1usize << metadata.trace_domain_log_size;

        assert_eq!(
            randomized_pcs_proof.sampled_values[1].len(),
            metadata.component_column_log_sizes[1].len()
        );
        assert!(randomized_pcs_proof.sampled_values[1]
            .iter()
            .all(|column| column.len() < trace_row_count));
        assert!(randomized_pcs_proof.queried_values[1]
            .iter()
            .all(|column| column.len() < trace_row_count));
        assert_eq!(
            randomized_pcs_proof.sampled_values[2].len(),
            metadata.component_column_log_sizes[2].len()
        );
        assert!(randomized_pcs_proof.sampled_values[2]
            .iter()
            .all(|column| column.len() < trace_row_count));
        assert!(randomized_pcs_proof.queried_values[2]
            .iter()
            .all(|column| column.len() < trace_row_count));
        assert_eq!(
            fixture.proof.0.public_metadata.privacy_map_hash,
            fixture.zk_verifier_audit.privacy_map.hash
        );
        assert_eq!(
            fixture
                .proof
                .0
                .public_metadata
                .witness_randomization
                .private_column_scope_hash,
            fixture.zk_verifier_audit.private_column_scope.hash
        );
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_wrong_privacy_map() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(801, 802);
        fixture.zk_verifier_audit.privacy_map.hash.0[0] ^= 1;

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_omitted_private_logup_range() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(803, 804);
        let metadata = poseidon_zk_canonical_metadata(&fixture.component);
        let omitted = metadata.interaction_trace_private_ranges[0];
        fixture
            .zk_verifier_audit
            .privacy_map
            .private_columns
            .retain(|range| *range != omitted);
        fixture.zk_verifier_audit.privacy_map.hash =
            canonical_zk_privacy_map_hash(&fixture.zk_verifier_audit.privacy_map);

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_wrong_logup_private_scope() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(825, 826);
        let metadata = poseidon_zk_canonical_metadata(&fixture.component);
        let interaction_range = metadata.interaction_trace_private_ranges[0];
        let entry = fixture
            .zk_verifier_audit
            .private_column_scope
            .entries
            .iter_mut()
            .find(|entry| entry.range == interaction_range)
            .expect("interaction range must be scoped");
        entry.usage = ZkPrivateColumnUsage::OrdinaryWitness;

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_wrong_public_statement_metadata() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(805, 806);
        fixture.zk_verifier_config.metadata.public_statement_hash.0[0] ^= 1;

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_underbudgeted_degree_metadata() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(807, 808);
        fixture
            .zk_verifier_config
            .metadata
            .degree_profile
            .fri_first_layer_log_size -= 1;
        fixture
            .zk_verifier_config
            .metadata
            .quotient_integration
            .fri_first_layer_log_size -= 1;

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_missing_prelookup_metadata_binding() {
        let fixture = prove_poseidon_zk_private_witness_for_test(809, 810);

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            false,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_malformed_commitment_structure_without_panic() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(813, 814);
        fixture.proof.0.randomized_pcs_proof.commitments.pop();

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_wrong_quotient_mask_profile() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(811, 812);
        let mut profile = fixture
            .zk_verifier_config
            .quotient_split_mask_profile
            .expect("Poseidon ZK verifier config must include quotient split profile");
        profile.right_masked_log_degree_bound += 1;
        fixture.zk_verifier_config.quotient_split_mask_profile = Some(profile);

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_tampered_masked_opening_after_transcript_fixed() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(821, 822);
        fixture.proof.0.randomized_pcs_proof.sampled_values[1][0][0] += SecureField::from(M31(1));

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_verifier_rejects_tampered_masked_logup_opening_after_transcript_fixed() {
        let mut fixture = prove_poseidon_zk_private_witness_for_test(823, 824);
        fixture.proof.0.randomized_pcs_proof.sampled_values[2][0][0] += SecureField::from(M31(1));

        assert!(verify_poseidon_zk_private_witness_proof_result(
            fixture.config,
            &fixture.component,
            fixture.proof,
            &fixture.zk_verifier_config,
            &fixture.zk_verifier_audit,
            true,
        )
        .is_err());
    }

    #[ignore = "Blocked: private LogUp claimed_sum is witness-derived until a reviewed private-claim protocol replaces public scalar claims."]
    #[test]
    fn test_poseidon_zk_private_witness_repeated_proofs_hide_private_material() {
        let fixture_a = prove_poseidon_zk_private_witness_for_test(401, 402);
        let fixture_b = prove_poseidon_zk_private_witness_for_test(501, 502);
        let PoseidonZkPrivateProofFixture {
            config: config_a,
            component: component_a,
            proof: proof_a,
            zk_verifier_config: zk_verifier_config_a,
            zk_verifier_audit: zk_verifier_audit_a,
        } = fixture_a;
        let PoseidonZkPrivateProofFixture {
            config: config_b,
            component: component_b,
            proof: proof_b,
            zk_verifier_config: zk_verifier_config_b,
            zk_verifier_audit: zk_verifier_audit_b,
        } = fixture_b;

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
            proof_a.0.randomized_pcs_proof.commitments[3],
            proof_b.0.randomized_pcs_proof.commitments[3]
        );
        assert_ne!(
            proof_a.0.randomized_pcs_proof.sampled_values[1],
            proof_b.0.randomized_pcs_proof.sampled_values[1]
        );
        assert_ne!(
            proof_a.0.randomized_pcs_proof.sampled_values[2],
            proof_b.0.randomized_pcs_proof.sampled_values[2]
        );
        assert_ne!(
            proof_a.0.fri_batch_mask.commitment,
            proof_b.0.fri_batch_mask.commitment
        );

        verify_poseidon_zk_private_witness_proof(
            config_a,
            &component_a,
            proof_a,
            &zk_verifier_config_a,
            &zk_verifier_audit_a,
        );
        verify_poseidon_zk_private_witness_proof(
            config_b,
            &component_b,
            proof_b,
            &zk_verifier_config_b,
            &zk_verifier_audit_b,
        );
    }

    #[ignore = "AIRs with constraint degree >= 2 are not supported yet in the lifted protocol."]
    #[cfg(feature = "tracing")]
    #[test]
    fn trace_simd_poseidon_prove() {
        use stwo::tracing::SpanAccumulator;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::Registry;

        let collector = SpanAccumulator::default();
        let layer = collector.clone();
        let subscriber = Registry::default().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64, 1),
            lifting_log_size: None,
            hiding: None,
        };

        // Prove.
        let _ = prove_poseidon(log_n_instances, config);

        let csv = collector.export_csv();

        println!("{csv}");
    }
}
