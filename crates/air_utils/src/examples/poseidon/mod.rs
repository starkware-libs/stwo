//! AIR for Poseidon2 hash function from <https://eprint.iacr.org/2023/323.pdf>.
#![allow(unused)]
use std::ops::{Add, AddAssign, Mul, Sub};

use num_traits::One;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use stwo_air_utils_derive::{IterMut, ParIterMut, Uninitialized};
use stwo_prover::constraint_framework::logup::LogupTraceGenerator;
use stwo_prover::constraint_framework::preprocessed_columns::gen_is_first;
use stwo_prover::constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, Relation, RelationEntry, TraceLocationAllocator,
};
use stwo_prover::core::backend::simd::m31::{PackedBaseField, PackedM31, LOG_N_LANES};
use stwo_prover::core::backend::simd::qm31::PackedSecureField;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::channel::Blake2sChannel;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::pcs::{CommitmentSchemeProver, PcsConfig};
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::prover::{prove, StarkProof};
use stwo_prover::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo_prover::core::ColumnVec;
use stwo_prover::relation;
use tracing::{info, span, Level};

use crate::trace::component_trace::ComponentTrace;

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
    pub total_sum: SecureField,
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

#[derive(Uninitialized, IterMut, ParIterMut)]
pub struct LookupData {
    initial_state: [Vec<[PackedM31; N_STATE]>; N_INSTANCES_PER_ROW],
    final_state: [Vec<[PackedM31; N_STATE]>; N_INSTANCES_PER_ROW],
}

pub fn gen_trace(
    log_size: u32,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    LookupData,
) {
    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);
    let mut trace = unsafe { ComponentTrace::<N_COLUMNS>::uninitialized(log_size) };
    let mut lookup_data = unsafe { LookupData::uninitialized(log_size - LOG_N_LANES) };

    trace
        .par_iter_mut()
        .zip(lookup_data.par_iter_mut())
        .enumerate()
        .for_each(|(vec_index, (mut row, lookup_data))| {
            // Initial state.
            let mut col_index = 0;
            for rep_i in 0..N_INSTANCES_PER_ROW {
                let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
                    PackedBaseField::from_array(std::array::from_fn(|i| {
                        BaseField::from_u32_unchecked((vec_index * 16 + i + state_i + rep_i) as u32)
                    }))
                });
                state.iter().copied().for_each(|s| {
                    *row[col_index] = s;
                    col_index += 1;
                });
                *lookup_data.initial_state[rep_i] = state;

                // 4 full rounds.
                (0..N_HALF_FULL_ROUNDS).for_each(|round| {
                    (0..N_STATE).for_each(|i| {
                        state[i] += PackedBaseField::broadcast(EXTERNAL_ROUND_CONSTS[round][i]);
                    });
                    apply_external_round_matrix(&mut state);
                    state = std::array::from_fn(|i| pow5(state[i]));
                    state.iter().copied().for_each(|s| {
                        *row[col_index] = s;
                        col_index += 1;
                    });
                });

                // Partial rounds.
                (0..N_PARTIAL_ROUNDS).for_each(|round| {
                    state[0] += PackedBaseField::broadcast(INTERNAL_ROUND_CONSTS[round]);
                    apply_internal_round_matrix(&mut state);
                    state[0] = pow5(state[0]);
                    *row[col_index] = state[0];
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
                        *row[col_index] = s;
                        col_index += 1;
                    });
                });

                *lookup_data.final_state[rep_i] = state;
            }
        });
    _span.exit();
    (trace.to_evals().to_vec(), lookup_data)
}

pub fn gen_interaction_trace(
    log_size: u32,
    mut lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    #[allow(clippy::needless_range_loop)]
    for rep_i in 0..N_INSTANCES_PER_ROW {
        let mut col_gen = logup_gen.new_col();
        for (vec_row, lookup_data) in lookup_data.iter_mut().enumerate() {
            // Batch the 2 lookups together.
            let denom0: PackedSecureField =
                lookup_elements.combine(lookup_data.initial_state[rep_i]);
            let denom1: PackedSecureField = lookup_elements.combine(lookup_data.final_state[rep_i]);
            // (1 / denom1) - (1 / denom1) = (denom1 - denom0) / (denom0 * denom1).
            col_gen.write_frac(vec_row, denom1 - denom0, denom0 * denom1);
        }
        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}

pub fn prove_poseidon(
    log_n_instances: u32,
    config: PcsConfig,
) -> (PoseidonComponent, StarkProof<Blake2sMerkleHasher>) {
    rayon::ThreadPoolBuilder::new()
        .stack_size(8388608)
        .build_global()
        .unwrap();
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

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let constant_trace = vec![gen_is_first(log_n_rows)];
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
    let (trace, total_sum) = gen_interaction_trace(log_n_rows, lookup_data, &lookup_elements);
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
            total_sum,
        },
        (total_sum, None),
    );
    info!("Poseidon component info:\n{}", component);
    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

#[cfg(test)]
mod tests {
    use std::env;

    use itertools::Itertools;
    use num_traits::One;
    use stwo_prover::constraint_framework::assert_constraints;
    use stwo_prover::constraint_framework::preprocessed_columns::gen_is_first;
    use stwo_prover::core::air::Component;
    use stwo_prover::core::channel::Blake2sChannel;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::fri::FriConfig;
    use stwo_prover::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo_prover::core::poly::circle::CanonicCoset;
    use stwo_prover::core::prover::verify;
    use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo_prover::math::matrix::{RowMajorMatrix, SquareMatrix};

    use crate::examples::poseidon::{
        apply_internal_round_matrix, apply_m4, eval_poseidon_constraints, gen_interaction_trace,
        gen_trace, prove_poseidon, PoseidonElements,
    };

    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn test_poseidon_prove_wasm() {
        const LOG_N_INSTANCES: u32 = 10;
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        // Prove.
        prove_poseidon(LOG_N_INSTANCES, config);
    }

    #[test]
    fn test_apply_m4() {
        let m4 = RowMajorMatrix::<BaseField, 4>::new(
            [5, 7, 1, 3, 4, 6, 1, 1, 1, 3, 5, 7, 1, 1, 4, 6]
                .map(BaseField::from_u32_unchecked)
                .into_iter()
                .collect_vec(),
        );
        let state = (0..4)
            .map(BaseField::from_u32_unchecked)
            .collect_vec()
            .try_into()
            .unwrap();

        assert_eq!(apply_m4(state), m4.mul(state));
    }

    #[test]
    fn test_apply_internal() {
        let mut state: [BaseField; 16] = (0..16)
            .map(|i| BaseField::from_u32_unchecked(i * 3 + 187))
            .collect_vec()
            .try_into()
            .unwrap();
        let mut internal_matrix = [[BaseField::one(); 16]; 16];
        #[allow(clippy::needless_range_loop)]
        for i in 0..16 {
            internal_matrix[i][i] += BaseField::from_u32_unchecked(1 << (i + 1));
        }
        let matrix = RowMajorMatrix::<BaseField, 16>::new(internal_matrix.as_flattened().to_vec());

        let expected_state = matrix.mul(state);
        apply_internal_round_matrix(&mut state);

        assert_eq!(state, expected_state);
    }

    #[test]
    fn test_poseidon_constraints() {
        const LOG_N_ROWS: u32 = 8;

        // Trace.
        let (trace0, interaction_data) = gen_trace(LOG_N_ROWS);
        let lookup_elements = PoseidonElements::dummy();
        let (trace1, total_sum) =
            gen_interaction_trace(LOG_N_ROWS, interaction_data, &lookup_elements);

        let traces = TreeVec::new(vec![vec![gen_is_first(LOG_N_ROWS)], trace0, trace1]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());
        assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_N_ROWS),
            |mut eval| {
                eval_poseidon_constraints(&mut eval, &lookup_elements);
            },
            (total_sum, None),
        );
    }

    #[test_log::test]
    fn test_simd_poseidon_par_prove() {
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
            fri_config: FriConfig::new(5, 1, 64),
        };

        // Prove.
        let (component, proof) = prove_poseidon(log_n_instances, config);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

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
}
