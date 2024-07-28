//! AIR for Poseidon2 hash function from <https://eprint.iacr.org/2023/323.pdf>.

use std::array;
use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{span, Level};

use crate::constraint_framework::constant_columns::gen_is_first;
use crate::constraint_framework::logup::{LogupTraceGenerator, LookupElements};
use crate::constraint_framework::{EvalAtRow, PointEvaluator, SimdDomainEvaluator};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::mask::fixed_mask_points;
use crate::core::air::{Air, AirProver, Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedBaseField, PackedM31, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::channel::{Blake2sChannel, Channel as _};
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::fields::{FieldExpOps, IntoSlice};
use crate::core::pcs::{CommitmentSchemeProver, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, StarkProof, VerificationError, LOG_BLOWUP_FACTOR};
use crate::core::utils::bit_reverse;
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::hasher::Hasher;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::trace_generation::{AirTraceGenerator, AirTraceVerifier, ComponentTraceGenerator};

const N_LOG_INSTANCES_PER_ROW: usize = 3;
const N_INSTANCES_PER_ROW: usize = 1 << N_LOG_INSTANCES_PER_ROW;
const N_STATE: usize = 16;
const N_PARTIAL_ROUNDS: usize = 14;
const N_HALF_FULL_ROUNDS: usize = 4;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
const N_COLUMNS_PER_REP: usize = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const N_COLUMNS: usize = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const LOG_EXPAND: u32 = 2;
// TODO(spapini): Pick better constants.
const EXTERNAL_ROUND_CONSTS: [[BaseField; N_STATE]; 2 * N_HALF_FULL_ROUNDS] =
    [[BaseField::from_u32_unchecked(1234); N_STATE]; 2 * N_HALF_FULL_ROUNDS];
const INTERNAL_ROUND_CONSTS: [BaseField; N_PARTIAL_ROUNDS] =
    [BaseField::from_u32_unchecked(1234); N_PARTIAL_ROUNDS];

#[derive(Clone)]
pub struct PoseidonComponent {
    pub log_n_rows: u32,
}

impl PoseidonComponent {
    pub fn log_column_size(&self) -> u32 {
        self.log_n_rows
    }

    pub fn n_columns(&self) -> usize {
        N_COLUMNS
    }
}

#[derive(Clone)]
pub struct PoseidonAir {
    pub component: PoseidonComponent,
}

impl Air for PoseidonAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }

    fn verify_lookups(&self, _lookup_values: &LookupValues) -> Result<(), VerificationError> {
        Ok(())
    }
}

impl AirTraceVerifier for PoseidonAir {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirTraceGenerator<SimdBackend> for PoseidonAir {
    fn interact(
        &self,
        _trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn to_air_prover(&self) -> impl AirProver<SimdBackend> {
        self.clone()
    }

    fn composition_log_degree_bound(&self) -> u32 {
        self.component.max_constraint_log_degree_bound()
    }
}

impl Component for PoseidonComponent {
    fn n_constraints(&self) -> usize {
        (N_COLUMNS_PER_REP - N_STATE) * N_INSTANCES_PER_ROW
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + LOG_EXPAND
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![
            vec![self.log_column_size(); N_COLUMNS],
            vec![self.log_column_size(); N_INSTANCES_PER_ROW * SECURE_EXTENSION_DEGREE],
            vec![self.log_column_size()],
        ])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![
            fixed_mask_points(&vec![vec![0_usize]; N_COLUMNS], point),
            vec![vec![]; N_INSTANCES_PER_ROW * SECURE_EXTENSION_DEGREE],
            vec![vec![point]],
        ])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        let mut poseidon_eval = PoseidonEval {
            eval: PointEvaluator::new(mask.as_ref(), evaluation_accumulator, denom_inverse),
        };
        for _ in 0..N_INSTANCES_PER_ROW {
            poseidon_eval.eval();
        }
        assert_eq!(poseidon_eval.eval.col_index[0], N_COLUMNS);
    }
}

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Copy + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0] + x[1];
    let t02 = t0 + t0;
    let t1 = x[2] + x[3];
    let t12 = t1 + t1;
    let t2 = x[1] + x[1] + t1;
    let t3 = x[3] + x[3] + t0;
    let t4 = t12 + t12 + t3;
    let t5 = t02 + t02 + t2;
    let t6 = t3 + t5;
    let t7 = t2 + t4;
    [t6, t5, t7, t4]
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix<F>(state: &mut [F; 16])
where
    F: Copy + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ]);
    }
    for j in 0..4 {
        let s = state[j] + state[j + 4] + state[j + 8] + state[j + 12];
        for i in 0..4 {
            state[4 * i + j] += s;
        }
    }
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Copy + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // TODO(spapini): Check that these coefficients are good according to section  5.3 of Poseidon2
    // paper.
    let sum = state[1..].iter().fold(state[0], |acc, s| acc + *s);
    state.iter_mut().enumerate().for_each(|(i, s)| {
        // TODO(spapini): Change to rotations.
        *s = *s * BaseField::from_u32_unchecked(1 << (i + 1)) + sum;
    });
}

fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

struct PoseidonEval<E: EvalAtRow> {
    eval: E,
}

impl<E: EvalAtRow> PoseidonEval<E> {
    fn eval(&mut self) {
        let mut state: [_; N_STATE] = std::array::from_fn(|_| self.eval.next_trace_mask());

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i]));
            state.iter_mut().for_each(|s| {
                let m = self.eval.next_trace_mask();
                self.eval.add_constraint(*s - m);
                *s = m;
            });
        });

        // Partial rounds.
        (0..N_PARTIAL_ROUNDS).for_each(|round| {
            state[0] += INTERNAL_ROUND_CONSTS[round];
            apply_internal_round_matrix(&mut state);
            state[0] = pow5(state[0]);
            let m = self.eval.next_trace_mask();
            self.eval.add_constraint(state[0] - m);
            state[0] = m;
        });

        // 4 full rounds.
        (0..N_HALF_FULL_ROUNDS).for_each(|round| {
            (0..N_STATE).for_each(|i| {
                state[i] += EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i];
            });
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i]));
            state.iter_mut().for_each(|s| {
                let m = self.eval.next_trace_mask();
                self.eval.add_constraint(*s - m);
                *s = m;
            });
        });
    }
}

impl AirProver<SimdBackend> for PoseidonAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![&self.component]
    }
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
                .for_each(|(res, state)| res.data[vec_index] = state);

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
                .for_each(|(res, state)| res.data[vec_index] = state);
        }
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec();
    (trace, lookup_data)
}

pub fn gen_interaction_trace(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: LookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    #[allow(clippy::needless_range_loop)]
    for rep_i in 0..N_INSTANCES_PER_ROW {
        let mut col_gen = logup_gen.new_col();
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            // Batch the 2 lookups together.
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
            // (1 / denom1) - (1 / denom1) = (denom1 - denom0) / (denom0 * denom1).
            col_gen.write_frac(vec_row, denom1 - denom0, denom0 * denom1);
        }
        col_gen.finalize_col();
    }

    logup_gen.finalize()
}

impl ComponentTraceGenerator<SimdBackend> for PoseidonComponent {
    type Component = Self;
    type Inputs = ();

    fn add_inputs(&mut self, _inputs: &Self::Inputs) {
        todo!()
    }

    fn write_trace(
        _component_id: &str,
        _registry: &mut crate::trace_generation::registry::ComponentGenerationRegistry,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        todo!()
    }

    fn write_interaction_trace(
        &self,
        _trace: &ColumnVec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn component(&self) -> Self::Component {
        todo!()
    }
}

impl ComponentProver<SimdBackend> for PoseidonComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        assert_eq!(trace.polys[0].len(), self.n_columns());
        let eval_domain = CanonicCoset::new(self.log_column_size() + LOG_EXPAND).circle_domain();

        // Create a new evaluation.
        let span = span!(Level::INFO, "Deg4 eval").entered();
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(self.max_constraint_log_degree_bound())
                .circle_domain()
                .half_coset,
        );
        let trace_eval = trace
            .polys
            .as_cols_ref()
            .map_cols(|col| col.evaluate_with_twiddles(eval_domain, &twiddles));
        let trace_eval_ref = trace_eval.as_ref().map(|t| t.iter().collect_vec());
        span.exit();

        // Denoms.
        let span = span!(Level::INFO, "Constraint eval denominators").entered();
        let zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let denoms_inv: [BaseField; 1 << LOG_EXPAND] =
            array::from_fn(|i| coset_vanishing(zero_domain, eval_domain.at(i)).inverse());
        let mut packed_denoms_inv = denoms_inv.map(PackedM31::broadcast);
        bit_reverse(&mut packed_denoms_inv);
        span.exit();

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();

        let constraint_log_degree_bound = self.max_constraint_log_degree_bound();
        let n_constraints = self.n_constraints();
        let [accum] =
            evaluation_accumulator.columns([(constraint_log_degree_bound, n_constraints)]);
        let mut pows = accum.random_coeff_powers.clone();
        pows.reverse();

        const CHUNK_SIZE: usize = 16;
        assert_eq!(accum.col.columns[0].length % (CHUNK_SIZE << LOG_N_LANES), 0);

        #[cfg(not(feature = "parallel"))]
        let iter = (0..(1 << (eval_domain.log_size() - LOG_N_LANES)))
            .step_by(CHUNK_SIZE)
            .zip(accum.col.chunks_mut(CHUNK_SIZE));

        #[cfg(feature = "parallel")]
        let iter = (0..(1 << (eval_domain.log_size() - LOG_N_LANES)))
            .into_par_iter()
            .step_by(CHUNK_SIZE)
            .zip(accum.col.chunks_mut(CHUNK_SIZE));

        iter.for_each(|(chunk_offset, mut col_chunk)| {
            for offset in 0..CHUNK_SIZE {
                let vec_row = chunk_offset + offset;
                let mut evaluator = PoseidonEval {
                    eval: SimdDomainEvaluator::new(
                        &trace_eval_ref,
                        vec_row,
                        &pows,
                        self.log_n_rows,
                        self.log_n_rows + LOG_EXPAND,
                    ),
                };
                for _ in 0..N_INSTANCES_PER_ROW {
                    evaluator.eval();
                }

                let packed_denom_inv =
                    packed_denoms_inv[vec_row >> (zero_domain.log_size() - LOG_N_LANES)];
                let quotient = evaluator.eval.row_res * packed_denom_inv;
                unsafe { col_chunk.set_packed(offset, col_chunk.packed_at(offset) + quotient) };
                assert_eq!(evaluator.eval.constraint_index, n_constraints);
            }
        });
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, SimdBackend>) -> LookupValues {
        LookupValues::default()
    }
}

pub fn prove_poseidon(log_n_instances: u32) -> (PoseidonAir, StarkProof) {
    assert!(log_n_instances >= N_LOG_INSTANCES_PER_ROW as u32);
    let log_n_rows = log_n_instances - N_LOG_INSTANCES_PER_ROW as u32;

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + LOG_EXPAND + LOG_BLOWUP_FACTOR)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    let commitment_scheme = &mut CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR, &twiddles);

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let (trace, lookup_data) = gen_trace(log_n_rows);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = LookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, _claimed_logup_sum) =
        gen_interaction_trace(log_n_rows, lookup_data, lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Constant trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![gen_is_first(log_n_rows)]);
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonComponent { log_n_rows };
    let air = PoseidonAir { component };
    let proof = prove::<SimdBackend>(
        &air,
        channel,
        &InteractionElements::default(),
        commitment_scheme,
    )
    .unwrap();

    (air, proof)
}

#[cfg(test)]
mod tests {
    use std::env;

    use itertools::Itertools;
    use num_traits::One;

    use crate::constraint_framework::assert_constraints;
    use crate::constraint_framework::constant_columns::gen_is_first;
    use crate::constraint_framework::logup::LookupElements;
    use crate::core::air::AirExt;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::pcs::{CommitmentSchemeVerifier, TreeVec};
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::prover::verify;
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::hasher::Hasher;
    use crate::core::InteractionElements;
    use crate::examples::poseidon::{
        apply_internal_round_matrix, apply_m4, gen_interaction_trace, gen_trace, prove_poseidon,
        PoseidonEval,
    };
    use crate::math::matrix::{RowMajorMatrix, SquareMatrix};
    use crate::qm31;

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
        let matrix = RowMajorMatrix::<BaseField, 16>::new(internal_matrix.flatten().to_vec());

        let expected_state = matrix.mul(state);
        apply_internal_round_matrix(&mut state);

        assert_eq!(state, expected_state);
    }

    #[test]
    fn test_poseidon_constraints() {
        const LOG_N_ROWS: u32 = 8;

        // Trace.
        let (trace0, interaction_data) = gen_trace(LOG_N_ROWS);
        let lookup_elements = LookupElements {
            z: qm31!(1, 2, 3, 4),
            alpha: qm31!(5, 6, 7, 8),
        };
        let (trace1, _claimed_logup_sum) =
            gen_interaction_trace(LOG_N_ROWS, interaction_data, lookup_elements);
        let trace2 = vec![gen_is_first(LOG_N_ROWS)];

        let traces = TreeVec::new(vec![trace0, trace1, trace2]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());
        assert_constraints(&trace_polys, CanonicCoset::new(LOG_N_ROWS), |eval| {
            PoseidonEval { eval }.eval();
        });
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

        // Prove.
        let (air, proof) = prove_poseidon(log_n_instances);

        // Verify.
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let commitment_scheme = &mut CommitmentSchemeVerifier::new();

        // Decommit.
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = air.column_log_sizes();
        // Trace columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Constant columns.
        commitment_scheme.commit(proof.commitments[2], &[air.component.log_n_rows], channel);

        verify(
            &air,
            channel,
            &InteractionElements::default(),
            commitment_scheme,
            proof,
        )
        .unwrap();
    }
}
