use itertools::Itertools;
use num_traits::{One, Zero};
use tracing::{span, Level};

use super::component::LOG_N_COLUMNS;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::mask::fixed_mask_points;
use crate::core::air::{Air, AirProver, Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::channel::Channel;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::VerificationError;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::wide_fibonacci::component::N_COLUMNS;
use crate::trace_generation::registry::ComponentGenerationRegistry;
use crate::trace_generation::{
    AirTraceGenerator, AirTraceVerifier, ComponentTraceGenerator, BASE_TRACE,
};

// TODO(AlonH): Remove this once the Cpu and Simd implementations are aligned.
#[derive(Clone)]
pub struct SimdWideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl SimdWideFibComponent {
    /// Returns the log of the size of the columns in the trace (which could also be looked at as
    /// the log number of rows).
    pub fn log_column_size(&self) -> u32 {
        self.log_n_instances + self.log_fibonacci_size - LOG_N_COLUMNS as u32
    }

    pub fn log_n_columns(&self) -> usize {
        LOG_N_COLUMNS
    }

    pub fn n_columns(&self) -> usize {
        N_COLUMNS
    }
}

// TODO(AlonH): Remove this once the Cpu and Simd implementations are aligned.
#[derive(Clone)]
pub struct SimdWideFibAir {
    pub component: SimdWideFibComponent,
}

impl Air for SimdWideFibAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }
}

impl AirTraceVerifier for SimdWideFibAir {
    fn interaction_elements(&self, _channel: &mut impl Channel) -> InteractionElements {
        InteractionElements::default()
    }

    fn verify_lookups(&self, _lookup_values: &LookupValues) -> Result<(), VerificationError> {
        Ok(())
    }
}

impl AirTraceGenerator<SimdBackend> for SimdWideFibAir {
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

impl Component for SimdWideFibComponent {
    fn n_constraints(&self) -> usize {
        self.n_columns() - 2
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![self.log_column_size(); self.n_columns()]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![fixed_mask_points(
            &vec![vec![0_usize]; self.n_columns()],
            point,
        )])
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
        for i in 0..self.n_columns() - 2 {
            let numerator = mask[0][i][0].square() + mask[0][i + 1][0].square() - mask[0][i + 2][0];
            evaluation_accumulator.accumulate(numerator * denom_inverse);
        }
    }
}

impl AirProver<SimdBackend> for SimdWideFibAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![&self.component]
    }
}

pub fn gen_trace(
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    assert!(log_size >= LOG_N_LANES);
    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for vec_index in 0..(1 << (log_size - LOG_N_LANES)) {
        let mut a = PackedBaseField::one();
        let mut b = PackedBaseField::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked((vec_index * 16 + i) as u32)
        }));
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

impl ComponentTraceGenerator<SimdBackend> for SimdWideFibComponent {
    type Component = Self;
    type Inputs = ();

    fn add_inputs(&mut self, _inputs: &Self::Inputs) {}

    fn write_trace(
        _component_id: &str,
        _registry: &mut ComponentGenerationRegistry,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn write_interaction_trace(
        &self,
        _trace: &ColumnVec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn component(&self) -> Self::Component {
        self.clone()
    }
}

impl ComponentProver<SimdBackend> for SimdWideFibComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        assert_eq!(trace.polys[BASE_TRACE].len(), self.n_columns());
        // TODO(spapini): Steal evaluation from commitment.
        let eval_domain = CanonicCoset::new(self.log_column_size() + 1).circle_domain();
        let trace_eval = &trace.evals;

        // Denoms.
        let span = span!(Level::INFO, "Constraint eval denominators").entered();
        // TODO(spapini): Make this prettier.
        let zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let mut denoms =
            BaseColumn::from_iter(eval_domain.iter().map(|p| coset_vanishing(zero_domain, p)));
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut denoms);
        let mut denom_inverses = BaseColumn::zeros(denoms.len());
        <SimdBackend as FieldOps<BaseField>>::batch_inverse(&denoms, &mut denom_inverses);
        span.exit();

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();

        let constraint_log_degree_bound = self.max_constraint_log_degree_bound();
        let n_constraints = self.n_constraints();
        let [accum] =
            evaluation_accumulator.columns([(constraint_log_degree_bound, n_constraints)]);

        for vec_row in 0..(1 << (eval_domain.log_size() - LOG_N_LANES)) {
            // Numerator.
            let a = trace_eval[BASE_TRACE][0].data[vec_row];
            let mut row_res = PackedSecureField::zero();
            let mut a_sq = a.square();
            let mut b_sq = trace_eval[BASE_TRACE][1].data[vec_row].square();
            #[allow(clippy::needless_range_loop)]
            for i in 0..(self.n_columns() - 2) {
                unsafe {
                    let c = *trace_eval[BASE_TRACE]
                        .get_unchecked(i + 2)
                        .data
                        .get_unchecked(vec_row);
                    row_res += PackedSecureField::broadcast(
                        accum.random_coeff_powers[self.n_columns() - 3 - i],
                    ) * (a_sq + b_sq - c);
                    (a_sq, b_sq) = (b_sq, c.square());
                }
            }

            unsafe {
                accum.col.set_packed(
                    vec_row,
                    accum.col.packed_at(vec_row) + row_res * denom_inverses.data[vec_row],
                )
            }
        }
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, SimdBackend>) -> LookupValues {
        LookupValues::default()
    }
}

#[cfg(test)]
mod tests {
    use tracing::{span, Level};

    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::examples::wide_fibonacci::component::LOG_N_COLUMNS;
    use crate::examples::wide_fibonacci::simd::{gen_trace, SimdWideFibAir, SimdWideFibComponent};
    use crate::trace_generation::{commit_and_prove, commit_and_verify};

    #[test_log::test]
    fn test_simd_wide_fib_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_wide_fib_prove -- --nocapture

        // Note: 17 means 128MB of trace.
        const LOG_N_ROWS: u32 = 12;
        let component = SimdWideFibComponent {
            log_fibonacci_size: LOG_N_COLUMNS as u32,
            log_n_instances: LOG_N_ROWS,
        };
        let span = span!(Level::INFO, "Trace generation").entered();
        let trace = gen_trace(component.log_column_size());
        span.exit();
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let air = SimdWideFibAir { component };
        let proof = commit_and_prove(&air, channel, trace).unwrap();

        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        commit_and_verify(proof, &air, channel).unwrap();
    }
}
