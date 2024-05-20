use std::collections::BTreeMap;

use itertools::{zip_eq, Itertools};
use num_traits::Zero;

use super::component::{Input, WideFibAir, WideFibComponent, ALPHA_ID, Z_ID};
use super::trace_gen::write_trace_row;
use crate::core::air::accumulation::DomainEvaluationAccumulator;
use crate::core::air::{
    AirProver, AirTraceVerifier, AirTraceWriter, Component, ComponentProver, ComponentTrace,
    ComponentTraceWriter,
};
use crate::core::backend::CpuBackend;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::constraints::{coset_vanishing, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse, shifted_secure_combination};
use crate::core::{ColumnVec, ComponentVec, InteractionElements};
use crate::examples::wide_fibonacci::component::LOG_N_COLUMNS;

// TODO(AlonH): Rename file to `cpu.rs`.

impl AirTraceVerifier for WideFibAir {
    fn interaction_elements(&self, channel: &mut Blake2sChannel) -> InteractionElements {
        let ids = self.component.interaction_element_ids();
        let elements = channel.draw_felts(ids.len()).into_iter().map(|e| e.0 .0);
        InteractionElements::new(BTreeMap::from_iter(zip_eq(ids, elements)))
    }
}

impl AirTraceWriter<CpuBackend> for WideFibAir {
    fn interact(
        &self,
        trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> ComponentVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        ComponentVec(vec![self
            .component
            .write_interaction_trace(&trace.iter().collect(), elements)])
    }

    fn to_air_prover(&self) -> &impl AirProver<CpuBackend> {
        self
    }
}

impl AirProver<CpuBackend> for WideFibAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![&self.component]
    }
}

impl ComponentProver<CpuBackend> for WideFibComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        interaction_elements: &InteractionElements,
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let trace_eval_domain = CanonicCoset::new(max_constraint_degree).circle_domain();
        let trace_evals = &trace.evals;
        let zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let mut denoms = vec![];
        let mut lookup_denoms = vec![];
        for point in trace_eval_domain.iter() {
            denoms.push(coset_vanishing(zero_domain, point));
            lookup_denoms.push(point_vanishing(zero_domain.at(0), point));
        }
        bit_reverse(&mut denoms);
        let mut denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&denoms, &mut denom_inverses);
        bit_reverse(&mut lookup_denoms);
        let mut lookup_denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&lookup_denoms, &mut lookup_denom_inverses);
        let mut numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let mut lookup_numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let [mut accum] =
            evaluation_accumulator.columns([(max_constraint_degree, self.n_constraints())]);
        let (alpha, z) = (interaction_elements[ALPHA_ID], interaction_elements[Z_ID]);

        for i in 0..trace_eval_domain.size() {
            // Step constraints.
            for j in 0..self.n_columns() - 2 {
                numerators[i] += accum.random_coeff_powers[self.n_columns() - 3 - j]
                    * (trace_evals[0][j][i].square() + trace_evals[0][j + 1][i].square()
                        - trace_evals[0][j + 2][i]);
            }

            // Lookup constraints.
            lookup_numerators[i] = accum.random_coeff_powers[self.n_columns() - 2]
                * ((trace_evals[1][0][i]
                    * shifted_secure_combination(
                        &[
                            trace_evals[0][self.n_columns() - 2][i],
                            trace_evals[0][self.n_columns() - 1][i],
                        ],
                        alpha,
                        z,
                    ))
                    - shifted_secure_combination(
                        &[trace_evals[0][0][i], trace_evals[0][1][i]],
                        alpha,
                        z,
                    ));
        }
        for (i, (num, denom_inverse)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
            accum.accumulate(i, *num * *denom_inverse);
        }
        for (i, (num, denom_inverse)) in lookup_numerators
            .iter()
            .zip(lookup_denom_inverses.iter())
            .enumerate()
        {
            accum.accumulate(i, *num * *denom_inverse);
        }
    }
}

/// Generates the trace for the wide Fibonacci example.
pub fn gen_trace(
    wide_fib: &WideFibComponent,
    private_input: Vec<Input>,
) -> ColumnVec<Vec<BaseField>> {
    let n_instances = 1 << wide_fib.log_n_instances;
    assert_eq!(
        private_input.len(),
        n_instances,
        "The number of inputs must match the number of instances"
    );
    assert!(
        wide_fib.log_fibonacci_size >= LOG_N_COLUMNS as u32,
        "The fibonacci size must be at least equal to the length of a row"
    );
    let n_rows_per_instance = 1 << (wide_fib.log_fibonacci_size - wide_fib.log_n_columns() as u32);
    let n_rows = n_instances * n_rows_per_instance;
    let zero_vec = vec![BaseField::zero(); n_rows];
    let mut dst = vec![zero_vec; wide_fib.n_columns()];
    (0..n_rows_per_instance).fold(private_input, |input, row| {
        (0..n_instances)
            .map(|instance| {
                let (a, b) =
                    write_trace_row(&mut dst, &input[instance], row * n_instances + instance);
                Input { a, b }
            })
            .collect_vec()
    });
    dst
}
