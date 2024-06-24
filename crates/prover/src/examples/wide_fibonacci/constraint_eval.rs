use std::collections::BTreeMap;

use itertools::{zip_eq, Itertools};
use num_traits::Zero;

use super::component::{
    Input, WideFibAir, WideFibComponent, ALPHA_ID, LOOKUP_VALUE_0_ID, LOOKUP_VALUE_1_ID,
    LOOKUP_VALUE_N_MINUS_1_ID, LOOKUP_VALUE_N_MINUS_2_ID, Z_ID,
};
use super::trace_gen::write_trace_row;
use crate::core::air::accumulation::{ColumnAccumulator, DomainEvaluationAccumulator};
use crate::core::air::{AirProver, Component, ComponentProver, ComponentTrace};
use crate::core::backend::CpuBackend;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::Coset;
use crate::core::constraints::{coset_vanishing, point_excluder, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, SecureCirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{BASE_TRACE, INTERACTION_TRACE};
use crate::core::utils::{
    bit_reverse, previous_bit_reversed_circle_domain_index, shifted_secure_combination,
};
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::wide_fibonacci::component::LOG_N_COLUMNS;
use crate::trace_generation::{AirTraceGenerator, AirTraceVerifier, ComponentTraceGenerator};

// TODO(AlonH): Rename file to `cpu.rs`.

impl AirTraceVerifier for WideFibAir {
    fn interaction_elements(&self, channel: &mut Blake2sChannel) -> InteractionElements {
        let ids = self.component.interaction_element_ids();
        let elements = channel.draw_felts(ids.len());
        InteractionElements::new(BTreeMap::from_iter(zip_eq(ids, elements)))
    }
}

impl AirTraceGenerator<CpuBackend> for WideFibAir {
    fn interact(
        &self,
        trace: &ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        self.component
            .write_interaction_trace(&trace.iter().collect(), elements)
    }

    fn to_air_prover(&self) -> impl AirProver<CpuBackend> {
        self.clone()
    }

    fn composition_log_degree_bound(&self) -> u32 {
        self.component.max_constraint_log_degree_bound()
    }
}

impl AirProver<CpuBackend> for WideFibAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<CpuBackend>> {
        vec![&self.component]
    }
}

impl WideFibComponent {
    fn evaluate_trace_boundary_constraints(
        &self,
        trace_evals: &TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
        trace_eval_domain: CircleDomain,
        zero_domain: Coset,
        accum: &mut ColumnAccumulator<'_, CpuBackend>,
        lookup_values: &LookupValues,
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let mut first_point_denoms = vec![];
        let mut last_point_denoms = vec![];
        for point in trace_eval_domain.iter() {
            // TODO(AlonH): Use `point_vanishing_fraction` instead of `point_vanishing` everywhere.
            first_point_denoms.push(point_vanishing(zero_domain.at(0), point));
            last_point_denoms.push(point_vanishing(zero_domain.at(zero_domain.size()), point));
        }
        bit_reverse(&mut first_point_denoms);
        bit_reverse(&mut last_point_denoms);
        let mut first_point_denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        let mut last_point_denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&first_point_denoms, &mut first_point_denom_inverses);
        BaseField::batch_inverse(&last_point_denoms, &mut last_point_denom_inverses);
        let mut first_point_numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let mut last_point_numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let (lookup_value_0, lookup_value_1, lookup_value_n_minus_2, lookup_value_n_minus_1) = (
            lookup_values[LOOKUP_VALUE_0_ID],
            lookup_values[LOOKUP_VALUE_1_ID],
            lookup_values[LOOKUP_VALUE_N_MINUS_2_ID],
            lookup_values[LOOKUP_VALUE_N_MINUS_1_ID],
        );

        #[allow(clippy::needless_range_loop)]
        for i in 0..trace_eval_domain.size() {
            first_point_numerators[i] = accum.random_coeff_powers[self.n_columns() + 4]
                * (trace_evals[BASE_TRACE][0][i] - lookup_value_0)
                + accum.random_coeff_powers[self.n_columns() + 3]
                    * (trace_evals[BASE_TRACE][1][i] - lookup_value_1);
            last_point_numerators[i] = accum.random_coeff_powers[self.n_columns() + 2]
                * (trace_evals[BASE_TRACE][self.n_columns() - 2][i] - lookup_value_n_minus_2)
                + accum.random_coeff_powers[self.n_columns() + 1]
                    * (trace_evals[BASE_TRACE][self.n_columns() - 1][i] - lookup_value_n_minus_1);
        }
        for (i, (num, denom_inverse)) in first_point_numerators
            .iter()
            .zip(first_point_denom_inverses.iter())
            .enumerate()
        {
            accum.accumulate(i, *num * *denom_inverse);
        }
        for (i, (num, denom_inverse)) in last_point_numerators
            .iter()
            .zip(last_point_denom_inverses.iter())
            .enumerate()
        {
            accum.accumulate(i, *num * *denom_inverse);
        }
    }

    fn evaluate_trace_step_constraints(
        &self,
        trace_evals: &TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
        trace_eval_domain: CircleDomain,
        zero_domain: Coset,
        accum: &mut ColumnAccumulator<'_, CpuBackend>,
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let mut denoms = vec![];
        for point in trace_eval_domain.iter() {
            denoms.push(coset_vanishing(zero_domain, point));
        }
        bit_reverse(&mut denoms);
        let mut denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&denoms, &mut denom_inverses);
        let mut numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];

        #[allow(clippy::needless_range_loop)]
        for i in 0..trace_eval_domain.size() {
            for j in 0..self.n_columns() - 2 {
                numerators[i] += accum.random_coeff_powers[self.n_columns() - 3 - j]
                    * (trace_evals[BASE_TRACE][j][i].square()
                        + trace_evals[BASE_TRACE][j + 1][i].square()
                        - trace_evals[BASE_TRACE][j + 2][i]);
            }
        }
        for (i, (num, denom_inverse)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
            accum.accumulate(i, *num * *denom_inverse);
        }
    }

    fn evaluate_lookup_boundary_constraints(
        &self,
        trace_evals: &TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
        trace_eval_domain: CircleDomain,
        zero_domain: Coset,
        accum: &mut ColumnAccumulator<'_, CpuBackend>,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let mut first_point_denoms = vec![];
        let mut last_point_denoms = vec![];
        for point in trace_eval_domain.iter() {
            first_point_denoms.push(point_vanishing(zero_domain.at(0), point));
            last_point_denoms.push(point_vanishing(zero_domain.at(zero_domain.size()), point));
        }
        bit_reverse(&mut first_point_denoms);
        bit_reverse(&mut last_point_denoms);
        let mut first_point_denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        let mut last_point_denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&first_point_denoms, &mut first_point_denom_inverses);
        BaseField::batch_inverse(&last_point_denoms, &mut last_point_denom_inverses);
        let mut first_point_numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let mut last_point_numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let (alpha, z) = (interaction_elements[ALPHA_ID], interaction_elements[Z_ID]);
        let (lookup_value_0, lookup_value_1, lookup_value_n_minus_2, lookup_value_n_minus_1) = (
            lookup_values[LOOKUP_VALUE_0_ID],
            lookup_values[LOOKUP_VALUE_1_ID],
            lookup_values[LOOKUP_VALUE_N_MINUS_2_ID],
            lookup_values[LOOKUP_VALUE_N_MINUS_1_ID],
        );

        #[allow(clippy::needless_range_loop)]
        for i in 0..trace_eval_domain.size() {
            let value =
                SecureCirclePoly::<CpuBackend>::eval_from_partial_evals(std::array::from_fn(|j| {
                    trace_evals[INTERACTION_TRACE][j][i].into()
                }));
            first_point_numerators[i] = accum.random_coeff_powers[self.n_columns() - 1]
                * ((value
                    * shifted_secure_combination(
                        &[
                            trace_evals[BASE_TRACE][self.n_columns() - 2][i],
                            trace_evals[BASE_TRACE][self.n_columns() - 1][i],
                        ],
                        alpha,
                        z,
                    ))
                    - shifted_secure_combination(
                        &[trace_evals[BASE_TRACE][0][i], trace_evals[BASE_TRACE][1][i]],
                        alpha,
                        z,
                    ));
            last_point_numerators[i] = accum.random_coeff_powers[self.n_columns() - 2]
                * ((value
                    * shifted_secure_combination(
                        &[lookup_value_n_minus_2, lookup_value_n_minus_1],
                        alpha,
                        z,
                    ))
                    - shifted_secure_combination(&[lookup_value_0, lookup_value_1], alpha, z));
        }
        for (i, (num, denom_inverse)) in first_point_numerators
            .iter()
            .zip(first_point_denom_inverses.iter())
            .enumerate()
        {
            accum.accumulate(i, *num * *denom_inverse);
        }
        for (i, (num, denom_inverse)) in last_point_numerators
            .iter()
            .zip(last_point_denom_inverses.iter())
            .enumerate()
        {
            accum.accumulate(i, *num * *denom_inverse);
        }
    }

    // TODO(AlonH): Simplify this function by using utility functions.
    fn evaluate_lookup_step_constraints(
        &self,
        trace_evals: &TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>>,
        trace_eval_domain: CircleDomain,
        zero_domain: Coset,
        accum: &mut ColumnAccumulator<'_, CpuBackend>,
        interaction_elements: &InteractionElements,
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let mut denoms = vec![];
        for point in trace_eval_domain.iter() {
            denoms.push(
                coset_vanishing(zero_domain, point) / point_excluder(zero_domain.at(0), point),
            );
        }
        bit_reverse(&mut denoms);
        let mut denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&denoms, &mut denom_inverses);
        let mut numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let (alpha, z) = (interaction_elements[ALPHA_ID], interaction_elements[Z_ID]);

        #[allow(clippy::needless_range_loop)]
        for i in 0..trace_eval_domain.size() {
            let value =
                SecureCirclePoly::<CpuBackend>::eval_from_partial_evals(std::array::from_fn(|j| {
                    trace_evals[INTERACTION_TRACE][j][i].into()
                }));
            let prev_index = previous_bit_reversed_circle_domain_index(
                i,
                zero_domain.log_size,
                trace_eval_domain.log_size(),
            );
            let prev_value =
                SecureCirclePoly::<CpuBackend>::eval_from_partial_evals(std::array::from_fn(|j| {
                    trace_evals[INTERACTION_TRACE][j][prev_index].into()
                }));
            numerators[i] = accum.random_coeff_powers[self.n_columns()]
                * ((value
                    * shifted_secure_combination(
                        &[
                            trace_evals[BASE_TRACE][self.n_columns() - 2][i],
                            trace_evals[BASE_TRACE][self.n_columns() - 1][i],
                        ],
                        alpha,
                        z,
                    ))
                    - (prev_value
                        * shifted_secure_combination(
                            &[trace_evals[BASE_TRACE][0][i], trace_evals[BASE_TRACE][1][i]],
                            alpha,
                            z,
                        )));
        }
        for (i, (num, denom_inverse)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
            accum.accumulate(i, *num * *denom_inverse);
        }
    }
}

impl ComponentProver<CpuBackend> for WideFibComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let trace_eval_domain = CanonicCoset::new(max_constraint_degree).circle_domain();
        let trace_evals = &trace.evals;
        let zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let [mut accum] =
            evaluation_accumulator.columns([(max_constraint_degree, self.n_constraints())]);

        // TODO(AlonH): Evaluate the numerators together and the denominators together (i.e. in the
        // same for loop)
        self.evaluate_trace_boundary_constraints(
            trace_evals,
            trace_eval_domain,
            zero_domain,
            &mut accum,
            lookup_values,
        );
        self.evaluate_lookup_step_constraints(
            trace_evals,
            trace_eval_domain,
            zero_domain,
            &mut accum,
            interaction_elements,
        );
        self.evaluate_lookup_boundary_constraints(
            trace_evals,
            trace_eval_domain,
            zero_domain,
            &mut accum,
            interaction_elements,
            lookup_values,
        );
        self.evaluate_trace_step_constraints(
            trace_evals,
            trace_eval_domain,
            zero_domain,
            &mut accum,
        );
    }

    fn lookup_values(&self, trace: &ComponentTrace<'_, CpuBackend>) -> LookupValues {
        let domain = CanonicCoset::new(self.log_column_size());
        let trace_poly = &trace.polys[BASE_TRACE];
        let values = BTreeMap::from_iter([
            (
                LOOKUP_VALUE_0_ID.to_string(),
                trace_poly[0]
                    .eval_at_point(domain.at(0).into_ef())
                    .try_into()
                    .unwrap(),
            ),
            (
                LOOKUP_VALUE_1_ID.to_string(),
                trace_poly[1]
                    .eval_at_point(domain.at(0).into_ef())
                    .try_into()
                    .unwrap(),
            ),
            (
                LOOKUP_VALUE_N_MINUS_2_ID.to_string(),
                trace_poly[self.n_columns() - 2]
                    .eval_at_point(domain.at(domain.size()).into_ef())
                    .try_into()
                    .unwrap(),
            ),
            (
                LOOKUP_VALUE_N_MINUS_1_ID.to_string(),
                trace_poly[self.n_columns() - 1]
                    .eval_at_point(domain.at(domain.size()).into_ef())
                    .try_into()
                    .unwrap(),
            ),
        ]);
        LookupValues::new(values)
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
