use itertools::Itertools;

use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::mask::fixed_mask_points;
use crate::core::air::{Air, Component};
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::backend::CpuBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::{coset_vanishing, point_excluder, point_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::shifted_secure_combination;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::wide_fibonacci::trace_gen::write_lookup_column;
use crate::trace_generation::registry::ComponentGenerationRegistry;
use crate::trace_generation::ComponentTraceGenerator;

pub const LOG_N_COLUMNS: usize = 8;
pub const N_COLUMNS: usize = 1 << LOG_N_COLUMNS;

pub const ALPHA_ID: &str = "wide_fibonacci_alpha";
pub const Z_ID: &str = "wide_fibonacci_z";
pub const LOOKUP_VALUE_0_ID: &str = "wide_fibonacci_0";
pub const LOOKUP_VALUE_1_ID: &str = "wide_fibonacci_1";
pub const LOOKUP_VALUE_N_MINUS_2_ID: &str = "wide_fibonacci_n-2";
pub const LOOKUP_VALUE_N_MINUS_1_ID: &str = "wide_fibonacci_n-1";

/// Component that computes 2^`self.log_n_instances` instances of fibonacci sequences of size
/// 2^`self.log_fibonacci_size`. The numbers are computes over [N_COLUMNS] trace columns. The
/// number of rows (i.e the size of the columns) is determined by the parameters above (see
/// [WideFibComponent::log_column_size()]).
#[derive(Clone)]
pub struct WideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl WideFibComponent {
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

    pub fn interaction_element_ids(&self) -> Vec<String> {
        vec![ALPHA_ID.to_string(), Z_ID.to_string()]
    }

    fn evaluate_trace_boundary_constraints_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        constraint_zero_domain: Coset,
        lookup_values: &LookupValues,
    ) {
        let numerator = mask[0][0][0] - lookup_values[LOOKUP_VALUE_0_ID];
        let denom = point_vanishing(constraint_zero_domain.at(0), point);
        evaluation_accumulator.accumulate(numerator / denom);
        let numerator = mask[0][1][0] - lookup_values[LOOKUP_VALUE_1_ID];
        evaluation_accumulator.accumulate(numerator / denom);

        let numerator = mask[0][self.n_columns() - 2][0] - lookup_values[LOOKUP_VALUE_N_MINUS_2_ID];
        let denom = point_vanishing(
            constraint_zero_domain.at(constraint_zero_domain.size() - 1),
            point,
        );
        evaluation_accumulator.accumulate(numerator / denom);
        let numerator = mask[0][self.n_columns() - 1][0] - lookup_values[LOOKUP_VALUE_N_MINUS_1_ID];
        evaluation_accumulator.accumulate(numerator / denom);
    }

    fn evaluate_trace_step_constraints_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        constraint_zero_domain: Coset,
    ) {
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        for i in 0..self.n_columns() - 2 {
            let numerator = mask[i][0].square() + mask[i + 1][0].square() - mask[i + 2][0];
            evaluation_accumulator.accumulate(numerator * denom_inverse);
        }
    }

    fn evaluate_lookup_boundary_constraints_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        constraint_zero_domain: Coset,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        let (alpha, z) = (interaction_elements[ALPHA_ID], interaction_elements[Z_ID]);
        let value = SecureField::from_partial_evals(std::array::from_fn(|i| mask[1][i][0]));
        let numerator = (value
            * shifted_secure_combination(
                &[
                    mask[0][self.n_columns() - 2][0],
                    mask[0][self.n_columns() - 1][0],
                ],
                alpha,
                z,
            ))
            - shifted_secure_combination(&[mask[0][0][0], mask[0][1][0]], alpha, z);
        let denom = point_vanishing(constraint_zero_domain.at(0), point);
        evaluation_accumulator.accumulate(numerator / denom);

        let numerator = (value
            * shifted_secure_combination(
                &[
                    lookup_values[LOOKUP_VALUE_N_MINUS_2_ID],
                    lookup_values[LOOKUP_VALUE_N_MINUS_1_ID],
                ],
                alpha,
                z,
            ))
            - shifted_secure_combination(
                &[
                    lookup_values[LOOKUP_VALUE_0_ID],
                    lookup_values[LOOKUP_VALUE_1_ID],
                ],
                alpha,
                z,
            );
        let denom = point_vanishing(
            constraint_zero_domain.at(constraint_zero_domain.size() - 1),
            point,
        );
        evaluation_accumulator.accumulate(numerator / denom);
    }

    fn evaluate_lookup_step_constraints_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        constraint_zero_domain: Coset,
        interaction_elements: &InteractionElements,
    ) {
        let (alpha, z) = (interaction_elements[ALPHA_ID], interaction_elements[Z_ID]);
        let value = SecureField::from_partial_evals(std::array::from_fn(|i| mask[1][i][0]));
        let prev_value = SecureField::from_partial_evals(std::array::from_fn(|i| mask[1][i][1]));
        let numerator = (value
            * shifted_secure_combination(
                &[
                    mask[0][self.n_columns() - 2][0],
                    mask[0][self.n_columns() - 1][0],
                ],
                alpha,
                z,
            ))
            - (prev_value * shifted_secure_combination(&[mask[0][0][0], mask[0][1][0]], alpha, z));
        let denom = coset_vanishing(constraint_zero_domain, point)
            / point_excluder(constraint_zero_domain.at(0), point);
        evaluation_accumulator.accumulate(numerator / denom);
    }
}

#[derive(Clone)]
pub struct WideFibAir {
    pub component: WideFibComponent,
}

impl Air for WideFibAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }
}

impl Component for WideFibComponent {
    fn n_constraints(&self) -> usize {
        self.n_columns() + 5
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![
            vec![self.log_column_size(); self.n_columns()],
            vec![self.log_column_size(); SECURE_EXTENSION_DEGREE],
        ])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let domain = CanonicCoset::new(self.log_column_size());
        TreeVec::new(vec![
            fixed_mask_points(&vec![vec![0_usize]; self.n_columns()], point),
            vec![vec![point, point - domain.step().into_ef()]; SECURE_EXTENSION_DEGREE],
        ])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<Vec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        self.evaluate_trace_boundary_constraints_at_point(
            point,
            mask,
            evaluation_accumulator,
            constraint_zero_domain,
            lookup_values,
        );
        self.evaluate_lookup_step_constraints_at_point(
            point,
            mask,
            evaluation_accumulator,
            constraint_zero_domain,
            interaction_elements,
        );
        self.evaluate_lookup_boundary_constraints_at_point(
            point,
            mask,
            evaluation_accumulator,
            constraint_zero_domain,
            interaction_elements,
            lookup_values,
        );
        self.evaluate_trace_step_constraints_at_point(
            point,
            &mask[0],
            evaluation_accumulator,
            constraint_zero_domain,
        );
    }
}

impl ComponentTraceGenerator<CpuBackend> for WideFibComponent {
    type Component = Self;
    type Inputs = ();

    fn add_inputs(&mut self, _inputs: &Self::Inputs) {}

    fn write_trace(
        _component_id: &str,
        _registry: &mut ComponentGenerationRegistry,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn write_interaction_trace(
        &self,
        trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let trace_values = trace.iter().map(|eval| &eval.values[..]).collect_vec();
        let (alpha, z) = (elements[ALPHA_ID], elements[Z_ID]);
        // TODO(AlonH): Return a secure column directly.
        let values = write_lookup_column(&trace_values, alpha, z);
        let secure_column: SecureColumnByCoords<CpuBackend> = values.into_iter().collect();
        secure_column
            .columns
            .into_iter()
            .map(|eval| {
                let coset = CanonicCoset::new(trace[0].domain.log_size());
                CpuCircleEvaluation::new_canonical_ordered(coset, eval)
            })
            .collect_vec()
    }

    fn component(&self) -> Self::Component {
        self.clone()
    }
}

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}
