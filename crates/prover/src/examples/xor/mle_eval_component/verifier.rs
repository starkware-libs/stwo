use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::Component;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CanonicCoset;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

#[derive(Debug, Clone)]
pub struct MleEvalVerifierComponent {
    eval_point: Vec<SecureField>,
    _mle_claim: SecureField,
}

impl MleEvalVerifierComponent {
    pub fn new(eval_point: &[SecureField], mle_claim: SecureField) -> Self {
        Self {
            eval_point: eval_point.to_vec(),
            _mle_claim: mle_claim,
        }
    }

    pub fn mle_n_variables(&self) -> usize {
        self.eval_point.len()
    }
}

impl Component for MleEvalVerifierComponent {
    // TODO: Bad this is a duplicate of MleEvalProverComponent component.
    fn n_constraints(&self) -> usize {
        // TODO(andrew): Use constraint counter.
        // TODO(andrew): Prevent code duplication in verifier.
        // 1. eq eval column initial value constraint.
        // 2. eq eval column periodic constraints (`n_variable` many)
        // TODO: let n_eq_eval_periodic_constraints = self.multilinear_n_variables;
        let n_eq_eval_periodic_constraints = 0;
        // 3. multilinear term column constant coefficient constraint.
        // 4. Inner product was computed correctly.
        // n_eq_eval_periodic_constraints + 3
        n_eq_eval_periodic_constraints + 1
    }

    // TODO: Bad this is a duplicate of MleEvalProverComponent component.
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.mle_n_variables() as u32 + 1
    }

    // TODO: Bad this is a duplicate of MleEvalProverComponent component.
    fn n_interaction_phases(&self) -> u32 {
        2
    }

    // TODO: Bad this is a duplicate of MleEvalProverComponent component.
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        let n_variables = self.mle_n_variables() as u32;

        let mut interaction_trace_degree_bounds = Vec::new();

        let eq_eval_col_log_degree_bounds = [n_variables; SECURE_EXTENSION_DEGREE];
        interaction_trace_degree_bounds.extend(eq_eval_col_log_degree_bounds);

        let mle_terms_prefix_sum_col_degree_bounds = [n_variables; SECURE_EXTENSION_DEGREE];
        interaction_trace_degree_bounds.extend(mle_terms_prefix_sum_col_degree_bounds);

        TreeVec::new(vec![vec![], interaction_trace_degree_bounds])
    }

    // TODO: Bad this is a duplicate of MleEvalProverComponent component.
    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mle_n_variables = self.mle_n_variables() as u32;
        let trace_domain_log_size = mle_n_variables;
        let _trace_step = CanonicCoset::new(trace_domain_log_size).step();

        // For checking eq evals
        let eq_evals_secure_col_mask_points = vec![point];

        // TODO: For preiodic constraints:
        // for log_step in 0..self.multilinear_n_variables {
        //     eq_evals_col_mask_points.push(point + trace_step.repeated_double(log_step))
        // }

        let mut interaction_trace_mask_points = Vec::new();
        // Copy `SECURE_EXTENSION_DEGREE` many times since columns are stored as base field columns.
        (0..SECURE_EXTENSION_DEGREE).for_each(|_| {
            interaction_trace_mask_points.push(eq_evals_secure_col_mask_points.clone())
        });
        // (0..SECURE_EXTENSION_DEGREE)
        //     .for_each(|_| interaction_mask_points.push(multilinear_term_mask_points.clone()));

        TreeVec::new(vec![vec![], interaction_trace_mask_points])
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        todo!()
    }
}
