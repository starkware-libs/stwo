use std::collections::BTreeMap;
use std::iter::zip;

use itertools::Itertools;

use crate::core::air::Component;
use crate::core::channel::Channel;
use crate::core::circle::CirclePoint;
use crate::core::constraints::point_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::utils::horner_eval;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{eval_from_partial_evals, CanonicCoset};

pub struct BatchMultilinearEvalIopVerfier {
    eval_claims_by_n_variables: BTreeMap<u32, Vec<SecureField>>,
    aggregation_coeff: SecureField,
}

impl BatchMultilinearEvalIopVerfier {
    pub fn new(
        channel: &mut impl Channel,
        eval_claims_by_n_variables: BTreeMap<u32, Vec<SecureField>>,
    ) -> Self {
        Self {
            eval_claims_by_n_variables,
            aggregation_coeff: channel.draw_felt(),
        }
    }

    fn univariate_sumcheck_constant_coeff_claim_by_log_size(&self) -> BTreeMap<u32, SecureField> {
        self.eval_claims_by_n_variables
            .iter()
            .map(|(&log_size, eval_claims)| {
                let n_claims = BaseField::from(eval_claims.len());
                let constant_coeff_claim =
                    horner_eval(eval_claims, self.aggregation_coeff) / n_claims;
                (log_size, constant_coeff_claim)
            })
            .collect()
    }
}

impl Component for BatchMultilinearEvalIopVerfier {
    fn n_constraints(&self) -> usize {
        let mut n_constraints = 0;

        for (&n_variables, _) in &self.eval_claims_by_n_variables {
            // Column for eq evals has a constraint per variable
            // TODO: n_constraints += n_variables as usize;
            // Constraint to check constant coefficient on sumcheck g poly.
            n_constraints += 1;
        }

        n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.eval_claims_by_n_variables
            .iter()
            .map(|(n_vars, _)| n_vars + 1)
            .max()
            .unwrap()
    }

    fn n_interaction_phases(&self) -> u32 {
        0
    }

    fn trace_log_degree_bounds(&self) -> crate::core::pcs::TreeVec<crate::core::ColumnVec<u32>> {
        let mut interaction_trace_log_degree_bounds = Vec::new();

        for (&n_variables, _) in &self.eval_claims_by_n_variables {
            // Three trace columns per multilinear n variables:
            // 1. eq evals (eq_evals) (secure column)
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            // 2. sumcheck g column (secure column)
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            // 3. sumcheck h column (secure column)
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
            interaction_trace_log_degree_bounds.push(n_variables);
        }

        TreeVec::new(vec![vec![], interaction_trace_log_degree_bounds])
    }

    fn mask_points(
        &self,
        point: crate::core::circle::CirclePoint<SecureField>,
    ) -> crate::core::pcs::TreeVec<
        crate::core::ColumnVec<Vec<crate::core::circle::CirclePoint<SecureField>>>,
    > {
        let mut interaction_trace_mask_points = Vec::new();

        for (&n_variables, _) in &self.eval_claims_by_n_variables {
            let trace_step = CanonicCoset::new(n_variables).step().into_ef();

            let eq_evals_col_mask_points = (0..n_variables)
                .map(|i| point + trace_step.repeated_double(i))
                .collect_vec();
            let g_mask_points = [point, -point];
            let h_mask_points = vec![];

            interaction_trace_mask_points
                .extend([&eq_evals_col_mask_points; SECURE_EXTENSION_DEGREE].map(|v| v.to_vec()));
            interaction_trace_mask_points
                .extend([&g_mask_points; SECURE_EXTENSION_DEGREE].map(|v| v.to_vec()));
            interaction_trace_mask_points
                .extend([&h_mask_points; SECURE_EXTENSION_DEGREE].map(|v| v.to_vec()));
        }

        TreeVec::new(vec![vec![], interaction_trace_mask_points])
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: crate::core::circle::CirclePoint<SecureField>,
        mask: &crate::core::ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut crate::core::air::accumulation::PointEvaluationAccumulator,
        _interaction_elements: &crate::core::InteractionElements,
        _lookup_values: &crate::core::LookupValues,
    ) {
        let interaction_mask_elements = mask;
        let multilinear_col_groups =
            interaction_mask_elements.array_chunks::<{ 3 * SECURE_EXTENSION_DEGREE }>();
        let univariate_sumcheck_constant_coeff_claim_by_log_size =
            self.univariate_sumcheck_constant_coeff_claim_by_log_size();

        for (
            (_log_size, claim),
            [
                // eq evals secure col
                _eq_col0_mask, _eq_col1_mask, _eq_col2_mask, _eq_col3_mask, 
                // g secure col
                g_col0_mask, g_col1_mask, g_col2_mask, g_col3_mask, 
                // h secure col
                _h_col0_mask, _h_col1_mask, _h_col2_mask, _h_col3_mask
            ],
        ) in zip(
            univariate_sumcheck_constant_coeff_claim_by_log_size,
            multilinear_col_groups,
        ) {
            // TODO: Evaluate eq eval constraints.

            // Let `g(x, y) = g0(x) + y * g1(x)`
            let g_at_p = eval_from_partial_evals([
                g_col0_mask[0],
                g_col1_mask[0],
                g_col2_mask[0],
                g_col3_mask[0],
            ]);
            let g_at_neg_p = eval_from_partial_evals([
                g_col0_mask[1],
                g_col1_mask[1],
                g_col2_mask[1],
                g_col3_mask[1],
            ]);
            let g0_at_p_x = (g_at_p + g_at_neg_p) / BaseField::from(2);

            // Since we constrain at `(0, 1)` we are essentially checking `g(0, 0) = claim`.
            let g_constant_coeff_numerator = g0_at_p_x - claim;
            let g_constant_coeff_denominator =
                point_vanishing(CirclePoint::<BaseField>::zero(), point);
            evaluation_accumulator
                .accumulate(g_constant_coeff_numerator / g_constant_coeff_denominator);
        }
    }

    fn gkr_lookup_instance_configs(&self) -> Vec<crate::core::air::LookupInstanceConfig> {
        vec![]
    }

    fn eval_at_point_iop_claims_by_n_variables(
        &self,
        _multilinear_eval_claims_by_instance: &[Vec<SecureField>],
    ) -> BTreeMap<u32, Vec<SecureField>> {
        BTreeMap::new()
    }
}
