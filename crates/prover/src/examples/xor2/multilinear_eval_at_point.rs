use std::collections::BTreeMap;

use crate::core::air::{Component, ComponentProver};
use crate::core::backend::CpuBackend;
use crate::core::channel::Channel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::horner_eval;

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
            .map(|(log_size, eval_claims)| {
                let n_claims = BaseField::from(eval_claims.len());
                let constant_coeff_claim =
                    horner_eval(eval_claims, self.aggregation_coeff) / n_claims;
                (log_size, constant_coeff_claim)
            })
            .collect()
    }
}

struct MultilinearEvalAtPoint;

impl Component for MultilinearEvalAtPoint {
    fn n_constraints(&self) -> usize {
        todo!()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        todo!()
    }

    fn n_interaction_phases(&self) -> u32 {
        todo!()
    }

    fn trace_log_degree_bounds(&self) -> crate::core::pcs::TreeVec<crate::core::ColumnVec<u32>> {
        todo!()
    }

    fn mask_points(
        &self,
        point: crate::core::circle::CirclePoint<SecureField>,
    ) -> crate::core::pcs::TreeVec<
        crate::core::ColumnVec<Vec<crate::core::circle::CirclePoint<SecureField>>>,
    > {
        todo!()
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        todo!()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: crate::core::circle::CirclePoint<SecureField>,
        mask: &crate::core::ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut crate::core::air::accumulation::PointEvaluationAccumulator,
        interaction_elements: &crate::core::InteractionElements,
        lookup_values: &crate::core::LookupValues,
    ) {
        todo!()
    }
}

impl ComponentProver<CpuBackend> for MultilinearEvalAtPoint {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &crate::core::air::ComponentTrace<'_, CpuBackend>,
        evaluation_accumulator: &mut crate::core::air::accumulation::DomainEvaluationAccumulator<
            CpuBackend,
        >,
        interaction_elements: &crate::core::InteractionElements,
        lookup_values: &crate::core::LookupValues,
    ) {
        todo!()
    }

    fn lookup_values(
        &self,
        _trace: &crate::core::air::ComponentTrace<'_, CpuBackend>,
    ) -> crate::core::LookupValues {
    }
}
