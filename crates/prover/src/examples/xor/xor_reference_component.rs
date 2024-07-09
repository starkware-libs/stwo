use std::collections::BTreeSet;

use itertools::izip;
use num_traits::One;

use super::gkr_lookup_component::prover::{GkrLookupComponentProver, MleAccumulator};
use super::gkr_lookup_component::verifier::{
    GkrLookupComponent, LookupInstanceConfig, MleClaimAccumulator, UnivariateClaimAccumulator,
};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace, ComponentTraceWriter};
use crate::core::backend::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::lookups::gkr_prover::Layer;
use crate::core::lookups::gkr_verifier::Gate;
use crate::core::lookups::mle::Mle;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements, LookupValues};
use crate::examples::xor::xor_table_component::{XOR_ALPHA_ID, XOR_LOOKUP_TABLE_ID, XOR_Z_ID};

/// Component full of random 8-bit XOR operations.
pub struct XorReferenceComponent;

impl Component for XorReferenceComponent {
    fn n_constraints(&self) -> usize {
        0
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        u8::BITS + u8::BITS
    }

    /// Returns the degree bounds of each trace column.
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![u8::BITS + u8::BITS; 3], vec![]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        // TODO: Forcing base and interaction mask points is annoying.
        TreeVec::new(vec![vec![vec![point], vec![point], vec![point]], vec![]])
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![XOR_ALPHA_ID.to_string(), XOR_Z_ID.to_string()]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
    }

    fn n_interaction_phases(&self) -> u32 {
        0
    }

    // fn gkr_lookup_instance_configs(&self) -> Vec<LookupInstanceConfig> {}

    // fn eval_at_point_iop_claims_by_n_variables(
    //     &self,
    //
}

impl GkrLookupComponent for XorReferenceComponent {
    fn lookup_config(&self) -> Vec<LookupInstanceConfig> {
        vec![LookupInstanceConfig {
            variant: Gate::LogUp,
            is_table: false,
            table_id: XOR_LOOKUP_TABLE_ID.to_string(),
        }]
    }

    fn mle_n_variables_for_univariate_iop(&self) -> BTreeSet<u32> {
        BTreeSet::from([u8::BITS + u8::BITS])
    }

    fn validate_succinct_mle_claims(
        &self,
        _ood_point: &[SecureField],
        multilinear_claims_by_instance: &[Vec<SecureField>],
        _interaction_elements: &InteractionElements,
    ) -> bool {
        let numerator_claim = multilinear_claims_by_instance[0][0];
        numerator_claim.is_one()
    }

    fn accumulate_mle_claims_for_univariate_iop(
        &self,
        mle_claims_by_instance: &[Vec<SecureField>],
        accumulator: &mut MleClaimAccumulator,
    ) {
        let denominators_n_variables = 16;
        let denominators_claim = mle_claims_by_instance[0][1];
        accumulator.accumulate(denominators_n_variables, denominators_claim);
    }

    fn evaluate_lookup_columns_for_univariate_iop_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        accumulator: &mut UnivariateClaimAccumulator,
        interaction_elements: &InteractionElements,
    ) {
        let z = interaction_elements[XOR_Z_ID];
        let alpha = interaction_elements[XOR_ALPHA_ID];

        let lhs = mask[0][0];
        let rhs = mask[1][0];
        let res = mask[2][0];

        let denominators_log_size = 16;
        let denominators_eval = z - lhs - rhs * alpha - res * alpha.square();

        accumulator.accumulate(denominators_log_size, denominators_eval)
    }
}

impl ComponentTraceWriter<CpuBackend> for XorReferenceComponent {
    fn write_interaction_trace(
        &self,
        _trace: &ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        vec![]
    }
}

impl ComponentProver<CpuBackend> for XorReferenceComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        _trace: &ComponentTrace<'_, CpuBackend>,
        _evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
    }
}

impl GkrLookupComponentProver<CpuBackend> for XorReferenceComponent {
    fn accumulate_mle_for_univariate_iop(
        &self,
        lookup_instances: Vec<Layer<CpuBackend>>,
        accumulator: &mut MleAccumulator<CpuBackend>,
    ) {
        let max_column_log_degree = 16;
        let acc_coeff = accumulator.accumulation_coeff();
        let acc_mle = accumulator.column(max_column_log_degree);

        let denominators = match &lookup_instances[0] {
            Layer::LogUpSingles { denominators, .. } => denominators,
            _ => panic!(),
        };

        for (i, &eval) in denominators.iter().enumerate() {
            acc_mle[i] = acc_mle[i] * acc_coeff + eval;
        }
    }

    fn write_lookup_instances(
        &self,
        trace: ColumnVec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>,
        interaction_elements: &InteractionElements,
    ) -> Vec<Layer<CpuBackend>> {
        let z = interaction_elements[XOR_Z_ID];
        let alpha = interaction_elements[XOR_ALPHA_ID];

        let xor_lhs_col = &**trace[0];
        let xor_rhs_col = &**trace[1];
        let xor_res_col = &**trace[2];

        let denominators = izip!(xor_lhs_col, xor_rhs_col, xor_res_col)
            .map(|(&lhs, &rhs, &xor)| z - lhs - alpha * rhs - alpha.square() * xor)
            .collect();

        vec![Layer::LogUpSingles {
            denominators: Mle::new(denominators),
        }]
    }
}
