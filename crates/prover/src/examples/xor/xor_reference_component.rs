use std::collections::BTreeMap;

use itertools::izip;

use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace, LookupInstanceConfig};
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

    fn gkr_lookup_instance_configs(&self) -> Vec<LookupInstanceConfig> {
        vec![LookupInstanceConfig {
            variant: Gate::LogUp,
            is_table: false,
            table_id: XOR_LOOKUP_TABLE_ID.to_string(),
        }]
    }

    fn eval_at_point_iop_claims_by_n_variables(
        &self,
        multilinear_eval_claims_by_instance: &[Vec<SecureField>],
    ) -> BTreeMap<u32, Vec<SecureField>> {
        let mut claims_by_n_variables = BTreeMap::new();
        let n_variables = u8::BITS + u8::BITS;
        let denominator_claim = multilinear_eval_claims_by_instance[0][1];
        claims_by_n_variables.insert(n_variables, vec![denominator_claim]);
        claims_by_n_variables
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

    fn build_lookup_instances(
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

    fn lookup_multilinears_for_eval_at_point_iop(
        &self,
        lookup_layers: Vec<Layer<CpuBackend>>,
    ) -> Vec<Mle<CpuBackend, SecureField>> {
        match lookup_layers.into_iter().next().unwrap() {
            Layer::LogUpSingles { denominators } => {
                vec![denominators]
            }
            _ => panic!(),
        }
    }
}
