use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace, LookupInstanceConfig};
use crate::core::backend::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::gkr_verifier::Gate;
use crate::core::pcs::TreeVec;
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
        TreeVec::new(vec![vec![u8::BITS + u8::BITS; 3]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![vec![vec![point], vec![point], vec![point]]])
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

    // fn evaluate_lookup_instances_at_point

    fn n_interaction_phases(&self) -> u32 {
        1
    }

    fn gkr_lookup_instance_configs(&self) -> Vec<crate::core::air::LookupInstanceConfig> {
        vec![LookupInstanceConfig {
            variant: Gate::_LogUp,
            is_table: false,
            table_id: XOR_LOOKUP_TABLE_ID.to_string(),
        }]
    }

    fn eval_at_point_iop_claims_by_n_variables(
        &self,
        _multilinear_eval_claims_by_instance: &[Vec<crate::core::fields::qm31::SecureField>],
    ) -> std::collections::BTreeMap<u32, Vec<crate::core::fields::qm31::SecureField>> {
        todo!()
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
        todo!()
    }
}
