use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::Component;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::{ColumnVec, InteractionElements};
use crate::examples::xor::xor_table_component::{XOR_ALPHA_ID, XOR_Z_ID};

/// Component full of random 8-bit XOR operations.
pub struct UnorderedXorComponent;

impl Component for UnorderedXorComponent {
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
    ) {
        todo!()
    }
}
