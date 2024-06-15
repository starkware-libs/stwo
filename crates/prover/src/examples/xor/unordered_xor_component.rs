use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::Component;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::ColumnVec;
use crate::examples::xor::xor_table_component::{XOR_ALPHA_ID, XOR_Z_ID};

/// Component full of random 8-bit XOR operations.
pub struct UnorderedXorComponent;

impl Component for UnorderedXorComponent {
    fn n_constraints(&self) -> usize {
        0
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.trace_log_degree_bounds().into_iter().max().unwrap()
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        // TODO: Allow this component to be arbitrary size.
        // Three columns: 1. LHS operand 2. RHS operand 3. result (LHS ^ RHS)
        vec![u8::BITS + u8::BITS; 3]
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> crate::core::ColumnVec<Vec<CirclePoint<SecureField>>> {
        vec![vec![point], vec![point], vec![point]]
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![XOR_ALPHA_ID.to_string(), XOR_Z_ID.to_string()]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
    }
}
