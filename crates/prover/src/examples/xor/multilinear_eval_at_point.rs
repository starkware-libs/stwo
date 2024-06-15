use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::Component;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::ColumnVec;

struct UnivariateSumcheckComponent;

impl Component for UnivariateSumcheckComponent {
    fn n_constraints(&self) -> usize {
        todo!()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        todo!()
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        todo!()
    }

    fn mask_points(
        &self,
        _point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
        todo!()
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        todo!()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        todo!()
    }
}
