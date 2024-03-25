use super::structs::WideFibComponent;
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace, Mask};
use crate::core::backend::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::ColumnVec;

impl Component<CPUBackend> for WideFibComponent {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_size; 256]
    }

    fn mask(&self) -> Mask {
        Mask(vec![vec![0_usize]; 256])
    }

    fn evaluate_constraint_quotients_on_domain(
        &self,
        _trace: &ComponentTrace<'_, CPUBackend>,
        _evaluation_accumulator: &mut DomainEvaluationAccumulator<CPUBackend>,
    ) {
        unimplemented!("not implemented")
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        _point: CirclePoint<SecureField>,
        _mask: &ColumnVec<Vec<SecureField>>,
        _evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        unimplemented!("not implemented")
    }
}
