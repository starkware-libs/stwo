use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::air::mask::fixed_mask_points;
use crate::core::air::{Air, Component};
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CanonicCoset;
use crate::core::ColumnVec;

/// Component that computes fibonacci numbers over 64 columns.
pub struct WideFibComponent {
    pub log_size: u32,
}

pub struct WideFibAir {
    pub component: WideFibComponent,
}

impl Air for WideFibAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }
}

impl Component for WideFibComponent {
    fn n_constraints(&self) -> usize {
        255
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_size; 256]
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
        fixed_mask_points(&vec![vec![0_usize]; 256], point)
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_size).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        let numerator = mask[0][0] - BaseField::from_u32_unchecked(1);
        evaluation_accumulator.accumulate(numerator * denom_inverse);

        for i in 0..254 {
            let numerator = mask[i][0].square() + mask[i + 1][0].square() - mask[i + 2][0];
            evaluation_accumulator.accumulate(numerator * denom_inverse);
        }
    }
}

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}
