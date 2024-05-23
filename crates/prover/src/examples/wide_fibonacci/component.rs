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

pub const LOG_N_COLUMNS: usize = 8;
pub const N_COLUMNS: usize = 1 << LOG_N_COLUMNS;

/// Component that computes 2^`self.log_n_instances` instances of fibonacci sequences of size
/// 2^`self.log_fibonacci_size`. The numbers are computes over [N_COLUMNS] trace columns. The
/// number of rows (i.e the size of the columns) is determined by the parameters above (see
/// [WideFibComponent::log_column_size()]).
pub struct WideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl WideFibComponent {
    /// Returns the log of the size of the columns in the trace (which could also be looked at as
    /// the log number of rows).
    pub fn log_column_size(&self) -> u32 {
        self.log_n_instances + self.log_fibonacci_size - LOG_N_COLUMNS as u32
    }

    pub fn log_n_columns(&self) -> usize {
        LOG_N_COLUMNS
    }

    pub fn n_columns(&self) -> usize {
        N_COLUMNS
    }
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
        self.n_columns() - 2
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + 1
    }

    fn trace_log_degree_bounds(&self) -> Vec<u32> {
        vec![self.log_column_size(); self.n_columns()]
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
        fixed_mask_points(&vec![vec![0_usize]; self.n_columns()], point)
    }

    fn interaction_element_ids(&self) -> Vec<String> {
        vec![
            "wide_fibonacci_alpha".to_string(),
            "wide_fibonacci_z".to_string(),
        ]
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        for i in 0..self.n_columns() - 2 {
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
