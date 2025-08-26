use std_shims::Vec;

use self::accumulation::PointEvaluationAccumulator;
use super::circle::CirclePoint;
use super::fields::qm31::SecureField;
use super::pcs::TreeVec;
use super::ColumnVec;

pub mod accumulation;
mod components;
pub use components::Components;

/// Arithmetic Intermediate Representation (AIR).
///
/// An Air instance is assumed to already contain all the information needed to evaluate the
/// constraints. For instance, all interaction elements are assumed to be present in it. Therefore,
/// an AIR is generated only after the initial trace commitment phase.
pub trait Air {
    fn components(&self) -> Vec<&dyn Component>;
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component {
    fn n_constraints(&self) -> usize;

    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Returns the degree bounds of each trace column. The returned TreeVec should be of size
    /// `n_interaction_phases`.
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>>;

    /// Returns the mask points for each trace column. The returned TreeVec should be of size
    /// `n_interaction_phases`.
    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>;

    fn preprocessed_column_indices(&self) -> ColumnVec<usize>;

    /// Evaluates the constraint quotients combination of the component at a point.
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    );
}
