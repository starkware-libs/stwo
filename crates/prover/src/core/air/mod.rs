pub use components::{ComponentProvers, Components};

use self::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::backend::Backend;
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::pcs::TreeVec;
use super::poly::circle::{CircleEvaluation, CirclePoly};
use super::poly::BitReversedOrder;
use super::ColumnVec;

pub mod accumulation;
mod components;
pub mod mask;

/// Arithmetic Intermediate Representation (AIR).
///
/// An Air instance is assumed to already contain all the information needed to evaluate the
/// constraints. For instance, all interaction elements are assumed to be present in it. Therefore,
/// an AIR is generated only after the initial trace commitment phase.
pub trait Air {
    fn components(&self) -> Vec<&dyn Component>;
}

pub trait AirProver<B: Backend>: Air {
    fn component_provers(&self) -> Vec<&dyn ComponentProver<B>>;
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

    fn preproccessed_column_indices(&self) -> ColumnVec<usize>;

    /// Evaluates the constraint quotients combination of the component at a point.
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    );
}

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );
}

/// The set of polynomials that make up the trace.
///
/// Each polynomial is stored both in a coefficients, and evaluations form (for efficiency)
pub struct Trace<'a, B: Backend> {
    /// Polynomials for each column.
    pub polys: TreeVec<ColumnVec<&'a CirclePoly<B>>>,
    /// Evaluations for each column (evaluated on their commitment domains).
    pub evals: TreeVec<ColumnVec<&'a CircleEvaluation<B, BaseField, BitReversedOrder>>>,
}
