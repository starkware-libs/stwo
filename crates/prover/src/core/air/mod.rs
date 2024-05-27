use self::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::backend::Backend;
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::poly::circle::{CircleEvaluation, CirclePoly};
use super::poly::BitReversedOrder;
use super::ColumnVec;

pub mod accumulation;
mod air_ext;
pub mod mask;

pub use air_ext::{AirExt, AirProverExt};

/// Arithmetic Intermediate Representation (AIR).
/// An Air instance is assumed to already contain all the information needed to
/// evaluate the constraints.
/// For instance, all interaction elements are assumed to be present in it.
/// Therefore, an AIR is generated only after the initial trace commitment phase.
// TODO(spapini): consider renaming this struct.
pub trait Air {
    fn components(&self) -> Vec<&dyn Component>;
}
pub trait AirProver<B: Backend>: Air {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<B>>;
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component {
    fn n_constraints(&self) -> usize;

    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Returns the degree bounds of each trace column.
    fn trace_log_degree_bounds(&self) -> Vec<u32>;

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>>;

    /// Returns the ids of the interaction elements used by the component.
    fn interaction_element_ids(&self) -> Vec<String>;

    /// Evaluates the constraint quotients combination of the component, given the mask values.
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    );
}

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );
}

/// A component trace is a set of polynomials for each column on that component.
/// Each polynomial is stored both in a coefficients, and evaluations form (for efficiency)
pub struct ComponentTrace<'a, B: Backend> {
    /// Polynomials for each column.
    pub polys: Vec<&'a CirclePoly<B>>,
    /// Evaluations for each column. The evaluation domain is the commitment domain for that column
    /// obtained from [AirExt::trace_commitment_domains()].
    pub evals: Vec<&'a CircleEvaluation<B, BaseField, BitReversedOrder>>,
}

impl<'a, B: Backend> ComponentTrace<'a, B> {
    pub fn new(
        polys: Vec<&'a CirclePoly<B>>,
        evals: Vec<&'a CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> Self {
        Self { polys, evals }
    }
}
