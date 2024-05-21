use self::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::backend::Backend;
use super::channel::Blake2sChannel;
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::pcs::TreeVec;
use super::poly::circle::{CircleEvaluation, CirclePoly};
use super::poly::BitReversedOrder;
use super::{ColumnVec, ComponentVec, InteractionElements};

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

pub trait AirTraceVerifier {
    fn interaction_elements(&self, channel: &mut Blake2sChannel) -> InteractionElements;
}

pub trait AirTraceWriter<B: Backend>: AirTraceVerifier {
    fn interact(
        &self,
        trace: &ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> ComponentVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;

    fn to_air_prover(&self) -> &impl AirProver<B>;
}

pub trait AirProver<B: Backend>: Air {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<B>>;
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component {
    fn n_constraints(&self) -> usize;

    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Returns the number of interaction phases done by the component.
    fn n_interaction_phases(&self) -> u32;

    /// Returns the degree bounds of each trace column. The returned TreeVec should be of size
    /// `n_interaction_phases`.
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>>;

    /// Returns the mask points for each trace column. The returned TreeVec should be of size
    /// `n_interaction_phases`.
    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>;

    /// Returns the ids of the interaction elements used by the component.
    fn interaction_element_ids(&self) -> Vec<String>;

    /// Evaluates the constraint quotients combination of the component, given the mask values.
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        interaction_elements: &InteractionElements,
    );
}

pub trait ComponentTraceWriter<B: Backend> {
    fn write_interaction_trace(
        &self,
        trace: &ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>,
        elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>;
}

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
        interaction_elements: &InteractionElements,
    );
}

/// A component trace is a set of polynomials for each column on that component.
/// Each polynomial is stored both in a coefficients, and evaluations form (for efficiency)
pub struct ComponentTrace<'a, B: Backend> {
    /// Polynomials for each column.
    pub polys: TreeVec<ColumnVec<&'a CirclePoly<B>>>,
    /// Evaluations for each column (evaluated on the commitment domains).
    pub evals: TreeVec<ColumnVec<&'a CircleEvaluation<B, BaseField, BitReversedOrder>>>,
}

impl<'a, B: Backend> ComponentTrace<'a, B> {
    pub fn new(
        polys: TreeVec<ColumnVec<&'a CirclePoly<B>>>,
        evals: TreeVec<ColumnVec<&'a CircleEvaluation<B, BaseField, BitReversedOrder>>>,
    ) -> Self {
        Self { polys, evals }
    }
}
