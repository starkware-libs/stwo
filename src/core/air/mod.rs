use self::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::poly::circle::CirclePoly;

pub mod evaluation;

/// Arithmetic Intermediate Representation (AIR).
/// An Air instance is assumed to already contain all the information needed to
/// evaluate the constraints.
/// For instance, all interaction elements are assumed to be present in it.
/// Therefore, an AIR is generated only after the initial trace commitment phase.
// TODO(spapini): consider renaming this struct.
pub trait Air {
    fn visit_components<V: ComponentVisitor>(&self, v: &mut V);
}
pub trait ComponentVisitor {
    fn visit<C: Component>(&mut self, component: &C);
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component {
    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Evaluates the constraint quotients of the component on constraint evaluation domains.
    /// See [`super::poly::circle::CircleDomain::constraint_evaluation_domain`].
    /// Accumulates quotients in `evaluation_accumulator`.
    // Note: This will be computed using a MaterializedGraph.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator,
    );

    /// Calculates the mask points starting at the given point.
    fn mask_points(&self, point: CirclePoint<SecureField>) -> Vec<CirclePoint<SecureField>>;

    /// Evaluates the mask values for the constraints at a point.
    fn mask_values_at_point(
        &self,
        point: CirclePoint<SecureField>,
        component_trace: &ComponentTrace<'_>,
    ) -> (Vec<CirclePoint<SecureField>>, Vec<SecureField>);

    /// Evaluates the constraint quotients combination of the component, given the mask values.
    fn evaluate_quotients_by_mask(
        &self,
        point: CirclePoint<SecureField>,
        mask: &[SecureField],
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    );

    // TODO(spapini): Extra functions for FRI and decommitment.
}

pub struct ComponentTrace<'a> {
    pub columns: Vec<&'a CirclePoly<BaseField>>,
}

impl<'a> ComponentTrace<'a> {
    pub fn new(columns: Vec<&'a CirclePoly<BaseField>>) -> Self {
        Self { columns }
    }
}
