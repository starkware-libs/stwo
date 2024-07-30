use self::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::backend::Backend;
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::pcs::TreeVec;
use super::poly::circle::{CircleEvaluation, CirclePoly, SecureCirclePoly};
use super::poly::BitReversedOrder;
use super::{ColumnVec, InteractionElements, LookupValues};

pub mod accumulation;
mod air_ext;
pub mod mask;

pub use air_ext::AirExt;
use itertools::zip_eq;

/// Arithmetic Intermediate Representation (AIR).
/// An Air instance is assumed to already contain all the information needed to
/// evaluate the constraints.
/// For instance, all interaction elements are assumed to be present in it.
/// Therefore, an AIR is generated only after the initial trace commitment phase.
// TODO(spapini): consider renaming this struct.
pub trait Air {
    fn components(&self) -> Vec<&dyn Component>;
}

pub struct AirProver<B: Backend> {
    pub prover_components: Vec<Box<dyn ComponentProver<B>>>,
}

impl<B: Backend> AirProver<B> {
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self
            .prover_components
            .iter()
            .map(|c| c.n_constraints())
            .sum();
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.composition_log_degree_bound(),
            total_constraints,
        );
        zip_eq(&self.prover_components, component_traces).for_each(|(component, trace)| {
            component.evaluate_constraint_quotients_on_domain(
                trace,
                &mut accumulator,
                interaction_elements,
                lookup_values,
            )
        });
        accumulator.finalize()
    }

    pub fn lookup_values(&self, component_traces: &[ComponentTrace<'_, B>]) -> LookupValues {
        let mut values = LookupValues::default();
        zip_eq(&self.prover_components, component_traces)
            .for_each(|(component, trace)| values.extend(component.lookup_values(trace)));
        values
    }
}

impl<B: Backend> Air for AirProver<B> {
    fn components(&self) -> Vec<&dyn Component> {
        self.prover_components
            .iter()
            .map(|c| c.as_ref() as &dyn Component)
            .collect()
    }
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

    /// Evaluates the constraint quotients combination of the component at a point.
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    );
}

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    );

    /// Returns the values needed to evaluate the components lookup boundary constraints.
    fn lookup_values(&self, _trace: &ComponentTrace<'_, B>) -> LookupValues;
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
