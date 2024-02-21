use std::iter::zip;
use std::ops::Deref;

use self::evaluation::{
    ConstraintEvaluator, DomainEvaluationAccumulator, PointEvaluationAccumulator,
};
use super::backend::{Backend, CPUBackend};
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::poly::circle::{CanonicCoset, CirclePoly};
use super::ColumnVec;

pub mod evaluation;

/// Arithmetic Intermediate Representation (AIR).
/// An Air instance is assumed to already contain all the information needed to
/// evaluate the constraints.
/// For instance, all interaction elements are assumed to be present in it.
/// Therefore, an AIR is generated only after the initial trace commitment phase.
// TODO(spapini): consider renaming this struct.
pub trait Air<B: Backend> {
    fn visit_components<V: ComponentVisitor<B>>(&self, v: &mut V);
}

pub trait AirExt: Air<CPUBackend> {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        let mut visitor = MaxConstraintLogDegreeBoundVisitor::new();
        self.visit_components(&mut visitor);
        visitor.finalize()
    }

    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, CPUBackend>],
    ) -> CirclePoly<CPUBackend, SecureField> {
        let mut evaluator = ConstraintEvaluator::new(
            component_traces,
            self.max_constraint_log_degree_bound(),
            random_coeff,
        );
        self.visit_components(&mut evaluator);
        evaluator.finalize()
    }
}

impl<A: Air<CPUBackend>> AirExt for A {}

pub trait ComponentVisitor<B: Backend> {
    fn visit<C: Component<B>>(&mut self, component: &C);
}

struct MaxConstraintLogDegreeBoundVisitor(u32);

impl MaxConstraintLogDegreeBoundVisitor {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn finalize(self) -> u32 {
        self.0
    }
}

impl<B: Backend> ComponentVisitor<B> for MaxConstraintLogDegreeBoundVisitor {
    fn visit<C: Component<B>>(&mut self, component: &C) {
        self.0 = self.0.max(component.max_constraint_log_degree_bound());
    }
}

/// Holds the mask offsets at each column.
/// Holds a vector with an entry for each column. Each entry holds the offsets
/// of the mask at that column.
pub struct Mask(pub ColumnVec<Vec<usize>>);

impl Mask {
    pub fn to_points(
        &self,
        domains: Vec<CanonicCoset>,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>> {
        self.iter()
            .zip(domains.iter())
            .map(|(col, domain)| {
                col.iter()
                    .map(|i| point + domain.at(*i).into_ef())
                    .collect()
            })
            .collect()
    }
}

impl Deref for Mask {
    type Target = ColumnVec<Vec<usize>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component<B: Backend> {
    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Returns the degree bounds of each trace column.
    fn trace_log_degree_bounds(&self) -> Vec<u32>;

    /// Evaluates the constraint quotients of the component on constraint evaluation domains.
    /// See [`super::poly::circle::CircleDomain::constraint_evaluation_domain`].
    /// Accumulates quotients in `evaluation_accumulator`.
    // Note: This will be computed using a MaterializedGraph.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );

    fn mask(&self) -> Mask;

    /// Calculates the mask points and evaluates them at each column.
    /// The mask values are used to evaluate the composition polynomial at a certain point.
    /// Returns two vectors with an entry for each column. Each entry holds the points/values
    /// of the mask at that column.
    fn mask_points_and_values(
        &self,
        point: CirclePoint<SecureField>,
        trace: &ComponentTrace<'_, B>,
    ) -> (
        ColumnVec<Vec<CirclePoint<SecureField>>>,
        ColumnVec<Vec<SecureField>>,
    ) {
        let domains = trace
            .columns
            .iter()
            .map(|col| CanonicCoset::new(col.log_size()))
            .collect();
        let points = self.mask().to_points(domains, point);
        let values = zip(&points, &trace.columns)
            .map(|(col_points, col)| {
                col_points
                    .iter()
                    .map(|point| col.eval_at_point(*point))
                    .collect()
            })
            .collect();

        (points, values)
    }

    /// Evaluates the constraint quotients combination of the component, given the mask values.
    fn evaluate_quotients_by_mask(
        &self,
        point: CirclePoint<SecureField>,
        mask: &[SecureField],
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    );

    // TODO(spapini): Extra functions for FRI and decommitment.
}

pub struct ComponentTrace<'a, B: Backend> {
    pub columns: Vec<&'a CirclePoly<B, BaseField>>,
}

impl<'a, B: Backend> ComponentTrace<'a, B> {
    pub fn new(columns: Vec<&'a CirclePoly<B, BaseField>>) -> Self {
        Self { columns }
    }
}
