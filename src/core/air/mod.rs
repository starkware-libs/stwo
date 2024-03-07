use core::slice;
use std::collections::BTreeMap;
use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;

use self::evaluation::{
    ConstraintEvaluator, ConstraintPointEvaluator, DomainEvaluationAccumulator,
    PointEvaluationAccumulator, SECURE_EXTENSION_DEGREE,
};
use super::backend::{Backend, CPUBackend};
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fri::CirclePolyDegreeBound;
use super::poly::circle::{CanonicCoset, CirclePoly, SecureCirclePoly};
use super::{ColumnVec, ComponentVec};

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
    ) -> SecureCirclePoly {
        let mut evaluator = ConstraintEvaluator::new(
            component_traces,
            self.max_constraint_log_degree_bound(),
            random_coeff,
        );
        self.visit_components(&mut evaluator);
        evaluator.finalize()
    }

    fn mask_points_and_values(
        &self,
        point: CirclePoint<SecureField>,
        component_traces: &[ComponentTrace<'_, CPUBackend>],
    ) -> (
        ComponentVec<Vec<CirclePoint<SecureField>>>,
        ComponentVec<Vec<SecureField>>,
    ) {
        let mut visitor = MaskEvaluator::new(point, component_traces);
        self.visit_components(&mut visitor);
        visitor.finalize()
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ComponentVec<Vec<CirclePoint<SecureField>>> {
        let mut visitor = MaskPointsEvaluator::new(point);
        self.visit_components(&mut visitor);
        visitor.finalize()
    }

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &ComponentVec<Vec<SecureField>>,
        random_coeff: SecureField,
    ) -> SecureField {
        let mut evaluator = ConstraintPointEvaluator::new(
            point,
            mask_values,
            self.max_constraint_log_degree_bound(),
            random_coeff,
        );
        self.visit_components(&mut evaluator);
        evaluator.finalize()
    }

    /// Returns the log degree bounds of the quotient polynomials in descending order.
    fn quotient_log_bounds(&self) -> Vec<CirclePolyDegreeBound> {
        let mut bounds_visitor = QuotientLogBoundsVisitor::new();
        self.visit_components(&mut bounds_visitor);
        let mut bounds = bounds_visitor.finalize();
        // Add the composition polynomial's log degree bounds.
        bounds.extend([self.max_constraint_log_degree_bound(); SECURE_EXTENSION_DEGREE]);
        bounds
            .into_iter()
            .rev()
            .map(CirclePolyDegreeBound::new)
            .collect()
    }

    fn component_traces<'a>(
        &'a self,
        polynomials: &'a [CirclePoly<CPUBackend, BaseField>],
    ) -> Vec<ComponentTrace<'_, CPUBackend>> {
        let mut component_traces_visitor = ComponentTracesVisitor::new(polynomials);
        self.visit_components(&mut component_traces_visitor);
        component_traces_visitor.finalize()
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

struct MaskEvaluator<'a, B: Backend> {
    point: CirclePoint<SecureField>,
    component_traces: slice::Iter<'a, ComponentTrace<'a, B>>,
    component_points: ComponentVec<Vec<CirclePoint<SecureField>>>,
    component_values: ComponentVec<Vec<SecureField>>,
}

impl<'a, B: Backend> MaskEvaluator<'a, B> {
    pub fn new(
        point: CirclePoint<SecureField>,
        component_traces: &'a [ComponentTrace<'a, B>],
    ) -> Self {
        Self {
            point,
            component_traces: component_traces.iter(),
            component_points: Vec::new(),
            component_values: Vec::new(),
        }
    }

    pub fn finalize(
        self,
    ) -> (
        ComponentVec<Vec<CirclePoint<SecureField>>>,
        ComponentVec<Vec<SecureField>>,
    ) {
        (self.component_points, self.component_values)
    }
}

impl<'a, B: Backend> ComponentVisitor<B> for MaskEvaluator<'a, B> {
    fn visit<C: Component<B>>(&mut self, component: &C) {
        let trace = self.component_traces.next().unwrap();
        let (points, values) = component.mask_points_and_values(self.point, trace);
        self.component_points.push(points);
        self.component_values.push(values);
    }
}

struct MaskPointsEvaluator {
    point: CirclePoint<SecureField>,
    points: ComponentVec<Vec<CirclePoint<SecureField>>>,
}

impl MaskPointsEvaluator {
    pub fn new(point: CirclePoint<SecureField>) -> Self {
        Self {
            point,
            points: Vec::new(),
        }
    }

    pub fn finalize(self) -> ComponentVec<Vec<CirclePoint<SecureField>>> {
        self.points
    }
}

impl<B: Backend> ComponentVisitor<B> for MaskPointsEvaluator {
    fn visit<C: Component<B>>(&mut self, component: &C) {
        let domains = component
            .trace_log_degree_bounds()
            .iter()
            .map(|&log_size| CanonicCoset::new(log_size))
            .collect_vec();
        self.points
            .push(component.mask().to_points(&domains, self.point));
    }
}

struct QuotientLogBoundsVisitor {
    // Maps the log degree bound to the number of quotients with that bound.
    bounds: BTreeMap<u32, usize>,
}

impl QuotientLogBoundsVisitor {
    pub fn new() -> Self {
        Self {
            bounds: BTreeMap::new(),
        }
    }

    pub fn finalize(self) -> Vec<u32> {
        self.bounds
            .into_iter()
            .flat_map(|(bound, n)| (0..n).map(|_| bound).collect_vec())
            .collect()
    }
}

impl<B: Backend> ComponentVisitor<B> for QuotientLogBoundsVisitor {
    fn visit<C: Component<B>>(&mut self, component: &C) {
        for (mask_points, trace_bound) in zip(
            component.mask().iter(),
            &component.trace_log_degree_bounds(),
        ) {
            let n = self.bounds.entry(*trace_bound);
            *n.or_default() += mask_points.len();
        }
    }
}

struct ComponentTracesVisitor<'a, B: Backend> {
    polynomials: slice::Iter<'a, CirclePoly<B, BaseField>>,
    component_traces: Vec<ComponentTrace<'a, B>>,
}

impl<'a, B: Backend> ComponentTracesVisitor<'a, B> {
    pub fn new(polynomials: &'a [CirclePoly<B, BaseField>]) -> Self {
        Self {
            polynomials: polynomials.iter(),
            component_traces: Vec::new(),
        }
    }

    pub fn finalize(self) -> Vec<ComponentTrace<'a, B>> {
        self.component_traces
    }
}

impl<'a, B: Backend> ComponentVisitor<B> for ComponentTracesVisitor<'a, B> {
    fn visit<C: Component<B>>(&mut self, component: &C) {
        let n_columns = component.trace_log_degree_bounds().len();
        let columns = (&mut self.polynomials).take(n_columns).collect();
        self.component_traces.push(ComponentTrace::new(columns));
    }
}

/// Holds the mask offsets at each column.
/// Holds a vector with an entry for each column. Each entry holds the offsets
/// of the mask at that column.
pub struct Mask(pub ColumnVec<Vec<usize>>);

impl Mask {
    pub fn to_points(
        &self,
        domains: &[CanonicCoset],
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
            .collect_vec();
        let points = self.mask().to_points(&domains, point);
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
        mask: &ColumnVec<Vec<SecureField>>,
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
