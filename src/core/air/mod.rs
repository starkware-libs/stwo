use std::iter::zip;
use std::ops::Deref;

use self::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::circle::CirclePoint;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fri::CirclePolyDegreeBound;
use super::poly::circle::{CanonicCoset, CirclePoly};

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

/// Holds the mask offsets at each column.
/// Holds a vector with an entry for each column. Each entry holds the offsets
/// of the mask at that column.
pub struct Mask(pub Vec<Vec<usize>>);

impl Mask {
    pub fn to_points(
        &self,
        domains: Vec<CanonicCoset>,
        point: CirclePoint<SecureField>,
    ) -> Vec<Vec<CirclePoint<SecureField>>> {
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
    type Target = Vec<Vec<usize>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component {
    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Returns the degree bounds of each trace column.
    fn trace_log_degree_bounds(&self) -> Vec<u32>;

    /// Evaluates the constraint quotients of the component on constraint evaluation domains.
    /// See [`super::poly::circle::CircleDomain::constraint_evaluation_domain`].
    /// Accumulates quotients in `evaluation_accumulator`.
    // Note: This will be computed using a MaterializedGraph.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator,
    );

    fn mask(&self) -> Mask;

    /// Calculates the mask points and evaluates them at each column.
    /// The mask values are used to evaluate the composition polynomial at a certain point.
    /// Returns two vectors with an entry for each column. Each entry holds the points/values
    /// of the mask at that column.
    fn mask_points_and_values(
        &self,
        point: CirclePoint<SecureField>,
        trace: &ComponentTrace<'_>,
    ) -> (Vec<Vec<CirclePoint<SecureField>>>, Vec<Vec<SecureField>>) {
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

pub trait ComponentExt: Component {
    fn get_quotient_log_bounds(&self) -> Vec<CirclePolyDegreeBound> {
        zip(self.mask().iter(), &self.trace_log_degree_bounds())
            .flat_map(|(trace_points, trace_bound)| {
                trace_points
                    .iter()
                    .map(|_| CirclePolyDegreeBound::new(*trace_bound))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl<C: Component> ComponentExt for C {}

pub struct ComponentTrace<'a> {
    pub columns: Vec<&'a CirclePoly<BaseField>>,
}

impl<'a> ComponentTrace<'a> {
    pub fn new(columns: Vec<&'a CirclePoly<BaseField>>) -> Self {
        Self { columns }
    }
}
