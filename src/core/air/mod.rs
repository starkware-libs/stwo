use std::iter::zip;

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

pub use air_ext::AirExt;
use itertools::Itertools;

/// Arithmetic Intermediate Representation (AIR).
/// An Air instance is assumed to already contain all the information needed to
/// evaluate the constraints.
/// For instance, all interaction elements are assumed to be present in it.
/// Therefore, an AIR is generated only after the initial trace commitment phase.
// TODO(spapini): consider renaming this struct.
pub trait Air<B: Backend> {
    fn components(&self) -> Vec<&dyn Component<B>>;
}

/// A component is a set of trace columns of various sizes along with a set of
/// constraints on them.
pub trait Component<B: Backend> {
    fn n_constraints(&self) -> usize;

    fn max_constraint_log_degree_bound(&self) -> u32;

    /// Returns the degree bounds of each trace column.
    fn trace_log_degree_bounds(&self) -> Vec<u32>;

    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    // Note: This will be computed using a MaterializedGraph.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ColumnVec<Vec<CirclePoint<SecureField>>>;

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
        let points = self.mask_points(point);
        let values = zip(&points, &trace.polys)
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
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &ColumnVec<Vec<SecureField>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    );

    // TODO(spapini): Extra functions for FRI and decommitment.
}

/// A component trace is a set of polynomials and evaluations for each column on that component.
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
        assert_eq!(
            polys.iter().map(|p| p.log_size()).collect_vec(),
            evals.iter().map(|e| e.domain.log_size()).collect_vec()
        );
        Self { polys, evals }
    }
}
