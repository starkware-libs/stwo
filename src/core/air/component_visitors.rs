use std::collections::BTreeMap;
use std::iter::zip;

use itertools::Itertools;
use tracing::{span, Level};

use super::{Air, ComponentTrace};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure::{SecureCirclePoly, SECURE_EXTENSION_DEGREE};
use crate::core::fri::CirclePolyDegreeBound;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::ComponentVec;

pub trait AirExt<B: Backend>: Air<B> {
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.components()
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
    ) -> SecureCirclePoly<B> {
        let mut accumulator =
            DomainEvaluationAccumulator::new(random_coeff, self.max_constraint_log_degree_bound());
        zip(self.components(), component_traces).for_each(|(component, trace)| {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        });
        accumulator.finalize()
    }

    fn mask_points_and_values(
        &self,
        point: CirclePoint<SecureField>,
        component_traces: &[ComponentTrace<'_, B>],
    ) -> (
        ComponentVec<Vec<CirclePoint<SecureField>>>,
        ComponentVec<Vec<SecureField>>,
    ) {
        let _span = span!(Level::INFO, "Eval columns ood").entered();
        let mut component_points = ComponentVec(Vec::new());
        let mut component_values = ComponentVec(Vec::new());
        zip(self.components(), component_traces).for_each(|(component, trace)| {
            let (points, values) = component.mask_points_and_values(point, trace);
            component_points.0.push(points);
            component_values.0.push(values);
        });
        (component_points, component_values)
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ComponentVec<Vec<CirclePoint<SecureField>>> {
        let mut points = ComponentVec(Vec::new());
        self.components().iter().for_each(|component| {
            let domains = component
                .trace_log_degree_bounds()
                .iter()
                .map(|&log_size| CanonicCoset::new(log_size))
                .collect_vec();
            points.0.push(component.mask().to_points(&domains, point));
        });
        points
    }

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &ComponentVec<Vec<SecureField>>,
        random_coeff: SecureField,
    ) -> SecureField {
        let mut evaluation_accumulator =
            PointEvaluationAccumulator::new(random_coeff, self.max_constraint_log_degree_bound());
        zip(self.components(), &mask_values.0).for_each(|(component, mask)| {
            component.evaluate_constraint_quotients_at_point(
                point,
                mask,
                &mut evaluation_accumulator,
            )
        });
        evaluation_accumulator.finalize()
    }

    fn quotient_log_bounds(&self) -> Vec<CirclePolyDegreeBound> {
        let mut bounds = BTreeMap::new();
        self.components().iter().for_each(|component| {
            for (mask_points, trace_bound) in zip(
                component.mask().iter(),
                &component.trace_log_degree_bounds(),
            ) {
                let n = bounds.entry(*trace_bound);
                *n.or_default() += mask_points.len();
            }
        });
        let mut bounds = bounds
            .into_iter()
            .flat_map(|(bound, n)| (0..n).map(|_| bound).collect_vec())
            .collect_vec();
        // Add the composition polynomial's log degree bounds.
        bounds.extend([self.max_constraint_log_degree_bound(); SECURE_EXTENSION_DEGREE]);
        bounds
            .into_iter()
            .rev()
            .map(CirclePolyDegreeBound::new)
            .collect()
    }

    /// Returns the log degree bounds of the quotient polynomials in descending order.
    fn column_log_sizes(&self) -> Vec<u32> {
        self.components()
            .iter()
            .flat_map(|component| component.trace_log_degree_bounds())
            .collect()
    }

    fn component_traces<'a>(
        &'a self,
        polynomials: &'a [CirclePoly<B>],
        evals: &'a [CircleEvaluation<B, BaseField, BitReversedOrder>],
    ) -> Vec<ComponentTrace<'_, B>> {
        self.components()
            .iter()
            .map(|component| {
                let n_columns = component.trace_log_degree_bounds().len();
                let polys = polynomials.iter().take(n_columns).collect();
                let evals = evals.iter().take(n_columns).collect();
                ComponentTrace::new(polys, evals)
            })
            .collect()
    }
}

impl<B: Backend, A: Air<B>> AirExt<B> for A {}
