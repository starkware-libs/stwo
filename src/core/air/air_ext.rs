use std::iter::zip;

use itertools::Itertools;
use tracing::{span, Level};

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Air, ComponentTrace};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, SecureCirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::LOG_BLOWUP_FACTOR;
use crate::core::ComponentVec;

pub trait AirExt<B: Backend>: Air<B> {
    fn composition_log_degree_bound(&self) -> u32 {
        self.components()
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    fn trace_commitment_domains(&self) -> Vec<CanonicCoset> {
        self.column_log_sizes()
            .iter()
            .map(|&log_size| CanonicCoset::new(log_size + LOG_BLOWUP_FACTOR))
            .collect_vec()
    }

    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.components().iter().map(|c| c.n_constraints()).sum();
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.composition_log_degree_bound(),
            total_constraints,
        );
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
            component_points.push(points);
            component_values.push(values);
        });
        (component_points, component_values)
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> ComponentVec<Vec<CirclePoint<SecureField>>> {
        let mut component_points = ComponentVec(Vec::new());
        for component in self.components() {
            let points = component.mask_points(point);
            component_points.push(points);
        }
        component_points
    }

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &ComponentVec<Vec<SecureField>>,
        random_coeff: SecureField,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        zip(self.components(), &mask_values.0).for_each(|(component, mask)| {
            component.evaluate_constraint_quotients_at_point(
                point,
                mask,
                &mut evaluation_accumulator,
            )
        });
        evaluation_accumulator.finalize()
    }

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
        let poly_iter = &mut polynomials.iter();
        let eval_iter = &mut evals.iter();
        self.components()
            .iter()
            .map(|component| {
                let n_columns = component.trace_log_degree_bounds().len();
                let polys = poly_iter.take(n_columns).collect();
                let evals = eval_iter.take(n_columns).collect();
                ComponentTrace::new(polys, evals)
            })
            .collect()
    }
}

impl<B: Backend, A: Air<B>> AirExt<B> for A {}
