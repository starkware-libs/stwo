use std::iter::zip;

use itertools::{zip_eq, Itertools};

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Air, AirProver, ComponentTrace};
use crate::core::backend::Backend;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly, SecureCirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::LOG_BLOWUP_FACTOR;
use crate::core::{ComponentVec, InteractionElements};

pub trait AirExt: Air {
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

    fn interaction_elements(&self, channel: &mut Blake2sChannel) -> InteractionElements {
        let ids = self
            .components()
            .iter()
            .flat_map(|component| component.interaction_element_ids())
            .sorted()
            .dedup()
            .collect_vec();
        let elements = channel.draw_felts(ids.len()).into_iter().map(|e| e.0 .0);
        InteractionElements(zip_eq(ids, elements).collect_vec())
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

    fn component_traces<'a, B: Backend>(
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
impl<A: Air + ?Sized> AirExt for A {}

pub trait AirProverExt<B: Backend>: AirProver<B> {
    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self
            .prover_components()
            .iter()
            .map(|c| c.n_constraints())
            .sum();
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.composition_log_degree_bound(),
            total_constraints,
        );
        zip(self.prover_components(), component_traces).for_each(|(component, trace)| {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        });
        accumulator.finalize()
    }
}
impl<B: Backend, A: AirProver<B>> AirProverExt<B> for A {}
