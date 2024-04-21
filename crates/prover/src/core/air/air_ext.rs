use itertools::{izip, zip_eq, Itertools};

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Air, AirProver, ComponentTrace};
use crate::core::backend::Backend;
use crate::core::channel::Blake2sChannel;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentTreeProver, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, SecureCirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::LOG_BLOWUP_FACTOR;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::MerkleOps;
use crate::core::{ColumnVec, ComponentVec};

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

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &ComponentVec<Vec<SecureField>>,
        random_coeff: SecureField,
        interaction_elements: &[Vec<BaseField>],
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        izip!(self.components(), &mask_values.0, interaction_elements).for_each(
            |(component, mask, elements)| {
                component.evaluate_constraint_quotients_at_point(
                    point,
                    mask,
                    &mut evaluation_accumulator,
                    elements,
                )
            },
        );
        evaluation_accumulator.finalize()
    }

    fn column_log_sizes(&self) -> Vec<u32> {
        self.components()
            .iter()
            .flat_map(|component| component.trace_log_degree_bounds())
            .collect()
    }

    fn component_traces<'a, B: Backend + MerkleOps<Blake2sMerkleHasher>>(
        &'a self,
        trees: &'a [CommitmentTreeProver<B>],
    ) -> Vec<ComponentTrace<'_, B>> {
        let poly_iter = &mut trees[0].polynomials.iter();
        let eval_iter = &mut trees[0].evaluations.iter();
        let mut component_traces = vec![];
        self.components().iter().for_each(|component| {
            let n_columns = component.trace_log_degree_bounds().len();
            let polys = poly_iter.take(n_columns).collect_vec();
            let evals = eval_iter.take(n_columns).collect_vec();

            component_traces.push(ComponentTrace {
                polys: TreeVec::new(vec![polys]),
                evals: TreeVec::new(vec![evals]),
            });
        });

        if trees.len() > 1 {
            let poly_iter = &mut trees[1].polynomials.iter();
            let eval_iter = &mut trees[1].evaluations.iter();
            self.components()
                .iter()
                .zip_eq(&mut component_traces)
                .for_each(|(_component, component_trace)| {
                    // TODO(AlonH): Implement n_interaction_columns() for component.
                    let polys = poly_iter.take(1).collect_vec();
                    let evals = eval_iter.take(1).collect_vec();
                    component_trace.polys.push(polys);
                    component_trace.evals.push(evals);
                });
        }
        component_traces
    }

    fn interaction_elements(&self, channel: &mut Blake2sChannel) -> Vec<Vec<BaseField>> {
        self.components()
            .iter()
            .map(|component| component.interaction_elements(channel))
            .collect()
    }
}

impl<A: Air + ?Sized> AirExt for A {}

pub trait AirProverExt<B: Backend>: AirProver<B> {
    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
        interaction_elements: &[Vec<BaseField>],
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
        izip!(
            self.prover_components(),
            component_traces,
            interaction_elements
        )
        .for_each(|(component, trace, elements)| {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator, elements)
        });
        accumulator.finalize()
    }

    fn interact(
        &self,
        trace: &ColumnVec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
        elements: &[Vec<BaseField>],
    ) -> ComponentVec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
        let trace_iter = &mut trace.iter();
        ComponentVec(
            zip_eq(self.prover_components().iter(), elements.iter())
                .map(|(component, component_elements)| {
                    let n_columns = component.trace_log_degree_bounds().len();
                    let trace_columns = trace_iter.take(n_columns).collect_vec();
                    component.interact(&trace_columns, component_elements)
                })
                .collect(),
        )
    }
}

impl<B: Backend, A: AirProver<B>> AirProverExt<B> for A {}
