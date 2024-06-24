use itertools::{zip_eq, Itertools};

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Air, AirProver, ComponentTrace};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::pcs::{CommitmentTreeProver, TreeVec};
use crate::core::poly::circle::SecureCirclePoly;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::MerkleOps;
use crate::core::{ColumnVec, ComponentVec, InteractionElements};

pub trait AirExt: Air {
    fn composition_log_degree_bound(&self) -> u32 {
        self.components()
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    fn n_interaction_phases(&self) -> u32 {
        self.components()
            .iter()
            .map(|component| component.n_interaction_phases())
            .max()
            .unwrap()
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mut trace_component_points = vec![];
        let mut interaction_component_points = vec![];
        for component in self.components() {
            let points = component.mask_points(point);
            trace_component_points.extend(points[0].clone());
            interaction_component_points.extend(points[1].clone());
        }
        let mut points = TreeVec::new(vec![trace_component_points]);
        if !interaction_component_points
            .iter()
            .all(|column| column.is_empty())
        {
            points.push(interaction_component_points);
        }
        // Add the composition polynomial mask points.
        points.push(vec![vec![point]; SECURE_EXTENSION_DEGREE]);
        points
    }

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &ComponentVec<Vec<SecureField>>,
        random_coeff: SecureField,
        interaction_elements: &InteractionElements,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        zip_eq(self.components(), &mask_values.0).for_each(|(component, mask)| {
            component.evaluate_constraint_quotients_at_point(
                point,
                mask,
                &mut evaluation_accumulator,
                interaction_elements,
            )
        });
        evaluation_accumulator.finalize()
    }

    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        let mut trace_tree = vec![];
        let mut interaction_tree = vec![];
        self.components().iter().for_each(|component| {
            let bounds = component.trace_log_degree_bounds();
            trace_tree.extend(bounds[0].clone());
            interaction_tree.extend(bounds[1].clone());
        });
        let mut sizes = TreeVec::new(vec![trace_tree]);
        if !interaction_tree.is_empty() {
            sizes.push(interaction_tree);
        }
        sizes
    }

    fn component_traces<'a, B: Backend + MerkleOps<Blake2sMerkleHasher>>(
        &'a self,
        trees: &'a [CommitmentTreeProver<B>],
    ) -> Vec<ComponentTrace<'_, B>> {
        let poly_iter = &mut trees[0].polynomials.iter();
        let eval_iter = &mut trees[0].evaluations.iter();
        let mut component_traces = vec![];
        self.components().iter().for_each(|component| {
            let n_columns = component.trace_log_degree_bounds()[0].len();
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
                .for_each(|(component, component_trace)| {
                    let n_columns = component.trace_log_degree_bounds()[1].len();
                    let polys = poly_iter.take(n_columns).collect_vec();
                    let evals = eval_iter.take(n_columns).collect_vec();
                    component_trace.polys.push(polys);
                    component_trace.evals.push(evals);
                });
        }
        component_traces
    }
}

impl<A: Air + ?Sized> AirExt for A {}

pub trait AirProverExt<B: Backend>: AirProver<B> {
    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
        interaction_elements: &InteractionElements,
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
        zip_eq(self.prover_components(), component_traces).for_each(|(component, trace)| {
            component.evaluate_constraint_quotients_on_domain(
                trace,
                &mut accumulator,
                interaction_elements,
            )
        });
        accumulator.finalize()
    }
}

impl<B: Backend, A: AirProver<B>> AirProverExt<B> for A {}
