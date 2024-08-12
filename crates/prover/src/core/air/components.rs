use itertools::{zip_eq, Itertools};

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::{Backend, BackendForChannel};
use crate::core::channel::MerkleChannel;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentTreeProver, TreeVec};
use crate::core::poly::circle::SecureCirclePoly;
use crate::core::{ColumnVec, InteractionElements, LookupValues};

pub struct Components<'a>(pub Vec<&'a dyn Component>);

impl<'a> Components<'a> {
    pub fn composition_log_degree_bound(&self) -> u32 {
        self.0
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    pub fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::concat_cols(self.0.iter().map(|component| component.mask_points(point)))
    }

    pub fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &Vec<TreeVec<Vec<Vec<SecureField>>>>,
        random_coeff: SecureField,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        zip_eq(&self.0, mask_values).for_each(|(component, mask)| {
            component.evaluate_constraint_quotients_at_point(
                point,
                mask,
                &mut evaluation_accumulator,
                interaction_elements,
                lookup_values,
            )
        });
        evaluation_accumulator.finalize()
    }

    pub fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::concat_cols(
            self.0
                .iter()
                .map(|component| component.trace_log_degree_bounds()),
        )
    }
}

pub struct ComponentProvers<'a, B: Backend>(pub Vec<&'a dyn ComponentProver<B>>);

impl<'a, B: Backend> ComponentProvers<'a, B> {
    pub fn components(&self) -> Components<'_> {
        Components(self.0.iter().map(|c| *c as &dyn Component).collect_vec())
    }
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.0.iter().map(|c| c.n_constraints()).sum();
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.components().composition_log_degree_bound(),
            total_constraints,
        );
        zip_eq(&self.0, component_traces).for_each(|(component, trace)| {
            component.evaluate_constraint_quotients_on_domain(
                trace,
                &mut accumulator,
                interaction_elements,
                lookup_values,
            )
        });
        accumulator.finalize()
    }

    pub fn component_traces<'b, MC: MerkleChannel>(
        &'b self,
        trees: &'b [CommitmentTreeProver<B, MC>],
    ) -> Vec<ComponentTrace<'b, B>>
    where
        B: BackendForChannel<MC>,
    {
        let mut poly_iters = trees
            .iter()
            .map(|tree| tree.polynomials.iter())
            .collect_vec();
        let mut eval_iters = trees
            .iter()
            .map(|tree| tree.evaluations.iter())
            .collect_vec();

        self.0
            .iter()
            .map(|component| {
                let col_sizes_per_tree = component
                    .trace_log_degree_bounds()
                    .iter()
                    .map(|col_sizes| col_sizes.len())
                    .collect_vec();
                let polys = col_sizes_per_tree
                    .iter()
                    .zip_eq(poly_iters.iter_mut())
                    .map(|(n_columns, iter)| iter.take(*n_columns).collect_vec())
                    .collect_vec();
                let evals = col_sizes_per_tree
                    .iter()
                    .zip_eq(eval_iters.iter_mut())
                    .map(|(n_columns, iter)| iter.take(*n_columns).collect_vec())
                    .collect_vec();
                ComponentTrace {
                    polys: TreeVec::new(polys),
                    evals: TreeVec::new(evals),
                }
            })
            .collect_vec()
    }

    pub fn lookup_values(&self, component_traces: &[ComponentTrace<'_, B>]) -> LookupValues {
        let mut values = LookupValues::default();
        zip_eq(&self.0, component_traces)
            .for_each(|(component, trace)| values.extend(component.lookup_values(trace)));
        values
    }
}
