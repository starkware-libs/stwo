use itertools::{zip_eq, Itertools};

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Air, AirProver, ComponentTrace};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::pcs::{CommitmentTreeProver, TreeVec};
use crate::core::poly::circle::SecureCirclePoly;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::{ColumnVec, InteractionElements, LookupValues};

pub trait AirExt: Air {
    fn composition_log_degree_bound(&self) -> u32 {
        self.components()
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mut air_points = TreeVec::default();
        for component in self.components() {
            let component_points = component.mask_points(point);
            if air_points.len() < component_points.len() {
                air_points.resize(component_points.len(), vec![]);
            }
            air_points.as_mut().zip_eq(component_points).map(
                |(air_tree_points, component_tree_points)| {
                    air_tree_points.extend(component_tree_points);
                },
            );
        }
        // Add the composition polynomial mask points.
        air_points.push(vec![vec![point]; SECURE_EXTENSION_DEGREE]);
        air_points
    }

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &Vec<TreeVec<Vec<Vec<SecureField>>>>,
        random_coeff: SecureField,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        zip_eq(self.components(), mask_values).for_each(|(component, mask)| {
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

    fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        let mut air_sizes = TreeVec::default();
        self.components().iter().for_each(|component| {
            let component_sizes = component.trace_log_degree_bounds();
            if air_sizes.len() < component_sizes.len() {
                air_sizes.resize(component_sizes.len(), vec![]);
            }
            air_sizes.as_mut().zip_eq(component_sizes).map(
                |(air_tree_sizes, component_tree_sizes)| {
                    air_tree_sizes.extend(component_tree_sizes)
                },
            );
        });
        air_sizes
    }

    fn component_traces<'a, B: Backend + MerkleOps<H>, H: MerkleHasher>(
        &'a self,
        trees: &'a [CommitmentTreeProver<B, H>],
    ) -> Vec<ComponentTrace<'_, B>> {
        let mut poly_iters = trees
            .iter()
            .map(|tree| tree.polynomials.iter())
            .collect_vec();
        let mut eval_iters = trees
            .iter()
            .map(|tree| tree.evaluations.iter())
            .collect_vec();

        self.components()
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
}

impl<A: Air + ?Sized> AirExt for A {}

pub trait AirProverExt<B: Backend>: AirProver<B> {
    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        component_traces: &[ComponentTrace<'_, B>],
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
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
                lookup_values,
            )
        });
        accumulator.finalize()
    }

    fn lookup_values(&self, component_traces: &[ComponentTrace<'_, B>]) -> LookupValues {
        let mut values = LookupValues::default();
        zip_eq(self.prover_components(), component_traces)
            .for_each(|(component, trace)| values.extend(component.lookup_values(trace)));
        values
    }
}

impl<B: Backend, A: AirProver<B>> AirProverExt<B> for A {}
