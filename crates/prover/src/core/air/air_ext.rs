use itertools::{zip_eq, Itertools};

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Air, AirProver, ComponentTrace};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentTreeProver, TreeVec};
use crate::core::poly::circle::SecureCirclePoly;
use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
use crate::core::vcs::ops::MerkleOps;
use crate::core::ColumnVec;

pub fn cmt_to_cmp<A: Air + ?Sized, T>(
    air: &A,
    cmt: TreeVec<ColumnVec<T>>,
) -> Vec<TreeVec<ColumnVec<T>>> {
    let mut iter_tree = cmt.map(|v| v.into_iter().enumerate());
    air.components()
        .iter()
        .map(|component| {
            let mut component_tree = TreeVec::default();
            for location in component.chunk_locations() {
                let chunk = (&mut iter_tree[location.tree_index])
                    .take(location.col_end - location.col_start)
                    .collect_vec();
                assert_eq!(chunk[0].0, location.col_start);
                let chunk = chunk.into_iter().map(|(_, v)| v).collect_vec();
                component_tree.push(chunk);
            }
            component_tree
        })
        .collect_vec()
}

pub fn cmp_to_cmt<A: Air + ?Sized, T>(
    air: &A,
    cmp: Vec<TreeVec<ColumnVec<T>>>,
) -> TreeVec<ColumnVec<T>> {
    let mut cmt = TreeVec::default();
    for (component, cmp_tree) in zip_eq(air.components(), cmp) {
        for (location, chunk) in component
            .chunk_locations()
            .iter()
            .zip(cmp_tree.0.into_iter())
        {
            let cur_len = cmt.len();
            cmt.resize_with(cur_len.max(location.tree_index + 1), || {
                ColumnVec::default()
            });
            assert_eq!(cmt[location.tree_index].len(), location.col_start);
            cmt[location.tree_index].extend(chunk.into_iter());
        }
    }
    cmt
}

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
        cmp_to_cmt(
            self,
            self.components()
                .iter()
                .map(|c| c.mask_points(point))
                .collect(),
        )
    }

    fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: TreeVec<ColumnVec<Vec<SecureField>>>,
        random_coeff: SecureField,
    ) -> SecureField {
        let component_masks = cmt_to_cmp(self, mask_values);
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        // zip_eq(self.components(), &mask_values.0).for_each(|(component, mask)| {
        self.components()
            .iter()
            .zip(component_masks)
            .for_each(|(component, mask)| {
                component.evaluate_constraint_quotients_at_point(
                    point,
                    &mask,
                    &mut evaluation_accumulator,
                )
            });
        evaluation_accumulator.finalize()
    }

    fn component_traces<'a, B: Backend + MerkleOps<Blake2sMerkleHasher>>(
        &'a self,
        trees: &'a [CommitmentTreeProver<B>],
    ) -> Vec<ComponentTrace<'_, B>> {
        let polys = cmt_to_cmp(
            self,
            TreeVec::new(
                trees
                    .iter()
                    .map(|tree| tree.polynomials.iter().collect_vec())
                    .collect_vec(),
            ),
        );
        let evals = cmt_to_cmp(
            self,
            TreeVec::new(
                trees
                    .iter()
                    .map(|tree| tree.evaluations.iter().collect_vec())
                    .collect_vec(),
            ),
        );
        zip_eq(polys, evals)
            .map(|(polys, evals)| ComponentTrace::new(polys, evals))
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
        zip_eq(self.prover_components(), component_traces).for_each(|(component, trace)| {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        });
        accumulator.finalize()
    }
}

impl<B: Backend, A: AirProver<B>> AirProverExt<B> for A {}
