use std::collections::BTreeSet;

use itertools::Itertools;

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Component, ComponentProver, Trace, CONST_INTERACTION};
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::SecureCirclePoly;
use crate::core::ColumnVec;

pub struct Components<'a> {
    pub components: Vec<&'a dyn Component>,
}

impl<'a> Components<'a> {
    pub fn composition_log_degree_bound(&self) -> u32 {
        self.components
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    pub fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.mask_points(point)),
        )
    }

    // Returns the unique mask points for each column.
    pub fn mask_points_by_column(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mut static_column_masks: Vec<BTreeSet<CirclePoint<SecureField>>> = vec![];
        for component in &self.components {
            let component_static_masks = &component.mask_points(point)[0];
            component_static_masks
                .iter()
                .zip(component.constant_column_locations())
                .for_each(|(points, index)| {
                    if index >= static_column_masks.len() {
                        static_column_masks.resize_with(index + 1, Default::default);
                    }
                    static_column_masks[index].extend(points);
                });
        }
        let mut components_masks = TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.mask_points(point)),
        );
        components_masks[0] = static_column_masks
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();

        components_masks
    }

    // Returns the mask evaluations according to the original mask for each component.
    pub fn evaluations_by_component(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &[Vec<SecureField>],
    ) -> ColumnVec<Vec<SecureField>> {
        let condensed_mask_points = &self.mask_points_by_column(point)[0];
        let mut expanded_values = vec![];

        for component in &self.components {
            let component_static_masks = &component.mask_points(point)[0];
            component_static_masks
                .iter()
                .zip(component.constant_column_locations())
                .for_each(|(points, index)| {
                    let mut self_evals = vec![];
                    let all_column_masks = &condensed_mask_points[index];
                    for &point in points {
                        let value = mask_values[index]
                            [all_column_masks.iter().position(|&p| p == point).unwrap()];
                        self_evals.push(value);
                    }
                    expanded_values.push(self_evals);
                });
        }

        expanded_values
    }

    pub fn reorganize_const_values_by_component(
        &self,
        point: CirclePoint<SecureField>,
        mut mask_values: TreeVec<ColumnVec<Vec<SecureField>>>,
    ) -> TreeVec<ColumnVec<Vec<SecureField>>> {
        mask_values[CONST_INTERACTION] =
            self.evaluations_by_component(point, &mask_values[CONST_INTERACTION]);
        mask_values
    }

    pub fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &TreeVec<Vec<Vec<SecureField>>>,
        random_coeff: SecureField,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        for component in &self.components {
            component.evaluate_constraint_quotients_at_point(
                point,
                mask_values,
                &mut evaluation_accumulator,
            );
            println!(
                "eval_composition_polynomial_at_point: evaluation_accumulator: {:?}",
                evaluation_accumulator.accumulation
            );
        }
        evaluation_accumulator.finalize()
    }

    pub fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.trace_log_degree_bounds()),
        )
    }
}

pub struct ComponentProvers<'a, B: Backend> {
    pub components: Vec<&'a dyn ComponentProver<B>>,
}

impl<'a, B: Backend> ComponentProvers<'a, B> {
    pub fn components(&self) -> Components<'_> {
        Components {
            components: self
                .components
                .iter()
                .map(|c| *c as &dyn Component)
                .collect_vec(),
        }
    }
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.components().composition_log_degree_bound(),
            total_constraints,
        );
        for component in &self.components {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator)
        }
        accumulator.finalize()
    }
}
