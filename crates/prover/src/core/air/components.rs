use std::iter::zip;

use itertools::Itertools;

use super::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use super::{Component, ComponentProver, Trace};
use crate::constraint_framework::PREPROCESSED_TRACE_IDX;
use crate::core::backend::Backend;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::SecureCirclePoly;
use crate::core::ColumnVec;

pub struct Components<'a> {
    pub components: Vec<&'a dyn Component>,
    pub n_preprocessed_columns: usize,
}

impl Components<'_> {
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
        let mut mask_points = TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.mask_points(point)),
        );

        let preprocessed_mask_points = &mut mask_points[PREPROCESSED_TRACE_IDX];
        *preprocessed_mask_points = vec![vec![]; self.n_preprocessed_columns];

        for component in &self.components {
            for idx in component.preproccessed_column_indices() {
                preprocessed_mask_points[idx] = vec![point];
            }
        }

        mask_points
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
            )
        }
        evaluation_accumulator.finalize()
    }

    pub fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        let mut preprocessed_columns_trace_log_sizes = vec![0; self.n_preprocessed_columns];
        let mut visited_columns = vec![false; self.n_preprocessed_columns];

        let mut column_log_sizes = TreeVec::concat_cols(self.components.iter().map(|component| {
            let component_trace_log_sizes = component.trace_log_degree_bounds();

            for (column_index, &log_size) in zip(
                component.preproccessed_column_indices(),
                &component_trace_log_sizes[PREPROCESSED_TRACE_IDX],
            ) {
                let column_log_size = &mut preprocessed_columns_trace_log_sizes[column_index];
                if visited_columns[column_index] {
                    assert!(
                        *column_log_size == log_size,
                        "Preprocessed column size mismatch for column {}",
                        column_index
                    );
                } else {
                    *column_log_size = log_size;
                    visited_columns[column_index] = true;
                }
            }

            component_trace_log_sizes
        }));

        assert!(
            visited_columns.iter().all(|&updated| updated),
            "Column size not set for all reprocessed columns"
        );

        column_log_sizes[PREPROCESSED_TRACE_IDX] = preprocessed_columns_trace_log_sizes;

        column_log_sizes
    }
}

pub struct ComponentProvers<'a, B: Backend> {
    pub components: Vec<&'a dyn ComponentProver<B>>,
    pub n_preprocessed_columns: usize,
}

impl<B: Backend> ComponentProvers<'_, B> {
    pub fn components(&self) -> Components<'_> {
        Components {
            components: self
                .components
                .iter()
                .map(|c| *c as &dyn Component)
                .collect_vec(),
            n_preprocessed_columns: self.n_preprocessed_columns,
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
