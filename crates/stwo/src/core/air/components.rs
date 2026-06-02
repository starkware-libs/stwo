use core::iter::zip;

use std_shims::{vec, Vec};

use super::accumulation::PointEvaluationAccumulator;
use super::{AbsoluteColumnMaskSemanticStepLogSizes, Component};
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::verifier::PREPROCESSED_TRACE_IDX;
use crate::core::zk::{
    apply_zk_column_degree_bounds, ZkColumnDegreeBound, ZkColumnDegreeBoundApplicationError,
};
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
        max_log_degree_bound: u32,
        include_all_preprocessed_columns: bool,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let composition_log_degree_bound = self.composition_log_degree_bound();
        let mut mask_points = TreeVec::concat_cols(self.components.iter().map(|component| {
            let component_lift = if component.n_constraints() == 0 {
                0
            } else {
                composition_log_degree_bound - component.max_constraint_log_degree_bound()
            };
            let component_point = point.repeated_double(component_lift);
            component.mask_points(component_point, max_log_degree_bound)
        }));

        let preprocessed_mask_points = &mut mask_points[PREPROCESSED_TRACE_IDX];
        if include_all_preprocessed_columns {
            *preprocessed_mask_points = vec![vec![point]; self.n_preprocessed_columns];
        } else {
            *preprocessed_mask_points = vec![vec![]; self.n_preprocessed_columns];
            for component in &self.components {
                for idx in component.preprocessed_column_indices() {
                    preprocessed_mask_points[idx] = vec![point];
                }
            }
        }

        for component in &self.components {
            let component_lift = if component.n_constraints() == 0 {
                0
            } else {
                composition_log_degree_bound - component.max_constraint_log_degree_bound()
            };
            let component_point = point.repeated_double(component_lift);
            for extra in component.absolute_mask_points(component_point, max_log_degree_bound) {
                mask_points[extra.tree_index][extra.column_index].extend(extra.points);
            }
        }

        mask_points
    }

    pub fn mask_offsets(
        &self,
        include_all_preprocessed_columns: bool,
    ) -> Option<TreeVec<ColumnVec<Vec<isize>>>> {
        let component_mask_offsets = self
            .components
            .iter()
            .map(|component| component.mask_offsets())
            .collect::<Option<Vec<_>>>()?;
        let mut mask_offsets = TreeVec::concat_cols(component_mask_offsets.into_iter());

        let preprocessed_mask_offsets = &mut mask_offsets[PREPROCESSED_TRACE_IDX];
        if include_all_preprocessed_columns {
            *preprocessed_mask_offsets = vec![vec![0]; self.n_preprocessed_columns];
        } else {
            *preprocessed_mask_offsets = vec![vec![]; self.n_preprocessed_columns];
            for component in &self.components {
                for idx in component.preprocessed_column_indices() {
                    preprocessed_mask_offsets[idx] = vec![0];
                }
            }
        }

        for component in &self.components {
            for extra in component.absolute_mask_offsets()? {
                mask_offsets[extra.tree_index][extra.column_index].extend(extra.offsets);
            }
        }

        Some(mask_offsets)
    }

    pub fn semantic_step_log_sizes_and_lifts(
        &self,
        include_all_preprocessed_columns: bool,
    ) -> (TreeVec<ColumnVec<u32>>, TreeVec<ColumnVec<u32>>) {
        let composition_log_degree_bound = self.composition_log_degree_bound();
        let mut semantic_step_log_sizes = TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.trace_log_degree_bounds()),
        );
        let mut semantic_base_lifts =
            TreeVec::concat_cols(self.components.iter().map(|component| {
                let lift = if component.n_constraints() == 0 {
                    0
                } else {
                    composition_log_degree_bound - component.max_constraint_log_degree_bound()
                };
                component.trace_log_degree_bounds().map_cols(|_| lift)
            }));

        let preprocessed_step_log_sizes = &mut semantic_step_log_sizes[PREPROCESSED_TRACE_IDX];
        let preprocessed_base_lifts = &mut semantic_base_lifts[PREPROCESSED_TRACE_IDX];
        *preprocessed_step_log_sizes = vec![0; self.n_preprocessed_columns];
        *preprocessed_base_lifts = vec![0; self.n_preprocessed_columns];
        if include_all_preprocessed_columns {
            return (semantic_step_log_sizes, semantic_base_lifts);
        }

        for component in &self.components {
            let component_bounds = component.trace_log_degree_bounds();
            let lift = if component.n_constraints() == 0 {
                0
            } else {
                composition_log_degree_bound - component.max_constraint_log_degree_bound()
            };
            for (local_index, column_index) in component
                .preprocessed_column_indices()
                .into_iter()
                .enumerate()
            {
                preprocessed_step_log_sizes[column_index] =
                    component_bounds[PREPROCESSED_TRACE_IDX][local_index];
                preprocessed_base_lifts[column_index] = lift;
            }
        }

        (semantic_step_log_sizes, semantic_base_lifts)
    }

    pub fn semantic_sample_step_log_sizes_and_lifts(
        &self,
        include_all_preprocessed_columns: bool,
    ) -> Option<(TreeVec<ColumnVec<Vec<u32>>>, TreeVec<ColumnVec<Vec<u32>>>)> {
        let composition_log_degree_bound = self.composition_log_degree_bound();
        let component_step_log_sizes = self
            .components
            .iter()
            .map(|component| {
                let component_offsets = component.mask_offsets()?;
                let component_bounds = component.trace_log_degree_bounds();
                Some(TreeVec(
                    component_offsets
                        .iter()
                        .enumerate()
                        .map(|(tree_index, offset_tree)| {
                            if tree_index == PREPROCESSED_TRACE_IDX {
                                return vec![vec![]; offset_tree.len()];
                            }
                            offset_tree
                                .iter()
                                .zip(&component_bounds[tree_index])
                                .map(|(column_offsets, &log_size)| {
                                    vec![log_size; column_offsets.len()]
                                })
                                .collect()
                        })
                        .collect(),
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let component_base_lifts = self
            .components
            .iter()
            .map(|component| {
                let lift = if component.n_constraints() == 0 {
                    0
                } else {
                    composition_log_degree_bound - component.max_constraint_log_degree_bound()
                };
                component.mask_offsets().map(|offsets| {
                    offsets.map_cols(|column_offsets| vec![lift; column_offsets.len()])
                })
            })
            .collect::<Option<Vec<_>>>()?;

        let mut semantic_step_log_sizes =
            TreeVec::concat_cols(component_step_log_sizes.into_iter());
        let mut semantic_base_lifts = TreeVec::concat_cols(component_base_lifts.into_iter());

        let preprocessed_step_log_sizes = &mut semantic_step_log_sizes[PREPROCESSED_TRACE_IDX];
        let preprocessed_base_lifts = &mut semantic_base_lifts[PREPROCESSED_TRACE_IDX];
        *preprocessed_step_log_sizes = vec![vec![]; self.n_preprocessed_columns];
        *preprocessed_base_lifts = vec![vec![]; self.n_preprocessed_columns];
        if include_all_preprocessed_columns {
            *preprocessed_step_log_sizes = vec![vec![0]; self.n_preprocessed_columns];
            *preprocessed_base_lifts = vec![vec![0]; self.n_preprocessed_columns];
        } else {
            for component in &self.components {
                let component_bounds = component.trace_log_degree_bounds();
                let lift = if component.n_constraints() == 0 {
                    0
                } else {
                    composition_log_degree_bound - component.max_constraint_log_degree_bound()
                };
                for (local_index, column_index) in component
                    .preprocessed_column_indices()
                    .into_iter()
                    .enumerate()
                {
                    preprocessed_step_log_sizes[column_index] =
                        vec![component_bounds[PREPROCESSED_TRACE_IDX][local_index]];
                    preprocessed_base_lifts[column_index] = vec![lift];
                }
            }
        }

        for component in &self.components {
            let lift = if component.n_constraints() == 0 {
                0
            } else {
                composition_log_degree_bound - component.max_constraint_log_degree_bound()
            };
            let absolute_steps: Vec<AbsoluteColumnMaskSemanticStepLogSizes> =
                component.absolute_mask_semantic_step_log_sizes()?;
            for extra in absolute_steps {
                let opening_count = extra.step_log_sizes.len();
                semantic_step_log_sizes[extra.tree_index][extra.column_index]
                    .extend(extra.step_log_sizes);
                semantic_base_lifts[extra.tree_index][extra.column_index]
                    .extend(vec![lift; opening_count]);
            }
        }

        Some((semantic_step_log_sizes, semantic_base_lifts))
    }

    pub fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &TreeVec<Vec<Vec<SecureField>>>,
        random_coeff: SecureField,
        max_log_degree_bound: u32,
    ) -> SecureField {
        let composition_log_degree_bound = self.composition_log_degree_bound();
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        for component in &self.components {
            let component_lift = if component.n_constraints() == 0 {
                0
            } else {
                composition_log_degree_bound - component.max_constraint_log_degree_bound()
            };
            let component_point = point.repeated_double(component_lift);
            component.evaluate_constraint_quotients_at_point(
                component_point,
                mask_values,
                &mut evaluation_accumulator,
                max_log_degree_bound,
            )
        }
        evaluation_accumulator.finalize()
    }

    pub fn eval_zk_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &TreeVec<Vec<Vec<SecureField>>>,
        random_coeff: SecureField,
        max_log_degree_bound: u32,
    ) -> SecureField {
        let composition_log_degree_bound = self.composition_log_degree_bound();
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        for component in &self.components {
            let component_lift = if component.n_constraints() == 0 {
                0
            } else {
                composition_log_degree_bound - component.max_constraint_log_degree_bound()
            };
            let component_point = point.repeated_double(component_lift);
            component.evaluate_zk_constraint_quotients_at_point(
                component_point,
                mask_values,
                &mut evaluation_accumulator,
                max_log_degree_bound,
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
                component.preprocessed_column_indices(),
                &component_trace_log_sizes[PREPROCESSED_TRACE_IDX],
            ) {
                let column_log_size = &mut preprocessed_columns_trace_log_sizes[column_index];
                if visited_columns[column_index] {
                    assert!(
                        *column_log_size == log_size,
                        "Preprocessed column size mismatch for column {column_index}"
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

    pub fn column_log_sizes_with_zk_bounds(
        &self,
        zk_bounds: &[ZkColumnDegreeBound],
    ) -> Result<TreeVec<ColumnVec<u32>>, ZkColumnDegreeBoundApplicationError> {
        apply_zk_column_degree_bounds(self.column_log_sizes(), zk_bounds)
    }
}
