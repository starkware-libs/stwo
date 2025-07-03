use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;
use stwo::core::air::accumulation::PointEvaluationAccumulator;
use stwo::core::air::Component;
use stwo::core::circle::CirclePoint;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::{TreeSubspan, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;

use super::preprocessed_columns::PreProcessedColumnId;
use super::{EvalAtRow, InfoEvaluator, PointEvaluator, PREPROCESSED_TRACE_IDX};

#[derive(Debug, Default)]
enum PreprocessedColumnsAllocationMode {
    #[default]
    Dynamic,
    Static,
}

// TODO(andrew): Docs.
// TODO(andrew): Consider better location for this.
#[derive(Debug, Default)]
pub struct TraceLocationAllocator {
    /// Mapping of tree index to next available column offset.
    next_tree_offsets: TreeVec<usize>,
    /// Mapping of preprocessed columns to their index.
    preprocessed_columns: Vec<PreProcessedColumnId>,
    /// Controls whether the preprocessed columns are dynamic or static (default=Dynamic).
    preprocessed_columns_allocation_mode: PreprocessedColumnsAllocationMode,
}

impl TraceLocationAllocator {
    pub fn next_for_structure<T>(
        &mut self,
        structure: &TreeVec<ColumnVec<T>>,
    ) -> TreeVec<TreeSubspan> {
        if structure.len() > self.next_tree_offsets.len() {
            self.next_tree_offsets.resize(structure.len(), 0);
        }

        TreeVec::new(
            zip(&mut *self.next_tree_offsets, &**structure)
                .enumerate()
                .map(|(tree_index, (offset, cols))| {
                    let col_start = *offset;
                    let col_end = col_start + cols.len();
                    *offset = col_end;
                    TreeSubspan {
                        tree_index,
                        col_start,
                        col_end,
                    }
                })
                .collect(),
        )
    }

    /// Create a new `TraceLocationAllocator` with fixed preprocessed columns setup.
    pub fn new_with_preproccessed_columns(preprocessed_columns: &[PreProcessedColumnId]) -> Self {
        assert!(
            preprocessed_columns.iter().all_unique(),
            "Duplicate preprocessed columns are not allowed!"
        );
        Self {
            next_tree_offsets: Default::default(),
            preprocessed_columns: preprocessed_columns.to_vec(),
            preprocessed_columns_allocation_mode: PreprocessedColumnsAllocationMode::Static,
        }
    }

    pub const fn preprocessed_columns(&self) -> &Vec<PreProcessedColumnId> {
        &self.preprocessed_columns
    }

    // Validates that `self.preprocessed_columns` is a permutation of
    // `preprocessed_columns`.
    pub fn validate_preprocessed_columns(&self, preprocessed_columns: &[PreProcessedColumnId]) {
        let mut self_columns = self.preprocessed_columns.clone();
        let mut input_columns = preprocessed_columns.to_vec();
        self_columns.sort_by_key(|col| col.id.clone());
        input_columns.sort_by_key(|col| col.id.clone());
        assert_eq!(
            self_columns, input_columns,
            "Preprocessed columns are not a permutation."
        );
    }
}

/// A component defined solely in means of the constraints framework.
///
/// Implementing this trait introduces implementations for [`Component`] and [`ComponentProver`] for
/// the SIMD backend. Note that the constraint framework only supports components with columns of
/// the same size.
///
/// [`ComponentProver`]: stwo::prover::ComponentProver
pub trait FrameworkEval {
    fn log_size(&self) -> u32;

    fn max_constraint_log_degree_bound(&self) -> u32;

    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E;
}

pub struct FrameworkComponent<C: FrameworkEval> {
    pub(super) eval: C,
    pub(super) trace_locations: TreeVec<TreeSubspan>,
    pub(super) preprocessed_column_indices: Vec<usize>,
    pub(super) claimed_sum: SecureField,
    info: InfoEvaluator,
}

impl<E: FrameworkEval> FrameworkComponent<E> {
    pub fn new(
        location_allocator: &mut TraceLocationAllocator,
        eval: E,
        claimed_sum: SecureField,
    ) -> Self {
        let info = eval.evaluate(InfoEvaluator::new(eval.log_size(), vec![], claimed_sum));
        let trace_locations = location_allocator.next_for_structure(&info.mask_offsets);

        let preprocessed_column_indices = info
            .preprocessed_columns
            .iter()
            .map(|col| {
                let next_column = location_allocator.preprocessed_columns.len();
                if let Some(pos) = location_allocator
                    .preprocessed_columns
                    .iter()
                    .position(|x| x.id == col.id)
                {
                    pos
                } else {
                    if matches!(
                        location_allocator.preprocessed_columns_allocation_mode,
                        PreprocessedColumnsAllocationMode::Static
                    ) {
                        panic!(
                            "Preprocessed column {:?} is missing from static allocation",
                            col
                        );
                    }
                    location_allocator.preprocessed_columns.push(col.clone());
                    next_column
                }
            })
            .collect();
        Self {
            eval,
            trace_locations,
            info,
            preprocessed_column_indices,
            claimed_sum,
        }
    }

    pub fn trace_locations(&self) -> &[TreeSubspan] {
        &self.trace_locations
    }

    pub fn preproccessed_column_indices(&self) -> &[usize] {
        &self.preprocessed_column_indices
    }

    pub const fn claimed_sum(&self) -> SecureField {
        self.claimed_sum
    }

    pub fn logup_counts(&self) -> RelationCounts {
        let size = 1 << self.eval.log_size();
        RelationCounts(
            self.info
                .logup_counts
                .iter()
                .map(|(k, v)| (k.clone(), v * size))
                .collect(),
        )
    }
}

pub struct RelationCounts(HashMap<String, usize>);
impl Deref for RelationCounts {
    type Target = HashMap<String, usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<E: FrameworkEval> Component for FrameworkComponent<E> {
    fn n_constraints(&self) -> usize {
        self.info.n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.eval.max_constraint_log_degree_bound()
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        let mut log_degree_bounds = self
            .info
            .mask_offsets
            .as_ref()
            .map(|tree_offsets| vec![self.eval.log_size(); tree_offsets.len()]);

        log_degree_bounds[0] = self
            .preprocessed_column_indices
            .iter()
            .map(|_| self.eval.log_size())
            .collect();

        log_degree_bounds
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let trace_step = CanonicCoset::new(self.eval.log_size()).step();
        self.info.mask_offsets.as_ref().map_cols(|col_offsets| {
            col_offsets
                .iter()
                .map(|offset| point + trace_step.mul_signed(*offset).into_ef())
                .collect()
        })
    }

    fn preproccessed_column_indices(&self) -> ColumnVec<usize> {
        self.preprocessed_column_indices.clone()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let preprocessed_mask = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &mask[PREPROCESSED_TRACE_IDX][*idx])
            .collect_vec();

        let mut mask_points = mask.sub_tree(&self.trace_locations);
        mask_points[PREPROCESSED_TRACE_IDX] = preprocessed_mask;

        self.eval.evaluate(PointEvaluator::new(
            mask_points,
            evaluation_accumulator,
            coset_vanishing(CanonicCoset::new(self.eval.log_size()).coset, point).inverse(),
            self.eval.log_size(),
            self.claimed_sum,
        ));
    }
}

impl<E: FrameworkEval> Deref for FrameworkComponent<E> {
    type Target = E;

    fn deref(&self) -> &E {
        &self.eval
    }
}

impl<E: FrameworkEval> Display for FrameworkComponent<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let log_n_rows = self.log_size();
        let mut n_cols = vec![];
        self.trace_log_degree_bounds()
            .0
            .iter()
            .for_each(|interaction| {
                n_cols.push(interaction.len());
            });
        writeln!(f, "n_rows 2^{}", log_n_rows)?;
        writeln!(f, "n_constraints {}", self.n_constraints())?;
        writeln!(
            f,
            "constraint_log_degree_bound {}",
            self.max_constraint_log_degree_bound()
        )?;
        writeln!(
            f,
            "total felts: 2^{} * {}",
            log_n_rows,
            n_cols.iter().sum::<usize>()
        )?;
        for (j, n_cols) in n_cols.into_iter().enumerate() {
            writeln!(f, "\t Interaction {}: n_cols {}", j, n_cols)?;
        }
        Ok(())
    }
}
