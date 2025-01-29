use std::borrow::Cow;
use std::fmt::{self, Display, Formatter};
use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{span, Level};

use super::cpu_domain::CpuDomainEvaluator;
use super::preprocessed_columns::PreProcessedColumnId;
use super::{
    EvalAtRow, InfoEvaluator, PointEvaluator, SimdDomainEvaluator, PREPROCESSED_TRACE_IDX,
};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, Trace};
use crate::core::backend::cpu::bit_reverse;
use crate::core::backend::simd::column::VeryPackedSecureColumnByCoords;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use crate::core::backend::simd::SimdBackend;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::{TreeSubspan, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::ColumnVec;

const CHUNK_SIZE: usize = 1;

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

    // validates that `self.preprocessed_columns` is consistent with
    // `preprocessed_columns`.
    // I.e. preprocessed_columns[i] == self.preprocessed_columns[i].
    // TODO(Gali): Change to only validating that this is a permutation (not necessarily the same
    // order).
    pub fn validate_preprocessed_columns(&self, preprocessed_columns: &[PreProcessedColumnId]) {
        assert_eq!(self.preprocessed_columns, preprocessed_columns);
    }
}

/// A component defined solely in means of the constraints framework.
///
/// Implementing this trait introduces implementations for [`Component`] and [`ComponentProver`] for
/// the SIMD backend. Note that the constraint framework only supports components with columns of
/// the same size.
pub trait FrameworkEval {
    fn log_size(&self) -> u32;

    fn max_constraint_log_degree_bound(&self) -> u32;

    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E;
}

pub struct FrameworkComponent<C: FrameworkEval> {
    pub(super) eval: C,
    pub(super) trace_locations: TreeVec<TreeSubspan>,
    pub(super) preprocessed_column_indices: Vec<usize>,
    info: InfoEvaluator,
    claimed_sum: SecureField,
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

impl<E: FrameworkEval + Sync> ComponentProver<SimdBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
    ) {
        if self.n_constraints() == 0 {
            return;
        }

        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let trace_domain = CanonicCoset::new(self.eval.log_size());

        let mut component_polys = trace.polys.sub_tree(&self.trace_locations);
        component_polys[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.polys[PREPROCESSED_TRACE_IDX][*idx])
            .collect();

        let mut component_evals = trace.evals.sub_tree(&self.trace_locations);
        component_evals[PREPROCESSED_TRACE_IDX] = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| &trace.evals[PREPROCESSED_TRACE_IDX][*idx])
            .collect();

        // Extend trace if necessary.
        // TODO: Don't extend when eval_size < committed_size. Instead, pick a good
        // subdomain. (For larger blowup factors).
        let need_to_extend = component_evals
            .iter()
            .flatten()
            .any(|c| c.domain != eval_domain);
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        > = if need_to_extend {
            let _span = span!(Level::INFO, "Extension").entered();
            let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);
            component_polys
                .as_cols_ref()
                .map_cols(|col| Cow::Owned(col.evaluate_with_twiddles(eval_domain, &twiddles)))
        } else {
            component_evals.clone().map_cols(|c| Cow::Borrowed(*c))
        };

        // Denom inverses.
        let log_expand = eval_domain.log_size() - trace_domain.log_size();
        let mut denom_inv = (0..1 << log_expand)
            .map(|i| coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse())
            .collect_vec();
        bit_reverse(&mut denom_inv);

        // Accumulator.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(Level::INFO, "Constraint point-wise eval").entered();

        if trace_domain.log_size() < LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS {
            // Fall back to CPU if the trace is too small.
            let mut col = accum.col.to_cpu();

            for row in 0..(1 << eval_domain.log_size()) {
                let trace_cols = trace.as_cols_ref().map_cols(|c| c.to_cpu());
                let trace_cols = trace_cols.as_cols_ref();

                // Evaluate constrains at row.
                let eval = CpuDomainEvaluator::new(
                    &trace_cols,
                    row,
                    &accum.random_coeff_powers,
                    trace_domain.log_size(),
                    eval_domain.log_size(),
                    self.eval.log_size(),
                    self.claimed_sum,
                );
                let row_res = self.eval.evaluate(eval).row_res;

                // Finalize row.
                let denom_inv = denom_inv[row >> trace_domain.log_size()];
                col.set(row, col.at(row) + row_res * denom_inv)
            }
            let col = SecureColumnByCoords::from_cpu(col);
            *accum.col = col;
            return;
        }

        let col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(accum.col) };

        let range = 0..(1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS));

        #[cfg(not(feature = "parallel"))]
        let iter = range.step_by(CHUNK_SIZE).zip(col.chunks_mut(CHUNK_SIZE));

        #[cfg(feature = "parallel")]
        let iter = range
            .into_par_iter()
            .step_by(CHUNK_SIZE)
            .zip(col.chunks_mut(CHUNK_SIZE));

        // Define any `self` values outside the loop to prevent the compiler thinking there is a
        // `Sync` requirement on `Self`.
        let self_eval = &self.eval;
        let self_claimed_sum = self.claimed_sum;

        iter.for_each(|(chunk_idx, mut chunk)| {
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

            for idx_in_chunk in 0..CHUNK_SIZE {
                let vec_row = chunk_idx * CHUNK_SIZE + idx_in_chunk;
                // Evaluate constrains at row.
                let eval = SimdDomainEvaluator::new(
                    &trace_cols,
                    vec_row,
                    &accum.random_coeff_powers,
                    trace_domain.log_size(),
                    eval_domain.log_size(),
                    self_eval.log_size(),
                    self_claimed_sum,
                );
                let row_res = self_eval.evaluate(eval).row_res;

                // Finalize row.
                unsafe {
                    let denom_inv = VeryPackedBaseField::broadcast(
                        denom_inv[vec_row
                            >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
                    );
                    chunk.set_packed(
                        idx_in_chunk,
                        chunk.packed_at(idx_in_chunk) + row_res * denom_inv,
                    )
                }
            }
        });
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
