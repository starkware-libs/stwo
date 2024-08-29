use std::borrow::Cow;
use std::iter::zip;
use std::ops::Deref;

use itertools::Itertools;
use tracing::{span, Level};

use super::{EvalAtRow, InfoEvaluator, PointEvaluator, SimdDomainEvaluator};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, Trace};
use crate::core::backend::simd::column::VeryPackedSecureColumnByCoords;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::very_packed_m31::{VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS};
use crate::core::backend::simd::SimdBackend;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::{TreeSubspan, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::{utils, ColumnVec};

// TODO(andrew): Docs.
// TODO(andrew): Consider better location for this.
#[derive(Debug, Default)]
pub struct TraceLocationAllocator {
    /// Mapping of tree index to next available column offset.
    next_tree_offsets: TreeVec<usize>,
}

impl TraceLocationAllocator {
    fn next_for_structure<T>(&mut self, structure: &TreeVec<ColumnVec<T>>) -> TreeVec<TreeSubspan> {
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
}

/// A component defined solely in means of the constraints framework.
/// Implementing this trait introduces implementations for [`Component`] and [`ComponentProver`] for
/// the SIMD backend.
/// Note that the constraint framework only support components with columns of the same size.
pub trait FrameworkEval {
    fn log_size(&self) -> u32;

    fn max_constraint_log_degree_bound(&self) -> u32;

    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E;
}

pub struct FrameworkComponent<C: FrameworkEval> {
    eval: C,
    trace_locations: TreeVec<TreeSubspan>,
}

impl<E: FrameworkEval> FrameworkComponent<E> {
    pub fn new(provider: &mut TraceLocationAllocator, eval: E) -> Self {
        let eval_tree_structure = eval.evaluate(InfoEvaluator::default()).mask_offsets;
        let trace_locations = provider.next_for_structure(&eval_tree_structure);
        Self {
            eval,
            trace_locations,
        }
    }
}

impl<E: FrameworkEval> Component for FrameworkComponent<E> {
    fn n_constraints(&self) -> usize {
        self.eval.evaluate(InfoEvaluator::default()).n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.eval.max_constraint_log_degree_bound()
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(
            self.eval
                .evaluate(InfoEvaluator::default())
                .mask_offsets
                .iter()
                .map(|tree_masks| vec![self.eval.log_size(); tree_masks.len()])
                .collect(),
        )
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let info = self.eval.evaluate(InfoEvaluator::default());
        let trace_step = CanonicCoset::new(self.eval.log_size()).step();
        info.mask_offsets.map_cols(|col_mask| {
            col_mask
                .iter()
                .map(|off| point + trace_step.mul_signed(*off).into_ef())
                .collect()
        })
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        self.eval.evaluate(PointEvaluator::new(
            mask.sub_tree(&self.trace_locations),
            evaluation_accumulator,
            coset_vanishing(CanonicCoset::new(self.eval.log_size()).coset, point).inverse(),
        ));
    }
}

impl<E: FrameworkEval> ComponentProver<SimdBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
    ) {
        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let trace_domain = CanonicCoset::new(self.eval.log_size());

        let component_polys = trace.polys.sub_tree(&self.trace_locations);
        let component_evals = trace.evals.sub_tree(&self.trace_locations);

        // Extend trace if necessary.
        // TODO(spapini): Don't extend when eval_size < committed_size. Instead, pick a good
        // subdomain.
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
        utils::bit_reverse(&mut denom_inv);

        // Accumulator.
        let [mut accum] =
            evaluation_accumulator.columns([(eval_domain.log_size(), self.n_constraints())]);
        accum.random_coeff_powers.reverse();

        let _span = span!(Level::INFO, "Constraint pointwise eval").entered();
        let col = unsafe { VeryPackedSecureColumnByCoords::transform_under_mut(accum.col) };

        for vec_row in 0..(1 << (eval_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)) {
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

            // Evaluate constrains at row.
            let eval = SimdDomainEvaluator::new(
                &trace_cols,
                vec_row,
                &accum.random_coeff_powers,
                trace_domain.log_size(),
                eval_domain.log_size(),
            );
            let row_res = self.eval.evaluate(eval).row_res;

            // Finalize row.
            unsafe {
                let denom_inv = VeryPackedBaseField::broadcast(
                    denom_inv[vec_row
                        >> (trace_domain.log_size() - LOG_N_LANES - LOG_N_VERY_PACKED_ELEMS)],
                );
                col.set_packed(vec_row, col.packed_at(vec_row) + row_res * denom_inv)
            }
        }
    }
}

impl<E: FrameworkEval> Deref for FrameworkComponent<E> {
    type Target = E;

    fn deref(&self) -> &E {
        &self.eval
    }
}
