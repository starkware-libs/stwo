use std::borrow::Cow;

use itertools::Itertools;
use tracing::{span, Level};

use super::{EvalAtRow, InfoEvaluator, PointEvaluator, SimdDomainEvaluator};
use crate::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentProver, ComponentTrace};
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::circle::CirclePoint;
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::LOG_BLOWUP_FACTOR;
use crate::core::{utils, ColumnVec, InteractionElements, LookupValues};

/// A component defined solely in means of the constraints framework.
/// Implementing this trait introduces implementations for [Component] and [ComponentProver] for the
/// SIMD backend.
/// Note that the constraint framework only support components with columns of the same size.
pub trait FrameworkComponent {
    fn log_size(&self) -> u32;
    fn max_constraint_log_degree_bound(&self) -> u32;
    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E;
}

impl<C: FrameworkComponent> Component for C {
    fn n_constraints(&self) -> usize {
        self.evaluate(InfoEvaluator::default()).n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        FrameworkComponent::max_constraint_log_degree_bound(self)
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(
            self.evaluate(InfoEvaluator::default())
                .mask_offsets
                .iter()
                .map(|tree_masks| vec![self.log_size(); tree_masks.len()])
                .collect(),
        )
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let info = self.evaluate(InfoEvaluator::default());
        let trace_step = CanonicCoset::new(self.log_size()).step();
        info.mask_offsets.map(|tree_mask| {
            tree_mask
                .iter()
                .map(|col_mask| {
                    col_mask
                        .iter()
                        .map(|off| point + trace_step.mul_signed(*off).into_ef())
                        .collect()
                })
                .collect()
        })
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        self.evaluate(PointEvaluator::new(
            mask.as_ref(),
            evaluation_accumulator,
            coset_vanishing(CanonicCoset::new(self.log_size()).coset, point).inverse(),
        ));
    }
}

impl<C: FrameworkComponent> ComponentProver<SimdBackend> for C {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, SimdBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<SimdBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let eval_domain = CanonicCoset::new(self.max_constraint_log_degree_bound()).circle_domain();
        let trace_domain = CanonicCoset::new(self.log_size());

        // Extend trace if necessary.
        // TODO(spapini): Don't extend when eval_size < committed_size. Instead, pick a good
        // subdomain.
        let trace: TreeVec<
            Vec<Cow<'_, CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        > = if eval_domain.log_size() != self.log_size() + LOG_BLOWUP_FACTOR {
            let _span = span!(Level::INFO, "Extension").entered();
            let twiddles = SimdBackend::precompute_twiddles(eval_domain.half_coset);
            trace
                .polys
                .as_cols_ref()
                .map_cols(|col| Cow::Owned(col.evaluate_with_twiddles(eval_domain, &twiddles)))
        } else {
            trace.evals.as_cols_ref().map_cols(|c| Cow::Borrowed(*c))
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
        for vec_row in 0..(1 << (eval_domain.log_size() - LOG_N_LANES)) {
            let trace_cols = trace.as_cols_ref().map_cols(|c| c.as_ref());

            // Evaluate constrains at row.
            let eval = SimdDomainEvaluator::new(
                &trace_cols,
                vec_row,
                &accum.random_coeff_powers,
                trace_domain.log_size(),
                eval_domain.log_size(),
            );
            let row_res = self.evaluate(eval).row_res;

            // Finalize row.
            unsafe {
                let denom_inv = PackedBaseField::broadcast(
                    denom_inv[vec_row >> (trace_domain.log_size() - LOG_N_LANES)],
                );
                accum
                    .col
                    .set_packed(vec_row, accum.col.packed_at(vec_row) + row_res * denom_inv)
            }
        }
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, SimdBackend>) -> LookupValues {
        LookupValues::default()
    }
}
