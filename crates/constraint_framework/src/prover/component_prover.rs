use std::borrow::Cow;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo_prover::core::air::Component;
use stwo_prover::core::constraints::coset_vanishing;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::CanonicCoset;
use stwo_prover::core::utils::bit_reverse;
use stwo_prover::prover::backend::simd::column::VeryPackedSecureColumnByCoords;
use stwo_prover::prover::backend::simd::m31::LOG_N_LANES;
use stwo_prover::prover::backend::simd::very_packed_m31::{
    VeryPackedBaseField, LOG_N_VERY_PACKED_ELEMS,
};
use stwo_prover::prover::backend::simd::SimdBackend;
use stwo_prover::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo_prover::prover::poly::BitReversedOrder;
use stwo_prover::prover::secure_column::SecureColumnByCoords;
use stwo_prover::prover::{ComponentProver, DomainEvaluationAccumulator, Trace};
use tracing::{span, Level};

use crate::cpu_domain::CpuDomainEvaluator;
use crate::simd_domain::SimdDomainEvaluator;
use crate::{FrameworkComponent, FrameworkEval, PREPROCESSED_TRACE_IDX};

const CHUNK_SIZE: usize = 1;

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
            let _span = span!(Level::INFO, "Constraint Extension").entered();
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

        let _span = span!(
            Level::INFO,
            "Constraint point-wise eval",
            class = "ConstraintEval"
        )
        .entered();

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
