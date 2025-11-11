use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::{
    bit_reverse_index, circle_domain_index_to_coset_index, coset_index_to_circle_domain_index,
};
use stwo::core::Fraction;
use stwo::parallel_iter;
use stwo::prover::backend::{Backend, Column};
use stwo::prover::poly::circle::CircleCoefficients;

use crate::logup::LogupAtRow;
use crate::{EvalAtRow, INTERACTION_TRACE_IDX};

/// Evaluates expressions at a trace domain row, and asserts constraints. Mainly used for testing.
pub struct AssertEvaluator<'a> {
    pub trace: &'a TreeVec<Vec<&'a Vec<BaseField>>>,
    pub col_index: TreeVec<usize>,
    pub row: usize,
    pub constraint_counter: usize,
    pub logup: LogupAtRow<Self>,
}
impl<'a> AssertEvaluator<'a> {
    pub fn new(
        trace: &'a TreeVec<Vec<&Vec<BaseField>>>,
        row: usize,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace,
            col_index: TreeVec::new(vec![0; trace.len()]),
            row,
            constraint_counter: 0,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}
impl EvalAtRow for AssertEvaluator<'_> {
    type F = BaseField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.col_index[interaction];
        self.col_index[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, we can just return the value directly from this row.
            if off == 0 {
                let col = &self.trace[interaction][col_index];
                return col[self.row];
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            let log_size = self.logup.log_size;
            let domain_size = 1 << log_size;

            let coset_index =
                circle_domain_index_to_coset_index(bit_reverse_index(self.row, log_size), log_size);
            let next_coset_index = (coset_index as isize + off).rem_euclid(domain_size);
            let next_index = bit_reverse_index(
                coset_index_to_circle_domain_index(next_coset_index as usize, log_size),
                log_size,
            );
            self.trace[interaction][col_index].at(next_index)
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF> + From<G>,
    {
        // Cast to SecureField.
        // The constraint should be zero at the given row, since we are evaluating on the trace
        // domain.
        assert_eq!(
            Self::EF::from(constraint),
            SecureField::zero(),
            "row: #{}, constraint #{}",
            self.row,
            self.constraint_counter
        );
        self.constraint_counter += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        SecureField::from_m31_array(values)
    }

    crate::logup_proxy!();
}

pub fn assert_constraints_on_polys<B: Backend>(
    trace_polys: &TreeVec<Vec<CircleCoefficients<B>>>,
    trace_domain: CanonicCoset,
    assert_func: impl Fn(AssertEvaluator<'_>) + Sync,
    claimed_sum: SecureField,
) {
    let traces = trace_polys.as_ref().map(|tree| {
        tree.iter()
            .map(|poly| poly.evaluate(trace_domain.circle_domain()).values.to_cpu())
            .collect_vec()
    });
    let traces = &traces.as_ref();
    let traces = traces.into();
    assert_constraints_on_trace(&traces, trace_domain.log_size(), assert_func, claimed_sum);
}

pub fn assert_constraints_on_trace(
    evals: &TreeVec<Vec<&Vec<M31>>>,
    log_size: u32,
    assert_func: impl Fn(AssertEvaluator<'_>) + Sync,
    claimed_sum: SecureField,
) {
    let n_rows = 1 << log_size;

    let iter = parallel_iter!(0..n_rows);
    iter.for_each(|row| {
        let eval = AssertEvaluator::new(evals, row, log_size, claimed_sum);
        assert_func(eval);
    });
}
