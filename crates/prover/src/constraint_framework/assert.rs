use num_traits::Zero;

use super::logup::LogupAtRow;
use super::{EvalAtRow, INTERACTION_TRACE_IDX};
use crate::core::backend::{Backend, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CirclePoly};
use crate::core::utils::circle_domain_order_to_coset_order;

/// Evaluates expressions at a trace domain row, and asserts constraints. Mainly used for testing.
pub struct AssertEvaluator<'a> {
    pub trace: &'a TreeVec<Vec<Vec<BaseField>>>,
    pub col_index: TreeVec<usize>,
    pub row: usize,
    pub logup: LogupAtRow<Self>,
}
impl<'a> AssertEvaluator<'a> {
    pub fn new(
        trace: &'a TreeVec<Vec<Vec<BaseField>>>,
        row: usize,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace,
            col_index: TreeVec::new(vec![0; trace.len()]),
            row,
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
            // The mask row might wrap around the column size.
            let col_size = self.trace[interaction][col_index].len() as isize;
            self.trace[interaction][col_index]
                [(self.row as isize + off).rem_euclid(col_size) as usize]
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
            "row: {}",
            self.row
        );
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        SecureField::from_m31_array(values)
    }

    super::logup_proxy!();
}

pub fn assert_constraints<B: Backend>(
    trace_polys: &TreeVec<Vec<CirclePoly<B>>>,
    trace_domain: CanonicCoset,
    assert_func: impl Fn(AssertEvaluator<'_>),
    claimed_sum: SecureField,
) {
    let traces = trace_polys.as_ref().map(|tree| {
        tree.iter()
            .map(|poly| {
                circle_domain_order_to_coset_order(
                    &poly
                        .evaluate(trace_domain.circle_domain())
                        .bit_reverse()
                        .values
                        .to_cpu(),
                )
            })
            .collect()
    });
    for row in 0..trace_domain.size() {
        let eval = AssertEvaluator::new(&traces, row, trace_domain.log_size(), claimed_sum);

        assert_func(eval);
    }
}
