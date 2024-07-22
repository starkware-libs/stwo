use num_traits::{One, Zero};

use super::EvalAtRow;
use crate::core::backend::{Backend, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CirclePoly};
use crate::core::utils::circle_domain_order_to_coset_order;

/// Evaluates expressions at a trace domain row, and asserts constraints. Mainly used for testing.
pub struct AssertEvaluator<'a> {
    pub trace: &'a TreeVec<Vec<Vec<BaseField>>>,
    pub col_index: TreeVec<usize>,
    pub row: usize,
}
impl<'a> AssertEvaluator<'a> {
    pub fn new(trace: &'a TreeVec<Vec<Vec<BaseField>>>, row: usize) -> Self {
        Self {
            trace,
            col_index: TreeVec::new(vec![0; trace.len()]),
            row,
        }
    }
}
impl<'a> EvalAtRow for AssertEvaluator<'a> {
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
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        // Cast to SecureField.
        let res = SecureField::one() * constraint;
        // The constraint should be zero at the given row, since we are evaluating on the trace
        // domain.
        assert_eq!(res, SecureField::zero(), "row: {}", self.row);
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        SecureField::from_m31_array(values)
    }
}

pub fn assert_constraints<B: Backend>(
    trace_polys: &TreeVec<Vec<CirclePoly<B>>>,
    trace_domain: CanonicCoset,
    assert_func: impl Fn(AssertEvaluator<'_>),
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
        let eval = AssertEvaluator::new(&traces, row);
        assert_func(eval);
    }
}
