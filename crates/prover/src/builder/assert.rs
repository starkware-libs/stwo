use num_traits::{One, Zero};

use super::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;

pub struct AssertEvaluator<'a> {
    pub trace: &'a TreeVec<Vec<Vec<BaseField>>>,
    pub col_index: TreeVec<usize>,
    pub row: usize,
}
impl<'a> AssertEvaluator<'a> {
    pub fn new(trace: &'a TreeVec<Vec<Vec<BaseField>>>) -> Self {
        Self {
            trace,
            col_index: TreeVec::new(vec![0; trace.len()]),
            row: 0,
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
            self.trace[interaction][col_index][(self.row as isize + off)
                .rem_euclid(self.trace[interaction][col_index].len() as isize)
                as usize]
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        let res = SecureField::one() * constraint;
        assert_eq!(res, SecureField::zero(), "row: {}", self.row);
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        SecureField::from_m31_array(values)
    }
}
