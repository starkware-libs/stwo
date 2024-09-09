use std::ops::Mul;

use super::EvalAtRow;
use crate::core::air::accumulation::PointEvaluationAccumulator;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::pcs::TreeVec;
use crate::core::ColumnVec;

/// Evaluates expressions at a point out of domain.
pub struct PointEvaluator<'a> {
    pub mask: TreeVec<ColumnVec<&'a Vec<SecureField>>>,
    pub evaluation_accumulator: &'a mut PointEvaluationAccumulator,
    pub col_index: Vec<usize>,
    pub denom_inverse: SecureField,
}
impl<'a> PointEvaluator<'a> {
    pub fn new(
        mask: TreeVec<ColumnVec<&'a Vec<SecureField>>>,
        evaluation_accumulator: &'a mut PointEvaluationAccumulator,
        denom_inverse: SecureField,
    ) -> Self {
        let col_index = vec![0; mask.len()];
        Self {
            mask,
            evaluation_accumulator,
            col_index,
            denom_inverse,
        }
    }
}
impl<'a> EvalAtRow for PointEvaluator<'a> {
    type F = SecureField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        _offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.col_index[interaction];
        self.col_index[interaction] += 1;
        let mask = self.mask[interaction][col_index].clone();
        assert_eq!(mask.len(), N);
        mask.try_into().unwrap()
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        self.evaluation_accumulator
            .accumulate(self.denom_inverse * constraint);
    }
    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        SecureField::from_partial_evals(values)
    }
}

// /// Evaluates expressions at a point out of domain.
// pub struct MleCoeffColEvalAccumulator<'a> {
//     pub mask: TreeVec<ColumnVec<&'a Vec<SecureField>>>,
//     pub evaluation_accumulator: &'a mut PointEvaluationAccumulator,
//     pub col_index: Vec<usize>,
//     pub denom_inverse: SecureField,
// }
// impl<'a> MleCoeffColEvalAccumulator<'a> {
//     pub fn new(
//         mask: TreeVec<ColumnVec<&'a Vec<SecureField>>>,
//         evaluation_accumulator: &'a mut PointEvaluationAccumulator,
//         denom_inverse: SecureField,
//     ) -> Self {
//         let col_index = vec![0; mask.len()];
//         Self {
//             mask,
//             evaluation_accumulator,
//             col_index,
//             denom_inverse,
//         }
//     }
// }
// impl<'a> EvalAtRow for MleCoeffColEvalAccumulator<'a> {
//     type F = SecureField;
//     type EF = SecureField;

//     fn next_interaction_mask<const N: usize>(
//         &mut self,
//         interaction: usize,
//         _offsets: [isize; N],
//     ) -> [Self::F; N] {
//         let col_index = self.col_index[interaction];
//         self.col_index[interaction] += 1;
//         let mask = self.mask[interaction][col_index].clone();
//         assert_eq!(mask.len(), N);
//         mask.try_into().unwrap()
//     }
//     fn add_constraint<G>(&mut self, constraint: G)
//     where
//         Self::EF: Mul<G, Output = Self::EF>,
//     {
//     }
//     fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
//         SecureField::from_partial_evals(values)
//     }
// }

// impl<'a> EvalAtRowWithMle for MleCoeffColEvalAccumulator<'a> {
//     fn add_mle_coeff_col_eval(&mut self, eval: Self::EF) {
//         todo!()
//     }
// }
