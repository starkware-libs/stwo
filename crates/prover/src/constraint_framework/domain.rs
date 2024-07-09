use std::ops::Mul;

use num_traits::Zero;

use super::EvalAtRow;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::offset_bit_reversed_circle_domain_index;

/// Evaluates expressions at an evaluation domain rows.
pub struct DomainEvaluator<'a> {
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    pub col_index: Vec<usize>,
    pub vec_row: usize,
    pub random_coeff_powers: &'a [SecureField],
    pub row_res: PackedSecureField,
    pub constraint_index: usize,
    pub domain_log_size: u32,
    pub eval_log_size: u32,
}
impl<'a> DomainEvaluator<'a> {
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
        random_coeff_powers: &'a [SecureField],
        domain_log_size: u32,
        eval_log_size: u32,
    ) -> Self {
        Self {
            trace_eval,
            col_index: vec![0; trace_eval.len()],
            vec_row,
            random_coeff_powers,
            row_res: PackedSecureField::zero(),
            constraint_index: 0,
            domain_log_size,
            eval_log_size,
        }
    }
}
impl<'a> EvalAtRow for DomainEvaluator<'a> {
    type F = PackedBaseField;
    type EF = PackedSecureField;

    // TODO(spapini): Remove all boundary checks.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.col_index[interaction];
        self.col_index[interaction] += 1;
        offsets.map(|off| {
            // TODO(spapini): Optimize.
            if off == 0 {
                return self.trace_eval[interaction][col_index].data[self.vec_row];
            }
            PackedBaseField::from_array(std::array::from_fn(|i| {
                let index = offset_bit_reversed_circle_domain_index(
                    (self.vec_row << LOG_N_LANES) + i,
                    self.domain_log_size,
                    self.eval_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index].at(index)
            }))
        })
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        self.row_res +=
            PackedSecureField::broadcast(self.random_coeff_powers[self.constraint_index])
                * constraint;
        self.constraint_index += 1;
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        PackedSecureField::from_packed_m31s(values)
    }
}
