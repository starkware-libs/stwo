use std::ops::Mul;

use num_traits::Zero;

use super::EvalAtRow;
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::utils::offset_bit_reversed_circle_domain_index;

/// Evaluates constraints at an evaluation domain points.
pub struct SimdDomainEvaluator<'a> {
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
    pub column_index_per_interaction: Vec<usize>,
    /// The row index of the simd-vector row to evaluate the constraints at.
    pub vec_row: usize,
    pub random_coeff_powers: &'a [SecureField],
    pub row_res: PackedSecureField,
    pub constraint_index: usize,
    pub domain_log_size: u32,
    pub eval_domain_log_size: u32,
}
impl<'a> SimdDomainEvaluator<'a> {
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
        random_coeff_powers: &'a [SecureField],
        domain_log_size: u32,
        eval_log_size: u32,
    ) -> Self {
        Self {
            trace_eval,
            column_index_per_interaction: vec![0; trace_eval.len()],
            vec_row,
            random_coeff_powers,
            row_res: PackedSecureField::zero(),
            constraint_index: 0,
            domain_log_size,
            eval_domain_log_size: eval_log_size,
        }
    }
}
impl<'a> EvalAtRow for SimdDomainEvaluator<'a> {
    type F = PackedBaseField;
    type EF = PackedSecureField;

    // TODO(spapini): Remove all boundary checks.
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;
        offsets.map(|off| {
            // If the offset is 0, we can just return the value directly from this row.
            if off == 0 {
                return unsafe {
                    *self
                        .trace_eval
                        .get_unchecked(interaction)
                        .get_unchecked(col_index)
                        .data
                        .get_unchecked(self.vec_row)
                };
            }
            // Otherwise, we need to look up the value at the offset.
            // Since the domain is bit-reversed circle domain ordered, we need to look up the value
            // at the bit-reversed natural order index at an offset.
            PackedBaseField::from_array(std::array::from_fn(|i| {
                let row_index = offset_bit_reversed_circle_domain_index(
                    (self.vec_row << LOG_N_LANES) + i,
                    self.domain_log_size,
                    self.eval_domain_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index].at(row_index)
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

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        PackedSecureField::from_packed_m31s(values)
    }
}
