use num_traits::One;

use crate::constraint_framework::preprocessed_columns::PreProcessedColumnId;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};

/// A column with `1` at every `2^log_step` positions, `0` elsewhere, shifted by offset.
#[derive(Debug)]
pub struct IsStepWithOffset {
    log_size: u32,
    log_step: u32,
    offset: usize,
}
impl IsStepWithOffset {
    pub const fn new(log_size: u32, log_step: u32, offset: usize) -> Self {
        Self {
            log_size,
            log_step,
            offset,
        }
    }

    // TODO(andrew): Consider optimizing. Is a quotients of two coset_vanishing (use succinct rep
    // for verifier).
    pub fn gen_column_simd(&self) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        let mut col = Col::<SimdBackend, BaseField>::zeros(1 << self.log_size);
        let size = 1 << self.log_size;
        let step = 1 << self.log_step;
        let step_offset = self.offset % step;
        for i in (step_offset..size).step_by(step) {
            let circle_domain_index = coset_index_to_circle_domain_index(i, self.log_size);
            let circle_domain_index_bit_rev = bit_reverse_index(circle_domain_index, self.log_size);
            col.set(circle_domain_index_bit_rev, BaseField::one());
        }
        CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
    }

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!(
                "preprocessed_is_step_with_offset_{}_{}_{}",
                self.log_size, self.log_step, self.offset
            ),
        }
    }
}
