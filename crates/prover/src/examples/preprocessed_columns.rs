use std::simd::Simd;

use num_traits::{One, Zero};

use crate::constraint_framework::preprocessed_columns::PreprocessedColumnTrait;
use crate::core::backend::simd::m31::{PackedM31, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;

/// A column with `1` at the first position, and `0` elsewhere.
#[derive(Debug)]
pub struct IsFirst {
    pub log_size: u32,
}
impl IsFirst {
    pub const fn new(log_size: u32) -> Self {
        Self { log_size }
    }

    pub fn packed_at(&self, vec_row: usize) -> PackedM31 {
        assert!(vec_row < (1 << self.log_size) / N_LANES);
        if vec_row == 0 {
            unsafe {
                PackedM31::from_simd_unchecked(Simd::from_array(std::array::from_fn(|i| {
                    if i == 0 {
                        1
                    } else {
                        0
                    }
                })))
            }
        } else {
            PackedM31::zero()
        }
    }

    pub fn gen_column_simd(&self) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
        let mut col = Col::<SimdBackend, BaseField>::zeros(1 << self.log_size);
        col.set(0, BaseField::one());
        CircleEvaluation::new(CanonicCoset::new(self.log_size).circle_domain(), col)
    }
}
impl PreprocessedColumnTrait for IsFirst {
    fn name(&self) -> &'static str {
        "preprocessed_is_first"
    }

    fn id(&self) -> String {
        format!("IsFirst(log_size: {})", self.log_size)
    }

    fn log_size(&self) -> u32 {
        self.log_size
    }
}
