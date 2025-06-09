use std::simd::Simd;

use num_traits::{One, Zero};

use crate::core::backend::simd::m31::{PackedM31, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column};
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;

/// Used for comparing preprocessed columns.
/// Column IDs must be unique in a given context.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreProcessedColumnId {
    pub id: String,
}

/// A column with `1` at the first position, and `0` elsewhere.
#[derive(Debug, Clone)]
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

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!("preprocessed_is_first_{}", self.log_size).to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IsFirst;
    use crate::core::backend::simd::m31::N_LANES;
    const LOG_SIZE: u32 = 8;

    #[test]
    fn test_packed_at_is_first() {
        let is_first = IsFirst::new(LOG_SIZE);
        let expected_is_first = is_first.gen_column_simd().to_cpu();

        for i in 0..(1 << LOG_SIZE) / N_LANES {
            assert_eq!(
                is_first.packed_at(i).to_array(),
                expected_is_first[i * N_LANES..(i + 1) * N_LANES]
            );
        }
    }
}
