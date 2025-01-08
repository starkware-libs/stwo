use std::fmt::Debug;
use std::hash::Hash;
use std::simd::Simd;

use num_traits::{One, Zero};

use crate::core::backend::simd::m31::{PackedM31, N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Backend, Col, Column};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};

const SIMD_ENUMERATION_0: PackedM31 = unsafe {
    PackedM31::from_simd_unchecked(Simd::from_array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ]))
};

pub trait PreprocessedColumn: Debug {
    fn name(&self) -> &'static str;
    /// Used for comparing preprocessed columns.
    /// Column IDs must be unique in a given context.
    fn id(&self) -> String;
    fn log_size(&self) -> u32;
}
impl PartialEq for dyn PreprocessedColumn {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for dyn PreprocessedColumn {}
impl Hash for dyn PreprocessedColumn {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
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
}
impl PreprocessedColumn for IsFirst {
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

// TODO(Gali): Remove Enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreprocessedColumnEnum {
    /// A column with `1` at the first position, and `0` elsewhere.
    IsFirst(u32),
    Plonk(usize),
    /// A column with the numbers [0..2^log_size-1].
    Seq(u32),
    XorTable(u32, u32, usize),
}

impl PreprocessedColumnEnum {
    pub const fn name(&self) -> &'static str {
        match self {
            PreprocessedColumnEnum::IsFirst(_) => "preprocessed_is_first",
            PreprocessedColumnEnum::Plonk(_) => "preprocessed_plonk",
            PreprocessedColumnEnum::Seq(_) => "preprocessed_seq",
            PreprocessedColumnEnum::XorTable(..) => "preprocessed_xor_table",
        }
    }

    pub fn log_size(&self) -> u32 {
        match self {
            PreprocessedColumnEnum::IsFirst(log_size) => *log_size,
            PreprocessedColumnEnum::Seq(log_size) => *log_size,
            PreprocessedColumnEnum::XorTable(log_size, ..) => *log_size,
            PreprocessedColumnEnum::Plonk(_) => unimplemented!(),
        }
    }

    /// Returns the values of the column at the given row.
    pub fn packed_at(&self, vec_row: usize) -> PackedM31 {
        match self {
            PreprocessedColumnEnum::IsFirst(log_size) => {
                assert!(vec_row < (1 << log_size) / N_LANES);
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
            PreprocessedColumnEnum::Seq(log_size) => {
                assert!(vec_row < (1 << log_size) / N_LANES);
                PackedM31::broadcast(M31::from(vec_row * N_LANES)) + SIMD_ENUMERATION_0
            }

            _ => unimplemented!(),
        }
    }

    /// Generates a column according to the preprocessed column chosen.
    pub fn gen_preprocessed_column<B: Backend>(
        preprocessed_column: &PreprocessedColumnEnum,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        match preprocessed_column {
            PreprocessedColumnEnum::IsFirst(log_size) => gen_is_first(*log_size),
            PreprocessedColumnEnum::Plonk(_) | PreprocessedColumnEnum::XorTable(..) => {
                unimplemented!("eval_preprocessed_column: Plonk and XorTable are not supported.")
            }
            PreprocessedColumnEnum::Seq(log_size) => gen_seq(*log_size),
        }
    }
}

/// Generates a column with a single one at the first position, and zeros elsewhere.
pub fn gen_is_first<B: Backend>(log_size: u32) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);
    col.set(0, BaseField::one());
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

/// Generates a column with `1` at every `2^log_step` positions, `0` elsewhere, shifted by offset.
// TODO(andrew): Consider optimizing. Is a quotients of two coset_vanishing (use succinct rep for
// verifier).
pub fn gen_is_step_with_offset<B: Backend>(
    log_size: u32,
    log_step: u32,
    offset: usize,
) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let mut col = Col::<B, BaseField>::zeros(1 << log_size);

    let size = 1 << log_size;
    let step = 1 << log_step;
    let step_offset = offset % step;

    for i in (step_offset..size).step_by(step) {
        let circle_domain_index = coset_index_to_circle_domain_index(i, log_size);
        let circle_domain_index_bit_rev = bit_reverse_index(circle_domain_index, log_size);
        col.set(circle_domain_index_bit_rev, BaseField::one());
    }

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

/// Generates a column with sequence of numbers from 0 to 2^log_size - 1.
pub fn gen_seq<B: Backend>(log_size: u32) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    let col = Col::<B, BaseField>::from_iter((0..(1 << log_size)).map(BaseField::from));
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn gen_preprocessed_columns<'a, B: Backend>(
    columns: impl Iterator<Item = &'a PreprocessedColumnEnum>,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    columns
        .map(PreprocessedColumnEnum::gen_preprocessed_column)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::IsFirst;
    use crate::core::backend::simd::m31::N_LANES;
    const LOG_SIZE: u32 = 8;

    // TODO(Gali): Add packed_at tests for xor_table and plonk.
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
