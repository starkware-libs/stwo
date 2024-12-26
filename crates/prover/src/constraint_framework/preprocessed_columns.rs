use num_traits::{One, Zero};

use crate::core::backend::simd::m31::{PackedM31, N_LANES};
use crate::core::backend::{Backend, Col, Column};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};

// TODO(ilya): Where should this enum be placed?
// TODO(Gali): Consider making it a trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreprocessedColumn {
    XorTable(u32, u32, usize),
    IsFirst(u32),
    Seq(u32),
    Plonk(usize),
}

impl PreprocessedColumn {
    pub const fn name(&self) -> &'static str {
        match self {
            PreprocessedColumn::XorTable(..) => "preprocessed.xor_table",
            PreprocessedColumn::IsFirst(_) => "preprocessed.is_first",
            PreprocessedColumn::Plonk(_) => "preprocessed.plonk",
            PreprocessedColumn::Seq(_) => "preprocessed.seq",
        }
    }

    /// Returns the values of the column at the given row.
    pub fn packed_at(&self, vec_row: usize) -> PackedM31 {
        match self {
            PreprocessedColumn::Seq(log_size) => {
                assert!(vec_row <= (1 << log_size) / N_LANES);
                let row: [M31; N_LANES] = (0..N_LANES)
                    .map(|i| M31::from((vec_row * N_LANES + i) as u32))
                    .collect::<Vec<M31>>()
                    .try_into()
                    .unwrap();
                PackedM31::from_array(row)
            }
            PreprocessedColumn::IsFirst(log_size) => {
                assert!(vec_row <= (1 << log_size) / N_LANES);
                if vec_row == 0 {
                    let mut res = [M31::zero(); N_LANES];
                    res[0] = M31::one();
                    PackedM31::from_array(res)
                } else {
                    PackedM31::zero()
                }
            }
            _ => unimplemented!(),
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

// TODO(Gali): Move inside the impl of PreprocessedColumn.
pub fn gen_preprocessed_column<B: Backend>(
    preprocessed_column: &PreprocessedColumn,
) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
    match preprocessed_column {
        PreprocessedColumn::IsFirst(log_size) => gen_is_first(*log_size),
        PreprocessedColumn::Plonk(_) | PreprocessedColumn::XorTable(..) => {
            unimplemented!("eval_preprocessed_column: Plonk and XorTable are not supported.")
        }
        PreprocessedColumn::Seq(log_size) => gen_seq(*log_size),
    }
}

pub fn gen_preprocessed_columns<'a, B: Backend>(
    columns: impl Iterator<Item = &'a PreprocessedColumn>,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    columns.map(gen_preprocessed_column).collect()
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};

    use crate::core::backend::simd::m31::N_LANES;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::Column;
    use crate::core::fields::m31::BaseField;

    #[test]
    fn test_gen_seq() {
        let log_size = 8;
        let seq = super::gen_seq::<SimdBackend>(log_size);

        for i in 0..(1 << log_size) {
            assert_eq!(seq.at(i), BaseField::from_u32_unchecked(i as u32));
        }
    }

    #[test]
    fn test_packed_at_seq() {
        let log_size = 8;
        let seq = super::PreprocessedColumn::Seq(8);

        for i in 0..(1 << log_size) / N_LANES {
            let packed = seq.packed_at(i);
            for j in 0..N_LANES {
                assert_eq!(
                    packed.to_array()[j],
                    BaseField::from_u32_unchecked((i * N_LANES + j) as u32)
                );
            }
        }
    }

    #[test]
    fn test_packed_at_is_first() {
        let log_size = 8;
        let is_first = super::PreprocessedColumn::IsFirst(log_size);
        let packed_0 = is_first.packed_at(0);

        let expected_not_first = [BaseField::zero(); N_LANES];

        assert_eq!(packed_0.to_array()[0], BaseField::one());
        for i in 1..N_LANES {
            assert_eq!(packed_0.to_array()[i], BaseField::zero());
        }
        for i in 1..log_size {
            assert_eq!(
                is_first.packed_at(i as usize).to_array(),
                expected_not_first
            );
        }
    }
}
