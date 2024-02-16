pub mod bit_reverse;

use std::ops::Index;

use bytemuck::checked::cast_slice_mut;
use num_traits::Zero;

use self::bit_reverse::bit_reverse_m31;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Column, FieldOps};
use crate::core::utils;

#[derive(Copy, Clone, Debug)]
pub struct AVX512Backend;

// BaseField.
// TODO(spapini): Unite with the M31AVX512 type.
pub const K_ELEMENTS: usize = 16;
type PackedBaseField = [BaseField; K_ELEMENTS];
#[derive(Clone, Debug)]
pub struct BaseFieldVec {
    data: Vec<PackedBaseField>,
    length: usize,
}
impl FieldOps<BaseField> for AVX512Backend {
    type Column = BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        // Fallback to cpu bit_reverse.
        if column.data.len().ilog2() < bit_reverse::MIN_LOG_SIZE {
            let data: &mut [BaseField] = cast_slice_mut(&mut column.data[..]);
            utils::bit_reverse(&mut data[..column.length]);
            return;
        }
        bit_reverse_m31(&mut column.data);
    }
}

impl Column<BaseField> for BaseFieldVec {
    fn zeros(len: usize) -> Self {
        Self {
            data: vec![PackedBaseField::default(); len.div_ceil(K_ELEMENTS)],
            length: len,
        }
    }
    fn to_vec(&self) -> Vec<BaseField> {
        self.data
            .iter()
            .flatten()
            .copied()
            .take(self.length)
            .collect()
    }
    fn len(&self) -> usize {
        self.length
    }
}

impl Index<usize> for BaseFieldVec {
    type Output = BaseField;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index / K_ELEMENTS][index % K_ELEMENTS]
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut res: Vec<_> = (&mut chunks).collect();
        let mut length = res.len() * K_ELEMENTS;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let pad_len = 16 - remainder.len();
                let last: PackedBaseField = remainder
                    .chain(std::iter::repeat(BaseField::zero()).take(pad_len))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                res.push(last);
            }
        }

        Self { data: res, length }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::{Col, Column};

    type B = AVX512Backend;

    #[test]
    fn test_column() {
        for i in 0..100 {
            let col = Col::<B, BaseField>::from_iter((0..i).map(BaseField::from));
            assert_eq!(
                col.to_vec(),
                (0..i).map(BaseField::from).collect::<Vec<_>>()
            );
            for j in 0..i {
                assert_eq!(col[j], BaseField::from(j));
            }
        }
    }

    #[test]
    fn test_bit_reverse() {
        for i in 1..16 {
            let len = 1 << i;
            let mut col = Col::<B, BaseField>::from_iter((0..len).map(BaseField::from));
            B::bit_reverse_column(&mut col);
            assert_eq!(
                col.to_vec(),
                (0..len)
                    .map(|x| BaseField::from(utils::bit_reverse_index(x, i as u32)))
                    .collect::<Vec<_>>()
            );
        }
    }
}
