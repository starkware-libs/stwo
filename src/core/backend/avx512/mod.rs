pub mod bit_reverse;

use std::ops::Index;

use bytemuck::checked::cast_slice_mut;
use bytemuck::{cast_slice, Pod, Zeroable};
use num_traits::Zero;

use self::bit_reverse::bit_reverse_m31;
use crate::core::fields::avx512_m31::M31AVX512;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Column, FieldOps};
use crate::core::utils;


#[derive(Copy, Clone, Debug)]
pub struct AVX512Backend;

// BaseField.
#[repr(align(64))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedBaseField([BaseField; 16]);
unsafe impl Pod for PackedBaseField {}
unsafe impl Zeroable for PackedBaseField {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseFieldVec {
    pub data: Vec<PackedBaseField>,
    length: usize,
}
impl FieldOps<BaseField> for AVX512Backend {
    type Column = BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        if column.data.len().ilog2() < bit_reverse::MIN_LOG_SIZE {
            let data: &mut [BaseField] = cast_slice_mut(&mut column.data[..]);
            utils::bit_reverse(&mut data[..column.length]);
            return;
        }
        bit_reverse_m31(&mut column.data);
    }

    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        Self::inverse_unoptimised(&column.data[..], &mut dst.data[..])
    }
}

impl AVX512Backend {
    pub fn inverse_unoptimised(column: &[PackedBaseField], dst: &mut [PackedBaseField]) {
        let n = column.len();
        let column: &[M31AVX512] = cast_slice(column);
        let dst: &mut [M31AVX512] = cast_slice_mut(dst);

        dst[0] = column[0];
        // First pass.
        for i in 1..n {
            dst[i] = dst[i - 1] * column[i];
        }

        // Inverse cumulative product.
        let mut curr_inverse = dst[n - 1];

        // Second pass.
        for i in (1..n).rev() {
            dst[i] = dst[i - 1] * curr_inverse;
            curr_inverse *= column[i];
        }
        dst[0] = curr_inverse;
    }
}

impl Column<BaseField> for BaseFieldVec {
    fn zeros(len: usize) -> Self {
        Self {
            data: vec![PackedBaseField::default(); (len + 15) / 16],
            length: len,
        }
    }
    fn to_vec(&self) -> Vec<BaseField> {
        self.data
            .iter()
            .flat_map(|x| x.0)
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
        &self.data[index / 16].0[index % 16]
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut res: Vec<_> = (&mut chunks).map(PackedBaseField).collect();
        let mut length = res.len() * 16;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let pad_len = 16 - remainder.len();
                let last = PackedBaseField(
                    remainder
                        .chain(std::iter::repeat(BaseField::zero()).take(pad_len))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                );
                res.push(last);
            }
        }

        Self { data: res, length }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::{Col, Column, Field};

    type B = AVX512Backend;

    #[test]
    fn test_column() {
        for i in 0..100 {
            let col = Col::<B, BaseField>::from_iter((0..i).map(BaseField::from));
            assert_eq!(
                col.to_vec(),
                (0..i).map(BaseField::from).collect::<Vec<_>>()
            );
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

    #[test]
    fn test_inverse_unoptimized() {
        let len = 1 << 10;
        let col = Col::<B, BaseField>::from_iter((1..len+1).map(BaseField::from));
        let mut dst = Col::<B, BaseField>::zeros(len);
        B::batch_inverse(&col, &mut dst);
        assert_eq!(dst.to_vec(), (0..len).map(|i|BaseField::from(i).inverse()).collect::<Vec<_>>());
    }
}
