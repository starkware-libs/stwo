pub mod bit_reverse;

use std::ops::Index;

use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use num_traits::Zero;

use self::bit_reverse::bit_reverse_m31;
use crate::core::fields::avx512_m31::M31AVX512;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Column, FieldOps};
use crate::core::utils;

#[derive(Copy, Clone, Debug)]
pub struct AVX512Backend;

// BaseField.
// TODO(spapini): Unite with the M31AVX512 type.
pub const K_ELEMENTS: usize = 16;

#[repr(align(64))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedBaseField([BaseField; K_ELEMENTS]);
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

impl BaseFieldVec {
    pub fn as_slice(&self) -> &[BaseField] {
        let data: &[BaseField] = cast_slice(&self.data[..]);
        &data[..self.length]
    }
    pub fn as_mut_slice(&mut self) -> &mut [BaseField] {
        let data: &mut [BaseField] = cast_slice_mut(&mut self.data[..]);
        &mut data[..self.length]
    }
}

impl FieldOps<BaseField> for AVX512Backend {
    type Column = BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        // Fallback to cpu bit_reverse.
        if column.data.len().ilog2() < bit_reverse::MIN_LOG_SIZE {
            utils::bit_reverse(column.as_mut_slice());
            return;
        }
        bit_reverse_m31(&mut column.data);
    }

    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        const W: usize = 4;
        let n = column.len() / K_ELEMENTS;
        debug_assert!(n.is_power_of_two());
        if n < W {
            Self::inverse_unoptimised(&column.data, &mut dst.data);
            return;
        }

        let column: &[M31AVX512] = cast_slice(&column.data);
        let dst: &mut [M31AVX512] = cast_slice_mut(&mut dst.data);

        // First pass.
        let mut cum_prod: [M31AVX512; W] = column[..W].try_into().unwrap();
        dst[..W].copy_from_slice(&cum_prod);
        for i in W..n {
            cum_prod[i % W] *= column[i];
            dst[i] = cum_prod[i % W];
        }
        debug_assert_eq!(dst.len(), n);

        // Inverse cumulative products.
        // Use classic batch inversion.
        let mut tail_inverses = [M31AVX512::zeroed(); W];
        Self::inverse_unoptimised(
            cast_slice(&dst[n - W..]),
            cast_slice_mut(&mut tail_inverses),
        );

        // Second pass.
        for i in (W..n).rev() {
            dst[i] = dst[i - W] * tail_inverses[i % W];
            tail_inverses[i % W] *= column[i];
        }
        dst[0..W].copy_from_slice(&tail_inverses);
    }
}

impl AVX512Backend {
    // TODO(Ohad): unroll.
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
        let mut curr_inverse = dst[n - 1].inverse();

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
            data: vec![PackedBaseField::default(); len.div_ceil(K_ELEMENTS)],
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
        &self.data[index / K_ELEMENTS].0[index % K_ELEMENTS]
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut res: Vec<_> = (&mut chunks).map(PackedBaseField).collect();
        let mut length = res.len() * K_ELEMENTS;

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

    #[test]
    fn test_inverse_unoptimized() {
        let len = 1 << 10;
        let col = Col::<B, BaseField>::from_iter((1..len + 1).map(BaseField::from));
        let mut dst = Col::<B, BaseField>::zeros(len);
        B::batch_inverse(&col, &mut dst);
        assert_eq!(
            dst.to_vec(),
            (1..len + 1)
                .map(|i| BaseField::from(i).inverse())
                .collect::<Vec<_>>()
        );
    }
}
