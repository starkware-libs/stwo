pub mod bit_reverse;
pub mod circle;
pub mod fft;

use std::ops::Index;

use bytemuck::checked::cast_slice_mut;
use bytemuck::{Pod, Zeroable};
use num_traits::Zero;

use self::bit_reverse::bit_reverse_m31;
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

fn as_cpu_vec(values: BaseFieldVec) -> Vec<BaseField> {
    let capacity = values.len() * 16;
    unsafe {
        Vec::from_raw_parts(
            values.data.as_ptr() as *mut BaseField,
            values.length,
            capacity,
        )
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
