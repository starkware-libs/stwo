pub mod bit_reverse;
pub mod circle;
pub mod fft;
pub mod m31;

use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use num_traits::Zero;

use self::bit_reverse::bit_reverse_m31;
use self::m31::PackedBaseField;
use crate::core::fields::m31::BaseField;
use crate::core::fields::{Column, FieldOps};
use crate::core::utils;

#[derive(Copy, Clone, Debug)]
pub struct AVX512Backend;

// BaseField.
// TODO(spapini): Unite with the M31AVX512 type.
pub const K_ELEMENTS: usize = 16;

unsafe impl Pod for PackedBaseField {}
unsafe impl Zeroable for PackedBaseField {
    fn zeroed() -> Self {
        unsafe { core::mem::zeroed() }
    }
}

#[derive(Clone, Debug)]
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
}

impl Column<BaseField> for BaseFieldVec {
    fn zeros(len: usize) -> Self {
        Self {
            data: vec![PackedBaseField::zeroed(); len.div_ceil(K_ELEMENTS)],
            length: len,
        }
    }
    fn to_vec(&self) -> Vec<BaseField> {
        self.data
            .iter()
            .flat_map(|x| x.to_array())
            .take(self.length)
            .collect()
    }
    fn len(&self) -> usize {
        self.length
    }
    fn at(&self, index: usize) -> BaseField {
        self.data[index / K_ELEMENTS].to_array()[index % K_ELEMENTS]
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

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut res: Vec<_> = (&mut chunks).map(PackedBaseField::from_array).collect();
        let mut length = res.len() * K_ELEMENTS;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let pad_len = 16 - remainder.len();
                let last = PackedBaseField::from_array(
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
            for j in 0..i {
                assert_eq!(col.at(j), BaseField::from(j));
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
