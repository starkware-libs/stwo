pub mod bit_reverse;
mod blake2s;
pub mod blake2s_avx;
pub mod circle;
pub mod cm31;
pub mod fft;
pub mod m31;
pub mod qm31;
pub mod quotients;
pub mod tranpose_utils;

use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
use itertools::{izip, Itertools};
use num_traits::Zero;

use self::bit_reverse::bit_reverse_m31;
use self::cm31::PackedCM31;
pub use self::m31::{PackedBaseField, K_BLOCK_SIZE};
use self::qm31::PackedSecureField;
use super::{Backend, Column, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::utils;

pub const VECS_LOG_SIZE: usize = 4;

#[derive(Copy, Clone, Debug)]
pub struct AVX512Backend;

impl Backend for AVX512Backend {}

// BaseField.
// TODO(spapini): Unite with the M31AVX512 type.

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

impl ColumnOps<BaseField> for AVX512Backend {
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

impl FieldOps<BaseField> for AVX512Backend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        PackedBaseField::batch_inverse(&column.data, &mut dst.data);
    }
}

impl Column<BaseField> for BaseFieldVec {
    fn zeros(len: usize) -> Self {
        Self {
            data: vec![PackedBaseField::zeroed(); len.div_ceil(K_BLOCK_SIZE)],
            length: len,
        }
    }
    fn to_cpu(&self) -> Vec<BaseField> {
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
        self.data[index / K_BLOCK_SIZE].to_array()[index % K_BLOCK_SIZE]
    }
}

fn as_cpu_vec(values: BaseFieldVec) -> Vec<BaseField> {
    let capacity = values.data.capacity() * K_BLOCK_SIZE;
    unsafe {
        let res = Vec::from_raw_parts(
            values.data.as_ptr() as *mut BaseField,
            values.length,
            capacity,
        );
        std::mem::forget(values);
        res
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut res: Vec<_> = (&mut chunks).map(PackedBaseField::from_array).collect();
        let mut length = res.len() * K_BLOCK_SIZE;

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

#[derive(Clone, Debug)]
pub struct SecureFieldVec {
    pub data: Vec<PackedSecureField>,
    length: usize,
}

impl ColumnOps<SecureField> for AVX512Backend {
    type Column = SecureFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        // Fallback to cpu bit_reverse.
        // TODO(AlonH): Implement AVX512 bit_reverse for SecureField.
        utils::bit_reverse(column.to_vec().as_mut_slice());
    }
}

impl FieldOps<SecureField> for AVX512Backend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        PackedSecureField::batch_inverse(&column.data, &mut dst.data);
    }
}

impl Column<SecureField> for SecureFieldVec {
    fn zeros(len: usize) -> Self {
        Self {
            data: vec![PackedSecureField::zeroed(); len.div_ceil(K_BLOCK_SIZE)],
            length: len,
        }
    }
    fn to_vec(&self) -> Vec<SecureField> {
        self.data
            .iter()
            .flat_map(|x| x.to_array())
            .take(self.length)
            .collect()
    }
    fn len(&self) -> usize {
        self.length
    }
    fn at(&self, index: usize) -> SecureField {
        self.data[index / K_BLOCK_SIZE].to_array()[index % K_BLOCK_SIZE]
    }
}

impl FromIterator<SecureField> for SecureFieldVec {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut res: Vec<_> = (&mut chunks).map(PackedSecureField::from_array).collect();
        let mut length = res.len() * K_BLOCK_SIZE;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let pad_len = 16 - remainder.len();
                let last = PackedSecureField::from_array(
                    remainder
                        .chain(std::iter::repeat(SecureField::zero()).take(pad_len))
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

impl FromIterator<PackedSecureField> for SecureFieldVec {
    fn from_iter<I: IntoIterator<Item = PackedSecureField>>(iter: I) -> Self {
        let data = (&mut iter.into_iter()).collect_vec();
        let length = data.len() * K_BLOCK_SIZE;

        Self { data, length }
    }
}

impl SecureColumn<AVX512Backend> {
    pub fn packed_at(&self, vec_index: usize) -> PackedSecureField {
        unsafe {
            PackedSecureField([
                PackedCM31([
                    *self.columns[0].data.get_unchecked(vec_index),
                    *self.columns[1].data.get_unchecked(vec_index),
                ]),
                PackedCM31([
                    *self.columns[2].data.get_unchecked(vec_index),
                    *self.columns[3].data.get_unchecked(vec_index),
                ]),
            ])
        }
    }

    pub fn set_packed(&mut self, vec_index: usize, value: PackedSecureField) {
        unsafe {
            *self.columns[0].data.get_unchecked_mut(vec_index) = value.a().a();
            *self.columns[1].data.get_unchecked_mut(vec_index) = value.a().b();
            *self.columns[2].data.get_unchecked_mut(vec_index) = value.b().a();
            *self.columns[3].data.get_unchecked_mut(vec_index) = value.b().b();
        }
    }

    pub fn to_vec(&self) -> Vec<SecureField> {
        izip!(
            self.columns[0].to_cpu(),
            self.columns[1].to_cpu(),
            self.columns[2].to_cpu(),
            self.columns[3].to_cpu(),
        )
        .map(|(a, b, c, d)| SecureField::from_m31_array([a, b, c, d]))
        .collect()
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::backend::{Col, Column};
    use crate::core::fields::m31::P;

    type B = AVX512Backend;

    #[test]
    fn test_column() {
        for i in 0..100 {
            let col = Col::<B, BaseField>::from_iter((0..i).map(BaseField::from));
            assert_eq!(
                col.to_cpu(),
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
            <B as ColumnOps<BaseField>>::bit_reverse_column(&mut col);
            assert_eq!(
                col.to_cpu(),
                (0..len)
                    .map(|x| BaseField::from(utils::bit_reverse_index(x, i as u32)))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_as_cpu_vec() {
        let original_vec = (1000..1100).map(BaseField::from).collect::<Vec<_>>();
        let col = Col::<B, BaseField>::from_iter(original_vec.clone());
        let vec = as_cpu_vec(col);
        assert_eq!(vec, original_vec);
    }

    #[test]
    fn test_packed_basefield_batch_inverse() {
        let mut rng = StdRng::seed_from_u64(0);
        let column = BaseFieldVec::from_iter(
            (0..64).map(|_| BaseField::from_u32_unchecked(rng.gen::<u32>() % P)),
        );
        let expected = column.data.iter().map(|e| e.inverse()).collect::<Vec<_>>();
        let mut dst = BaseFieldVec::from_iter((0..64).map(|_| BaseField::zero()));

        <AVX512Backend as FieldOps<BaseField>>::batch_inverse(&column, &mut dst);

        dst.data.iter().zip(expected.iter()).for_each(|(a, b)| {
            assert_eq!(a.to_array(), b.to_array());
        });
    }
}
