use std::mem::{forget, transmute};

use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use itertools::{izip, Itertools};
use num_traits::Zero;

use super::cm31::PackedCM31;
use super::m31::{PackedBaseField, N_LANES};
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::backend::Column;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;

// impl Backend for SimdBackend {}
#[derive(Clone, Debug)]
pub struct BaseFieldVec {
    pub data: Vec<PackedBaseField>,
    pub length: usize,
}

impl BaseFieldVec {
    pub fn as_slice(&self) -> &[BaseField] {
        &cast_slice(&self.data)[..self.length]
    }
    pub fn as_mut_slice(&mut self) -> &mut [BaseField] {
        &mut cast_slice_mut(&mut self.data)[..self.length]
    }
}

impl Column<BaseField> for BaseFieldVec {
    fn zeros(length: usize) -> Self {
        let data = vec![PackedBaseField::zeroed(); length.div_ceil(N_LANES)];
        Self { data, length }
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
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }
}

// TODO: This is unsafe because casting unreduced PackedBaseField to BaseField.
fn _as_cpu_vec(mut v: BaseFieldVec) -> Vec<BaseField> {
    let capacity = v.data.capacity() * N_LANES;
    let res = unsafe { Vec::from_raw_parts(transmute(v.data.as_mut_ptr()), v.length, capacity) };
    forget(v);
    res
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut data = (&mut chunks).map(PackedBaseField::from_array).collect_vec();
        let mut length = data.len() * N_LANES;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let mut last = [BaseField::zero(); N_LANES];
                last[..remainder.len()].copy_from_slice(remainder.as_slice());
                data.push(PackedBaseField::from_array(last));
            }
        }

        Self { data, length }
    }
}

#[derive(Clone, Debug)]
pub struct SecureFieldVec {
    pub data: Vec<PackedSecureField>,
    pub length: usize,
}

impl Column<SecureField> for SecureFieldVec {
    fn zeros(len: usize) -> Self {
        Self {
            data: vec![PackedSecureField::zeroed(); len.div_ceil(N_LANES)],
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
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }
}

impl FromIterator<SecureField> for SecureFieldVec {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut data = (&mut chunks)
            .map(PackedSecureField::from_array)
            .collect_vec();
        let mut length = data.len() * N_LANES;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let mut last = [SecureField::zero(); N_LANES];
                last[..remainder.len()].copy_from_slice(remainder.as_slice());
                data.push(PackedSecureField::from_array(last));
            }
        }

        Self { data, length }
    }
}

impl FromIterator<PackedSecureField> for SecureFieldVec {
    fn from_iter<I: IntoIterator<Item = PackedSecureField>>(iter: I) -> Self {
        let data = (&mut iter.into_iter()).collect_vec();
        let length = data.len() * N_LANES;

        Self { data, length }
    }
}

impl SecureColumn<SimdBackend> {
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
            self.columns[0].to_vec(),
            self.columns[1].to_vec(),
            self.columns[2].to_vec(),
            self.columns[3].to_vec(),
        )
        .map(|(a, b, c, d)| SecureField::from_m31_array([a, b, c, d]))
        .collect()
    }
}
