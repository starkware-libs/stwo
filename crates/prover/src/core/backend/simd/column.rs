use std::mem;

use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use itertools::{izip, Itertools};
use num_traits::Zero;

use super::cm31::PackedCM31;
use super::m31::{PackedBaseField, N_LANES};
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::backend::{CPUBackend, Column};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;

#[derive(Clone, Debug)]
pub struct BaseFieldVec {
    pub data: Vec<PackedBaseField>,
    pub length: usize,
}

impl BaseFieldVec {
    pub fn into_cpu_vec(mut self) -> Vec<BaseField> {
        let capacity = self.data.capacity() * N_LANES;
        let length = self.length;
        let ptr = self.data.as_mut_ptr() as *mut BaseField;
        let res = unsafe { Vec::from_raw_parts(ptr, length, capacity) };
        mem::forget(self);
        res
    }
}

impl AsRef<[BaseField]> for BaseFieldVec {
    fn as_ref(&self) -> &[BaseField] {
        &cast_slice(&self.data)[..self.length]
    }
}

impl AsMut<[BaseField]> for BaseFieldVec {
    fn as_mut(&mut self) -> &mut [BaseField] {
        &mut cast_slice_mut(&mut self.data)[..self.length]
    }
}

impl Column<BaseField> for BaseFieldVec {
    fn zeros(length: usize) -> Self {
        let data = vec![PackedBaseField::zeroed(); length.div_ceil(N_LANES)];
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.as_ref().to_vec()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> BaseField {
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }
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
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![PackedSecureField::zeroed(); length.div_ceil(N_LANES)],
            length,
        }
    }

    fn to_cpu(&self) -> Vec<SecureField> {
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
    /// # Safety
    ///
    /// `vec_index` must be a valid index.
    pub unsafe fn packed_at(&self, vec_index: usize) -> PackedSecureField {
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

    /// # Safety
    ///
    /// `vec_index` must be a valid index.
    pub unsafe fn set_packed(&mut self, vec_index: usize, value: PackedSecureField) {
        let PackedSecureField([PackedCM31([a, b]), PackedCM31([c, d])]) = value;
        *self.columns[0].data.get_unchecked_mut(vec_index) = a;
        *self.columns[1].data.get_unchecked_mut(vec_index) = b;
        *self.columns[2].data.get_unchecked_mut(vec_index) = c;
        *self.columns[3].data.get_unchecked_mut(vec_index) = d;
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

impl FromIterator<SecureField> for SecureColumn<SimdBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let cpu_col = SecureColumn::<CPUBackend>::from_iter(iter);
        let columns = cpu_col.columns.map(|col| col.into_iter().collect());
        SecureColumn { columns }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::BaseFieldVec;
    use crate::core::backend::simd::column::SecureFieldVec;
    use crate::core::backend::Column;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;

    #[test]
    fn base_field_vec_from_iter_works() {
        let values: [BaseField; 30] = array::from_fn(BaseField::from);

        let res = values.into_iter().collect::<BaseFieldVec>();

        assert_eq!(res.to_cpu(), values);
    }

    #[test]
    fn secure_field_vec_from_iter_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values: [SecureField; 30] = rng.gen();

        let res = values.into_iter().collect::<SecureFieldVec>();

        assert_eq!(res.to_cpu(), values);
    }
}
