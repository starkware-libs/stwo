use std::mem;

use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use itertools::{izip, Itertools};
use num_traits::Zero;

use super::cm31::PackedCM31;
use super::m31::{PackedBaseField, N_LANES};
use super::qm31::{PackedQM31, PackedSecureField};
use super::SimdBackend;
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::{SecureColumnByCoords, SECURE_EXTENSION_DEGREE};
use crate::core::fields::{FieldExpOps, FieldOps};

impl FieldOps<BaseField> for SimdBackend {
    fn batch_inverse(column: &BaseColumn, dst: &mut BaseColumn) {
        PackedBaseField::batch_inverse(&column.data, &mut dst.data);
    }
}

impl FieldOps<SecureField> for SimdBackend {
    fn batch_inverse(column: &SecureColumn, dst: &mut SecureColumn) {
        PackedSecureField::batch_inverse(&column.data, &mut dst.data);
    }
}

/// An efficient structure for storing and operating on a arbitrary number of [`BaseField`] values.
#[derive(Clone, Debug)]
pub struct BaseColumn {
    pub data: Vec<PackedBaseField>,
    /// The number of [`BaseField`]s in the vector.
    pub length: usize,
}

impl BaseColumn {
    /// Extracts a slice containing the entire vector of [`BaseField`]s.
    pub fn as_slice(&self) -> &[BaseField] {
        &cast_slice(&self.data)[..self.length]
    }

    /// Extracts a mutable slice containing the entire vector of [`BaseField`]s.
    pub fn as_mut_slice(&mut self) -> &mut [BaseField] {
        &mut cast_slice_mut(&mut self.data)[..self.length]
    }

    pub fn into_cpu_vec(mut self) -> Vec<BaseField> {
        let capacity = self.data.capacity() * N_LANES;
        let length = self.length;
        let ptr = self.data.as_mut_ptr() as *mut BaseField;
        let res = unsafe { Vec::from_raw_parts(ptr, length, capacity) };
        mem::forget(self);
        res
    }

    /// Returns a vector of `BaseColumnMutSlice`s, each mutably owning
    /// `chunk_size` `PackedBasedField`s (i.e, `chuck_size` * `N_LANES` elements).
    pub fn chunks_mut(&mut self, chunk_size: usize) -> Vec<BaseColumnMutSlice<'_>> {
        self.data
            .chunks_mut(chunk_size)
            .map(BaseColumnMutSlice)
            .collect_vec()
    }
}

impl Column<BaseField> for BaseColumn {
    fn zeros(length: usize) -> Self {
        let data = vec![PackedBaseField::zeroed(); length.div_ceil(N_LANES)];
        Self { data, length }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = Vec::with_capacity(length.div_ceil(N_LANES));
        data.set_len(length.div_ceil(N_LANES));
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.as_slice().to_vec()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> BaseField {
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }

    fn set(&mut self, index: usize, value: BaseField) {
        let mut packed = self.data[index / N_LANES].to_array();
        packed[index % N_LANES] = value;
        self.data[index / N_LANES] = PackedBaseField::from_array(packed)
    }
}

impl FromIterator<BaseField> for BaseColumn {
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

// A efficient structure for storing and operating on a arbitrary number of [`SecureField`] values.
#[derive(Clone, Debug)]
pub struct CM31Column {
    pub data: Vec<PackedCM31>,
    pub length: usize,
}

impl Column<CM31> for CM31Column {
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![PackedCM31::zeroed(); length.div_ceil(N_LANES)],
            length,
        }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = Vec::with_capacity(length.div_ceil(N_LANES));
        data.set_len(length.div_ceil(N_LANES));
        Self { data, length }
    }

    fn to_cpu(&self) -> Vec<CM31> {
        self.data
            .iter()
            .flat_map(|x| x.to_array())
            .take(self.length)
            .collect()
    }

    fn len(&self) -> usize {
        self.length
    }

    fn at(&self, index: usize) -> CM31 {
        self.data[index / N_LANES].to_array()[index % N_LANES]
    }

    fn set(&mut self, index: usize, value: CM31) {
        let mut packed = self.data[index / N_LANES].to_array();
        packed[index % N_LANES] = value;
        self.data[index / N_LANES] = PackedCM31::from_array(packed)
    }
}

impl FromIterator<CM31> for CM31Column {
    fn from_iter<I: IntoIterator<Item = CM31>>(iter: I) -> Self {
        let mut chunks = iter.into_iter().array_chunks();
        let mut data = (&mut chunks).map(PackedCM31::from_array).collect_vec();
        let mut length = data.len() * N_LANES;

        if let Some(remainder) = chunks.into_remainder() {
            if !remainder.is_empty() {
                length += remainder.len();
                let mut last = [CM31::zero(); N_LANES];
                last[..remainder.len()].copy_from_slice(remainder.as_slice());
                data.push(PackedCM31::from_array(last));
            }
        }

        Self { data, length }
    }
}

impl FromIterator<PackedCM31> for CM31Column {
    fn from_iter<I: IntoIterator<Item = PackedCM31>>(iter: I) -> Self {
        let data = (&mut iter.into_iter()).collect_vec();
        let length = data.len() * N_LANES;

        Self { data, length }
    }
}

/// A mutable slice of a BaseColumn.
pub struct BaseColumnMutSlice<'a>(pub &'a mut [PackedBaseField]);

impl<'a> BaseColumnMutSlice<'a> {
    pub fn at(&self, index: usize) -> BaseField {
        self.0[index / N_LANES].to_array()[index % N_LANES]
    }

    pub fn set(&mut self, index: usize, value: BaseField) {
        let mut packed = self.0[index / N_LANES].to_array();
        packed[index % N_LANES] = value;
        self.0[index / N_LANES] = PackedBaseField::from_array(packed)
    }
}

/// An efficient structure for storing and operating on a arbitrary number of [`SecureField`]
/// values.
#[derive(Clone, Debug)]
pub struct SecureColumn {
    pub data: Vec<PackedSecureField>,
    /// The number of [`SecureField`]s in the vector.
    pub length: usize,
}

impl Column<SecureField> for SecureColumn {
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![PackedSecureField::zeroed(); length.div_ceil(N_LANES)],
            length,
        }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = Vec::with_capacity(length.div_ceil(N_LANES));
        data.set_len(length.div_ceil(N_LANES));
        Self { data, length }
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

    fn set(&mut self, index: usize, value: SecureField) {
        let mut packed = self.data[index / N_LANES].to_array();
        packed[index % N_LANES] = value;
        self.data[index / N_LANES] = PackedSecureField::from_array(packed)
    }
}

impl FromIterator<SecureField> for SecureColumn {
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

impl FromIterator<PackedSecureField> for SecureColumn {
    fn from_iter<I: IntoIterator<Item = PackedSecureField>>(iter: I) -> Self {
        let data = (&mut iter.into_iter()).collect_vec();
        let length = data.len() * N_LANES;

        Self { data, length }
    }
}

/// A mutable slice of a SecureColumnByCoords.
pub struct SecureColumnByCoordsMutSlice<'a>(pub [BaseColumnMutSlice<'a>; SECURE_EXTENSION_DEGREE]);

impl<'a> SecureColumnByCoordsMutSlice<'a> {
    /// # Safety
    ///
    /// `vec_index` must be a valid index.
    pub unsafe fn packed_at(&self, vec_index: usize) -> PackedSecureField {
        PackedQM31([
            PackedCM31([
                *self.0[0].0.get_unchecked(vec_index),
                *self.0[1].0.get_unchecked(vec_index),
            ]),
            PackedCM31([
                *self.0[2].0.get_unchecked(vec_index),
                *self.0[3].0.get_unchecked(vec_index),
            ]),
        ])
    }

    /// # Safety
    ///
    /// `vec_index` must be a valid index.
    pub unsafe fn set_packed(&mut self, vec_index: usize, value: PackedSecureField) {
        let PackedQM31([PackedCM31([a, b]), PackedCM31([c, d])]) = value;
        *self.0[0].0.get_unchecked_mut(vec_index) = a;
        *self.0[1].0.get_unchecked_mut(vec_index) = b;
        *self.0[2].0.get_unchecked_mut(vec_index) = c;
        *self.0[3].0.get_unchecked_mut(vec_index) = d;
    }
}

impl SecureColumnByCoords<SimdBackend> {
    pub fn packed_len(&self) -> usize {
        self.columns[0].data.len()
    }

    /// # Safety
    ///
    /// `vec_index` must be a valid index.
    pub unsafe fn packed_at(&self, vec_index: usize) -> PackedSecureField {
        PackedQM31([
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
        let PackedQM31([PackedCM31([a, b]), PackedCM31([c, d])]) = value;
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

    /// Returns a vector of `SecureColumnByCoordsMutSlice`s, each mutably owning
    /// `SECURE_EXTENSION_DEGREE` slices of `chunk_size` `PackedBasedField`s
    /// (i.e, `chuck_size` * `N_LANES` secure field elements, by coordinates).
    pub fn chunks_mut(&mut self, chunk_size: usize) -> Vec<SecureColumnByCoordsMutSlice<'_>> {
        let [a, b, c, d] = self
            .columns
            .get_many_mut([0, 1, 2, 3])
            .unwrap()
            .map(|x| x.chunks_mut(chunk_size));
        izip!(a, b, c, d)
            .map(|(a, b, c, d)| SecureColumnByCoordsMutSlice([a, b, c, d]))
            .collect_vec()
    }
}

impl FromIterator<SecureField> for SecureColumnByCoords<SimdBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let cpu_col = SecureColumnByCoords::<CpuBackend>::from_iter(iter);
        let columns = cpu_col.columns.map(|col| col.into_iter().collect());
        SecureColumnByCoords { columns }
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::BaseColumn;
    use crate::core::backend::simd::column::SecureColumn;
    use crate::core::backend::simd::m31::N_LANES;
    use crate::core::backend::simd::qm31::PackedQM31;
    use crate::core::backend::Column;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::secure_column::SecureColumnByCoords;

    #[test]
    fn base_field_vec_from_iter_works() {
        let values: [BaseField; 30] = array::from_fn(BaseField::from);

        let res = values.into_iter().collect::<BaseColumn>();

        assert_eq!(res.to_cpu(), values);
    }

    #[test]
    fn secure_field_vec_from_iter_works() {
        let mut rng = SmallRng::seed_from_u64(0);
        let values: [SecureField; 30] = rng.gen();

        let res = values.into_iter().collect::<SecureColumn>();

        assert_eq!(res.to_cpu(), values);
    }

    #[test]
    fn test_base_column_chunks_mut() {
        let values: [BaseField; N_LANES * 7] = array::from_fn(BaseField::from);
        let mut col = values.into_iter().collect::<BaseColumn>();

        const CHUNK_SIZE: usize = 2;
        let mut chunks = col.chunks_mut(CHUNK_SIZE);
        chunks[2].set(19, BaseField::from(1234));
        chunks[3].set(1, BaseField::from(5678));

        assert_eq!(col.at(2 * CHUNK_SIZE * N_LANES + 19), BaseField::from(1234));
        assert_eq!(col.at(3 * CHUNK_SIZE * N_LANES + 1), BaseField::from(5678));
    }

    #[test]
    fn test_secure_column_by_coords_chunks_mut() {
        const COL_PACKED_SIZE: usize = 16;
        let a: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let b: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let c: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let d: [BaseField; N_LANES * COL_PACKED_SIZE] = array::from_fn(BaseField::from);
        let mut col = SecureColumnByCoords {
            columns: [a, b, c, d].map(|values| values.into_iter().collect::<BaseColumn>()),
        };

        let mut rng = SmallRng::seed_from_u64(0);
        let rand0 = PackedQM31::from_array(rng.gen());
        let rand1 = PackedQM31::from_array(rng.gen());

        const CHUNK_SIZE: usize = 4;
        let mut chunks = col.chunks_mut(CHUNK_SIZE);
        unsafe {
            chunks[2].set_packed(3, rand0);
            chunks[3].set_packed(1, rand1);

            assert_eq!(
                col.packed_at(2 * CHUNK_SIZE + 3).to_array(),
                rand0.to_array()
            );
            assert_eq!(
                col.packed_at(3 * CHUNK_SIZE + 1).to_array(),
                rand1.to_array()
            );
        }
    }
}
