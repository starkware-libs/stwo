use itertools::izip;

use super::WgpuBackend;
use crate::core::backend::simd::cm31::PackedCM31;
use crate::core::backend::simd::column::{BaseColumn, SecureColumnByCoordsMutSlice};
use crate::core::backend::simd::qm31::{PackedQM31, PackedSecureField};
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;

impl SecureColumnByCoords<WgpuBackend> {
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
    /// `SECURE_EXTENSION_DEGREE` slices of `chunk_size` `PackedBaseField`s
    /// (i.e, `chuck_size` * `N_LANES` secure field elements, by coordinates).
    pub fn chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl ExactSizeIterator<Item = SecureColumnByCoordsMutSlice<'_>> {
        let [a, b, c, d] = self
            .columns
            .get_many_mut([0, 1, 2, 3])
            .unwrap()
            .map(|x| x.chunks_mut(chunk_size));
        izip!(a, b, c, d).map(|(a, b, c, d)| SecureColumnByCoordsMutSlice([a, b, c, d]))
    }

    #[cfg(feature = "parallel")]
    pub fn par_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl IndexedParallelIterator<Item = SecureColumnByCoordsMutSlice<'_>> {
        let [a, b, c, d] = self.columns.each_mut().map(|c| c.chunks_mut(chunk_size));
        (a, b, c, d)
            .into_par_iter()
            .map(|(a, b, c, d)| SecureColumnByCoordsMutSlice([a, b, c, d]))
    }

    pub fn from_cpu(cpu: SecureColumnByCoords<CpuBackend>) -> Self {
        Self {
            columns: cpu.columns.map(BaseColumn::from_cpu),
        }
    }
}
