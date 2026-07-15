use itertools::izip;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::backend::{Column, ColumnOps};
use crate::stwo_cuda as interface;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::stwo_cuda::bindings;
use crate::stwo_cuda::blake_2s_hash_vec::Blake2sHashVec;
use crate::stwo_cuda::secure_field_vec::SecureFieldVec;

impl ColumnOps<BaseField> for CudaBackend {
    type Column = BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            interface::bindings::bit_reverse_base_field(column.device_ptr, size);
        }
    }
}

// 74951f79 makes `PackLeavesOps` a supertrait of `MerkleOpsLifted`. v1: host-delegate to the
// SIMD reference (correctness-first; the value is unique so this is byte-identical). A device
// packing kernel is a later optimization (pairs with the device FRI pack_leaves work).
impl crate::prover::vcs_lifted::ops::PackLeavesOps for CudaBackend {
    fn pack_leaves_input(
        values: &[&crate::prover::backend::Col<Self, BaseField>;
             crate::core::fields::qm31::SECURE_EXTENSION_DEGREE],
    ) -> [crate::prover::backend::Col<Self, BaseField>;
           crate::core::fields::qm31::SECURE_EXTENSION_DEGREE
               * crate::core::vcs_lifted::verifier::PACKED_LEAF_SIZE] {
        use crate::prover::backend::simd::column::BaseColumn as SimdBaseColumn;
        use crate::prover::backend::simd::SimdBackend;
        use crate::prover::vcs_lifted::ops::PackLeavesOps;
        let simd_cols: [SimdBaseColumn; crate::core::fields::qm31::SECURE_EXTENSION_DEGREE] =
            std::array::from_fn(|i| values[i].to_cpu().into_iter().collect());
        let simd_refs: [&SimdBaseColumn; crate::core::fields::qm31::SECURE_EXTENSION_DEGREE] =
            std::array::from_fn(|i| &simd_cols[i]);
        let packed = <SimdBackend as PackLeavesOps>::pack_leaves_input(&simd_refs);
        packed.map(|c| BaseFieldVec::from_vec(c.to_cpu()))
    }
}

// PolyOps::Twiddles = BaseFieldVec must satisfy `TwiddleBuffer<BitReversedOrder>` (a 74951f79
// addition). v1: host-delegate to the `Vec<T>` reference impl — twiddle buffers are small, so the
// round-trip is negligible and byte-identical. (Device-side subdomain extraction is a later opt.)
impl crate::prover::poly::twiddles::TwiddleBuffer<crate::prover::poly::BitReversedOrder>
    for BaseFieldVec
{
    fn empty() -> Self {
        BaseFieldVec::from_vec(Vec::new())
    }

    fn extract_subdomain_twiddles(&self, domain_log_size: u32, subdomain_log_size: u32) -> Self {
        use crate::prover::poly::twiddles::TwiddleBuffer;
        let host: Vec<BaseField> = self.to_cpu();
        let extracted = <Vec<BaseField> as TwiddleBuffer<
            crate::prover::poly::BitReversedOrder,
        >>::extract_subdomain_twiddles(&host, domain_log_size, subdomain_log_size);
        BaseFieldVec::from_vec(extracted)
    }
}

impl ColumnOps<SecureField> for CudaBackend {
    type Column = SecureFieldVec;
    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            interface::bindings::bit_reverse_secure_field(column.device_ptr, size);
        }
    }
}

impl Column<BaseField> for interface::base_field_vec::BaseFieldVec {
    fn zeros(len: usize) -> Self {
        Self::new_zeroes(len)
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> BaseField {
        // STEP 2 decommit reader-site (GATE_AIR_STREAM_COMMIT): the generic
        // `MerkleProverLifted::decommit` reads each committed column at ~70 sparse query positions
        // via `col.at(idx)`. Under the streamed commit the device buffer was freed and the exact
        // bytes live in the streaming-commit-layer stash (fused_commit), so serve the read from
        // there — byte-identical to the resident device read. This keeps the recovery in the CUDA
        // backend layer (not the shared generic PCS/verifier). For a resident column the stash has
        // no entry and this is the plain device read.
        if crate::prover::backend::cuda::fused_commit::is_staged(self) {
            return crate::prover::backend::cuda::fused_commit::host_batch_get(self, &[index])[0];
        }
        Self::get_data(self, index)
    }

    /// Bulk gather: one device→host copy for all `indices` instead of one per `at`. Returns the
    /// SAME values in the SAME order as `indices.iter().map(|&i| self.at(i))` — the only change vs.
    /// the default trait impl is that the read is batched, so the produced bytes are identical.
    ///
    /// Routes exactly like `at`: a host-staged column (freed device buffer, bytes in the streaming
    /// stash) is served from the stash via `host_batch_get`; a resident column is served by the
    /// device batch-gather kernel. Both preserve `indices` order.
    fn batch_at(&self, indices: &[usize]) -> Vec<BaseField> {
        if crate::prover::backend::cuda::fused_commit::is_staged(self) {
            return crate::prover::backend::cuda::fused_commit::host_batch_get(self, indices);
        }
        self.batch_get(indices)
    }

    fn set(&mut self, _index: usize, _value: BaseField) {
        Self::set_data(self, _index, _value);
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_uint32_t(len),
            size: len,
            owns_memory: true,
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        let mid = self.size / 2;
        let second_len = self.size - mid;
        let first = BaseFieldVec::new_uninitialized(mid);
        let second = BaseFieldVec::new_uninitialized(second_len);
        unsafe {
            // Copy first half
            bindings::copy_uint32_t_vec_from_device_to_device(
                self.device_ptr,
                first.device_ptr,
                mid as u32,
            );
            // Copy second half (offset source pointer by mid elements)
            bindings::copy_uint32_t_vec_from_device_to_device(
                self.device_ptr.add(mid),
                second.device_ptr,
                second_len as u32,
            );
        }
        (first, second)
    }
}

// Device-batched gather helpers (not part of the 74951f79 `Column` trait — kept as inherent
// methods so internal CUDA callers can still use them).
impl BaseFieldVec {
    pub fn batch_at(&self, indices: &[usize]) -> Vec<BaseField> {
        self.batch_get(indices)
    }

    pub fn batch_at_multi(columns: &[&Self], indices: &[usize]) -> Vec<BaseField> {
        Self::batch_gather_multi(columns, indices)
    }
}

impl FromIterator<BaseField> for BaseFieldVec {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        let vec: Vec<BaseField> = iter.into_iter().collect();
        BaseFieldVec::from_vec(vec)
    }
}

impl IntoIterator for BaseFieldVec {
    type Item = BaseField;

    type IntoIter = std::vec::IntoIter<BaseField>;

    fn into_iter(self) -> Self::IntoIter {
        self.to_cpu().into_iter()
    }
}

impl Column<SecureField> for SecureFieldVec {
    fn zeros(_len: usize) -> Self {
        Self::new_zeroes(_len)
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, _index: usize) -> SecureField {
        Self::get_data(self, _index)
    }

    fn set(&mut self, _index: usize, _value: SecureField) {
        todo!()
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_uint32_t(4 * len),
            size: len,
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        let mid = self.size / 2;
        let second_len = self.size - mid;
        let first = SecureFieldVec::new_uninitialized(mid);
        let second = SecureFieldVec::new_uninitialized(second_len);
        unsafe {
            // Each SecureField is 4 u32s, so multiply offsets by 4
            bindings::copy_uint32_t_vec_from_device_to_device(
                self.device_ptr,
                first.device_ptr,
                (4 * mid) as u32,
            );
            bindings::copy_uint32_t_vec_from_device_to_device(
                self.device_ptr.add(4 * mid),
                second.device_ptr,
                (4 * second_len) as u32,
            );
        }
        (first, second)
    }
}

impl FromIterator<SecureField> for SecureFieldVec {
    fn from_iter<T: IntoIterator<Item = SecureField>>(_iter: T) -> Self {
        todo!()
    }
}

// Device-batched gather (not part of the 74951f79 `Column` trait — kept inherent).
impl Blake2sHashVec {
    pub fn batch_at(&self, indices: &[usize]) -> Vec<Blake2sHash> {
        self.batch_get(indices)
    }
}

impl Column<Blake2sHash> for Blake2sHashVec {
    fn zeros(len: usize) -> Self {
        Self::new_zeroes(len)
    }

    fn to_cpu(&self) -> Vec<Blake2sHash> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> Blake2sHash {
        Self::get_data(self, index)
    }

    /// Option B: pinned minimal-latency root read. Same 32 bytes as `at(index)`; the transfer uses a
    /// pinned staging buffer + copy-stream sync instead of the pageable blocking default-stream D2H.
    /// Merkle-tree hash layers are always device-resident (never host-staged), so there is no stash
    /// path here.
    fn at_root_pinned(&self, index: usize) -> Blake2sHash {
        Self::get_data_pinned(self, index)
    }

    /// Bulk gather: one device→host copy for all `indices` (each hash is 32 bytes) instead of one
    /// per `at`. Returns the SAME hashes in the SAME order as
    /// `indices.iter().map(|&i| self.at(i))`; the only change vs. the default trait impl is that the
    /// read is batched. Merkle-tree hash layers are always device-resident (never host-staged), so
    /// there is no stash path here.
    fn batch_at(&self, indices: &[usize]) -> Vec<Blake2sHash> {
        self.batch_get(indices)
    }

    fn set(&mut self, _index: usize, _value: Blake2sHash) {
        todo!()
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self {
            device_ptr: bindings::cuda_malloc_blake_2s_hash(len),
            size: len,
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        let mid = self.size / 2;
        let second_len = self.size - mid;
        let first = Blake2sHashVec::new_uninitialized(mid);
        let second = Blake2sHashVec::new_uninitialized(second_len);
        unsafe {
            bindings::copy_blake_2s_hash_vec_from_device_to_device(
                self.device_ptr,
                first.device_ptr,
                mid,
            );
            bindings::copy_blake_2s_hash_vec_from_device_to_device(
                self.device_ptr.add(mid),
                second.device_ptr,
                second_len,
            );
        }
        (first, second)
    }
}

impl FromIterator<Blake2sHash> for Blake2sHashVec {
    fn from_iter<T: IntoIterator<Item = Blake2sHash>>(_iter: T) -> Self {
        todo!()
    }
}
use crate::prover::secure_column::SecureColumnByCoords;
impl SecureColumnByCoords<CudaBackend> {
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

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::prover::backend::cuda::CudaBackend;
    use crate::prover::backend::{Column, ColumnOps, CpuBackend};
    use crate::stwo_cuda::base_field_vec::BaseFieldVec;
    use crate::stwo_cuda::secure_field_vec::SecureFieldVec;

    #[test]
    fn test_bit_reverse_base_field() {
        let size: usize = 1 << 10;
        let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
        let mut expected_result = column_data.clone();
        CpuBackend::bit_reverse_column(&mut expected_result);

        let mut column = BaseFieldVec::from_vec(column_data);
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut column);

        assert_eq!(column.to_cpu(), expected_result);
    }

    #[test]
    fn test_bit_reverse_secure_field() {
        let size: usize = 1 << 16;

        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();
        let from_cpu = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect::<Vec<_>>();
        let mut array_expected = from_cpu.clone();

        CpuBackend::bit_reverse_column(&mut array_expected);

        let mut array = SecureFieldVec::from_vec(from_cpu.clone());
        <CudaBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut array);

        assert_eq!(array.to_cpu(), array_expected);
    }
}
