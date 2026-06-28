use crate::stwo_cuda::bindings;
use std::{ffi::c_void, fmt::Debug};
use crate::core::vcs::blake2_hash::Blake2sHash;

#[derive(Debug)]
pub struct Blake2sHashVec {
    pub(crate) device_ptr: *const Blake2sHash,
    pub(crate) size: usize,
}

// SAFETY: CUDA device pointers are safe to send/share between threads.
// CUDA manages GPU memory synchronization internally.
unsafe impl Send for Blake2sHashVec {}
unsafe impl Sync for Blake2sHashVec {}


impl Blake2sHashVec {
    pub fn new(device_ptr: *const Blake2sHash, size: usize) -> Self {
        Self { device_ptr, size }
    }

    pub fn from_vec(host_array: Vec<Blake2sHash>) -> Self {
        let size = host_array.len();
        let device_ptr = unsafe {
            bindings::copy_blake_2s_hash_vec_from_host_to_device(host_array.as_ptr(), size)
        };
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        Self::new(unsafe { bindings::cuda_malloc_blake_2s_hash(size) }, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        Self::new(
            unsafe { bindings::cuda_alloc_zeroes_blake_2s_hash(size) },
            size,
        )
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_blake_2s_hash_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                other.size,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<Blake2sHash> {
        let mut host_data: Vec<Blake2sHash> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_blake_2s_hash_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr(),
                self.size,
            );
        }
        host_data
    }

    pub fn get_data(&self, index: usize) -> Blake2sHash {
        let host_value = Blake2sHash([0u8; 32]);
        unsafe {
            bindings::cuda_get_blake_2s_hash(self.device_ptr, &host_value as *const Blake2sHash, index)
        };
        host_value
    }

    /// Batch get multiple Blake2s hashes by indices
    ///
    /// # Arguments
    /// * `indices` - List of indices to fetch
    ///
    /// # Returns
    /// Vector of hashes in the same order as indices
    pub fn batch_get(&self, indices: &[usize]) -> Vec<Blake2sHash> {
        if indices.is_empty() {
            return Vec::new();
        }

        // Prepare output buffer
        let mut result: Vec<Blake2sHash> = Vec::with_capacity(indices.len());

        // Convert usize to u32 (CUDA uses u32)
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();

        unsafe {
            result.set_len(indices.len());
            bindings::cuda_batch_get_blake_2s_hash(
                self.device_ptr,
                result.as_mut_ptr(),
                indices_u32.as_ptr(),
                indices.len() as u32,
            );
        }

        result
    }

    /// Batch get hashes from multiple layers in a single GPU kernel call.
    /// This replaces N per-layer batch_get calls with one multi-layer call,
    /// reducing GPU→CPU round-trips from N to 1.
    pub fn batch_get_multi_layer(
        layers: &[&Blake2sHashVec],
        pairs: &[bindings::LayerIndexPair],
    ) -> Vec<Blake2sHash> {
        if pairs.is_empty() {
            return Vec::new();
        }

        let layer_ptrs: Vec<*const Blake2sHash> =
            layers.iter().map(|l| l.device_ptr).collect();

        let mut result: Vec<Blake2sHash> = Vec::with_capacity(pairs.len());
        unsafe {
            result.set_len(pairs.len());
            bindings::cuda_multi_layer_batch_get_blake_2s_hash(
                layer_ptrs.as_ptr(),
                result.as_mut_ptr(),
                pairs.as_ptr(),
                pairs.len() as u32,
            );
        }
        result
    }
}

impl Clone for Blake2sHashVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for Blake2sHashVec {
    fn drop(&mut self) {
        unsafe { bindings::cuda_free_memory(self.device_ptr as *const c_void) };
    }
}
