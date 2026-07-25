use std::ffi::c_void;

use super::bindings;
use crate::core::fields::m31::BaseField;

#[derive(Debug)]
pub struct BaseFieldVec {
    pub device_ptr: *const u32,
    pub size: usize,
}
unsafe impl Send for BaseFieldVec {}
unsafe impl Sync for BaseFieldVec {}

impl BaseFieldVec {
    pub const fn new(device_ptr: *const u32, size: usize) -> Self {
        Self { device_ptr, size }
    }

    pub fn from_vec(host_array: Vec<BaseField>) -> Self {
        let device_ptr = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                host_array.len() as u32,
            )
        };
        let size = host_array.len();
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        let device_ptr = unsafe { bindings::cuda_malloc_uint32_t(size) };
        Self::new(device_ptr, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        let device_ptr = unsafe { bindings::cuda_alloc_zeroes_uint32_t(size) };
        Self::new(device_ptr, size)
    }

    pub fn get_data(&self, index: usize) -> BaseField {
        let value = unsafe { bindings::cuda_get_uint32_t(self.device_ptr as *const c_void, index) };
        BaseField::from_u32_unchecked(value)
    }

    // `result` is filled in full by the following D2H gather cudaMemcpy before any read.
    #[allow(clippy::uninit_vec)]
    pub fn batch_get(&self, indices: &[usize]) -> Vec<BaseField> {
        if indices.is_empty() {
            return Vec::new();
        }
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let mut result: Vec<BaseField> = Vec::with_capacity(indices.len());
        unsafe {
            result.set_len(indices.len());
            bindings::cuda_batch_get_uint32_t(
                self.device_ptr,
                result.as_mut_ptr() as *mut u32,
                indices_u32.as_ptr(),
                indices.len() as u32,
            );
        }
        result
    }

    pub fn set_data(&mut self, index: usize, value: BaseField) {
        unsafe {
            bindings::cuda_set_uint32_t(self.device_ptr as *const c_void, index, value.0);
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                other.size as u32,
            );
        }
    }

    pub fn copy_from_offset(&mut self, other: &Self, offset: usize) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device_offset(
                other.device_ptr,
                self.device_ptr,
                other.size as u32,
                offset as u32,
            );
        }
    }

    // `host_data` is filled in full by the following D2H cudaMemcpy before any read.
    #[allow(clippy::uninit_vec)]
    pub fn to_vec(&self) -> Vec<BaseField> {
        let mut host_data: Vec<BaseField> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr() as *const u32,
                self.size as u32,
            );
        }
        host_data
    }

    /// Gather the same `indices` from multiple columns in a single GPU kernel launch.
    /// Returns a flat Vec in row-major order: for each index, values from all columns.
    /// i.e. result[idx * n_cols + col] = columns[col][indices[idx]]
    // `result` is filled in full by the following D2H gather cudaMemcpy before any read.
    #[allow(clippy::uninit_vec)]
    pub fn batch_gather_multi(columns: &[&BaseFieldVec], indices: &[usize]) -> Vec<BaseField> {
        if indices.is_empty() || columns.is_empty() {
            return Vec::new();
        }
        let n_cols = columns.len();
        let n_indices = indices.len();
        let total = n_cols * n_indices;
        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.device_ptr).collect();
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let mut result: Vec<BaseField> = Vec::with_capacity(total);
        unsafe {
            result.set_len(total);
            bindings::cuda_batch_gather_multi_uint32(
                col_ptrs.as_ptr(),
                n_cols as u32,
                indices_u32.as_ptr(),
                n_indices as u32,
                result.as_mut_ptr() as *mut u32,
            );
        }
        result
    }

    /// Pad the GPU array in-place by cycling the first `cycle_len` elements.
    /// Reallocates to `padded_size`, copies existing `actual_size` elements via D2D,
    /// then fills [actual_size, padded_size) with data[idx % cycle_len] on GPU.
    pub fn pad_with_cycle(&mut self, actual_size: usize, padded_size: usize, cycle_len: usize) {
        if padded_size <= actual_size {
            return;
        }
        // Realloc: allocate new buffer, D2D copy, replace self
        let new_vec = Self::new_uninitialized(padded_size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                self.device_ptr,
                new_vec.device_ptr,
                actual_size as u32,
            );
        }
        let old = std::mem::replace(self, new_vec);
        drop(old);
        // Fill padding region on GPU
        unsafe {
            bindings::pad_with_cycle(
                self.device_ptr,
                actual_size as u32,
                padded_size as u32,
                cycle_len as u32,
            );
        }
        self.size = padded_size;
    }
}

impl Clone for BaseFieldVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for BaseFieldVec {
    fn drop(&mut self) {
        unsafe {
            bindings::cuda_free_memory(self.device_ptr as *const c_void);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_constructor() {
        let size = 1 << 25;
        let host_data = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let base_field_vec = BaseFieldVec::from_vec(host_data.clone());
        assert_eq!(base_field_vec.to_vec(), host_data);
        assert_eq!(base_field_vec.size, host_data.len());
    }

    #[test]
    fn test_zeroes() {
        let size = 64;
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        for a in new_zeroes.to_vec().iter() {
            assert_eq!(a, &BaseField::from(0));
        }
    }
}
