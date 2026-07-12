use std::ffi::c_void;
use std::time::{Duration, Instant};

// use crate::stwo_cuda::mem_pool; // DEPRECATED: Using CUDA allocation directly
use super::bindings;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::qm31;
use crate::prover::backend::simd::very_packed_m31::VeryPackedSecureField;

#[derive(Debug)]
pub struct SecureFieldVec {
    pub device_ptr: *const u32,
    pub(crate) size: usize,
}

unsafe impl Send for SecureFieldVec {}
unsafe impl Sync for SecureFieldVec {}

impl SecureFieldVec {
    pub const fn new(device_ptr: *const u32, size: usize) -> Self {
        Self { device_ptr, size }
    }
    pub fn from_vec(host_array: Vec<SecureField>) -> Self {
        let _data_size_bytes = (host_array.len() * 16) as u64;

        // let start_time = Instant::now();
        let device_ptr = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                4 * host_array.len() as u32,
            )
        };
        // let elapsed_time = Instant::now().duration_since(start_time);

        // let transfer_speed_gbps = (data_size_bytes as f64 / 1_000_000_000.0) /
        // elapsed_time.as_secs_f64();

        let size = host_array.len();
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        let device_ptr = unsafe { bindings::cuda_malloc_uint32_t(4 * size) };
        Self::new(device_ptr, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        let device_ptr = unsafe { bindings::cuda_alloc_zeroes_uint32_t(4 * size) };
        Self::new(device_ptr, size)
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                4 * other.size as u32,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<SecureField> {
        let mut host_data: Vec<SecureField> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr() as *const u32,
                4 * self.size as u32,
            );
        }
        host_data
    }

    pub fn get_data(&self, index: usize) -> SecureField {
        let cuda_val =
            unsafe { bindings::cuda_get_secure_field(self.device_ptr as *const c_void, index) };
        SecureField::from(cuda_val)
    }
}

impl Clone for SecureFieldVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for SecureFieldVec {
    fn drop(&mut self) {
        unsafe {
            bindings::cuda_free_memory(self.device_ptr as *const c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::qm31::SecureField;

    #[test]
    fn test_constructor() {
        let size = 1 << 5;
        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();
        let host_data = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect::<Vec<_>>();
        let secure_field_vec = SecureFieldVec::from_vec(host_data.clone());

        assert_eq!(secure_field_vec.to_vec(), host_data);
        assert_eq!(secure_field_vec.size, host_data.len());
    }
}
