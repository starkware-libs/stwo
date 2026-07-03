use std::ffi::c_void;
use std::fmt::Debug;

use starknet_ff::FieldElement as FieldElement252;

use crate::prover::backend::Column;
use crate::stwo_cuda::bindings;

#[derive(Debug)]
pub struct Poseidon252HashVec {
    pub(crate) device_ptr: *const [u8; 32], // Device pointer to array of 32-byte hashes
    pub(crate) size: usize,
}

// SAFETY: CUDA device pointers are safe to send/share between threads.
// CUDA manages GPU memory synchronization internally.
unsafe impl Send for Poseidon252HashVec {}
unsafe impl Sync for Poseidon252HashVec {}

impl Poseidon252HashVec {
    pub fn new(device_ptr: *const [u8; 32], size: usize) -> Self {
        Self { device_ptr, size }
    }

    pub(crate) fn device_ptr(&self) -> *const [u8; 32] {
        self.device_ptr
    }

    pub fn from_vec(host_array: Vec<FieldElement252>) -> Self {
        let size = host_array.len();

        // Convert to array of [u8; 32]
        let host_bytes: Vec<[u8; 32]> = host_array.iter().map(|hash| hash.to_bytes_be()).collect();

        // Use the proper CUDA copy function
        let device_ptr = unsafe {
            bindings::copy_poseidon252_hash_vec_from_host_to_device(host_bytes.as_ptr(), size)
        };

        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        let device_ptr = unsafe { bindings::cuda_malloc_poseidon252_hash(size) };

        if device_ptr.is_null() {
            panic!("RUST ERROR: cuda_malloc_poseidon252_hash returned null pointer!");
        }

        Self::new(device_ptr, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        let device_ptr = unsafe { bindings::cuda_alloc_zeroes_poseidon252_hash(size) };
        Self::new(device_ptr, size)
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_poseidon252_hash_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr as *mut [u8; 32],
                other.size,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<FieldElement252> {
        // Allocate array of [u8; 32]
        let mut host_bytes = vec![[0u8; 32]; self.size];

        unsafe {
            bindings::copy_poseidon252_hash_vec_from_device_to_host(
                self.device_ptr,
                host_bytes.as_mut_ptr(),
                self.size,
            );
        }

        // Convert to FieldElement252
        let host_array: Vec<FieldElement252> = host_bytes
            .into_iter()
            .map(|bytes| FieldElement252::from_bytes_be(&bytes).unwrap())
            .collect();

        host_array
    }

    pub fn to_cpu(&self) -> Vec<FieldElement252> {
        self.to_vec()
    }

    pub fn get_data(&self, index: usize) -> FieldElement252 {
        let mut host_bytes = [0u8; 32];
        unsafe { bindings::cuda_get_poseidon252_hash(self.device_ptr, &mut host_bytes, index) };
        FieldElement252::from_bytes_be(&host_bytes).unwrap()
    }
}

impl Clone for Poseidon252HashVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for Poseidon252HashVec {
    fn drop(&mut self) {
        unsafe { bindings::cuda_free_memory(self.device_ptr as *const c_void) };
    }
}

impl Column<FieldElement252> for Poseidon252HashVec {
    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> FieldElement252 {
        self.get_data(index)
    }

    fn to_cpu(&self) -> Vec<FieldElement252> {
        self.to_vec()
    }

    fn zeros(len: usize) -> Self {
        Self::new_zeroes(len)
    }

    unsafe fn uninitialized(len: usize) -> Self {
        Self::new_uninitialized(len)
    }

    fn set(&mut self, index: usize, value: FieldElement252) {
        unsafe {
            let value_bytes = value.to_bytes_be();
            // Use proper CUDA memory copy for device memory
            bindings::cuda_set_poseidon252_hash(
                self.device_ptr as *mut [u8; 32],
                index,
                &value_bytes as *const [u8; 32],
            );
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        let mid = self.size / 2;
        let second_len = self.size - mid;
        let first = Poseidon252HashVec::new_uninitialized(mid);
        let second = Poseidon252HashVec::new_uninitialized(second_len);
        unsafe {
            bindings::copy_poseidon252_hash_vec_from_device_to_device(
                self.device_ptr,
                first.device_ptr as *mut [u8; 32],
                mid,
            );
            bindings::copy_poseidon252_hash_vec_from_device_to_device(
                self.device_ptr.add(mid),
                second.device_ptr as *mut [u8; 32],
                second_len,
            );
        }
        (first, second)
    }
}

impl FromIterator<FieldElement252> for Poseidon252HashVec {
    fn from_iter<T: IntoIterator<Item = FieldElement252>>(iter: T) -> Self {
        let data: Vec<FieldElement252> = iter.into_iter().collect();
        Self::from_vec(data)
    }
}
