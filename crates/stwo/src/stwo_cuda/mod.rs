#![allow(unused_imports)]

pub mod base_field_vec;
pub mod bindings;
pub mod bindings_airs;
pub mod blake_2s_hash_vec;
pub mod poseidon252;
pub mod secure_field_vec;

pub use base_field_vec::BaseFieldVec;
pub use blake_2s_hash_vec::Blake2sHashVec;
pub use secure_field_vec::SecureFieldVec;

/// Get CUDA memory info (free and total memory in bytes)
/// Returns (free_bytes, total_bytes)
pub fn get_cuda_memory_info() -> (usize, usize) {
    let mut free_mem: usize = 0;
    let mut total_mem: usize = 0;
    unsafe {
        bindings::cuda_get_memory_info(&mut free_mem, &mut total_mem);
    }
    (free_mem, total_mem)
}

/// Print CUDA memory usage
pub fn print_cuda_memory(label: &str) {
    let (free, total) = get_cuda_memory_info();
    let used = total - free;
    println!(
        "[CUDA MEM] {}: used={:.2} MB, free={:.2} MB, total={:.2} MB",
        label,
        used as f64 / 1024.0 / 1024.0,
        free as f64 / 1024.0 / 1024.0,
        total as f64 / 1024.0 / 1024.0
    );
}
