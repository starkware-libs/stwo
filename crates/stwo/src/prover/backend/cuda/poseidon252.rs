use std::ffi::c_void;

use starknet_ff::FieldElement as FieldElement252;

use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher as Poseidon252MerkleHasherOld;
use crate::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher as Poseidon252MerkleHasherLifted;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::prover::vcs::ops::MerkleOps;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::stwo_cuda::bindings;
use crate::stwo_cuda::poseidon252::Poseidon252HashVec;

impl ColumnOps<FieldElement252> for CudaBackend {
    type Column = Poseidon252HashVec;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Poseidon252MerkleHasherOld> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Poseidon252HashVec>,
        columns: &[&BaseFieldVec],
    ) -> Poseidon252HashVec {
        let size = 1 << log_size;
        let number_of_columns = columns.len();

        let result: Poseidon252HashVec = Poseidon252HashVec::new_uninitialized(size);
        unsafe {
            Self::poseidon252_commit_on_layer_using_gpu(
                size,
                number_of_columns,
                columns,
                prev_layer,
                result.device_ptr(),
            );
        }

        result
    }
}

impl CudaBackend {
    unsafe fn poseidon252_commit_on_layer_using_gpu(
        size: usize,
        number_of_columns: usize,
        columns: &[&BaseFieldVec],
        prev_layer: Option<&Poseidon252HashVec>,
        result_pointer: *const [u8; 32],
    ) {
        let device_column_pointers_vector: Vec<*const u32> =
            columns.iter().map(|column| column.device_ptr).collect();
        let device_column_pointers: *const *const u32 =
            bindings::copy_device_pointer_vec_from_host_to_device(
                device_column_pointers_vector.as_ptr(),
                number_of_columns,
            );

        if let Some(previous_layer) = prev_layer {
            bindings::poseidon252_commit_on_layer_with_previous(
                size,
                number_of_columns,
                device_column_pointers,
                previous_layer.device_ptr(),
                result_pointer as *mut [u8; 32],
            );
        } else {
            bindings::poseidon252_commit_on_first_layer(
                size,
                number_of_columns,
                device_column_pointers,
                result_pointer as *mut [u8; 32],
            );
        }
        bindings::cuda_free_memory(device_column_pointers as *const c_void);
    }
}

impl MerkleOpsLifted<Poseidon252MerkleHasherLifted> for CudaBackend {
    fn build_leaves(
        columns: &[&Col<Self, crate::core::fields::m31::BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, FieldElement252> {
        if columns.is_empty() {
            let cpu_result =
                <CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasherLifted>>::build_leaves(
                    &[],
                    lifting_log_size,
                );
            return Poseidon252HashVec::from_vec(cpu_result);
        }

        assert!(columns[0].len() >= 2, "A column must be of length >= 2.");

        // Fast path: all columns same log_size AND equal to lifting_log_size
        // → single fused kernel, state stays in registers, no global memory state R/W
        let all_same_size = columns.iter().all(|c| c.len() == columns[0].len());
        let max_log_size = columns.last().map(|c| c.len().ilog2()).unwrap_or(0);
        if all_same_size && max_log_size == lifting_log_size {
            let size = 1usize << lifting_log_size;
            let result = Poseidon252HashVec::new_uninitialized(size);

            let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.device_ptr).collect();
            let device_col_ptrs = unsafe {
                bindings::copy_device_pointer_vec_from_host_to_device(
                    col_ptrs.as_ptr(),
                    col_ptrs.len(),
                )
            };

            unsafe {
                bindings::poseidon252_build_leaves_fused(
                    size,
                    columns.len(),
                    device_col_ptrs,
                    result.device_ptr() as *mut [u8; 32],
                );
                bindings::cuda_free_memory(device_col_ptrs as *const c_void);
            }
            return result;
        }

        // Slow path: columns of different sizes → fall back to CPU
        let cpu_cols: Vec<Vec<crate::core::fields::m31::BaseField>> = columns
            .iter()
            .map(|c| if c.len() == 0 { vec![] } else { c.to_cpu() })
            .collect();
        let cpu_col_refs: Vec<&Vec<crate::core::fields::m31::BaseField>> =
            cpu_cols.iter().collect();
        let cpu_result =
            <CpuBackend as MerkleOpsLifted<Poseidon252MerkleHasherLifted>>::build_leaves(
                &cpu_col_refs,
                lifting_log_size,
            );
        Poseidon252HashVec::from_vec(cpu_result)
    }

    fn build_next_layer(prev_layer: &Col<Self, FieldElement252>) -> Col<Self, FieldElement252> {
        if prev_layer.len() == 0 {
            return Poseidon252HashVec::from_vec(vec![]);
        }
        let output_size = prev_layer.len() / 2;
        let result = Poseidon252HashVec::new_uninitialized(output_size);
        unsafe {
            bindings::poseidon252_lifted_build_next_layer(
                output_size as u32,
                prev_layer.device_ptr(),
                result.device_ptr() as *mut [u8; 32],
            );
        }
        result
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher as Poseidon252MerkleHasherOld;
    use crate::prover::backend::cuda::CudaBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::vcs::ops::MerkleOps;
    use crate::stwo_cuda::base_field_vec::BaseFieldVec;
    use crate::stwo_cuda::poseidon252::Poseidon252HashVec;

    #[test]
    fn test_commit_on_first_layer_with_many_columns_compared_with_cpu() {
        let log_size = 8;
        let size = 1 << log_size;

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(16, size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result =
            <CpuBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                log_size,
                None,
                &cpu_columns_vector.iter().collect::<Vec<_>>(),
            );
        let result: Poseidon252HashVec =
            <CudaBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        assert_eq!(result.to_cpu(), expected_result);
    }

    #[test]
    fn test_commit_on_layer_with_previous_layer_compared_with_cpu() {
        let current_layer_log_size = 7;
        let current_layer_size = 1 << current_layer_log_size;
        let previous_layer_log_size = current_layer_log_size + 1;
        let previous_layer_size = 1 << previous_layer_log_size;

        // First layer
        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(12, previous_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let cpu_previous_layer =
            <CpuBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                previous_layer_log_size,
                None,
                &cpu_columns_vector.iter().collect::<Vec<_>>(),
            );
        let gpu_previous_layer: Poseidon252HashVec =
            <CudaBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                previous_layer_log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        // Current layer
        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(10, current_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result =
            <CpuBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                current_layer_log_size,
                Some(&cpu_previous_layer),
                &cpu_columns_vector.iter().collect::<Vec<_>>(),
            );
        let result: Poseidon252HashVec =
            <CudaBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                current_layer_log_size,
                Some(&gpu_previous_layer),
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        assert_eq!(result.to_cpu(), expected_result);
    }

    fn gpu_columns_from(columns: &Vec<Vec<BaseField>>) -> Vec<BaseFieldVec> {
        columns
            .clone()
            .into_iter()
            .map(|vector| BaseFieldVec::from_vec(vector))
            .collect()
    }

    fn columns_test_vector(
        number_of_columns: usize,
        size_of_columns: usize,
    ) -> Vec<Vec<BaseField>> {
        (0..number_of_columns)
            .map(|index_of_column| {
                (0..size_of_columns)
                    .map(|index_in_column| M31::from(index_in_column * index_of_column))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_commit_on_layer_with_previous_no_columns() {
        // Test the case where prev_layer exists but columns is empty.
        // This triggers poseidon_hash(left, right) on CPU.
        let current_layer_log_size = 7;
        let previous_layer_log_size = current_layer_log_size + 1;
        let previous_layer_size = 1 << previous_layer_log_size;

        // Build a previous layer from some columns
        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(12, previous_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let cpu_previous_layer =
            <CpuBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                previous_layer_log_size,
                None,
                &cpu_columns_vector.iter().collect::<Vec<_>>(),
            );
        let gpu_previous_layer: Poseidon252HashVec =
            <CudaBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                previous_layer_log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        // Now build a layer WITH previous but WITHOUT columns
        let empty_cpu_columns: Vec<Vec<BaseField>> = vec![];
        let empty_gpu_columns: Vec<BaseFieldVec> = vec![];

        let expected_result =
            <CpuBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                current_layer_log_size,
                Some(&cpu_previous_layer),
                &empty_cpu_columns.iter().collect::<Vec<_>>(),
            );
        let result: Poseidon252HashVec =
            <CudaBackend as MerkleOps<Poseidon252MerkleHasherOld>>::commit_on_layer(
                current_layer_log_size,
                Some(&gpu_previous_layer),
                &empty_gpu_columns.iter().collect::<Vec<_>>(),
            );

        assert_eq!(result.to_cpu(), expected_result);
    }
}
