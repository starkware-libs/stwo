use std::ffi::c_void;

use crate::core::vcs::blake2_hash::{Blake2sHash, reduce_to_m31};
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as Blake2sMerkleHasherLifted;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::backend::{Col, Column, ColumnOps, CpuBackend};
use crate::prover::vcs::ops::MerkleOps;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::stwo_cuda::base_field_vec::BaseFieldVec;
use crate::stwo_cuda::bindings;
use crate::stwo_cuda::blake_2s_hash_vec::Blake2sHashVec;

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = Blake2sHashVec;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Blake2sHashVec>,
        columns: &[&BaseFieldVec],
    ) -> Blake2sHashVec {
        let size = 1 << log_size;
        let number_of_columns = columns.len();

        let result: Blake2sHashVec = Blake2sHashVec::new_uninitialized(size);
        unsafe {
            Self::commit_on_layer_using_gpu(
                size,
                number_of_columns,
                columns,
                prev_layer,
                result.device_ptr,
            );
        }

        result
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for CudaBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Blake2sHashVec>,
        columns: &[&BaseFieldVec],
    ) -> Blake2sHashVec {
        let result = <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size, prev_layer, columns,
        );
        // Apply M31 reduction to each hash on CPU (result is copied back)
        let mut hashes = result.to_vec();
        for hash in hashes.iter_mut() {
            hash.0 = reduce_to_m31(hash.0);
        }
        Blake2sHashVec::from_vec(hashes)
    }
}

impl CudaBackend {
    unsafe fn commit_on_layer_using_gpu(
        size: usize,
        number_of_columns: usize,
        columns: &[&BaseFieldVec],
        prev_layer: Option<&Blake2sHashVec>,
        result_pointer: *const Blake2sHash,
    ) {
        let device_column_pointers_vector: Vec<*const u32> =
            columns.iter().map(|column| column.device_ptr).collect();

        let device_column_pointers: *const *const u32 =
            bindings::copy_device_pointer_vec_from_host_to_device(
                device_column_pointers_vector.as_ptr(),
                number_of_columns,
            );

        if let Some(previous_layer) = prev_layer {
            bindings::commit_on_layer_with_previous(
                size,
                number_of_columns,
                device_column_pointers,
                previous_layer.device_ptr,
                result_pointer as *mut Blake2sHash,
            );
        } else {
            bindings::commit_on_first_layer(
                size,
                number_of_columns,
                device_column_pointers,
                result_pointer as *mut Blake2sHash,
            );
        }
        bindings::cuda_free_memory(device_column_pointers as *const c_void);
    }
}

// PR #1425: the lifted Blake2s Merkle hasher is now the *standard* (non-M31-reduced)
// `Blake2sMerkleHasher = Blake2sHasherGeneric<false>` — the same hasher for BOTH
// `Blake2sMerkleChannel` and `Blake2sM31MerkleChannel` (only the Fiat-Shamir channel differs).
// The former `<const IS_M31_OUTPUT: bool>` generic collapses to the single `false` case, so the
// device kernels below are always called with the non-reduced (`false`) finalize path.
impl MerkleOpsLifted<Blake2sMerkleHasherLifted> for CudaBackend {
    fn build_leaves(
        columns: &[&Col<Self, crate::core::fields::m31::BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Blake2sHash> {
        if columns.is_empty() {
            let cpu_result =
                <CpuBackend as MerkleOpsLifted<Blake2sMerkleHasherLifted>>::build_leaves(
                    &[],
                    lifting_log_size,
                );
            return Blake2sHashVec::from_vec(cpu_result);
        }

        assert!(columns[0].len() >= 2, "A column must be of length >= 2.");

        // Fast path: all columns same log_size AND equal to lifting_log_size
        // → single fused kernel, state stays in registers, no global memory state R/W
        let all_same_size = columns.iter().all(|c| c.len() == columns[0].len());
        let max_log_size = columns.last().map(|c| c.len().ilog2()).unwrap_or(0);
        if all_same_size && max_log_size == lifting_log_size {
            let size = 1u32 << lifting_log_size;

            let result = Blake2sHashVec::new_uninitialized(size as usize);

            let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.device_ptr).collect();
            let device_col_ptrs = unsafe {
                bindings::copy_device_pointer_vec_from_host_to_device(
                    col_ptrs.as_ptr(),
                    col_ptrs.len(),
                )
            };

            unsafe {
                bindings::blake2s_build_leaves_fused(
                    size,
                    columns.len() as u32,
                    device_col_ptrs,
                    result.device_ptr as *mut Blake2sHash,
                    // #1425: lifted hasher is the non-reduced `Blake2sMerkleHasher` (<false>).
                    false,
                );
                bindings::cuda_free_memory(device_col_ptrs as *const c_void);
            }
            return result;
        }

        // GPU path for heterogeneous columns using incremental state management.
        // Uses blake2s_alloc_init_states / blake2s_lift_states / blake2s_update_columns /
        // blake2s_finalize_all CUDA kernels to process column groups of different sizes
        // entirely on GPU without any CPU/SIMD fallback.

        // Sort columns by size (ascending) and group by log_size.
        let mut sorted_cols: Vec<(u32, &BaseFieldVec)> =
            columns.iter().map(|c| (c.len().ilog2(), *c)).collect();
        sorted_cols.sort_by_key(|(log_size, _)| *log_size);

        let mut groups: Vec<(u32, Vec<&BaseFieldVec>)> = Vec::new();
        let mut current_log_size = sorted_cols[0].0;
        let mut current_cols: Vec<&BaseFieldVec> = vec![sorted_cols[0].1];
        for &(log_size, col) in &sorted_cols[1..] {
            if log_size == current_log_size {
                current_cols.push(col);
            } else {
                groups.push((current_log_size, std::mem::take(&mut current_cols)));
                current_log_size = log_size;
                current_cols.push(col);
            }
        }
        groups.push((current_log_size, current_cols));

        unsafe {
            // Initialize Blake2s states (start with 2 states, matching lifted algorithm).
            let mut states: *mut c_void = bindings::blake2s_alloc_init_states(2);
            let mut prev_log_size: u32 = 1;
            let mut current_size: u32 = 2;

            for (log_size, cols) in &groups {
                let log_size = *log_size;
                let new_size = 1u32 << log_size;

                // Lift states to match the current column group's size.
                if log_size > prev_log_size {
                    let log_ratio = log_size - prev_log_size;
                    let mut next_states: *mut c_void = std::ptr::null_mut();
                    bindings::blake2s_lift_states(
                        states,
                        current_size,
                        &mut next_states,
                        new_size,
                        log_ratio,
                    );
                    bindings::cuda_free_memory(states as *const c_void);
                    states = next_states;
                    current_size = new_size;
                }

                // Single multi-column absorb over the group's resident device buffers.
                let col_ptrs: Vec<*const u32> = cols.iter().map(|c| c.device_ptr).collect();
                bindings::blake2s_update_columns(
                    states,
                    current_size,
                    col_ptrs.as_ptr(),
                    col_ptrs.len() as u32,
                );

                prev_log_size = log_size;
            }

            // Final lift to lifting_log_size if columns don't reach it.
            let final_size = 1u32 << lifting_log_size;
            if lifting_log_size > prev_log_size {
                let log_ratio = lifting_log_size - prev_log_size;
                let mut next_states: *mut c_void = std::ptr::null_mut();
                bindings::blake2s_lift_states(
                    states,
                    current_size,
                    &mut next_states,
                    final_size,
                    log_ratio,
                );
                bindings::cuda_free_memory(states as *const c_void);
                states = next_states;
            }

            // Finalize all states into output hashes.
            let result = Blake2sHashVec::new_uninitialized(final_size as usize);
            bindings::blake2s_finalize_all(
                states,
                result.device_ptr as *mut Blake2sHash,
                final_size,
                // #1425: lifted hasher is the non-reduced `Blake2sMerkleHasher` (<false>).
                false,
            );
            bindings::cuda_free_memory(states as *const c_void);

            result
        }
    }

    fn build_next_layer(prev_layer: &Col<Self, Blake2sHash>) -> Col<Self, Blake2sHash> {
        if prev_layer.len() == 0 {
            return Blake2sHashVec::from_vec(vec![]);
        }
        let output_size = prev_layer.len() / 2;
        let result = Blake2sHashVec::new_uninitialized(output_size);
        unsafe {
            bindings::blake2s_lifted_build_next_layer(
                output_size as u32,
                prev_layer.device_ptr,
                result.device_ptr as *mut Blake2sHash,
                // #1425: lifted hasher is the non-reduced `Blake2sMerkleHasher` (<false>).
                false,
            );
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::prover::backend::cuda::CudaBackend;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::vcs::ops::MerkleOps;
    use crate::stwo_cuda::base_field_vec::BaseFieldVec;
    use crate::stwo_cuda::blake_2s_hash_vec::Blake2sHashVec;

    #[test]
    fn test_commit_on_first_layer_with_many_columns_compared_with_cpu() {
        let log_size = 16;
        let size = 1 << log_size;

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(100, size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        assert_eq!(result.to_cpu(), expected_result);
    }

    #[test]
    fn test_commit_on_layer_with_previous_layer_compared_with_cpu() {
        let current_layer_log_size = 10;
        let current_layer_size = 1 << current_layer_log_size;
        let previous_layer_log_size = current_layer_log_size + 1;
        let previous_layer_size = 1 << previous_layer_log_size;

        // First layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(35, previous_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let cpu_previous_layer = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            previous_layer_log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let gpu_previous_layer: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                previous_layer_log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        // Current layer

        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(16, current_layer_size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            current_layer_log_size,
            Some(&cpu_previous_layer),
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                current_layer_log_size,
                Some(&gpu_previous_layer),
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        assert_eq!(result.to_cpu(), expected_result);
    }

    fn gpu_columns_from(columns: &[Vec<BaseField>]) -> Vec<BaseFieldVec> {
        columns.iter().cloned().map(BaseFieldVec::from_vec).collect()
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
    fn test_build_leaves_heterogeneous_vs_cpu() {
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as LiftedBlake2s;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        // Columns with different sizes: 2^4, 2^6, 2^8, 2^10
        let lifting_log_size = 10u32;
        let cpu_cols: Vec<Vec<BaseField>> = vec![
            (0..1 << 4).map(|i| M31::from(i * 3)).collect(),
            (0..1 << 6).map(|i| M31::from(i * 7)).collect(),
            (0..1 << 8).map(|i| M31::from(i * 11)).collect(),
            (0..1 << 10).map(|i| M31::from(i * 13)).collect(),
        ];
        let gpu_cols: Vec<BaseFieldVec> = gpu_columns_from(&cpu_cols);

        let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_cols.iter().collect();
        let expected = <CpuBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &cpu_col_refs,
            lifting_log_size,
        );
        let gpu_col_refs: Vec<&BaseFieldVec> = gpu_cols.iter().collect();
        let result = <CudaBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &gpu_col_refs,
            lifting_log_size,
        );

        assert_eq!(result.to_cpu(), expected);
    }

    #[test]
    fn test_build_leaves_same_size_vs_cpu() {
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as LiftedBlake2s;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        // All columns same size = lifting_log_size (exercises fused kernel path)
        let lifting_log_size = 12u32;
        let size = 1 << lifting_log_size;
        let cpu_cols: Vec<Vec<BaseField>> =
            (0..20).map(|col| (0..size).map(|i| M31::from(i * (col + 1))).collect()).collect();
        let gpu_cols: Vec<BaseFieldVec> = gpu_columns_from(&cpu_cols);

        let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_cols.iter().collect();
        let expected = <CpuBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &cpu_col_refs,
            lifting_log_size,
        );
        let gpu_col_refs: Vec<&BaseFieldVec> = gpu_cols.iter().collect();
        let result = <CudaBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &gpu_col_refs,
            lifting_log_size,
        );

        assert_eq!(result.to_cpu(), expected);
    }

    #[test]
    fn test_build_leaves_with_extra_lifting_vs_cpu() {
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as LiftedBlake2s;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        // Columns smaller than lifting_log_size (exercises final lift step)
        let lifting_log_size = 12u32;
        let cpu_cols: Vec<Vec<BaseField>> = vec![
            (0..1 << 4).map(|i| M31::from(i * 5)).collect(),
            (0..1 << 8).map(|i| M31::from(i * 9)).collect(),
        ];
        let gpu_cols: Vec<BaseFieldVec> = gpu_columns_from(&cpu_cols);

        let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_cols.iter().collect();
        let expected = <CpuBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &cpu_col_refs,
            lifting_log_size,
        );
        let gpu_col_refs: Vec<&BaseFieldVec> = gpu_cols.iter().collect();
        let result = <CudaBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &gpu_col_refs,
            lifting_log_size,
        );

        assert_eq!(result.to_cpu(), expected);
    }

    #[test]
    fn test_commit_on_first_layer_log24() {
        // Test at log_size=24 to check for size-related issues
        let log_size = 24u32;
        let size = 1 << log_size;

        // Use fewer columns to save memory - just 4 columns
        let cpu_columns_vector: Vec<Vec<BaseField>> = columns_test_vector(4, size);
        let gpu_columns_vector: Vec<BaseFieldVec> = gpu_columns_from(&cpu_columns_vector);

        let expected_result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            None,
            &cpu_columns_vector.iter().collect::<Vec<_>>(),
        );
        let result: Blake2sHashVec =
            <CudaBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                log_size,
                None,
                &gpu_columns_vector.iter().collect::<Vec<_>>(),
            );

        // Compare first and last hashes
        let cpu_hashes = expected_result.clone();
        let gpu_hashes = result.to_cpu();

        assert_eq!(gpu_hashes.len(), size);
        assert_eq!(cpu_hashes.len(), size);

        // Check first 100 hashes
        assert_eq!(gpu_hashes[..100], cpu_hashes[..100], "First 100 hashes mismatch");
        // Check last 100 hashes
        assert_eq!(gpu_hashes[size - 100..], cpu_hashes[size - 100..], "Last 100 hashes mismatch");
        // Full equality
        assert_eq!(gpu_hashes, cpu_hashes);
    }
}
