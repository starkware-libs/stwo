use std::ffi::c_void;

use crate::core::vcs::blake2_hash::{reduce_to_m31, Blake2sHash};
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
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

impl<const IS_M31_OUTPUT: bool> MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>
    for CudaBackend
{
    fn build_leaves(
        columns: &[&Col<Self, crate::core::fields::m31::BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Blake2sHash> {
        if columns.is_empty() {
            let cpu_result = <CpuBackend as MerkleOpsLifted<
                Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>,
            >>::build_leaves(&[], lifting_log_size);
            return Blake2sHashVec::from_vec(cpu_result);
        }

        assert!(columns[0].len() >= 2, "A column must be of length >= 2.");

        // Fast path: all columns same log_size AND equal to lifting_log_size
        // → single fused kernel, state stays in registers, no global memory state R/W
        let all_same_size = columns.iter().all(|c| c.len() == columns[0].len());
        let max_log_size = columns.last().map(|c| c.len().ilog2()).unwrap_or(0);
        if all_same_size && max_log_size == lifting_log_size {
            let size = 1u32 << lifting_log_size;

            // Fail loud if the same-size fast path is ever handed HOST-STAGED columns: their device
            // buffers are freed, so the resident fused-kernel hash below (`blake2s_build_leaves_fused`)
            // or the GATE_AIR_STREAM_MERKLE loop would dereference freed pointers. The streamed
            // (staged) main tree is MIXED-SIZE and is handled by the heterogeneous path (which
            // rehydrates staged columns from the host stash before absorbing), so it never reaches
            // here; an all-staged same-size set arriving here is a bug, never a silent
            // use-after-free. Guarded to IS_M31_OUTPUT == true (the streamed base main tree uses the
            // M31-output reduction, Blake2sM31MerkleHasher). `all()` on an empty slice is true, but
            // we only reach here with columns non-empty.
            if IS_M31_OUTPUT {
                let columns_all_staged = columns
                    .iter()
                    .all(|c| crate::prover::backend::cuda::fused_commit::is_staged(c));
                assert!(
                    !columns_all_staged,
                    "all columns are host-staged in the same-size fast path — refusing to hash \
                     freed device buffers (staged trees are mixed-size and must use the \
                     heterogeneous rehydrate path)"
                );
            }

            // P-S0 (streaming-Merkle redesign): optionally build the leaf hash by streaming one
            // eval column at a time into a persistent per-row Blake2s state array, instead of the
            // fused all-at-once kernel. This is the prerequisite for the later residency lever
            // (P-S1+ frees each column right after absorbing it) but P-S0 itself does NOT free or
            // change residency — it only proves the streamed-hash route produces the identical
            // root. Gated behind GATE_AIR_STREAM_MERKLE so it's A/B-comparable.
            //
            // BYTE-IDENTITY: the fused kernel computes leaf[i] = Blake2s(col0[i]‖col1[i]‖…) by
            // looping `data[col][i]` in the ORDER OF THE INPUT `columns` slice (which `commit`
            // has already `sorted_by_key(|c| c.len())`). blake2s_update is an incremental absorb,
            // so H(c0‖c1‖…) == update(update(init,c0),c1)…  The streamed loop below absorbs the
            // columns ONE AT A TIME into the same per-row state, IN THE SAME `columns` ORDER, with
            // the SAME blake2s_update_columns / blake2s_finalize_all kernels (identical LE-byte
            // extraction + identical IS_M31_OUTPUT reduction as the fused kernel). By
            // absorb-associativity the running state after col0..col_{n-1} is bit-identical to the
            // fused result. THE COLUMN ORDER IS LOAD-BEARING: a reorder silently changes every
            // leaf — preserve the input `columns` order exactly.
            if std::env::var("GATE_AIR_STREAM_MERKLE").is_ok() {
                let result = Blake2sHashVec::new_uninitialized(size as usize);
                unsafe {
                    // One Blake2s state per leaf row, all already at lifting size (same-size
                    // path) → no lift step is needed.
                    let states: *mut c_void = bindings::blake2s_alloc_init_states(size);
                    // Absorb each column individually, preserving the input column order.
                    for c in columns.iter() {
                        let col_ptr: [*const u32; 1] = [c.device_ptr];
                        bindings::blake2s_update_columns(states, size, col_ptr.as_ptr(), 1);
                    }
                    bindings::blake2s_finalize_all(
                        states,
                        result.device_ptr as *mut Blake2sHash,
                        size,
                        IS_M31_OUTPUT,
                    );
                    bindings::cuda_free_memory(states as *const c_void);
                }
                return result;
            }

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
                    IS_M31_OUTPUT,
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

        // Carry the column references (not just device pointers) so a group whose columns were
        // host-staged by the streamed commit (`fused_commit::is_staged`) can be rehydrated before
        // absorb — the staged columns' `device_ptr`s are freed, so reading them here would be a
        // use-after-free. Resident columns are absorbed by pointer exactly as before.
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

                // INVARIANT: the tree1 root/leaves/nodes are byte-identical to the flag-off
                // resident build. When a group has host-staged columns, only the byte SOURCE of
                // those columns moves (host stash -> a temp device buffer); the values, column
                // order, state-lift orchestration, absorb order, and M31 finalize are all
                // unchanged. Absorbing the group's columns ONE AT A TIME in the SAME order is
                // bit-identical to the single multi-column call by absorb-associativity (the same
                // property the streamed fast path relies on above).
                let any_staged = cols
                    .iter()
                    .any(|c| crate::prover::backend::cuda::fused_commit::is_staged(c));
                if any_staged {
                    // Absorb the group ONE COLUMN AT A TIME, in the SAME group order. A staged
                    // column is rehydrated into a fresh device buffer (its committed host bytes)
                    // just before its absorb, then freed immediately after. The free is a
                    // stream-ordered `cudaFreeAsync` deferred behind the absorb kernel, while the
                    // NEXT rehydrate's `cudaMallocFromPoolAsync` is immediate — so without an
                    // explicit reclaim the pool would reserve a fresh ~256 MiB segment per staged
                    // column (188 × ~256 MiB ≈ 48 GB → OOM at 2^25). `cuda_stream_reclaim_freed(0)`
                    // drains the deferred free + trims the pool before the next alloc, mirroring the
                    // `dehydrate_column` stopgap. Resident columns are absorbed directly by pointer.
                    let t1 = crate::prover::backend::cuda::fused_commit::t1_timers_on();
                    for col in cols {
                        if crate::prover::backend::cuda::fused_commit::is_staged(col) {
                            // T1: build_leaves rehydrate H2D (whole staged column back to device).
                            let h2d_start = t1.then(std::time::Instant::now);
                            let hydrated =
                                crate::prover::backend::cuda::fused_commit::rehydrate_owned(col);
                            if let Some(s) = h2d_start {
                                crate::prover::prove_ex_sync();
                                crate::prover::backend::cuda::fused_commit::t1_add(
                                    3,
                                    s.elapsed().as_nanos(),
                                );
                            }
                            // T1: absorb (single incremental blake2s update).
                            let absorb_start = t1.then(std::time::Instant::now);
                            let one: [*const u32; 1] = [hydrated.device_ptr];
                            bindings::blake2s_update_columns(states, current_size, one.as_ptr(), 1);
                            if let Some(s) = absorb_start {
                                crate::prover::prove_ex_sync();
                                crate::prover::backend::cuda::fused_commit::t1_add(
                                    4,
                                    s.elapsed().as_nanos(),
                                );
                            }
                            drop(hydrated); // free before rehydrating the next staged column
                            // PART A reclaim (no per-column OS trim by default; sync bounds memory).
                            crate::prover::backend::cuda::fused_commit::reclaim_after_free();
                        } else {
                            let one: [*const u32; 1] = [col.device_ptr];
                            bindings::blake2s_update_columns(states, current_size, one.as_ptr(), 1);
                        }
                    }
                } else {
                    // No staged column in this group (empty stash / flag-off / SIMD / other
                    // callers): keep the existing single multi-column absorb, unchanged.
                    let col_ptrs: Vec<*const u32> = cols.iter().map(|c| c.device_ptr).collect();
                    bindings::blake2s_update_columns(
                        states,
                        current_size,
                        col_ptrs.as_ptr(),
                        col_ptrs.len() as u32,
                    );
                }

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
                IS_M31_OUTPUT,
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
                IS_M31_OUTPUT,
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
        let cpu_cols: Vec<Vec<BaseField>> = (0..20)
            .map(|col| (0..size).map(|i| M31::from(i * (col + 1))).collect())
            .collect();
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
    fn test_build_leaves_streamed_same_size_vs_cpu() {
        // P-S0: the GATE_AIR_STREAM_MERKLE per-column-streamed leaf hash must produce a
        // bit-identical leaf layer to the CPU reference (and hence to the fused kernel) for the
        // same-size path (the gate_air main tree). Validates absorb-associativity + identical
        // input column order. NOTE: env var is process-global; this test owns the toggle.
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as LiftedBlake2s;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        let lifting_log_size = 12u32;
        let size = 1 << lifting_log_size;
        let cpu_cols: Vec<Vec<BaseField>> = (0..20)
            .map(|col| (0..size).map(|i| M31::from(i * (col + 1))).collect())
            .collect();
        let gpu_cols: Vec<BaseFieldVec> = gpu_columns_from(&cpu_cols);

        let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_cols.iter().collect();
        let expected = <CpuBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &cpu_col_refs,
            lifting_log_size,
        );
        let gpu_col_refs: Vec<&BaseFieldVec> = gpu_cols.iter().collect();

        // SAFETY: tests in this module that depend on the toggle are not run concurrently with
        // it set; set, run, restore.
        std::env::set_var("GATE_AIR_STREAM_MERKLE", "1");
        let result = <CudaBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &gpu_col_refs,
            lifting_log_size,
        );
        std::env::remove_var("GATE_AIR_STREAM_MERKLE");

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
        assert_eq!(
            gpu_hashes[..100],
            cpu_hashes[..100],
            "First 100 hashes mismatch"
        );
        // Check last 100 hashes
        assert_eq!(
            gpu_hashes[size - 100..],
            cpu_hashes[size - 100..],
            "Last 100 hashes mismatch"
        );
        // Full equality
        assert_eq!(gpu_hashes, cpu_hashes);
    }

    #[test]
    fn test_build_leaves_heterogeneous_staged_vs_resident_vs_cpu() {
        // PART 2 byte-identity net: a MIXED-SIZE lifted-Merkle build must produce a bit-identical
        // leaf layer whether the large columns are RESIDENT or host-STAGED (their device buffers
        // freed, bytes moved to the fused_commit stash). The staged build forces the Part-2
        // `is_staged` rehydrate branch in `build_leaves`; the invariant is that only the byte
        // SOURCE moves (host stash -> temp device buffer) — values, column order, state-lift
        // orchestration, absorb order and M31 finalize are all unchanged, so the root/leaves are
        // byte-identical to the resident build AND to the CpuBackend reference.
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher as LiftedBlake2s;
        use crate::prover::backend::cuda::fused_commit;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        // A few large columns at 2^10 (dehydrated below) + a few small ones at 2^7 (stay resident).
        let lifting_log_size = 10u32;
        let cpu_cols: Vec<Vec<BaseField>> = vec![
            (0..1 << 10).map(|i| M31::from(i * 3)).collect(),
            (0..1 << 10).map(|i| M31::from(i * 5)).collect(),
            (0..1 << 10).map(|i| M31::from(i * 7)).collect(),
            (0..1 << 7).map(|i| M31::from(i * 11)).collect(),
            (0..1 << 7).map(|i| M31::from(i * 13)).collect(),
        ];

        // CPU reference.
        let cpu_col_refs: Vec<&Vec<BaseField>> = cpu_cols.iter().collect();
        let expected = <CpuBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &cpu_col_refs,
            lifting_log_size,
        );

        // Fully-resident CUDA build.
        let resident_cols: Vec<BaseFieldVec> = gpu_columns_from(&cpu_cols);
        let resident_refs: Vec<&BaseFieldVec> = resident_cols.iter().collect();
        let resident = <CudaBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &resident_refs,
            lifting_log_size,
        );

        // Staged CUDA build: dehydrate the large (2^10) columns so they hit the rehydrate branch.
        // Fresh stash so no prior test's freed pointer can alias a live column key.
        fused_commit::clear_stash();
        let mut staged_cols: Vec<BaseFieldVec> = gpu_columns_from(&cpu_cols);
        for col in staged_cols.iter_mut() {
            if col.len() == (1usize << 10) {
                fused_commit::dehydrate_column(col);
                assert!(fused_commit::is_staged(col), "large column must be staged");
            }
        }
        let staged_refs: Vec<&BaseFieldVec> = staged_cols.iter().collect();
        let staged = <CudaBackend as MerkleOpsLifted<LiftedBlake2s>>::build_leaves(
            &staged_refs,
            lifting_log_size,
        );
        fused_commit::clear_stash();

        // Byte-identity: staged == resident == CPU reference.
        assert_eq!(staged.to_cpu(), resident.to_cpu());
        assert_eq!(resident.to_cpu(), expected);
        assert_eq!(staged.to_cpu(), expected);
    }
}
