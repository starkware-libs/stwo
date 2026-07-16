use std::ffi::c_void;

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::vcs::blake2_hash::Blake2sHash;
// use crate::stwo_cuda::mem_pool; // DEPRECATED: No longer needed

#[repr(C)]
pub struct CudaSecureField {
    a: BaseField,
    b: BaseField,
    c: BaseField,
    d: BaseField,
}

impl CudaSecureField {
    pub fn zero() -> Self {
        Self {
            a: BaseField::from(0),
            b: BaseField::from(0),
            c: BaseField::from(0),
            d: BaseField::from(0),
        }
    }
}

impl From<SecureField> for CudaSecureField {
    fn from(value: SecureField) -> Self {
        Self {
            a: value.0 .0,
            b: value.0 .1,
            c: value.1 .0,
            d: value.1 .1,
        }
    }
}

impl From<CudaSecureField> for SecureField {
    fn from(value: CudaSecureField) -> Self {
        SecureField::from_m31(value.a, value.b, value.c, value.d)
    }
}

// This is needed since `CirclePoint<BaseField>` is not FFI safe.
#[repr(C)]
pub struct CirclePointBaseField {
    x: BaseField,
    y: BaseField,
}

#[repr(C)]
pub struct LayerIndexPair {
    pub layer_idx: u32,
    pub hash_idx: u32,
}

impl From<CirclePoint<BaseField>> for CirclePointBaseField {
    fn from(value: CirclePoint<BaseField>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[repr(C)]
pub struct CirclePointSecureField {
    x: CudaSecureField,
    y: CudaSecureField,
}

impl From<CirclePoint<SecureField>> for CirclePointSecureField {
    fn from(value: CirclePoint<SecureField>) -> Self {
        Self {
            x: CudaSecureField::from(value.x),
            y: CudaSecureField::from(value.y),
        }
    }
}

#[link(name = "stwo_cuda")]
extern "C" {
    pub fn copy_uint32_t_vec_from_device_to_host(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );

    pub fn copy_uint32_t_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32;

    pub fn copy_uint32_t_vec_from_device_to_device(
        from: *const u32,
        dst: *const u32,
        size: u32,
    ) -> *const u32;

    pub fn copy_uint32_t_vec_from_device_to_device_offset(
        from: *const u32,
        dst: *const u32,
        size: u32,
        offset: u32,
    );

    pub fn cuda_malloc_uint32_t(size: usize) -> *const u32;

    pub fn cuda_set_uint32_t(device_ptr: *const c_void, index: usize, val: u32);

    pub fn cuda_get_uint32_t(device_ptr: *const c_void, index: usize) -> u32;

    pub fn cuda_increase_at(device_ptr: *const c_void, addr: u32);

    pub fn histogram_by_binary_search(
        input_values: *const u32,
        n_inputs: u32,
        sorted_keys: *const u32,
        n_keys: u32,
        mults: *const u32,
    );

    pub fn cuda_get_secure_field(device_ptr: *const c_void, index: usize) -> CudaSecureField;

    pub fn cuda_malloc_blake_2s_hash(size: usize) -> *const Blake2sHash;

    pub fn cuda_alloc_zeroes_uint32_t(size: usize) -> *const u32;

    pub fn cuda_alloc_zeroes_blake_2s_hash(size: usize) -> *const Blake2sHash;

    pub fn cuda_free_memory(device_ptr: *const c_void);

    // One-shot pool defrag: sync + cudaMemPoolTrimTo(0), releasing ALL cached already-freed segments
    // to the OS. Used by the resident multi-shard path (GATE_AIR_POOL_TRIM) at a shard boundary so
    // the next shard starts from a clean pool. Live buffers untouched. See utils.cu.
    pub fn cuda_pool_trim();

    // MULTI-GPU ("option A"): bind the calling host thread to CUDA device `ordinal` (per-thread
    // runtime current-device). Returns 0 on success, -1 on error. See utils.cu.
    pub fn cuda_set_device(ordinal: i32) -> i32;

    // MULTI-GPU: number of visible CUDA devices (0 on error). See utils.cu.
    pub fn cuda_device_count() -> i32;

    pub fn cuda_get_memory_info(free_mem: *mut usize, total_mem: *mut usize);

    // MEM PROBE (diagnostic, read-only): prints driver free/total + pool reserved/used at a labeled
    // boundary. Does not allocate/free/trim. `tag` is a NUL-terminated C string. See utils.cu.
    pub fn cuda_mem_probe(tag: *const core::ffi::c_char);

    pub fn bit_reverse_base_field(array: *const u32, size: usize);

    pub fn bit_reverse_secure_field(array: *const u32, size: usize);

    pub fn batch_inverse_base_field(from: *const u32, dst: *const u32, size: usize);

    pub fn batch_inverse_secure_field(from: *const u32, dst: *const u32, size: usize);

    pub fn barycentric_weights_cuda(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        domain_size: i32,
        log_size: i32,
        vn_p: CudaSecureField,
        p_x: CudaSecureField,
        p_y: CudaSecureField,
        exp_val: u32,
        result: *const u32,
    );

    pub fn barycentric_eval_at_point_cuda(
        evals: *const u32,
        weights: *const u32,
        size: i32,
        result: *mut CudaSecureField,
    );

    // Option A (batched OODS): one call launches all `n` independent dot-product kernels (no interior
    // sync), does a single terminal device sync + single bulk D2H, then the same per-column CPU
    // reduction the single wrapper does. `evals`/`weights` are arrays of `n` device pointers, `sizes`
    // an array of `n` domain sizes; `results` (len `n`) receives one value per eval. Byte-identical
    // to calling `barycentric_eval_at_point_cuda` `n` times; only the schedule changes. The caller
    // MUST keep each `evals[e]` buffer alive until this returns (the terminal sync guarantees all
    // kernels have consumed their inputs by then).
    pub fn barycentric_eval_at_point_batched_cuda(
        evals: *const *const u32,
        weights: *const *const u32,
        sizes: *const i32,
        n: i32,
        results: *mut CudaSecureField,
    );

    pub fn sort_values_and_permute_with_bit_reverse_order(
        from: *const u32,
        size: usize,
    ) -> *const u32;

    pub fn precompute_twiddles(
        initial: CirclePointBaseField,
        step: CirclePointBaseField,
        total_size: usize,
    ) -> *const u32;

    pub fn evaluate_columns(
        eval_domain_sizes: *const u32,
        values: *const *const u32,
        twiddles_tree: *const u32,
        twiddle_tree_size: u32,
        number_of_columns: u32,
        column_sizes: *const u32,
    );

    pub fn eval_at_point(
        coeffs: *const u32,
        coeffs_size: u32,
        point_x: CudaSecureField,
        point_y: CudaSecureField,
    ) -> CudaSecureField;

    pub fn batch_eval_at_points(
        coeffs_ptrs: *const *const u32,
        coeffs_size: i32,
        num_polys: i32,
        point_x: CudaSecureField,
        point_y: CudaSecureField,
        results: *mut CudaSecureField,
    );

    pub fn fold_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    pub fn fold_circle_into_line(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    pub fn accumulate(size: u32, left_columns: *const *const u32, right_columns: *const *const u32);

    pub fn lift_and_accumulate(
        col_size: u32,
        col_0: *const u32,
        col_1: *const u32,
        col_2: *const u32,
        col_3: *const u32,
        curr_0: *const u32,
        curr_1: *const u32,
        curr_2: *const u32,
        curr_3: *const u32,
        log_ratio: u32,
    );

    pub fn blake2s_lifted_build_next_layer(
        size: u32,
        prev_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
        is_m31_output: bool,
    );

    pub fn blake2s_alloc_init_states(count: u32) -> *mut c_void;

    pub fn blake2s_lift_states(
        prev_states: *mut c_void,
        prev_size: u32,
        next_states_out: *mut *mut c_void,
        next_size: u32,
        log_ratio: u32,
    );

    pub fn blake2s_update_columns(
        states: *mut c_void,
        size: u32,
        column_ptrs_host: *const *const u32,
        num_columns: u32,
    );

    pub fn blake2s_finalize_all(
        states: *mut c_void,
        output: *mut Blake2sHash,
        size: u32,
        is_m31_output: bool,
    );

    pub fn blake2s_build_leaves_fused(
        size: u32,
        number_of_columns: u32,
        device_columns: *const *const u32,
        result: *mut Blake2sHash,
        is_m31_output: bool,
    );

    pub fn commit_on_first_layer(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut Blake2sHash,
    );

    pub fn commit_on_layer_with_previous(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
    );

    pub fn copy_blake_2s_hash_vec_from_host_to_device(
        from: *const Blake2sHash,
        size: usize,
    ) -> *mut Blake2sHash;

    pub fn copy_blake_2s_hash_vec_from_device_to_host(
        from: *const Blake2sHash,
        to: *const Blake2sHash,
        size: usize,
    );

    pub fn copy_blake_2s_hash_vec_from_device_to_device(
        from: *const Blake2sHash,
        dst: *const Blake2sHash,
        size: usize,
    );

    pub fn cuda_get_blake_2s_hash(
        device_ptr: *const Blake2sHash,
        host_ptr: *const Blake2sHash,
        index: usize,
    );

    // Option B: pinned minimal-latency single-root read. Byte-identical value to
    // `cuda_get_blake_2s_hash`; copies via a pinned staging buffer on a dedicated copy stream and
    // syncs only that stream, instead of the pageable blocking default-stream read. See utils.cu.
    pub fn cuda_get_blake_2s_hash_pinned(
        device_ptr: *const Blake2sHash,
        host_ptr: *const Blake2sHash,
        index: usize,
    );

    pub fn cuda_batch_get_blake_2s_hash(
        device_ptr: *const Blake2sHash,
        host_ptr: *mut Blake2sHash,
        indices: *const u32,
        n_indices: u32,
    );

    pub fn cuda_batch_get_uint32_t(
        device_ptr: *const u32,
        host_ptr: *mut u32,
        indices: *const u32,
        n_indices: u32,
    );

    pub fn cuda_batch_gather_multi_uint32(
        column_device_ptrs: *const *const u32,
        n_columns: u32,
        host_indices: *const u32,
        n_indices: u32,
        host_output: *mut u32,
    );

    pub fn cuda_multi_layer_batch_get_blake_2s_hash(
        layer_device_ptrs: *const *const Blake2sHash,
        host_ptr: *mut Blake2sHash,
        pairs: *const LayerIndexPair,
        n_pairs: u32,
    );

    pub fn copy_device_pointer_vec_from_host_to_device(
        from: *const *const u32,
        size: usize,
    ) -> *const *const u32;

    /// M31 modular add offset in-place: data[i] = (data[i] + offset) mod P for all i < n.
    pub fn m31_vector_add_offset(data: *const u32, n: u32, offset: u32);

    /// Pad GPU array by cycling: data[idx] = data[idx % cycle_len] for idx in [actual_size,
    /// padded_size).
    pub fn pad_with_cycle(data: *const u32, actual_size: u32, padded_size: u32, cycle_len: u32);

    /// Fill GPU array with zeros: data[idx] = 0 for idx in [start, end).
    pub fn fill_zero_from(data: *const u32, start: u32, end: u32);

    /// Vector add in-place: dst[i] += src[i] for i in [0, n).
    pub fn vector_add_u32(dst: *const u32, src: *const u32, n: u32);

    /// GPU scatter-add: mults[indices[i] - offset] += 1 for each i.
    pub fn scatter_add(mults: *const u32, device_indices: *const u32, n_indices: u32, offset: u32);

    pub fn accumulate_numerators_batch(
        size: u32,
        columns: *const *const u32,
        line_coeffs_b: *const CudaSecureField,
        line_coeffs_c: *const CudaSecureField,
        column_indices: *const u32,
        num_coeffs: u32,
        result_0: *const u32,
        result_1: *const u32,
        result_2: *const u32,
        result_3: *const u32,
    );

    pub fn compute_quotients_and_combine(
        max_size: u32,
        max_log_size: u32,
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        num_accumulations: u32,
        acc_partial_columns: *const *const u32,
        acc_log_sizes: *const i32,
        first_linear_term_accs: *const CudaSecureField,
        sample_points: *const CirclePointSecureField,
        result_0: *const u32,
        result_1: *const u32,
        result_2: *const u32,
        result_3: *const u32,
    );

    pub fn accumulate_quotients(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        domain_size: u32,
        columns: *const *const u32,
        number_of_columns: usize,
        random_coeff: CudaSecureField,
        sample_points: *const u32,
        sample_columns_indexes: *const u32,
        sample_columns_indexes_size: u32,
        sample_column_values: *const CudaSecureField,
        sample_column_and_values_sizes: *const u32,
        sample_size: u32,
        result_column_0: *const u32,
        result_column_1: *const u32,
        result_column_2: *const u32,
        result_column_3: *const u32,
        flattened_line_coeffs_size: u32,
    );

    pub fn gen_eq_evals(
        v: CudaSecureField,
        y: *const CudaSecureField,
        y_size: u32,
        evals: *const CudaSecureField,
        evals_size: u32,
    );

    pub fn fix_first_variable_base_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );

    pub fn fix_first_variable_secure_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );

    pub fn generate_wide_fibonacci_trace(
        input_a: *const u32,
        input_b: *const u32,
        input_len: u32,
        traces: *const *const u32,
        trace_len: u32,
        n_columns: u32,
    );

    pub fn generate_poseidon_traces(
        traces: *const *const u32,
        lookup_init: *const *const u32,
        lookup_final: *const *const u32,
        trace_log_len: u32,
    );

    pub fn generate_poseidon_interaction_traces(
        lookup_element: *mut c_void,
        lookup_init: *const *const u32,
        lookup_final: *const *const u32,
        log_size: u32,
        interaction_traces: *const *const u32,
        claimed_sum: *const u32,
    );

    // Assert EQ FP IMM trace generation
    pub fn generate_assert_eq_fp_imm_traces(
        trace_columns: *const *const u32,
        trace_columns_len: u32,
        registers_lookups: *const *const u32,
        registers_lookups_len: u32,
        memory_lookups: *const *const u32,
        memory_lookups_len: u32,
        range_check_20_lookups: *const *const u32,
        range_check_20_lookups_len: u32,
        inputs: *const c_void, // AssertEqFpImmInput*
        inputs_len: u32,
        data_accesses: *const c_void, // DataAccess*
        data_accesses_len: u32,
        log_size: u32,
        non_padded_length: u32,
    );

    pub fn evaluate_constraint_quotients_on_domain(
        quotients_0: *const u32,
        quotients_1: *const u32,
        quotients_2: *const u32,
        quotients_3: *const u32,
        trace0_evaluations: *const *const u32,
        trace0_evaluations_len: u32,
        trace1_evaluations: *const *const u32,
        trace1_evaluations_len: u32,
        trace2_evaluations: *const *const u32,
        trace2_evaluations_len: u32,
        random_coeff_powers: *const u32,
        denominator_inverses: *const u32,
        domain_log_size: u32,
        eval_domain_log_size: u32,
        number_of_columns: u32,
        logup_counts: u32,
        eval: *mut c_void,
        cumsum_shift: CudaSecureField,
        should_accumulate: bool,
        use_assert_evaluator: bool,
    ) -> bool;

    pub fn ntt_n2b_native_batch(
        value: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        start_stage: u32,
        end_stage: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    pub fn ntt_b2n_column(
        values_columns: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    pub fn ntt_n2b_columns(
        values_columns: *mut *mut u32,
        log_n: u32,
        num_poly: u32,
        g_twiddles: *const u32,
        twiddles_size: u32,
        eval_domain_size: u32,
    );

    // pub fn inclusive_prefix_sum(
    //     device_bit_rev_circle_domain_evals:  *const u32,
    //     len: u32,
    // );

    // GPU-accelerated PoW grinding for Blake2s channel
    pub fn grind_blake2s(prefixed_digest: *const u32, pow_bits: u32) -> u64;

    // Test function to compute offset_bit_reversed_circle_domain_index on GPU
    pub fn test_offset_bit_reversed_indices(
        result_host: *mut u32,
        domain_log_size: u32,
        eval_log_size: u32,
        offset: i32,
        n: u32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::offset_bit_reversed_circle_domain_index;

    #[test]
    fn test_cuda_offset_bit_reversed_indices_with_blowup() {
        // Test with blowup: eval_log_size = domain_log_size + 1
        const DOMAIN_LOG_SIZE: u32 = 10;
        const EVAL_LOG_SIZE: u32 = 11;
        const OFFSET: isize = -1;
        const N: usize = 1 << EVAL_LOG_SIZE; // Test all indices in eval domain

        // Compute Rust reference
        let rust_results: Vec<usize> = (0..N)
            .map(|i| {
                offset_bit_reversed_circle_domain_index(i, DOMAIN_LOG_SIZE, EVAL_LOG_SIZE, OFFSET)
            })
            .collect();

        // Compute CUDA results
        let mut cuda_results = vec![0u32; N];
        unsafe {
            test_offset_bit_reversed_indices(
                cuda_results.as_mut_ptr(),
                DOMAIN_LOG_SIZE,
                EVAL_LOG_SIZE,
                OFFSET as i32,
                N as u32,
            );
        }

        // Compare
        let mut mismatch_count = 0;
        for i in 0..N {
            if rust_results[i] as u32 != cuda_results[i] {
                if mismatch_count < 10 {
                    eprintln!(
                        "Mismatch at i={}: Rust={}, CUDA={}",
                        i, rust_results[i], cuda_results[i]
                    );
                }
                mismatch_count += 1;
            }
        }

        assert_eq!(
            mismatch_count, 0,
            "Found {} mismatches between Rust and CUDA offset computation",
            mismatch_count
        );
    }

    #[test]
    fn test_cuda_offset_bit_reversed_indices_log24_blowup() {
        // Test at log_size=24 with blowup (the actual failure case)
        const DOMAIN_LOG_SIZE: u32 = 24;
        const EVAL_LOG_SIZE: u32 = 25;
        const OFFSET: isize = -1;
        // Test first 10000 indices
        const N_SAMPLE: usize = 10000;

        // Test first N_SAMPLE indices
        let rust_first: Vec<usize> = (0..N_SAMPLE)
            .map(|i| {
                offset_bit_reversed_circle_domain_index(i, DOMAIN_LOG_SIZE, EVAL_LOG_SIZE, OFFSET)
            })
            .collect();

        let mut cuda_first = vec![0u32; N_SAMPLE];
        unsafe {
            test_offset_bit_reversed_indices(
                cuda_first.as_mut_ptr(),
                DOMAIN_LOG_SIZE,
                EVAL_LOG_SIZE,
                OFFSET as i32,
                N_SAMPLE as u32,
            );
        }

        let mut mismatch_first = 0;
        for i in 0..N_SAMPLE {
            if rust_first[i] as u32 != cuda_first[i] {
                if mismatch_first < 5 {
                    eprintln!(
                        "First batch mismatch at i={}: Rust={}, CUDA={}",
                        i, rust_first[i], cuda_first[i]
                    );
                }
                mismatch_first += 1;
            }
        }

        assert_eq!(
            mismatch_first, 0,
            "Found {} mismatches in first batch of log24 test",
            mismatch_first
        );

        println!(
            "test_cuda_offset_bit_reversed_indices_log24_blowup: First {} indices match!",
            N_SAMPLE
        );
    }

    #[test]
    fn test_cuda_offset_bit_reversed_indices_various_offsets() {
        // Test with various offsets
        const DOMAIN_LOG_SIZE: u32 = 8;
        const EVAL_LOG_SIZE: u32 = 9;
        const N: usize = 1 << EVAL_LOG_SIZE;

        for offset in [-2, -1, 1, 2] {
            let rust_results: Vec<usize> = (0..N)
                .map(|i| {
                    offset_bit_reversed_circle_domain_index(
                        i,
                        DOMAIN_LOG_SIZE,
                        EVAL_LOG_SIZE,
                        offset,
                    )
                })
                .collect();

            let mut cuda_results = vec![0u32; N];
            unsafe {
                test_offset_bit_reversed_indices(
                    cuda_results.as_mut_ptr(),
                    DOMAIN_LOG_SIZE,
                    EVAL_LOG_SIZE,
                    offset as i32,
                    N as u32,
                );
            }

            let mut mismatch_count = 0;
            for i in 0..N {
                if rust_results[i] as u32 != cuda_results[i] {
                    if mismatch_count < 5 {
                        eprintln!(
                            "offset={}, i={}: Rust={}, CUDA={}",
                            offset, i, rust_results[i], cuda_results[i]
                        );
                    }
                    mismatch_count += 1;
                }
            }

            assert_eq!(
                mismatch_count, 0,
                "Found {} mismatches for offset={}",
                mismatch_count, offset
            );
        }
    }
}
