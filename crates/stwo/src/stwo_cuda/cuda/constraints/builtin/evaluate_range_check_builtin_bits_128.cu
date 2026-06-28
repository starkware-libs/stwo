#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_range_check_builtin_bits_128.cuh"
#include "evaluate_range_check_last_limb_bits_in_ms_limb.cuh"
#include "evaluate_common.cuh"

#define RANGE_CHECK_BUILTIN_BITS_128_THREAD_COUNT_MAX 256

// Corresponds to CPU component range_check_builtin_bits_128.rs:
// - preprocessed column: Seq(log_size)
// - 17 trace columns: id + 15 limbs + partial_limb_msb
// - uses ReadPositiveNumBits128:
//      ReadId + ReadPositiveKnownIdNumBits128

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_range_check_builtin_bits_128_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    RangeCheckBuiltinBits128_Eval *range_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    // Evaluator for preprocessed trace (trace0)
    EvaluatorT cuda_evaluator0(
        trace0_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // Evaluator for base trace (trace1)
    EvaluatorT cuda_evaluator1(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    m31 seq = cuda_evaluator0.next_trace_mask();

    m31 value_id_col0       = cuda_evaluator1.next_trace_mask();
    m31 value_limb_0_col1   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_1_col2   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_2_col3   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_3_col4   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_4_col5   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_5_col6   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_6_col7   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_7_col8   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_8_col9   = cuda_evaluator1.next_trace_mask();
    m31 value_limb_9_col10  = cuda_evaluator1.next_trace_mask();
    m31 value_limb_10_col11 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_11_col12 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_12_col13 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_13_col14 = cuda_evaluator1.next_trace_mask();
    m31 value_limb_14_col15 = cuda_evaluator1.next_trace_mask();
    m31 partial_limb_msb_col16 = cuda_evaluator1.next_trace_mask();

    // address = segment_start + seq
    m31 segment_start = m31(range_eval->Claim.range_check_builtin_segment_start);
    m31 addr = add(segment_start, seq);

    // ReadId: address-to-id
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, addr, value_id_col0};
        cuda_evaluator1.add_to_relation<3>(range_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // ReadPositiveKnownIdNumBits128:
    // 1) RangeCheckLastLimbBitsInMsLimb2(value_limb_14, partial_limb_msb)
    evaluate_range_check_last_limb_bits_in_ms_limb_2(
        value_limb_14_col15,
        partial_limb_msb_col16,
        &cuda_evaluator1
    );

    // 2) id-to-big lookup with 16 value limbs (padded to 30 slots with RELATION_ID)
    {
        m31 M31_0 = m31(0);
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            value_id_col0,
            value_limb_0_col1,
            value_limb_1_col2,
            value_limb_2_col3,
            value_limb_3_col4,
            value_limb_4_col5,
            value_limb_5_col6,
            value_limb_6_col7,
            value_limb_7_col8,
            value_limb_8_col9,
            value_limb_9_col10,
            value_limb_10_col11,
            value_limb_11_col12,
            value_limb_12_col13,
            value_limb_13_col14,
            value_limb_14_col15,
            // padding to 30
            M31_0, M31_0, M31_0, M31_0, M31_0,
            M31_0, M31_0, M31_0, M31_0, M31_0,
            M31_0, M31_0, M31_0
        };
        cuda_evaluator1.add_to_relation<30>(range_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

extern "C"
void evaluate_range_check_builtin_bits_128(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator,
    cudaStream_t stream
) {
    (void)number_of_columns;

    RangeCheckBuiltinBits128_Eval *range_eval =
        (RangeCheckBuiltinBits128_Eval *) eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    RangeCheckBuiltinBits128_Eval *device_range_eval =
        cuda_malloc<RangeCheckBuiltinBits128_Eval>(1);
    cuda_mem_copy_host_to_device<RangeCheckBuiltinBits128_Eval>(
        range_eval, device_range_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_range_check_builtin_bits_128");

    int block_dim = eval_domain_size < RANGE_CHECK_BUILTIN_BITS_128_THREAD_COUNT_MAX
        ? eval_domain_size
        : RANGE_CHECK_BUILTIN_BITS_128_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_range_check_builtin_bits_128_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_range_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_range_check_builtin_bits_128_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_range_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (unsigned i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = batching[logup_counts - 1];

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    } else {
        generic_constraint_post_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generic_constraint_quotients_finalize_kernel<<<num_blocks, block_dim, 0, stream>>>(
        quotients_0,
        quotients_1,
        quotients_2,
        quotients_3,
        numerators,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        should_accumulate
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_range_check_builtin_bits_128");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_range_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
