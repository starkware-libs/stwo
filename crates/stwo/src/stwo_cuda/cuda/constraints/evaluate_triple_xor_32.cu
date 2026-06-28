#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_triple_xor_32.cuh"
#include "evaluate_xor_rot_32R8.cuh"
#include "evaluate_common.cuh"

#define TRIPLE_XOR_32_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_triple_xor_32_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    TripleXor32_Eval *eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    EvaluatorT cuda_evaluator(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        {0},
        0,
        {0},
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );
    const m31 M31_256 = { 256 };
    m31 input_limb_0_col0 = cuda_evaluator.next_trace_mask();
    m31 input_limb_1_col1 = cuda_evaluator.next_trace_mask();
    m31 input_limb_2_col2 = cuda_evaluator.next_trace_mask();
    m31 input_limb_3_col3 = cuda_evaluator.next_trace_mask();
    m31 input_limb_4_col4 = cuda_evaluator.next_trace_mask();
    m31 input_limb_5_col5 = cuda_evaluator.next_trace_mask();
    m31 ms_8_bits_col6 = cuda_evaluator.next_trace_mask();
    m31 ms_8_bits_col7 = cuda_evaluator.next_trace_mask();
    m31 ms_8_bits_col8 = cuda_evaluator.next_trace_mask();
    m31 ms_8_bits_col9 = cuda_evaluator.next_trace_mask();
    m31 ms_8_bits_col10 = cuda_evaluator.next_trace_mask();
    m31 ms_8_bits_col11 = cuda_evaluator.next_trace_mask();
    m31 xor_col12 = cuda_evaluator.next_trace_mask();
    m31 xor_col13 = cuda_evaluator.next_trace_mask();
    m31 xor_col14 = cuda_evaluator.next_trace_mask();
    m31 xor_col15 = cuda_evaluator.next_trace_mask();
    m31 xor_col16 = cuda_evaluator.next_trace_mask();
    m31 xor_col17 = cuda_evaluator.next_trace_mask();
    m31 xor_col18 = cuda_evaluator.next_trace_mask();
    m31 xor_col19 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    m31 split_16_low_part_size_8_output_tmp_298db_1_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_298db_1_limb_1 = {0};
    splite16_low_part_size8_32R8_evaluate(
        input_limb_0_col0,
        ms_8_bits_col6,
        &split_16_low_part_size_8_output_tmp_298db_1_limb_0,
        &split_16_low_part_size_8_output_tmp_298db_1_limb_1
    );

    m31 split_16_low_part_size_8_output_tmp_298db_3_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_298db_3_limb_1 = {0};
    splite16_low_part_size8_32R8_evaluate(
        input_limb_1_col1,
        ms_8_bits_col7,
        &split_16_low_part_size_8_output_tmp_298db_3_limb_0,
        &split_16_low_part_size_8_output_tmp_298db_3_limb_1
    );

    m31 split_16_low_part_size_8_output_tmp_298db_5_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_298db_5_limb_1 = {0};
    splite16_low_part_size8_32R8_evaluate(
        input_limb_2_col2,
        ms_8_bits_col8,
        &split_16_low_part_size_8_output_tmp_298db_5_limb_0,
        &split_16_low_part_size_8_output_tmp_298db_5_limb_1
    );

    m31 split_16_low_part_size_8_output_tmp_298db_7_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_298db_7_limb_1 = {0};
    splite16_low_part_size8_32R8_evaluate(
        input_limb_3_col3,
        ms_8_bits_col9,
        &split_16_low_part_size_8_output_tmp_298db_7_limb_0,
        &split_16_low_part_size_8_output_tmp_298db_7_limb_1
    );

    m31 split_16_low_part_size_8_output_tmp_298db_9_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_298db_9_limb_1 = {0};
    splite16_low_part_size8_32R8_evaluate(
        input_limb_4_col4,
        ms_8_bits_col10,
        &split_16_low_part_size_8_output_tmp_298db_9_limb_0,
        &split_16_low_part_size_8_output_tmp_298db_9_limb_1
    );

    m31 split_16_low_part_size_8_output_tmp_298db_11_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_298db_11_limb_1 = {0};
    splite16_low_part_size8_32R8_evaluate(
        input_limb_5_col5,
        ms_8_bits_col11,
        &split_16_low_part_size_8_output_tmp_298db_11_limb_0,
        &split_16_low_part_size_8_output_tmp_298db_11_limb_1
    );

    bitwise_xor_numbits8_32R8_evaluate(
        split_16_low_part_size_8_output_tmp_298db_1_limb_0,
        split_16_low_part_size_8_output_tmp_298db_5_limb_0,
        xor_col12,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    bitwise_xor_numbits8_32R8_evaluate(
        xor_col12,
        split_16_low_part_size_8_output_tmp_298db_9_limb_0,
        xor_col13,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    bitwise_xor_numbits8_32R8_evaluate(
        ms_8_bits_col6,
        ms_8_bits_col8,
        xor_col14,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    bitwise_xor_numbits8_32R8_evaluate(
        xor_col14,
        ms_8_bits_col10,
        xor_col15,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Last 4 XORs use verify_bitwise_xor_8_b_lookup_elements (BitwiseXorNumBits8B)
    bitwise_xor_numbits8_b_32R8_evaluate(
        split_16_low_part_size_8_output_tmp_298db_3_limb_0,
        split_16_low_part_size_8_output_tmp_298db_7_limb_0,
        xor_col16,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    bitwise_xor_numbits8_b_32R8_evaluate(
        xor_col16,
        split_16_low_part_size_8_output_tmp_298db_11_limb_0,
        xor_col17,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    bitwise_xor_numbits8_b_32R8_evaluate(
        ms_8_bits_col7,
        ms_8_bits_col9,
        xor_col18,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    bitwise_xor_numbits8_b_32R8_evaluate(
        xor_col18,
        ms_8_bits_col11,
        xor_col19,
        eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 triple_xor32_output_tmp_298db_28_limb_0 =  mul(xor_col15, M31_256);
    triple_xor32_output_tmp_298db_28_limb_0 = add(xor_col13, triple_xor32_output_tmp_298db_28_limb_0);

    m31 triple_xor32_output_tmp_298db_28_limb_1 = mul(xor_col19, M31_256);
    triple_xor32_output_tmp_298db_28_limb_1 = add(xor_col17, triple_xor32_output_tmp_298db_28_limb_1);

    m31 values[9] = {
        TRIPLE_XOR_32_RELATION_ID,
        input_limb_0_col0,
        input_limb_1_col1,
        input_limb_2_col2,
        input_limb_3_col3,
        input_limb_4_col4,
        input_limb_5_col5,
        triple_xor32_output_tmp_298db_28_limb_0,
        triple_xor32_output_tmp_298db_28_limb_1,
    };

    cuda_evaluator.add_to_relation<9>(eval->common_lookup_elements, qm31{{neg(enabler), 0}, {0, 0}}, values);

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

extern "C" void evaluate_triple_xor_32(
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

    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    TripleXor32_Eval *device_triple_xor_32_eval = cuda_malloc<TripleXor32_Eval>(1);
    cuda_mem_copy_host_to_device<TripleXor32_Eval>((TripleXor32_Eval*)eval, device_triple_xor_32_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_triple_xor_32");

    int block_dim = eval_domain_size < TRIPLE_XOR_32_THREAD_COUNT_MAX ? eval_domain_size : TRIPLE_XOR_32_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_triple_xor_32_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_triple_xor_32_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_triple_xor_32_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_triple_xor_32_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = batching[logup_counts - 1];

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        generic_constraint_post_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        g_should_accumulate_host  // Read from global variable set by dispatcher
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_triple_xor_32");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_triple_xor_32_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}

