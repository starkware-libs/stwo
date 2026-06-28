
#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_blake_g.cuh"
#include "evaluate_xor_rot_32R7.cuh"
#include "evaluate_xor_rot_32R8.cuh"
#include "evaluate_xor_rot_32R12.cuh"
#include "evaluate_xor_rot_32R16.cuh"
#include "evaluate_common.cuh"

#define BLAKE_G_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_blake_g_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    BlakeG_Eval *eval,
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

    for (unsigned instance_index = 0; instance_index < N_BLAKE_G_INSTANCES_PER_ROW; instance_index++) {
        // TODO: Tony (Move into L1/L2)
        const m31 M31_0 = { 0 };
        m31 input_limb_0_col0 = cuda_evaluator.next_trace_mask();
        m31 input_limb_1_col1 = cuda_evaluator.next_trace_mask();
        m31 input_limb_2_col2 = cuda_evaluator.next_trace_mask();
        m31 input_limb_3_col3 = cuda_evaluator.next_trace_mask();
        m31 input_limb_4_col4 = cuda_evaluator.next_trace_mask();
        m31 input_limb_5_col5 = cuda_evaluator.next_trace_mask();
        m31 input_limb_6_col6 = cuda_evaluator.next_trace_mask();
        m31 input_limb_7_col7 = cuda_evaluator.next_trace_mask();
        m31 input_limb_8_col8 = cuda_evaluator.next_trace_mask();
        m31 input_limb_9_col9 = cuda_evaluator.next_trace_mask();
        m31 input_limb_10_col10 = cuda_evaluator.next_trace_mask();
        m31 input_limb_11_col11 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_0_col12 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_1_col13 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col14 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col15 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col16 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col17 = cuda_evaluator.next_trace_mask();
        m31 xor_col18 = cuda_evaluator.next_trace_mask();
        m31 xor_col19 = cuda_evaluator.next_trace_mask();
        m31 xor_col20 = cuda_evaluator.next_trace_mask();
        m31 xor_col21 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_0_col22 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_1_col23 = cuda_evaluator.next_trace_mask();
        m31 ms_4_bits_col24 = cuda_evaluator.next_trace_mask();
        m31 ms_4_bits_col25 = cuda_evaluator.next_trace_mask();
        m31 ms_4_bits_col26 = cuda_evaluator.next_trace_mask();
        m31 ms_4_bits_col27 = cuda_evaluator.next_trace_mask();
        m31 xor_col28 = cuda_evaluator.next_trace_mask();
        m31 xor_col29 = cuda_evaluator.next_trace_mask();
        m31 xor_col30 = cuda_evaluator.next_trace_mask();
        m31 xor_col31 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_0_col32 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_1_col33 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col34 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col35 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col36 = cuda_evaluator.next_trace_mask();
        m31 ms_8_bits_col37 = cuda_evaluator.next_trace_mask();
        m31 xor_col38 = cuda_evaluator.next_trace_mask();
        m31 xor_col39 = cuda_evaluator.next_trace_mask();
        m31 xor_col40 = cuda_evaluator.next_trace_mask();
        m31 xor_col41 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_0_col42 = cuda_evaluator.next_trace_mask();
        m31 triple_sum32_res_limb_1_col43 = cuda_evaluator.next_trace_mask();
        m31 ms_9_bits_col44 = cuda_evaluator.next_trace_mask();
        m31 ms_9_bits_col45 = cuda_evaluator.next_trace_mask();
        m31 ms_9_bits_col46 = cuda_evaluator.next_trace_mask();
        m31 ms_9_bits_col47 = cuda_evaluator.next_trace_mask();
        m31 xor_col48 = cuda_evaluator.next_trace_mask();
        m31 xor_col49 = cuda_evaluator.next_trace_mask();
        m31 xor_col50 = cuda_evaluator.next_trace_mask();
        m31 xor_col51 = cuda_evaluator.next_trace_mask();
        m31 enabler = cuda_evaluator.next_trace_mask();

        triple_sum32_evaluate(
            input_limb_0_col0,
            input_limb_1_col1,
            input_limb_2_col2,
            input_limb_3_col3,
            input_limb_8_col8,
            input_limb_9_col9,
            triple_sum32_res_limb_0_col12,
            triple_sum32_res_limb_1_col13,
            &cuda_evaluator
        );

        m31 xor_rot_32_r_16_output_tmp_f72c8_21_limb_0 = {0};
        m31 xor_rot_32_r_16_output_tmp_f72c8_21_limb_1 = {0};
        xor_rot_32R16_evaluate(
            triple_sum32_res_limb_0_col12,
            triple_sum32_res_limb_1_col13,
            input_limb_6_col6,
            input_limb_7_col7,
            ms_8_bits_col14,
            ms_8_bits_col15,
            ms_8_bits_col16,
            ms_8_bits_col17,
            xor_col18,
            xor_col19,
            xor_col20,
            xor_col21,
            &xor_rot_32_r_16_output_tmp_f72c8_21_limb_0,
            &xor_rot_32_r_16_output_tmp_f72c8_21_limb_1,

            eval->common_lookup_elements,

            &cuda_evaluator
        );

        triple_sum32_evaluate(
            input_limb_4_col4,
            input_limb_5_col5,
            xor_rot_32_r_16_output_tmp_f72c8_21_limb_0,
            xor_rot_32_r_16_output_tmp_f72c8_21_limb_1,
            M31_0,
            M31_0,
            triple_sum32_res_limb_0_col22,
            triple_sum32_res_limb_1_col23,
            &cuda_evaluator
        );

        m31 xor_rot_32_r_12_output_tmp_f72c8_43_limb_0 = {0};
        m31 xor_rot_32_r_12_output_tmp_f72c8_43_limb_1 = {0};
        xor_rot_32R12_evaluate(
            input_limb_2_col2,
            input_limb_3_col3,
            triple_sum32_res_limb_0_col22,
            triple_sum32_res_limb_1_col23,
            ms_4_bits_col24,
            ms_4_bits_col25,
            ms_4_bits_col26,
            ms_4_bits_col27,
            xor_col28,
            xor_col29,
            xor_col30,
            xor_col31,

            &xor_rot_32_r_12_output_tmp_f72c8_43_limb_0,
            &xor_rot_32_r_12_output_tmp_f72c8_43_limb_1,

            eval->common_lookup_elements,

            &cuda_evaluator
        );

        triple_sum32_evaluate(
            triple_sum32_res_limb_0_col12,
            triple_sum32_res_limb_1_col13,
            xor_rot_32_r_12_output_tmp_f72c8_43_limb_0,
            xor_rot_32_r_12_output_tmp_f72c8_43_limb_1,
            input_limb_10_col10,
            input_limb_11_col11,
            triple_sum32_res_limb_0_col32,
            triple_sum32_res_limb_1_col33,
            &cuda_evaluator
        );

        m31 xor_rot_32_r_8_output_tmp_f72c8_65_limb_0 = {0};
        m31 xor_rot_32_r_8_output_tmp_f72c8_65_limb_1 = {0};
        xor_rot_32R8_evaluate(
            triple_sum32_res_limb_0_col32,
            triple_sum32_res_limb_1_col33,
            xor_rot_32_r_16_output_tmp_f72c8_21_limb_0,
            xor_rot_32_r_16_output_tmp_f72c8_21_limb_1,
            ms_8_bits_col34,
            ms_8_bits_col35,
            ms_8_bits_col36,
            ms_8_bits_col37,
            xor_col38,
            xor_col39,
            xor_col40,
            xor_col41,
            &xor_rot_32_r_8_output_tmp_f72c8_65_limb_0,
            &xor_rot_32_r_8_output_tmp_f72c8_65_limb_1,

            eval->common_lookup_elements,

            &cuda_evaluator
        );

        triple_sum32_evaluate(
            triple_sum32_res_limb_0_col22,
            triple_sum32_res_limb_1_col23,
            xor_rot_32_r_8_output_tmp_f72c8_65_limb_0,
            xor_rot_32_r_8_output_tmp_f72c8_65_limb_1,
            M31_0,
            M31_0,
            triple_sum32_res_limb_0_col42,
            triple_sum32_res_limb_1_col43,
            &cuda_evaluator
        );

        m31 xor_rot_32_r_7_output_tmp_f72c8_87_limb_0 = {0};
        m31 xor_rot_32_r_7_output_tmp_f72c8_87_limb_1 = {0};
        xor_rot_32R7_evaluate(
            xor_rot_32_r_12_output_tmp_f72c8_43_limb_0,
            xor_rot_32_r_12_output_tmp_f72c8_43_limb_1,
            triple_sum32_res_limb_0_col42,
            triple_sum32_res_limb_1_col43,
            ms_9_bits_col44,
            ms_9_bits_col45,
            ms_9_bits_col46,
            ms_9_bits_col47,
            xor_col48,
            xor_col49,
            xor_col50,
            xor_col51,
            &xor_rot_32_r_7_output_tmp_f72c8_87_limb_0,
            &xor_rot_32_r_7_output_tmp_f72c8_87_limb_1,

            eval->common_lookup_elements,

            &cuda_evaluator
        );

        // enabler is boolean (v1.1.0: moved after subroutines)
        cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

        m31 values[21] = {
            BLAKE_G_RELATION_ID,
            input_limb_0_col0,
            input_limb_1_col1,
            input_limb_2_col2,
            input_limb_3_col3,
            input_limb_4_col4,
            input_limb_5_col5,
            input_limb_6_col6,
            input_limb_7_col7,
            input_limb_8_col8,
            input_limb_9_col9,
            input_limb_10_col10,
            input_limb_11_col11,
            triple_sum32_res_limb_0_col32,
            triple_sum32_res_limb_1_col33,
            xor_rot_32_r_7_output_tmp_f72c8_87_limb_0,
            xor_rot_32_r_7_output_tmp_f72c8_87_limb_1,
            triple_sum32_res_limb_0_col42,
            triple_sum32_res_limb_1_col43,
            xor_rot_32_r_8_output_tmp_f72c8_65_limb_0,
            xor_rot_32_r_8_output_tmp_f72c8_65_limb_1,
        };

        cuda_evaluator.add_to_relation<21>(eval->common_lookup_elements, qm31{{neg(enabler), 0}, {0, 0}}, values);
    }

    numerators[row] = cuda_evaluator.row_res;
    constraint_index_array[row] = cuda_evaluator.constraint_index;
}

extern "C" void evaluate_blake_g(
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

    BlakeG_Eval *device_blake_g_eval = cuda_malloc<BlakeG_Eval>(1);
    cuda_mem_copy_host_to_device<BlakeG_Eval>((BlakeG_Eval*)eval, device_blake_g_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_blake_g");

    int block_dim = eval_domain_size < BLAKE_G_THREAD_COUNT_MAX ? eval_domain_size : BLAKE_G_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_blake_g_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_g_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_blake_g_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_g_eval,
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
    global_timer.end("evaluate_blake_g");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_blake_g_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}

