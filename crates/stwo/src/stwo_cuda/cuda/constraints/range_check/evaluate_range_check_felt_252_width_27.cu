#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_range_check_felt_252_width_27.cuh"
#include "evaluate_common.cuh"

#define RANGE_CHECK_FELT_252_WIDTH_27_THREAD_COUNT_MAX 256

// Corresponds to CPU component range_check_felt_252_width_27.rs.

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_range_check_felt_252_width_27_pre_kernel(
    qm31 *numerators,
    m31 **trace_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    RangeCheckFelt252Width27_Eval *range_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    EvaluatorT cuda_evaluator(
        trace_evaluations,
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

    const m31 M31_262144 = m31(262144);
    const m31 M31_4194304 = m31(4194304);

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
    m31 limb_0_high_part_col10 = cuda_evaluator.next_trace_mask();
    m31 limb_1_low_part_col11 = cuda_evaluator.next_trace_mask();
    m31 limb_2_high_part_col12 = cuda_evaluator.next_trace_mask();
    m31 limb_3_low_part_col13 = cuda_evaluator.next_trace_mask();
    m31 limb_4_high_part_col14 = cuda_evaluator.next_trace_mask();
    m31 limb_5_low_part_col15 = cuda_evaluator.next_trace_mask();
    m31 limb_6_high_part_col16 = cuda_evaluator.next_trace_mask();
    m31 limb_7_low_part_col17 = cuda_evaluator.next_trace_mask();
    m31 limb_8_high_part_col18 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // enabler^2 = enabler
    cuda_evaluator.add_constraint(
        sub(mul(enabler, enabler), enabler)
    );

    // range_check_9_9 on limb_0_high_part, limb_1_low_part
    {
        m31 values[3] = {RANGE_CHECK_9_9_RELATION_ID, limb_0_high_part_col10, limb_1_low_part_col11};
        cuda_evaluator.add_to_relation<3>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_0 - limb_0_high_part * 2^18) in RangeCheck_18
    {
        m31 expr = sub(
            input_limb_0_col0,
            mul(limb_0_high_part_col10, M31_262144)
        );
        m31 values[2] = {RANGE_CHECK_18_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_1 - limb_1_low_part) * 2^22 in RangeCheck_18
    {
        m31 expr = mul(sub(input_limb_1_col1, limb_1_low_part_col11),
                       M31_4194304);
        m31 values[2] = {RANGE_CHECK_18_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // range_check_9_9_B on limb_2_high_part, limb_3_low_part
    {
        m31 values[3] = {RANGE_CHECK_9_9_B_RELATION_ID, limb_2_high_part_col12, limb_3_low_part_col13};
        cuda_evaluator.add_to_relation<3>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_2 - limb_2_high_part * 2^18) in RangeCheck_18_B
    {
        m31 expr = sub(
            input_limb_2_col2,
            mul(limb_2_high_part_col12, M31_262144)
        );
        m31 values[2] = {RANGE_CHECK_18_B_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_3 - limb_3_low_part) * 2^22 in RangeCheck_18
    {
        m31 expr = mul(sub(input_limb_3_col3, limb_3_low_part_col13),
                       M31_4194304);
        m31 values[2] = {RANGE_CHECK_18_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // range_check_9_9_C on limb_4_high_part, limb_5_low_part
    {
        m31 values[3] = {RANGE_CHECK_9_9_C_RELATION_ID, limb_4_high_part_col14, limb_5_low_part_col15};
        cuda_evaluator.add_to_relation<3>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_4 - limb_4_high_part * 2^18) in RangeCheck_18
    {
        m31 expr = sub(
            input_limb_4_col4,
            mul(limb_4_high_part_col14, M31_262144)
        );
        m31 values[2] = {RANGE_CHECK_18_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_5 - limb_5_low_part) * 2^22 in RangeCheck_18
    {
        m31 expr = mul(sub(input_limb_5_col5, limb_5_low_part_col15),
                       M31_4194304);
        m31 values[2] = {RANGE_CHECK_18_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // range_check_9_9_D on limb_6_high_part, limb_7_low_part
    {
        m31 values[3] = {RANGE_CHECK_9_9_D_RELATION_ID, limb_6_high_part_col16, limb_7_low_part_col17};
        cuda_evaluator.add_to_relation<3>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_6 - limb_6_high_part * 2^18) in RangeCheck_18_B
    {
        m31 expr = sub(
            input_limb_6_col6,
            mul(limb_6_high_part_col16, M31_262144)
        );
        m31 values[2] = {RANGE_CHECK_18_B_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_7 - limb_7_low_part) * 2^22 in RangeCheck_18
    {
        m31 expr = mul(sub(input_limb_7_col7, limb_7_low_part_col17),
                       M31_4194304);
        m31 values[2] = {RANGE_CHECK_18_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // range_check_9_9_E on limb_8_high_part, input_limb_9
    {
        m31 values[3] = {RANGE_CHECK_9_9_E_RELATION_ID, limb_8_high_part_col18, input_limb_9_col9};
        cuda_evaluator.add_to_relation<3>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (input_limb_8 - limb_8_high_part * 2^18) in RangeCheck_18
    {
        m31 expr = sub(
            input_limb_8_col8,
            mul(limb_8_high_part_col18, M31_262144)
        );
        m31 values[2] = {RANGE_CHECK_18_RELATION_ID, expr};
        cuda_evaluator.add_to_relation<2>(range_eval->common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // Felt252 width-27 main relation, multiplicity = -enabler
    {
        m31 neg_enabler = neg(enabler);
        qm31 multiplicity_ext = {{neg_enabler, 0}, {0, 0}};

        m31 values[11] = {
            RANGE_CHECK_252_WIDTH_27_RELATION_ID,
            input_limb_0_col0,
            input_limb_1_col1,
            input_limb_2_col2,
            input_limb_3_col3,
            input_limb_4_col4,
            input_limb_5_col5,
            input_limb_6_col6,
            input_limb_7_col7,
            input_limb_8_col8,
            input_limb_9_col9
        };
        cuda_evaluator.add_to_relation<11>(range_eval->common_lookup_elements, multiplicity_ext, values);
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

extern "C"
void evaluate_range_check_felt_252_width_27(
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
    (void)trace0_evaluations;
    (void)trace0_evaluations_len;

    RangeCheckFelt252Width27_Eval *range_eval =
        (RangeCheckFelt252Width27_Eval *) eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    // All columns for this component are in trace1, so directly use trace1_evaluations
    m31 **device_trace_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    RangeCheckFelt252Width27_Eval *device_range_eval =
        cuda_malloc<RangeCheckFelt252Width27_Eval>(1);
    cuda_mem_copy_host_to_device<RangeCheckFelt252Width27_Eval>(
        range_eval, device_range_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_range_check_felt_252_width_27");

    int block_dim =
        eval_domain_size < RANGE_CHECK_FELT_252_WIDTH_27_THREAD_COUNT_MAX
            ? eval_domain_size
            : RANGE_CHECK_FELT_252_WIDTH_27_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_range_check_felt_252_width_27_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace1_evaluations_len,
            device_range_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_range_check_felt_252_width_27_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace1_evaluations_len,
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
    global_timer.end("evaluate_range_check_felt_252_width_27");

    cuda_free_memory(device_trace_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_range_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}

