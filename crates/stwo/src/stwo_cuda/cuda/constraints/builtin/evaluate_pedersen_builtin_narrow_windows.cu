// Same as evaluate_pedersen_builtin.cu but uses PedersenAggregatorWindowBits9
// instead of PedersenAggregatorWindowBits18.
// Translated from: cairo-air/src/components/pedersen_builtin_narrow_windows.rs

#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_pedersen_builtin_narrow_windows.cuh"
#include "evaluate_common.cuh"

template<typename EvaluatorT>
__global__ void evaluate_pedersen_builtin_narrow_windows_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PedersenBuiltinNarrowWindows_Eval *ped_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    EvaluatorT cuda_evaluator0(
        trace0_evaluations,
        random_coeff_powers,
        0,
        row,
        {0},
        0,
        {0},
        domain_log_size,
        eval_domain_log_size,
        nullptr,
        0
    );

    m31 seq = cuda_evaluator0.next_trace_mask();

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

    m31 input_state_0_id_col0 = cuda_evaluator.next_trace_mask();
    m31 input_state_1_id_col1 = cuda_evaluator.next_trace_mask();
    m31 output_state_id_col2 = cuda_evaluator.next_trace_mask();

    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_3 = m31(3);
    m31 instance_addr = add(mul(seq, M31_3), m31(ped_eval->claim.pedersen_builtin_segment_start));

    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, instance_addr, input_state_0_id_col0};
        cuda_evaluator.add_to_relation<3>(
            ped_eval->common_lookup_elements,
            qm31{M31_1, m31(0)},
            values
        );
    }

    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, add(instance_addr, M31_1), input_state_1_id_col1};
        cuda_evaluator.add_to_relation<3>(
            ped_eval->common_lookup_elements,
            qm31{M31_1, m31(0)},
            values
        );
    }

    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, add(instance_addr, M31_2), output_state_id_col2};
        cuda_evaluator.add_to_relation<3>(
            ped_eval->common_lookup_elements,
            qm31{M31_1, m31(0)},
            values
        );
    }

    // PedersenAggregatorWindowBits9 (instead of WindowBits18)
    {
        m31 values[4] = {
            PEDERSEN_AGGREGATOR_WINDOW_BITS_9_RELATION_ID,
            input_state_0_id_col0,
            input_state_1_id_col1,
            output_state_id_col2
        };
        cuda_evaluator.add_to_relation<4>(
            ped_eval->common_lookup_elements,
            qm31{M31_1, m31(0)},
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

extern "C"
void evaluate_pedersen_builtin_narrow_windows(
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

    PedersenBuiltinNarrowWindows_Eval *device_ped_eval = cuda_malloc<PedersenBuiltinNarrowWindows_Eval>(1);
    cuda_mem_copy_host_to_device<PedersenBuiltinNarrowWindows_Eval>(
        (PedersenBuiltinNarrowWindows_Eval*)eval, device_ped_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_pedersen_builtin_narrow_windows");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_pedersen_builtin_narrow_windows_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_ped_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_pedersen_builtin_narrow_windows_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_ped_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < (int)logup_counts; ++i) {
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
        should_accumulate
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_pedersen_builtin_narrow_windows");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_ped_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
