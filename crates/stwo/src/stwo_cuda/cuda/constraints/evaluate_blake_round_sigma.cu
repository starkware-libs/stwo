#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_blake_round_sigma.cuh"
#include "evaluate_common.cuh"

#define BLAKE_ROUND_SIGMA_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_blake_round_sigma_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    BlakeRoundSigma_Eval *eval,
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
        intermediate_fractions,
        logup_counts
    );

    // All preprocessed columns - use get_preprocessed_column() not next_trace_mask()
    m31 seq = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_0 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_1 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_2 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_3 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_4 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_5 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_6 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_7 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_8 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_9 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_10 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_11 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_12 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_13 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_14 = cuda_evaluator0.get_preprocessed_column();
    m31 blakesigma_15 = cuda_evaluator0.get_preprocessed_column();

    EvaluatorT cuda_evaluator1(
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
    m31 multiplicity = cuda_evaluator1.next_trace_mask();

    m31 values[18] = {
        BLAKE_ROUND_SIGMA_RELATION_ID,
        seq,
        blakesigma_0,
        blakesigma_1,
        blakesigma_2,
        blakesigma_3,
        blakesigma_4,
        blakesigma_5,
        blakesigma_6,
        blakesigma_7,
        blakesigma_8,
        blakesigma_9,
        blakesigma_10,
        blakesigma_11,
        blakesigma_12,
        blakesigma_13,
        blakesigma_14,
        blakesigma_15
    };

    cuda_evaluator1.add_to_relation<18>(eval->common_lookup_elements, qm31{{(P - multiplicity), 0}, {0, 0}}, values);

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

extern "C" void evaluate_blake_round_sigma(
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

    BlakeRoundSigma_Eval *device_blake_round_sigma_eval = cuda_malloc<BlakeRoundSigma_Eval>(1);
    cuda_mem_copy_host_to_device<BlakeRoundSigma_Eval>((BlakeRoundSigma_Eval*)eval, device_blake_round_sigma_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_blake_round_sigma");

    int block_dim = eval_domain_size < BLAKE_ROUND_SIGMA_THREAD_COUNT_MAX ? eval_domain_size : BLAKE_ROUND_SIGMA_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_blake_round_sigma_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_round_sigma_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_blake_round_sigma_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_blake_round_sigma_eval,
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
    global_timer.end("evaluate_blake_round_sigma");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_blake_round_sigma_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
