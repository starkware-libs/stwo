#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_memory_id_to_big.cuh"
#include "evaluate_common.cuh"

#define MEMORY_ID_TO_BIG_THREAD_COUNT_MAX 256

// BigEval kernel - handles 28 limbs with 8 different range_check_9_9 relations
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_memory_id_to_big_big_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    MemoryIdToBig_BigEval *memory_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) {
        return;
    }

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

    // Get preprocessed Seq column
    m31 seq_val = cuda_evaluator0.next_trace_mask();

    // Read 28 limbs
    m31 value[N_M31_IN_FELT252];
    for (unsigned i = 0; i < N_M31_IN_FELT252; ++i) {
        value[i] = cuda_evaluator1.next_trace_mask();
    }
    m31 multiplicity = cuda_evaluator1.next_trace_mask();

    // Range check limbs in pairs using 8 different range_check_9_9 relations
    // 28 limbs = 14 pairs
    static constexpr m31 rc_relation_ids[8] = {
        RANGE_CHECK_9_9_RELATION_ID,
        RANGE_CHECK_9_9_B_RELATION_ID,
        RANGE_CHECK_9_9_C_RELATION_ID,
        RANGE_CHECK_9_9_D_RELATION_ID,
        RANGE_CHECK_9_9_E_RELATION_ID,
        RANGE_CHECK_9_9_F_RELATION_ID,
        RANGE_CHECK_9_9_G_RELATION_ID,
        RANGE_CHECK_9_9_H_RELATION_ID,
    };
    for (unsigned i = 0; i < N_M31_IN_FELT252 / 2; ++i) {
        qm31 multiplicity_one = {{1, 0}, {0, 0}};
        unsigned relation_index = i % 8;
        m31 values[3] = {rc_relation_ids[relation_index], value[2 * i], value[2 * i + 1]};
        cuda_evaluator1.add_to_relation<3>(memory_eval->common_lookup_elements, multiplicity_one, values);
    }

    // Yield the value with MemoryIdToBig relation
    // id = seq + LARGE_MEMORY_VALUE_ID_BASE + offset
    m31 id = add(add(seq_val, m31(LARGE_MEMORY_VALUE_ID_BASE)), m31(memory_eval->offset));

    // Prepare values array: [RELATION_ID, id, value[0..28]] - total 30 elements
    m31 id_and_value[30];
    id_and_value[0] = MEMORY_ID_TO_BIG_RELATION_ID;
    id_and_value[1] = id;
    for (unsigned i = 0; i < N_M31_IN_FELT252; ++i) {
        id_and_value[i + 2] = value[i];
    }

    m31 neg_mult = neg(multiplicity);
    qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};

    cuda_evaluator1.add_to_relation<30>(memory_eval->common_lookup_elements, multiplicity_ext, id_and_value);

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// SmallEval kernel - handles 8 limbs with 4 different range_check_9_9 relations
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_memory_id_to_big_small_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    MemoryIdToBig_SmallEval *memory_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) {
        return;
    }

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

    // Get preprocessed Seq column
    m31 seq_val = cuda_evaluator0.next_trace_mask();

    // Read 8 limbs
    m31 value[N_M31_IN_SMALL_FELT252];
    for (unsigned i = 0; i < N_M31_IN_SMALL_FELT252; ++i) {
        value[i] = cuda_evaluator1.next_trace_mask();
    }
    m31 multiplicity = cuda_evaluator1.next_trace_mask();

    // Range check limbs in pairs using 4 different range_check_9_9 relations
    // 8 limbs = 4 pairs
    static constexpr m31 rc_relation_ids[4] = {
        RANGE_CHECK_9_9_RELATION_ID,
        RANGE_CHECK_9_9_B_RELATION_ID,
        RANGE_CHECK_9_9_C_RELATION_ID,
        RANGE_CHECK_9_9_D_RELATION_ID,
    };
    for (unsigned i = 0; i < N_M31_IN_SMALL_FELT252 / 2; ++i) {
        qm31 multiplicity_one = {{1, 0}, {0, 0}};
        unsigned relation_index = i % 4;
        m31 values[3] = {rc_relation_ids[relation_index], value[2 * i], value[2 * i + 1]};
        cuda_evaluator1.add_to_relation<3>(memory_eval->common_lookup_elements, multiplicity_one, values);
    }

    // Yield the value with MemoryIdToBig relation
    // For small eval, id = seq (no offset)
    m31 id = seq_val;

    // Prepare values array: [RELATION_ID, id, value[0..8]] - but MemoryIdToBig still expects 29 elements + 1 relation ID
    // Pad with zeros to match the relation size
    m31 id_and_value[30];
    id_and_value[0] = MEMORY_ID_TO_BIG_RELATION_ID;
    id_and_value[1] = id;
    for (unsigned i = 0; i < N_M31_IN_SMALL_FELT252; ++i) {
        id_and_value[i + 2] = value[i];
    }
    // Pad remaining elements with zeros
    for (unsigned i = N_M31_IN_SMALL_FELT252 + 2; i < 30; ++i) {
        id_and_value[i] = m31(0);
    }

    m31 neg_mult = neg(multiplicity);
    qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};

    cuda_evaluator1.add_to_relation<30>(memory_eval->common_lookup_elements, multiplicity_ext, id_and_value);

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// BigEval extern C function
extern "C"
void evaluate_memory_id_to_big_big(
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

    MemoryIdToBig_BigEval *memory_eval = (MemoryIdToBig_BigEval *) eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    MemoryIdToBig_BigEval *device_memory_eval =
        cuda_malloc<MemoryIdToBig_BigEval>(1);
    cuda_mem_copy_host_to_device<MemoryIdToBig_BigEval>(
        memory_eval, device_memory_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_memory_id_to_big_big");

    int block_dim = eval_domain_size < MEMORY_ID_TO_BIG_THREAD_COUNT_MAX
        ? eval_domain_size
        : MEMORY_ID_TO_BIG_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_memory_id_to_big_big_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_memory_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_memory_id_to_big_big_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_memory_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // BigEval uses finalize_logup_in_pairs
    // 14 range check pairs + 1 memory_id_to_big = 15 logup relations
    // 15 relations / 2 = 7 pairs (with last one being a pair if odd)
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
    global_timer.end("evaluate_memory_id_to_big_big");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_memory_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}

// SmallEval extern C function
extern "C"
void evaluate_memory_id_to_big_small(
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

    MemoryIdToBig_SmallEval *memory_eval = (MemoryIdToBig_SmallEval *) eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    MemoryIdToBig_SmallEval *device_memory_eval =
        cuda_malloc<MemoryIdToBig_SmallEval>(1);
    cuda_mem_copy_host_to_device<MemoryIdToBig_SmallEval>(
        memory_eval, device_memory_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_memory_id_to_big_small");

    int block_dim = eval_domain_size < MEMORY_ID_TO_BIG_THREAD_COUNT_MAX
        ? eval_domain_size
        : MEMORY_ID_TO_BIG_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_memory_id_to_big_small_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_memory_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_memory_id_to_big_small_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_memory_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // SmallEval uses finalize_logup_in_pairs
    // 4 range check pairs + 1 memory_id_to_big = 5 logup relations
    // 5 relations / 2 = 2 pairs (with last one being a pair if odd)
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
    global_timer.end("evaluate_memory_id_to_big_small");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_memory_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
