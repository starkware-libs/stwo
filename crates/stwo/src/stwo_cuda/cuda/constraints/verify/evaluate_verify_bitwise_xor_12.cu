#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_verify_bitwise_xor_12.cuh"
#include "evaluate_common.cuh"

#define VERIFY_BITWISE_XOR_12_THREAD_COUNT_MAX 256

// Constants consistent with CPU-side verify_bitwise_xor_12.rs
static const unsigned VERIFY_BITWISE_XOR_12_ELEM_BITS = 12;
static const unsigned VERIFY_BITWISE_XOR_12_EXPAND_BITS = 2;
static const unsigned VERIFY_BITWISE_XOR_12_LIMB_BITS =
    VERIFY_BITWISE_XOR_12_ELEM_BITS - VERIFY_BITWISE_XOR_12_EXPAND_BITS;

template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_verify_bitwise_xor_12_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    VerifyBitwiseXor_12_Eval *xor_eval,
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

    // Preprocessed columns: a_low, b_low, c_low, from BitwiseXor::new(LIMB_BITS, i)
    m31 a_low = cuda_evaluator0.next_trace_mask();
    m31 b_low = cuda_evaluator0.next_trace_mask();
    m31 c_low = cuda_evaluator0.next_trace_mask();

    // Each (i, j) pair corresponds to one multiplicity column
    for (unsigned i = 0; i < (1u << VERIFY_BITWISE_XOR_12_EXPAND_BITS); ++i) {
        for (unsigned j = 0; j < (1u << VERIFY_BITWISE_XOR_12_EXPAND_BITS); ++j) {
            m31 multiplicity = cuda_evaluator1.next_trace_mask();

            m31 a_shift = (m31)(i << VERIFY_BITWISE_XOR_12_LIMB_BITS);
            m31 b_shift = (m31)(j << VERIFY_BITWISE_XOR_12_LIMB_BITS);
            m31 c_shift = (m31)((i ^ j) << VERIFY_BITWISE_XOR_12_LIMB_BITS);

            m31 a = add(a_low, a_shift);
            m31 b = add(b_low, b_shift);
            m31 c = add(c_low, c_shift);

            m31 values[4] = {VERIFY_BITWISE_XOR_12_RELATION_ID, a, b, c};

            m31 neg_mult = neg(multiplicity);
            qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};

            cuda_evaluator1.add_to_relation<4>(xor_eval->common_lookup_elements, multiplicity_ext, values);
        }
    }

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

extern "C"
void evaluate_verify_bitwise_xor_12(
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

    VerifyBitwiseXor_12_Eval *xor_eval =
        (VerifyBitwiseXor_12_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    VerifyBitwiseXor_12_Eval *device_xor_eval =
        cuda_malloc<VerifyBitwiseXor_12_Eval>(1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_12_Eval>(
        xor_eval, device_xor_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_verify_bitwise_xor_12");

    int block_dim = eval_domain_size < VERIFY_BITWISE_XOR_12_THREAD_COUNT_MAX
        ? eval_domain_size
        : VERIFY_BITWISE_XOR_12_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_verify_bitwise_xor_12_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_xor_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_verify_bitwise_xor_12_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_xor_eval,
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
    global_timer.end("evaluate_verify_bitwise_xor_12");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_xor_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}

