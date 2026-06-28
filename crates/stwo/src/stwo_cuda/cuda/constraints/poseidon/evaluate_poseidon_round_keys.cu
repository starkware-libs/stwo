/*
============================================
Poseidon Round Keys CUDA AIR Evaluator
============================================

Component: PoseidonRoundKeys
translated from: cairo-air/src/components/poseidon_round_keys.rs
AIR version: 54d95c0d

Functionality:
- Lookup table for Poseidon hash round keys (precomputed constants)

Data Structure:
- 1 trace column: multiplicity (lookup table usage count)
- 30 preprocessed columns: poseidonroundkeys_0 to poseidonroundkeys_29
- 1 preprocessed column: seq (sequence ID)
- Total: 1 trace column, 31 preprocessed columns

Constraint Logic:
- No algebraic constraints
- Only 1 relation lookup: PoseidonRoundKeys (PROVIDE side)
- Entry: (seq, poseidonroundkeys_0..29) with negative multiplicity

Relation Lookups:
- PoseidonRoundKeys (PROVIDE): 1 instance

Key Algorithms:
- This is a lookup table provider component
- LOG_SIZE = 6 (fixed, 64 entries)
- Provides round key constants for Poseidon full and partial round chains
============================================
*/

#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_poseidon_round_keys.cuh"
#include "evaluate_common.cuh"

#define POSEIDON_ROUND_KEYS_THREAD_COUNT_MAX 256

// ============================================================================
// PoseidonRoundKeys Pre-Kernel: PROVIDE side relation lookup
// ============================================================================
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_poseidon_round_keys_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,  // Preprocessed columns
    m31 **trace1_evaluations,  // Trace columns
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PoseidonRoundKeys_Eval *poseidon_eval,
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

    // Evaluator for preprocessed columns (trace0)
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

    // ===================== read preprocessed columns =====================
    // Seq preprocessed column (table index)
    m31 seq = cuda_evaluator0.get_preprocessed_column();

    // Poseidon round keys preprocessed columns (30 round key constants)
    m31 poseidonroundkeys[30];
    for (int i = 0; i < 30; i++) {
        poseidonroundkeys[i] = cuda_evaluator0.get_preprocessed_column();
    }

    // Evaluator for trace columns (trace1)
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

    // ===================== read trace column =====================
    // Column 0: multiplicity (usage count)
    m31 multiplicity_col0 = cuda_evaluator1.next_trace_mask();

    // ===================== Relation Lookup: PoseidonRoundKeys (PROVIDE) =====================
    // Build lookup table entry: (RELATION_ID, seq, poseidonroundkeys[0..29])
    // PROVIDE side uses negative multiplicity
    m31 lookup_values[32];
    lookup_values[0] = POSEIDON_ROUND_KEYS_RELATION_ID;
    lookup_values[1] = seq;
    for (int i = 0; i < 30; i++) {
        lookup_values[i + 2] = poseidonroundkeys[i];
    }

    // PROVIDE side: multiplicity is negative
    // Use neg() to get proper M31 field negation (P - x), not m31(-1)
    qm31 multiplicity = qm31{{neg(multiplicity_col0), 0}, {0, 0}};
    cuda_evaluator1.add_to_relation<32>(poseidon_eval->common_lookup_elements, multiplicity, lookup_values);

    // ===================== Cleanup =====================
    // Store results back to global memory
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// ============================================================================
// Host Function: Evaluate Poseidon Round Keys
// ============================================================================
extern "C"
void evaluate_poseidon_round_keys(
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

    PoseidonRoundKeys_Eval *device_poseidon_eval = cuda_malloc<PoseidonRoundKeys_Eval>(1);
    cuda_mem_copy_host_to_device<PoseidonRoundKeys_Eval>((PoseidonRoundKeys_Eval*)eval, device_poseidon_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    int block_dim = eval_domain_size < POSEIDON_ROUND_KEYS_THREAD_COUNT_MAX ? eval_domain_size : POSEIDON_ROUND_KEYS_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_poseidon_round_keys_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_poseidon_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_poseidon_round_keys_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_poseidon_eval,
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

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_poseidon_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
