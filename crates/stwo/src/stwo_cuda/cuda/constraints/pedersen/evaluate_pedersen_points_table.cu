/*
============================================
Pedersen Points Table CUDA AIR Evaluator
============================================

Component: PedersenPointsTable
translated from: cairo-air/src/comptogethernts/pedersen_points_table.rs
AIR version: 54d95c0d

Functionality:
- Pedersen hash lookup table, provides precomputed elliptic curve points
- Lookup table for Pedersen hash precomputed elliptic curve points

Data Structure:
- 1 trace column: multiplicity (lookup table usage count)
- 56 preprocessed columns: pedersen_points_0 to pedersen_points_55
- 1 preprocessed column: pedersen_seq (sequence ID)
- Total: 1 trace column

Constraint Logic:
- No algebraic constraints
- Only 1 relation lookup: PedersenPointsTable (PROVIDE side)
- Entry: (seq, pedersen_points_0..55) with multiplicity

Relation Lookups:
- PedersenPointsTable (PROVIDE): 1 use

Key Algorithms:
- This is a lookup table provider component
- Multiplicity tracks how many times each table entry is used
- Provides point data for partial_ec_mul component
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
#include "evaluate_pedersen_points_table.cuh"
#include "evaluate_common.cuh"

#define PEDERSEN_POINTS_TABLE_THREAD_COUNT_MAX 256

// ============================================================================
// PedersenPointsTable Pre-Kernel: PROVIDE side relation lookup
// ============================================================================
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_pedersen_points_table_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PedersenPointsTable_Eval *pedersen_eval,
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

    // ===================== Read preprocessed columns =====================
    // Seq preprocessed column (table index)
    m31 pedersen_seq = cuda_evaluator0.next_trace_mask();

    // Pedersen points preprocessed columns (56 elliptic curve point coordinates)
    m31 pedersen_points[56];
    for (int i = 0; i < 56; i++) {
        pedersen_points[i] = cuda_evaluator0.next_trace_mask();
    }

    // ===================== Read Trace column =====================
    // Column 0: multiplicity (usage count)
    m31 multiplicity_col0 = cuda_evaluator1.next_trace_mask();

    // ===================== Relation Lookup: PedersenPointsTable (PROVIDE) =====================
    // Build lookup table entry: (RELATION_ID, seq, pedersen_points[0..55])
    // PROVIDE side uses negative multiplicity
    m31 lookup_values[58];
    lookup_values[0] = PEDERSEN_POINTS_TABLE_RELATION_ID;
    lookup_values[1] = pedersen_seq;
    for (int i = 0; i < 56; i++) {
        lookup_values[i + 2] = pedersen_points[i];
    }

    // PROVIDE side: multiplicity is negative
    // Use neg() to get proper M31 field negation (P - x), not m31(-1)
    qm31 multiplicity = qm31{{neg(multiplicity_col0), 0}, {0, 0}};
    cuda_evaluator1.add_to_relation<58>(pedersen_eval->common_lookup_elements, multiplicity, lookup_values);

    // ===================== Complete constraint evaluation =====================
    // No additional algebraic constraints, only relation lookup
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// ============================================================================
// PedersenPointsTable main function: orchestrates the entire evaluation process
// ============================================================================

extern "C"
void evaluate_pedersen_points_table(
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

    PedersenPointsTable_Eval *pedersen_eval =
        (PedersenPointsTable_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    // pedersen_points_tableuses：
    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    PedersenPointsTable_Eval *device_pedersen_eval =
        cuda_malloc<PedersenPointsTable_Eval>(1);
    cuda_mem_copy_host_to_device<PedersenPointsTable_Eval>(
        pedersen_eval, device_pedersen_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_pedersen_points_table");

    int block_dim = eval_domain_size < PEDERSEN_POINTS_TABLE_THREAD_COUNT_MAX
        ? eval_domain_size
        : PEDERSEN_POINTS_TABLE_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_pedersen_points_table_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_pedersen_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_pedersen_points_table_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_pedersen_eval,
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
    unsigned last_batch = logup_counts ? batching[logup_counts - 1] : 0;

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
    global_timer.end("evaluate_pedersen_points_table");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_pedersen_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
