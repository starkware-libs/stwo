/*
============================================
PedersenPointsTableWindowBits9 CUDA AIR Evaluator
============================================

Component: PedersenPointsTableWindowBits9
Translated from: cairo-air/src/components/pedersen_points_table_window_bits_9.rs

Functionality:
- Pedersen hash lookup table (window_bits_9 variant)
- Provides precomputed elliptic curve points for Pedersen hash

Data Structure:
- 1 trace column: multiplicity_0 (lookup table usage count)
- 57 preprocessed columns: seq_15, pedersen_points_small_0..55
- Total: 1 trace column

Constraint Logic:
- No algebraic constraints
- 1 relation lookup (PROVIDE side): uses PEDERSEN_POINTS_TABLE_WINDOW_BITS_9_RELATION_ID
  with 58 values: (relation_id, seq_15, pedersen_points_small_0..55)
  and multiplicity = -multiplicity_0

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
#include "evaluate_pedersen_points_table_window_bits_9.cuh"
#include "evaluate_common.cuh"

#define PEDERSEN_POINTS_TABLE_WINDOW_BITS_9_THREAD_COUNT_MAX 256

// ============================================================================
// Pre-Kernel: Read preprocessed + trace columns, build logup fraction
// ============================================================================
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_pedersen_points_table_window_bits_9_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PedersenPointsTableWindowBits9_Eval *component_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    // Evaluator for preprocessed trace (trace0): reads seq_15 + pedersen_points_small_0..55
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

    // ===================== Read 57 preprocessed columns =====================
    // seq_15: sequence column (table index)
    m31 seq_15 = cuda_evaluator0.next_trace_mask();

    // pedersen_points_small_0..55: precomputed elliptic curve point coordinates
    m31 pedersen_points[56];
    for (int i = 0; i < 56; i++) {
        pedersen_points[i] = cuda_evaluator0.next_trace_mask();
    }

    // Evaluator for base trace (trace1): reads 1 trace column + builds logup
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

    // ===================== Read trace column =====================
    // Column 0: multiplicity (usage count)
    m31 multiplicity_0 = cuda_evaluator1.next_trace_mask();

    // ===================== Relation Lookup (PROVIDE side) =====================
    // Build lookup entry: 58 values
    //   [0] = PEDERSEN_POINTS_TABLE_WINDOW_BITS_9_RELATION_ID  (M31(1791500038))
    //   [1] = seq_15
    //   [2..57] = pedersen_points_small_0..55
    m31 lookup_values[58];
    lookup_values[0] = PEDERSEN_POINTS_TABLE_WINDOW_BITS_9_RELATION_ID;
    lookup_values[1] = seq_15;
    for (int i = 0; i < 56; i++) {
        lookup_values[i + 2] = pedersen_points[i];
    }

    // PROVIDE side: multiplicity is negated  (-multiplicity_0)
    qm31 neg_multiplicity = {{neg(multiplicity_0), 0}, {0, 0}};
    cuda_evaluator1.add_to_relation<58>(
        component_eval->common_lookup_elements, neg_multiplicity, lookup_values
    );

    // ===================== Store results =====================
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// ============================================================================
// Host wrapper function
// ============================================================================
extern "C"
void evaluate_pedersen_points_table_window_bits_9(
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
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    PedersenPointsTableWindowBits9_Eval *device_component_eval =
        cuda_malloc<PedersenPointsTableWindowBits9_Eval>(1);
    cuda_mem_copy_host_to_device<PedersenPointsTableWindowBits9_Eval>(
        (PedersenPointsTableWindowBits9_Eval *)eval, device_component_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_pedersen_points_table_window_bits_9");

    int block_dim = eval_domain_size < PEDERSEN_POINTS_TABLE_WINDOW_BITS_9_THREAD_COUNT_MAX
        ? eval_domain_size
        : PEDERSEN_POINTS_TABLE_WINDOW_BITS_9_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_pedersen_points_table_window_bits_9_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_component_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_pedersen_points_table_window_bits_9_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_component_eval,
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
        g_should_accumulate_host
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_pedersen_points_table_window_bits_9");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_component_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
