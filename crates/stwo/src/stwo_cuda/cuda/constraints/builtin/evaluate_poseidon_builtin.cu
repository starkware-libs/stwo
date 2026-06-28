// ============================================================================
// Poseidon Builtin CUDA AIR Evaluator (stwo-cairo-now version)
// ============================================================================
//
// Component: PoseidonBuiltin (6 trace columns)
// Translated from: cairo-air/src/components/poseidon_builtin.rs
//
// This is the decomposed version where the heavy Poseidon permutation is
// handled by a separate PoseidonAggregator component.
// This component just reads 6 memory IDs and delegates to the aggregator.
//
// Trace columns (6):
//   col0: input_state_0_id
//   col1: input_state_1_id
//   col2: input_state_2_id
//   col3: output_state_0_id
//   col4: output_state_1_id
//   col5: output_state_2_id
//
// Constraints: 0 regular constraints
// Relation lookups: 7 (6x MemoryAddressToId via ReadId, 1x PoseidonAggregator)
// logup_counts = 7, logup pairs = 4
//
// ============================================================================

#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_poseidon_builtin.cuh"
#include "evaluate_common.cuh"

// =====================================================================
// Pre-Kernel: Read trace columns, evaluate constraints, build logup fractions
// =====================================================================
template<typename EvaluatorT>
__global__ void evaluate_poseidon_builtin_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PoseidonBuiltin_Eval *pos_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    // Evaluator for preprocessed trace (trace0)
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

    // Read preprocessed column: seq
    m31 seq = cuda_evaluator0.next_trace_mask();

    // Evaluator for base trace (trace1)
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

    // Read 6 trace columns
    m31 input_state_0_id_col0 = cuda_evaluator.next_trace_mask();
    m31 input_state_1_id_col1 = cuda_evaluator.next_trace_mask();
    m31 input_state_2_id_col2 = cuda_evaluator.next_trace_mask();
    m31 output_state_0_id_col3 = cuda_evaluator.next_trace_mask();
    m31 output_state_1_id_col4 = cuda_evaluator.next_trace_mask();
    m31 output_state_2_id_col5 = cuda_evaluator.next_trace_mask();

    // Compute instance_addr = seq * 6 + poseidon_builtin_segment_start
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_3 = m31(3);
    const m31 M31_4 = m31(4);
    const m31 M31_5 = m31(5);
    const m31 M31_6 = m31(6);
    m31 instance_addr = add(mul(seq, M31_6), m31(pos_eval->claim.poseidon_builtin_segment_start));

    // ReadId for input_state_0: MemoryAddressToId(instance_addr, input_state_0_id)
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, instance_addr, input_state_0_id_col0};
        cuda_evaluator.add_to_relation<3>(
            pos_eval->common_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
    }

    // ReadId for input_state_1: MemoryAddressToId(instance_addr + 1, input_state_1_id)
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, add(instance_addr, M31_1), input_state_1_id_col1};
        cuda_evaluator.add_to_relation<3>(
            pos_eval->common_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
    }

    // ReadId for input_state_2: MemoryAddressToId(instance_addr + 2, input_state_2_id)
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, add(instance_addr, M31_2), input_state_2_id_col2};
        cuda_evaluator.add_to_relation<3>(
            pos_eval->common_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
    }

    // ReadId for output_state_0: MemoryAddressToId(instance_addr + 3, output_state_0_id)
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, add(instance_addr, M31_3), output_state_0_id_col3};
        cuda_evaluator.add_to_relation<3>(
            pos_eval->common_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
    }

    // ReadId for output_state_1: MemoryAddressToId(instance_addr + 4, output_state_1_id)
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, add(instance_addr, M31_4), output_state_1_id_col4};
        cuda_evaluator.add_to_relation<3>(
            pos_eval->common_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
    }

    // ReadId for output_state_2: MemoryAddressToId(instance_addr + 5, output_state_2_id)
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, add(instance_addr, M31_5), output_state_2_id_col5};
        cuda_evaluator.add_to_relation<3>(
            pos_eval->common_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
    }

    // PoseidonAggregator relation lookup
    {
        m31 values[7] = {
            POSEIDON_AGGREGATOR_RELATION_ID,
            input_state_0_id_col0,
            input_state_1_id_col1,
            input_state_2_id_col2,
            output_state_0_id_col3,
            output_state_1_id_col4,
            output_state_2_id_col5
        };
        cuda_evaluator.add_to_relation<7>(
            pos_eval->common_lookup_elements,
            qm31{M31_1, M31_0},
            values
        );
    }

    // Store constraint index and row result
    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// =====================================================================
// Host Wrapper Function
// =====================================================================
extern "C"
void evaluate_poseidon_builtin(
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

    PoseidonBuiltin_Eval *device_pos_eval = cuda_malloc<PoseidonBuiltin_Eval>(1);
    cuda_mem_copy_host_to_device<PoseidonBuiltin_Eval>((PoseidonBuiltin_Eval*)eval, device_pos_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_poseidon_builtin");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_poseidon_builtin_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_pos_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_poseidon_builtin_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_pos_eval,
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
        g_should_accumulate_host
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_poseidon_builtin");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_pos_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
