#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_ret_opcode.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_common.cuh"

#define RET_OPCODE_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_ret_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    RetOpcode_Eval *ret_opcode_eval,
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

    // Load all 16 trace columns
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 next_pc_id_col3 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_0_col4 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_1_col5 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_2_col6 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_3_col7 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col8 = cuda_evaluator.next_trace_mask();
    m31 next_fp_id_col9 = cuda_evaluator.next_trace_mask();
    m31 next_fp_limb_0_col10 = cuda_evaluator.next_trace_mask();
    m31 next_fp_limb_1_col11 = cuda_evaluator.next_trace_mask();
    m31 next_fp_limb_2_col12 = cuda_evaluator.next_trace_mask();
    m31 next_fp_limb_3_col13 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col14 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_512 = m31(512);
    const m31 M31_262144 = m31(262144);
    const m31 M31_134217728 = m31(134217728);

    // DecodeInstruction15A61 — uses CommonLookupElements
    m31 decode_output[19];
    evaluate_decode_instruction_15a61(
        input_pc_col0,
        decode_output,
        ret_opcode_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Read next_pc from [fp-1] (ReadPositiveNumBits29) — uses CommonLookupElements
    m31 read_next_pc_output[29] = {0};
    evaluate_read_positive_num_bits_29(
        sub(input_fp_col2, M31_1),  // Address: fp - 1
        next_pc_id_col3,
        next_pc_limb_0_col4,
        next_pc_limb_1_col5,
        next_pc_limb_2_col6,
        next_pc_limb_3_col7,
        partial_limb_msb_col8,
        read_next_pc_output,
        ret_opcode_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Read next_fp from [fp-2] (ReadPositiveNumBits29) — uses CommonLookupElements
    m31 read_next_fp_output[29] = {0};
    evaluate_read_positive_num_bits_29(
        sub(input_fp_col2, M31_2),  // Address: fp - 2
        next_fp_id_col9,
        next_fp_limb_0_col10,
        next_fp_limb_1_col11,
        next_fp_limb_2_col12,
        next_fp_limb_3_col13,
        partial_limb_msb_col14,
        read_next_fp_output,
        ret_opcode_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Reconstruct next_pc from limbs
    m31 next_pc_reconstructed = add(
        add(
            next_pc_limb_0_col4,
            mul(next_pc_limb_1_col5, M31_512)
        ),
        add(
            mul(next_pc_limb_2_col6, M31_262144),
            mul(next_pc_limb_3_col7, M31_134217728)
        )
    );

    // Reconstruct next_fp from limbs
    m31 next_fp_reconstructed = add(
        add(
            next_fp_limb_0_col10,
            mul(next_fp_limb_1_col11, M31_512)
        ),
        add(
            mul(next_fp_limb_2_col12, M31_262144),
            mul(next_fp_limb_3_col13, M31_134217728)
        )
    );

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Add first opcodes relation entry (positive) — RELATION_ID prepended
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            input_pc_col0,
            input_ap_col1,
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            ret_opcode_eval->common_lookup_elements,
            qm31{enabler, 0, 0, 0},  // positive multiplicity
            values
        );
    }

    // Add second opcodes relation entry (negative) — RELATION_ID prepended
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            next_pc_reconstructed,
            input_ap_col1,
            next_fp_reconstructed
        };
        cuda_evaluator.add_to_relation<4>(
            ret_opcode_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},  // negative multiplicity
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host wrapper function
extern "C"
void evaluate_ret_opcode(
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

    RetOpcode_Eval *device_ret_eval = cuda_malloc<RetOpcode_Eval>(1);
    cuda_mem_copy_host_to_device<RetOpcode_Eval>((RetOpcode_Eval*)eval, device_ret_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_ret_opcode");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_ret_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_ret_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_ret_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_ret_eval,
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
    global_timer.end("evaluate_ret_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_ret_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
