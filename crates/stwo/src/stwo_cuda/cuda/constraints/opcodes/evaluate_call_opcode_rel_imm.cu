#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_call_opcode_rel_imm.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_read_small.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_common.cuh"

#define CALL_OPCODE_REL_IMM_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_call_opcode_rel_imm_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    CallOpcodeRelImm_Eval *call_opcode_rel_imm_eval,
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

    // Load all 24 trace columns
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 stored_fp_id_col3 = cuda_evaluator.next_trace_mask();
    m31 stored_fp_limb_0_col4 = cuda_evaluator.next_trace_mask();
    m31 stored_fp_limb_1_col5 = cuda_evaluator.next_trace_mask();
    m31 stored_fp_limb_2_col6 = cuda_evaluator.next_trace_mask();
    m31 stored_fp_limb_3_col7 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col8 = cuda_evaluator.next_trace_mask();
    m31 stored_ret_pc_id_col9 = cuda_evaluator.next_trace_mask();
    m31 stored_ret_pc_limb_0_col10 = cuda_evaluator.next_trace_mask();
    m31 stored_ret_pc_limb_1_col11 = cuda_evaluator.next_trace_mask();
    m31 stored_ret_pc_limb_2_col12 = cuda_evaluator.next_trace_mask();
    m31 stored_ret_pc_limb_3_col13 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col14 = cuda_evaluator.next_trace_mask();
    m31 distance_to_next_pc_id_col15 = cuda_evaluator.next_trace_mask();
    m31 msb_col16 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col17 = cuda_evaluator.next_trace_mask();
    m31 distance_to_next_pc_limb_0_col18 = cuda_evaluator.next_trace_mask();
    m31 distance_to_next_pc_limb_1_col19 = cuda_evaluator.next_trace_mask();
    m31 distance_to_next_pc_limb_2_col20 = cuda_evaluator.next_trace_mask();
    m31 remainder_bits_col21 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col22 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_512 = m31(512);
    const m31 M31_262144 = m31(262144);
    const m31 M31_134217728 = m31(134217728);

    // DecodeInstruction2A7A2 - simpler than DecodeInstructionF1Edd
    // No output values needed for call_opcode_rel_imm
    m31 decode_output[19];
    evaluate_decode_instruction_2a7a2(
        input_pc_col0,
        decode_output,
        call_opcode_rel_imm_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Read stored_fp (ReadPositiveNumBits29)
    // [ap] = fp
    m31 read_fp_output[29] = {0};
    evaluate_read_positive_num_bits_29(
        input_ap_col1,  // Address: ap (not ap+1)
        stored_fp_id_col3,
        stored_fp_limb_0_col4,
        stored_fp_limb_1_col5,
        stored_fp_limb_2_col6,
        stored_fp_limb_3_col7,
        partial_limb_msb_col8,
        read_fp_output,
        call_opcode_rel_imm_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Reconstruct stored_fp from limbs
    m31 stored_fp_reconstructed = add(
        add(
            stored_fp_limb_0_col4,
            mul(stored_fp_limb_1_col5, M31_512)
        ),
        add(
            mul(stored_fp_limb_2_col6, M31_262144),
            mul(stored_fp_limb_3_col7, M31_134217728)
        )
    );

    // Constraint: stored_fp_reconstructed == input_fp
    cuda_evaluator.add_constraint(
        sub(stored_fp_reconstructed, input_fp_col2)
    );

    // Read stored_ret_pc (ReadPositiveNumBits29)
    // [ap+1] = return_pc
    m31 read_ret_pc_output[29] = {0};
    evaluate_read_positive_num_bits_29(
        add(input_ap_col1, M31_1),  // Address: ap + 1
        stored_ret_pc_id_col9,
        stored_ret_pc_limb_0_col10,
        stored_ret_pc_limb_1_col11,
        stored_ret_pc_limb_2_col12,
        stored_ret_pc_limb_3_col13,
        partial_limb_msb_col14,
        read_ret_pc_output,
        call_opcode_rel_imm_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Reconstruct stored_ret_pc from limbs
    m31 stored_ret_pc_reconstructed = add(
        add(
            stored_ret_pc_limb_0_col10,
            mul(stored_ret_pc_limb_1_col11, M31_512)
        ),
        add(
            mul(stored_ret_pc_limb_2_col12, M31_262144),
            mul(stored_ret_pc_limb_3_col13, M31_134217728)
        )
    );

    // CRITICAL: Constraint: stored_ret_pc_reconstructed == input_pc + 2
    // This is different from call_opcode! Call rel imm is a 2-word instruction.
    cuda_evaluator.add_constraint(
        sub(stored_ret_pc_reconstructed, add(input_pc_col0, M31_2))
    );

    // Read distance using ReadSmall
    // The distance immediate is at [input_pc + 1]
    m31 read_small_output[2] = {0};
    evaluate_read_small(
        add(input_pc_col0, M31_1),  // Address: pc + 1 (immediate value location)
        distance_to_next_pc_id_col15,
        msb_col16,
        mid_limbs_set_col17,
        distance_to_next_pc_limb_0_col18,
        distance_to_next_pc_limb_1_col19,
        distance_to_next_pc_limb_2_col20,
        remainder_bits_col21,
        partial_limb_msb_col22,
        read_small_output,
        call_opcode_rel_imm_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // read_small_output[0] is the reconstructed distance value (limb_0)
    m31 distance = read_small_output[0];

    // Compute next_pc = input_pc + distance
    m31 next_pc = add(input_pc_col0, distance);

    // Compute new_ap and new_fp for opcodes relation
    m31 new_ap = add(input_ap_col1, M31_2);
    m31 new_fp = new_ap;

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Add first opcodes relation entry (positive)
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            input_pc_col0,
            input_ap_col1,
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            call_opcode_rel_imm_eval->common_lookup_elements,
            qm31{enabler, 0, 0, 0},  // positive multiplicity
            values
        );
    }

    // Add second opcodes relation entry (negative)
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            next_pc,
            new_ap,
            new_fp
        };
        cuda_evaluator.add_to_relation<4>(
            call_opcode_rel_imm_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},  // negative multiplicity
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host wrapper function
extern "C"
void evaluate_call_opcode_rel_imm(
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

    CallOpcodeRelImm_Eval *device_call_eval = cuda_malloc<CallOpcodeRelImm_Eval>(1);
    cuda_mem_copy_host_to_device<CallOpcodeRelImm_Eval>((CallOpcodeRelImm_Eval*)eval, device_call_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_call_opcode_rel_imm");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_call_opcode_rel_imm_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_call_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_call_opcode_rel_imm_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_call_eval,
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
    global_timer.end("evaluate_call_opcode_rel_imm");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_call_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
