#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_jump_opcode_rel.cuh"
#include "evaluate_read_small.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_common.cuh"

#define JUMP_OPCODE_REL_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_jump_opcode_rel_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    JumpOpcodeRel_Eval *jump_opcode_rel_eval,
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

    // Load all 16 trace columns (was 17 in legacy; op1_base_ap removed)
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset2_col3 = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp_col4 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col5 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_col6 = cuda_evaluator.next_trace_mask();
    m31 next_pc_id_col7 = cuda_evaluator.next_trace_mask();
    m31 msb_col8 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col9 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_0_col10 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_1_col11 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_2_col12 = cuda_evaluator.next_trace_mask();
    m31 remainder_bits_col13 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col14 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // DecodeInstructionBa944 (no separate op1_base_ap column)
    m31 decode_output[2];
    evaluate_decode_instruction_ba944(
        input_pc_col0,
        offset2_col3,
        op1_base_fp_col4,
        ap_update_add_1_col5,
        decode_output,
        jump_opcode_rel_eval->common_lookup_elements,
        &cuda_evaluator
    );
    // decode_output[0] = offset2 - 32768
    // decode_output[1] = op1_base_ap = 1 - op1_base_fp

    // Constraint: mem1_base = op1_base_fp * input_fp + op1_base_ap * input_ap
    m31 mem1_base_expected = add(
        mul(op1_base_fp_col4, input_fp_col2),
        mul(decode_output[1], input_ap_col1)
    );
    cuda_evaluator.add_constraint(sub(mem1_base_col6, mem1_base_expected));

    // Read relative offset from [mem1_base + offset2] (ReadSmall)
    m31 read_small_output[2] = {0};
    evaluate_read_small(
        add(mem1_base_col6, decode_output[0]),  // Address: mem1_base + offset2
        next_pc_id_col7,
        msb_col8,
        mid_limbs_set_col9,
        next_pc_limb_0_col10,
        next_pc_limb_1_col11,
        next_pc_limb_2_col12,
        remainder_bits_col13,
        partial_limb_msb_col14,
        read_small_output,
        jump_opcode_rel_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // read_small_output[0] contains the relative offset
    // next_pc = input_pc + relative_offset

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
            jump_opcode_rel_eval->common_lookup_elements,
            qm31{enabler, 0, 0, 0},  // positive multiplicity
            values
        );
    }

    // Add second opcodes relation entry (negative)
    // next_pc = input_pc + relative_offset
    // next_ap = input_ap + ap_update_add_1
    // next_fp = input_fp (unchanged for jump)
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            add(input_pc_col0, read_small_output[0]),  // next_pc = input_pc + offset
            add(input_ap_col1, ap_update_add_1_col5),
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            jump_opcode_rel_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},  // negative multiplicity
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host wrapper function
extern "C"
void evaluate_jump_opcode_rel(
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

    JumpOpcodeRel_Eval *device_jump_rel_eval = cuda_malloc<JumpOpcodeRel_Eval>(1);
    cuda_mem_copy_host_to_device<JumpOpcodeRel_Eval>((JumpOpcodeRel_Eval*)eval, device_jump_rel_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_jump_opcode_rel");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_jump_opcode_rel_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jump_rel_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_jump_opcode_rel_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jump_rel_eval,
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
    global_timer.end("evaluate_jump_opcode_rel");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_jump_rel_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
