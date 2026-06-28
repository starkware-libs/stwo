#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_jump_opcode_double_deref.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_common.cuh"

#define JUMP_OPCODE_DOUBLE_DEREF_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_jump_opcode_double_deref_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    JumpOpcodeDoubleDeref_Eval *jump_opcode_double_deref_eval,
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

    // Load all 21 trace columns
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset1_col3 = cuda_evaluator.next_trace_mask();
    m31 offset2_col4 = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp_col5 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col6 = cuda_evaluator.next_trace_mask();
    m31 mem0_base_col7 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_id_col8 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_0_col9 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_1_col10 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_2_col11 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_limb_3_col12 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col13 = cuda_evaluator.next_trace_mask();
    m31 next_pc_id_col14 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_0_col15 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_1_col16 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_2_col17 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_3_col18 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col19 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_512 = m31(512);
    const m31 M31_262144 = m31(262144);
    const m31 M31_134217728 = m31(134217728);

    // DecodeInstruction9Bd86
    m31 decode_output[19];
    evaluate_decode_instruction_9bd86(
        input_pc_col0,
        offset1_col3,
        offset2_col4,
        op0_base_fp_col5,
        ap_update_add_1_col6,
        decode_output,
        jump_opcode_double_deref_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Constraint: mem0_base = op0_base_fp * input_fp + (1 - op0_base_fp) * input_ap
    m31 mem0_base_expected = add(
        mul(op0_base_fp_col5, input_fp_col2),
        mul(sub(M31_1, op0_base_fp_col5), input_ap_col1)
    );
    cuda_evaluator.add_constraint(sub(mem0_base_col7, mem0_base_expected));

    // Read mem1_base from [mem0_base + offset1] (ReadPositiveNumBits29)
    m31 read_mem1_base_output[29] = {0};
    evaluate_read_positive_num_bits_29(
        add(mem0_base_col7, decode_output[0]),  // Address: mem0_base + offset1
        mem1_base_id_col8,
        mem1_base_limb_0_col9,
        mem1_base_limb_1_col10,
        mem1_base_limb_2_col11,
        mem1_base_limb_3_col12,
        partial_limb_msb_col13,
        read_mem1_base_output,
        jump_opcode_double_deref_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Reconstruct mem1_base from limbs
    m31 mem1_base_reconstructed = add(
        add(
            mem1_base_limb_0_col9,
            mul(mem1_base_limb_1_col10, M31_512)
        ),
        add(
            mul(mem1_base_limb_2_col11, M31_262144),
            mul(mem1_base_limb_3_col12, M31_134217728)
        )
    );

    // Read next_pc from [mem1_base + offset2] (ReadPositiveNumBits29)
    m31 read_next_pc_output[29] = {0};
    evaluate_read_positive_num_bits_29(
        add(mem1_base_reconstructed, decode_output[1]),  // Address: mem1_base + offset2
        next_pc_id_col14,
        next_pc_limb_0_col15,
        next_pc_limb_1_col16,
        next_pc_limb_2_col17,
        next_pc_limb_3_col18,
        partial_limb_msb_col19,
        read_next_pc_output,
        jump_opcode_double_deref_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Reconstruct next_pc from limbs
    m31 next_pc_reconstructed = add(
        add(
            next_pc_limb_0_col15,
            mul(next_pc_limb_1_col16, M31_512)
        ),
        add(
            mul(next_pc_limb_2_col17, M31_262144),
            mul(next_pc_limb_3_col18, M31_134217728)
        )
    );

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
            jump_opcode_double_deref_eval->common_lookup_elements,
            qm31{enabler, 0, 0, 0},  // positive multiplicity
            values
        );
    }

    // Add second opcodes relation entry (negative)
    // next_ap = input_ap + ap_update_add_1
    // next_fp = input_fp (unchanged for jump)
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            next_pc_reconstructed,
            add(input_ap_col1, ap_update_add_1_col6),
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            jump_opcode_double_deref_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},  // negative multiplicity
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host wrapper function
extern "C"
void evaluate_jump_opcode_double_deref(
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

    JumpOpcodeDoubleDeref_Eval *device_jump_double_deref_eval = cuda_malloc<JumpOpcodeDoubleDeref_Eval>(1);
    cuda_mem_copy_host_to_device<JumpOpcodeDoubleDeref_Eval>((JumpOpcodeDoubleDeref_Eval*)eval, device_jump_double_deref_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_jump_opcode_double_deref");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_jump_opcode_double_deref_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jump_double_deref_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_jump_opcode_double_deref_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jump_double_deref_eval,
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
    global_timer.end("evaluate_jump_opcode_double_deref");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_jump_double_deref_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
