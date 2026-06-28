// mul_opcode CUDA AIR Evaluator
// This evaluator handles the multiplication opcode constraints for Cairo VM
// 130 trace columns, complex 252-bit multiplication verification with Karatsuba algorithm

#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_mul_opcode.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_common.cuh"
#include "../builtin/verify_mul_252.cuh"

// =====================================================================
// Pre-Kernel: Read trace columns, evaluate constraints, build logup fractions
// =====================================================================
template<typename EvaluatorT>
__global__ void evaluate_mul_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    MulOpcode_Eval *mul_eval,
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

    // Read all 130 trace columns (trace1)
    // Columns 0-2: Input state (PC, AP, FP)
    m31 input_pc = cuda_evaluator.next_trace_mask();
    m31 input_ap = cuda_evaluator.next_trace_mask();
    m31 input_fp = cuda_evaluator.next_trace_mask();

    // Columns 3-10: Decode flags and offsets
    m31 offset0 = cuda_evaluator.next_trace_mask();
    m31 offset1 = cuda_evaluator.next_trace_mask();
    m31 offset2 = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp = cuda_evaluator.next_trace_mask();
    m31 op1_imm = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1 = cuda_evaluator.next_trace_mask();

    // Columns 11-13: Memory base addresses
    m31 mem_dst_base = cuda_evaluator.next_trace_mask();
    m31 mem0_base = cuda_evaluator.next_trace_mask();
    m31 mem1_base = cuda_evaluator.next_trace_mask();

    // Columns 14-42: dst (28 limbs + id)
    m31 dst_id = cuda_evaluator.next_trace_mask();
    m31 dst_limbs[28];
    for (int i = 0; i < 28; i++) {
        dst_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Columns 43-71: op0 (28 limbs + id)
    m31 op0_id = cuda_evaluator.next_trace_mask();
    m31 op0_limbs[28];
    for (int i = 0; i < 28; i++) {
        op0_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Columns 72-100: op1 (28 limbs + id)
    m31 op1_id = cuda_evaluator.next_trace_mask();
    m31 op1_limbs[28];
    for (int i = 0; i < 28; i++) {
        op1_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Column 101: k (modular reduction parameter)
    m31 k = cuda_evaluator.next_trace_mask();

    // Columns 102-128: carries (27 carry values)
    m31 carries[27];
    for (int i = 0; i < 27; i++) {
        carries[i] = cuda_evaluator.next_trace_mask();
    }

    // Column 129: enabler
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);

    // Call DecodeInstruction4B8Cf subroutine
    m31 decode_outputs[19];
    evaluate_decode_instruction_4b8cf(
        input_pc,
        offset0, offset1, offset2,
        dst_base_fp, op0_base_fp,
        op1_imm, op1_base_fp,
        ap_update_add_1,
        decode_outputs,
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 decode_offset0 = decode_outputs[0];
    m31 decode_offset1 = decode_outputs[1];
    m31 decode_offset2 = decode_outputs[2];
    // Note: CUDA decode function stores op1_base_ap at index 7
    // (1 - op1_imm) - op1_base_fp = op1_base_ap
    m31 decode_op1_base_ap = decode_outputs[7];

    // Constraint 1: if imm then offset2 is 1
    cuda_evaluator.add_constraint(mul(op1_imm, sub(M31_1, decode_offset2)));

    // Constraint 2: mem_dst_base
    cuda_evaluator.add_constraint(
        sub(mem_dst_base, add(mul(dst_base_fp, input_fp), mul(sub(M31_1, dst_base_fp), input_ap)))
    );

    // Constraint 3: mem0_base
    cuda_evaluator.add_constraint(
        sub(mem0_base, add(mul(op0_base_fp, input_fp), mul(sub(M31_1, op0_base_fp), input_ap)))
    );

    // Constraint 4: mem1_base
    cuda_evaluator.add_constraint(
        sub(mem1_base, add(add(mul(op1_imm, input_pc), mul(op1_base_fp, input_fp)), mul(decode_op1_base_ap, input_ap)))
    );

    // Call ReadPositiveNumBits252 for dst
    m31 dst_output[29];
    evaluate_read_positive_num_bits_252(
        add(mem_dst_base, decode_offset0),
        dst_id,
        dst_limbs[0], dst_limbs[1], dst_limbs[2], dst_limbs[3],
        dst_limbs[4], dst_limbs[5], dst_limbs[6], dst_limbs[7],
        dst_limbs[8], dst_limbs[9], dst_limbs[10], dst_limbs[11],
        dst_limbs[12], dst_limbs[13], dst_limbs[14], dst_limbs[15],
        dst_limbs[16], dst_limbs[17], dst_limbs[18], dst_limbs[19],
        dst_limbs[20], dst_limbs[21], dst_limbs[22], dst_limbs[23],
        dst_limbs[24], dst_limbs[25], dst_limbs[26], dst_limbs[27],
        dst_output,
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Call ReadPositiveNumBits252 for op0
    m31 op0_output[29];
    evaluate_read_positive_num_bits_252(
        add(mem0_base, decode_offset1),
        op0_id,
        op0_limbs[0], op0_limbs[1], op0_limbs[2], op0_limbs[3],
        op0_limbs[4], op0_limbs[5], op0_limbs[6], op0_limbs[7],
        op0_limbs[8], op0_limbs[9], op0_limbs[10], op0_limbs[11],
        op0_limbs[12], op0_limbs[13], op0_limbs[14], op0_limbs[15],
        op0_limbs[16], op0_limbs[17], op0_limbs[18], op0_limbs[19],
        op0_limbs[20], op0_limbs[21], op0_limbs[22], op0_limbs[23],
        op0_limbs[24], op0_limbs[25], op0_limbs[26], op0_limbs[27],
        op0_output,
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Call ReadPositiveNumBits252 for op1
    m31 op1_output[29];
    evaluate_read_positive_num_bits_252(
        add(mem1_base, decode_offset2),
        op1_id,
        op1_limbs[0], op1_limbs[1], op1_limbs[2], op1_limbs[3],
        op1_limbs[4], op1_limbs[5], op1_limbs[6], op1_limbs[7],
        op1_limbs[8], op1_limbs[9], op1_limbs[10], op1_limbs[11],
        op1_limbs[12], op1_limbs[13], op1_limbs[14], op1_limbs[15],
        op1_limbs[16], op1_limbs[17], op1_limbs[18], op1_limbs[19],
        op1_limbs[20], op1_limbs[21], op1_limbs[22], op1_limbs[23],
        op1_limbs[24], op1_limbs[25], op1_limbs[26], op1_limbs[27],
        op1_output,
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Call VerifyMul252: verify op0 * op1 = dst (mod Cairo prime)
    // Uses the shared implementation from builtin/verify_mul_252.cuh
    verify_mul_252_evaluate<EvaluatorT>(
        op0_limbs,
        op1_limbs,
        dst_limbs,
        k,
        carries,
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Add opcodes relation entries (state transition)
    // Forward entry: (input_pc, input_ap, input_fp) with multiplicity +enabler
    {
        m31 values[4] = {OPCODES_RELATION_ID, input_pc, input_ap, input_fp};
        cuda_evaluator.add_to_relation<4>(
            mul_eval->common_lookup_elements,
            qm31{enabler, M31_0},
            values
        );
    }

    // Backward entry: (next_pc, next_ap, input_fp) with multiplicity -enabler
    m31 next_pc = add(add(input_pc, M31_1), op1_imm);
    m31 next_ap = add(input_ap, ap_update_add_1);
    {
        m31 values[4] = {OPCODES_RELATION_ID, next_pc, next_ap, input_fp};
        cuda_evaluator.add_to_relation<4>(
            mul_eval->common_lookup_elements,
            sub(qm31{{M31_0, M31_0}, {M31_0, M31_0}}, qm31{enabler, M31_0}),
            values
        );
    }

    // Store constraint index
    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
    // numerators[row] = cuda_evaluator.numerator;
}

// =====================================================================
// Host Wrapper Function
// =====================================================================
extern "C"
void evaluate_mul_opcode(
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

    MulOpcode_Eval *device_mul_eval = cuda_malloc<MulOpcode_Eval>(1);
    cuda_mem_copy_host_to_device<MulOpcode_Eval>((MulOpcode_Eval*)eval, device_mul_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_mul_opcode");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_mul_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_mul_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_mul_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_mul_eval,
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
    global_timer.end("evaluate_mul_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_mul_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
