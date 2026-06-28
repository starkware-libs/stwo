// mul_opcode_small CUDA AIR Evaluator
// This evaluator handles the small multiplication opcode constraints for Cairo VM
// 37 trace columns, simple 36-bit multiplication verification (factors < 2^36)
// Result is 72 bits, much simpler than full 252-bit multiplication

#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_mul_opcode_small.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_common.cuh"

// =====================================================================
// Helper: VerifyMulSmall
// Verifies that op0 * op1 = dst for 36-bit factors
// op0, op1 are 36 bits (4 limbs of 9 bits each)
// dst is 72 bits (8 limbs of 9 bits each)
// Uses 3 carry values for verification
// =====================================================================
template<typename EvaluatorT>
__device__ __forceinline__ void VerifyMulSmall(
    EvaluatorT* eval,
    m31 op0_limbs[4],  // 4 limbs × 9 bits = 36 bits
    m31 op1_limbs[4],  // 4 limbs × 9 bits = 36 bits
    m31 dst_limbs[8],  // 8 limbs × 9 bits = 72 bits
    m31 carries[3],    // 3 carry values
    MulOpcodeSmall_Eval* mul_eval
) {
    // Constants
    const m31 M31_0_local = m31(0);
    const m31 M31_1_local = m31(1);
    const m31 M31_512_local = m31(512);
    const m31 M31_262144_local = m31(262144);

    // Multiplication expansion (schoolbook):
    // op0 * op1 = (op0[0] + op0[1]*2^9 + op0[2]*2^18 + op0[3]*2^27) *
    //             (op1[0] + op1[1]*2^9 + op1[2]*2^18 + op1[3]*2^27)
    //
    // Grouping by powers of 2^9 (limb positions):
    // Position 0: op0[0] * op1[0]
    // Position 1: op0[0] * op1[1] + op0[1] * op1[0]
    // Position 2: op0[0] * op1[2] + op0[1] * op1[1] + op0[2] * op1[0]
    // Position 3: op0[0] * op1[3] + op0[1] * op1[2] + op0[2] * op1[1] + op0[3] * op1[0]
    // Position 4: op0[1] * op1[3] + op0[2] * op1[2] + op0[3] * op1[1]
    // Position 5: op0[2] * op1[3] + op0[3] * op1[2]
    // Position 6: op0[3] * op1[3]
    // Position 7: 0 (overflow prevention)

    // Carry chain verification:
    // carry_1 * 8192 = (op0[0]*op1[0] - dst[0]) +
    //                   (op0[0]*op1[1] + op0[1]*op1[0] - dst[1]) * 512
    //
    // carry_3 * 8192 = carry_1 +
    //                   (op0[0]*op1[2] + op0[1]*op1[1] + op0[2]*op1[0] - dst[2]) +
    //                   (op0[0]*op1[3] + op0[1]*op1[2] + op0[2]*op1[1] + op0[3]*op1[0] - dst[3]) * 512
    //
    // carry_5 * 8192 = carry_3 +
    //                   (op0[1]*op1[3] + op0[2]*op1[2] + op0[3]*op1[1] - dst[4]) +
    //                   (op0[2]*op1[3] + op0[3]*op1[2] - dst[5]) * 512
    //
    // Final constraint:
    // carry_5 + (op0[3]*op1[3] - dst[6]) + (-dst[7]) * 512 = 0

    // Constraint for carry_1
    m31 term0 = sub(mul(op0_limbs[0], op1_limbs[0]), dst_limbs[0]);
    m31 term1 = sub(add(mul(op0_limbs[0], op1_limbs[1]), mul(op0_limbs[1], op1_limbs[0])), dst_limbs[1]);
    m31 carry_1_lhs = mul(carries[0], M31_262144_local);
    m31 carry_1_rhs = add(term0, mul(term1, M31_512_local));

    eval->add_constraint(sub(carry_1_lhs, carry_1_rhs));

    // Range check carry_1
    {
        m31 values[2] = {RANGE_CHECK_11_RELATION_ID, carries[0]};
        eval->add_to_relation<2>(
            mul_eval->common_lookup_elements,
            qm31{M31_1_local, M31_0_local},
            values
        );
    }

    // Constraint for carry_3
    m31 term2 = sub(add(add(mul(op0_limbs[0], op1_limbs[2]),
                 mul(op0_limbs[1], op1_limbs[1])),
                 mul(op0_limbs[2], op1_limbs[0])), dst_limbs[2]);
    m31 term3 = sub(add(add(add(mul(op0_limbs[0], op1_limbs[3]),
                 mul(op0_limbs[1], op1_limbs[2])),
                 mul(op0_limbs[2], op1_limbs[1])),
                 mul(op0_limbs[3], op1_limbs[0])), dst_limbs[3]);
    m31 carry_3_lhs = mul(carries[1], M31_262144_local);
    m31 carry_3_rhs = add(add(carries[0], term2), mul(term3, M31_512_local));

    eval->add_constraint(sub(carry_3_lhs, carry_3_rhs));

    // Range check carry_3
    {
        m31 values[2] = {RANGE_CHECK_11_RELATION_ID, carries[1]};
        eval->add_to_relation<2>(
            mul_eval->common_lookup_elements,
            qm31{M31_1_local, M31_0_local},
            values
        );
    }

    // Constraint for carry_5
    m31 term4 = sub(add(add(mul(op0_limbs[1], op1_limbs[3]),
                 mul(op0_limbs[2], op1_limbs[2])),
                 mul(op0_limbs[3], op1_limbs[1])), dst_limbs[4]);
    m31 term5 = sub(add(mul(op0_limbs[2], op1_limbs[3]),
                 mul(op0_limbs[3], op1_limbs[2])), dst_limbs[5]);
    m31 carry_5_lhs = mul(carries[2], M31_262144_local);
    m31 carry_5_rhs = add(add(carries[1], term4), mul(term5, M31_512_local));

    eval->add_constraint(sub(carry_5_lhs, carry_5_rhs));

    // Range check carry_5
    {
        m31 values[2] = {RANGE_CHECK_11_RELATION_ID, carries[2]};
        eval->add_to_relation<2>(
            mul_eval->common_lookup_elements,
            qm31{M31_1_local, M31_0_local},
            values
        );
    }

    // Final constraint: carry_5 + (op0[3]*op1[3] - dst[6]) + (-dst[7]) * 512 = 0
    m31 term6 = sub(mul(op0_limbs[3], op1_limbs[3]), dst_limbs[6]);
    m31 term7 = sub(m31(0), dst_limbs[7]);
    m31 final_constraint = add(add(carries[2], term6), mul(term7, M31_512_local));

    eval->add_constraint(final_constraint);
}

// =====================================================================
// Pre-Kernel: Read trace columns, evaluate constraints, build logup fractions
// =====================================================================
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_mul_opcode_small_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    MulOpcodeSmall_Eval *mul_eval,
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
        {{0,0},{0,0}},
        0,
        {{0,0},{0,0}},
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // Read all 37 trace columns (trace1)
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

    // Columns 14-22: dst (8 limbs + id)
    m31 dst_id = cuda_evaluator.next_trace_mask();
    m31 dst_limbs[8];
    for (int i = 0; i < 8; i++) {
        dst_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Columns 23-27: op0 (4 limbs + id)
    m31 op0_id = cuda_evaluator.next_trace_mask();
    m31 op0_limbs[4];
    for (int i = 0; i < 4; i++) {
        op0_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Columns 28-32: op1 (4 limbs + id)
    m31 op1_id = cuda_evaluator.next_trace_mask();
    m31 op1_limbs[4];
    for (int i = 0; i < 4; i++) {
        op1_limbs[i] = cuda_evaluator.next_trace_mask();
    }

    // Columns 33-35: carries (3 carry values)
    m31 carries[3];
    for (int i = 0; i < 3; i++) {
        carries[i] = cuda_evaluator.next_trace_mask();
    }

    // Column 36: enabler
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Define constants
    const m31 M31_0_local = m31(0);
    const m31 M31_1_local = m31(1);

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
    m31 decode_op1_base_ap = decode_outputs[7];

    // Add memory read lookups for dst (72 bits)
    evaluate_read_positive_num_bits_72(
        add(mem_dst_base, decode_offset0),  // address
        dst_id,
        dst_limbs[0], dst_limbs[1], dst_limbs[2], dst_limbs[3],
        dst_limbs[4], dst_limbs[5], dst_limbs[6], dst_limbs[7],
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Add memory read lookups for op0 (36 bits)
    evaluate_read_positive_num_bits_36(
        add(mem0_base, decode_offset1),  // address
        op0_id,
        op0_limbs[0], op0_limbs[1], op0_limbs[2], op0_limbs[3],
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Add memory read lookups for op1 (36 bits)
    evaluate_read_positive_num_bits_36(
        add(mem1_base, decode_offset2),  // address
        op1_id,
        op1_limbs[0], op1_limbs[1], op1_limbs[2], op1_limbs[3],
        mul_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Constraint 1: if imm then offset2 is 1
    cuda_evaluator.add_constraint(mul(op1_imm, sub(M31_1_local, decode_offset2)));

    // Constraint 2: mem_dst_base
    cuda_evaluator.add_constraint(
        sub(mem_dst_base, add(mul(dst_base_fp, input_fp), mul(sub(M31_1_local, dst_base_fp), input_ap)))
    );

    // Constraint 3: mem0_base
    cuda_evaluator.add_constraint(
        sub(mem0_base, add(mul(op0_base_fp, input_fp), mul(sub(M31_1_local, op0_base_fp), input_ap)))
    );

    // Constraint 4: mem1_base
    cuda_evaluator.add_constraint(
        sub(mem1_base, add(add(mul(op1_imm, input_pc), mul(op1_base_fp, input_fp)), mul(decode_op1_base_ap, input_ap)))
    );

    // Call VerifyMulSmall: verify op0 * op1 = dst for 36-bit factors
    VerifyMulSmall(
        &cuda_evaluator,
        op0_limbs,
        op1_limbs,
        dst_limbs,
        carries,
        mul_eval
    );

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Add opcodes relation entries (state transition)
    // Forward entry: (input_pc, input_ap, input_fp) with multiplicity +enabler
    {
        m31 values[4] = {OPCODES_RELATION_ID, input_pc, input_ap, input_fp};
        cuda_evaluator.add_to_relation<4>(
            mul_eval->common_lookup_elements,
            qm31{enabler, M31_0_local},
            values
        );
    }

    // Backward entry: (next_pc, next_ap, input_fp) with multiplicity -enabler
    m31 next_pc = add(add(input_pc, M31_1_local), op1_imm);
    m31 next_ap = add(input_ap, ap_update_add_1);
    {
        m31 values[4] = {OPCODES_RELATION_ID, next_pc, next_ap, input_fp};
        cuda_evaluator.add_to_relation<4>(
            mul_eval->common_lookup_elements,
            sub(qm31{{M31_0_local, M31_0_local}, {M31_0_local, M31_0_local}}, qm31{enabler, M31_0_local}),
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
void evaluate_mul_opcode_small(
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

    MulOpcodeSmall_Eval *device_mul_eval = cuda_malloc<MulOpcodeSmall_Eval>(1);
    cuda_mem_copy_host_to_device<MulOpcodeSmall_Eval>((MulOpcodeSmall_Eval*)eval, device_mul_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_mul_opcode_small");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_mul_opcode_small_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        evaluate_mul_opcode_small_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
    global_timer.end("evaluate_mul_opcode_small");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_mul_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
