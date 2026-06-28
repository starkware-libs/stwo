#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_qm_31_add_mul_opcode.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_common.cuh"

// Helper: inline implementation of Qm31ReadReduced (CPU subroutine)
template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_qm31_read_reduced(
    EvaluatorT* cuda_evaluator,
    Qm31AddMulOpcode_Eval* qm31_eval,
    m31 base_addr,
    m31 id_col0,
    m31 value_limb_0_col1,
    m31 value_limb_1_col2,
    m31 value_limb_2_col3,
    m31 value_limb_3_col4,
    m31 value_limb_4_col5,
    m31 value_limb_5_col6,
    m31 value_limb_6_col7,
    m31 value_limb_7_col8,
    m31 value_limb_8_col9,
    m31 value_limb_9_col10,
    m31 value_limb_10_col11,
    m31 value_limb_11_col12,
    m31 value_limb_12_col13,
    m31 value_limb_13_col14,
    m31 value_limb_14_col15,
    m31 value_limb_15_col16,
    m31 delta_ab_inv_col17,
    m31 delta_cd_inv_col18,
    m31 *out_limbs // len 4
) {
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_512 = m31(512);
    const m31 M31_262144 = m31(262144);
    const m31 M31_134217728 = m31(134217728);
    const m31 M31_1548 = m31(1548);

    // ReadId: MemoryAddressToId
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, base_addr, id_col0};
        cuda_evaluator->add_to_relation<3>(
            qm31_eval->common_lookup_elements,
            qm31{{1, 0}, {0, 0}},
            values
        );
    }

    // ReadPositiveKnownIdNumBits144: MemoryIdToBig (id + 16 limbs + zeros)
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col1,  value_limb_1_col2,  value_limb_2_col3,  value_limb_3_col4,
            value_limb_4_col5,  value_limb_5_col6,  value_limb_6_col7,  value_limb_7_col8,
            value_limb_8_col9,  value_limb_9_col10, value_limb_10_col11, value_limb_11_col12,
            value_limb_12_col13, value_limb_13_col14, value_limb_14_col15, value_limb_15_col16,
            M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0
        };
        cuda_evaluator->add_to_relation<30>(
            qm31_eval->common_lookup_elements,
            qm31{{1, 0}, {0, 0}},
            values
        );
    }

    // RangeCheck_4_4_4_4
    {
        m31 values[5] = {
            RANGE_CHECK_4_4_4_4_RELATION_ID,
            value_limb_3_col4,
            value_limb_7_col8,
            value_limb_11_col12,
            value_limb_15_col16
        };
        cuda_evaluator->add_to_relation<5>(
            qm31_eval->common_lookup_elements,
            qm31{{1, 0}, {0, 0}},
            values
        );
    }

    // delta_ab != 0
    {
        m31 sum_ab1 = add(
            add(add(value_limb_0_col1, value_limb_1_col2), value_limb_2_col3),
            value_limb_3_col4
        );
        sum_ab1 = sub(sum_ab1, M31_1548);
        m31 sum_ab2 = add(
            add(add(value_limb_4_col5, value_limb_5_col6), value_limb_6_col7),
            value_limb_7_col8
        );
        sum_ab2 = sub(sum_ab2, M31_1548);
        m31 prod = mul(mul(sum_ab1, sum_ab2), delta_ab_inv_col17);
        cuda_evaluator->add_constraint(sub(prod, M31_1));
    }

    // delta_cd != 0
    {
        m31 sum_cd1 = add(
            add(add(value_limb_8_col9, value_limb_9_col10), value_limb_10_col11),
            value_limb_11_col12
        );
        sum_cd1 = sub(sum_cd1, M31_1548);
        m31 sum_cd2 = add(
            add(add(value_limb_12_col13, value_limb_13_col14), value_limb_14_col15),
            value_limb_15_col16
        );
        sum_cd2 = sub(sum_cd2, M31_1548);
        m31 prod = mul(mul(sum_cd1, sum_cd2), delta_cd_inv_col18);
        cuda_evaluator->add_constraint(sub(prod, M31_1));
    }

    // Outputs: packed QM31 coordinates
    out_limbs[0] = add(
        add(value_limb_0_col1, mul(value_limb_1_col2, M31_512)),
        add(mul(value_limb_2_col3, M31_262144), mul(value_limb_3_col4, M31_134217728))
    );
    out_limbs[1] = add(
        add(value_limb_4_col5, mul(value_limb_5_col6, M31_512)),
        add(mul(value_limb_6_col7, M31_262144), mul(value_limb_7_col8, M31_134217728))
    );
    out_limbs[2] = add(
        add(value_limb_8_col9, mul(value_limb_9_col10, M31_512)),
        add(mul(value_limb_10_col11, M31_262144), mul(value_limb_11_col12, M31_134217728))
    );
    out_limbs[3] = add(
        add(value_limb_12_col13, mul(value_limb_13_col14, M31_512)),
        add(mul(value_limb_14_col15, M31_262144), mul(value_limb_15_col16, M31_134217728))
    );
}

#define QM31_ADD_MUL_OPCODE_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_qm_31_add_mul_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    Qm31AddMulOpcode_Eval *qm31_eval,
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
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // Constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);

    // Trace columns (73)
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset0_col3 = cuda_evaluator.next_trace_mask();
    m31 offset1_col4 = cuda_evaluator.next_trace_mask();
    m31 offset2_col5 = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col6 = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp_col7 = cuda_evaluator.next_trace_mask();
    m31 op1_imm_col8 = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp_col9 = cuda_evaluator.next_trace_mask();
    m31 res_add_col10 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col11 = cuda_evaluator.next_trace_mask();
    m31 mem_dst_base_col12 = cuda_evaluator.next_trace_mask();
    m31 mem0_base_col13 = cuda_evaluator.next_trace_mask();
    m31 mem1_base_col14 = cuda_evaluator.next_trace_mask();
    m31 dst_id_col15 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_0_col16 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_1_col17 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_2_col18 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_3_col19 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_4_col20 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_5_col21 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_6_col22 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_7_col23 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_8_col24 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_9_col25 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_10_col26 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_11_col27 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_12_col28 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_13_col29 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_14_col30 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_15_col31 = cuda_evaluator.next_trace_mask();
    m31 dst_delta_ab_inv_col32 = cuda_evaluator.next_trace_mask();
    m31 dst_delta_cd_inv_col33 = cuda_evaluator.next_trace_mask();
    m31 op0_id_col34 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_0_col35 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_1_col36 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_2_col37 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_3_col38 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_4_col39 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_5_col40 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_6_col41 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_7_col42 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_8_col43 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_9_col44 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_10_col45 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_11_col46 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_12_col47 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_13_col48 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_14_col49 = cuda_evaluator.next_trace_mask();
    m31 op0_limb_15_col50 = cuda_evaluator.next_trace_mask();
    m31 op0_delta_ab_inv_col51 = cuda_evaluator.next_trace_mask();
    m31 op0_delta_cd_inv_col52 = cuda_evaluator.next_trace_mask();
    m31 op1_id_col53 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_0_col54 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_1_col55 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_2_col56 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_3_col57 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_4_col58 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_5_col59 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_6_col60 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_7_col61 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_8_col62 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_9_col63 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_10_col64 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_11_col65 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_12_col66 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_13_col67 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_14_col68 = cuda_evaluator.next_trace_mask();
    m31 op1_limb_15_col69 = cuda_evaluator.next_trace_mask();
    m31 op1_delta_ab_inv_col70 = cuda_evaluator.next_trace_mask();
    m31 op1_delta_cd_inv_col71 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // DecodeInstruction3802D
    m31 decode_out[19] = {0};
    evaluate_decode_instruction_3802d(
        input_pc_col0,
        offset0_col3,
        offset1_col4,
        offset2_col5,
        dst_base_fp_col6,
        op0_base_fp_col7,
        op1_imm_col8,
        op1_base_fp_col9,
        res_add_col10,
        ap_update_add_1_col11,
        decode_out,
        qm31_eval->common_lookup_elements,
        &cuda_evaluator
    );

    m31 decode_offset0 = decode_out[0];
    m31 decode_offset1 = decode_out[1];
    m31 decode_offset2 = decode_out[2];
    m31 decode_op1_base_ap = decode_out[7];
    m31 decode_res_mul = decode_out[9];

    // Either flag op1_imm is off or offset2 is equal to 1.
    cuda_evaluator.add_constraint(
        mul(op1_imm_col8, sub(decode_offset2, M31_1))
    );

    // mem_dst_base.
    cuda_evaluator.add_constraint(
        sub(
            mem_dst_base_col12,
            add(
                mul(dst_base_fp_col6, input_fp_col2),
                mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)
            )
        )
    );

    // mem0_base.
    cuda_evaluator.add_constraint(
        sub(
            mem0_base_col13,
            add(
                mul(op0_base_fp_col7, input_fp_col2),
                mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)
            )
        )
    );

    // mem1_base.
    cuda_evaluator.add_constraint(
        sub(
            mem1_base_col14,
            add(
                add(
                    mul(op1_base_fp_col9, input_fp_col2),
                    mul(decode_op1_base_ap, input_ap_col1)
                ),
                mul(op1_imm_col8, input_pc_col0)
            )
        )
    );

    // Qm31ReadReduced for dst, op0, op1
    m31 dst_qm31[4];
    evaluate_qm31_read_reduced(
        &cuda_evaluator,
        qm31_eval,
        add(mem_dst_base_col12, decode_offset0),
        dst_id_col15,
        dst_limb_0_col16,
        dst_limb_1_col17,
        dst_limb_2_col18,
        dst_limb_3_col19,
        dst_limb_4_col20,
        dst_limb_5_col21,
        dst_limb_6_col22,
        dst_limb_7_col23,
        dst_limb_8_col24,
        dst_limb_9_col25,
        dst_limb_10_col26,
        dst_limb_11_col27,
        dst_limb_12_col28,
        dst_limb_13_col29,
        dst_limb_14_col30,
        dst_limb_15_col31,
        dst_delta_ab_inv_col32,
        dst_delta_cd_inv_col33,
        dst_qm31
    );

    m31 op0_qm31[4];
    evaluate_qm31_read_reduced(
        &cuda_evaluator,
        qm31_eval,
        add(mem0_base_col13, decode_offset1),
        op0_id_col34,
        op0_limb_0_col35,
        op0_limb_1_col36,
        op0_limb_2_col37,
        op0_limb_3_col38,
        op0_limb_4_col39,
        op0_limb_5_col40,
        op0_limb_6_col41,
        op0_limb_7_col42,
        op0_limb_8_col43,
        op0_limb_9_col44,
        op0_limb_10_col45,
        op0_limb_11_col46,
        op0_limb_12_col47,
        op0_limb_13_col48,
        op0_limb_14_col49,
        op0_limb_15_col50,
        op0_delta_ab_inv_col51,
        op0_delta_cd_inv_col52,
        op0_qm31
    );

    m31 op1_qm31[4];
    evaluate_qm31_read_reduced(
        &cuda_evaluator,
        qm31_eval,
        add(mem1_base_col14, decode_offset2),
        op1_id_col53,
        op1_limb_0_col54,
        op1_limb_1_col55,
        op1_limb_2_col56,
        op1_limb_3_col57,
        op1_limb_4_col58,
        op1_limb_5_col59,
        op1_limb_6_col60,
        op1_limb_7_col61,
        op1_limb_8_col62,
        op1_limb_9_col63,
        op1_limb_10_col64,
        op1_limb_11_col65,
        op1_limb_12_col66,
        op1_limb_13_col67,
        op1_limb_14_col68,
        op1_limb_15_col69,
        op1_delta_ab_inv_col70,
        op1_delta_cd_inv_col71,
        op1_qm31
    );

    // dst equals (op0 * op1)*flag_res_mul + (op0 + op1)*(1-flag_res_mul).
    // Coordinates are in QM31; multiplication is implemented via explicit formulas.

    // Coordinate 0
    {
        m31 term0 = mul(op0_qm31[0], op1_qm31[0]);
        m31 term1 = mul(op0_qm31[1], op1_qm31[1]);
        m31 term2 = mul(op0_qm31[2], op1_qm31[2]);
        m31 term3 = mul(op0_qm31[3], op1_qm31[3]);
        m31 t = sub(term0, term1);
        t = add(t, mul(M31_2, sub(term2, term3)));
        t = sub(t, mul(op0_qm31[2], op1_qm31[3]));
        t = sub(t, mul(op0_qm31[3], op1_qm31[2]));
        m31 mul_part = mul(t, decode_res_mul);
        m31 add_part = mul(add(op0_qm31[0], op1_qm31[0]), res_add_col10);
        cuda_evaluator.add_constraint(
            sub(sub(dst_qm31[0], mul_part), add_part)
        );
    }

    // Coordinate 1
    {
        m31 term0 = mul(op0_qm31[0], op1_qm31[1]);
        m31 term1 = mul(op0_qm31[1], op1_qm31[0]);
        m31 term2 = mul(op0_qm31[2], op1_qm31[3]);
        m31 term3 = mul(op0_qm31[3], op1_qm31[2]);
        m31 term4 = mul(op0_qm31[2], op1_qm31[2]);
        m31 term5 = mul(op0_qm31[3], op1_qm31[3]);
        m31 t = add(term0, term1);
        t = add(t, mul(M31_2, add(term2, term3)));
        t = add(t, term4);
        t = sub(t, term5);
        m31 mul_part = mul(t, decode_res_mul);
        m31 add_part = mul(add(op0_qm31[1], op1_qm31[1]), res_add_col10);
        cuda_evaluator.add_constraint(
            sub(sub(dst_qm31[1], mul_part), add_part)
        );
    }

    // Coordinate 2
    {
        m31 term0 = mul(op0_qm31[0], op1_qm31[2]);
        m31 term1 = mul(op0_qm31[1], op1_qm31[3]);
        m31 term2 = mul(op0_qm31[2], op1_qm31[0]);
        m31 term3 = mul(op0_qm31[3], op1_qm31[1]);
        m31 t = sub(term0, term1);
        t = add(t, term2);
        t = sub(t, term3);
        m31 mul_part = mul(t, decode_res_mul);
        m31 add_part = mul(add(op0_qm31[2], op1_qm31[2]), res_add_col10);
        cuda_evaluator.add_constraint(
            sub(sub(dst_qm31[2], mul_part), add_part)
        );
    }

    // Coordinate 3
    {
        m31 term0 = mul(op0_qm31[0], op1_qm31[3]);
        m31 term1 = mul(op0_qm31[1], op1_qm31[2]);
        m31 term2 = mul(op0_qm31[2], op1_qm31[1]);
        m31 term3 = mul(op0_qm31[3], op1_qm31[0]);
        m31 t = add(term0, term1);
        t = add(t, term2);
        t = add(t, term3);
        m31 mul_part = mul(t, decode_res_mul);
        m31 add_part = mul(add(op0_qm31[3], op1_qm31[3]), res_add_col10);
        cuda_evaluator.add_constraint(
            sub(sub(dst_qm31[3], mul_part), add_part)
        );
    }

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Opcodes relation entries
    {
        m31 values[4] = {OPCODES_RELATION_ID, input_pc_col0, input_ap_col1, input_fp_col2};
        cuda_evaluator.add_to_relation<4>(
            qm31_eval->common_lookup_elements,
            qm31{{enabler, 0}, {0, 0}},
            values
        );
    }

    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            add(add(input_pc_col0, M31_1), op1_imm_col8),
            add(input_ap_col1, ap_update_add_1_col11),
            input_fp_col2
        };
        qm31 minus_enabler = (enabler == 0) ? qm31{{0, 0}, {0, 0}}
                                            : qm31{{sub(P, enabler), 0}, {0, 0}};
        cuda_evaluator.add_to_relation<4>(
            qm31_eval->common_lookup_elements,
            minus_enabler,
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

extern "C"
void evaluate_qm_31_add_mul_opcode(
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
    (void)should_accumulate; // behavior controlled by global flag in finalize kernel

    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    Qm31AddMulOpcode_Eval *device_eval = cuda_malloc<Qm31AddMulOpcode_Eval>(1);
    cuda_mem_copy_host_to_device<Qm31AddMulOpcode_Eval>((Qm31AddMulOpcode_Eval*)eval, device_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_qm_31_add_mul_opcode");

    int block_dim = eval_domain_size < QM31_ADD_MUL_OPCODE_THREAD_COUNT_MAX
        ? eval_domain_size
        : QM31_ADD_MUL_OPCODE_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_qm_31_add_mul_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_qm_31_add_mul_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_eval,
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
    global_timer.end("evaluate_qm_31_add_mul_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
