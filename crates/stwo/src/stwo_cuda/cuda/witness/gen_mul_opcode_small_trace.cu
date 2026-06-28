#include "relations.cuh"

#include <cstdint>
#include <cstdio>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include <stdint.h>

#include "gen_mul_opcode_small_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

__launch_bounds__(256, 2)
__global__ void generate_mul_opcode_small_trace_kernel(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,

    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,

    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    unsigned **lookup_range_check_11_0,
    unsigned **lookup_range_check_11_1,
    unsigned **lookup_range_check_11_2,

    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_11,

    unsigned **mul_opcode_small_inputs,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0 = {0};
    const m31 M31_1 = {1};
    const m31 M31_8 = {8};
    const m31 M31_16 = {16};
    const m31 M31_32 = {32};
    const m31 M31_64 = {64};
    const m31 M31_128 = {128};
    const m31 M31_256 = {256};
    const m31 M31_512 = {512};
    const m31 M31_8192 = {8192};
    const m31 M31_32768 = {32768};

    const uint16_t UInt16_0 = 0;
    const uint16_t UInt16_1 = 1;
    const uint16_t UInt16_2 = 2;
    const uint16_t UInt16_3 = 3;
    const uint16_t UInt16_4 = 4;
    const uint16_t UInt16_5 = 5;
    const uint16_t UInt16_6 = 6;
    const uint16_t UInt16_7 = 7;
    const uint16_t UInt16_9 = 9;
    const uint16_t UInt16_11 = 11;
    const uint16_t UInt16_13 = 13;
    const uint16_t UInt16_31 = 31;
    const uint16_t UInt16_127 = 127;

    if (row < trace_size) {
        // === input ===
        m31 input_pc_col0 = {mul_opcode_small_inputs[0][row]};
        traces[0][row] = input_pc_col0;
        m31 input_ap_col1 = {mul_opcode_small_inputs[1][row]};
        traces[1][row] = input_ap_col1;
        m31 input_fp_col2 = {mul_opcode_small_inputs[2][row]};
        traces[2][row] = input_fp_col2;

        // === Decode Instruction ===
        m31 memory_address_to_id_value_tmp_0 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_pc_col0,
            &memory_address_to_id_value_tmp_0
        );
        m31 memory_id_to_big_value_tmp_1[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_0,
            memory_id_to_big_value_tmp_1
        );

        // offset0
        uint16_t offset0_tmp_2 =
            ((uint16_t)(memory_id_to_big_value_tmp_1[0]))
            + ((((uint16_t)(memory_id_to_big_value_tmp_1[1]) & UInt16_127)) << UInt16_9);
        m31 offset0_col3 = m31{offset0_tmp_2};
        traces[3][row] = offset0_col3;

        // offset1
        uint16_t offset1_tmp_3 =
            (((((uint16_t)(memory_id_to_big_value_tmp_1[1])) >> UInt16_7)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[2])) << UInt16_2))
            + ((((uint16_t)(memory_id_to_big_value_tmp_1[3])) & UInt16_31) << UInt16_11));
        m31 offset1_col4 = m31{offset1_tmp_3};
        traces[4][row] = offset1_col4;

        // offset2
        uint16_t offset2_tmp_4 =
            (((((uint16_t)(memory_id_to_big_value_tmp_1[3])) >> UInt16_5)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[4])) << UInt16_4))
            + ((((uint16_t)(memory_id_to_big_value_tmp_1[5])) & UInt16_7) << UInt16_13));
        m31 offset2_col5 = m31{offset2_tmp_4};
        traces[5][row] = offset2_col5;

        // dst_base_fp
        uint16_t dst_base_fp_tmp_5 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6))
            >> UInt16_0) & UInt16_1);
        m31 dst_base_fp_col6 = m31{dst_base_fp_tmp_5};
        traces[6][row] = dst_base_fp_col6;

        // op0_base_fp
        uint16_t op0_base_fp_tmp_6 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6))
            >> UInt16_1) & UInt16_1);
        m31 op0_base_fp_col7 = m31{op0_base_fp_tmp_6};
        traces[7][row] = op0_base_fp_col7;

        // op1_imm
        uint16_t op1_imm_tmp_7 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6))
            >> UInt16_2) & UInt16_1);
        m31 op1_imm_col8 = m31{op1_imm_tmp_7};
        traces[8][row] = op1_imm_col8;

        // op1_base_fp
        uint16_t op1_base_fp_tmp_8 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6))
            >> UInt16_3) & UInt16_1);
        m31 op1_base_fp_col9 = m31{op1_base_fp_tmp_8};
        traces[9][row] = op1_base_fp_col9;

        // ap_update_add_1
        uint16_t ap_update_add_1_tmp_9 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6))
            >> UInt16_11) & UInt16_1);
        m31 ap_update_add_1_col10 = m31{ap_update_add_1_tmp_9};
        traces[10][row] = ap_update_add_1_col10;

        // op1_base_op0 = (1 - op1_imm) - op1_base_fp
        m31 op1_base_op0 = sub(sub(M31_1, op1_imm_col8), op1_base_fp_col9);

        // === sub_component_inputs.verify_instruction ===
        sub_component_inputs_verify_instruction[0][row] = input_pc_col0;
        sub_component_inputs_verify_instruction[1][row] = offset0_col3;
        sub_component_inputs_verify_instruction[2][row] = offset1_col4;
        sub_component_inputs_verify_instruction[3][row] = offset2_col5;
        m31 flags_a = add(
            add(
                add(
                    add(
                        mul(dst_base_fp_col6, M31_8),
                        mul(op0_base_fp_col7, M31_16)
                    ),
                    mul(op1_imm_col8, M31_32)
                ),
                mul(op1_base_fp_col9, M31_64)
            ),
            mul(op1_base_op0, M31_128)
        );
        sub_component_inputs_verify_instruction[4][row] = flags_a;
        m31 flags_b = add(add(M31_1, mul(ap_update_add_1_col10, M31_32)), M31_256);
        sub_component_inputs_verify_instruction[5][row] = flags_b;
        sub_component_inputs_verify_instruction[6][row] = M31_0;

        // === lookup_verify_instruction_0 ===
        lookup_verify_instruction_0[0][row] = input_pc_col0;
        lookup_verify_instruction_0[1][row] = offset0_col3;
        lookup_verify_instruction_0[2][row] = offset1_col4;
        lookup_verify_instruction_0[3][row] = offset2_col5;
        lookup_verify_instruction_0[4][row] = flags_a;
        lookup_verify_instruction_0[5][row] = flags_b;
        lookup_verify_instruction_0[6][row] = M31_0;

        // === decode_instruction output ===
        m31 decode_offsets[3] = {
            sub(offset0_col3, M31_32768),
            sub(offset1_col4, M31_32768),
            sub(offset2_col5, M31_32768)
        };

        // === mem_dst_base_col11 ===
        m31 mem_dst_base_col11 = add(
            mul(dst_base_fp_col6, input_fp_col2),
            mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)
        );
        traces[11][row] = mem_dst_base_col11;

        // === mem0_base_col12 ===
        m31 mem0_base_col12 = add(
            mul(op0_base_fp_col7, input_fp_col2),
            mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)
        );
        traces[12][row] = mem0_base_col12;

        // === mem1_base_col13 ===
        m31 mem1_base_col13 = add(
            add(
                mul(op1_imm_col8, input_pc_col0),
                mul(op1_base_fp_col9, input_fp_col2)
            ),
            mul(op1_base_op0, input_ap_col1)
        );
        traces[13][row] = mem1_base_col13;

        // === Read Positive Num Bits 72 for dst ===
        m31 dst_addr = add(mem_dst_base_col11, decode_offsets[0]);
        m31 memory_address_to_id_value_tmp_11 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            dst_addr,
            &memory_address_to_id_value_tmp_11
        );
        m31 dst_id_col14 = memory_address_to_id_value_tmp_11;
        traces[14][row] = dst_id_col14;

        sub_component_inputs_memory_address_to_id[0][row] = dst_addr;
        lookup_memory_address_to_id_0[0][row] = dst_addr;
        lookup_memory_address_to_id_0[1][row] = dst_id_col14;

        m31 memory_id_to_big_value_tmp_13[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            dst_id_col14,
            memory_id_to_big_value_tmp_13
        );

        m31 dst_limb_0_col15 = memory_id_to_big_value_tmp_13[0];
        traces[15][row] = dst_limb_0_col15;
        m31 dst_limb_1_col16 = memory_id_to_big_value_tmp_13[1];
        traces[16][row] = dst_limb_1_col16;
        m31 dst_limb_2_col17 = memory_id_to_big_value_tmp_13[2];
        traces[17][row] = dst_limb_2_col17;
        m31 dst_limb_3_col18 = memory_id_to_big_value_tmp_13[3];
        traces[18][row] = dst_limb_3_col18;
        m31 dst_limb_4_col19 = memory_id_to_big_value_tmp_13[4];
        traces[19][row] = dst_limb_4_col19;
        m31 dst_limb_5_col20 = memory_id_to_big_value_tmp_13[5];
        traces[20][row] = dst_limb_5_col20;
        m31 dst_limb_6_col21 = memory_id_to_big_value_tmp_13[6];
        traces[21][row] = dst_limb_6_col21;
        m31 dst_limb_7_col22 = memory_id_to_big_value_tmp_13[7];
        traces[22][row] = dst_limb_7_col22;

        sub_component_inputs_memory_id_to_big[0][row] = dst_id_col14;
        lookup_memory_id_to_big_0[0][row] = dst_id_col14;
        lookup_memory_id_to_big_0[1][row] = dst_limb_0_col15;
        lookup_memory_id_to_big_0[2][row] = dst_limb_1_col16;
        lookup_memory_id_to_big_0[3][row] = dst_limb_2_col17;
        lookup_memory_id_to_big_0[4][row] = dst_limb_3_col18;
        lookup_memory_id_to_big_0[5][row] = dst_limb_4_col19;
        lookup_memory_id_to_big_0[6][row] = dst_limb_5_col20;
        lookup_memory_id_to_big_0[7][row] = dst_limb_6_col21;
        lookup_memory_id_to_big_0[8][row] = dst_limb_7_col22;
        for (int i = 9; i < 29; i++) {
            lookup_memory_id_to_big_0[i][row] = M31_0;
        }

        // === Read Positive Num Bits 36 for op0 ===
        m31 op0_addr = add(mem0_base_col12, decode_offsets[1]);
        m31 memory_address_to_id_value_tmp_16 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op0_addr,
            &memory_address_to_id_value_tmp_16
        );
        m31 op0_id_col23 = memory_address_to_id_value_tmp_16;
        traces[23][row] = op0_id_col23;

        sub_component_inputs_memory_address_to_id[1][row] = op0_addr;
        lookup_memory_address_to_id_1[0][row] = op0_addr;
        lookup_memory_address_to_id_1[1][row] = op0_id_col23;

        m31 memory_id_to_big_value_tmp_18[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            op0_id_col23,
            memory_id_to_big_value_tmp_18
        );

        m31 op0_limb_0_col24 = memory_id_to_big_value_tmp_18[0];
        traces[24][row] = op0_limb_0_col24;
        m31 op0_limb_1_col25 = memory_id_to_big_value_tmp_18[1];
        traces[25][row] = op0_limb_1_col25;
        m31 op0_limb_2_col26 = memory_id_to_big_value_tmp_18[2];
        traces[26][row] = op0_limb_2_col26;
        m31 op0_limb_3_col27 = memory_id_to_big_value_tmp_18[3];
        traces[27][row] = op0_limb_3_col27;

        sub_component_inputs_memory_id_to_big[1][row] = op0_id_col23;
        lookup_memory_id_to_big_1[0][row] = op0_id_col23;
        lookup_memory_id_to_big_1[1][row] = op0_limb_0_col24;
        lookup_memory_id_to_big_1[2][row] = op0_limb_1_col25;
        lookup_memory_id_to_big_1[3][row] = op0_limb_2_col26;
        lookup_memory_id_to_big_1[4][row] = op0_limb_3_col27;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_1[i][row] = M31_0;
        }

        // === Read Positive Num Bits 36 for op1 ===
        m31 op1_addr = add(mem1_base_col13, decode_offsets[2]);
        m31 memory_address_to_id_value_tmp_21 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op1_addr,
            &memory_address_to_id_value_tmp_21
        );
        m31 op1_id_col28 = memory_address_to_id_value_tmp_21;
        traces[28][row] = op1_id_col28;

        sub_component_inputs_memory_address_to_id[2][row] = op1_addr;
        lookup_memory_address_to_id_2[0][row] = op1_addr;
        lookup_memory_address_to_id_2[1][row] = op1_id_col28;

        m31 memory_id_to_big_value_tmp_23[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            op1_id_col28,
            memory_id_to_big_value_tmp_23
        );

        m31 op1_limb_0_col29 = memory_id_to_big_value_tmp_23[0];
        traces[29][row] = op1_limb_0_col29;
        m31 op1_limb_1_col30 = memory_id_to_big_value_tmp_23[1];
        traces[30][row] = op1_limb_1_col30;
        m31 op1_limb_2_col31 = memory_id_to_big_value_tmp_23[2];
        traces[31][row] = op1_limb_2_col31;
        m31 op1_limb_3_col32 = memory_id_to_big_value_tmp_23[3];
        traces[32][row] = op1_limb_3_col32;

        sub_component_inputs_memory_id_to_big[2][row] = op1_id_col28;
        lookup_memory_id_to_big_2[0][row] = op1_id_col28;
        lookup_memory_id_to_big_2[1][row] = op1_limb_0_col29;
        lookup_memory_id_to_big_2[2][row] = op1_limb_1_col30;
        lookup_memory_id_to_big_2[3][row] = op1_limb_2_col31;
        lookup_memory_id_to_big_2[4][row] = op1_limb_3_col32;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_2[i][row] = M31_0;
        }

        // === Verify Mul Small ===
        // carry_1 = (op0_limb_0 * op1_limb_0 - dst_limb_0
        //          + (op0_limb_0 * op1_limb_1 + op0_limb_1 * op1_limb_0 - dst_limb_1) * 512) * 8192
        m31 carry_1_col33 = mul(
            sub(
                add(
                    sub(mul(op0_limb_0_col24, op1_limb_0_col29), dst_limb_0_col15),
                    mul(
                        sub(
                            add(
                                mul(op0_limb_0_col24, op1_limb_1_col30),
                                mul(op0_limb_1_col25, op1_limb_0_col29)
                            ),
                            dst_limb_1_col16
                        ),
                        M31_512
                    )
                ),
                M31_0
            ),
            M31_8192
        );
        traces[33][row] = carry_1_col33;
        sub_component_inputs_range_check_11[0][row] = carry_1_col33;
        lookup_range_check_11_0[0][row] = carry_1_col33;

        // carry_3 calculation
        m31 carry_3_col34 = mul(
            sub(
                add(
                    sub(
                        add(
                            add(
                                add(carry_1_col33, mul(op0_limb_0_col24, op1_limb_2_col31)),
                                mul(op0_limb_1_col25, op1_limb_1_col30)
                            ),
                            mul(op0_limb_2_col26, op1_limb_0_col29)
                        ),
                        dst_limb_2_col17
                    ),
                    mul(
                        sub(
                            add(
                                add(
                                    add(
                                        mul(op0_limb_0_col24, op1_limb_3_col32),
                                        mul(op0_limb_1_col25, op1_limb_2_col31)
                                    ),
                                    mul(op0_limb_2_col26, op1_limb_1_col30)
                                ),
                                mul(op0_limb_3_col27, op1_limb_0_col29)
                            ),
                            dst_limb_3_col18
                        ),
                        M31_512
                    )
                ),
                M31_0
            ),
            M31_8192
        );
        traces[34][row] = carry_3_col34;
        sub_component_inputs_range_check_11[1][row] = carry_3_col34;
        lookup_range_check_11_1[0][row] = carry_3_col34;

        // carry_5 calculation
        m31 carry_5_col35 = mul(
            sub(
                add(
                    sub(
                        add(
                            add(
                                add(carry_3_col34, mul(op0_limb_1_col25, op1_limb_3_col32)),
                                mul(op0_limb_2_col26, op1_limb_2_col31)
                            ),
                            mul(op0_limb_3_col27, op1_limb_1_col30)
                        ),
                        dst_limb_4_col19
                    ),
                    mul(
                        sub(
                            add(
                                mul(op0_limb_2_col26, op1_limb_3_col32),
                                mul(op0_limb_3_col27, op1_limb_2_col31)
                            ),
                            dst_limb_5_col20
                        ),
                        M31_512
                    )
                ),
                M31_0
            ),
            M31_8192
        );
        traces[35][row] = carry_5_col35;
        sub_component_inputs_range_check_11[2][row] = carry_5_col35;
        lookup_range_check_11_2[0][row] = carry_5_col35;

        // === lookup_opcodes_0 ===
        lookup_opcodes_0[0][row] = input_pc_col0;
        lookup_opcodes_0[1][row] = input_ap_col1;
        lookup_opcodes_0[2][row] = input_fp_col2;

        // === lookup_opcodes_1 ===
        lookup_opcodes_1[0][row] = add(add(input_pc_col0, M31_1), op1_imm_col8);
        lookup_opcodes_1[1][row] = add(input_ap_col1, ap_update_add_1_col10);
        lookup_opcodes_1[2][row] = input_fp_col2;

        // === enabler column (col36) ===
        m31 enabler_col = (row < n_rows) ? M31_1 : M31_0;
        traces[36][row] = enabler_col;
    }
}

void generate_mul_opcode_small_traces(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_11_0,
    unsigned **lookup_range_check_11_1,
    unsigned **lookup_range_check_11_2,
    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_11,

    unsigned **mul_opcode_small_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    unsigned trace_size = 1 << log_size;

    timer global_timer;
    global_timer.start("generate mul_opcode_small base trace");

    unsigned **device_traces = clone_to_device<unsigned *>(traces, MUL_OPCODE_SMALL_N_TRACE_COLUMNS);
    unsigned **device_lookup_memory_address_to_id_0 = clone_to_device<unsigned *>(lookup_memory_address_to_id_0, 2);
    unsigned **device_lookup_memory_address_to_id_1 = clone_to_device<unsigned *>(lookup_memory_address_to_id_1, 2);
    unsigned **device_lookup_memory_address_to_id_2 = clone_to_device<unsigned *>(lookup_memory_address_to_id_2, 2);
    unsigned **device_lookup_memory_id_to_big_0 = clone_to_device<unsigned *>(lookup_memory_id_to_big_0, 29);
    unsigned **device_lookup_memory_id_to_big_1 = clone_to_device<unsigned *>(lookup_memory_id_to_big_1, 29);
    unsigned **device_lookup_memory_id_to_big_2 = clone_to_device<unsigned *>(lookup_memory_id_to_big_2, 29);
    unsigned **device_lookup_opcodes_0 = clone_to_device<unsigned *>(lookup_opcodes_0, 3);
    unsigned **device_lookup_opcodes_1 = clone_to_device<unsigned *>(lookup_opcodes_1, 3);
    unsigned **device_lookup_range_check_11_0 = clone_to_device<unsigned *>(lookup_range_check_11_0, 1);
    unsigned **device_lookup_range_check_11_1 = clone_to_device<unsigned *>(lookup_range_check_11_1, 1);
    unsigned **device_lookup_range_check_11_2 = clone_to_device<unsigned *>(lookup_range_check_11_2, 1);
    unsigned **device_lookup_verify_instruction_0 = clone_to_device<unsigned *>(lookup_verify_instruction_0, 7);

    unsigned **device_sub_component_inputs_verify_instruction = clone_to_device<unsigned *>(sub_component_inputs_verify_instruction, 7);
    unsigned **device_sub_component_inputs_memory_address_to_id = clone_to_device<unsigned *>(sub_component_inputs_memory_address_to_id, 3);
    unsigned **device_sub_component_inputs_memory_id_to_big = clone_to_device<unsigned *>(sub_component_inputs_memory_id_to_big, 3);
    unsigned **device_sub_component_inputs_range_check_11 = clone_to_device<unsigned *>(sub_component_inputs_range_check_11, 3);

    unsigned **device_mul_opcode_small_input = clone_to_device<unsigned *>(mul_opcode_small_input, 3);
    unsigned **device_memory_id_to_big_transpose_big_value_ptr = clone_to_device<unsigned *>(memory_id_to_big_transposed_big_values, N_M31_IN_FELT252);

    int block_dim = trace_size < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_mul_opcode_small_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_id_to_big_1,
        device_lookup_memory_id_to_big_2,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        device_lookup_range_check_11_0,
        device_lookup_range_check_11_1,
        device_lookup_range_check_11_2,
        device_lookup_verify_instruction_0,
        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        device_sub_component_inputs_range_check_11,
        device_mul_opcode_small_input,
        memory_address_to_id_address_to_raw_id,
        device_memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_values,
        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate mul_opcode_small base trace");

    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_11_0);
    cuda_free_memory(device_lookup_range_check_11_1);
    cuda_free_memory(device_lookup_range_check_11_2);
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);
    cuda_free_memory(device_sub_component_inputs_range_check_11);
    cuda_free_memory(device_mul_opcode_small_input);
    cuda_free_memory(device_memory_id_to_big_transpose_big_value_ptr);
}

// === Interaction Trace Kernels ===

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_mul_opcode_small_interaction_trace_col_gen_kernel_round0(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    unsigned **lookup_state_0,
    unsigned **lookup_state_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 init_combine_reg[N] = {0};
    m31 final_combine_reg[M] = {0};

    for (int i = 0; i < 7; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }
    for (int i = 0; i < M; i++) {
        final_combine_reg[i] = lookup_state_1[i][vec_index];
    }
    if (vec_index < trace_size) {
        qm31 denom0 = lookup_elements_n->combine(init_combine_reg, N);
        qm31 denom1 = lookup_elements_m->combine(final_combine_reg, M);
        logup_col_write_frac(vec_index, add(denom1, denom0), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_mul_opcode_small_interaction_trace_col_gen_kernel(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    unsigned **lookup_state_0,
    unsigned **lookup_state_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index >= trace_size) return;

    m31 init_combine_reg[N] = {0};
    m31 final_combine_reg[M] = {0};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }
    for (int i = 0; i < M; i++) {
        final_combine_reg[i] = lookup_state_1[i][vec_index];
    }

    qm31 denom0 = lookup_elements_n->combine(init_combine_reg, N);
    qm31 denom1 = lookup_elements_m->combine(final_combine_reg, M);
    logup_col_write_frac(vec_index, add(denom1, denom0), mul(denom0, denom1),
                        denom_ptr, numerator0, numerator1, numerator2, numerator3);
}

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_mul_opcode_small_interaction_trace_col_gen_kernel_opcodes(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    unsigned **lookup_state_0,
    unsigned **lookup_state_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned n_rows
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index >= trace_size) return;

    qm31 enabler_col = (vec_index < n_rows) ? qm31{1, 0, 0, 0} : qm31{0, 0, 0, 0};

    m31 init_combine_reg[N] = {0};
    m31 final_combine_reg[M] = {0};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }
    for (int i = 0; i < M; i++) {
        final_combine_reg[i] = lookup_state_1[i][vec_index];
    }

    qm31 denom0 = lookup_elements_n->combine(init_combine_reg, N);
    qm31 denom1 = lookup_elements_m->combine(final_combine_reg, M);
    // numerator = denom1 * enabler - denom0 * enabler
    qm31 num = sub(mul(denom1, enabler_col), mul(denom0, enabler_col));
    logup_col_write_frac(vec_index, num, mul(denom0, denom1),
                        denom_ptr, numerator0, numerator1, numerator2, numerator3);
}

__global__ void generate_mul_opcode_small_interaction_trace_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = rep_index - 1;

    if (vec_index < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[vec_index], numerator1[vec_index]},
                cm31{numerator2[vec_index], numerator3[vec_index]}
            },
            denom_inv_ptr[vec_index]
        );

        // Add previous column's value (column-wise accumulation)
        qm31 prev_value;
        if (pre_index < 0) {
            prev_value = qm31{0, 0, 0, 0};
        } else {
            prev_value = qm31 {
                cm31{m31{interaction_traces[pre_index * 4 + 0][vec_index]}, m31{interaction_traces[pre_index * 4 + 1][vec_index]}},
                cm31{m31{interaction_traces[pre_index * 4 + 2][vec_index]}, m31{interaction_traces[pre_index * 4 + 3][vec_index]}}
            };
        }
        qm31 result = add(value, prev_value);

        interaction_traces[rep_index * 4 + 0][vec_index] = result.a.a;
        interaction_traces[rep_index * 4 + 1][vec_index] = result.a.b;
        interaction_traces[rep_index * 4 + 2][vec_index] = result.b.a;
        interaction_traces[rep_index * 4 + 3][vec_index] = result.b.b;
    }
}

__global__ void generate_mul_opcode_small_interaction_trace_cumsum_shift(
    unsigned last_index,
    unsigned trace_size,
    unsigned **interactive_traces,
    m31 *coordinate_sums
) {
    int idx0 = 4 * last_index - 4;
    int idx1 = 4 * last_index - 3;
    int idx2 = 4 * last_index - 2;
    int idx3 = 4 * last_index - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = {0};
    m31 sum1 = {0};
    m31 sum2 = {0};
    m31 sum3 = {0};

    for (int i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, m31{interactive_traces[idx0][i]});
        sum1 = add(sum1, m31{interactive_traces[idx1][i]});
        sum2 = add(sum2, m31{interactive_traces[idx2][i]});
        sum3 = add(sum3, m31{interactive_traces[idx3][i]});
    }

    extern __shared__ m31 shared[];
    m31* sdata0 = &shared[0];
    m31* sdata1 = &shared[blockDim.x];
    m31* sdata2 = &shared[2 * blockDim.x];
    m31* sdata3 = &shared[3 * blockDim.x];

    sdata0[threadIdx.x] = sum0;
    sdata1[threadIdx.x] = sum1;
    sdata2[threadIdx.x] = sum2;
    sdata3[threadIdx.x] = sum3;

    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata0[threadIdx.x] = add(sdata0[threadIdx.x], sdata0[threadIdx.x + s]);
            sdata1[threadIdx.x] = add(sdata1[threadIdx.x], sdata1[threadIdx.x + s]);
            sdata2[threadIdx.x] = add(sdata2[threadIdx.x], sdata2[threadIdx.x + s]);
            sdata3[threadIdx.x] = add(sdata3[threadIdx.x], sdata3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sdata0[0]);
        atomic_add(&coordinate_sums[1], sdata1[0]);
        atomic_add(&coordinate_sums[2], sdata2[0]);
        atomic_add(&coordinate_sums[3], sdata3[0]);
    }
}

__global__ void generate_mul_opcode_small_interaction_trace_coord_prefix_sum(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    unsigned **interactive_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interactive_traces[4 * last_index - 4][vec_index] = sub(m31{interactive_traces[4 * last_index - 4][vec_index]}, cumsum_shift.a.a);
        interactive_traces[4 * last_index - 3][vec_index] = sub(m31{interactive_traces[4 * last_index - 3][vec_index]}, cumsum_shift.a.b);
        interactive_traces[4 * last_index - 2][vec_index] = sub(m31{interactive_traces[4 * last_index - 2][vec_index]}, cumsum_shift.b.a);
        interactive_traces[4 * last_index - 1][vec_index] = sub(m31{interactive_traces[4 * last_index - 1][vec_index]}, cumsum_shift.b.b);
    }
}

void generate_mul_opcode_small_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,
    void *range_check_11,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_11_0,
    unsigned **lookup_range_check_11_1,
    unsigned **lookup_range_check_11_2,
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    MemoryAddressToId *memory_address_to_id_lookup_elements = (MemoryAddressToId *)memory_address_to_id;
    MemoryIdToBig *memory_id_to_big_lookup_elements = (MemoryIdToBig *)memory_id_to_big;
    Opcodes *opcodes_lookup_elements = (Opcodes *)opcodes;
    VerifyInstruction *verify_instruction_lookup_elements = (VerifyInstruction *)verify_instruction;
    RangeCheck_11 *range_check_11_lookup_elements = (RangeCheck_11 *)range_check_11;

    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes_lookup_elements = cuda_malloc<Opcodes>(1);
    VerifyInstruction *device_verify_instruction_lookup_elements = cuda_malloc<VerifyInstruction>(1);
    RangeCheck_11 *device_range_check_11_lookup_elements = cuda_malloc<RangeCheck_11>(1);

    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id_lookup_elements, 1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big_lookup_elements, 1);
    cuda_mem_copy_host_to_device<Opcodes>(opcodes_lookup_elements, device_opcodes_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyInstruction>(verify_instruction_lookup_elements, device_verify_instruction_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_11>(range_check_11_lookup_elements, device_range_check_11_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    unsigned **device_lookup_memory_address_to_id_0 = clone_to_device<unsigned *>(lookup_memory_address_to_id_0, 2);
    unsigned **device_lookup_memory_address_to_id_1 = clone_to_device<unsigned *>(lookup_memory_address_to_id_1, 2);
    unsigned **device_lookup_memory_address_to_id_2 = clone_to_device<unsigned *>(lookup_memory_address_to_id_2, 2);
    unsigned **device_lookup_memory_id_to_big_0 = clone_to_device<unsigned *>(lookup_memory_id_to_big_0, 29);
    unsigned **device_lookup_memory_id_to_big_1 = clone_to_device<unsigned *>(lookup_memory_id_to_big_1, 29);
    unsigned **device_lookup_memory_id_to_big_2 = clone_to_device<unsigned *>(lookup_memory_id_to_big_2, 29);
    unsigned **device_lookup_opcodes_0 = clone_to_device<unsigned *>(lookup_opcodes_0, 3);
    unsigned **device_lookup_opcodes_1 = clone_to_device<unsigned *>(lookup_opcodes_1, 3);
    unsigned **device_lookup_range_check_11_0 = clone_to_device<unsigned *>(lookup_range_check_11_0, 1);
    unsigned **device_lookup_range_check_11_1 = clone_to_device<unsigned *>(lookup_range_check_11_1, 1);
    unsigned **device_lookup_range_check_11_2 = clone_to_device<unsigned *>(lookup_range_check_11_2, 1);
    unsigned **device_lookup_verify_instruction_0 = clone_to_device<unsigned *>(lookup_verify_instruction_0, 7);

    unsigned **device_interaction_traces = clone_to_device<unsigned *>(interaction_trace, 4 * MUL_OPCODE_SMALL_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);

    timer global_timer;
    global_timer.start("generate mul_opcode_small interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For verify_instruction_0 & memory_address_to_id_0
    generate_mul_opcode_small_interaction_trace_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
        device_verify_instruction_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_verify_instruction_0,
        device_lookup_memory_address_to_id_0,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        0, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #1 Interaction trace For memory_id_to_big_0 & memory_address_to_id_1
    block_dim = trace_size < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_col_gen_kernel<29, 2><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_address_to_id_1,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        1, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #2 Interaction trace For memory_id_to_big_1 & memory_address_to_id_2
    block_dim = trace_size < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_col_gen_kernel<29, 2><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_memory_id_to_big_1,
        device_lookup_memory_address_to_id_2,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        2, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #3 Interaction trace For memory_id_to_big_2 & range_check_11_0
    block_dim = trace_size < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_col_gen_kernel<29, 1><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_11_lookup_elements,
        device_lookup_memory_id_to_big_2,
        device_lookup_range_check_11_0,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        3, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #4 Interaction trace For range_check_11_1 & range_check_11_2
    block_dim = trace_size < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_col_gen_kernel<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_11_lookup_elements,
        device_range_check_11_lookup_elements,
        device_lookup_range_check_11_1,
        device_lookup_range_check_11_2,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        4, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #5 Interaction trace For opcodes_0 & opcodes_1 (with enabler)
    block_dim = trace_size < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_SMALL_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_col_gen_kernel_opcodes<3, 3><<<num_blocks, block_dim>>>(
        device_opcodes_lookup_elements,
        device_opcodes_lookup_elements,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        n_rows
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        5, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Compute claimed_sum
    m31 *device_coordinate_sums = cuda_malloc<m31>(4);
    cudaMemsetAsync(device_coordinate_sums, 0, 4 * sizeof(m31), 0);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_small_interaction_trace_cumsum_shift<<<num_blocks, block_dim, 4 * block_dim * sizeof(m31)>>>(
        MUL_OPCODE_SMALL_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        device_coordinate_sums
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Apply cumsum shift to last column
    generate_mul_opcode_small_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        device_coordinate_sums,
        MUL_OPCODE_SMALL_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Apply inclusive prefix sum to the last column
    inclusive_prefix_sum((m31 *)interaction_trace[4 * MUL_OPCODE_SMALL_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum((m31 *)interaction_trace[4 * MUL_OPCODE_SMALL_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum((m31 *)interaction_trace[4 * MUL_OPCODE_SMALL_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum((m31 *)interaction_trace[4 * MUL_OPCODE_SMALL_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    // Copy claimed_sum back to host
    cuda_mem_copy_device_to_host<m31>(device_coordinate_sums, (m31 *)claimed_sum, 4);

    global_timer.end("generate mul_opcode_small interaction trace");

    // Cleanup
    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_opcodes_lookup_elements);
    cuda_free_memory(device_verify_instruction_lookup_elements);
    cuda_free_memory(device_range_check_11_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_11_0);
    cuda_free_memory(device_lookup_range_check_11_1);
    cuda_free_memory(device_lookup_range_check_11_2);
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_coordinate_sums);
}
