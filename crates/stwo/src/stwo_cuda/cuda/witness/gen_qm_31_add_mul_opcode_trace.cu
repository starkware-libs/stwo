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

#include "gen_qm_31_add_mul_opcode_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

__launch_bounds__(256, 2)
__global__ void generate_qm_31_add_mul_opcode_trace_kernel(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,

    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,

    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    unsigned **lookup_range_check_4_4_4_4_0,
    unsigned **lookup_range_check_4_4_4_4_1,
    unsigned **lookup_range_check_4_4_4_4_2,

    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_4_4_4_4,

    unsigned **qm_31_add_mul_opcode_inputs,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0 = {0};
    const m31 M31_1 = {1};
    const m31 M31_3 = {3};
    const m31 M31_8 = {8};
    const m31 M31_16 = {16};
    const m31 M31_32 = {32};
    const m31 M31_64 = {64};
    const m31 M31_128 = {128};
    const m31 M31_256 = {256};
    const m31 M31_512 = {512};
    const m31 M31_1548 = {1548};
    const m31 M31_32768 = {32768};
    const m31 M31_262144 = {262144};
    const m31 M31_134217728 = {134217728};

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
        m31 input_pc_col0 = {qm_31_add_mul_opcode_inputs[0][row]};
        traces[0][row] = input_pc_col0;
        m31 input_ap_col1 = {qm_31_add_mul_opcode_inputs[1][row]};
        traces[1][row] = input_ap_col1;
        m31 input_fp_col2 = {qm_31_add_mul_opcode_inputs[2][row]};
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

        // flags extraction
        uint16_t flags_combined = (((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6);

        // dst_base_fp
        uint16_t dst_base_fp_tmp_5 = (flags_combined >> UInt16_0) & UInt16_1;
        m31 dst_base_fp_col6 = m31{dst_base_fp_tmp_5};
        traces[6][row] = dst_base_fp_col6;

        // op0_base_fp
        uint16_t op0_base_fp_tmp_6 = (flags_combined >> UInt16_1) & UInt16_1;
        m31 op0_base_fp_col7 = m31{op0_base_fp_tmp_6};
        traces[7][row] = op0_base_fp_col7;

        // op1_imm
        uint16_t op1_imm_tmp_7 = (flags_combined >> UInt16_2) & UInt16_1;
        m31 op1_imm_col8 = m31{op1_imm_tmp_7};
        traces[8][row] = op1_imm_col8;

        // op1_base_fp
        uint16_t op1_base_fp_tmp_8 = (flags_combined >> UInt16_3) & UInt16_1;
        m31 op1_base_fp_col9 = m31{op1_base_fp_tmp_8};
        traces[9][row] = op1_base_fp_col9;

        // res_add
        uint16_t res_add_tmp_9 = (flags_combined >> UInt16_5) & UInt16_1;
        m31 res_add_col10 = m31{res_add_tmp_9};
        traces[10][row] = res_add_col10;

        // ap_update_add_1
        uint16_t ap_update_add_1_tmp_10 = (flags_combined >> UInt16_11) & UInt16_1;
        m31 ap_update_add_1_col11 = m31{ap_update_add_1_tmp_10};
        traces[11][row] = ap_update_add_1_col11;

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
                ),
                mul(res_add_col10, M31_256)
            ),
            M31_0  // no additional offset for res_mul in this opcode
        );
        sub_component_inputs_verify_instruction[4][row] = flags_a;
        // flags_b = (1 - res_add) + ap_update_add_1 * 32 + 256
        m31 flags_b = add(add(sub(M31_1, res_add_col10), mul(ap_update_add_1_col11, M31_32)), M31_256);
        sub_component_inputs_verify_instruction[5][row] = flags_b;
        sub_component_inputs_verify_instruction[6][row] = M31_3;

        // === lookup_verify_instruction_0 ===
        lookup_verify_instruction_0[0][row] = input_pc_col0;
        lookup_verify_instruction_0[1][row] = offset0_col3;
        lookup_verify_instruction_0[2][row] = offset1_col4;
        lookup_verify_instruction_0[3][row] = offset2_col5;
        lookup_verify_instruction_0[4][row] = flags_a;
        lookup_verify_instruction_0[5][row] = flags_b;
        lookup_verify_instruction_0[6][row] = M31_3;

        // === decode_instruction output ===
        m31 decode_offsets[3] = {
            sub(offset0_col3, M31_32768),
            sub(offset1_col4, M31_32768),
            sub(offset2_col5, M31_32768)
        };

        // === mem_dst_base_col12 ===
        m31 mem_dst_base_col12 = add(
            mul(dst_base_fp_col6, input_fp_col2),
            mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)
        );
        traces[12][row] = mem_dst_base_col12;

        // === mem0_base_col13 ===
        m31 mem0_base_col13 = add(
            mul(op0_base_fp_col7, input_fp_col2),
            mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)
        );
        traces[13][row] = mem0_base_col13;

        // === mem1_base_col14 ===
        m31 mem1_base_col14 = add(
            add(
                mul(op1_base_fp_col9, input_fp_col2),
                mul(op1_base_op0, input_ap_col1)
            ),
            mul(op1_imm_col8, input_pc_col0)
        );
        traces[14][row] = mem1_base_col14;

        // === Qm 31 Read Reduced for dst ===
        m31 dst_addr = add(mem_dst_base_col12, decode_offsets[0]);
        m31 memory_address_to_id_value_tmp_12 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            dst_addr,
            &memory_address_to_id_value_tmp_12
        );
        m31 dst_id_col15 = memory_address_to_id_value_tmp_12;
        traces[15][row] = dst_id_col15;

        sub_component_inputs_memory_address_to_id[0][row] = dst_addr;
        lookup_memory_address_to_id_0[0][row] = dst_addr;
        lookup_memory_address_to_id_0[1][row] = dst_id_col15;

        m31 memory_id_to_big_value_tmp_14[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            dst_id_col15,
            memory_id_to_big_value_tmp_14
        );

        // dst limbs 0-15 (col16-31)
        m31 dst_limb_0_col16 = memory_id_to_big_value_tmp_14[0];
        traces[16][row] = dst_limb_0_col16;
        m31 dst_limb_1_col17 = memory_id_to_big_value_tmp_14[1];
        traces[17][row] = dst_limb_1_col17;
        m31 dst_limb_2_col18 = memory_id_to_big_value_tmp_14[2];
        traces[18][row] = dst_limb_2_col18;
        m31 dst_limb_3_col19 = memory_id_to_big_value_tmp_14[3];
        traces[19][row] = dst_limb_3_col19;
        m31 dst_limb_4_col20 = memory_id_to_big_value_tmp_14[4];
        traces[20][row] = dst_limb_4_col20;
        m31 dst_limb_5_col21 = memory_id_to_big_value_tmp_14[5];
        traces[21][row] = dst_limb_5_col21;
        m31 dst_limb_6_col22 = memory_id_to_big_value_tmp_14[6];
        traces[22][row] = dst_limb_6_col22;
        m31 dst_limb_7_col23 = memory_id_to_big_value_tmp_14[7];
        traces[23][row] = dst_limb_7_col23;
        m31 dst_limb_8_col24 = memory_id_to_big_value_tmp_14[8];
        traces[24][row] = dst_limb_8_col24;
        m31 dst_limb_9_col25 = memory_id_to_big_value_tmp_14[9];
        traces[25][row] = dst_limb_9_col25;
        m31 dst_limb_10_col26 = memory_id_to_big_value_tmp_14[10];
        traces[26][row] = dst_limb_10_col26;
        m31 dst_limb_11_col27 = memory_id_to_big_value_tmp_14[11];
        traces[27][row] = dst_limb_11_col27;
        m31 dst_limb_12_col28 = memory_id_to_big_value_tmp_14[12];
        traces[28][row] = dst_limb_12_col28;
        m31 dst_limb_13_col29 = memory_id_to_big_value_tmp_14[13];
        traces[29][row] = dst_limb_13_col29;
        m31 dst_limb_14_col30 = memory_id_to_big_value_tmp_14[14];
        traces[30][row] = dst_limb_14_col30;
        m31 dst_limb_15_col31 = memory_id_to_big_value_tmp_14[15];
        traces[31][row] = dst_limb_15_col31;

        sub_component_inputs_memory_id_to_big[0][row] = dst_id_col15;
        lookup_memory_id_to_big_0[0][row] = dst_id_col15;
        lookup_memory_id_to_big_0[1][row] = dst_limb_0_col16;
        lookup_memory_id_to_big_0[2][row] = dst_limb_1_col17;
        lookup_memory_id_to_big_0[3][row] = dst_limb_2_col18;
        lookup_memory_id_to_big_0[4][row] = dst_limb_3_col19;
        lookup_memory_id_to_big_0[5][row] = dst_limb_4_col20;
        lookup_memory_id_to_big_0[6][row] = dst_limb_5_col21;
        lookup_memory_id_to_big_0[7][row] = dst_limb_6_col22;
        lookup_memory_id_to_big_0[8][row] = dst_limb_7_col23;
        lookup_memory_id_to_big_0[9][row] = dst_limb_8_col24;
        lookup_memory_id_to_big_0[10][row] = dst_limb_9_col25;
        lookup_memory_id_to_big_0[11][row] = dst_limb_10_col26;
        lookup_memory_id_to_big_0[12][row] = dst_limb_11_col27;
        lookup_memory_id_to_big_0[13][row] = dst_limb_12_col28;
        lookup_memory_id_to_big_0[14][row] = dst_limb_13_col29;
        lookup_memory_id_to_big_0[15][row] = dst_limb_14_col30;
        lookup_memory_id_to_big_0[16][row] = dst_limb_15_col31;
        for (int i = 17; i < 29; i++) {
            lookup_memory_id_to_big_0[i][row] = M31_0;
        }

        // range_check_4_4_4_4 for dst
        sub_component_inputs_range_check_4_4_4_4[0][row] = dst_limb_3_col19;
        sub_component_inputs_range_check_4_4_4_4[1][row] = dst_limb_7_col23;
        sub_component_inputs_range_check_4_4_4_4[2][row] = dst_limb_11_col27;
        sub_component_inputs_range_check_4_4_4_4[3][row] = dst_limb_15_col31;
        lookup_range_check_4_4_4_4_0[0][row] = dst_limb_3_col19;
        lookup_range_check_4_4_4_4_0[1][row] = dst_limb_7_col23;
        lookup_range_check_4_4_4_4_0[2][row] = dst_limb_11_col27;
        lookup_range_check_4_4_4_4_0[3][row] = dst_limb_15_col31;

        // dst_delta_ab_inv_col32
        m31 dst_sum_a = add(add(add(dst_limb_0_col16, dst_limb_1_col17), dst_limb_2_col18), dst_limb_3_col19);
        m31 dst_sum_b = add(add(add(dst_limb_4_col20, dst_limb_5_col21), dst_limb_6_col22), dst_limb_7_col23);
        m31 dst_delta_ab = mul(sub(dst_sum_a, M31_1548), sub(dst_sum_b, M31_1548));
        m31 dst_delta_ab_inv_col32 = inv(dst_delta_ab);
        traces[32][row] = dst_delta_ab_inv_col32;

        // dst_delta_cd_inv_col33
        m31 dst_sum_c = add(add(add(dst_limb_8_col24, dst_limb_9_col25), dst_limb_10_col26), dst_limb_11_col27);
        m31 dst_sum_d = add(add(add(dst_limb_12_col28, dst_limb_13_col29), dst_limb_14_col30), dst_limb_15_col31);
        m31 dst_delta_cd = mul(sub(dst_sum_c, M31_1548), sub(dst_sum_d, M31_1548));
        m31 dst_delta_cd_inv_col33 = inv(dst_delta_cd);
        traces[33][row] = dst_delta_cd_inv_col33;

        // === Qm 31 Read Reduced for op0 ===
        m31 op0_addr = add(mem0_base_col13, decode_offsets[1]);
        m31 memory_address_to_id_value_tmp_18 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op0_addr,
            &memory_address_to_id_value_tmp_18
        );
        m31 op0_id_col34 = memory_address_to_id_value_tmp_18;
        traces[34][row] = op0_id_col34;

        sub_component_inputs_memory_address_to_id[1][row] = op0_addr;
        lookup_memory_address_to_id_1[0][row] = op0_addr;
        lookup_memory_address_to_id_1[1][row] = op0_id_col34;

        m31 memory_id_to_big_value_tmp_20[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            op0_id_col34,
            memory_id_to_big_value_tmp_20
        );

        // op0 limbs 0-15 (col35-50)
        m31 op0_limb_0_col35 = memory_id_to_big_value_tmp_20[0];
        traces[35][row] = op0_limb_0_col35;
        m31 op0_limb_1_col36 = memory_id_to_big_value_tmp_20[1];
        traces[36][row] = op0_limb_1_col36;
        m31 op0_limb_2_col37 = memory_id_to_big_value_tmp_20[2];
        traces[37][row] = op0_limb_2_col37;
        m31 op0_limb_3_col38 = memory_id_to_big_value_tmp_20[3];
        traces[38][row] = op0_limb_3_col38;
        m31 op0_limb_4_col39 = memory_id_to_big_value_tmp_20[4];
        traces[39][row] = op0_limb_4_col39;
        m31 op0_limb_5_col40 = memory_id_to_big_value_tmp_20[5];
        traces[40][row] = op0_limb_5_col40;
        m31 op0_limb_6_col41 = memory_id_to_big_value_tmp_20[6];
        traces[41][row] = op0_limb_6_col41;
        m31 op0_limb_7_col42 = memory_id_to_big_value_tmp_20[7];
        traces[42][row] = op0_limb_7_col42;
        m31 op0_limb_8_col43 = memory_id_to_big_value_tmp_20[8];
        traces[43][row] = op0_limb_8_col43;
        m31 op0_limb_9_col44 = memory_id_to_big_value_tmp_20[9];
        traces[44][row] = op0_limb_9_col44;
        m31 op0_limb_10_col45 = memory_id_to_big_value_tmp_20[10];
        traces[45][row] = op0_limb_10_col45;
        m31 op0_limb_11_col46 = memory_id_to_big_value_tmp_20[11];
        traces[46][row] = op0_limb_11_col46;
        m31 op0_limb_12_col47 = memory_id_to_big_value_tmp_20[12];
        traces[47][row] = op0_limb_12_col47;
        m31 op0_limb_13_col48 = memory_id_to_big_value_tmp_20[13];
        traces[48][row] = op0_limb_13_col48;
        m31 op0_limb_14_col49 = memory_id_to_big_value_tmp_20[14];
        traces[49][row] = op0_limb_14_col49;
        m31 op0_limb_15_col50 = memory_id_to_big_value_tmp_20[15];
        traces[50][row] = op0_limb_15_col50;

        sub_component_inputs_memory_id_to_big[1][row] = op0_id_col34;
        lookup_memory_id_to_big_1[0][row] = op0_id_col34;
        lookup_memory_id_to_big_1[1][row] = op0_limb_0_col35;
        lookup_memory_id_to_big_1[2][row] = op0_limb_1_col36;
        lookup_memory_id_to_big_1[3][row] = op0_limb_2_col37;
        lookup_memory_id_to_big_1[4][row] = op0_limb_3_col38;
        lookup_memory_id_to_big_1[5][row] = op0_limb_4_col39;
        lookup_memory_id_to_big_1[6][row] = op0_limb_5_col40;
        lookup_memory_id_to_big_1[7][row] = op0_limb_6_col41;
        lookup_memory_id_to_big_1[8][row] = op0_limb_7_col42;
        lookup_memory_id_to_big_1[9][row] = op0_limb_8_col43;
        lookup_memory_id_to_big_1[10][row] = op0_limb_9_col44;
        lookup_memory_id_to_big_1[11][row] = op0_limb_10_col45;
        lookup_memory_id_to_big_1[12][row] = op0_limb_11_col46;
        lookup_memory_id_to_big_1[13][row] = op0_limb_12_col47;
        lookup_memory_id_to_big_1[14][row] = op0_limb_13_col48;
        lookup_memory_id_to_big_1[15][row] = op0_limb_14_col49;
        lookup_memory_id_to_big_1[16][row] = op0_limb_15_col50;
        for (int i = 17; i < 29; i++) {
            lookup_memory_id_to_big_1[i][row] = M31_0;
        }

        // range_check_4_4_4_4 for op0
        sub_component_inputs_range_check_4_4_4_4[4][row] = op0_limb_3_col38;
        sub_component_inputs_range_check_4_4_4_4[5][row] = op0_limb_7_col42;
        sub_component_inputs_range_check_4_4_4_4[6][row] = op0_limb_11_col46;
        sub_component_inputs_range_check_4_4_4_4[7][row] = op0_limb_15_col50;
        lookup_range_check_4_4_4_4_1[0][row] = op0_limb_3_col38;
        lookup_range_check_4_4_4_4_1[1][row] = op0_limb_7_col42;
        lookup_range_check_4_4_4_4_1[2][row] = op0_limb_11_col46;
        lookup_range_check_4_4_4_4_1[3][row] = op0_limb_15_col50;

        // op0_delta_ab_inv_col51
        m31 op0_sum_a = add(add(add(op0_limb_0_col35, op0_limb_1_col36), op0_limb_2_col37), op0_limb_3_col38);
        m31 op0_sum_b = add(add(add(op0_limb_4_col39, op0_limb_5_col40), op0_limb_6_col41), op0_limb_7_col42);
        m31 op0_delta_ab = mul(sub(op0_sum_a, M31_1548), sub(op0_sum_b, M31_1548));
        m31 op0_delta_ab_inv_col51 = inv(op0_delta_ab);
        traces[51][row] = op0_delta_ab_inv_col51;

        // op0_delta_cd_inv_col52
        m31 op0_sum_c = add(add(add(op0_limb_8_col43, op0_limb_9_col44), op0_limb_10_col45), op0_limb_11_col46);
        m31 op0_sum_d = add(add(add(op0_limb_12_col47, op0_limb_13_col48), op0_limb_14_col49), op0_limb_15_col50);
        m31 op0_delta_cd = mul(sub(op0_sum_c, M31_1548), sub(op0_sum_d, M31_1548));
        m31 op0_delta_cd_inv_col52 = inv(op0_delta_cd);
        traces[52][row] = op0_delta_cd_inv_col52;

        // === Qm 31 Read Reduced for op1 ===
        m31 op1_addr = add(mem1_base_col14, decode_offsets[2]);
        m31 memory_address_to_id_value_tmp_24 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op1_addr,
            &memory_address_to_id_value_tmp_24
        );
        m31 op1_id_col53 = memory_address_to_id_value_tmp_24;
        traces[53][row] = op1_id_col53;

        sub_component_inputs_memory_address_to_id[2][row] = op1_addr;
        lookup_memory_address_to_id_2[0][row] = op1_addr;
        lookup_memory_address_to_id_2[1][row] = op1_id_col53;

        m31 memory_id_to_big_value_tmp_26[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            op1_id_col53,
            memory_id_to_big_value_tmp_26
        );

        // op1 limbs 0-15 (col54-69)
        m31 op1_limb_0_col54 = memory_id_to_big_value_tmp_26[0];
        traces[54][row] = op1_limb_0_col54;
        m31 op1_limb_1_col55 = memory_id_to_big_value_tmp_26[1];
        traces[55][row] = op1_limb_1_col55;
        m31 op1_limb_2_col56 = memory_id_to_big_value_tmp_26[2];
        traces[56][row] = op1_limb_2_col56;
        m31 op1_limb_3_col57 = memory_id_to_big_value_tmp_26[3];
        traces[57][row] = op1_limb_3_col57;
        m31 op1_limb_4_col58 = memory_id_to_big_value_tmp_26[4];
        traces[58][row] = op1_limb_4_col58;
        m31 op1_limb_5_col59 = memory_id_to_big_value_tmp_26[5];
        traces[59][row] = op1_limb_5_col59;
        m31 op1_limb_6_col60 = memory_id_to_big_value_tmp_26[6];
        traces[60][row] = op1_limb_6_col60;
        m31 op1_limb_7_col61 = memory_id_to_big_value_tmp_26[7];
        traces[61][row] = op1_limb_7_col61;
        m31 op1_limb_8_col62 = memory_id_to_big_value_tmp_26[8];
        traces[62][row] = op1_limb_8_col62;
        m31 op1_limb_9_col63 = memory_id_to_big_value_tmp_26[9];
        traces[63][row] = op1_limb_9_col63;
        m31 op1_limb_10_col64 = memory_id_to_big_value_tmp_26[10];
        traces[64][row] = op1_limb_10_col64;
        m31 op1_limb_11_col65 = memory_id_to_big_value_tmp_26[11];
        traces[65][row] = op1_limb_11_col65;
        m31 op1_limb_12_col66 = memory_id_to_big_value_tmp_26[12];
        traces[66][row] = op1_limb_12_col66;
        m31 op1_limb_13_col67 = memory_id_to_big_value_tmp_26[13];
        traces[67][row] = op1_limb_13_col67;
        m31 op1_limb_14_col68 = memory_id_to_big_value_tmp_26[14];
        traces[68][row] = op1_limb_14_col68;
        m31 op1_limb_15_col69 = memory_id_to_big_value_tmp_26[15];
        traces[69][row] = op1_limb_15_col69;

        sub_component_inputs_memory_id_to_big[2][row] = op1_id_col53;
        lookup_memory_id_to_big_2[0][row] = op1_id_col53;
        lookup_memory_id_to_big_2[1][row] = op1_limb_0_col54;
        lookup_memory_id_to_big_2[2][row] = op1_limb_1_col55;
        lookup_memory_id_to_big_2[3][row] = op1_limb_2_col56;
        lookup_memory_id_to_big_2[4][row] = op1_limb_3_col57;
        lookup_memory_id_to_big_2[5][row] = op1_limb_4_col58;
        lookup_memory_id_to_big_2[6][row] = op1_limb_5_col59;
        lookup_memory_id_to_big_2[7][row] = op1_limb_6_col60;
        lookup_memory_id_to_big_2[8][row] = op1_limb_7_col61;
        lookup_memory_id_to_big_2[9][row] = op1_limb_8_col62;
        lookup_memory_id_to_big_2[10][row] = op1_limb_9_col63;
        lookup_memory_id_to_big_2[11][row] = op1_limb_10_col64;
        lookup_memory_id_to_big_2[12][row] = op1_limb_11_col65;
        lookup_memory_id_to_big_2[13][row] = op1_limb_12_col66;
        lookup_memory_id_to_big_2[14][row] = op1_limb_13_col67;
        lookup_memory_id_to_big_2[15][row] = op1_limb_14_col68;
        lookup_memory_id_to_big_2[16][row] = op1_limb_15_col69;
        for (int i = 17; i < 29; i++) {
            lookup_memory_id_to_big_2[i][row] = M31_0;
        }

        // range_check_4_4_4_4 for op1
        sub_component_inputs_range_check_4_4_4_4[8][row] = op1_limb_3_col57;
        sub_component_inputs_range_check_4_4_4_4[9][row] = op1_limb_7_col61;
        sub_component_inputs_range_check_4_4_4_4[10][row] = op1_limb_11_col65;
        sub_component_inputs_range_check_4_4_4_4[11][row] = op1_limb_15_col69;
        lookup_range_check_4_4_4_4_2[0][row] = op1_limb_3_col57;
        lookup_range_check_4_4_4_4_2[1][row] = op1_limb_7_col61;
        lookup_range_check_4_4_4_4_2[2][row] = op1_limb_11_col65;
        lookup_range_check_4_4_4_4_2[3][row] = op1_limb_15_col69;

        // op1_delta_ab_inv_col70
        m31 op1_sum_a = add(add(add(op1_limb_0_col54, op1_limb_1_col55), op1_limb_2_col56), op1_limb_3_col57);
        m31 op1_sum_b = add(add(add(op1_limb_4_col58, op1_limb_5_col59), op1_limb_6_col60), op1_limb_7_col61);
        m31 op1_delta_ab = mul(sub(op1_sum_a, M31_1548), sub(op1_sum_b, M31_1548));
        m31 op1_delta_ab_inv_col70 = inv(op1_delta_ab);
        traces[70][row] = op1_delta_ab_inv_col70;

        // op1_delta_cd_inv_col71
        m31 op1_sum_c = add(add(add(op1_limb_8_col62, op1_limb_9_col63), op1_limb_10_col64), op1_limb_11_col65);
        m31 op1_sum_d = add(add(add(op1_limb_12_col66, op1_limb_13_col67), op1_limb_14_col68), op1_limb_15_col69);
        m31 op1_delta_cd = mul(sub(op1_sum_c, M31_1548), sub(op1_sum_d, M31_1548));
        m31 op1_delta_cd_inv_col71 = inv(op1_delta_cd);
        traces[71][row] = op1_delta_cd_inv_col71;

        // === lookup_opcodes_0 ===
        lookup_opcodes_0[0][row] = input_pc_col0;
        lookup_opcodes_0[1][row] = input_ap_col1;
        lookup_opcodes_0[2][row] = input_fp_col2;

        // === lookup_opcodes_1 ===
        lookup_opcodes_1[0][row] = add(add(input_pc_col0, M31_1), op1_imm_col8);
        lookup_opcodes_1[1][row] = add(input_ap_col1, ap_update_add_1_col11);
        lookup_opcodes_1[2][row] = input_fp_col2;

        // === enabler column (col72) ===
        m31 enabler_col = (row < n_rows) ? M31_1 : M31_0;
        traces[72][row] = enabler_col;
    }
}

void generate_qm_31_add_mul_opcode_traces(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_4_4_4_4_0,
    unsigned **lookup_range_check_4_4_4_4_1,
    unsigned **lookup_range_check_4_4_4_4_2,
    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_4_4_4_4,

    unsigned **qm_31_add_mul_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    unsigned trace_size = 1 << log_size;

    timer global_timer;
    global_timer.start("generate qm_31_add_mul_opcode base trace");

    unsigned **device_traces = clone_to_device<unsigned *>(traces, QM_31_ADD_MUL_OPCODE_N_TRACE_COLUMNS);
    unsigned **device_lookup_memory_address_to_id_0 = clone_to_device<unsigned *>(lookup_memory_address_to_id_0, 2);
    unsigned **device_lookup_memory_address_to_id_1 = clone_to_device<unsigned *>(lookup_memory_address_to_id_1, 2);
    unsigned **device_lookup_memory_address_to_id_2 = clone_to_device<unsigned *>(lookup_memory_address_to_id_2, 2);
    unsigned **device_lookup_memory_id_to_big_0 = clone_to_device<unsigned *>(lookup_memory_id_to_big_0, 29);
    unsigned **device_lookup_memory_id_to_big_1 = clone_to_device<unsigned *>(lookup_memory_id_to_big_1, 29);
    unsigned **device_lookup_memory_id_to_big_2 = clone_to_device<unsigned *>(lookup_memory_id_to_big_2, 29);
    unsigned **device_lookup_opcodes_0 = clone_to_device<unsigned *>(lookup_opcodes_0, 3);
    unsigned **device_lookup_opcodes_1 = clone_to_device<unsigned *>(lookup_opcodes_1, 3);
    unsigned **device_lookup_range_check_4_4_4_4_0 = clone_to_device<unsigned *>(lookup_range_check_4_4_4_4_0, 4);
    unsigned **device_lookup_range_check_4_4_4_4_1 = clone_to_device<unsigned *>(lookup_range_check_4_4_4_4_1, 4);
    unsigned **device_lookup_range_check_4_4_4_4_2 = clone_to_device<unsigned *>(lookup_range_check_4_4_4_4_2, 4);
    unsigned **device_lookup_verify_instruction_0 = clone_to_device<unsigned *>(lookup_verify_instruction_0, 7);

    unsigned **device_sub_component_inputs_verify_instruction = clone_to_device<unsigned *>(sub_component_inputs_verify_instruction, 7);
    unsigned **device_sub_component_inputs_memory_address_to_id = clone_to_device<unsigned *>(sub_component_inputs_memory_address_to_id, 3);
    unsigned **device_sub_component_inputs_memory_id_to_big = clone_to_device<unsigned *>(sub_component_inputs_memory_id_to_big, 3);
    unsigned **device_sub_component_inputs_range_check_4_4_4_4 = clone_to_device<unsigned *>(sub_component_inputs_range_check_4_4_4_4, 12);

    unsigned **device_qm_31_add_mul_opcode_input = clone_to_device<unsigned *>(qm_31_add_mul_opcode_input, 3);
    unsigned **device_memory_id_to_big_transpose_big_value_ptr = clone_to_device<unsigned *>(memory_id_to_big_transposed_big_values, N_M31_IN_FELT252);

    int block_dim = trace_size < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_qm_31_add_mul_opcode_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_id_to_big_1,
        device_lookup_memory_id_to_big_2,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        device_lookup_range_check_4_4_4_4_0,
        device_lookup_range_check_4_4_4_4_1,
        device_lookup_range_check_4_4_4_4_2,
        device_lookup_verify_instruction_0,
        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        device_sub_component_inputs_range_check_4_4_4_4,
        device_qm_31_add_mul_opcode_input,
        memory_address_to_id_address_to_raw_id,
        device_memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_values,
        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate qm_31_add_mul_opcode base trace");

    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_4_4_4_4_0);
    cuda_free_memory(device_lookup_range_check_4_4_4_4_1);
    cuda_free_memory(device_lookup_range_check_4_4_4_4_2);
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);
    cuda_free_memory(device_sub_component_inputs_range_check_4_4_4_4);
    cuda_free_memory(device_qm_31_add_mul_opcode_input);
    cuda_free_memory(device_memory_id_to_big_transpose_big_value_ptr);
}

// === Interaction Trace Kernels ===

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel_round0(
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
__global__ void generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel(
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
__global__ void generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel_opcodes(
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

__global__ void generate_qm_31_add_mul_opcode_interaction_trace_finalize_col_kernel(
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

__global__ void generate_qm_31_add_mul_opcode_interaction_trace_cumsum_shift(
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

__global__ void generate_qm_31_add_mul_opcode_interaction_trace_coord_prefix_sum(
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

void generate_qm_31_add_mul_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,
    void *range_check_4_4_4_4,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_4_4_4_4_0,
    unsigned **lookup_range_check_4_4_4_4_1,
    unsigned **lookup_range_check_4_4_4_4_2,
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
    RangeCheck_4_4_4_4 *range_check_4_4_4_4_lookup_elements = (RangeCheck_4_4_4_4 *)range_check_4_4_4_4;

    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes_lookup_elements = cuda_malloc<Opcodes>(1);
    VerifyInstruction *device_verify_instruction_lookup_elements = cuda_malloc<VerifyInstruction>(1);
    RangeCheck_4_4_4_4 *device_range_check_4_4_4_4_lookup_elements = cuda_malloc<RangeCheck_4_4_4_4>(1);

    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id_lookup_elements, 1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big_lookup_elements, 1);
    cuda_mem_copy_host_to_device<Opcodes>(opcodes_lookup_elements, device_opcodes_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyInstruction>(verify_instruction_lookup_elements, device_verify_instruction_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_4_4_4_4>(range_check_4_4_4_4_lookup_elements, device_range_check_4_4_4_4_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    unsigned **device_lookup_memory_address_to_id_0 = clone_to_device<unsigned *>(lookup_memory_address_to_id_0, 2);
    unsigned **device_lookup_memory_address_to_id_1 = clone_to_device<unsigned *>(lookup_memory_address_to_id_1, 2);
    unsigned **device_lookup_memory_address_to_id_2 = clone_to_device<unsigned *>(lookup_memory_address_to_id_2, 2);
    unsigned **device_lookup_memory_id_to_big_0 = clone_to_device<unsigned *>(lookup_memory_id_to_big_0, 29);
    unsigned **device_lookup_memory_id_to_big_1 = clone_to_device<unsigned *>(lookup_memory_id_to_big_1, 29);
    unsigned **device_lookup_memory_id_to_big_2 = clone_to_device<unsigned *>(lookup_memory_id_to_big_2, 29);
    unsigned **device_lookup_opcodes_0 = clone_to_device<unsigned *>(lookup_opcodes_0, 3);
    unsigned **device_lookup_opcodes_1 = clone_to_device<unsigned *>(lookup_opcodes_1, 3);
    unsigned **device_lookup_range_check_4_4_4_4_0 = clone_to_device<unsigned *>(lookup_range_check_4_4_4_4_0, 4);
    unsigned **device_lookup_range_check_4_4_4_4_1 = clone_to_device<unsigned *>(lookup_range_check_4_4_4_4_1, 4);
    unsigned **device_lookup_range_check_4_4_4_4_2 = clone_to_device<unsigned *>(lookup_range_check_4_4_4_4_2, 4);
    unsigned **device_lookup_verify_instruction_0 = clone_to_device<unsigned *>(lookup_verify_instruction_0, 7);

    unsigned **device_interaction_traces = clone_to_device<unsigned *>(interaction_trace, 4 * QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);

    timer global_timer;
    global_timer.start("generate qm_31_add_mul_opcode interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For verify_instruction_0 & memory_address_to_id_0
    generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
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
    generate_qm_31_add_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        0, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #1 Interaction trace For memory_id_to_big_0 & range_check_4_4_4_4_0
    block_dim = trace_size < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel<29, 4><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_4_4_4_4_lookup_elements,
        device_lookup_memory_id_to_big_0,
        device_lookup_range_check_4_4_4_4_0,
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
    generate_qm_31_add_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        1, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #2 Interaction trace For memory_address_to_id_1 & memory_id_to_big_1
    block_dim = trace_size < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_id_to_big_1,
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
    generate_qm_31_add_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        2, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #3 Interaction trace For range_check_4_4_4_4_1 & memory_address_to_id_2
    block_dim = trace_size < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel<4, 2><<<num_blocks, block_dim>>>(
        device_range_check_4_4_4_4_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_4_4_4_4_1,
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
    generate_qm_31_add_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        3, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #4 Interaction trace For memory_id_to_big_2 & range_check_4_4_4_4_2
    block_dim = trace_size < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel<29, 4><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_4_4_4_4_lookup_elements,
        device_lookup_memory_id_to_big_2,
        device_lookup_range_check_4_4_4_4_2,
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
    generate_qm_31_add_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        4, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #5 Interaction trace For opcodes_0 & opcodes_1 (with enabler)
    block_dim = trace_size < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_qm_31_add_mul_opcode_interaction_trace_col_gen_kernel_opcodes<3, 3><<<num_blocks, block_dim>>>(
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
    generate_qm_31_add_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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
    generate_qm_31_add_mul_opcode_interaction_trace_cumsum_shift<<<num_blocks, block_dim, 4 * block_dim * sizeof(m31)>>>(
        QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        device_coordinate_sums
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Apply cumsum shift to last column
    generate_qm_31_add_mul_opcode_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        device_coordinate_sums,
        QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Apply inclusive prefix sum to the last column
    inclusive_prefix_sum((m31 *)interaction_trace[4 * QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum((m31 *)interaction_trace[4 * QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum((m31 *)interaction_trace[4 * QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum((m31 *)interaction_trace[4 * QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    // Copy claimed_sum back to host
    cuda_mem_copy_device_to_host<m31>(device_coordinate_sums, (m31 *)claimed_sum, 4);

    global_timer.end("generate qm_31_add_mul_opcode interaction trace");

    // Cleanup
    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_opcodes_lookup_elements);
    cuda_free_memory(device_verify_instruction_lookup_elements);
    cuda_free_memory(device_range_check_4_4_4_4_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_4_4_4_4_0);
    cuda_free_memory(device_lookup_range_check_4_4_4_4_1);
    cuda_free_memory(device_lookup_range_check_4_4_4_4_2);
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_coordinate_sums);
}
