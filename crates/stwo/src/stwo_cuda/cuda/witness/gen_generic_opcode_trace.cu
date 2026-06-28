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

#include "gen_generic_opcode_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

__launch_bounds__(256, 2)
__global__ void generate_generic_opcode_trace_kernel(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_range_check_9_9_0,
    m31 **lookup_range_check_9_9_1,
    m31 **lookup_range_check_9_9_2,
    m31 **lookup_range_check_9_9_3,
    m31 **lookup_range_check_9_9_b_0,
    m31 **lookup_range_check_9_9_b_1,
    m31 **lookup_range_check_9_9_b_2,
    m31 **lookup_range_check_9_9_b_3,
    m31 **lookup_range_check_9_9_c_0,
    m31 **lookup_range_check_9_9_c_1,
    m31 **lookup_range_check_9_9_c_2,
    m31 **lookup_range_check_9_9_c_3,
    m31 **lookup_range_check_9_9_d_0,
    m31 **lookup_range_check_9_9_d_1,
    m31 **lookup_range_check_9_9_d_2,
    m31 **lookup_range_check_9_9_d_3,
    m31 **lookup_range_check_9_9_e_0,
    m31 **lookup_range_check_9_9_e_1,
    m31 **lookup_range_check_9_9_e_2,
    m31 **lookup_range_check_9_9_e_3,
    m31 **lookup_range_check_9_9_f_0,
    m31 **lookup_range_check_9_9_f_1,
    m31 **lookup_range_check_9_9_f_2,
    m31 **lookup_range_check_9_9_f_3,
    m31 **lookup_range_check_9_9_g_0,
    m31 **lookup_range_check_9_9_g_1,
    m31 **lookup_range_check_9_9_h_0,
    m31 **lookup_range_check_9_9_h_1,
    m31 **lookup_range_check_19_0,
    m31 **lookup_range_check_19_1,
    m31 **lookup_range_check_19_2,
    m31 **lookup_range_check_19_3,
    m31 **lookup_range_check_19_b_0,
    m31 **lookup_range_check_19_b_1,
    m31 **lookup_range_check_19_b_2,
    m31 **lookup_range_check_19_b_3,
    m31 **lookup_range_check_19_c_0,
    m31 **lookup_range_check_19_c_1,
    m31 **lookup_range_check_19_c_2,
    m31 **lookup_range_check_19_c_3,
    m31 **lookup_range_check_19_d_0,
    m31 **lookup_range_check_19_d_1,
    m31 **lookup_range_check_19_d_2,
    m31 **lookup_range_check_19_e_0,
    m31 **lookup_range_check_19_e_1,
    m31 **lookup_range_check_19_e_2,
    m31 **lookup_range_check_19_f_0,
    m31 **lookup_range_check_19_f_1,
    m31 **lookup_range_check_19_f_2,
    m31 **lookup_range_check_19_g_0,
    m31 **lookup_range_check_19_g_1,
    m31 **lookup_range_check_19_g_2,
    m31 **lookup_range_check_19_h_0,
    m31 **lookup_range_check_19_h_1,
    m31 **lookup_range_check_19_h_2,
    m31 **lookup_range_check_19_h_3,
    m31 **lookup_range_check_18_0,
    m31 **lookup_range_check_11_0,
    m31 **lookup_verify_instruction_0,

    m31 **sub_component_inputs_verify_instruction,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,

    m31 **generic_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0         = {0};
    const m31 M31_1         = {1};
    const m31 M31_2         = {2};
    const m31 M31_4         = {4};
    const m31 M31_8         = {8};
    const m31 M31_16        = {16};
    const m31 M31_32        = {32};
    const m31 M31_64        = {64};
    const m31 M31_128       = {128};
    const m31 M31_136       = {136};
    const m31 M31_256       = {256};
    const m31 M31_508       = {508};
    const m31 M31_511       = {511};
    const m31 M31_512       = {512};
    const m31 M31_32768     = {32768};
    const m31 M31_65536     = {65536};
    const m31 M31_131072    = {131072};
    const m31 M31_262144    = {262144};
    const m31 M31_1048576   = {1048576};
    const m31 M31_4194304   = {4194304};
    const m31 M31_134217728 = {134217728};
    const m31 M31_536870912 = {536870912};

    const uint16_t UInt16_0 = 0;
    const uint16_t UInt16_1 = 1;
    const uint16_t UInt16_2 = 2;
    const uint16_t UInt16_3 = 3;
    const uint16_t UInt16_4 = 4;
    const uint16_t UInt16_5 = 5;
    const uint16_t UInt16_6 = 6;
    const uint16_t UInt16_7 = 7;
    const uint16_t UInt16_8 = 8;
    const uint16_t UInt16_9 = 9;
    const uint16_t UInt16_10 = 10;
    const uint16_t UInt16_11 = 11;
    const uint16_t UInt16_12 = 12;
    const uint16_t UInt16_13 = 13;
    const uint16_t UInt16_14 = 14;
    const uint16_t UInt16_31 = 31;
    const uint16_t UInt16_127 = 127;
    const uint32_t UInt32_2047 = 2047;

    if (row < trace_size) {
        // Input columns
        m31 input_pc_col0 = generic_opcode_input[0][row];
        traces[0][row] = input_pc_col0;
        m31 input_ap_col1 = generic_opcode_input[1][row];
        traces[1][row] = input_ap_col1;
        m31 input_fp_col2 = generic_opcode_input[2][row];
        traces[2][row] = input_fp_col2;

        // Decode Instruction - read instruction from memory
        m31 mem_id_pc = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_pc_col0,
            &mem_id_pc
        );

        m31 inst_limbs[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            mem_id_pc,
            inst_limbs
        );

        // Extract offsets
        uint16_t offset0_tmp = ((uint16_t)inst_limbs[0]) +
                               ((((uint16_t)inst_limbs[1] & UInt16_127) << UInt16_9));
        m31 offset0_col3 = m31{offset0_tmp};
        traces[3][row] = offset0_col3;

        uint16_t offset1_tmp = (((uint16_t)inst_limbs[1] >> UInt16_7) +
                               (((uint16_t)inst_limbs[2]) << UInt16_2)) +
                               ((((uint16_t)inst_limbs[3] & UInt16_31) << UInt16_11));
        m31 offset1_col4 = m31{offset1_tmp};
        traces[4][row] = offset1_col4;

        uint16_t offset2_tmp = (((uint16_t)inst_limbs[3] >> UInt16_5) +
                               (((uint16_t)inst_limbs[4]) << UInt16_4)) +
                               ((((uint16_t)inst_limbs[5] & UInt16_3) << UInt16_13));
        m31 offset2_col5 = m31{offset2_tmp};
        traces[5][row] = offset2_col5;

        // Extract flags
        uint16_t flags_tmp = ((uint16_t)inst_limbs[5] >> UInt16_3) +
                            (((uint16_t)inst_limbs[6]) << UInt16_6);
        m31 dst_base_fp_col6 = m31{(unsigned)((flags_tmp >> UInt16_0) & UInt16_1)};
        m31 op0_base_fp_col7 = m31{(unsigned)((flags_tmp >> UInt16_1) & UInt16_1)};
        m31 op1_imm_col8 = m31{(unsigned)((flags_tmp >> UInt16_2) & UInt16_1)};
        m31 op1_base_fp_col9 = m31{(unsigned)((flags_tmp >> UInt16_3) & UInt16_1)};
        m31 op1_base_ap_col10 = m31{(unsigned)((flags_tmp >> UInt16_4) & UInt16_1)};
        m31 res_add_col11 = m31{(unsigned)((flags_tmp >> UInt16_5) & UInt16_1)};
        m31 res_mul_col12 = m31{(unsigned)((flags_tmp >> UInt16_6) & UInt16_1)};
        m31 pc_update_jump_col13 = m31{(unsigned)((flags_tmp >> UInt16_7) & UInt16_1)};
        m31 pc_update_jump_rel_col14 = m31{(unsigned)((flags_tmp >> UInt16_8) & UInt16_1)};
        m31 pc_update_jnz_col15 = m31{(unsigned)((flags_tmp >> UInt16_9) & UInt16_1)};
        m31 ap_update_add_col16 = m31{(unsigned)((flags_tmp >> UInt16_10) & UInt16_1)};
        m31 ap_update_add_1_col17 = m31{(unsigned)((flags_tmp >> UInt16_11) & UInt16_1)};
        m31 opcode_call_col18 = m31{(unsigned)((flags_tmp >> UInt16_12) & UInt16_1)};
        m31 opcode_ret_col19 = m31{(unsigned)((flags_tmp >> UInt16_13) & UInt16_1)};
        m31 opcode_assert_eq_col20 = m31{(unsigned)((flags_tmp >> UInt16_14) & UInt16_1)};

        traces[6][row] = dst_base_fp_col6;
        traces[7][row] = op0_base_fp_col7;
        traces[8][row] = op1_imm_col8;
        traces[9][row] = op1_base_fp_col9;
        traces[10][row] = op1_base_ap_col10;
        traces[11][row] = res_add_col11;
        traces[12][row] = res_mul_col12;
        traces[13][row] = pc_update_jump_col13;
        traces[14][row] = pc_update_jump_rel_col14;
        traces[15][row] = pc_update_jnz_col15;
        traces[16][row] = ap_update_add_col16;
        traces[17][row] = ap_update_add_1_col17;
        traces[18][row] = opcode_call_col18;
        traces[19][row] = opcode_ret_col19;
        traces[20][row] = opcode_assert_eq_col20;

        // verify_instruction lookup
        m31 flag0_tmp = {dst_base_fp_col6 * 8 + op0_base_fp_col7 * 16 +
                        op1_imm_col8 * 32 + op1_base_fp_col9 * 64 +
                        op1_base_ap_col10 * 128 + res_add_col11 * 256};
        m31 flag1_tmp = {res_mul_col12 + pc_update_jump_col13 * 2 +
                        pc_update_jump_rel_col14 * 4 + pc_update_jnz_col15 * 8 +
                        ap_update_add_col16 * 16 + ap_update_add_1_col17 * 32 +
                        opcode_call_col18 * 64 + opcode_ret_col19 * 128 +
                        opcode_assert_eq_col20 * 256};

        lookup_verify_instruction_0[0][row] = input_pc_col0;
        lookup_verify_instruction_0[1][row] = offset0_col3;
        lookup_verify_instruction_0[2][row] = offset1_col4;
        lookup_verify_instruction_0[3][row] = offset2_col5;
        lookup_verify_instruction_0[4][row] = flag0_tmp;
        lookup_verify_instruction_0[5][row] = flag1_tmp;
        lookup_verify_instruction_0[6][row] = M31_0;

        sub_component_inputs_verify_instruction[0][row] = input_pc_col0;
        sub_component_inputs_verify_instruction[1][row] = offset0_col3;
        sub_component_inputs_verify_instruction[2][row] = offset1_col4;
        sub_component_inputs_verify_instruction[3][row] = offset2_col5;
        sub_component_inputs_verify_instruction[4][row] = flag0_tmp;
        sub_component_inputs_verify_instruction[5][row] = flag1_tmp;
        sub_component_inputs_verify_instruction[6][row] = M31_0;

        // Decode instruction outputs
        m31 offset0_signed = {offset0_col3 - 32768};
        m31 offset1_signed = {offset1_col4 - 32768};
        m31 offset2_signed = {offset2_col5 - 32768};
        m31 op1_base_op0 = {1 - op1_imm_col8 - op1_base_fp_col9 - op1_base_ap_col10};
        m31 res_op1 = {1 - res_add_col11 - res_mul_col12 - pc_update_jnz_col15};
        m31 pc_update_regular = {1 - pc_update_jump_col13 - pc_update_jump_rel_col14 - pc_update_jnz_col15};
        m31 fp_update_regular = {1 - opcode_call_col18 - opcode_ret_col19};
        m31 inst_size = {1 + op1_imm_col8};

        // Eval Operands
        // dst_src = dst_base_fp * fp + (1 - dst_base_fp) * ap
        m31 dst_src_col21 = {dst_base_fp_col6 * input_fp_col2 +
                            (1 - dst_base_fp_col6) * input_ap_col1};
        traces[21][row] = dst_src_col21;

        // Read dst
        m31 dst_addr = {dst_src_col21 + offset0_signed};
        m31 dst_id_col22 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            dst_addr,
            &dst_id_col22
        );
        traces[22][row] = dst_id_col22;

        lookup_memory_address_to_id_0[0][row] = dst_addr;
        lookup_memory_address_to_id_0[1][row] = dst_id_col22;
        sub_component_inputs_memory_address_to_id[0][row] = dst_addr;

        m31 dst_limbs[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            dst_id_col22,
            dst_limbs
        );

        for (int i = 0; i < 28; i++) {
            traces[23 + i][row] = dst_limbs[i];
        }
        sub_component_inputs_memory_id_to_big[0][row] = dst_id_col22;

        // memory_id_to_big_0 lookup (29 fields)
        lookup_memory_id_to_big_0[0][row] = dst_id_col22;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_0[1 + i][row] = dst_limbs[i];
        }

        // op0_src = op0_base_fp * fp + (1 - op0_base_fp) * ap
        m31 op0_src_col51 = {op0_base_fp_col7 * input_fp_col2 +
                            (1 - op0_base_fp_col7) * input_ap_col1};
        traces[51][row] = op0_src_col51;

        // Read op0
        m31 op0_addr = {op0_src_col51 + offset1_signed};
        m31 op0_id_col52 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op0_addr,
            &op0_id_col52
        );
        traces[52][row] = op0_id_col52;

        lookup_memory_address_to_id_1[0][row] = op0_addr;
        lookup_memory_address_to_id_1[1][row] = op0_id_col52;
        sub_component_inputs_memory_address_to_id[1][row] = op0_addr;

        m31 op0_limbs[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            op0_id_col52,
            op0_limbs
        );

        for (int i = 0; i < 28; i++) {
            traces[53 + i][row] = op0_limbs[i];
        }
        sub_component_inputs_memory_id_to_big[1][row] = op0_id_col52;

        // memory_id_to_big_1 lookup
        lookup_memory_id_to_big_1[0][row] = op0_id_col52;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_1[1 + i][row] = op0_limbs[i];
        }

        // Cond Felt 252 As Addr for op0
        m31 partial_limb_msb_col81 = {((uint16_t)op0_limbs[3] & UInt16_2) >> UInt16_1};
        traces[81][row] = partial_limb_msb_col81;

        m31 op0_as_addr = {op0_limbs[0] + op0_limbs[1] * 512 +
                          op0_limbs[2] * 262144 + op0_limbs[3] * 134217728};

        // op1_src = op1_base_fp * fp + op1_base_ap * ap + op1_imm * pc + op1_base_op0 * op0_as_addr
        m31 op1_src_col82 = {op1_base_fp_col9 * input_fp_col2 +
                            op1_base_ap_col10 * input_ap_col1 +
                            op1_imm_col8 * input_pc_col0 +
                            op1_base_op0 * op0_as_addr};
        traces[82][row] = op1_src_col82;

        // Read op1
        m31 op1_addr = {op1_src_col82 + offset2_signed};
        m31 op1_id_col83 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op1_addr,
            &op1_id_col83
        );
        traces[83][row] = op1_id_col83;

        lookup_memory_address_to_id_2[0][row] = op1_addr;
        lookup_memory_address_to_id_2[1][row] = op1_id_col83;
        sub_component_inputs_memory_address_to_id[2][row] = op1_addr;

        m31 op1_limbs[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            op1_id_col83,
            op1_limbs
        );

        for (int i = 0; i < 28; i++) {
            traces[84 + i][row] = op1_limbs[i];
        }
        sub_component_inputs_memory_id_to_big[2][row] = op1_id_col83;

        // memory_id_to_big_2 lookup
        lookup_memory_id_to_big_2[0][row] = op1_id_col83;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_2[1 + i][row] = op1_limbs[i];
        }

        // Add 252: add_res = op0 + op1
        // Simple per-limb addition (actual implementation needs carry propagation and mod P)
        m31 add_res_limbs[28];
        uint64_t carry = 0;
        for (int i = 0; i < 28; i++) {
            uint64_t sum = (uint64_t)op0_limbs[i] + (uint64_t)op1_limbs[i] + carry;
            add_res_limbs[i] = {(unsigned)(sum & 0x1FF)};
            carry = sum >> 9;
            traces[112 + i][row] = add_res_limbs[i];
        }

        // Range check for add_res
        lookup_range_check_9_9_0[0][row] = add_res_limbs[0];
        lookup_range_check_9_9_0[1][row] = add_res_limbs[1];
        lookup_range_check_9_9_b_0[0][row] = add_res_limbs[2];
        lookup_range_check_9_9_b_0[1][row] = add_res_limbs[3];
        lookup_range_check_9_9_c_0[0][row] = add_res_limbs[4];
        lookup_range_check_9_9_c_0[1][row] = add_res_limbs[5];
        lookup_range_check_9_9_d_0[0][row] = add_res_limbs[6];
        lookup_range_check_9_9_d_0[1][row] = add_res_limbs[7];
        lookup_range_check_9_9_e_0[0][row] = add_res_limbs[8];
        lookup_range_check_9_9_e_0[1][row] = add_res_limbs[9];
        lookup_range_check_9_9_f_0[0][row] = add_res_limbs[10];
        lookup_range_check_9_9_f_0[1][row] = add_res_limbs[11];
        lookup_range_check_9_9_g_0[0][row] = add_res_limbs[12];
        lookup_range_check_9_9_g_0[1][row] = add_res_limbs[13];
        lookup_range_check_9_9_h_0[0][row] = add_res_limbs[14];
        lookup_range_check_9_9_h_0[1][row] = add_res_limbs[15];
        lookup_range_check_9_9_1[0][row] = add_res_limbs[16];
        lookup_range_check_9_9_1[1][row] = add_res_limbs[17];
        lookup_range_check_9_9_b_1[0][row] = add_res_limbs[18];
        lookup_range_check_9_9_b_1[1][row] = add_res_limbs[19];
        lookup_range_check_9_9_c_1[0][row] = add_res_limbs[20];
        lookup_range_check_9_9_c_1[1][row] = add_res_limbs[21];
        lookup_range_check_9_9_d_1[0][row] = add_res_limbs[22];
        lookup_range_check_9_9_d_1[1][row] = add_res_limbs[23];
        lookup_range_check_9_9_e_1[0][row] = add_res_limbs[24];
        lookup_range_check_9_9_e_1[1][row] = add_res_limbs[25];
        lookup_range_check_9_9_f_1[0][row] = add_res_limbs[26];
        lookup_range_check_9_9_f_1[1][row] = add_res_limbs[27];

        // Verify Add 252 - sub_p_bit
        uint16_t sub_p_bit_tmp = UInt16_1 & ((((uint16_t)op0_limbs[0]) ^ ((uint16_t)op1_limbs[0])) ^
                                              ((uint16_t)add_res_limbs[0]));
        m31 sub_p_bit_col140 = {sub_p_bit_tmp};
        traces[140][row] = sub_p_bit_col140;

        // Mul 252: mul_res = op0 * op1 (simplified - needs proper implementation)
        m31 mul_res_limbs[28];
        uint64_t products[56] = {0};
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                products[i + j] += (uint64_t)op0_limbs[i] * (uint64_t)op1_limbs[j];
            }
        }
        for (int i = 0; i < 28; i++) {
            if (i < 55) products[i + 1] += products[i] >> 9;
            mul_res_limbs[i] = {(unsigned)(products[i] & 0x1FF)};
            traces[141 + i][row] = mul_res_limbs[i];
        }

        // Range check for mul_res
        lookup_range_check_9_9_2[0][row] = mul_res_limbs[0];
        lookup_range_check_9_9_2[1][row] = mul_res_limbs[1];
        lookup_range_check_9_9_b_2[0][row] = mul_res_limbs[2];
        lookup_range_check_9_9_b_2[1][row] = mul_res_limbs[3];
        lookup_range_check_9_9_c_2[0][row] = mul_res_limbs[4];
        lookup_range_check_9_9_c_2[1][row] = mul_res_limbs[5];
        lookup_range_check_9_9_d_2[0][row] = mul_res_limbs[6];
        lookup_range_check_9_9_d_2[1][row] = mul_res_limbs[7];
        lookup_range_check_9_9_e_2[0][row] = mul_res_limbs[8];
        lookup_range_check_9_9_e_2[1][row] = mul_res_limbs[9];
        lookup_range_check_9_9_f_2[0][row] = mul_res_limbs[10];
        lookup_range_check_9_9_f_2[1][row] = mul_res_limbs[11];
        lookup_range_check_9_9_g_1[0][row] = mul_res_limbs[12];
        lookup_range_check_9_9_g_1[1][row] = mul_res_limbs[13];
        lookup_range_check_9_9_h_1[0][row] = mul_res_limbs[14];
        lookup_range_check_9_9_h_1[1][row] = mul_res_limbs[15];
        lookup_range_check_9_9_3[0][row] = mul_res_limbs[16];
        lookup_range_check_9_9_3[1][row] = mul_res_limbs[17];
        lookup_range_check_9_9_b_3[0][row] = mul_res_limbs[18];
        lookup_range_check_9_9_b_3[1][row] = mul_res_limbs[19];
        lookup_range_check_9_9_c_3[0][row] = mul_res_limbs[20];
        lookup_range_check_9_9_c_3[1][row] = mul_res_limbs[21];
        lookup_range_check_9_9_d_3[0][row] = mul_res_limbs[22];
        lookup_range_check_9_9_d_3[1][row] = mul_res_limbs[23];
        lookup_range_check_9_9_e_3[0][row] = mul_res_limbs[24];
        lookup_range_check_9_9_e_3[1][row] = mul_res_limbs[25];
        lookup_range_check_9_9_f_3[0][row] = mul_res_limbs[26];
        lookup_range_check_9_9_f_3[1][row] = mul_res_limbs[27];

        // === Verify Mul 252: Double Karatsuba N=7 ===
        // Using signed 64-bit integers to handle signed intermediate values
        int64_t op0[28], op1[28];
        for (int i = 0; i < 28; i++) {
            op0[i] = (int64_t)op0_limbs[i];
            op1[i] = (int64_t)op1_limbs[i];
        }

        // First single Karatsuba: op0[0..6] × op1[0..6] and op0[7..13] × op1[7..13]
        int64_t z0_76[13], z2_77[13];
        // z0: schoolbook multiplication of op0[0..6] × op1[0..6]
        z0_76[0] = op0[0]*op1[0];
        z0_76[1] = op0[0]*op1[1] + op0[1]*op1[0];
        z0_76[2] = op0[0]*op1[2] + op0[1]*op1[1] + op0[2]*op1[0];
        z0_76[3] = op0[0]*op1[3] + op0[1]*op1[2] + op0[2]*op1[1] + op0[3]*op1[0];
        z0_76[4] = op0[0]*op1[4] + op0[1]*op1[3] + op0[2]*op1[2] + op0[3]*op1[1] + op0[4]*op1[0];
        z0_76[5] = op0[0]*op1[5] + op0[1]*op1[4] + op0[2]*op1[3] + op0[3]*op1[2] + op0[4]*op1[1] + op0[5]*op1[0];
        z0_76[6] = op0[0]*op1[6] + op0[1]*op1[5] + op0[2]*op1[4] + op0[3]*op1[3] + op0[4]*op1[2] + op0[5]*op1[1] + op0[6]*op1[0];
        z0_76[7] = op0[1]*op1[6] + op0[2]*op1[5] + op0[3]*op1[4] + op0[4]*op1[3] + op0[5]*op1[2] + op0[6]*op1[1];
        z0_76[8] = op0[2]*op1[6] + op0[3]*op1[5] + op0[4]*op1[4] + op0[5]*op1[3] + op0[6]*op1[2];
        z0_76[9] = op0[3]*op1[6] + op0[4]*op1[5] + op0[5]*op1[4] + op0[6]*op1[3];
        z0_76[10] = op0[4]*op1[6] + op0[5]*op1[5] + op0[6]*op1[4];
        z0_76[11] = op0[5]*op1[6] + op0[6]*op1[5];
        z0_76[12] = op0[6]*op1[6];
        // z2: schoolbook multiplication of op0[7..13] × op1[7..13]
        z2_77[0] = op0[7]*op1[7];
        z2_77[1] = op0[7]*op1[8] + op0[8]*op1[7];
        z2_77[2] = op0[7]*op1[9] + op0[8]*op1[8] + op0[9]*op1[7];
        z2_77[3] = op0[7]*op1[10] + op0[8]*op1[9] + op0[9]*op1[8] + op0[10]*op1[7];
        z2_77[4] = op0[7]*op1[11] + op0[8]*op1[10] + op0[9]*op1[9] + op0[10]*op1[8] + op0[11]*op1[7];
        z2_77[5] = op0[7]*op1[12] + op0[8]*op1[11] + op0[9]*op1[10] + op0[10]*op1[9] + op0[11]*op1[8] + op0[12]*op1[7];
        z2_77[6] = op0[7]*op1[13] + op0[8]*op1[12] + op0[9]*op1[11] + op0[10]*op1[10] + op0[11]*op1[9] + op0[12]*op1[8] + op0[13]*op1[7];
        z2_77[7] = op0[8]*op1[13] + op0[9]*op1[12] + op0[10]*op1[11] + op0[11]*op1[10] + op0[12]*op1[9] + op0[13]*op1[8];
        z2_77[8] = op0[9]*op1[13] + op0[10]*op1[12] + op0[11]*op1[11] + op0[12]*op1[10] + op0[13]*op1[9];
        z2_77[9] = op0[10]*op1[13] + op0[11]*op1[12] + op0[12]*op1[11] + op0[13]*op1[10];
        z2_77[10] = op0[11]*op1[13] + op0[12]*op1[12] + op0[13]*op1[11];
        z2_77[11] = op0[12]*op1[13] + op0[13]*op1[12];
        z2_77[12] = op0[13]*op1[13];
        // x_sum_78 = op0[0..6] + op0[7..13], y_sum_79 = op1[0..6] + op1[7..13]
        int64_t x_sum_78[7], y_sum_79[7];
        for (int i = 0; i < 7; i++) {
            x_sum_78[i] = op0[i] + op0[i+7];
            y_sum_79[i] = op1[i] + op1[i+7];
        }
        // z1_partial = x_sum_78 × y_sum_79 (schoolbook)
        int64_t z1_partial[13];
        z1_partial[0] = x_sum_78[0]*y_sum_79[0];
        z1_partial[1] = x_sum_78[0]*y_sum_79[1] + x_sum_78[1]*y_sum_79[0];
        z1_partial[2] = x_sum_78[0]*y_sum_79[2] + x_sum_78[1]*y_sum_79[1] + x_sum_78[2]*y_sum_79[0];
        z1_partial[3] = x_sum_78[0]*y_sum_79[3] + x_sum_78[1]*y_sum_79[2] + x_sum_78[2]*y_sum_79[1] + x_sum_78[3]*y_sum_79[0];
        z1_partial[4] = x_sum_78[0]*y_sum_79[4] + x_sum_78[1]*y_sum_79[3] + x_sum_78[2]*y_sum_79[2] + x_sum_78[3]*y_sum_79[1] + x_sum_78[4]*y_sum_79[0];
        z1_partial[5] = x_sum_78[0]*y_sum_79[5] + x_sum_78[1]*y_sum_79[4] + x_sum_78[2]*y_sum_79[3] + x_sum_78[3]*y_sum_79[2] + x_sum_78[4]*y_sum_79[1] + x_sum_78[5]*y_sum_79[0];
        z1_partial[6] = x_sum_78[0]*y_sum_79[6] + x_sum_78[1]*y_sum_79[5] + x_sum_78[2]*y_sum_79[4] + x_sum_78[3]*y_sum_79[3] + x_sum_78[4]*y_sum_79[2] + x_sum_78[5]*y_sum_79[1] + x_sum_78[6]*y_sum_79[0];
        z1_partial[7] = x_sum_78[1]*y_sum_79[6] + x_sum_78[2]*y_sum_79[5] + x_sum_78[3]*y_sum_79[4] + x_sum_78[4]*y_sum_79[3] + x_sum_78[5]*y_sum_79[2] + x_sum_78[6]*y_sum_79[1];
        z1_partial[8] = x_sum_78[2]*y_sum_79[6] + x_sum_78[3]*y_sum_79[5] + x_sum_78[4]*y_sum_79[4] + x_sum_78[5]*y_sum_79[3] + x_sum_78[6]*y_sum_79[2];
        z1_partial[9] = x_sum_78[3]*y_sum_79[6] + x_sum_78[4]*y_sum_79[5] + x_sum_78[5]*y_sum_79[4] + x_sum_78[6]*y_sum_79[3];
        z1_partial[10] = x_sum_78[4]*y_sum_79[6] + x_sum_78[5]*y_sum_79[5] + x_sum_78[6]*y_sum_79[4];
        z1_partial[11] = x_sum_78[5]*y_sum_79[6] + x_sum_78[6]*y_sum_79[5];
        z1_partial[12] = x_sum_78[6]*y_sum_79[6];
        // single_karatsuba_80[27]: combines z0_76, z2_77, z1_partial
        int64_t sk80[27];
        for (int i = 0; i < 7; i++) sk80[i] = z0_76[i];
        for (int i = 7; i < 13; i++) sk80[i] = z0_76[i] + (z1_partial[i-7] - z0_76[i-7] - z2_77[i-7]);
        sk80[13] = z1_partial[6] - z0_76[6] - z2_77[6];
        for (int i = 14; i < 20; i++) sk80[i] = z2_77[i-14] + (z1_partial[i-7] - z0_76[i-7] - z2_77[i-7]);
        for (int i = 20; i < 27; i++) sk80[i] = z2_77[i-14];

        // Second single Karatsuba: op0[14..20] × op1[14..20] and op0[21..27] × op1[21..27]
        int64_t z0_81[13], z2_82[13];
        // z0_81: schoolbook of op0[14..20] × op1[14..20]
        z0_81[0] = op0[14]*op1[14];
        z0_81[1] = op0[14]*op1[15] + op0[15]*op1[14];
        z0_81[2] = op0[14]*op1[16] + op0[15]*op1[15] + op0[16]*op1[14];
        z0_81[3] = op0[14]*op1[17] + op0[15]*op1[16] + op0[16]*op1[15] + op0[17]*op1[14];
        z0_81[4] = op0[14]*op1[18] + op0[15]*op1[17] + op0[16]*op1[16] + op0[17]*op1[15] + op0[18]*op1[14];
        z0_81[5] = op0[14]*op1[19] + op0[15]*op1[18] + op0[16]*op1[17] + op0[17]*op1[16] + op0[18]*op1[15] + op0[19]*op1[14];
        z0_81[6] = op0[14]*op1[20] + op0[15]*op1[19] + op0[16]*op1[18] + op0[17]*op1[17] + op0[18]*op1[16] + op0[19]*op1[15] + op0[20]*op1[14];
        z0_81[7] = op0[15]*op1[20] + op0[16]*op1[19] + op0[17]*op1[18] + op0[18]*op1[17] + op0[19]*op1[16] + op0[20]*op1[15];
        z0_81[8] = op0[16]*op1[20] + op0[17]*op1[19] + op0[18]*op1[18] + op0[19]*op1[17] + op0[20]*op1[16];
        z0_81[9] = op0[17]*op1[20] + op0[18]*op1[19] + op0[19]*op1[18] + op0[20]*op1[17];
        z0_81[10] = op0[18]*op1[20] + op0[19]*op1[19] + op0[20]*op1[18];
        z0_81[11] = op0[19]*op1[20] + op0[20]*op1[19];
        z0_81[12] = op0[20]*op1[20];
        // z2_82: schoolbook of op0[21..27] × op1[21..27]
        z2_82[0] = op0[21]*op1[21];
        z2_82[1] = op0[21]*op1[22] + op0[22]*op1[21];
        z2_82[2] = op0[21]*op1[23] + op0[22]*op1[22] + op0[23]*op1[21];
        z2_82[3] = op0[21]*op1[24] + op0[22]*op1[23] + op0[23]*op1[22] + op0[24]*op1[21];
        z2_82[4] = op0[21]*op1[25] + op0[22]*op1[24] + op0[23]*op1[23] + op0[24]*op1[22] + op0[25]*op1[21];
        z2_82[5] = op0[21]*op1[26] + op0[22]*op1[25] + op0[23]*op1[24] + op0[24]*op1[23] + op0[25]*op1[22] + op0[26]*op1[21];
        z2_82[6] = op0[21]*op1[27] + op0[22]*op1[26] + op0[23]*op1[25] + op0[24]*op1[24] + op0[25]*op1[23] + op0[26]*op1[22] + op0[27]*op1[21];
        z2_82[7] = op0[22]*op1[27] + op0[23]*op1[26] + op0[24]*op1[25] + op0[25]*op1[24] + op0[26]*op1[23] + op0[27]*op1[22];
        z2_82[8] = op0[23]*op1[27] + op0[24]*op1[26] + op0[25]*op1[25] + op0[26]*op1[24] + op0[27]*op1[23];
        z2_82[9] = op0[24]*op1[27] + op0[25]*op1[26] + op0[26]*op1[25] + op0[27]*op1[24];
        z2_82[10] = op0[25]*op1[27] + op0[26]*op1[26] + op0[27]*op1[25];
        z2_82[11] = op0[26]*op1[27] + op0[27]*op1[26];
        z2_82[12] = op0[27]*op1[27];
        // x_sum_83, y_sum_84
        int64_t x_sum_83[7], y_sum_84[7];
        for (int i = 0; i < 7; i++) {
            x_sum_83[i] = op0[14+i] + op0[21+i];
            y_sum_84[i] = op1[14+i] + op1[21+i];
        }
        int64_t z1_partial_2[13];
        z1_partial_2[0] = x_sum_83[0]*y_sum_84[0];
        z1_partial_2[1] = x_sum_83[0]*y_sum_84[1] + x_sum_83[1]*y_sum_84[0];
        z1_partial_2[2] = x_sum_83[0]*y_sum_84[2] + x_sum_83[1]*y_sum_84[1] + x_sum_83[2]*y_sum_84[0];
        z1_partial_2[3] = x_sum_83[0]*y_sum_84[3] + x_sum_83[1]*y_sum_84[2] + x_sum_83[2]*y_sum_84[1] + x_sum_83[3]*y_sum_84[0];
        z1_partial_2[4] = x_sum_83[0]*y_sum_84[4] + x_sum_83[1]*y_sum_84[3] + x_sum_83[2]*y_sum_84[2] + x_sum_83[3]*y_sum_84[1] + x_sum_83[4]*y_sum_84[0];
        z1_partial_2[5] = x_sum_83[0]*y_sum_84[5] + x_sum_83[1]*y_sum_84[4] + x_sum_83[2]*y_sum_84[3] + x_sum_83[3]*y_sum_84[2] + x_sum_83[4]*y_sum_84[1] + x_sum_83[5]*y_sum_84[0];
        z1_partial_2[6] = x_sum_83[0]*y_sum_84[6] + x_sum_83[1]*y_sum_84[5] + x_sum_83[2]*y_sum_84[4] + x_sum_83[3]*y_sum_84[3] + x_sum_83[4]*y_sum_84[2] + x_sum_83[5]*y_sum_84[1] + x_sum_83[6]*y_sum_84[0];
        z1_partial_2[7] = x_sum_83[1]*y_sum_84[6] + x_sum_83[2]*y_sum_84[5] + x_sum_83[3]*y_sum_84[4] + x_sum_83[4]*y_sum_84[3] + x_sum_83[5]*y_sum_84[2] + x_sum_83[6]*y_sum_84[1];
        z1_partial_2[8] = x_sum_83[2]*y_sum_84[6] + x_sum_83[3]*y_sum_84[5] + x_sum_83[4]*y_sum_84[4] + x_sum_83[5]*y_sum_84[3] + x_sum_83[6]*y_sum_84[2];
        z1_partial_2[9] = x_sum_83[3]*y_sum_84[6] + x_sum_83[4]*y_sum_84[5] + x_sum_83[5]*y_sum_84[4] + x_sum_83[6]*y_sum_84[3];
        z1_partial_2[10] = x_sum_83[4]*y_sum_84[6] + x_sum_83[5]*y_sum_84[5] + x_sum_83[6]*y_sum_84[4];
        z1_partial_2[11] = x_sum_83[5]*y_sum_84[6] + x_sum_83[6]*y_sum_84[5];
        z1_partial_2[12] = x_sum_83[6]*y_sum_84[6];
        // single_karatsuba_85[27]
        int64_t sk85[27];
        for (int i = 0; i < 7; i++) sk85[i] = z0_81[i];
        for (int i = 7; i < 13; i++) sk85[i] = z0_81[i] + (z1_partial_2[i-7] - z0_81[i-7] - z2_82[i-7]);
        sk85[13] = z1_partial_2[6] - z0_81[6] - z2_82[6];
        for (int i = 14; i < 20; i++) sk85[i] = z2_82[i-14] + (z1_partial_2[i-7] - z0_81[i-7] - z2_82[i-7]);
        for (int i = 20; i < 27; i++) sk85[i] = z2_82[i-14];

        // Third single Karatsuba: (op0[0..13] + op0[14..27]) × (op1[0..13] + op1[14..27])
        int64_t x_sum_86[14], y_sum_87[14];
        for (int i = 0; i < 14; i++) {
            x_sum_86[i] = op0[i] + op0[14+i];
            y_sum_87[i] = op1[i] + op1[14+i];
        }
        // z0_88: schoolbook of x_sum_86[0..6] × y_sum_87[0..6]
        int64_t z0_88[13], z2_89[13];
        z0_88[0] = x_sum_86[0]*y_sum_87[0];
        z0_88[1] = x_sum_86[0]*y_sum_87[1] + x_sum_86[1]*y_sum_87[0];
        z0_88[2] = x_sum_86[0]*y_sum_87[2] + x_sum_86[1]*y_sum_87[1] + x_sum_86[2]*y_sum_87[0];
        z0_88[3] = x_sum_86[0]*y_sum_87[3] + x_sum_86[1]*y_sum_87[2] + x_sum_86[2]*y_sum_87[1] + x_sum_86[3]*y_sum_87[0];
        z0_88[4] = x_sum_86[0]*y_sum_87[4] + x_sum_86[1]*y_sum_87[3] + x_sum_86[2]*y_sum_87[2] + x_sum_86[3]*y_sum_87[1] + x_sum_86[4]*y_sum_87[0];
        z0_88[5] = x_sum_86[0]*y_sum_87[5] + x_sum_86[1]*y_sum_87[4] + x_sum_86[2]*y_sum_87[3] + x_sum_86[3]*y_sum_87[2] + x_sum_86[4]*y_sum_87[1] + x_sum_86[5]*y_sum_87[0];
        z0_88[6] = x_sum_86[0]*y_sum_87[6] + x_sum_86[1]*y_sum_87[5] + x_sum_86[2]*y_sum_87[4] + x_sum_86[3]*y_sum_87[3] + x_sum_86[4]*y_sum_87[2] + x_sum_86[5]*y_sum_87[1] + x_sum_86[6]*y_sum_87[0];
        z0_88[7] = x_sum_86[1]*y_sum_87[6] + x_sum_86[2]*y_sum_87[5] + x_sum_86[3]*y_sum_87[4] + x_sum_86[4]*y_sum_87[3] + x_sum_86[5]*y_sum_87[2] + x_sum_86[6]*y_sum_87[1];
        z0_88[8] = x_sum_86[2]*y_sum_87[6] + x_sum_86[3]*y_sum_87[5] + x_sum_86[4]*y_sum_87[4] + x_sum_86[5]*y_sum_87[3] + x_sum_86[6]*y_sum_87[2];
        z0_88[9] = x_sum_86[3]*y_sum_87[6] + x_sum_86[4]*y_sum_87[5] + x_sum_86[5]*y_sum_87[4] + x_sum_86[6]*y_sum_87[3];
        z0_88[10] = x_sum_86[4]*y_sum_87[6] + x_sum_86[5]*y_sum_87[5] + x_sum_86[6]*y_sum_87[4];
        z0_88[11] = x_sum_86[5]*y_sum_87[6] + x_sum_86[6]*y_sum_87[5];
        z0_88[12] = x_sum_86[6]*y_sum_87[6];
        // z2_89: schoolbook of x_sum_86[7..13] × y_sum_87[7..13]
        z2_89[0] = x_sum_86[7]*y_sum_87[7];
        z2_89[1] = x_sum_86[7]*y_sum_87[8] + x_sum_86[8]*y_sum_87[7];
        z2_89[2] = x_sum_86[7]*y_sum_87[9] + x_sum_86[8]*y_sum_87[8] + x_sum_86[9]*y_sum_87[7];
        z2_89[3] = x_sum_86[7]*y_sum_87[10] + x_sum_86[8]*y_sum_87[9] + x_sum_86[9]*y_sum_87[8] + x_sum_86[10]*y_sum_87[7];
        z2_89[4] = x_sum_86[7]*y_sum_87[11] + x_sum_86[8]*y_sum_87[10] + x_sum_86[9]*y_sum_87[9] + x_sum_86[10]*y_sum_87[8] + x_sum_86[11]*y_sum_87[7];
        z2_89[5] = x_sum_86[7]*y_sum_87[12] + x_sum_86[8]*y_sum_87[11] + x_sum_86[9]*y_sum_87[10] + x_sum_86[10]*y_sum_87[9] + x_sum_86[11]*y_sum_87[8] + x_sum_86[12]*y_sum_87[7];
        z2_89[6] = x_sum_86[7]*y_sum_87[13] + x_sum_86[8]*y_sum_87[12] + x_sum_86[9]*y_sum_87[11] + x_sum_86[10]*y_sum_87[10] + x_sum_86[11]*y_sum_87[9] + x_sum_86[12]*y_sum_87[8] + x_sum_86[13]*y_sum_87[7];
        z2_89[7] = x_sum_86[8]*y_sum_87[13] + x_sum_86[9]*y_sum_87[12] + x_sum_86[10]*y_sum_87[11] + x_sum_86[11]*y_sum_87[10] + x_sum_86[12]*y_sum_87[9] + x_sum_86[13]*y_sum_87[8];
        z2_89[8] = x_sum_86[9]*y_sum_87[13] + x_sum_86[10]*y_sum_87[12] + x_sum_86[11]*y_sum_87[11] + x_sum_86[12]*y_sum_87[10] + x_sum_86[13]*y_sum_87[9];
        z2_89[9] = x_sum_86[10]*y_sum_87[13] + x_sum_86[11]*y_sum_87[12] + x_sum_86[12]*y_sum_87[11] + x_sum_86[13]*y_sum_87[10];
        z2_89[10] = x_sum_86[11]*y_sum_87[13] + x_sum_86[12]*y_sum_87[12] + x_sum_86[13]*y_sum_87[11];
        z2_89[11] = x_sum_86[12]*y_sum_87[13] + x_sum_86[13]*y_sum_87[12];
        z2_89[12] = x_sum_86[13]*y_sum_87[13];
        // x_sum_90, y_sum_91
        int64_t x_sum_90[7], y_sum_91[7];
        for (int i = 0; i < 7; i++) {
            x_sum_90[i] = x_sum_86[i] + x_sum_86[i+7];
            y_sum_91[i] = y_sum_87[i] + y_sum_87[i+7];
        }
        int64_t z1_partial_3[13];
        z1_partial_3[0] = x_sum_90[0]*y_sum_91[0];
        z1_partial_3[1] = x_sum_90[0]*y_sum_91[1] + x_sum_90[1]*y_sum_91[0];
        z1_partial_3[2] = x_sum_90[0]*y_sum_91[2] + x_sum_90[1]*y_sum_91[1] + x_sum_90[2]*y_sum_91[0];
        z1_partial_3[3] = x_sum_90[0]*y_sum_91[3] + x_sum_90[1]*y_sum_91[2] + x_sum_90[2]*y_sum_91[1] + x_sum_90[3]*y_sum_91[0];
        z1_partial_3[4] = x_sum_90[0]*y_sum_91[4] + x_sum_90[1]*y_sum_91[3] + x_sum_90[2]*y_sum_91[2] + x_sum_90[3]*y_sum_91[1] + x_sum_90[4]*y_sum_91[0];
        z1_partial_3[5] = x_sum_90[0]*y_sum_91[5] + x_sum_90[1]*y_sum_91[4] + x_sum_90[2]*y_sum_91[3] + x_sum_90[3]*y_sum_91[2] + x_sum_90[4]*y_sum_91[1] + x_sum_90[5]*y_sum_91[0];
        z1_partial_3[6] = x_sum_90[0]*y_sum_91[6] + x_sum_90[1]*y_sum_91[5] + x_sum_90[2]*y_sum_91[4] + x_sum_90[3]*y_sum_91[3] + x_sum_90[4]*y_sum_91[2] + x_sum_90[5]*y_sum_91[1] + x_sum_90[6]*y_sum_91[0];
        z1_partial_3[7] = x_sum_90[1]*y_sum_91[6] + x_sum_90[2]*y_sum_91[5] + x_sum_90[3]*y_sum_91[4] + x_sum_90[4]*y_sum_91[3] + x_sum_90[5]*y_sum_91[2] + x_sum_90[6]*y_sum_91[1];
        z1_partial_3[8] = x_sum_90[2]*y_sum_91[6] + x_sum_90[3]*y_sum_91[5] + x_sum_90[4]*y_sum_91[4] + x_sum_90[5]*y_sum_91[3] + x_sum_90[6]*y_sum_91[2];
        z1_partial_3[9] = x_sum_90[3]*y_sum_91[6] + x_sum_90[4]*y_sum_91[5] + x_sum_90[5]*y_sum_91[4] + x_sum_90[6]*y_sum_91[3];
        z1_partial_3[10] = x_sum_90[4]*y_sum_91[6] + x_sum_90[5]*y_sum_91[5] + x_sum_90[6]*y_sum_91[4];
        z1_partial_3[11] = x_sum_90[5]*y_sum_91[6] + x_sum_90[6]*y_sum_91[5];
        z1_partial_3[12] = x_sum_90[6]*y_sum_91[6];
        // single_karatsuba_92[27]
        int64_t sk92[27];
        for (int i = 0; i < 7; i++) sk92[i] = z0_88[i];
        for (int i = 7; i < 13; i++) sk92[i] = z0_88[i] + (z1_partial_3[i-7] - z0_88[i-7] - z2_89[i-7]);
        sk92[13] = z1_partial_3[6] - z0_88[6] - z2_89[6];
        for (int i = 14; i < 20; i++) sk92[i] = z2_89[i-14] + (z1_partial_3[i-7] - z0_88[i-7] - z2_89[i-7]);
        for (int i = 20; i < 27; i++) sk92[i] = z2_89[i-14];

        // double_karatsuba_93[41]: combines sk80, sk85, sk92
        int64_t dk93[41];
        for (int i = 0; i < 14; i++) dk93[i] = sk80[i];
        for (int i = 14; i < 27; i++) dk93[i] = sk80[i] + (sk92[i-14] - sk80[i-14] - sk85[i-14]);
        dk93[27] = sk92[13] - sk80[13] - sk85[13];
        for (int i = 28; i < 41; i++) dk93[i] = sk85[i-28] + (sk92[i-14] - sk80[i-14] - sk85[i-14]);
        // Note: Elements 41-54 are sk85[13..26] but we only need up to 54 for conv
        // Full product is 55 elements
        int64_t full_prod[55];
        for (int i = 0; i < 41; i++) full_prod[i] = dk93[i];
        for (int i = 41; i < 55; i++) full_prod[i] = sk85[i - 28];

        // conv_tmp_94[55] = full_prod - mul_res (first 28 elements)
        int64_t conv[55];
        for (int i = 0; i < 28; i++) conv[i] = full_prod[i] - (int64_t)mul_res_limbs[i];
        for (int i = 28; i < 55; i++) conv[i] = full_prod[i];

        // conv_mod[28] = apply modular reduction
        int64_t conv_mod[28];
        conv_mod[0] = 32*conv[0] - 4*conv[21] + 8*conv[49];
        conv_mod[1] = conv[0] + 32*conv[1] - 4*conv[22] + 8*conv[50];
        conv_mod[2] = conv[1] + 32*conv[2] - 4*conv[23] + 8*conv[51];
        conv_mod[3] = conv[2] + 32*conv[3] - 4*conv[24] + 8*conv[52];
        conv_mod[4] = conv[3] + 32*conv[4] - 4*conv[25] + 8*conv[53];
        conv_mod[5] = conv[4] + 32*conv[5] - 4*conv[26] + 8*conv[54];
        conv_mod[6] = conv[5] + 32*conv[6] - 4*conv[27];
        conv_mod[7] = 2*conv[0] + conv[6] + 32*conv[7] - 4*conv[28];
        conv_mod[8] = 2*conv[1] + conv[7] + 32*conv[8] - 4*conv[29];
        conv_mod[9] = 2*conv[2] + conv[8] + 32*conv[9] - 4*conv[30];
        conv_mod[10] = 2*conv[3] + conv[9] + 32*conv[10] - 4*conv[31];
        conv_mod[11] = 2*conv[4] + conv[10] + 32*conv[11] - 4*conv[32];
        conv_mod[12] = 2*conv[5] + conv[11] + 32*conv[12] - 4*conv[33];
        conv_mod[13] = 2*conv[6] + conv[12] + 32*conv[13] - 4*conv[34];
        conv_mod[14] = 2*conv[7] + conv[13] + 32*conv[14] - 4*conv[35];
        conv_mod[15] = 2*conv[8] + conv[14] + 32*conv[15] - 4*conv[36];
        conv_mod[16] = 2*conv[9] + conv[15] + 32*conv[16] - 4*conv[37];
        conv_mod[17] = 2*conv[10] + conv[16] + 32*conv[17] - 4*conv[38];
        conv_mod[18] = 2*conv[11] + conv[17] + 32*conv[18] - 4*conv[39];
        conv_mod[19] = 2*conv[12] + conv[18] + 32*conv[19] - 4*conv[40];
        conv_mod[20] = 2*conv[13] + conv[19] + 32*conv[20] - 4*conv[41];
        conv_mod[21] = 2*conv[14] + conv[20] - 4*conv[42] + 64*conv[49];
        conv_mod[22] = 2*conv[15] - 4*conv[43] + 2*conv[49] + 64*conv[50];
        conv_mod[23] = 2*conv[16] - 4*conv[44] + 2*conv[50] + 64*conv[51];
        conv_mod[24] = 2*conv[17] - 4*conv[45] + 2*conv[51] + 64*conv[52];
        conv_mod[25] = 2*conv[18] - 4*conv[46] + 2*conv[52] + 64*conv[53];
        conv_mod[26] = 2*conv[19] - 4*conv[47] + 2*conv[53] + 64*conv[54];
        conv_mod[27] = 2*conv[20] - 4*conv[48] + 2*conv[54];

        // k_mod_2_18_biased = ((conv_mod[0] + 134217728) + ((conv_mod[1] + 134217728) & 511) << 9 + 65536) & 262143
        // Note: 134217728 = M31_134217728 = 2^27, 65536 = 2^16, 262143 = 2^18-1
        uint32_t conv_mod_0_biased = (uint32_t)(conv_mod[0] + 134217728);
        uint32_t conv_mod_1_biased = (uint32_t)(conv_mod[1] + 134217728);
        uint32_t k_mod_2_18_biased = (conv_mod_0_biased + ((conv_mod_1_biased & 511) << 9) + 65536) & 262143;
        // k_col169 = low_16_bits + (high_2_bits - 1) * 65536
        uint32_t low_16 = k_mod_2_18_biased & 0xFFFF;
        uint32_t high_2 = (k_mod_2_18_biased >> 16) & 0x3;
        int64_t k_val = (int64_t)low_16 + ((int64_t)high_2 - 1) * 65536;
        m31 k_col169 = {(unsigned)((k_val % P + P) % P)};
        traces[169][row] = k_col169;

        // Carry columns (170-196)
        // mul_carry_i = (conv_mod[i] + mul_carry_{i-1}) * M31_4194304
        // Note: M31_4194304 = 2^22 = 1/512 in M31 field (since 512 * 2^22 = 2^31 ≡ 1 mod P)
        // Actually we need division by 512 = right shift by 9 bits
        int64_t mul_carry[27];
        mul_carry[0] = (conv_mod[0] - k_val) / 512;  // = (conv_mod[0] - k) >> 9
        traces[170][row] = {(unsigned)((mul_carry[0] % P + P) % P)};
        for (int i = 1; i < 21; i++) {
            mul_carry[i] = (conv_mod[i] + mul_carry[i-1]) / 512;
            traces[170 + i][row] = {(unsigned)((mul_carry[i] % P + P) % P)};
        }
        // mul_carry_21 has special term: -136 * k
        mul_carry[21] = (conv_mod[21] - 136 * k_val + mul_carry[20]) / 512;
        traces[191][row] = {(unsigned)((mul_carry[21] % P + P) % P)};
        for (int i = 22; i < 27; i++) {
            mul_carry[i] = (conv_mod[i] + mul_carry[i-1]) / 512;
            traces[170 + i][row] = {(unsigned)((mul_carry[i] % P + P) % P)};
        }

        // Set range_check_19 lookups
        lookup_range_check_19_h_0[0][row] = add(k_col169, {262144});
        lookup_range_check_19_0[0][row] = {(unsigned)((mul_carry[0] + 131072) % P + P) % P};
        lookup_range_check_19_b_0[0][row] = {(unsigned)((mul_carry[1] + 131072) % P + P) % P};
        lookup_range_check_19_c_0[0][row] = {(unsigned)((mul_carry[2] + 131072) % P + P) % P};
        lookup_range_check_19_d_0[0][row] = {(unsigned)((mul_carry[3] + 131072) % P + P) % P};
        lookup_range_check_19_e_0[0][row] = {(unsigned)((mul_carry[4] + 131072) % P + P) % P};
        lookup_range_check_19_f_0[0][row] = {(unsigned)((mul_carry[5] + 131072) % P + P) % P};
        lookup_range_check_19_g_0[0][row] = {(unsigned)((mul_carry[6] + 131072) % P + P) % P};
        lookup_range_check_19_h_1[0][row] = {(unsigned)((mul_carry[7] + 131072) % P + P) % P};
        lookup_range_check_19_1[0][row] = {(unsigned)((mul_carry[8] + 131072) % P + P) % P};
        lookup_range_check_19_b_1[0][row] = {(unsigned)((mul_carry[9] + 131072) % P + P) % P};
        lookup_range_check_19_c_1[0][row] = {(unsigned)((mul_carry[10] + 131072) % P + P) % P};
        lookup_range_check_19_d_1[0][row] = {(unsigned)((mul_carry[11] + 131072) % P + P) % P};
        lookup_range_check_19_e_1[0][row] = {(unsigned)((mul_carry[12] + 131072) % P + P) % P};
        lookup_range_check_19_f_1[0][row] = {(unsigned)((mul_carry[13] + 131072) % P + P) % P};
        lookup_range_check_19_g_1[0][row] = {(unsigned)((mul_carry[14] + 131072) % P + P) % P};
        lookup_range_check_19_h_2[0][row] = {(unsigned)((mul_carry[15] + 131072) % P + P) % P};
        lookup_range_check_19_2[0][row] = {(unsigned)((mul_carry[16] + 131072) % P + P) % P};
        lookup_range_check_19_b_2[0][row] = {(unsigned)((mul_carry[17] + 131072) % P + P) % P};
        lookup_range_check_19_c_2[0][row] = {(unsigned)((mul_carry[18] + 131072) % P + P) % P};
        lookup_range_check_19_d_2[0][row] = {(unsigned)((mul_carry[19] + 131072) % P + P) % P};
        lookup_range_check_19_e_2[0][row] = {(unsigned)((mul_carry[20] + 131072) % P + P) % P};
        lookup_range_check_19_f_2[0][row] = {(unsigned)((mul_carry[21] + 131072) % P + P) % P};
        lookup_range_check_19_g_2[0][row] = {(unsigned)((mul_carry[22] + 131072) % P + P) % P};
        lookup_range_check_19_h_3[0][row] = {(unsigned)((mul_carry[23] + 131072) % P + P) % P};
        lookup_range_check_19_3[0][row] = {(unsigned)((mul_carry[24] + 131072) % P + P) % P};
        lookup_range_check_19_b_3[0][row] = {(unsigned)((mul_carry[25] + 131072) % P + P) % P};
        lookup_range_check_19_c_3[0][row] = {(unsigned)((mul_carry[26] + 131072) % P + P) % P};

        // res = res_op1 * op1 + res_mul * mul_res + res_add * add_res
        m31 res_limbs[28];
        for (int i = 0; i < 28; i++) {
            res_limbs[i] = {res_op1 * op1_limbs[i] +
                           res_mul_col12 * mul_res_limbs[i] +
                           res_add_col11 * add_res_limbs[i]};
            traces[197 + i][row] = res_limbs[i];
        }

        // Handle Opcodes columns (225-242)
        m31 partial_limb_msb_col225 = {((uint16_t)dst_limbs[3] & UInt16_2) >> UInt16_1};
        traces[225][row] = partial_limb_msb_col225;
        traces[226][row] = partial_limb_msb_col81;

        m31 partial_limb_msb_col227 = {((uint16_t)res_limbs[3] & UInt16_2) >> UInt16_1};
        traces[227][row] = partial_limb_msb_col227;
        traces[228][row] = partial_limb_msb_col225;

        m31 msb_col229 = (res_limbs[27] == 256) ? M31_1 : M31_0;
        traces[229][row] = msb_col229;
        m31 mid_limbs_set_col230 = (res_limbs[20] == 511) ? M31_1 : M31_0;
        traces[230][row] = mid_limbs_set_col230;

        m31 remainder_bits_res = {res_limbs[3] - mid_limbs_set_col230 * 508};
        m31 partial_limb_msb_col231 = {((uint16_t)remainder_bits_res & UInt16_2) >> UInt16_1};
        traces[231][row] = partial_limb_msb_col231;

        // dst_sum for jnz
        unsigned dst_sum = 0;
        for (int i = 0; i < 28; i++) {
            dst_sum += dst_limbs[i];
        }
        m31 dst_is_zero = (dst_sum == 0) ? M31_1 : M31_0;

        // Compute dst_sum_squares_inv_col232:
        // inverse of ((dst_0-1)^2 + dst_1..dst_20 + (dst_21-136)^2 + dst_22..dst_26 + (dst_27-256)^2)
        m31 diff_from_p_0 = sub({dst_limbs[0]}, M31_1);
        m31 diff_from_p_21 = sub({dst_limbs[21]}, {136});
        m31 diff_from_p_27 = sub({dst_limbs[27]}, {256});

        m31 dst_sum_squares = mul(diff_from_p_0, diff_from_p_0);
        for (int i = 1; i <= 20; i++) {
            dst_sum_squares = add(dst_sum_squares, {dst_limbs[i]});
        }
        dst_sum_squares = add(dst_sum_squares, mul(diff_from_p_21, diff_from_p_21));
        for (int i = 22; i <= 26; i++) {
            dst_sum_squares = add(dst_sum_squares, {dst_limbs[i]});
        }
        dst_sum_squares = add(dst_sum_squares, mul(diff_from_p_27, diff_from_p_27));
        traces[232][row] = inv(dst_sum_squares);

        // Compute dst_sum_inv_col233: inverse of (dst_sum + dst_is_zero)
        m31 dst_sum_m31 = {dst_sum};
        m31 dst_sum_plus_is_zero = add(dst_sum_m31, dst_is_zero);
        traces[233][row] = inv(dst_sum_plus_is_zero);

        m31 op1_as_rel_imm_cond_col234 = {pc_update_jnz_col15 * dst_sum};
        traces[234][row] = op1_as_rel_imm_cond_col234;

        m31 msb_col235 = (op1_limbs[27] == 256) ? M31_1 : M31_0;
        traces[235][row] = msb_col235;
        m31 mid_limbs_set_col236 = (op1_limbs[20] == 511) ? M31_1 : M31_0;
        traces[236][row] = mid_limbs_set_col236;

        m31 remainder_bits_op1 = {op1_limbs[3] - mid_limbs_set_col236 * 508};
        m31 partial_limb_msb_col237 = {((uint16_t)remainder_bits_op1 & UInt16_2) >> UInt16_1};
        traces[237][row] = partial_limb_msb_col237;

        // Compute addresses and relative immediates
        m31 res_as_addr = {res_limbs[0] + res_limbs[1] * 512 +
                          res_limbs[2] * 262144 + res_limbs[3] * 134217728};
        m31 res_as_rel_imm = {res_limbs[0] + res_limbs[1] * 512 +
                             res_limbs[2] * 262144 + remainder_bits_res * 134217728 -
                             msb_col229 - 536870912 * mid_limbs_set_col230};
        m31 op1_as_rel_imm = {op1_limbs[0] + op1_limbs[1] * 512 +
                             op1_limbs[2] * 262144 + remainder_bits_op1 * 134217728 -
                             msb_col235 - 536870912 * mid_limbs_set_col236};
        m31 dst_as_addr = {dst_limbs[0] + dst_limbs[1] * 512 +
                          dst_limbs[2] * 262144 + dst_limbs[3] * 134217728};

        // next_pc_jnz
        m31 next_pc_jnz_col238 = {dst_is_zero * (input_pc_col0 + inst_size) +
                                  (1 - dst_is_zero) * (input_pc_col0 + op1_as_rel_imm)};
        traces[238][row] = next_pc_jnz_col238;

        // next_pc
        m31 next_pc_col239 = {pc_update_regular * (input_pc_col0 + inst_size) +
                              pc_update_jump_col13 * res_as_addr +
                              pc_update_jump_rel_col14 * (input_pc_col0 + res_as_rel_imm) +
                              pc_update_jnz_col15 * next_pc_jnz_col238};
        traces[239][row] = next_pc_col239;

        // next_ap
        m31 next_ap_col240 = {input_ap_col1 + ap_update_add_col16 * res_as_rel_imm +
                              ap_update_add_1_col17 + opcode_call_col18 * 2};
        traces[240][row] = next_ap_col240;

        // Range check AP
        m31 range_check_ap_bot11bits_col241 = {((uint32_t)next_ap_col240) & UInt32_2047};
        traces[241][row] = range_check_ap_bot11bits_col241;

        lookup_range_check_18_0[0][row] = mul(next_ap_col240 - range_check_ap_bot11bits_col241, 1048576);
        lookup_range_check_11_0[0][row] = range_check_ap_bot11bits_col241;

        // next_fp
        m31 next_fp_col242 = {fp_update_regular * input_fp_col2 +
                              opcode_ret_col19 * dst_as_addr +
                              opcode_call_col18 * (input_ap_col1 + 2)};
        traces[242][row] = next_fp_col242;

        // opcodes lookups
        lookup_opcodes_0[0][row] = input_pc_col0;
        lookup_opcodes_0[1][row] = input_ap_col1;
        lookup_opcodes_0[2][row] = input_fp_col2;
        lookup_opcodes_1[0][row] = next_pc_col239;
        lookup_opcodes_1[1][row] = next_ap_col240;
        lookup_opcodes_1[2][row] = next_fp_col242;

        // enabler
        m31 enabler_col243 = (row < n_rows) ? M31_1 : M31_0;
        traces[243][row] = enabler_col243;
    }
}

extern "C"
void generate_generic_opcode_traces(
    unsigned **traces,
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_9_9_0,
    unsigned **lookup_range_check_9_9_1,
    unsigned **lookup_range_check_9_9_2,
    unsigned **lookup_range_check_9_9_3,
    unsigned **lookup_range_check_9_9_b_0,
    unsigned **lookup_range_check_9_9_b_1,
    unsigned **lookup_range_check_9_9_b_2,
    unsigned **lookup_range_check_9_9_b_3,
    unsigned **lookup_range_check_9_9_c_0,
    unsigned **lookup_range_check_9_9_c_1,
    unsigned **lookup_range_check_9_9_c_2,
    unsigned **lookup_range_check_9_9_c_3,
    unsigned **lookup_range_check_9_9_d_0,
    unsigned **lookup_range_check_9_9_d_1,
    unsigned **lookup_range_check_9_9_d_2,
    unsigned **lookup_range_check_9_9_d_3,
    unsigned **lookup_range_check_9_9_e_0,
    unsigned **lookup_range_check_9_9_e_1,
    unsigned **lookup_range_check_9_9_e_2,
    unsigned **lookup_range_check_9_9_e_3,
    unsigned **lookup_range_check_9_9_f_0,
    unsigned **lookup_range_check_9_9_f_1,
    unsigned **lookup_range_check_9_9_f_2,
    unsigned **lookup_range_check_9_9_f_3,
    unsigned **lookup_range_check_9_9_g_0,
    unsigned **lookup_range_check_9_9_g_1,
    unsigned **lookup_range_check_9_9_h_0,
    unsigned **lookup_range_check_9_9_h_1,
    unsigned **lookup_range_check_19_0,
    unsigned **lookup_range_check_19_1,
    unsigned **lookup_range_check_19_2,
    unsigned **lookup_range_check_19_3,
    unsigned **lookup_range_check_19_b_0,
    unsigned **lookup_range_check_19_b_1,
    unsigned **lookup_range_check_19_b_2,
    unsigned **lookup_range_check_19_b_3,
    unsigned **lookup_range_check_19_c_0,
    unsigned **lookup_range_check_19_c_1,
    unsigned **lookup_range_check_19_c_2,
    unsigned **lookup_range_check_19_c_3,
    unsigned **lookup_range_check_19_d_0,
    unsigned **lookup_range_check_19_d_1,
    unsigned **lookup_range_check_19_d_2,
    unsigned **lookup_range_check_19_e_0,
    unsigned **lookup_range_check_19_e_1,
    unsigned **lookup_range_check_19_e_2,
    unsigned **lookup_range_check_19_f_0,
    unsigned **lookup_range_check_19_f_1,
    unsigned **lookup_range_check_19_f_2,
    unsigned **lookup_range_check_19_g_0,
    unsigned **lookup_range_check_19_g_1,
    unsigned **lookup_range_check_19_g_2,
    unsigned **lookup_range_check_19_h_0,
    unsigned **lookup_range_check_19_h_1,
    unsigned **lookup_range_check_19_h_2,
    unsigned **lookup_range_check_19_h_3,
    unsigned **lookup_range_check_18_0,
    unsigned **lookup_range_check_11_0,
    unsigned **lookup_verify_instruction_0,
    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **generic_opcode_input,
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,
    unsigned n_rows,
    unsigned log_size
) {
    unsigned trace_size = 1u << log_size;
    unsigned block_size = GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    unsigned num_blocks = (trace_size + block_size - 1) / block_size;

    // Clone all pointer arrays to device memory (host->device copy of pointer arrays)
    m31 **device_traces = clone_to_device<m31*>(traces, GENERIC_OPCODE_N_TRACE_COLUMNS);
    m31 **device_generic_opcode_input = clone_to_device<m31*>(generic_opcode_input, 3);
    unsigned **device_memory_id_to_big_transposed_big_values = clone_to_device<unsigned*>(memory_id_to_big_transposed_big_values, 8);

    // Lookup data - memory_address_to_id (3 lookups × 2 fields)
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);

    // Lookup data - memory_id_to_big (3 lookups × 29 fields)
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);

    // Lookup data - opcodes (2 lookups × 3 fields)
    m31 **device_lookup_opcodes_0 = clone_to_device<m31*>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31*>(lookup_opcodes_1, 3);

    // Lookup data - range_check_9_9 (4 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_0 = clone_to_device<m31*>(lookup_range_check_9_9_0, 2);
    m31 **device_lookup_range_check_9_9_1 = clone_to_device<m31*>(lookup_range_check_9_9_1, 2);
    m31 **device_lookup_range_check_9_9_2 = clone_to_device<m31*>(lookup_range_check_9_9_2, 2);
    m31 **device_lookup_range_check_9_9_3 = clone_to_device<m31*>(lookup_range_check_9_9_3, 2);

    // Lookup data - range_check_9_9_b (4 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_b_0 = clone_to_device<m31*>(lookup_range_check_9_9_b_0, 2);
    m31 **device_lookup_range_check_9_9_b_1 = clone_to_device<m31*>(lookup_range_check_9_9_b_1, 2);
    m31 **device_lookup_range_check_9_9_b_2 = clone_to_device<m31*>(lookup_range_check_9_9_b_2, 2);
    m31 **device_lookup_range_check_9_9_b_3 = clone_to_device<m31*>(lookup_range_check_9_9_b_3, 2);

    // Lookup data - range_check_9_9_c (4 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_c_0 = clone_to_device<m31*>(lookup_range_check_9_9_c_0, 2);
    m31 **device_lookup_range_check_9_9_c_1 = clone_to_device<m31*>(lookup_range_check_9_9_c_1, 2);
    m31 **device_lookup_range_check_9_9_c_2 = clone_to_device<m31*>(lookup_range_check_9_9_c_2, 2);
    m31 **device_lookup_range_check_9_9_c_3 = clone_to_device<m31*>(lookup_range_check_9_9_c_3, 2);

    // Lookup data - range_check_9_9_d (4 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_d_0 = clone_to_device<m31*>(lookup_range_check_9_9_d_0, 2);
    m31 **device_lookup_range_check_9_9_d_1 = clone_to_device<m31*>(lookup_range_check_9_9_d_1, 2);
    m31 **device_lookup_range_check_9_9_d_2 = clone_to_device<m31*>(lookup_range_check_9_9_d_2, 2);
    m31 **device_lookup_range_check_9_9_d_3 = clone_to_device<m31*>(lookup_range_check_9_9_d_3, 2);

    // Lookup data - range_check_9_9_e (4 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_e_0 = clone_to_device<m31*>(lookup_range_check_9_9_e_0, 2);
    m31 **device_lookup_range_check_9_9_e_1 = clone_to_device<m31*>(lookup_range_check_9_9_e_1, 2);
    m31 **device_lookup_range_check_9_9_e_2 = clone_to_device<m31*>(lookup_range_check_9_9_e_2, 2);
    m31 **device_lookup_range_check_9_9_e_3 = clone_to_device<m31*>(lookup_range_check_9_9_e_3, 2);

    // Lookup data - range_check_9_9_f (4 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_f_0 = clone_to_device<m31*>(lookup_range_check_9_9_f_0, 2);
    m31 **device_lookup_range_check_9_9_f_1 = clone_to_device<m31*>(lookup_range_check_9_9_f_1, 2);
    m31 **device_lookup_range_check_9_9_f_2 = clone_to_device<m31*>(lookup_range_check_9_9_f_2, 2);
    m31 **device_lookup_range_check_9_9_f_3 = clone_to_device<m31*>(lookup_range_check_9_9_f_3, 2);

    // Lookup data - range_check_9_9_g (2 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_g_0 = clone_to_device<m31*>(lookup_range_check_9_9_g_0, 2);
    m31 **device_lookup_range_check_9_9_g_1 = clone_to_device<m31*>(lookup_range_check_9_9_g_1, 2);

    // Lookup data - range_check_9_9_h (2 lookups × 2 fields)
    m31 **device_lookup_range_check_9_9_h_0 = clone_to_device<m31*>(lookup_range_check_9_9_h_0, 2);
    m31 **device_lookup_range_check_9_9_h_1 = clone_to_device<m31*>(lookup_range_check_9_9_h_1, 2);

    // Lookup data - range_check_19 (4 lookups × 1 field)
    m31 **device_lookup_range_check_19_0 = clone_to_device<m31*>(lookup_range_check_19_0, 1);
    m31 **device_lookup_range_check_19_1 = clone_to_device<m31*>(lookup_range_check_19_1, 1);
    m31 **device_lookup_range_check_19_2 = clone_to_device<m31*>(lookup_range_check_19_2, 1);
    m31 **device_lookup_range_check_19_3 = clone_to_device<m31*>(lookup_range_check_19_3, 1);

    // Lookup data - range_check_19_b (4 lookups × 1 field)
    m31 **device_lookup_range_check_19_b_0 = clone_to_device<m31*>(lookup_range_check_19_b_0, 1);
    m31 **device_lookup_range_check_19_b_1 = clone_to_device<m31*>(lookup_range_check_19_b_1, 1);
    m31 **device_lookup_range_check_19_b_2 = clone_to_device<m31*>(lookup_range_check_19_b_2, 1);
    m31 **device_lookup_range_check_19_b_3 = clone_to_device<m31*>(lookup_range_check_19_b_3, 1);

    // Lookup data - range_check_19_c (4 lookups × 1 field)
    m31 **device_lookup_range_check_19_c_0 = clone_to_device<m31*>(lookup_range_check_19_c_0, 1);
    m31 **device_lookup_range_check_19_c_1 = clone_to_device<m31*>(lookup_range_check_19_c_1, 1);
    m31 **device_lookup_range_check_19_c_2 = clone_to_device<m31*>(lookup_range_check_19_c_2, 1);
    m31 **device_lookup_range_check_19_c_3 = clone_to_device<m31*>(lookup_range_check_19_c_3, 1);

    // Lookup data - range_check_19_d (3 lookups × 1 field)
    m31 **device_lookup_range_check_19_d_0 = clone_to_device<m31*>(lookup_range_check_19_d_0, 1);
    m31 **device_lookup_range_check_19_d_1 = clone_to_device<m31*>(lookup_range_check_19_d_1, 1);
    m31 **device_lookup_range_check_19_d_2 = clone_to_device<m31*>(lookup_range_check_19_d_2, 1);

    // Lookup data - range_check_19_e (3 lookups × 1 field)
    m31 **device_lookup_range_check_19_e_0 = clone_to_device<m31*>(lookup_range_check_19_e_0, 1);
    m31 **device_lookup_range_check_19_e_1 = clone_to_device<m31*>(lookup_range_check_19_e_1, 1);
    m31 **device_lookup_range_check_19_e_2 = clone_to_device<m31*>(lookup_range_check_19_e_2, 1);

    // Lookup data - range_check_19_f (3 lookups × 1 field)
    m31 **device_lookup_range_check_19_f_0 = clone_to_device<m31*>(lookup_range_check_19_f_0, 1);
    m31 **device_lookup_range_check_19_f_1 = clone_to_device<m31*>(lookup_range_check_19_f_1, 1);
    m31 **device_lookup_range_check_19_f_2 = clone_to_device<m31*>(lookup_range_check_19_f_2, 1);

    // Lookup data - range_check_19_g (3 lookups × 1 field)
    m31 **device_lookup_range_check_19_g_0 = clone_to_device<m31*>(lookup_range_check_19_g_0, 1);
    m31 **device_lookup_range_check_19_g_1 = clone_to_device<m31*>(lookup_range_check_19_g_1, 1);
    m31 **device_lookup_range_check_19_g_2 = clone_to_device<m31*>(lookup_range_check_19_g_2, 1);

    // Lookup data - range_check_19_h (4 lookups × 1 field)
    m31 **device_lookup_range_check_19_h_0 = clone_to_device<m31*>(lookup_range_check_19_h_0, 1);
    m31 **device_lookup_range_check_19_h_1 = clone_to_device<m31*>(lookup_range_check_19_h_1, 1);
    m31 **device_lookup_range_check_19_h_2 = clone_to_device<m31*>(lookup_range_check_19_h_2, 1);
    m31 **device_lookup_range_check_19_h_3 = clone_to_device<m31*>(lookup_range_check_19_h_3, 1);

    // Lookup data - range_check_18 (1 lookup × 1 field)
    m31 **device_lookup_range_check_18_0 = clone_to_device<m31*>(lookup_range_check_18_0, 1);

    // Lookup data - range_check_11 (1 lookup × 1 field)
    m31 **device_lookup_range_check_11_0 = clone_to_device<m31*>(lookup_range_check_11_0, 1);

    // Lookup data - verify_instruction (1 lookup × 7 fields)
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31*>(lookup_verify_instruction_0, 7);

    // Sub-component inputs
    m31 **device_sub_component_inputs_verify_instruction = clone_to_device<m31*>(sub_component_inputs_verify_instruction, 7);
    m31 **device_sub_component_inputs_memory_address_to_id = clone_to_device<m31*>(sub_component_inputs_memory_address_to_id, 3);
    m31 **device_sub_component_inputs_memory_id_to_big = clone_to_device<m31*>(sub_component_inputs_memory_id_to_big, 3);

    timer global_timer;
    global_timer.start("generate generic_opcode base trace");

    generate_generic_opcode_trace_kernel<<<num_blocks, block_size>>>(
        device_traces,
        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_id_to_big_1,
        device_lookup_memory_id_to_big_2,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        device_lookup_range_check_9_9_0,
        device_lookup_range_check_9_9_1,
        device_lookup_range_check_9_9_2,
        device_lookup_range_check_9_9_3,
        device_lookup_range_check_9_9_b_0,
        device_lookup_range_check_9_9_b_1,
        device_lookup_range_check_9_9_b_2,
        device_lookup_range_check_9_9_b_3,
        device_lookup_range_check_9_9_c_0,
        device_lookup_range_check_9_9_c_1,
        device_lookup_range_check_9_9_c_2,
        device_lookup_range_check_9_9_c_3,
        device_lookup_range_check_9_9_d_0,
        device_lookup_range_check_9_9_d_1,
        device_lookup_range_check_9_9_d_2,
        device_lookup_range_check_9_9_d_3,
        device_lookup_range_check_9_9_e_0,
        device_lookup_range_check_9_9_e_1,
        device_lookup_range_check_9_9_e_2,
        device_lookup_range_check_9_9_e_3,
        device_lookup_range_check_9_9_f_0,
        device_lookup_range_check_9_9_f_1,
        device_lookup_range_check_9_9_f_2,
        device_lookup_range_check_9_9_f_3,
        device_lookup_range_check_9_9_g_0,
        device_lookup_range_check_9_9_g_1,
        device_lookup_range_check_9_9_h_0,
        device_lookup_range_check_9_9_h_1,
        device_lookup_range_check_19_0,
        device_lookup_range_check_19_1,
        device_lookup_range_check_19_2,
        device_lookup_range_check_19_3,
        device_lookup_range_check_19_b_0,
        device_lookup_range_check_19_b_1,
        device_lookup_range_check_19_b_2,
        device_lookup_range_check_19_b_3,
        device_lookup_range_check_19_c_0,
        device_lookup_range_check_19_c_1,
        device_lookup_range_check_19_c_2,
        device_lookup_range_check_19_c_3,
        device_lookup_range_check_19_d_0,
        device_lookup_range_check_19_d_1,
        device_lookup_range_check_19_d_2,
        device_lookup_range_check_19_e_0,
        device_lookup_range_check_19_e_1,
        device_lookup_range_check_19_e_2,
        device_lookup_range_check_19_f_0,
        device_lookup_range_check_19_f_1,
        device_lookup_range_check_19_f_2,
        device_lookup_range_check_19_g_0,
        device_lookup_range_check_19_g_1,
        device_lookup_range_check_19_g_2,
        device_lookup_range_check_19_h_0,
        device_lookup_range_check_19_h_1,
        device_lookup_range_check_19_h_2,
        device_lookup_range_check_19_h_3,
        device_lookup_range_check_18_0,
        device_lookup_range_check_11_0,
        device_lookup_verify_instruction_0,
        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        device_generic_opcode_input,
        memory_address_to_id_address_to_raw_id,
        device_memory_id_to_big_transposed_big_values,
        memory_id_to_big_small_values,
        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate generic_opcode base trace");

    // Free cloned device arrays
    cuda_free_memory(device_traces);
    cuda_free_memory(device_generic_opcode_input);
    cuda_free_memory(device_memory_id_to_big_transposed_big_values);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_9_9_0);
    cuda_free_memory(device_lookup_range_check_9_9_1);
    cuda_free_memory(device_lookup_range_check_9_9_2);
    cuda_free_memory(device_lookup_range_check_9_9_3);
    cuda_free_memory(device_lookup_range_check_9_9_b_0);
    cuda_free_memory(device_lookup_range_check_9_9_b_1);
    cuda_free_memory(device_lookup_range_check_9_9_b_2);
    cuda_free_memory(device_lookup_range_check_9_9_b_3);
    cuda_free_memory(device_lookup_range_check_9_9_c_0);
    cuda_free_memory(device_lookup_range_check_9_9_c_1);
    cuda_free_memory(device_lookup_range_check_9_9_c_2);
    cuda_free_memory(device_lookup_range_check_9_9_c_3);
    cuda_free_memory(device_lookup_range_check_9_9_d_0);
    cuda_free_memory(device_lookup_range_check_9_9_d_1);
    cuda_free_memory(device_lookup_range_check_9_9_d_2);
    cuda_free_memory(device_lookup_range_check_9_9_d_3);
    cuda_free_memory(device_lookup_range_check_9_9_e_0);
    cuda_free_memory(device_lookup_range_check_9_9_e_1);
    cuda_free_memory(device_lookup_range_check_9_9_e_2);
    cuda_free_memory(device_lookup_range_check_9_9_e_3);
    cuda_free_memory(device_lookup_range_check_9_9_f_0);
    cuda_free_memory(device_lookup_range_check_9_9_f_1);
    cuda_free_memory(device_lookup_range_check_9_9_f_2);
    cuda_free_memory(device_lookup_range_check_9_9_f_3);
    cuda_free_memory(device_lookup_range_check_9_9_g_0);
    cuda_free_memory(device_lookup_range_check_9_9_g_1);
    cuda_free_memory(device_lookup_range_check_9_9_h_0);
    cuda_free_memory(device_lookup_range_check_9_9_h_1);
    cuda_free_memory(device_lookup_range_check_19_0);
    cuda_free_memory(device_lookup_range_check_19_1);
    cuda_free_memory(device_lookup_range_check_19_2);
    cuda_free_memory(device_lookup_range_check_19_3);
    cuda_free_memory(device_lookup_range_check_19_b_0);
    cuda_free_memory(device_lookup_range_check_19_b_1);
    cuda_free_memory(device_lookup_range_check_19_b_2);
    cuda_free_memory(device_lookup_range_check_19_b_3);
    cuda_free_memory(device_lookup_range_check_19_c_0);
    cuda_free_memory(device_lookup_range_check_19_c_1);
    cuda_free_memory(device_lookup_range_check_19_c_2);
    cuda_free_memory(device_lookup_range_check_19_c_3);
    cuda_free_memory(device_lookup_range_check_19_d_0);
    cuda_free_memory(device_lookup_range_check_19_d_1);
    cuda_free_memory(device_lookup_range_check_19_d_2);
    cuda_free_memory(device_lookup_range_check_19_e_0);
    cuda_free_memory(device_lookup_range_check_19_e_1);
    cuda_free_memory(device_lookup_range_check_19_e_2);
    cuda_free_memory(device_lookup_range_check_19_f_0);
    cuda_free_memory(device_lookup_range_check_19_f_1);
    cuda_free_memory(device_lookup_range_check_19_f_2);
    cuda_free_memory(device_lookup_range_check_19_g_0);
    cuda_free_memory(device_lookup_range_check_19_g_1);
    cuda_free_memory(device_lookup_range_check_19_g_2);
    cuda_free_memory(device_lookup_range_check_19_h_0);
    cuda_free_memory(device_lookup_range_check_19_h_1);
    cuda_free_memory(device_lookup_range_check_19_h_2);
    cuda_free_memory(device_lookup_range_check_19_h_3);
    cuda_free_memory(device_lookup_range_check_18_0);
    cuda_free_memory(device_lookup_range_check_11_0);
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);

}

// ============================================================================
// Interaction Trace Generation Kernels
// ============================================================================

// Kernel for combining two lookups with +1 numerators (standard round)
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generic_opcode_interaction_col_gen_kernel_round0(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    m31 **lookup_state_0,
    m31 **lookup_state_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 init_combine_reg[N] = {};
    m31 final_combine_reg[M] = {};

    for (int i = 0; i < N; i++) {
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

// Kernel for second-to-last column: first lookup with enabler, second with +1
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generic_opcode_interaction_col_gen_kernel_second2last(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    m31 **lookup_state_0,
    m31 **lookup_state_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned n_rows
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    qm31 enabler_col = {0};
    if (vec_index < n_rows) {
        enabler_col = {1};
    }

    m31 init_combine_reg[N] = {};
    m31 final_combine_reg[M] = {};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }
    for (int i = 0; i < M; i++) {
        final_combine_reg[i] = lookup_state_1[i][vec_index];
    }
    if (vec_index < trace_size) {
        qm31 denom0 = lookup_elements_n->combine(init_combine_reg, N);
        qm31 denom1 = lookup_elements_m->combine(final_combine_reg, M);
        logup_col_write_frac(vec_index, add(denom1, mul(denom0, enabler_col)), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Kernel for last column: single lookup with -1 * enabler
template <int N>
__launch_bounds__(256, 2)
__global__ void generic_opcode_interaction_col_gen_kernel_last(
    LookupElementsBasic<N> *lookup_elements_n,
    m31 **lookup_state_0,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned n_rows
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    qm31 enabler_col = {0};
    if (vec_index < n_rows) {
        enabler_col = {1};
    }

    m31 init_combine_reg[N] = {};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }

    if (vec_index < trace_size) {
        qm31 denom = lookup_elements_n->combine(init_combine_reg, N);
        logup_col_write_frac(vec_index, mul(qm31{P-1, 0, 0, 0}, enabler_col), denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Kernel to finalize column by computing cumulative sum
__global__ void generic_opcode_interaction_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_trace
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

        if (pre_index == -1) {
            interaction_trace[0][vec_index] = 0;
            interaction_trace[1][vec_index] = 0;
            interaction_trace[2][vec_index] = 0;
            interaction_trace[3][vec_index] = 0;
            qm31 pre_value = qm31 {0};
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        } else {
            qm31 pre_value = qm31 {
                cm31{interaction_trace[pre_index * 4 + 0][vec_index], interaction_trace[pre_index * 4 + 1][vec_index]},
                cm31{interaction_trace[pre_index * 4 + 2][vec_index], interaction_trace[pre_index * 4 + 3][vec_index]}
            };
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        }

        interaction_trace[rep_index * 4 + 0][vec_index] = numerator0[vec_index];
        interaction_trace[rep_index * 4 + 1][vec_index] = numerator1[vec_index];
        interaction_trace[rep_index * 4 + 2][vec_index] = numerator2[vec_index];
        interaction_trace[rep_index * 4 + 3][vec_index] = numerator3[vec_index];
    }
}

// Kernel to compute coordinate sums for cumsum shift
__global__ void generic_opcode_interaction_cumsum_shift(
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_trace,
    m31 *coordinate_sums
) {
    int idx0 = 4 * last_index - 4;
    int idx1 = 4 * last_index - 3;
    int idx2 = 4 * last_index - 2;
    int idx3 = 4 * last_index - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = 0;
    m31 sum1 = 0;
    m31 sum2 = 0;
    m31 sum3 = 0;

    for (int i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_trace[idx0][i]);
        sum1 = add(sum1, interaction_trace[idx1][i]);
        sum2 = add(sum2, interaction_trace[idx2][i]);
        sum3 = add(sum3, interaction_trace[idx3][i]);
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

// Kernel to apply cumsum shift prefix sum adjustment
__global__ void generic_opcode_interaction_coord_prefix_sum(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_trace
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interaction_trace[4 * last_index - 4][vec_index] = sub(interaction_trace[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interaction_trace[4 * last_index - 3][vec_index] = sub(interaction_trace[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interaction_trace[4 * last_index - 2][vec_index] = sub(interaction_trace[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interaction_trace[4 * last_index - 1][vec_index] = sub(interaction_trace[4 * last_index - 1][vec_index], cumsum_shift.b.b);
    }
}

// Helper macro for a standard round (2 lookups combined)
#define GENERIC_OPCODE_INTERACTION_ROUND0(col_idx, lookup_elements_type_0, lookup_elements_0, lookup_data_0, n_fields_0, lookup_elements_type_1, lookup_elements_1, lookup_data_1, n_fields_1) \
    do { \
        block_dim = trace_size < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX; \
        num_blocks = block_dim < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim; \
        generic_opcode_interaction_col_gen_kernel_round0<n_fields_0, n_fields_1><<<num_blocks, block_dim>>>( \
            lookup_elements_0, lookup_elements_1, \
            lookup_data_0, lookup_data_1, \
            trace_size, device_logup_denom, \
            device_numerator0, device_numerator1, device_numerator2, device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
        block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX; \
        num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim; \
        generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>( \
            col_idx, trace_size, denom_inv, \
            device_numerator0, device_numerator1, device_numerator2, device_numerator3, \
            device_interaction_trace \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
    } while(0)

extern "C"
void generate_generic_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *range_check_9_9,
    void *range_check_9_9_b,
    void *range_check_9_9_c,
    void *range_check_9_9_d,
    void *range_check_9_9_e,
    void *range_check_9_9_f,
    void *range_check_9_9_g,
    void *range_check_9_9_h,
    void *range_check_19,
    void *range_check_19_b,
    void *range_check_19_c,
    void *range_check_19_d,
    void *range_check_19_e,
    void *range_check_19_f,
    void *range_check_19_g,
    void *range_check_19_h,
    void *range_check_18,
    void *range_check_11,
    void *verify_instruction,
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_9_9_0,
    unsigned **lookup_range_check_9_9_1,
    unsigned **lookup_range_check_9_9_2,
    unsigned **lookup_range_check_9_9_3,
    unsigned **lookup_range_check_9_9_b_0,
    unsigned **lookup_range_check_9_9_b_1,
    unsigned **lookup_range_check_9_9_b_2,
    unsigned **lookup_range_check_9_9_b_3,
    unsigned **lookup_range_check_9_9_c_0,
    unsigned **lookup_range_check_9_9_c_1,
    unsigned **lookup_range_check_9_9_c_2,
    unsigned **lookup_range_check_9_9_c_3,
    unsigned **lookup_range_check_9_9_d_0,
    unsigned **lookup_range_check_9_9_d_1,
    unsigned **lookup_range_check_9_9_d_2,
    unsigned **lookup_range_check_9_9_d_3,
    unsigned **lookup_range_check_9_9_e_0,
    unsigned **lookup_range_check_9_9_e_1,
    unsigned **lookup_range_check_9_9_e_2,
    unsigned **lookup_range_check_9_9_e_3,
    unsigned **lookup_range_check_9_9_f_0,
    unsigned **lookup_range_check_9_9_f_1,
    unsigned **lookup_range_check_9_9_f_2,
    unsigned **lookup_range_check_9_9_f_3,
    unsigned **lookup_range_check_9_9_g_0,
    unsigned **lookup_range_check_9_9_g_1,
    unsigned **lookup_range_check_9_9_h_0,
    unsigned **lookup_range_check_9_9_h_1,
    unsigned **lookup_range_check_19_0,
    unsigned **lookup_range_check_19_1,
    unsigned **lookup_range_check_19_2,
    unsigned **lookup_range_check_19_3,
    unsigned **lookup_range_check_19_b_0,
    unsigned **lookup_range_check_19_b_1,
    unsigned **lookup_range_check_19_b_2,
    unsigned **lookup_range_check_19_b_3,
    unsigned **lookup_range_check_19_c_0,
    unsigned **lookup_range_check_19_c_1,
    unsigned **lookup_range_check_19_c_2,
    unsigned **lookup_range_check_19_c_3,
    unsigned **lookup_range_check_19_d_0,
    unsigned **lookup_range_check_19_d_1,
    unsigned **lookup_range_check_19_d_2,
    unsigned **lookup_range_check_19_e_0,
    unsigned **lookup_range_check_19_e_1,
    unsigned **lookup_range_check_19_e_2,
    unsigned **lookup_range_check_19_f_0,
    unsigned **lookup_range_check_19_f_1,
    unsigned **lookup_range_check_19_f_2,
    unsigned **lookup_range_check_19_g_0,
    unsigned **lookup_range_check_19_g_1,
    unsigned **lookup_range_check_19_g_2,
    unsigned **lookup_range_check_19_h_0,
    unsigned **lookup_range_check_19_h_1,
    unsigned **lookup_range_check_19_h_2,
    unsigned **lookup_range_check_19_h_3,
    unsigned **lookup_range_check_18_0,
    unsigned **lookup_range_check_11_0,
    unsigned **lookup_verify_instruction_0,
    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
) {

    timer global_timer;
    global_timer.start("generate generic_opcode interaction trace");

    unsigned trace_size = 1 << log_size;

    // Cast lookup element pointers
    VerifyInstruction *verify_instruction_lookup_elements = (VerifyInstruction *)verify_instruction;
    MemoryAddressToId *memory_address_to_id_lookup_elements = (MemoryAddressToId *)memory_address_to_id;
    MemoryIdToBig *memory_id_to_big_lookup_elements = (MemoryIdToBig *)memory_id_to_big;
    Opcodes *opcodes_lookup_elements = (Opcodes *)opcodes;
    RangeCheck_9_9 *range_check_9_9_lookup_elements = (RangeCheck_9_9 *)range_check_9_9;
    RangeCheck_9_9 *range_check_9_9_b_lookup_elements = (RangeCheck_9_9 *)range_check_9_9_b;
    RangeCheck_9_9 *range_check_9_9_c_lookup_elements = (RangeCheck_9_9 *)range_check_9_9_c;
    RangeCheck_9_9 *range_check_9_9_d_lookup_elements = (RangeCheck_9_9 *)range_check_9_9_d;
    RangeCheck_9_9 *range_check_9_9_e_lookup_elements = (RangeCheck_9_9 *)range_check_9_9_e;
    RangeCheck_9_9 *range_check_9_9_f_lookup_elements = (RangeCheck_9_9 *)range_check_9_9_f;
    RangeCheck_9_9 *range_check_9_9_g_lookup_elements = (RangeCheck_9_9 *)range_check_9_9_g;
    RangeCheck_9_9 *range_check_9_9_h_lookup_elements = (RangeCheck_9_9 *)range_check_9_9_h;
    RangeCheck_19 *range_check_19_lookup_elements = (RangeCheck_19 *)range_check_19;
    RangeCheck_19 *range_check_19_b_lookup_elements = (RangeCheck_19 *)range_check_19_b;
    RangeCheck_19 *range_check_19_c_lookup_elements = (RangeCheck_19 *)range_check_19_c;
    RangeCheck_19 *range_check_19_d_lookup_elements = (RangeCheck_19 *)range_check_19_d;
    RangeCheck_19 *range_check_19_e_lookup_elements = (RangeCheck_19 *)range_check_19_e;
    RangeCheck_19 *range_check_19_f_lookup_elements = (RangeCheck_19 *)range_check_19_f;
    RangeCheck_19 *range_check_19_g_lookup_elements = (RangeCheck_19 *)range_check_19_g;
    RangeCheck_19 *range_check_19_h_lookup_elements = (RangeCheck_19 *)range_check_19_h;
    RangeCheck_18 *range_check_18_lookup_elements = (RangeCheck_18 *)range_check_18;
    RangeCheck_11 *range_check_11_lookup_elements = (RangeCheck_11 *)range_check_11;

    // Allocate device memory for lookup elements
    VerifyInstruction *device_verify_instruction = cuda_malloc<VerifyInstruction>(1);
    MemoryAddressToId *device_memory_address_to_id = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig *device_memory_id_to_big = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes = cuda_malloc<Opcodes>(1);
    RangeCheck_9_9 *device_range_check_9_9 = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_9_9 *device_range_check_9_9_b = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_9_9 *device_range_check_9_9_c = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_9_9 *device_range_check_9_9_d = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_9_9 *device_range_check_9_9_e = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_9_9 *device_range_check_9_9_f = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_9_9 *device_range_check_9_9_g = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_9_9 *device_range_check_9_9_h = cuda_malloc<RangeCheck_9_9>(1);
    RangeCheck_19 *device_range_check_19 = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19 *device_range_check_19_b = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19 *device_range_check_19_c = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19 *device_range_check_19_d = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19 *device_range_check_19_e = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19 *device_range_check_19_f = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19 *device_range_check_19_g = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19 *device_range_check_19_h = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_18 *device_range_check_18 = cuda_malloc<RangeCheck_18>(1);
    RangeCheck_11 *device_range_check_11 = cuda_malloc<RangeCheck_11>(1);

    // Copy lookup elements to device
    cuda_mem_copy_host_to_device<VerifyInstruction>(verify_instruction_lookup_elements, device_verify_instruction, 1);
    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id, 1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big, 1);
    cuda_mem_copy_host_to_device<Opcodes>(opcodes_lookup_elements, device_opcodes, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_lookup_elements, device_range_check_9_9, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_b_lookup_elements, device_range_check_9_9_b, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_c_lookup_elements, device_range_check_9_9_c, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_d_lookup_elements, device_range_check_9_9_d, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_e_lookup_elements, device_range_check_9_9_e, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_f_lookup_elements, device_range_check_9_9_f, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_g_lookup_elements, device_range_check_9_9_g, 1);
    cuda_mem_copy_host_to_device<RangeCheck_9_9>(range_check_9_9_h_lookup_elements, device_range_check_9_9_h, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_lookup_elements, device_range_check_19, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_b_lookup_elements, device_range_check_19_b, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_c_lookup_elements, device_range_check_19_c, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_d_lookup_elements, device_range_check_19_d, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_e_lookup_elements, device_range_check_19_e, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_f_lookup_elements, device_range_check_19_f, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_g_lookup_elements, device_range_check_19_g, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_h_lookup_elements, device_range_check_19_h, 1);
    cuda_mem_copy_host_to_device<RangeCheck_18>(range_check_18_lookup_elements, device_range_check_18, 1);
    cuda_mem_copy_host_to_device<RangeCheck_11>(range_check_11_lookup_elements, device_range_check_11, 1);

    // Clone lookup data pointer arrays to device
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31*>(lookup_verify_instruction_0, 7);
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31*>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31*>(lookup_opcodes_1, 3);

    // Range check 9_9 variants (2 fields each)
    m31 **device_lookup_range_check_9_9_0 = clone_to_device<m31*>(lookup_range_check_9_9_0, 2);
    m31 **device_lookup_range_check_9_9_1 = clone_to_device<m31*>(lookup_range_check_9_9_1, 2);
    m31 **device_lookup_range_check_9_9_2 = clone_to_device<m31*>(lookup_range_check_9_9_2, 2);
    m31 **device_lookup_range_check_9_9_3 = clone_to_device<m31*>(lookup_range_check_9_9_3, 2);
    m31 **device_lookup_range_check_9_9_b_0 = clone_to_device<m31*>(lookup_range_check_9_9_b_0, 2);
    m31 **device_lookup_range_check_9_9_b_1 = clone_to_device<m31*>(lookup_range_check_9_9_b_1, 2);
    m31 **device_lookup_range_check_9_9_b_2 = clone_to_device<m31*>(lookup_range_check_9_9_b_2, 2);
    m31 **device_lookup_range_check_9_9_b_3 = clone_to_device<m31*>(lookup_range_check_9_9_b_3, 2);
    m31 **device_lookup_range_check_9_9_c_0 = clone_to_device<m31*>(lookup_range_check_9_9_c_0, 2);
    m31 **device_lookup_range_check_9_9_c_1 = clone_to_device<m31*>(lookup_range_check_9_9_c_1, 2);
    m31 **device_lookup_range_check_9_9_c_2 = clone_to_device<m31*>(lookup_range_check_9_9_c_2, 2);
    m31 **device_lookup_range_check_9_9_c_3 = clone_to_device<m31*>(lookup_range_check_9_9_c_3, 2);
    m31 **device_lookup_range_check_9_9_d_0 = clone_to_device<m31*>(lookup_range_check_9_9_d_0, 2);
    m31 **device_lookup_range_check_9_9_d_1 = clone_to_device<m31*>(lookup_range_check_9_9_d_1, 2);
    m31 **device_lookup_range_check_9_9_d_2 = clone_to_device<m31*>(lookup_range_check_9_9_d_2, 2);
    m31 **device_lookup_range_check_9_9_d_3 = clone_to_device<m31*>(lookup_range_check_9_9_d_3, 2);
    m31 **device_lookup_range_check_9_9_e_0 = clone_to_device<m31*>(lookup_range_check_9_9_e_0, 2);
    m31 **device_lookup_range_check_9_9_e_1 = clone_to_device<m31*>(lookup_range_check_9_9_e_1, 2);
    m31 **device_lookup_range_check_9_9_e_2 = clone_to_device<m31*>(lookup_range_check_9_9_e_2, 2);
    m31 **device_lookup_range_check_9_9_e_3 = clone_to_device<m31*>(lookup_range_check_9_9_e_3, 2);
    m31 **device_lookup_range_check_9_9_f_0 = clone_to_device<m31*>(lookup_range_check_9_9_f_0, 2);
    m31 **device_lookup_range_check_9_9_f_1 = clone_to_device<m31*>(lookup_range_check_9_9_f_1, 2);
    m31 **device_lookup_range_check_9_9_f_2 = clone_to_device<m31*>(lookup_range_check_9_9_f_2, 2);
    m31 **device_lookup_range_check_9_9_f_3 = clone_to_device<m31*>(lookup_range_check_9_9_f_3, 2);
    m31 **device_lookup_range_check_9_9_g_0 = clone_to_device<m31*>(lookup_range_check_9_9_g_0, 2);
    m31 **device_lookup_range_check_9_9_g_1 = clone_to_device<m31*>(lookup_range_check_9_9_g_1, 2);
    m31 **device_lookup_range_check_9_9_h_0 = clone_to_device<m31*>(lookup_range_check_9_9_h_0, 2);
    m31 **device_lookup_range_check_9_9_h_1 = clone_to_device<m31*>(lookup_range_check_9_9_h_1, 2);

    // Range check 19 variants (1 field each)
    m31 **device_lookup_range_check_19_0 = clone_to_device<m31*>(lookup_range_check_19_0, 1);
    m31 **device_lookup_range_check_19_1 = clone_to_device<m31*>(lookup_range_check_19_1, 1);
    m31 **device_lookup_range_check_19_2 = clone_to_device<m31*>(lookup_range_check_19_2, 1);
    m31 **device_lookup_range_check_19_3 = clone_to_device<m31*>(lookup_range_check_19_3, 1);
    m31 **device_lookup_range_check_19_b_0 = clone_to_device<m31*>(lookup_range_check_19_b_0, 1);
    m31 **device_lookup_range_check_19_b_1 = clone_to_device<m31*>(lookup_range_check_19_b_1, 1);
    m31 **device_lookup_range_check_19_b_2 = clone_to_device<m31*>(lookup_range_check_19_b_2, 1);
    m31 **device_lookup_range_check_19_b_3 = clone_to_device<m31*>(lookup_range_check_19_b_3, 1);
    m31 **device_lookup_range_check_19_c_0 = clone_to_device<m31*>(lookup_range_check_19_c_0, 1);
    m31 **device_lookup_range_check_19_c_1 = clone_to_device<m31*>(lookup_range_check_19_c_1, 1);
    m31 **device_lookup_range_check_19_c_2 = clone_to_device<m31*>(lookup_range_check_19_c_2, 1);
    m31 **device_lookup_range_check_19_c_3 = clone_to_device<m31*>(lookup_range_check_19_c_3, 1);
    m31 **device_lookup_range_check_19_d_0 = clone_to_device<m31*>(lookup_range_check_19_d_0, 1);
    m31 **device_lookup_range_check_19_d_1 = clone_to_device<m31*>(lookup_range_check_19_d_1, 1);
    m31 **device_lookup_range_check_19_d_2 = clone_to_device<m31*>(lookup_range_check_19_d_2, 1);
    m31 **device_lookup_range_check_19_e_0 = clone_to_device<m31*>(lookup_range_check_19_e_0, 1);
    m31 **device_lookup_range_check_19_e_1 = clone_to_device<m31*>(lookup_range_check_19_e_1, 1);
    m31 **device_lookup_range_check_19_e_2 = clone_to_device<m31*>(lookup_range_check_19_e_2, 1);
    m31 **device_lookup_range_check_19_f_0 = clone_to_device<m31*>(lookup_range_check_19_f_0, 1);
    m31 **device_lookup_range_check_19_f_1 = clone_to_device<m31*>(lookup_range_check_19_f_1, 1);
    m31 **device_lookup_range_check_19_f_2 = clone_to_device<m31*>(lookup_range_check_19_f_2, 1);
    m31 **device_lookup_range_check_19_g_0 = clone_to_device<m31*>(lookup_range_check_19_g_0, 1);
    m31 **device_lookup_range_check_19_g_1 = clone_to_device<m31*>(lookup_range_check_19_g_1, 1);
    m31 **device_lookup_range_check_19_g_2 = clone_to_device<m31*>(lookup_range_check_19_g_2, 1);
    m31 **device_lookup_range_check_19_h_0 = clone_to_device<m31*>(lookup_range_check_19_h_0, 1);
    m31 **device_lookup_range_check_19_h_1 = clone_to_device<m31*>(lookup_range_check_19_h_1, 1);
    m31 **device_lookup_range_check_19_h_2 = clone_to_device<m31*>(lookup_range_check_19_h_2, 1);
    m31 **device_lookup_range_check_19_h_3 = clone_to_device<m31*>(lookup_range_check_19_h_3, 1);
    m31 **device_lookup_range_check_18_0 = clone_to_device<m31*>(lookup_range_check_18_0, 1);
    m31 **device_lookup_range_check_11_0 = clone_to_device<m31*>(lookup_range_check_11_0, 1);

    // Clone interaction trace pointer array to device
    m31 **device_interaction_trace = clone_to_device<m31*>(interaction_trace, 4 * GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS);

    // Allocate working memory
    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    int block_dim = 0;
    int num_blocks = 0;

    // Process all 67 lookups in 34 interaction columns
    // The ordering follows the CPU evaluator's add_to_relation order:
    // 1. verify_instruction (1)
    // 2. memory_address_to_id x3
    // 3. memory_id_to_big x3
    // 4. range_check_9_9 variants (28 total)
    // 5. range_check_19 variants (28 total)
    // 6. range_check_18 (1)
    // 7. range_check_11 (1)
    // 8. opcodes x2

    // Column 0: verify_instruction_0 + memory_address_to_id_0
    block_dim = trace_size < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
        device_verify_instruction, device_memory_address_to_id,
        device_lookup_verify_instruction_0, device_lookup_memory_address_to_id_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        0, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 1: memory_id_to_big_0 + memory_address_to_id_1
    block_dim = trace_size < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_col_gen_kernel_round0<29, 2><<<num_blocks, block_dim>>>(
        device_memory_id_to_big, device_memory_address_to_id,
        device_lookup_memory_id_to_big_0, device_lookup_memory_address_to_id_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        1, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 2: memory_id_to_big_1 + memory_address_to_id_2
    block_dim = trace_size < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_col_gen_kernel_round0<29, 2><<<num_blocks, block_dim>>>(
        device_memory_id_to_big, device_memory_address_to_id,
        device_lookup_memory_id_to_big_1, device_lookup_memory_address_to_id_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        2, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 3: memory_id_to_big_2 + range_check_9_9_0
    block_dim = trace_size < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_col_gen_kernel_round0<29, 2><<<num_blocks, block_dim>>>(
        device_memory_id_to_big, device_range_check_9_9,
        device_lookup_memory_id_to_big_2, device_lookup_range_check_9_9_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        3, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Columns 4-17: range_check_9_9 variants following exact CPU order
    // Column 4: range_check_9_9_b_0 + range_check_9_9_c_0
    block_dim = trace_size < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_b, device_range_check_9_9_c,
        device_lookup_range_check_9_9_b_0, device_lookup_range_check_9_9_c_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        4, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 5: range_check_9_9_d_0 + range_check_9_9_e_0
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_d, device_range_check_9_9_e,
        device_lookup_range_check_9_9_d_0, device_lookup_range_check_9_9_e_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        5, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 6: range_check_9_9_f_0 + range_check_9_9_g_0
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_f, device_range_check_9_9_g,
        device_lookup_range_check_9_9_f_0, device_lookup_range_check_9_9_g_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        6, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 7: range_check_9_9_h_0 + range_check_9_9_1
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_h, device_range_check_9_9,
        device_lookup_range_check_9_9_h_0, device_lookup_range_check_9_9_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        7, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 8: range_check_9_9_b_1 + range_check_9_9_c_1
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_b, device_range_check_9_9_c,
        device_lookup_range_check_9_9_b_1, device_lookup_range_check_9_9_c_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        8, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 9: range_check_9_9_d_1 + range_check_9_9_e_1
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_d, device_range_check_9_9_e,
        device_lookup_range_check_9_9_d_1, device_lookup_range_check_9_9_e_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        9, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 10: range_check_9_9_f_1 + range_check_9_9_2
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_f, device_range_check_9_9,
        device_lookup_range_check_9_9_f_1, device_lookup_range_check_9_9_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        10, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 11: range_check_9_9_b_2 + range_check_9_9_c_2
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_b, device_range_check_9_9_c,
        device_lookup_range_check_9_9_b_2, device_lookup_range_check_9_9_c_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        11, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 12: range_check_9_9_d_2 + range_check_9_9_e_2
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_d, device_range_check_9_9_e,
        device_lookup_range_check_9_9_d_2, device_lookup_range_check_9_9_e_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        12, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 13: range_check_9_9_f_2 + range_check_9_9_g_1
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_f, device_range_check_9_9_g,
        device_lookup_range_check_9_9_f_2, device_lookup_range_check_9_9_g_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        13, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 14: range_check_9_9_h_1 + range_check_9_9_3
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_h, device_range_check_9_9,
        device_lookup_range_check_9_9_h_1, device_lookup_range_check_9_9_3,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        14, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 15: range_check_9_9_b_3 + range_check_9_9_c_3
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_b, device_range_check_9_9_c,
        device_lookup_range_check_9_9_b_3, device_lookup_range_check_9_9_c_3,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        15, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 16: range_check_9_9_d_3 + range_check_9_9_e_3
    generic_opcode_interaction_col_gen_kernel_round0<2, 2><<<num_blocks, block_dim>>>(
        device_range_check_9_9_d, device_range_check_9_9_e,
        device_lookup_range_check_9_9_d_3, device_lookup_range_check_9_9_e_3,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        16, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 17: range_check_9_9_f_3 + range_check_19_h_0
    generic_opcode_interaction_col_gen_kernel_round0<2, 1><<<num_blocks, block_dim>>>(
        device_range_check_9_9_f, device_range_check_19_h,
        device_lookup_range_check_9_9_f_3, device_lookup_range_check_19_h_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        17, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Continue with range_check_19 variants following exact CPU order
    // Column 18: range_check_19_0 + range_check_19_b_0
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19, device_range_check_19_b,
        device_lookup_range_check_19_0, device_lookup_range_check_19_b_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        18, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 19: range_check_19_c_0 + range_check_19_d_0
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_c, device_range_check_19_d,
        device_lookup_range_check_19_c_0, device_lookup_range_check_19_d_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        19, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 20: range_check_19_e_0 + range_check_19_f_0
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_e, device_range_check_19_f,
        device_lookup_range_check_19_e_0, device_lookup_range_check_19_f_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        20, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 21: range_check_19_g_0 + range_check_19_h_1
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_g, device_range_check_19_h,
        device_lookup_range_check_19_g_0, device_lookup_range_check_19_h_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        21, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 22: range_check_19_1 + range_check_19_b_1
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19, device_range_check_19_b,
        device_lookup_range_check_19_1, device_lookup_range_check_19_b_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        22, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 23: range_check_19_c_1 + range_check_19_d_1
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_c, device_range_check_19_d,
        device_lookup_range_check_19_c_1, device_lookup_range_check_19_d_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        23, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 24: range_check_19_e_1 + range_check_19_f_1
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_e, device_range_check_19_f,
        device_lookup_range_check_19_e_1, device_lookup_range_check_19_f_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        24, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 25: range_check_19_g_1 + range_check_19_h_2
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_g, device_range_check_19_h,
        device_lookup_range_check_19_g_1, device_lookup_range_check_19_h_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        25, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 26: range_check_19_2 + range_check_19_b_2
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19, device_range_check_19_b,
        device_lookup_range_check_19_2, device_lookup_range_check_19_b_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        26, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 27: range_check_19_c_2 + range_check_19_d_2
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_c, device_range_check_19_d,
        device_lookup_range_check_19_c_2, device_lookup_range_check_19_d_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        27, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 28: range_check_19_e_2 + range_check_19_f_2
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_e, device_range_check_19_f,
        device_lookup_range_check_19_e_2, device_lookup_range_check_19_f_2,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        28, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 29: range_check_19_g_2 + range_check_19_h_3
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_g, device_range_check_19_h,
        device_lookup_range_check_19_g_2, device_lookup_range_check_19_h_3,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        29, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 30: range_check_19_3 + range_check_19_b_3
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19, device_range_check_19_b,
        device_lookup_range_check_19_3, device_lookup_range_check_19_b_3,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        30, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 31: range_check_19_c_3 + range_check_18_0
    generic_opcode_interaction_col_gen_kernel_round0<1, 1><<<num_blocks, block_dim>>>(
        device_range_check_19_c, device_range_check_18,
        device_lookup_range_check_19_c_3, device_lookup_range_check_18_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        31, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 32: range_check_11_0 * enabler + opcodes_0
    generic_opcode_interaction_col_gen_kernel_second2last<1, 3><<<num_blocks, block_dim>>>(
        device_range_check_11, device_opcodes,
        device_lookup_range_check_11_0, device_lookup_opcodes_0,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        n_rows
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        32, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Column 33: -1 * enabler / opcodes_1 (last column)
    generic_opcode_interaction_col_gen_kernel_last<3><<<num_blocks, block_dim>>>(
        device_opcodes,
        device_lookup_opcodes_1,
        trace_size, device_logup_denom,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        n_rows
    );
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    generic_opcode_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
        33, trace_size, denom_inv,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3,
        device_interaction_trace
    );

    // Compute cumsum_shift
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generic_opcode_interaction_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_trace,
        claimed_sum
    );

    // Apply prefix sum adjustment
    generic_opcode_interaction_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_trace
    );

    // Apply inclusive prefix sum to last interaction column
    inclusive_prefix_sum(interaction_trace[4 * GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate generic_opcode interaction trace");

    // Free lookup element device memory
    cuda_free_memory(device_verify_instruction);
    cuda_free_memory(device_memory_address_to_id);
    cuda_free_memory(device_memory_id_to_big);
    cuda_free_memory(device_opcodes);
    cuda_free_memory(device_range_check_9_9);
    cuda_free_memory(device_range_check_9_9_b);
    cuda_free_memory(device_range_check_9_9_c);
    cuda_free_memory(device_range_check_9_9_d);
    cuda_free_memory(device_range_check_9_9_e);
    cuda_free_memory(device_range_check_9_9_f);
    cuda_free_memory(device_range_check_9_9_g);
    cuda_free_memory(device_range_check_9_9_h);
    cuda_free_memory(device_range_check_19);
    cuda_free_memory(device_range_check_19_b);
    cuda_free_memory(device_range_check_19_c);
    cuda_free_memory(device_range_check_19_d);
    cuda_free_memory(device_range_check_19_e);
    cuda_free_memory(device_range_check_19_f);
    cuda_free_memory(device_range_check_19_g);
    cuda_free_memory(device_range_check_19_h);
    cuda_free_memory(device_range_check_18);
    cuda_free_memory(device_range_check_11);

    // Free lookup data device memory
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_9_9_0);
    cuda_free_memory(device_lookup_range_check_9_9_1);
    cuda_free_memory(device_lookup_range_check_9_9_2);
    cuda_free_memory(device_lookup_range_check_9_9_3);
    cuda_free_memory(device_lookup_range_check_9_9_b_0);
    cuda_free_memory(device_lookup_range_check_9_9_b_1);
    cuda_free_memory(device_lookup_range_check_9_9_b_2);
    cuda_free_memory(device_lookup_range_check_9_9_b_3);
    cuda_free_memory(device_lookup_range_check_9_9_c_0);
    cuda_free_memory(device_lookup_range_check_9_9_c_1);
    cuda_free_memory(device_lookup_range_check_9_9_c_2);
    cuda_free_memory(device_lookup_range_check_9_9_c_3);
    cuda_free_memory(device_lookup_range_check_9_9_d_0);
    cuda_free_memory(device_lookup_range_check_9_9_d_1);
    cuda_free_memory(device_lookup_range_check_9_9_d_2);
    cuda_free_memory(device_lookup_range_check_9_9_d_3);
    cuda_free_memory(device_lookup_range_check_9_9_e_0);
    cuda_free_memory(device_lookup_range_check_9_9_e_1);
    cuda_free_memory(device_lookup_range_check_9_9_e_2);
    cuda_free_memory(device_lookup_range_check_9_9_e_3);
    cuda_free_memory(device_lookup_range_check_9_9_f_0);
    cuda_free_memory(device_lookup_range_check_9_9_f_1);
    cuda_free_memory(device_lookup_range_check_9_9_f_2);
    cuda_free_memory(device_lookup_range_check_9_9_f_3);
    cuda_free_memory(device_lookup_range_check_9_9_g_0);
    cuda_free_memory(device_lookup_range_check_9_9_g_1);
    cuda_free_memory(device_lookup_range_check_9_9_h_0);
    cuda_free_memory(device_lookup_range_check_9_9_h_1);
    cuda_free_memory(device_lookup_range_check_19_0);
    cuda_free_memory(device_lookup_range_check_19_1);
    cuda_free_memory(device_lookup_range_check_19_2);
    cuda_free_memory(device_lookup_range_check_19_3);
    cuda_free_memory(device_lookup_range_check_19_b_0);
    cuda_free_memory(device_lookup_range_check_19_b_1);
    cuda_free_memory(device_lookup_range_check_19_b_2);
    cuda_free_memory(device_lookup_range_check_19_b_3);
    cuda_free_memory(device_lookup_range_check_19_c_0);
    cuda_free_memory(device_lookup_range_check_19_c_1);
    cuda_free_memory(device_lookup_range_check_19_c_2);
    cuda_free_memory(device_lookup_range_check_19_c_3);
    cuda_free_memory(device_lookup_range_check_19_d_0);
    cuda_free_memory(device_lookup_range_check_19_d_1);
    cuda_free_memory(device_lookup_range_check_19_d_2);
    cuda_free_memory(device_lookup_range_check_19_e_0);
    cuda_free_memory(device_lookup_range_check_19_e_1);
    cuda_free_memory(device_lookup_range_check_19_e_2);
    cuda_free_memory(device_lookup_range_check_19_f_0);
    cuda_free_memory(device_lookup_range_check_19_f_1);
    cuda_free_memory(device_lookup_range_check_19_f_2);
    cuda_free_memory(device_lookup_range_check_19_g_0);
    cuda_free_memory(device_lookup_range_check_19_g_1);
    cuda_free_memory(device_lookup_range_check_19_g_2);
    cuda_free_memory(device_lookup_range_check_19_h_0);
    cuda_free_memory(device_lookup_range_check_19_h_1);
    cuda_free_memory(device_lookup_range_check_19_h_2);
    cuda_free_memory(device_lookup_range_check_19_h_3);
    cuda_free_memory(device_lookup_range_check_18_0);
    cuda_free_memory(device_lookup_range_check_11_0);

    // Free working memory
    cuda_free_memory(device_interaction_trace);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);

}
