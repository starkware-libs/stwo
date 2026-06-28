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

#include "gen_mul_opcode_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

// Constants defined in gen_mul_opcode_trace.cuh:
// MUL_OPCODE_N_TRACE_COLUMNS = 130
// MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS = 19
// MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX = 256

__launch_bounds__(256, 2)
__global__ void generate_mul_opcode_trace_kernel(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
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
    m31 **lookup_verify_instruction_0,

    m31 **sub_component_inputss_verify_instruction,
    m31 **sub_component_inputss_memory_address_to_id,
    m31 **sub_component_inputss_memory_id_to_big,
    m31 **sub_component_inputss_range_check_19,
    m31 **sub_component_inputss_range_check_19_b,
    m31 **sub_component_inputss_range_check_19_c,
    m31 **sub_component_inputss_range_check_19_d,
    m31 **sub_component_inputss_range_check_19_e,
    m31 **sub_component_inputss_range_check_19_f,
    m31 **sub_component_inputss_range_check_19_g,
    m31 **sub_component_inputss_range_check_19_h,

    m31 **mul_opcode_inputs,

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
    const m31 M31_131072    = {131072};
    const m31 M31_262144    = {262144};
    const m31 M31_32768     = {32768};
    const m31 M31_65536     = {65536};
    const m31 M31_4194304   = {4194304};
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

    const uint32_t UInt32_9 = 9;
    const uint32_t UInt32_511 = 511;
    const uint32_t UInt32_65536 = 65536;
    const uint32_t UInt32_262143 = 262143;

    if (row < trace_size) {
        // Input columns
        m31 input_pc_col0 = mul_opcode_inputs[0][row];
        traces[0][row] = input_pc_col0;
        m31 input_ap_col1 = mul_opcode_inputs[1][row];
        traces[1][row] = input_ap_col1;
        m31 input_fp_col2 = mul_opcode_inputs[2][row];
        traces[2][row] = input_fp_col2;

        // Decode Instruction
        m31 memory_address_to_id_value_tmp_0 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_pc_col0,
            &memory_address_to_id_value_tmp_0
        );

        m31 memory_id_to_big_value_tmp_1[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            memory_address_to_id_value_tmp_0,
            memory_id_to_big_value_tmp_1
        );

        // offset0 calculation
        uint16_t offset0_tmp =
            ((uint16_t)(memory_id_to_big_value_tmp_1[0]))
            + ((((uint16_t)(memory_id_to_big_value_tmp_1[1])) & UInt16_127) << UInt16_9);
        m31 offset0_col3 = m31{offset0_tmp};
        traces[3][row] = offset0_col3;

        // offset1 calculation
        uint16_t offset1_tmp =
            ((((uint16_t)(memory_id_to_big_value_tmp_1[1])) >> UInt16_7)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[2])) << UInt16_2))
            + ((((uint16_t)(memory_id_to_big_value_tmp_1[3])) & UInt16_31) << UInt16_11);
        m31 offset1_col4 = m31{offset1_tmp};
        traces[4][row] = offset1_col4;

        // offset2 calculation
        uint16_t offset2_tmp =
            ((((uint16_t)(memory_id_to_big_value_tmp_1[3])) >> UInt16_5)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[4])) << UInt16_4))
            + ((((uint16_t)(memory_id_to_big_value_tmp_1[5])) & UInt16_7) << UInt16_13);
        m31 offset2_col5 = m31{offset2_tmp};
        traces[5][row] = offset2_col5;

        // Flag decoding
        uint16_t flags_tmp = (((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6);

        uint16_t dst_base_fp_tmp = (flags_tmp >> UInt16_0) & UInt16_1;
        m31 dst_base_fp_col6 = m31{dst_base_fp_tmp};
        traces[6][row] = dst_base_fp_col6;

        uint16_t op0_base_fp_tmp = (flags_tmp >> UInt16_1) & UInt16_1;
        m31 op0_base_fp_col7 = m31{op0_base_fp_tmp};
        traces[7][row] = op0_base_fp_col7;

        uint16_t op1_imm_tmp = (flags_tmp >> UInt16_2) & UInt16_1;
        m31 op1_imm_col8 = m31{op1_imm_tmp};
        traces[8][row] = op1_imm_col8;

        uint16_t op1_base_fp_tmp = (flags_tmp >> UInt16_3) & UInt16_1;
        m31 op1_base_fp_col9 = m31{op1_base_fp_tmp};
        traces[9][row] = op1_base_fp_col9;

        uint16_t ap_update_add_1_tmp = (flags_tmp >> UInt16_11) & UInt16_1;
        m31 ap_update_add_1_col10 = m31{ap_update_add_1_tmp};
        traces[10][row] = ap_update_add_1_col10;

        // op1_base_ap = (1 - op1_imm) - op1_base_fp
        m31 op1_base_ap = sub(sub(M31_1, op1_imm_col8), op1_base_fp_col9);

        // verify_instruction lookup
        m31 flags0 = add(add(add(add(
            mul(dst_base_fp_col6, M31_8),
            mul(op0_base_fp_col7, M31_16)),
            mul(op1_imm_col8, M31_32)),
            mul(op1_base_fp_col9, M31_64)),
            mul(op1_base_ap, M31_128));
        m31 flags1 = add(add(M31_1, mul(ap_update_add_1_col10, M31_32)), M31_256);

        sub_component_inputss_verify_instruction[0][row] = input_pc_col0;
        sub_component_inputss_verify_instruction[1][row] = offset0_col3;
        sub_component_inputss_verify_instruction[2][row] = offset1_col4;
        sub_component_inputss_verify_instruction[3][row] = offset2_col5;
        sub_component_inputss_verify_instruction[4][row] = flags0;
        sub_component_inputss_verify_instruction[5][row] = flags1;
        sub_component_inputss_verify_instruction[6][row] = M31_0;

        lookup_verify_instruction_0[0][row] = input_pc_col0;
        lookup_verify_instruction_0[1][row] = offset0_col3;
        lookup_verify_instruction_0[2][row] = offset1_col4;
        lookup_verify_instruction_0[3][row] = offset2_col5;
        lookup_verify_instruction_0[4][row] = flags0;
        lookup_verify_instruction_0[5][row] = flags1;
        lookup_verify_instruction_0[6][row] = M31_0;

        // Memory base calculations
        m31 mem_dst_base_col11 = add(
            mul(dst_base_fp_col6, input_fp_col2),
            mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)
        );
        traces[11][row] = mem_dst_base_col11;

        m31 mem0_base_col12 = add(
            mul(op0_base_fp_col7, input_fp_col2),
            mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)
        );
        traces[12][row] = mem0_base_col12;

        m31 mem1_base_col13 = add(add(
            mul(op1_imm_col8, input_pc_col0),
            mul(op1_base_fp_col9, input_fp_col2)),
            mul(op1_base_ap, input_ap_col1)
        );
        traces[13][row] = mem1_base_col13;

        // Decode instruction output offsets (subtract 32768)
        m31 offset0_adj = sub(offset0_col3, M31_32768);
        m31 offset1_adj = sub(offset1_col4, M31_32768);
        m31 offset2_adj = sub(offset2_col5, M31_32768);

        // Read dst_id
        m31 dst_address = add(mem_dst_base_col11, offset0_adj);
        m31 dst_id_col14 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            dst_address,
            &dst_id_col14
        );
        traces[14][row] = dst_id_col14;

        sub_component_inputss_memory_address_to_id[0][row] = dst_address;
        lookup_memory_address_to_id_0[0][row] = dst_address;
        lookup_memory_address_to_id_0[1][row] = dst_id_col14;

        // Read dst limbs (memory_id_to_big)
        m31 dst_limbs[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            dst_id_col14,
            dst_limbs
        );

        // Store dst limbs (cols 15-42)
        for (int i = 0; i < 28; i++) {
            traces[15 + i][row] = dst_limbs[i];
        }

        sub_component_inputss_memory_id_to_big[0][row] = dst_id_col14;
        lookup_memory_id_to_big_0[0][row] = dst_id_col14;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_0[1 + i][row] = dst_limbs[i];
        }

        // Read op0_id
        m31 op0_address = add(mem0_base_col12, offset1_adj);
        m31 op0_id_col43 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op0_address,
            &op0_id_col43
        );
        traces[43][row] = op0_id_col43;

        sub_component_inputss_memory_address_to_id[1][row] = op0_address;
        lookup_memory_address_to_id_1[0][row] = op0_address;
        lookup_memory_address_to_id_1[1][row] = op0_id_col43;

        // Read op0 limbs
        m31 op0_limbs[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            op0_id_col43,
            op0_limbs
        );

        // Store op0 limbs (cols 44-71)
        for (int i = 0; i < 28; i++) {
            traces[44 + i][row] = op0_limbs[i];
        }

        sub_component_inputss_memory_id_to_big[1][row] = op0_id_col43;
        lookup_memory_id_to_big_1[0][row] = op0_id_col43;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_1[1 + i][row] = op0_limbs[i];
        }

        // Read op1_id
        m31 op1_address = add(mem1_base_col13, offset2_adj);
        m31 op1_id_col72 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op1_address,
            &op1_id_col72
        );
        traces[72][row] = op1_id_col72;

        sub_component_inputss_memory_address_to_id[2][row] = op1_address;
        lookup_memory_address_to_id_2[0][row] = op1_address;
        lookup_memory_address_to_id_2[1][row] = op1_id_col72;

        // Read op1 limbs
        m31 op1_limbs[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            op1_id_col72,
            op1_limbs
        );

        // Store op1 limbs (cols 73-100)
        for (int i = 0; i < 28; i++) {
            traces[73 + i][row] = op1_limbs[i];
        }

        sub_component_inputss_memory_id_to_big[2][row] = op1_id_col72;
        lookup_memory_id_to_big_2[0][row] = op1_id_col72;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_2[1 + i][row] = op1_limbs[i];
        }

        // ============================================================================
        // Double Karatsuba Multiplication for 252-bit numbers
        // ============================================================================

        // Single Karatsuba N=7 for lower halves (z0: op0[0..7] * op1[0..7])
        int64_t z0[13];
        z0[0] = (int64_t)(unsigned)op0_limbs[0] * (int64_t)(unsigned)op1_limbs[0];
        z0[1] = (int64_t)(unsigned)op0_limbs[0] * (int64_t)(unsigned)op1_limbs[1]
              + (int64_t)(unsigned)op0_limbs[1] * (int64_t)(unsigned)op1_limbs[0];
        z0[2] = (int64_t)(unsigned)op0_limbs[0] * (int64_t)(unsigned)op1_limbs[2]
              + (int64_t)(unsigned)op0_limbs[1] * (int64_t)(unsigned)op1_limbs[1]
              + (int64_t)(unsigned)op0_limbs[2] * (int64_t)(unsigned)op1_limbs[0];
        z0[3] = (int64_t)(unsigned)op0_limbs[0] * (int64_t)(unsigned)op1_limbs[3]
              + (int64_t)(unsigned)op0_limbs[1] * (int64_t)(unsigned)op1_limbs[2]
              + (int64_t)(unsigned)op0_limbs[2] * (int64_t)(unsigned)op1_limbs[1]
              + (int64_t)(unsigned)op0_limbs[3] * (int64_t)(unsigned)op1_limbs[0];
        z0[4] = (int64_t)(unsigned)op0_limbs[0] * (int64_t)(unsigned)op1_limbs[4]
              + (int64_t)(unsigned)op0_limbs[1] * (int64_t)(unsigned)op1_limbs[3]
              + (int64_t)(unsigned)op0_limbs[2] * (int64_t)(unsigned)op1_limbs[2]
              + (int64_t)(unsigned)op0_limbs[3] * (int64_t)(unsigned)op1_limbs[1]
              + (int64_t)(unsigned)op0_limbs[4] * (int64_t)(unsigned)op1_limbs[0];
        z0[5] = (int64_t)(unsigned)op0_limbs[0] * (int64_t)(unsigned)op1_limbs[5]
              + (int64_t)(unsigned)op0_limbs[1] * (int64_t)(unsigned)op1_limbs[4]
              + (int64_t)(unsigned)op0_limbs[2] * (int64_t)(unsigned)op1_limbs[3]
              + (int64_t)(unsigned)op0_limbs[3] * (int64_t)(unsigned)op1_limbs[2]
              + (int64_t)(unsigned)op0_limbs[4] * (int64_t)(unsigned)op1_limbs[1]
              + (int64_t)(unsigned)op0_limbs[5] * (int64_t)(unsigned)op1_limbs[0];
        z0[6] = (int64_t)(unsigned)op0_limbs[0] * (int64_t)(unsigned)op1_limbs[6]
              + (int64_t)(unsigned)op0_limbs[1] * (int64_t)(unsigned)op1_limbs[5]
              + (int64_t)(unsigned)op0_limbs[2] * (int64_t)(unsigned)op1_limbs[4]
              + (int64_t)(unsigned)op0_limbs[3] * (int64_t)(unsigned)op1_limbs[3]
              + (int64_t)(unsigned)op0_limbs[4] * (int64_t)(unsigned)op1_limbs[2]
              + (int64_t)(unsigned)op0_limbs[5] * (int64_t)(unsigned)op1_limbs[1]
              + (int64_t)(unsigned)op0_limbs[6] * (int64_t)(unsigned)op1_limbs[0];
        z0[7] = (int64_t)(unsigned)op0_limbs[1] * (int64_t)(unsigned)op1_limbs[6]
              + (int64_t)(unsigned)op0_limbs[2] * (int64_t)(unsigned)op1_limbs[5]
              + (int64_t)(unsigned)op0_limbs[3] * (int64_t)(unsigned)op1_limbs[4]
              + (int64_t)(unsigned)op0_limbs[4] * (int64_t)(unsigned)op1_limbs[3]
              + (int64_t)(unsigned)op0_limbs[5] * (int64_t)(unsigned)op1_limbs[2]
              + (int64_t)(unsigned)op0_limbs[6] * (int64_t)(unsigned)op1_limbs[1];
        z0[8] = (int64_t)(unsigned)op0_limbs[2] * (int64_t)(unsigned)op1_limbs[6]
              + (int64_t)(unsigned)op0_limbs[3] * (int64_t)(unsigned)op1_limbs[5]
              + (int64_t)(unsigned)op0_limbs[4] * (int64_t)(unsigned)op1_limbs[4]
              + (int64_t)(unsigned)op0_limbs[5] * (int64_t)(unsigned)op1_limbs[3]
              + (int64_t)(unsigned)op0_limbs[6] * (int64_t)(unsigned)op1_limbs[2];
        z0[9] = (int64_t)(unsigned)op0_limbs[3] * (int64_t)(unsigned)op1_limbs[6]
              + (int64_t)(unsigned)op0_limbs[4] * (int64_t)(unsigned)op1_limbs[5]
              + (int64_t)(unsigned)op0_limbs[5] * (int64_t)(unsigned)op1_limbs[4]
              + (int64_t)(unsigned)op0_limbs[6] * (int64_t)(unsigned)op1_limbs[3];
        z0[10] = (int64_t)(unsigned)op0_limbs[4] * (int64_t)(unsigned)op1_limbs[6]
               + (int64_t)(unsigned)op0_limbs[5] * (int64_t)(unsigned)op1_limbs[5]
               + (int64_t)(unsigned)op0_limbs[6] * (int64_t)(unsigned)op1_limbs[4];
        z0[11] = (int64_t)(unsigned)op0_limbs[5] * (int64_t)(unsigned)op1_limbs[6]
               + (int64_t)(unsigned)op0_limbs[6] * (int64_t)(unsigned)op1_limbs[5];
        z0[12] = (int64_t)(unsigned)op0_limbs[6] * (int64_t)(unsigned)op1_limbs[6];

        // z2: op0[7..14] * op1[7..14]
        int64_t z2[13];
        z2[0] = (int64_t)(unsigned)op0_limbs[7] * (int64_t)(unsigned)op1_limbs[7];
        z2[1] = (int64_t)(unsigned)op0_limbs[7] * (int64_t)(unsigned)op1_limbs[8]
              + (int64_t)(unsigned)op0_limbs[8] * (int64_t)(unsigned)op1_limbs[7];
        z2[2] = (int64_t)(unsigned)op0_limbs[7] * (int64_t)(unsigned)op1_limbs[9]
              + (int64_t)(unsigned)op0_limbs[8] * (int64_t)(unsigned)op1_limbs[8]
              + (int64_t)(unsigned)op0_limbs[9] * (int64_t)(unsigned)op1_limbs[7];
        z2[3] = (int64_t)(unsigned)op0_limbs[7] * (int64_t)(unsigned)op1_limbs[10]
              + (int64_t)(unsigned)op0_limbs[8] * (int64_t)(unsigned)op1_limbs[9]
              + (int64_t)(unsigned)op0_limbs[9] * (int64_t)(unsigned)op1_limbs[8]
              + (int64_t)(unsigned)op0_limbs[10] * (int64_t)(unsigned)op1_limbs[7];
        z2[4] = (int64_t)(unsigned)op0_limbs[7] * (int64_t)(unsigned)op1_limbs[11]
              + (int64_t)(unsigned)op0_limbs[8] * (int64_t)(unsigned)op1_limbs[10]
              + (int64_t)(unsigned)op0_limbs[9] * (int64_t)(unsigned)op1_limbs[9]
              + (int64_t)(unsigned)op0_limbs[10] * (int64_t)(unsigned)op1_limbs[8]
              + (int64_t)(unsigned)op0_limbs[11] * (int64_t)(unsigned)op1_limbs[7];
        z2[5] = (int64_t)(unsigned)op0_limbs[7] * (int64_t)(unsigned)op1_limbs[12]
              + (int64_t)(unsigned)op0_limbs[8] * (int64_t)(unsigned)op1_limbs[11]
              + (int64_t)(unsigned)op0_limbs[9] * (int64_t)(unsigned)op1_limbs[10]
              + (int64_t)(unsigned)op0_limbs[10] * (int64_t)(unsigned)op1_limbs[9]
              + (int64_t)(unsigned)op0_limbs[11] * (int64_t)(unsigned)op1_limbs[8]
              + (int64_t)(unsigned)op0_limbs[12] * (int64_t)(unsigned)op1_limbs[7];
        z2[6] = (int64_t)(unsigned)op0_limbs[7] * (int64_t)(unsigned)op1_limbs[13]
              + (int64_t)(unsigned)op0_limbs[8] * (int64_t)(unsigned)op1_limbs[12]
              + (int64_t)(unsigned)op0_limbs[9] * (int64_t)(unsigned)op1_limbs[11]
              + (int64_t)(unsigned)op0_limbs[10] * (int64_t)(unsigned)op1_limbs[10]
              + (int64_t)(unsigned)op0_limbs[11] * (int64_t)(unsigned)op1_limbs[9]
              + (int64_t)(unsigned)op0_limbs[12] * (int64_t)(unsigned)op1_limbs[8]
              + (int64_t)(unsigned)op0_limbs[13] * (int64_t)(unsigned)op1_limbs[7];
        z2[7] = (int64_t)(unsigned)op0_limbs[8] * (int64_t)(unsigned)op1_limbs[13]
              + (int64_t)(unsigned)op0_limbs[9] * (int64_t)(unsigned)op1_limbs[12]
              + (int64_t)(unsigned)op0_limbs[10] * (int64_t)(unsigned)op1_limbs[11]
              + (int64_t)(unsigned)op0_limbs[11] * (int64_t)(unsigned)op1_limbs[10]
              + (int64_t)(unsigned)op0_limbs[12] * (int64_t)(unsigned)op1_limbs[9]
              + (int64_t)(unsigned)op0_limbs[13] * (int64_t)(unsigned)op1_limbs[8];
        z2[8] = (int64_t)(unsigned)op0_limbs[9] * (int64_t)(unsigned)op1_limbs[13]
              + (int64_t)(unsigned)op0_limbs[10] * (int64_t)(unsigned)op1_limbs[12]
              + (int64_t)(unsigned)op0_limbs[11] * (int64_t)(unsigned)op1_limbs[11]
              + (int64_t)(unsigned)op0_limbs[12] * (int64_t)(unsigned)op1_limbs[10]
              + (int64_t)(unsigned)op0_limbs[13] * (int64_t)(unsigned)op1_limbs[9];
        z2[9] = (int64_t)(unsigned)op0_limbs[10] * (int64_t)(unsigned)op1_limbs[13]
              + (int64_t)(unsigned)op0_limbs[11] * (int64_t)(unsigned)op1_limbs[12]
              + (int64_t)(unsigned)op0_limbs[12] * (int64_t)(unsigned)op1_limbs[11]
              + (int64_t)(unsigned)op0_limbs[13] * (int64_t)(unsigned)op1_limbs[10];
        z2[10] = (int64_t)(unsigned)op0_limbs[11] * (int64_t)(unsigned)op1_limbs[13]
               + (int64_t)(unsigned)op0_limbs[12] * (int64_t)(unsigned)op1_limbs[12]
               + (int64_t)(unsigned)op0_limbs[13] * (int64_t)(unsigned)op1_limbs[11];
        z2[11] = (int64_t)(unsigned)op0_limbs[12] * (int64_t)(unsigned)op1_limbs[13]
               + (int64_t)(unsigned)op0_limbs[13] * (int64_t)(unsigned)op1_limbs[12];
        z2[12] = (int64_t)(unsigned)op0_limbs[13] * (int64_t)(unsigned)op1_limbs[13];

        // x_sum = op0[0..7] + op0[7..14], y_sum = op1[0..7] + op1[7..14]
        int64_t x_sum[7], y_sum[7];
        for (int i = 0; i < 7; i++) {
            x_sum[i] = (int64_t)(unsigned)op0_limbs[i] + (int64_t)(unsigned)op0_limbs[7 + i];
            y_sum[i] = (int64_t)(unsigned)op1_limbs[i] + (int64_t)(unsigned)op1_limbs[7 + i];
        }

        // z1_sum = x_sum * y_sum (convolution)
        int64_t z1_sum[13];
        for (int i = 0; i < 13; i++) z1_sum[i] = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                z1_sum[i + j] += x_sum[i] * y_sum[j];
            }
        }

        // single_karatsuba_output_0 = z0 for [0..7], then middle terms
        int64_t single_karatsuba_0[27];
        for (int i = 0; i < 7; i++) single_karatsuba_0[i] = z0[i];
        for (int i = 0; i < 6; i++) {
            single_karatsuba_0[7 + i] = z0[7 + i] + z1_sum[i] - z0[i] - z2[i];
        }
        single_karatsuba_0[13] = z1_sum[6] - z0[6] - z2[6];
        for (int i = 0; i < 6; i++) {
            single_karatsuba_0[14 + i] = z2[i] + z1_sum[7 + i] - z0[7 + i] - z2[7 + i];
        }
        for (int i = 0; i < 7; i++) single_karatsuba_0[20 + i] = z2[6 + i];

        // Second Karatsuba level: op0[14..21] * op1[14..21] and op0[21..28] * op1[21..28]
        int64_t z0_2[13], z2_2[13];

        // z0_2: op0[14..21] * op1[14..21]
        z0_2[0] = (int64_t)(unsigned)op0_limbs[14] * (int64_t)(unsigned)op1_limbs[14];
        z0_2[1] = (int64_t)(unsigned)op0_limbs[14] * (int64_t)(unsigned)op1_limbs[15]
                + (int64_t)(unsigned)op0_limbs[15] * (int64_t)(unsigned)op1_limbs[14];
        z0_2[2] = (int64_t)(unsigned)op0_limbs[14] * (int64_t)(unsigned)op1_limbs[16]
                + (int64_t)(unsigned)op0_limbs[15] * (int64_t)(unsigned)op1_limbs[15]
                + (int64_t)(unsigned)op0_limbs[16] * (int64_t)(unsigned)op1_limbs[14];
        z0_2[3] = (int64_t)(unsigned)op0_limbs[14] * (int64_t)(unsigned)op1_limbs[17]
                + (int64_t)(unsigned)op0_limbs[15] * (int64_t)(unsigned)op1_limbs[16]
                + (int64_t)(unsigned)op0_limbs[16] * (int64_t)(unsigned)op1_limbs[15]
                + (int64_t)(unsigned)op0_limbs[17] * (int64_t)(unsigned)op1_limbs[14];
        z0_2[4] = (int64_t)(unsigned)op0_limbs[14] * (int64_t)(unsigned)op1_limbs[18]
                + (int64_t)(unsigned)op0_limbs[15] * (int64_t)(unsigned)op1_limbs[17]
                + (int64_t)(unsigned)op0_limbs[16] * (int64_t)(unsigned)op1_limbs[16]
                + (int64_t)(unsigned)op0_limbs[17] * (int64_t)(unsigned)op1_limbs[15]
                + (int64_t)(unsigned)op0_limbs[18] * (int64_t)(unsigned)op1_limbs[14];
        z0_2[5] = (int64_t)(unsigned)op0_limbs[14] * (int64_t)(unsigned)op1_limbs[19]
                + (int64_t)(unsigned)op0_limbs[15] * (int64_t)(unsigned)op1_limbs[18]
                + (int64_t)(unsigned)op0_limbs[16] * (int64_t)(unsigned)op1_limbs[17]
                + (int64_t)(unsigned)op0_limbs[17] * (int64_t)(unsigned)op1_limbs[16]
                + (int64_t)(unsigned)op0_limbs[18] * (int64_t)(unsigned)op1_limbs[15]
                + (int64_t)(unsigned)op0_limbs[19] * (int64_t)(unsigned)op1_limbs[14];
        z0_2[6] = (int64_t)(unsigned)op0_limbs[14] * (int64_t)(unsigned)op1_limbs[20]
                + (int64_t)(unsigned)op0_limbs[15] * (int64_t)(unsigned)op1_limbs[19]
                + (int64_t)(unsigned)op0_limbs[16] * (int64_t)(unsigned)op1_limbs[18]
                + (int64_t)(unsigned)op0_limbs[17] * (int64_t)(unsigned)op1_limbs[17]
                + (int64_t)(unsigned)op0_limbs[18] * (int64_t)(unsigned)op1_limbs[16]
                + (int64_t)(unsigned)op0_limbs[19] * (int64_t)(unsigned)op1_limbs[15]
                + (int64_t)(unsigned)op0_limbs[20] * (int64_t)(unsigned)op1_limbs[14];
        z0_2[7] = (int64_t)(unsigned)op0_limbs[15] * (int64_t)(unsigned)op1_limbs[20]
                + (int64_t)(unsigned)op0_limbs[16] * (int64_t)(unsigned)op1_limbs[19]
                + (int64_t)(unsigned)op0_limbs[17] * (int64_t)(unsigned)op1_limbs[18]
                + (int64_t)(unsigned)op0_limbs[18] * (int64_t)(unsigned)op1_limbs[17]
                + (int64_t)(unsigned)op0_limbs[19] * (int64_t)(unsigned)op1_limbs[16]
                + (int64_t)(unsigned)op0_limbs[20] * (int64_t)(unsigned)op1_limbs[15];
        z0_2[8] = (int64_t)(unsigned)op0_limbs[16] * (int64_t)(unsigned)op1_limbs[20]
                + (int64_t)(unsigned)op0_limbs[17] * (int64_t)(unsigned)op1_limbs[19]
                + (int64_t)(unsigned)op0_limbs[18] * (int64_t)(unsigned)op1_limbs[18]
                + (int64_t)(unsigned)op0_limbs[19] * (int64_t)(unsigned)op1_limbs[17]
                + (int64_t)(unsigned)op0_limbs[20] * (int64_t)(unsigned)op1_limbs[16];
        z0_2[9] = (int64_t)(unsigned)op0_limbs[17] * (int64_t)(unsigned)op1_limbs[20]
                + (int64_t)(unsigned)op0_limbs[18] * (int64_t)(unsigned)op1_limbs[19]
                + (int64_t)(unsigned)op0_limbs[19] * (int64_t)(unsigned)op1_limbs[18]
                + (int64_t)(unsigned)op0_limbs[20] * (int64_t)(unsigned)op1_limbs[17];
        z0_2[10] = (int64_t)(unsigned)op0_limbs[18] * (int64_t)(unsigned)op1_limbs[20]
                 + (int64_t)(unsigned)op0_limbs[19] * (int64_t)(unsigned)op1_limbs[19]
                 + (int64_t)(unsigned)op0_limbs[20] * (int64_t)(unsigned)op1_limbs[18];
        z0_2[11] = (int64_t)(unsigned)op0_limbs[19] * (int64_t)(unsigned)op1_limbs[20]
                 + (int64_t)(unsigned)op0_limbs[20] * (int64_t)(unsigned)op1_limbs[19];
        z0_2[12] = (int64_t)(unsigned)op0_limbs[20] * (int64_t)(unsigned)op1_limbs[20];

        // z2_2: op0[21..28] * op1[21..28]
        z2_2[0] = (int64_t)(unsigned)op0_limbs[21] * (int64_t)(unsigned)op1_limbs[21];
        z2_2[1] = (int64_t)(unsigned)op0_limbs[21] * (int64_t)(unsigned)op1_limbs[22]
                + (int64_t)(unsigned)op0_limbs[22] * (int64_t)(unsigned)op1_limbs[21];
        z2_2[2] = (int64_t)(unsigned)op0_limbs[21] * (int64_t)(unsigned)op1_limbs[23]
                + (int64_t)(unsigned)op0_limbs[22] * (int64_t)(unsigned)op1_limbs[22]
                + (int64_t)(unsigned)op0_limbs[23] * (int64_t)(unsigned)op1_limbs[21];
        z2_2[3] = (int64_t)(unsigned)op0_limbs[21] * (int64_t)(unsigned)op1_limbs[24]
                + (int64_t)(unsigned)op0_limbs[22] * (int64_t)(unsigned)op1_limbs[23]
                + (int64_t)(unsigned)op0_limbs[23] * (int64_t)(unsigned)op1_limbs[22]
                + (int64_t)(unsigned)op0_limbs[24] * (int64_t)(unsigned)op1_limbs[21];
        z2_2[4] = (int64_t)(unsigned)op0_limbs[21] * (int64_t)(unsigned)op1_limbs[25]
                + (int64_t)(unsigned)op0_limbs[22] * (int64_t)(unsigned)op1_limbs[24]
                + (int64_t)(unsigned)op0_limbs[23] * (int64_t)(unsigned)op1_limbs[23]
                + (int64_t)(unsigned)op0_limbs[24] * (int64_t)(unsigned)op1_limbs[22]
                + (int64_t)(unsigned)op0_limbs[25] * (int64_t)(unsigned)op1_limbs[21];
        z2_2[5] = (int64_t)(unsigned)op0_limbs[21] * (int64_t)(unsigned)op1_limbs[26]
                + (int64_t)(unsigned)op0_limbs[22] * (int64_t)(unsigned)op1_limbs[25]
                + (int64_t)(unsigned)op0_limbs[23] * (int64_t)(unsigned)op1_limbs[24]
                + (int64_t)(unsigned)op0_limbs[24] * (int64_t)(unsigned)op1_limbs[23]
                + (int64_t)(unsigned)op0_limbs[25] * (int64_t)(unsigned)op1_limbs[22]
                + (int64_t)(unsigned)op0_limbs[26] * (int64_t)(unsigned)op1_limbs[21];
        z2_2[6] = (int64_t)(unsigned)op0_limbs[21] * (int64_t)(unsigned)op1_limbs[27]
                + (int64_t)(unsigned)op0_limbs[22] * (int64_t)(unsigned)op1_limbs[26]
                + (int64_t)(unsigned)op0_limbs[23] * (int64_t)(unsigned)op1_limbs[25]
                + (int64_t)(unsigned)op0_limbs[24] * (int64_t)(unsigned)op1_limbs[24]
                + (int64_t)(unsigned)op0_limbs[25] * (int64_t)(unsigned)op1_limbs[23]
                + (int64_t)(unsigned)op0_limbs[26] * (int64_t)(unsigned)op1_limbs[22]
                + (int64_t)(unsigned)op0_limbs[27] * (int64_t)(unsigned)op1_limbs[21];
        z2_2[7] = (int64_t)(unsigned)op0_limbs[22] * (int64_t)(unsigned)op1_limbs[27]
                + (int64_t)(unsigned)op0_limbs[23] * (int64_t)(unsigned)op1_limbs[26]
                + (int64_t)(unsigned)op0_limbs[24] * (int64_t)(unsigned)op1_limbs[25]
                + (int64_t)(unsigned)op0_limbs[25] * (int64_t)(unsigned)op1_limbs[24]
                + (int64_t)(unsigned)op0_limbs[26] * (int64_t)(unsigned)op1_limbs[23]
                + (int64_t)(unsigned)op0_limbs[27] * (int64_t)(unsigned)op1_limbs[22];
        z2_2[8] = (int64_t)(unsigned)op0_limbs[23] * (int64_t)(unsigned)op1_limbs[27]
                + (int64_t)(unsigned)op0_limbs[24] * (int64_t)(unsigned)op1_limbs[26]
                + (int64_t)(unsigned)op0_limbs[25] * (int64_t)(unsigned)op1_limbs[25]
                + (int64_t)(unsigned)op0_limbs[26] * (int64_t)(unsigned)op1_limbs[24]
                + (int64_t)(unsigned)op0_limbs[27] * (int64_t)(unsigned)op1_limbs[23];
        z2_2[9] = (int64_t)(unsigned)op0_limbs[24] * (int64_t)(unsigned)op1_limbs[27]
                + (int64_t)(unsigned)op0_limbs[25] * (int64_t)(unsigned)op1_limbs[26]
                + (int64_t)(unsigned)op0_limbs[26] * (int64_t)(unsigned)op1_limbs[25]
                + (int64_t)(unsigned)op0_limbs[27] * (int64_t)(unsigned)op1_limbs[24];
        z2_2[10] = (int64_t)(unsigned)op0_limbs[25] * (int64_t)(unsigned)op1_limbs[27]
                 + (int64_t)(unsigned)op0_limbs[26] * (int64_t)(unsigned)op1_limbs[26]
                 + (int64_t)(unsigned)op0_limbs[27] * (int64_t)(unsigned)op1_limbs[25];
        z2_2[11] = (int64_t)(unsigned)op0_limbs[26] * (int64_t)(unsigned)op1_limbs[27]
                 + (int64_t)(unsigned)op0_limbs[27] * (int64_t)(unsigned)op1_limbs[26];
        z2_2[12] = (int64_t)(unsigned)op0_limbs[27] * (int64_t)(unsigned)op1_limbs[27];

        // x_sum_2 = op0[14..21] + op0[21..28], y_sum_2 = op1[14..21] + op1[21..28]
        int64_t x_sum_2[7], y_sum_2[7];
        for (int i = 0; i < 7; i++) {
            x_sum_2[i] = (int64_t)(unsigned)op0_limbs[14 + i] + (int64_t)(unsigned)op0_limbs[21 + i];
            y_sum_2[i] = (int64_t)(unsigned)op1_limbs[14 + i] + (int64_t)(unsigned)op1_limbs[21 + i];
        }

        // z1_sum_2 = x_sum_2 * y_sum_2
        int64_t z1_sum_2[13];
        for (int i = 0; i < 13; i++) z1_sum_2[i] = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                z1_sum_2[i + j] += x_sum_2[i] * y_sum_2[j];
            }
        }

        // single_karatsuba_output_1
        int64_t single_karatsuba_1[27];
        for (int i = 0; i < 7; i++) single_karatsuba_1[i] = z0_2[i];
        for (int i = 0; i < 6; i++) {
            single_karatsuba_1[7 + i] = z0_2[7 + i] + z1_sum_2[i] - z0_2[i] - z2_2[i];
        }
        single_karatsuba_1[13] = z1_sum_2[6] - z0_2[6] - z2_2[6];
        for (int i = 0; i < 6; i++) {
            single_karatsuba_1[14 + i] = z2_2[i] + z1_sum_2[7 + i] - z0_2[7 + i] - z2_2[7 + i];
        }
        for (int i = 0; i < 7; i++) single_karatsuba_1[20 + i] = z2_2[6 + i];

        // Double Karatsuba: combine single_karatsuba_0 and single_karatsuba_1
        // x_sum_outer = op0[0..14] + op0[14..28], y_sum_outer = op1[0..14] + op1[14..28]
        int64_t x_sum_outer[14], y_sum_outer[14];
        for (int i = 0; i < 14; i++) {
            x_sum_outer[i] = (int64_t)(unsigned)op0_limbs[i] + (int64_t)(unsigned)op0_limbs[14 + i];
            y_sum_outer[i] = (int64_t)(unsigned)op1_limbs[i] + (int64_t)(unsigned)op1_limbs[14 + i];
        }

        // Compute x_sum_outer * y_sum_outer using another level of Karatsuba
        int64_t x_sum_outer_lo[7], x_sum_outer_hi[7], y_sum_outer_lo[7], y_sum_outer_hi[7];
        for (int i = 0; i < 7; i++) {
            x_sum_outer_lo[i] = x_sum_outer[i];
            x_sum_outer_hi[i] = x_sum_outer[7 + i];
            y_sum_outer_lo[i] = y_sum_outer[i];
            y_sum_outer_hi[i] = y_sum_outer[7 + i];
        }

        // z0_outer = x_sum_outer_lo * y_sum_outer_lo
        int64_t z0_outer[13];
        for (int i = 0; i < 13; i++) z0_outer[i] = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                z0_outer[i + j] += x_sum_outer_lo[i] * y_sum_outer_lo[j];
            }
        }

        // z2_outer = x_sum_outer_hi * y_sum_outer_hi
        int64_t z2_outer[13];
        for (int i = 0; i < 13; i++) z2_outer[i] = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                z2_outer[i + j] += x_sum_outer_hi[i] * y_sum_outer_hi[j];
            }
        }

        // z1_sum_outer = (x_sum_outer_lo + x_sum_outer_hi) * (y_sum_outer_lo + y_sum_outer_hi)
        int64_t x_sum_outer_sum[7], y_sum_outer_sum[7];
        for (int i = 0; i < 7; i++) {
            x_sum_outer_sum[i] = x_sum_outer_lo[i] + x_sum_outer_hi[i];
            y_sum_outer_sum[i] = y_sum_outer_lo[i] + y_sum_outer_hi[i];
        }
        int64_t z1_sum_outer[13];
        for (int i = 0; i < 13; i++) z1_sum_outer[i] = 0;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                z1_sum_outer[i + j] += x_sum_outer_sum[i] * y_sum_outer_sum[j];
            }
        }

        // single_karatsuba_outer
        int64_t single_karatsuba_outer[27];
        for (int i = 0; i < 7; i++) single_karatsuba_outer[i] = z0_outer[i];
        for (int i = 0; i < 6; i++) {
            single_karatsuba_outer[7 + i] = z0_outer[7 + i] + z1_sum_outer[i] - z0_outer[i] - z2_outer[i];
        }
        single_karatsuba_outer[13] = z1_sum_outer[6] - z0_outer[6] - z2_outer[6];
        for (int i = 0; i < 6; i++) {
            single_karatsuba_outer[14 + i] = z2_outer[i] + z1_sum_outer[7 + i] - z0_outer[7 + i] - z2_outer[7 + i];
        }
        for (int i = 0; i < 7; i++) single_karatsuba_outer[20 + i] = z2_outer[6 + i];

        // Final double Karatsuba result (55 limbs)
        int64_t double_karatsuba[55];
        for (int i = 0; i < 14; i++) double_karatsuba[i] = single_karatsuba_0[i];
        for (int i = 0; i < 13; i++) {
            double_karatsuba[14 + i] = single_karatsuba_0[14 + i] + single_karatsuba_outer[i] - single_karatsuba_0[i] - single_karatsuba_1[i];
        }
        double_karatsuba[27] = single_karatsuba_outer[13] - single_karatsuba_0[13] - single_karatsuba_1[13];
        for (int i = 0; i < 13; i++) {
            double_karatsuba[28 + i] = single_karatsuba_1[i] + single_karatsuba_outer[14 + i] - single_karatsuba_0[14 + i] - single_karatsuba_1[14 + i];
        }
        for (int i = 0; i < 14; i++) double_karatsuba[41 + i] = single_karatsuba_1[13 + i];

        // conv = double_karatsuba - dst (difference between product and stored result)
        int64_t conv[55];
        for (int i = 0; i < 28; i++) {
            conv[i] = double_karatsuba[i] - (int64_t)(unsigned)dst_limbs[i];
        }
        for (int i = 28; i < 55; i++) {
            conv[i] = double_karatsuba[i];
        }

        // conv_mod: apply modular reduction formula
        int64_t conv_mod[28];
        conv_mod[0] = 32 * conv[0] - 4 * conv[21] + 8 * conv[49];
        conv_mod[1] = conv[0] + 32 * conv[1] - 4 * conv[22] + 8 * conv[50];
        conv_mod[2] = conv[1] + 32 * conv[2] - 4 * conv[23] + 8 * conv[51];
        conv_mod[3] = conv[2] + 32 * conv[3] - 4 * conv[24] + 8 * conv[52];
        conv_mod[4] = conv[3] + 32 * conv[4] - 4 * conv[25] + 8 * conv[53];
        conv_mod[5] = conv[4] + 32 * conv[5] - 4 * conv[26] + 8 * conv[54];
        conv_mod[6] = conv[5] + 32 * conv[6] - 4 * conv[27];
        conv_mod[7] = 2 * conv[0] + conv[6] + 32 * conv[7] - 4 * conv[28];
        conv_mod[8] = 2 * conv[1] + conv[7] + 32 * conv[8] - 4 * conv[29];
        conv_mod[9] = 2 * conv[2] + conv[8] + 32 * conv[9] - 4 * conv[30];
        conv_mod[10] = 2 * conv[3] + conv[9] + 32 * conv[10] - 4 * conv[31];
        conv_mod[11] = 2 * conv[4] + conv[10] + 32 * conv[11] - 4 * conv[32];
        conv_mod[12] = 2 * conv[5] + conv[11] + 32 * conv[12] - 4 * conv[33];
        conv_mod[13] = 2 * conv[6] + conv[12] + 32 * conv[13] - 4 * conv[34];
        conv_mod[14] = 2 * conv[7] + conv[13] + 32 * conv[14] - 4 * conv[35];
        conv_mod[15] = 2 * conv[8] + conv[14] + 32 * conv[15] - 4 * conv[36];
        conv_mod[16] = 2 * conv[9] + conv[15] + 32 * conv[16] - 4 * conv[37];
        conv_mod[17] = 2 * conv[10] + conv[16] + 32 * conv[17] - 4 * conv[38];
        conv_mod[18] = 2 * conv[11] + conv[17] + 32 * conv[18] - 4 * conv[39];
        conv_mod[19] = 2 * conv[12] + conv[18] + 32 * conv[19] - 4 * conv[40];
        conv_mod[20] = 2 * conv[13] + conv[19] + 32 * conv[20] - 4 * conv[41];
        conv_mod[21] = 2 * conv[14] + conv[20] - 4 * conv[42] + 64 * conv[49];
        conv_mod[22] = 2 * conv[15] - 4 * conv[43] + 2 * conv[49] + 64 * conv[50];
        conv_mod[23] = 2 * conv[16] - 4 * conv[44] + 2 * conv[50] + 64 * conv[51];
        conv_mod[24] = 2 * conv[17] - 4 * conv[45] + 2 * conv[51] + 64 * conv[52];
        conv_mod[25] = 2 * conv[18] - 4 * conv[46] + 2 * conv[52] + 64 * conv[53];
        conv_mod[26] = 2 * conv[19] - 4 * conv[47] + 2 * conv[53] + 64 * conv[54];
        conv_mod[27] = 2 * conv[20] - 4 * conv[48] + 2 * conv[54];

        // k_mod_2_18_biased calculation
        uint32_t k_mod_tmp = ((uint32_t)(conv_mod[0] + 134217728) + (((uint32_t)(conv_mod[1] + 134217728) & 511) << 9) + 65536) & 262143;
        int64_t k_val = (int64_t)(k_mod_tmp & 0xFFFF) + (int64_t)((int32_t)((k_mod_tmp >> 16) & 0x3) - 1) * 65536;
        m31 k_col101 = m31{(unsigned)((k_val % P + P) % P)};
        traces[101][row] = k_col101;

        sub_component_inputss_range_check_19_h[0][row] = add(k_col101, M31_262144);
        lookup_range_check_19_h_0[0][row] = add(k_col101, M31_262144);

        // Carry computations
        int64_t carry[27];
        carry[0] = (conv_mod[0] - k_val) / 512;
        m31 carry_0_col102 = m31{(unsigned)((carry[0] % P + P) % P)};
        traces[102][row] = carry_0_col102;
        sub_component_inputss_range_check_19[0][row] = add(carry_0_col102, M31_131072);
        lookup_range_check_19_0[0][row] = add(carry_0_col102, M31_131072);

        for (int i = 1; i < 21; i++) {
            carry[i] = (conv_mod[i] + carry[i-1]) / 512;
        }
        carry[21] = (conv_mod[21] - 136 * k_val + carry[20]) / 512;
        for (int i = 22; i < 27; i++) {
            carry[i] = (conv_mod[i] + carry[i-1]) / 512;
        }

        // Store carries and range check lookups
        m31 carry_1_col103 = m31{(unsigned)((carry[1] % P + P) % P)};
        traces[103][row] = carry_1_col103;
        sub_component_inputss_range_check_19_b[0][row] = add(carry_1_col103, M31_131072);
        lookup_range_check_19_b_0[0][row] = add(carry_1_col103, M31_131072);

        m31 carry_2_col104 = m31{(unsigned)((carry[2] % P + P) % P)};
        traces[104][row] = carry_2_col104;
        sub_component_inputss_range_check_19_c[0][row] = add(carry_2_col104, M31_131072);
        lookup_range_check_19_c_0[0][row] = add(carry_2_col104, M31_131072);

        m31 carry_3_col105 = m31{(unsigned)((carry[3] % P + P) % P)};
        traces[105][row] = carry_3_col105;
        sub_component_inputss_range_check_19_d[0][row] = add(carry_3_col105, M31_131072);
        lookup_range_check_19_d_0[0][row] = add(carry_3_col105, M31_131072);

        m31 carry_4_col106 = m31{(unsigned)((carry[4] % P + P) % P)};
        traces[106][row] = carry_4_col106;
        sub_component_inputss_range_check_19_e[0][row] = add(carry_4_col106, M31_131072);
        lookup_range_check_19_e_0[0][row] = add(carry_4_col106, M31_131072);

        m31 carry_5_col107 = m31{(unsigned)((carry[5] % P + P) % P)};
        traces[107][row] = carry_5_col107;
        sub_component_inputss_range_check_19_f[0][row] = add(carry_5_col107, M31_131072);
        lookup_range_check_19_f_0[0][row] = add(carry_5_col107, M31_131072);

        m31 carry_6_col108 = m31{(unsigned)((carry[6] % P + P) % P)};
        traces[108][row] = carry_6_col108;
        sub_component_inputss_range_check_19_g[0][row] = add(carry_6_col108, M31_131072);
        lookup_range_check_19_g_0[0][row] = add(carry_6_col108, M31_131072);

        m31 carry_7_col109 = m31{(unsigned)((carry[7] % P + P) % P)};
        traces[109][row] = carry_7_col109;
        sub_component_inputss_range_check_19_h[1][row] = add(carry_7_col109, M31_131072);
        lookup_range_check_19_h_1[0][row] = add(carry_7_col109, M31_131072);

        m31 carry_8_col110 = m31{(unsigned)((carry[8] % P + P) % P)};
        traces[110][row] = carry_8_col110;
        sub_component_inputss_range_check_19[1][row] = add(carry_8_col110, M31_131072);
        lookup_range_check_19_1[0][row] = add(carry_8_col110, M31_131072);

        m31 carry_9_col111 = m31{(unsigned)((carry[9] % P + P) % P)};
        traces[111][row] = carry_9_col111;
        sub_component_inputss_range_check_19_b[1][row] = add(carry_9_col111, M31_131072);
        lookup_range_check_19_b_1[0][row] = add(carry_9_col111, M31_131072);

        m31 carry_10_col112 = m31{(unsigned)((carry[10] % P + P) % P)};
        traces[112][row] = carry_10_col112;
        sub_component_inputss_range_check_19_c[1][row] = add(carry_10_col112, M31_131072);
        lookup_range_check_19_c_1[0][row] = add(carry_10_col112, M31_131072);

        m31 carry_11_col113 = m31{(unsigned)((carry[11] % P + P) % P)};
        traces[113][row] = carry_11_col113;
        sub_component_inputss_range_check_19_d[1][row] = add(carry_11_col113, M31_131072);
        lookup_range_check_19_d_1[0][row] = add(carry_11_col113, M31_131072);

        m31 carry_12_col114 = m31{(unsigned)((carry[12] % P + P) % P)};
        traces[114][row] = carry_12_col114;
        sub_component_inputss_range_check_19_e[1][row] = add(carry_12_col114, M31_131072);
        lookup_range_check_19_e_1[0][row] = add(carry_12_col114, M31_131072);

        m31 carry_13_col115 = m31{(unsigned)((carry[13] % P + P) % P)};
        traces[115][row] = carry_13_col115;
        sub_component_inputss_range_check_19_f[1][row] = add(carry_13_col115, M31_131072);
        lookup_range_check_19_f_1[0][row] = add(carry_13_col115, M31_131072);

        m31 carry_14_col116 = m31{(unsigned)((carry[14] % P + P) % P)};
        traces[116][row] = carry_14_col116;
        sub_component_inputss_range_check_19_g[1][row] = add(carry_14_col116, M31_131072);
        lookup_range_check_19_g_1[0][row] = add(carry_14_col116, M31_131072);

        m31 carry_15_col117 = m31{(unsigned)((carry[15] % P + P) % P)};
        traces[117][row] = carry_15_col117;
        sub_component_inputss_range_check_19_h[2][row] = add(carry_15_col117, M31_131072);
        lookup_range_check_19_h_2[0][row] = add(carry_15_col117, M31_131072);

        m31 carry_16_col118 = m31{(unsigned)((carry[16] % P + P) % P)};
        traces[118][row] = carry_16_col118;
        sub_component_inputss_range_check_19[2][row] = add(carry_16_col118, M31_131072);
        lookup_range_check_19_2[0][row] = add(carry_16_col118, M31_131072);

        m31 carry_17_col119 = m31{(unsigned)((carry[17] % P + P) % P)};
        traces[119][row] = carry_17_col119;
        sub_component_inputss_range_check_19_b[2][row] = add(carry_17_col119, M31_131072);
        lookup_range_check_19_b_2[0][row] = add(carry_17_col119, M31_131072);

        m31 carry_18_col120 = m31{(unsigned)((carry[18] % P + P) % P)};
        traces[120][row] = carry_18_col120;
        sub_component_inputss_range_check_19_c[2][row] = add(carry_18_col120, M31_131072);
        lookup_range_check_19_c_2[0][row] = add(carry_18_col120, M31_131072);

        m31 carry_19_col121 = m31{(unsigned)((carry[19] % P + P) % P)};
        traces[121][row] = carry_19_col121;
        sub_component_inputss_range_check_19_d[2][row] = add(carry_19_col121, M31_131072);
        lookup_range_check_19_d_2[0][row] = add(carry_19_col121, M31_131072);

        m31 carry_20_col122 = m31{(unsigned)((carry[20] % P + P) % P)};
        traces[122][row] = carry_20_col122;
        sub_component_inputss_range_check_19_e[2][row] = add(carry_20_col122, M31_131072);
        lookup_range_check_19_e_2[0][row] = add(carry_20_col122, M31_131072);

        m31 carry_21_col123 = m31{(unsigned)((carry[21] % P + P) % P)};
        traces[123][row] = carry_21_col123;
        sub_component_inputss_range_check_19_f[2][row] = add(carry_21_col123, M31_131072);
        lookup_range_check_19_f_2[0][row] = add(carry_21_col123, M31_131072);

        m31 carry_22_col124 = m31{(unsigned)((carry[22] % P + P) % P)};
        traces[124][row] = carry_22_col124;
        sub_component_inputss_range_check_19_g[2][row] = add(carry_22_col124, M31_131072);
        lookup_range_check_19_g_2[0][row] = add(carry_22_col124, M31_131072);

        m31 carry_23_col125 = m31{(unsigned)((carry[23] % P + P) % P)};
        traces[125][row] = carry_23_col125;
        sub_component_inputss_range_check_19_h[3][row] = add(carry_23_col125, M31_131072);
        lookup_range_check_19_h_3[0][row] = add(carry_23_col125, M31_131072);

        m31 carry_24_col126 = m31{(unsigned)((carry[24] % P + P) % P)};
        traces[126][row] = carry_24_col126;
        sub_component_inputss_range_check_19[3][row] = add(carry_24_col126, M31_131072);
        lookup_range_check_19_3[0][row] = add(carry_24_col126, M31_131072);

        m31 carry_25_col127 = m31{(unsigned)((carry[25] % P + P) % P)};
        traces[127][row] = carry_25_col127;
        sub_component_inputss_range_check_19_b[3][row] = add(carry_25_col127, M31_131072);
        lookup_range_check_19_b_3[0][row] = add(carry_25_col127, M31_131072);

        m31 carry_26_col128 = m31{(unsigned)((carry[26] % P + P) % P)};
        traces[128][row] = carry_26_col128;
        sub_component_inputss_range_check_19_c[3][row] = add(carry_26_col128, M31_131072);
        lookup_range_check_19_c_3[0][row] = add(carry_26_col128, M31_131072);

        // Opcodes lookups
        lookup_opcodes_0[0][row] = input_pc_col0;
        lookup_opcodes_0[1][row] = input_ap_col1;
        lookup_opcodes_0[2][row] = input_fp_col2;

        // next_pc = pc + 1 + op1_imm
        lookup_opcodes_1[0][row] = add(add(input_pc_col0, M31_1), op1_imm_col8);
        lookup_opcodes_1[1][row] = add(input_ap_col1, ap_update_add_1_col10);
        lookup_opcodes_1[2][row] = input_fp_col2;

        // Enabler column (col 129)
        traces[129][row] = (row < n_rows) ? M31_1 : M31_0;
    }
}

// Wrapper function for base trace generation
void generate_mul_opcode_traces(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
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
    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_19,
    unsigned **sub_component_inputs_range_check_19_b,
    unsigned **sub_component_inputs_range_check_19_c,
    unsigned **sub_component_inputs_range_check_19_d,
    unsigned **sub_component_inputs_range_check_19_e,
    unsigned **sub_component_inputs_range_check_19_f,
    unsigned **sub_component_inputs_range_check_19_g,
    unsigned **sub_component_inputs_range_check_19_h,

    unsigned **mul_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    m31 **device_traces = clone_to_device<m31*>((m31 **)traces, MUL_OPCODE_N_TRACE_COLUMNS);

    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>((m31 **)lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>((m31 **)lookup_opcodes_1, 3);
    m31 **device_lookup_range_check_19_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_0, 1);
    m31 **device_lookup_range_check_19_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_1, 1);
    m31 **device_lookup_range_check_19_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_2, 1);
    m31 **device_lookup_range_check_19_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_3, 1);
    m31 **device_lookup_range_check_19_b_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_0, 1);
    m31 **device_lookup_range_check_19_b_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_1, 1);
    m31 **device_lookup_range_check_19_b_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_2, 1);
    m31 **device_lookup_range_check_19_b_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_3, 1);
    m31 **device_lookup_range_check_19_c_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_0, 1);
    m31 **device_lookup_range_check_19_c_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_1, 1);
    m31 **device_lookup_range_check_19_c_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_2, 1);
    m31 **device_lookup_range_check_19_c_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_3, 1);
    m31 **device_lookup_range_check_19_d_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_d_0, 1);
    m31 **device_lookup_range_check_19_d_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_d_1, 1);
    m31 **device_lookup_range_check_19_d_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_d_2, 1);
    m31 **device_lookup_range_check_19_e_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_e_0, 1);
    m31 **device_lookup_range_check_19_e_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_e_1, 1);
    m31 **device_lookup_range_check_19_e_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_e_2, 1);
    m31 **device_lookup_range_check_19_f_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_f_0, 1);
    m31 **device_lookup_range_check_19_f_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_f_1, 1);
    m31 **device_lookup_range_check_19_f_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_f_2, 1);
    m31 **device_lookup_range_check_19_g_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_g_0, 1);
    m31 **device_lookup_range_check_19_g_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_g_1, 1);
    m31 **device_lookup_range_check_19_g_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_g_2, 1);
    m31 **device_lookup_range_check_19_h_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_0, 1);
    m31 **device_lookup_range_check_19_h_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_1, 1);
    m31 **device_lookup_range_check_19_h_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_2, 1);
    m31 **device_lookup_range_check_19_h_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_3, 1);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>((m31 **)lookup_verify_instruction_0, 7);

    m31 **device_sub_component_inputs_verify_instruction = clone_to_device<m31 *>((m31 **)sub_component_inputs_verify_instruction, 7);
    m31 **device_sub_component_inputs_memory_address_to_id = clone_to_device<m31 *>((m31 **)sub_component_inputs_memory_address_to_id, 3);
    m31 **device_sub_component_inputs_memory_id_to_big = clone_to_device<m31 *>((m31 **)sub_component_inputs_memory_id_to_big, 3);
    m31 **device_sub_component_inputs_range_check_19 = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19, 4);
    m31 **device_sub_component_inputs_range_check_19_b = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19_b, 4);
    m31 **device_sub_component_inputs_range_check_19_c = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19_c, 4);
    m31 **device_sub_component_inputs_range_check_19_d = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19_d, 3);
    m31 **device_sub_component_inputs_range_check_19_e = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19_e, 3);
    m31 **device_sub_component_inputs_range_check_19_f = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19_f, 3);
    m31 **device_sub_component_inputs_range_check_19_g = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19_g, 3);
    m31 **device_sub_component_inputs_range_check_19_h = clone_to_device<m31 *>((m31 **)sub_component_inputs_range_check_19_h, 4);

    m31 **device_mul_opcode_input = clone_to_device<m31 *>((m31 **)mul_opcode_input, 3);
    unsigned **device_memory_id_to_big_transposed_big_values = clone_to_device<m31 *>((m31 **)memory_id_to_big_transposed_big_values, 8);


    timer global_timer;
    global_timer.start("generate mul_opcode base trace");

    unsigned trace_size = 1 << log_size;
    int block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_mul_opcode_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_id_to_big_1,
        device_lookup_memory_id_to_big_2,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
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
        device_lookup_verify_instruction_0,
        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        device_sub_component_inputs_range_check_19,
        device_sub_component_inputs_range_check_19_b,
        device_sub_component_inputs_range_check_19_c,
        device_sub_component_inputs_range_check_19_d,
        device_sub_component_inputs_range_check_19_e,
        device_sub_component_inputs_range_check_19_f,
        device_sub_component_inputs_range_check_19_g,
        device_sub_component_inputs_range_check_19_h,
        device_mul_opcode_input,
        memory_address_to_id_address_to_raw_id,
        device_memory_id_to_big_transposed_big_values,
        memory_id_to_big_small_values,
        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate mul_opcode base trace");

    // Free device memory
    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
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
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);
    cuda_free_memory(device_sub_component_inputs_range_check_19);
    cuda_free_memory(device_sub_component_inputs_range_check_19_b);
    cuda_free_memory(device_sub_component_inputs_range_check_19_c);
    cuda_free_memory(device_sub_component_inputs_range_check_19_d);
    cuda_free_memory(device_sub_component_inputs_range_check_19_e);
    cuda_free_memory(device_sub_component_inputs_range_check_19_f);
    cuda_free_memory(device_sub_component_inputs_range_check_19_g);
    cuda_free_memory(device_sub_component_inputs_range_check_19_h);
    cuda_free_memory(device_mul_opcode_input);
    cuda_free_memory(device_memory_id_to_big_transposed_big_values);
}

// ============================================================================
// Interaction trace generation kernels
// ============================================================================

// Round 0: verify_instruction_0 + memory_address_to_id_0
template <int N_LOOKUP_COLUMNS_0, int N_LOOKUP_COLUMNS_1>
__global__ void generate_mul_opcode_interaction_trace_col_gen_kernel_round0(
    VerifyInstruction *verify_instruction,
    MemoryAddressToId *memory_address_to_id,
    m31 **lookup_values_0,
    m31 **lookup_values_1,
    unsigned trace_size,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        m31 lookup_vals_0[N_LOOKUP_COLUMNS_0];
        m31 lookup_vals_1[N_LOOKUP_COLUMNS_1];

        for (int i = 0; i < N_LOOKUP_COLUMNS_0; i++) {
            lookup_vals_0[i] = lookup_values_0[i][vec_index];
        }
        for (int i = 0; i < N_LOOKUP_COLUMNS_1; i++) {
            lookup_vals_1[i] = lookup_values_1[i][vec_index];
        }

        qm31 denom0 = verify_instruction->combine(lookup_vals_0, N_LOOKUP_COLUMNS_0);
        qm31 denom1 = memory_address_to_id->combine(lookup_vals_1, N_LOOKUP_COLUMNS_1);

        qm31 num = add(denom0, denom1);
        denom[vec_index] = mul(denom0, denom1);

        numerator0[vec_index] = num.a.a;
        numerator1[vec_index] = num.a.b;
        numerator2[vec_index] = num.b.a;
        numerator3[vec_index] = num.b.b;
    }
}

// Round 1-2: memory_id_to_big + memory_address_to_id
template <int N_LOOKUP_COLUMNS_0, int N_LOOKUP_COLUMNS_1>
__global__ void generate_mul_opcode_interaction_trace_col_gen_kernel_mem_id_big_addr(
    MemoryIdToBig *memory_id_to_big,
    MemoryAddressToId *memory_address_to_id,
    m31 **lookup_values_0,
    m31 **lookup_values_1,
    unsigned trace_size,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        m31 lookup_vals_0[N_LOOKUP_COLUMNS_0];
        m31 lookup_vals_1[N_LOOKUP_COLUMNS_1];

        for (int i = 0; i < N_LOOKUP_COLUMNS_0; i++) {
            lookup_vals_0[i] = lookup_values_0[i][vec_index];
        }
        for (int i = 0; i < N_LOOKUP_COLUMNS_1; i++) {
            lookup_vals_1[i] = lookup_values_1[i][vec_index];
        }

        qm31 denom0 = memory_id_to_big->combine(lookup_vals_0, N_LOOKUP_COLUMNS_0);
        qm31 denom1 = memory_address_to_id->combine(lookup_vals_1, N_LOOKUP_COLUMNS_1);

        qm31 num = add(denom0, denom1);
        denom[vec_index] = mul(denom0, denom1);

        numerator0[vec_index] = num.a.a;
        numerator1[vec_index] = num.a.b;
        numerator2[vec_index] = num.b.a;
        numerator3[vec_index] = num.b.b;
    }
}

// Round 3: memory_id_to_big_2 + range_check_19_h_0
template <int N_LOOKUP_COLUMNS_0, int N_LOOKUP_COLUMNS_1>
__global__ void generate_mul_opcode_interaction_trace_col_gen_kernel_mem_id_big_rc(
    MemoryIdToBig *memory_id_to_big,
    RangeCheck_19_H *range_check_19_h,
    m31 **lookup_values_0,
    m31 **lookup_values_1,
    unsigned trace_size,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        m31 lookup_vals_0[N_LOOKUP_COLUMNS_0];
        m31 lookup_vals_1[N_LOOKUP_COLUMNS_1];

        for (int i = 0; i < N_LOOKUP_COLUMNS_0; i++) {
            lookup_vals_0[i] = lookup_values_0[i][vec_index];
        }
        for (int i = 0; i < N_LOOKUP_COLUMNS_1; i++) {
            lookup_vals_1[i] = lookup_values_1[i][vec_index];
        }

        qm31 denom0 = memory_id_to_big->combine(lookup_vals_0, N_LOOKUP_COLUMNS_0);
        qm31 denom1 = range_check_19_h->combine(lookup_vals_1, N_LOOKUP_COLUMNS_1);

        qm31 num = add(denom0, denom1);
        denom[vec_index] = mul(denom0, denom1);

        numerator0[vec_index] = num.a.a;
        numerator1[vec_index] = num.a.b;
        numerator2[vec_index] = num.b.a;
        numerator3[vec_index] = num.b.b;
    }
}

// Rounds 4-16: range_check pairs (generic for all range check combinations)
template <int N_LOOKUP_COLUMNS_0, int N_LOOKUP_COLUMNS_1, typename RC0, typename RC1>
__global__ void generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair(
    RC0 *range_check_0,
    RC1 *range_check_1,
    m31 **lookup_values_0,
    m31 **lookup_values_1,
    unsigned trace_size,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        m31 lookup_vals_0[N_LOOKUP_COLUMNS_0];
        m31 lookup_vals_1[N_LOOKUP_COLUMNS_1];

        for (int i = 0; i < N_LOOKUP_COLUMNS_0; i++) {
            lookup_vals_0[i] = lookup_values_0[i][vec_index];
        }
        for (int i = 0; i < N_LOOKUP_COLUMNS_1; i++) {
            lookup_vals_1[i] = lookup_values_1[i][vec_index];
        }

        qm31 denom0 = range_check_0->combine(lookup_vals_0, N_LOOKUP_COLUMNS_0);
        qm31 denom1 = range_check_1->combine(lookup_vals_1, N_LOOKUP_COLUMNS_1);

        qm31 num = add(denom0, denom1);
        denom[vec_index] = mul(denom0, denom1);

        numerator0[vec_index] = num.a.a;
        numerator1[vec_index] = num.a.b;
        numerator2[vec_index] = num.b.a;
        numerator3[vec_index] = num.b.b;
    }
}

// Round 17: range_check_19_c_3 * enabler + opcodes_0
template <int N_LOOKUP_COLUMNS_0, int N_LOOKUP_COLUMNS_1>
__global__ void generate_mul_opcode_interaction_trace_col_gen_kernel_second2last(
    RangeCheck_19_C *range_check_19_c,
    Opcodes *opcodes,
    m31 **lookup_values_0,
    m31 **lookup_values_1,
    unsigned trace_size,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned n_rows
) {
    unsigned vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        m31 lookup_vals_0[N_LOOKUP_COLUMNS_0];
        m31 lookup_vals_1[N_LOOKUP_COLUMNS_1];

        for (int i = 0; i < N_LOOKUP_COLUMNS_0; i++) {
            lookup_vals_0[i] = lookup_values_0[i][vec_index];
        }
        for (int i = 0; i < N_LOOKUP_COLUMNS_1; i++) {
            lookup_vals_1[i] = lookup_values_1[i][vec_index];
        }

        qm31 denom0 = range_check_19_c->combine(lookup_vals_0, N_LOOKUP_COLUMNS_0);
        qm31 denom1 = opcodes->combine(lookup_vals_1, N_LOOKUP_COLUMNS_1);

        // enabler = 1 if row < n_rows else 0
        m31 enabler = vec_index < n_rows ? m31{1} : m31{0};
        qm31 num = add(mul(denom0, qm31{cm31{enabler, m31{0}}, cm31{m31{0}, m31{0}}}), denom1);
        denom[vec_index] = mul(denom0, denom1);

        numerator0[vec_index] = num.a.a;
        numerator1[vec_index] = num.a.b;
        numerator2[vec_index] = num.b.a;
        numerator3[vec_index] = num.b.b;
    }
}

// Round 18 (Last): -enabler * opcodes_1
template <int N_LOOKUP_COLUMNS>
__global__ void generate_mul_opcode_interaction_trace_col_gen_kernel_last(
    Opcodes *opcodes,
    m31 **lookup_values,
    unsigned trace_size,
    qm31 *denom,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned n_rows
) {
    unsigned vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        m31 lookup_vals[N_LOOKUP_COLUMNS];

        for (int i = 0; i < N_LOOKUP_COLUMNS; i++) {
            lookup_vals[i] = lookup_values[i][vec_index];
        }

        qm31 denom0 = opcodes->combine(lookup_vals, N_LOOKUP_COLUMNS);
        denom[vec_index] = denom0;

        // enabler = 1 if row < n_rows else 0
        // numerator = -enabler = (P - 1) if row < n_rows else 0
        m31 neg_enabler = vec_index < n_rows ? m31{P - 1} : m31{0};

        numerator0[vec_index] = neg_enabler;
        numerator1[vec_index] = m31{0};
        numerator2[vec_index] = m31{0};
        numerator3[vec_index] = m31{0};
    }
}

__global__ void generate_mul_opcode_interaction_trace_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31 *denom_inv,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
) {
    unsigned vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 num = qm31 {
            cm31{numerator0[vec_index], numerator1[vec_index]},
            cm31{numerator2[vec_index], numerator3[vec_index]}
        };
        qm31 value = mul(num, denom_inv[vec_index]);

        // Add previous column's value at the same row (column-wise accumulation)
        // This matches the CPU LogUp implementation where each column accumulates
        // from the previous column's value at the same row position.
        qm31 prev_value;
        if (rep_index > 0) {
            prev_value.a.a = interaction_traces[(rep_index - 1) * 4 + 0][vec_index];
            prev_value.a.b = interaction_traces[(rep_index - 1) * 4 + 1][vec_index];
            prev_value.b.a = interaction_traces[(rep_index - 1) * 4 + 2][vec_index];
            prev_value.b.b = interaction_traces[(rep_index - 1) * 4 + 3][vec_index];
        } else {
            prev_value = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};
        }
        qm31 result = add(value, prev_value);

        interaction_traces[rep_index * 4 + 0][vec_index] = result.a.a;
        interaction_traces[rep_index * 4 + 1][vec_index] = result.a.b;
        interaction_traces[rep_index * 4 + 2][vec_index] = result.b.a;
        interaction_traces[rep_index * 4 + 3][vec_index] = result.b.b;
    }
}

__global__ void generate_mul_opcode_interaction_trace_cumsum_shift(
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_traces,
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
        sum0 = add(sum0, interaction_traces[idx0][i]);
        sum1 = add(sum1, interaction_traces[idx1][i]);
        sum2 = add(sum2, interaction_traces[idx2][i]);
        sum3 = add(sum3, interaction_traces[idx3][i]);
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

__global__ void generate_mul_opcode_interaction_trace_coord_prefix_sum(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interaction_traces[4 * last_index - 4][vec_index] = sub(interaction_traces[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interaction_traces[4 * last_index - 3][vec_index] = sub(interaction_traces[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interaction_traces[4 * last_index - 2][vec_index] = sub(interaction_traces[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interaction_traces[4 * last_index - 1][vec_index] = sub(interaction_traces[4 * last_index - 1][vec_index], cumsum_shift.b.b);
    }
}

void generate_mul_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,
    void *range_check_19,
    void *range_check_19_b,
    void *range_check_19_c,
    void *range_check_19_d,
    void *range_check_19_e,
    void *range_check_19_f,
    void *range_check_19_g,
    void *range_check_19_h,

    // Lookup data arrays
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
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
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_traces,
    unsigned *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    MemoryAddressToId *memory_address_to_id_lookup_elements = (MemoryAddressToId *)memory_address_to_id;
    MemoryIdToBig *memory_id_to_big_lookup_elements = (MemoryIdToBig *)memory_id_to_big;
    Opcodes *opcodes_lookup_elements = (Opcodes *)opcodes;
    VerifyInstruction *verify_instruction_lookup_elements = (VerifyInstruction *)verify_instruction;
    RangeCheck_19 *range_check_19_lookup_elements = (RangeCheck_19 *)range_check_19;
    RangeCheck_19_B *range_check_19_b_lookup_elements = (RangeCheck_19_B *)range_check_19_b;
    RangeCheck_19_C *range_check_19_c_lookup_elements = (RangeCheck_19_C *)range_check_19_c;
    RangeCheck_19_D *range_check_19_d_lookup_elements = (RangeCheck_19_D *)range_check_19_d;
    RangeCheck_19_E *range_check_19_e_lookup_elements = (RangeCheck_19_E *)range_check_19_e;
    RangeCheck_19_F *range_check_19_f_lookup_elements = (RangeCheck_19_F *)range_check_19_f;
    RangeCheck_19_G *range_check_19_g_lookup_elements = (RangeCheck_19_G *)range_check_19_g;
    RangeCheck_19_H *range_check_19_h_lookup_elements = (RangeCheck_19_H *)range_check_19_h;

    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes_lookup_elements = cuda_malloc<Opcodes>(1);
    VerifyInstruction *device_verify_instruction_lookup_elements = cuda_malloc<VerifyInstruction>(1);
    RangeCheck_19 *device_range_check_19_lookup_elements = cuda_malloc<RangeCheck_19>(1);
    RangeCheck_19_B *device_range_check_19_b_lookup_elements = cuda_malloc<RangeCheck_19_B>(1);
    RangeCheck_19_C *device_range_check_19_c_lookup_elements = cuda_malloc<RangeCheck_19_C>(1);
    RangeCheck_19_D *device_range_check_19_d_lookup_elements = cuda_malloc<RangeCheck_19_D>(1);
    RangeCheck_19_E *device_range_check_19_e_lookup_elements = cuda_malloc<RangeCheck_19_E>(1);
    RangeCheck_19_F *device_range_check_19_f_lookup_elements = cuda_malloc<RangeCheck_19_F>(1);
    RangeCheck_19_G *device_range_check_19_g_lookup_elements = cuda_malloc<RangeCheck_19_G>(1);
    RangeCheck_19_H *device_range_check_19_h_lookup_elements = cuda_malloc<RangeCheck_19_H>(1);

    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id_lookup_elements, 1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big_lookup_elements, 1);
    cuda_mem_copy_host_to_device<Opcodes>(opcodes_lookup_elements, device_opcodes_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyInstruction>(verify_instruction_lookup_elements, device_verify_instruction_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19>(range_check_19_lookup_elements, device_range_check_19_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19_B>(range_check_19_b_lookup_elements, device_range_check_19_b_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19_C>(range_check_19_c_lookup_elements, device_range_check_19_c_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19_D>(range_check_19_d_lookup_elements, device_range_check_19_d_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19_E>(range_check_19_e_lookup_elements, device_range_check_19_e_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19_F>(range_check_19_f_lookup_elements, device_range_check_19_f_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19_G>(range_check_19_g_lookup_elements, device_range_check_19_g_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_19_H>(range_check_19_h_lookup_elements, device_range_check_19_h_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>((m31 **)lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>((m31 **)lookup_opcodes_1, 3);
    m31 **device_lookup_range_check_19_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_0, 1);
    m31 **device_lookup_range_check_19_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_1, 1);
    m31 **device_lookup_range_check_19_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_2, 1);
    m31 **device_lookup_range_check_19_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_3, 1);
    m31 **device_lookup_range_check_19_b_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_0, 1);
    m31 **device_lookup_range_check_19_b_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_1, 1);
    m31 **device_lookup_range_check_19_b_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_2, 1);
    m31 **device_lookup_range_check_19_b_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_b_3, 1);
    m31 **device_lookup_range_check_19_c_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_0, 1);
    m31 **device_lookup_range_check_19_c_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_1, 1);
    m31 **device_lookup_range_check_19_c_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_2, 1);
    m31 **device_lookup_range_check_19_c_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_c_3, 1);
    m31 **device_lookup_range_check_19_d_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_d_0, 1);
    m31 **device_lookup_range_check_19_d_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_d_1, 1);
    m31 **device_lookup_range_check_19_d_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_d_2, 1);
    m31 **device_lookup_range_check_19_e_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_e_0, 1);
    m31 **device_lookup_range_check_19_e_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_e_1, 1);
    m31 **device_lookup_range_check_19_e_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_e_2, 1);
    m31 **device_lookup_range_check_19_f_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_f_0, 1);
    m31 **device_lookup_range_check_19_f_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_f_1, 1);
    m31 **device_lookup_range_check_19_f_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_f_2, 1);
    m31 **device_lookup_range_check_19_g_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_g_0, 1);
    m31 **device_lookup_range_check_19_g_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_g_1, 1);
    m31 **device_lookup_range_check_19_g_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_g_2, 1);
    m31 **device_lookup_range_check_19_h_0 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_0, 1);
    m31 **device_lookup_range_check_19_h_1 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_1, 1);
    m31 **device_lookup_range_check_19_h_2 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_2, 1);
    m31 **device_lookup_range_check_19_h_3 = clone_to_device<m31 *>((m31 **)lookup_range_check_19_h_3, 1);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>((m31 **)lookup_verify_instruction_0, 7);

    m31 **device_interaction_traces = clone_to_device<m31*>((m31 **)interaction_traces, 4 * MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);

    timer global_timer;
    global_timer.start("generate mul_opcode interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // Round 0: verify_instruction_0 + memory_address_to_id_0
    generate_mul_opcode_interaction_trace_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        0,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 1: memory_id_to_big_0 + memory_address_to_id_1
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_mem_id_big_addr<29, 2><<<num_blocks, block_dim>>>(
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        1,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 2: memory_id_to_big_1 + memory_address_to_id_2
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_mem_id_big_addr<29, 2><<<num_blocks, block_dim>>>(
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        2,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 3: memory_id_to_big_2 + range_check_19_h_0
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_mem_id_big_rc<29, 1><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_19_h_lookup_elements,
        device_lookup_memory_id_to_big_2,
        device_lookup_range_check_19_h_0,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        3,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 4: range_check_19_0 + range_check_19_b_0
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19, RangeCheck_19_B><<<num_blocks, block_dim>>>(
        device_range_check_19_lookup_elements,
        device_range_check_19_b_lookup_elements,
        device_lookup_range_check_19_0,
        device_lookup_range_check_19_b_0,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        4,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 5: range_check_19_c_0 + range_check_19_d_0
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_C, RangeCheck_19_D><<<num_blocks, block_dim>>>(
        device_range_check_19_c_lookup_elements,
        device_range_check_19_d_lookup_elements,
        device_lookup_range_check_19_c_0,
        device_lookup_range_check_19_d_0,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        5,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 6: range_check_19_e_0 + range_check_19_f_0
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_E, RangeCheck_19_F><<<num_blocks, block_dim>>>(
        device_range_check_19_e_lookup_elements,
        device_range_check_19_f_lookup_elements,
        device_lookup_range_check_19_e_0,
        device_lookup_range_check_19_f_0,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        6,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 7: range_check_19_g_0 + range_check_19_h_1
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_G, RangeCheck_19_H><<<num_blocks, block_dim>>>(
        device_range_check_19_g_lookup_elements,
        device_range_check_19_h_lookup_elements,
        device_lookup_range_check_19_g_0,
        device_lookup_range_check_19_h_1,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        7,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 8: range_check_19_1 + range_check_19_b_1
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19, RangeCheck_19_B><<<num_blocks, block_dim>>>(
        device_range_check_19_lookup_elements,
        device_range_check_19_b_lookup_elements,
        device_lookup_range_check_19_1,
        device_lookup_range_check_19_b_1,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        8,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 9: range_check_19_c_1 + range_check_19_d_1
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_C, RangeCheck_19_D><<<num_blocks, block_dim>>>(
        device_range_check_19_c_lookup_elements,
        device_range_check_19_d_lookup_elements,
        device_lookup_range_check_19_c_1,
        device_lookup_range_check_19_d_1,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        9,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 10: range_check_19_e_1 + range_check_19_f_1
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_E, RangeCheck_19_F><<<num_blocks, block_dim>>>(
        device_range_check_19_e_lookup_elements,
        device_range_check_19_f_lookup_elements,
        device_lookup_range_check_19_e_1,
        device_lookup_range_check_19_f_1,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        10,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 11: range_check_19_g_1 + range_check_19_h_2
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_G, RangeCheck_19_H><<<num_blocks, block_dim>>>(
        device_range_check_19_g_lookup_elements,
        device_range_check_19_h_lookup_elements,
        device_lookup_range_check_19_g_1,
        device_lookup_range_check_19_h_2,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        11,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 12: range_check_19_2 + range_check_19_b_2
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19, RangeCheck_19_B><<<num_blocks, block_dim>>>(
        device_range_check_19_lookup_elements,
        device_range_check_19_b_lookup_elements,
        device_lookup_range_check_19_2,
        device_lookup_range_check_19_b_2,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        12,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 13: range_check_19_c_2 + range_check_19_d_2
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_C, RangeCheck_19_D><<<num_blocks, block_dim>>>(
        device_range_check_19_c_lookup_elements,
        device_range_check_19_d_lookup_elements,
        device_lookup_range_check_19_c_2,
        device_lookup_range_check_19_d_2,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        13,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 14: range_check_19_e_2 + range_check_19_f_2
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_E, RangeCheck_19_F><<<num_blocks, block_dim>>>(
        device_range_check_19_e_lookup_elements,
        device_range_check_19_f_lookup_elements,
        device_lookup_range_check_19_e_2,
        device_lookup_range_check_19_f_2,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        14,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 15: range_check_19_g_2 + range_check_19_h_3
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19_G, RangeCheck_19_H><<<num_blocks, block_dim>>>(
        device_range_check_19_g_lookup_elements,
        device_range_check_19_h_lookup_elements,
        device_lookup_range_check_19_g_2,
        device_lookup_range_check_19_h_3,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        15,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 16: range_check_19_3 + range_check_19_b_3
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_rc_pair<1, 1, RangeCheck_19, RangeCheck_19_B><<<num_blocks, block_dim>>>(
        device_range_check_19_lookup_elements,
        device_range_check_19_b_lookup_elements,
        device_lookup_range_check_19_3,
        device_lookup_range_check_19_b_3,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        16,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 17: range_check_19_c_3 * enabler + opcodes_0
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_second2last<1, 3><<<num_blocks, block_dim>>>(
        device_range_check_19_c_lookup_elements,
        device_opcodes_lookup_elements,
        device_lookup_range_check_19_c_3,
        device_lookup_opcodes_0,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        17,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Round 18 (Last): -enabler * opcodes_1
    block_dim = trace_size < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_col_gen_kernel_last<3><<<num_blocks, block_dim>>>(
        device_opcodes_lookup_elements,
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
    generate_mul_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        18,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_mul_opcode_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        (m31 *)claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_mul_opcode_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        (m31 *)claimed_sum,
        MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum((m31 *)interaction_traces[4 * MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum((m31 *)interaction_traces[4 * MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum((m31 *)interaction_traces[4 * MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum((m31 *)interaction_traces[4 * MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate mul_opcode interaction trace");

    // Free device memory
    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_opcodes_lookup_elements);
    cuda_free_memory(device_verify_instruction_lookup_elements);
    cuda_free_memory(device_range_check_19_lookup_elements);
    cuda_free_memory(device_range_check_19_b_lookup_elements);
    cuda_free_memory(device_range_check_19_c_lookup_elements);
    cuda_free_memory(device_range_check_19_d_lookup_elements);
    cuda_free_memory(device_range_check_19_e_lookup_elements);
    cuda_free_memory(device_range_check_19_f_lookup_elements);
    cuda_free_memory(device_range_check_19_g_lookup_elements);
    cuda_free_memory(device_range_check_19_h_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
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
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}
