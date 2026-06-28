#include "relations.cuh"

#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include <stdint.h>

#include "gen_blake_g_trace.cuh"
#define N_TRACE_COLUMNS 53

#define GEN_TRACE_BLAKE_G_THREAD_COUNT_MAX 256

#define N_INTERACTION_TRACE_COLUMNS 9

__launch_bounds__(256, 2)
__global__ void generate_blake_g_trace_kernel(
    m31 **traces,
    m31 **lookup_blake_g_0,

    m31 **lookup_verify_bitwise_xor_12_0,
    m31 **lookup_verify_bitwise_xor_12_1,
    m31 **lookup_verify_bitwise_xor_4_0,
    m31 **lookup_verify_bitwise_xor_4_1,
    m31 **lookup_verify_bitwise_xor_7_0,
    m31 **lookup_verify_bitwise_xor_7_1,
    m31 **lookup_verify_bitwise_xor_8_0,
    m31 **lookup_verify_bitwise_xor_8_1,
    m31 **lookup_verify_bitwise_xor_8_2,
    m31 **lookup_verify_bitwise_xor_8_3,
    m31 **lookup_verify_bitwise_xor_8_4,
    m31 **lookup_verify_bitwise_xor_8_5,
    m31 **lookup_verify_bitwise_xor_8_6,
    m31 **lookup_verify_bitwise_xor_8_7,
    m31 **lookup_verify_bitwise_xor_9_0,
    m31 **lookup_verify_bitwise_xor_9_1,

    m31 **sub_component_inputs_verify_bitwise_xor_8,
    m31 **sub_component_inputs_verify_bitwise_xor_12,
    m31 **sub_component_inputs_verify_bitwise_xor_4,
    m31 **sub_component_inputs_verify_bitwise_xor_7,
    m31 **sub_component_inputs_verify_bitwise_xor_9,

    uint32_t **blake_g_input,

    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const m31 M31_128 = {128};
    const m31 M31_16 = {16};
    const m31 M31_256 = {256};
    const m31 M31_4096 = {4096};
    const m31 M31_512 = {512};

    const uint16_t UInt16_12 = 12;
    const uint16_t UInt16_7 = 7;
    const uint16_t UInt16_8 = 8;
    const uint32_t UInt32_0 = 0;

    if (row < trace_size) {
        m31 input_limb_0_col0 = low_as_m31(blake_g_input[0][row]);
        traces[0][row] = input_limb_0_col0;
        m31 input_limb_1_col1 = high_as_m31(blake_g_input[0][row]);
        traces[1][row] = input_limb_1_col1;
        m31 input_limb_2_col2 = low_as_m31(blake_g_input[1][row]);
        traces[2][row] = input_limb_2_col2;
        m31 input_limb_3_col3 = high_as_m31(blake_g_input[1][row]);
        traces[3][row] = input_limb_3_col3;
        m31 input_limb_4_col4 = low_as_m31(blake_g_input[2][row]);
        traces[4][row] = input_limb_4_col4;
        m31 input_limb_5_col5 = high_as_m31(blake_g_input[2][row]);
        traces[5][row] = input_limb_5_col5;
        m31 input_limb_6_col6 = low_as_m31(blake_g_input[3][row]);
        traces[6][row] = input_limb_6_col6;
        m31 input_limb_7_col7 = high_as_m31(blake_g_input[3][row]);
        traces[7][row] = input_limb_7_col7;
        m31 input_limb_8_col8 = low_as_m31(blake_g_input[4][row]);
        traces[8][row] = input_limb_8_col8;
        m31 input_limb_9_col9 = high_as_m31(blake_g_input[4][row]);
        traces[9][row] = input_limb_9_col9;
        m31 input_limb_10_col10 = low_as_m31(blake_g_input[5][row]);
        traces[10][row] = input_limb_10_col10;
        m31 input_limb_11_col11 = high_as_m31(blake_g_input[5][row]);
        traces[11][row] = input_limb_11_col11;
        Enabler enabler_col = Enabler(trace_size);

        // Triple Sum 32.

        uint32_t triple_sum32_res_tmp_f72c8_0 =
            (((blake_g_input[0][row]) + (blake_g_input[1][row])) + (blake_g_input[4][row]));
        m31 triple_sum32_res_limb_0_col12 = low_as_m31(triple_sum32_res_tmp_f72c8_0);
        traces[12][row] = triple_sum32_res_limb_0_col12;
        m31 triple_sum32_res_limb_1_col13 = high_as_m31(triple_sum32_res_tmp_f72c8_0);
        traces[13][row] = triple_sum32_res_limb_1_col13;
        uint32_t triple_sum_32_output_tmp_f72c8_3 = triple_sum32_res_tmp_f72c8_0;

        // Xor Rot 32 R 16.

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_4 = ((triple_sum_32_output_tmp_f72c8_3 & 0xFFFF) >> (UInt16_8));
        m31 ms_8_bits_col14 = m31 {ms_8_bits_tmp_f72c8_4};
        traces[14][row] = ms_8_bits_col14;
        m31 split_16_low_part_size_8_output_tmp_f72c8_5_0 = sub((triple_sum32_res_limb_0_col12), mul((ms_8_bits_col14), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_5_1 = ms_8_bits_col14;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_6 = ((triple_sum_32_output_tmp_f72c8_3 >> 16) >> (UInt16_8));
        m31 ms_8_bits_col15 = m31 {ms_8_bits_tmp_f72c8_6};
        traces[15][row] = ms_8_bits_col15;
        m31 split_16_low_part_size_8_output_tmp_f72c8_7_0 = sub((triple_sum32_res_limb_1_col13), mul((ms_8_bits_col15), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_7_1 = ms_8_bits_col15;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_8 = ((blake_g_input[3][row] & 0xFFFF) >> (UInt16_8));
        m31 ms_8_bits_col16 = m31 {ms_8_bits_tmp_f72c8_8};
        traces[16][row] = ms_8_bits_col16;
        m31 split_16_low_part_size_8_output_tmp_f72c8_9_0 = sub((input_limb_6_col6), mul((ms_8_bits_col16), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_9_1 = ms_8_bits_col16;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_10 = ((blake_g_input[3][row] >> 16) >> (UInt16_8));
        m31 ms_8_bits_col17 = m31 {ms_8_bits_tmp_f72c8_10};
        traces[17][row] = ms_8_bits_col17;
        m31 split_16_low_part_size_8_output_tmp_f72c8_11_0 = sub((input_limb_7_col7), mul((ms_8_bits_col17), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_11_1 = ms_8_bits_col17;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_12 = split_16_low_part_size_8_output_tmp_f72c8_5_0 ^ split_16_low_part_size_8_output_tmp_f72c8_9_0;
        m31 xor_col18 = m31 {xor_tmp_f72c8_12};
        traces[18][row] = xor_col18;
        sub_component_inputs_verify_bitwise_xor_8[0 * 3 + 0][row] = split_16_low_part_size_8_output_tmp_f72c8_5_0;
        sub_component_inputs_verify_bitwise_xor_8[0 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_f72c8_9_0;
        sub_component_inputs_verify_bitwise_xor_8[0 * 3 + 2][row] = xor_col18;
        lookup_verify_bitwise_xor_8_0[0][row] = split_16_low_part_size_8_output_tmp_f72c8_5_0;
        lookup_verify_bitwise_xor_8_0[1][row] = split_16_low_part_size_8_output_tmp_f72c8_9_0;
        lookup_verify_bitwise_xor_8_0[2][row] = xor_col18;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_14 = (ms_8_bits_col14 ^ ms_8_bits_col16);
        m31 xor_col19 = {xor_tmp_f72c8_14};
        traces[19][row] = xor_col19;
        sub_component_inputs_verify_bitwise_xor_8[1 * 3][row] = ms_8_bits_col14;
        sub_component_inputs_verify_bitwise_xor_8[1 * 3 + 1][row] = ms_8_bits_col16;
        sub_component_inputs_verify_bitwise_xor_8[1 * 3 + 2][row] = xor_col19;
        lookup_verify_bitwise_xor_8_1[0][row] = ms_8_bits_col14;
        lookup_verify_bitwise_xor_8_1[1][row] = ms_8_bits_col16;
        lookup_verify_bitwise_xor_8_1[2][row] = xor_col19;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_16 = split_16_low_part_size_8_output_tmp_f72c8_7_0 ^ split_16_low_part_size_8_output_tmp_f72c8_11_0;
        m31 xor_col20 = m31 {xor_tmp_f72c8_16};
        traces[20][row] = xor_col20;
        sub_component_inputs_verify_bitwise_xor_8[4 * 3 ][row] = split_16_low_part_size_8_output_tmp_f72c8_7_0;
        sub_component_inputs_verify_bitwise_xor_8[4 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_f72c8_11_0;
        sub_component_inputs_verify_bitwise_xor_8[4 * 3 + 2][row] = xor_col20;
        lookup_verify_bitwise_xor_8_2[0][row] = split_16_low_part_size_8_output_tmp_f72c8_7_0;
        lookup_verify_bitwise_xor_8_2[1][row] = split_16_low_part_size_8_output_tmp_f72c8_11_0;
        lookup_verify_bitwise_xor_8_2[2][row] = xor_col20;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_18 = ms_8_bits_col15 ^ ms_8_bits_col17;
        m31 xor_col21 = m31 {xor_tmp_f72c8_18};
        traces[21][row] = xor_col21;
        sub_component_inputs_verify_bitwise_xor_8[5 * 3 + 0][row] = ms_8_bits_col15;
        sub_component_inputs_verify_bitwise_xor_8[5 * 3 + 1][row] = ms_8_bits_col17;
        sub_component_inputs_verify_bitwise_xor_8[5 * 3 + 2][row] = xor_col21;

        lookup_verify_bitwise_xor_8_3[0][row] = ms_8_bits_col15;
        lookup_verify_bitwise_xor_8_3[1][row] = ms_8_bits_col17;
        lookup_verify_bitwise_xor_8_3[2][row] = xor_col21;
        uint32_t xor_rot_16_output_tmp_f72c8_20 = (add((xor_col20), mul((xor_col21), (M31_256)))) + (add((xor_col18), mul((xor_col19), (M31_256))) << 16);
        uint32_t xor_rot_32_r_16_output_tmp_f72c8_21 = xor_rot_16_output_tmp_f72c8_20;

        // Triple Sum 32.

        uint32_t triple_sum32_res_tmp_f72c8_22 = (((blake_g_input[2][row]) + (xor_rot_32_r_16_output_tmp_f72c8_21)) + (UInt32_0));
        m31 triple_sum32_res_limb_0_col22 = low_as_m31(triple_sum32_res_tmp_f72c8_22);
        traces[22][row] = triple_sum32_res_limb_0_col22;
        m31  triple_sum32_res_limb_1_col23 = high_as_m31(triple_sum32_res_tmp_f72c8_22);
        traces[23][row] = triple_sum32_res_limb_1_col23;
        uint32_t triple_sum_32_output_tmp_f72c8_25 = triple_sum32_res_tmp_f72c8_22;

        // Xor Rot 32 R 12.

        // Split 16 Low Part Size 12.

        uint16_t ms_4_bits_tmp_f72c8_26 = ((blake_g_input[1][row] & 0xFFFF) >> (UInt16_12));
        m31 ms_4_bits_col24 = m31 {ms_4_bits_tmp_f72c8_26};
        traces[24][row] = ms_4_bits_col24;
        m31 split_16_low_part_size_12_output_tmp_f72c8_27_0 = sub((input_limb_2_col2), mul((ms_4_bits_col24), (M31_4096)));
        m31 split_16_low_part_size_12_output_tmp_f72c8_27_1 = ms_4_bits_col24;

        // Split 16 Low Part Size 12.

        uint16_t ms_4_bits_tmp_f72c8_28 = ((blake_g_input[1][row] >> 16) >> (UInt16_12));
        m31 ms_4_bits_col25 = m31 {ms_4_bits_tmp_f72c8_28};
        traces[25][row] = ms_4_bits_col25;
        m31 split_16_low_part_size_12_output_tmp_f72c8_29_0 = sub((input_limb_3_col3), mul((ms_4_bits_col25), (M31_4096)));
        m31 split_16_low_part_size_12_output_tmp_f72c8_29_1 = ms_4_bits_col25;

        // Split 16 Low Part Size 12.

        uint16_t ms_4_bits_tmp_f72c8_30 = ((triple_sum_32_output_tmp_f72c8_25 & 0xFFFF) >> (UInt16_12));
        m31 ms_4_bits_col26 = m31 {ms_4_bits_tmp_f72c8_30};
        traces[26][row] = ms_4_bits_col26;
        m31 split_16_low_part_size_12_output_tmp_f72c8_31_0 = sub((triple_sum32_res_limb_0_col22), mul((ms_4_bits_col26), (M31_4096)));
        m31 split_16_low_part_size_12_output_tmp_f72c8_31_1 = ms_4_bits_col26;

        // Split 16 Low Part Size 12.

        uint16_t ms_4_bits_tmp_f72c8_32 = ((triple_sum_32_output_tmp_f72c8_25 >> 16) >> (UInt16_12));
        m31 ms_4_bits_col27 = m31 {ms_4_bits_tmp_f72c8_32};
        traces[27][row] = ms_4_bits_col27;
        m31 split_16_low_part_size_12_output_tmp_f72c8_33_0 = sub((triple_sum32_res_limb_1_col23), mul((ms_4_bits_col27), (M31_4096)));
        m31 split_16_low_part_size_12_output_tmp_f72c8_33_1 = ms_4_bits_col27;

        // Bitwise Xor Num Bits 12.

        uint16_t xor_tmp_f72c8_34 = split_16_low_part_size_12_output_tmp_f72c8_27_0 ^ split_16_low_part_size_12_output_tmp_f72c8_31_0;
        m31 xor_col28 = m31 {xor_tmp_f72c8_34};
        traces[28][row] = xor_col28;
        sub_component_inputs_verify_bitwise_xor_12[0 * 3 + 0][row] = split_16_low_part_size_12_output_tmp_f72c8_27_0;
        sub_component_inputs_verify_bitwise_xor_12[0 * 3 + 1][row] = split_16_low_part_size_12_output_tmp_f72c8_31_0;
        sub_component_inputs_verify_bitwise_xor_12[0 * 3 + 2][row] = xor_col28;
        lookup_verify_bitwise_xor_12_0[0][row] = split_16_low_part_size_12_output_tmp_f72c8_27_0;
        lookup_verify_bitwise_xor_12_0[1][row] = split_16_low_part_size_12_output_tmp_f72c8_31_0;
        lookup_verify_bitwise_xor_12_0[2][row] = xor_col28;

        // Bitwise Xor Num Bits 4.

        uint16_t xor_tmp_f72c8_36 = ms_4_bits_col24 ^ ms_4_bits_col26;
        m31 xor_col29 = m31 {xor_tmp_f72c8_36};
        traces[29][row] = xor_col29;
        sub_component_inputs_verify_bitwise_xor_4[0 * 3 + 0][row] = ms_4_bits_col24;
        sub_component_inputs_verify_bitwise_xor_4[0 * 3 + 1][row] = ms_4_bits_col26;
        sub_component_inputs_verify_bitwise_xor_4[0 * 3 + 2][row] = xor_col29;
        lookup_verify_bitwise_xor_4_0[0][row] = ms_4_bits_col24;
        lookup_verify_bitwise_xor_4_0[1][row] = ms_4_bits_col26;
        lookup_verify_bitwise_xor_4_0[2][row] = xor_col29;

        // Bitwise Xor Num Bits 12.

        uint16_t xor_tmp_f72c8_38 = split_16_low_part_size_12_output_tmp_f72c8_29_0 ^ split_16_low_part_size_12_output_tmp_f72c8_33_0;
        m31 xor_col30 = m31 {xor_tmp_f72c8_38};
        traces[30][row] = xor_col30;
        sub_component_inputs_verify_bitwise_xor_12[1 * 3 + 0][row] = split_16_low_part_size_12_output_tmp_f72c8_29_0;
        sub_component_inputs_verify_bitwise_xor_12[1 * 3 + 1][row] = split_16_low_part_size_12_output_tmp_f72c8_33_0;
        sub_component_inputs_verify_bitwise_xor_12[1 * 3 + 2][row] = xor_col30;
        lookup_verify_bitwise_xor_12_1[0][row] = split_16_low_part_size_12_output_tmp_f72c8_29_0;
        lookup_verify_bitwise_xor_12_1[1][row] = split_16_low_part_size_12_output_tmp_f72c8_33_0;
        lookup_verify_bitwise_xor_12_1[2][row] = xor_col30;

        // Bitwise Xor Num Bits 4.

        uint16_t xor_tmp_f72c8_40 = ms_4_bits_col25 ^ ms_4_bits_col27;
        m31 xor_col31 = m31 {xor_tmp_f72c8_40};
        traces[31][row] = xor_col31;
        sub_component_inputs_verify_bitwise_xor_4[1 * 3 + 0][row] = ms_4_bits_col25;
        sub_component_inputs_verify_bitwise_xor_4[1 * 3 + 1][row] = ms_4_bits_col27;
        sub_component_inputs_verify_bitwise_xor_4[1 * 3 + 2][row] = xor_col31;
        lookup_verify_bitwise_xor_4_1[0][row] = ms_4_bits_col25;
        lookup_verify_bitwise_xor_4_1[1][row] = ms_4_bits_col27;
        lookup_verify_bitwise_xor_4_1[2][row] = xor_col31;
        uint32_t xor_rot_12_output_tmp_f72c8_42 = add((xor_col29), mul((xor_col30), (M31_16))) + (add((xor_col31), mul((xor_col28), (M31_16))) << 16);
        uint32_t xor_rot_32_r_12_output_tmp_f72c8_43 = xor_rot_12_output_tmp_f72c8_42;

        // Triple Sum 32.

        uint32_t triple_sum32_res_tmp_f72c8_44 = (((triple_sum_32_output_tmp_f72c8_3) + (xor_rot_32_r_12_output_tmp_f72c8_43)) + (blake_g_input[5][row]));
        m31 triple_sum32_res_limb_0_col32 = low_as_m31(triple_sum32_res_tmp_f72c8_44);
        traces[32][row] = triple_sum32_res_limb_0_col32;
        m31 triple_sum32_res_limb_1_col33 = high_as_m31(triple_sum32_res_tmp_f72c8_44);
        traces[33][row] = triple_sum32_res_limb_1_col33;
        uint32_t triple_sum_32_output_tmp_f72c8_47 = triple_sum32_res_tmp_f72c8_44;

        // Xor Rot 32 R 8.

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_48 =
            ((triple_sum_32_output_tmp_f72c8_47 & 0xFFFF) >> (UInt16_8));
        m31 ms_8_bits_col34 = m31 {ms_8_bits_tmp_f72c8_48};
        traces[34][row] = ms_8_bits_col34;
        m31 split_16_low_part_size_8_output_tmp_f72c8_49_0 = sub((triple_sum32_res_limb_0_col32), mul((ms_8_bits_col34), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_49_1 = ms_8_bits_col34;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_50 =
            ((triple_sum_32_output_tmp_f72c8_47 >> 16) >> (UInt16_8));
        m31 ms_8_bits_col35 = m31 {ms_8_bits_tmp_f72c8_50};
        traces[35][row] = ms_8_bits_col35;
        m31 split_16_low_part_size_8_output_tmp_f72c8_51_0 = sub((triple_sum32_res_limb_1_col33), mul((ms_8_bits_col35), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_51_1 = ms_8_bits_col35;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_52 = ((xor_rot_32_r_16_output_tmp_f72c8_21 & 0xFFFF) >> (UInt16_8));
        m31 ms_8_bits_col36 = m31 {ms_8_bits_tmp_f72c8_52};
        traces[36][row] = ms_8_bits_col36;
        m31 split_16_low_part_size_8_output_tmp_f72c8_53_0 = sub(low_as_m31(xor_rot_32_r_16_output_tmp_f72c8_21), mul((ms_8_bits_col36), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_53_1 = ms_8_bits_col36;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_f72c8_54 = ((xor_rot_32_r_16_output_tmp_f72c8_21 >> 16) >> (UInt16_8));
        m31 ms_8_bits_col37 = m31 {ms_8_bits_tmp_f72c8_54};
        traces[37][row] = ms_8_bits_col37;
        m31 split_16_low_part_size_8_output_tmp_f72c8_55_0 = sub(high_as_m31(xor_rot_32_r_16_output_tmp_f72c8_21), mul((ms_8_bits_col37), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_f72c8_55_1 = ms_8_bits_col37;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_56 = split_16_low_part_size_8_output_tmp_f72c8_49_0 ^ split_16_low_part_size_8_output_tmp_f72c8_53_0;
        m31 xor_col38 = m31 {xor_tmp_f72c8_56};
        traces[38][row] = xor_col38;
        sub_component_inputs_verify_bitwise_xor_8[2 * 3 + 0][row] = split_16_low_part_size_8_output_tmp_f72c8_49_0;
        sub_component_inputs_verify_bitwise_xor_8[2 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_f72c8_53_0;
        sub_component_inputs_verify_bitwise_xor_8[2 * 3 + 2][row] = xor_col38;
        lookup_verify_bitwise_xor_8_4[0][row] = split_16_low_part_size_8_output_tmp_f72c8_49_0;
        lookup_verify_bitwise_xor_8_4[1][row] = split_16_low_part_size_8_output_tmp_f72c8_53_0;
        lookup_verify_bitwise_xor_8_4[2][row] = xor_col38;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_58 = ms_8_bits_col34 ^ ms_8_bits_col36;
        m31 xor_col39 = m31 {xor_tmp_f72c8_58};
        traces[39][row] = xor_col39;
        sub_component_inputs_verify_bitwise_xor_8[3 * 3 + 0][row] = ms_8_bits_col34;
        sub_component_inputs_verify_bitwise_xor_8[3 * 3 + 1][row] = ms_8_bits_col36;
        sub_component_inputs_verify_bitwise_xor_8[3 * 3 + 2][row] = xor_col39;
        lookup_verify_bitwise_xor_8_5[0][row] = ms_8_bits_col34;
        lookup_verify_bitwise_xor_8_5[1][row] = ms_8_bits_col36;
        lookup_verify_bitwise_xor_8_5[2][row] = xor_col39;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_60 = split_16_low_part_size_8_output_tmp_f72c8_51_0 ^ split_16_low_part_size_8_output_tmp_f72c8_55_0;
        m31 xor_col40 = m31 {xor_tmp_f72c8_60};
        traces[40][row] = xor_col40;
        sub_component_inputs_verify_bitwise_xor_8[6 * 3 + 0][row] = split_16_low_part_size_8_output_tmp_f72c8_51_0;
        sub_component_inputs_verify_bitwise_xor_8[6 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_f72c8_55_0;
        sub_component_inputs_verify_bitwise_xor_8[6 * 3 + 2][row] = xor_col40;
        lookup_verify_bitwise_xor_8_6[0][row] = split_16_low_part_size_8_output_tmp_f72c8_51_0;
        lookup_verify_bitwise_xor_8_6[1][row] = split_16_low_part_size_8_output_tmp_f72c8_55_0;
        lookup_verify_bitwise_xor_8_6[2][row] = xor_col40;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_f72c8_62 = ms_8_bits_col35 ^ ms_8_bits_col37;
        m31 xor_col41 = m31 {xor_tmp_f72c8_62};
        traces[41][row] = xor_col41;
        sub_component_inputs_verify_bitwise_xor_8[7 * 3 + 0][row] = ms_8_bits_col35;
        sub_component_inputs_verify_bitwise_xor_8[7 * 3 + 1][row] = ms_8_bits_col37;
        sub_component_inputs_verify_bitwise_xor_8[7 * 3 + 2][row] = xor_col41;
        lookup_verify_bitwise_xor_8_7[0][row] = ms_8_bits_col35;
        lookup_verify_bitwise_xor_8_7[1][row] = ms_8_bits_col37;
        lookup_verify_bitwise_xor_8_7[2][row] = xor_col41;
        uint32_t xor_rot_8_output_tmp_f72c8_64 = add((xor_col39), mul((xor_col40), (M31_256))) + (add((xor_col41), mul((xor_col38), (M31_256))) << 16);
        uint32_t xor_rot_32_r_8_output_tmp_f72c8_65 = xor_rot_8_output_tmp_f72c8_64;

        // Triple Sum 32.

        uint32_t triple_sum32_res_tmp_f72c8_66 = (((triple_sum_32_output_tmp_f72c8_25) + (xor_rot_32_r_8_output_tmp_f72c8_65)) + (UInt32_0));
        m31 triple_sum32_res_limb_0_col42 = low_as_m31(triple_sum32_res_tmp_f72c8_66);
        traces[42][row] = triple_sum32_res_limb_0_col42;
        m31 triple_sum32_res_limb_1_col43 = high_as_m31(triple_sum32_res_tmp_f72c8_66);
        traces[43][row] = triple_sum32_res_limb_1_col43;
        uint32_t triple_sum_32_output_tmp_f72c8_69 = triple_sum32_res_tmp_f72c8_66;

        // Xor Rot 32 R 7.

        // Split 16 Low Part Size 7.

        uint16_t ms_9_bits_tmp_f72c8_70 = ((xor_rot_32_r_12_output_tmp_f72c8_43 & 0xFFFF) >> (UInt16_7));
        m31 ms_9_bits_col44 = m31 {ms_9_bits_tmp_f72c8_70};
        traces[44][row] = ms_9_bits_col44;
        m31 split_16_low_part_size_7_output_tmp_f72c8_71_0 = sub(low_as_m31(xor_rot_32_r_12_output_tmp_f72c8_43), mul((ms_9_bits_col44), (M31_128)));

        // Split 16 Low Part Size 7.

        uint16_t ms_9_bits_tmp_f72c8_72 = ((xor_rot_32_r_12_output_tmp_f72c8_43 >> 16) >> (UInt16_7));
        m31 ms_9_bits_col45 = m31 {ms_9_bits_tmp_f72c8_72};
        traces[45][row] = ms_9_bits_col45;
        m31 split_16_low_part_size_7_output_tmp_f72c8_73_0 = sub(high_as_m31(xor_rot_32_r_12_output_tmp_f72c8_43), mul((ms_9_bits_col45), (M31_128)));
        m31 split_16_low_part_size_7_output_tmp_f72c8_73_1 = ms_9_bits_col45;

        // Split 16 Low Part Size 7.

        uint16_t ms_9_bits_tmp_f72c8_74 = ((triple_sum_32_output_tmp_f72c8_69 & 0xFFFF) >> (UInt16_7));
        m31 ms_9_bits_col46 = m31 {ms_9_bits_tmp_f72c8_74};
        traces[46][row] = ms_9_bits_col46;
        m31 split_16_low_part_size_7_output_tmp_f72c8_75_0 = sub((triple_sum32_res_limb_0_col42), mul((ms_9_bits_col46), (M31_128)));
        m31 split_16_low_part_size_7_output_tmp_f72c8_75_1 = ms_9_bits_col46;

        // Split 16 Low Part Size 7.

        uint16_t ms_9_bits_tmp_f72c8_76 = ((triple_sum_32_output_tmp_f72c8_69 >> 16) >> (UInt16_7));
        m31 ms_9_bits_col47 = m31 {ms_9_bits_tmp_f72c8_76};
        traces[47][row] = ms_9_bits_col47;
        m31 split_16_low_part_size_7_output_tmp_f72c8_77_0 = sub((triple_sum32_res_limb_1_col43), mul((ms_9_bits_col47), (M31_128)));
        m31 split_16_low_part_size_7_output_tmp_f72c8_77_1 = ms_9_bits_col47;

        // Bitwise Xor Num Bits 7.

        uint16_t xor_tmp_f72c8_78 = split_16_low_part_size_7_output_tmp_f72c8_71_0 ^ split_16_low_part_size_7_output_tmp_f72c8_75_0;
        m31 xor_col48 = m31 {xor_tmp_f72c8_78};
        traces[48][row] = xor_col48;
        sub_component_inputs_verify_bitwise_xor_7[0][row] = split_16_low_part_size_7_output_tmp_f72c8_71_0;
        sub_component_inputs_verify_bitwise_xor_7[1][row] = split_16_low_part_size_7_output_tmp_f72c8_75_0;
        sub_component_inputs_verify_bitwise_xor_7[2][row] = xor_col48;
        lookup_verify_bitwise_xor_7_0[0][row] = split_16_low_part_size_7_output_tmp_f72c8_71_0;
        lookup_verify_bitwise_xor_7_0[1][row] = split_16_low_part_size_7_output_tmp_f72c8_75_0;
        lookup_verify_bitwise_xor_7_0[2][row] = xor_col48;

        // Bitwise Xor Num Bits 9.

        uint16_t xor_tmp_f72c8_80 = ms_9_bits_col44 ^ ms_9_bits_col46;
        m31 xor_col49 = m31 {xor_tmp_f72c8_80};
        traces[49][row] = xor_col49;
        sub_component_inputs_verify_bitwise_xor_9[0][row] = ms_9_bits_col44;
        sub_component_inputs_verify_bitwise_xor_9[1][row] = ms_9_bits_col46;
        sub_component_inputs_verify_bitwise_xor_9[2][row] = xor_col49;
        lookup_verify_bitwise_xor_9_0[0][row] = ms_9_bits_col44;
        lookup_verify_bitwise_xor_9_0[1][row] = ms_9_bits_col46;
        lookup_verify_bitwise_xor_9_0[2][row] = xor_col49;

        // Bitwise Xor Num Bits 7.

        uint16_t xor_tmp_f72c8_82 = split_16_low_part_size_7_output_tmp_f72c8_73_0 ^ split_16_low_part_size_7_output_tmp_f72c8_77_0;
        m31 xor_col50 = m31 {xor_tmp_f72c8_82};
        traces[50][row] = xor_col50;
        sub_component_inputs_verify_bitwise_xor_7[1 * 3 + 0][row] = split_16_low_part_size_7_output_tmp_f72c8_73_0;
        sub_component_inputs_verify_bitwise_xor_7[1 * 3 + 1][row] = split_16_low_part_size_7_output_tmp_f72c8_77_0;
        sub_component_inputs_verify_bitwise_xor_7[1 * 3 + 2][row] = xor_col50;
        lookup_verify_bitwise_xor_7_1[0][row] = split_16_low_part_size_7_output_tmp_f72c8_73_0;
        lookup_verify_bitwise_xor_7_1[1][row] = split_16_low_part_size_7_output_tmp_f72c8_77_0;
        lookup_verify_bitwise_xor_7_1[2][row] = xor_col50;

        // Bitwise Xor Num Bits 9.

        uint16_t xor_tmp_f72c8_84 = ms_9_bits_col45 ^ ms_9_bits_col47;
        m31 xor_col51 = m31 {xor_tmp_f72c8_84};
        traces[51][row] = xor_col51;
        sub_component_inputs_verify_bitwise_xor_9[1 * 3 + 0][row] = ms_9_bits_col45;
        sub_component_inputs_verify_bitwise_xor_9[1 * 3 + 1][row] = ms_9_bits_col47;
        sub_component_inputs_verify_bitwise_xor_9[1 * 3 + 2][row] = xor_col51;
        lookup_verify_bitwise_xor_9_1[0][row] = ms_9_bits_col45;
        lookup_verify_bitwise_xor_9_1[1][row] = ms_9_bits_col47;
        lookup_verify_bitwise_xor_9_1[2][row] = xor_col51;
        uint32_t xor_rot_7_output_tmp_f72c8_86 =  add((xor_col49), mul((xor_col50), (M31_512))) + (add((xor_col51), mul((xor_col48), (M31_512))) << 16);
        uint32_t xor_rot_32_r_7_output_tmp_f72c8_87 = xor_rot_7_output_tmp_f72c8_86;

        lookup_blake_g_0[0][row] = input_limb_0_col0;
        lookup_blake_g_0[1][row] = input_limb_1_col1;
        lookup_blake_g_0[2][row] = input_limb_2_col2;
        lookup_blake_g_0[3][row] = input_limb_3_col3;
        lookup_blake_g_0[4][row] = input_limb_4_col4;
        lookup_blake_g_0[5][row] = input_limb_5_col5;
        lookup_blake_g_0[6][row] = input_limb_6_col6;
        lookup_blake_g_0[7][row] = input_limb_7_col7;
        lookup_blake_g_0[8][row] = input_limb_8_col8;
        lookup_blake_g_0[9][row] = input_limb_9_col9;
        lookup_blake_g_0[10][row] = input_limb_10_col10;
        lookup_blake_g_0[11][row] = input_limb_11_col11;
        lookup_blake_g_0[12][row] = triple_sum32_res_limb_0_col32;
        lookup_blake_g_0[13][row] = triple_sum32_res_limb_1_col33;
        lookup_blake_g_0[14][row] = low_as_m31(xor_rot_32_r_7_output_tmp_f72c8_87);
        lookup_blake_g_0[15][row] = high_as_m31(xor_rot_32_r_7_output_tmp_f72c8_87);
        lookup_blake_g_0[16][row] = triple_sum32_res_limb_0_col42;
        lookup_blake_g_0[17][row] = triple_sum32_res_limb_1_col43;
        lookup_blake_g_0[18][row] = low_as_m31(xor_rot_32_r_8_output_tmp_f72c8_65);
        lookup_blake_g_0[19][row] = high_as_m31(xor_rot_32_r_8_output_tmp_f72c8_65);

        traces[52][row] = 1;
    }
}


void generate_blake_g_traces(
    m31 **traces,
    m31 **lookup_blake_g_0,
    m31 **lookup_verify_bitwise_xor_12_0,
    m31 **lookup_verify_bitwise_xor_12_1,
    m31 **lookup_verify_bitwise_xor_4_0 ,
    m31 **lookup_verify_bitwise_xor_4_1 ,
    m31 **lookup_verify_bitwise_xor_7_0 ,
    m31 **lookup_verify_bitwise_xor_7_1 ,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_4 ,
    m31 **lookup_verify_bitwise_xor_8_5 ,
    m31 **lookup_verify_bitwise_xor_8_6 ,
    m31 **lookup_verify_bitwise_xor_8_7 ,
    m31 **lookup_verify_bitwise_xor_9_0 ,
    m31 **lookup_verify_bitwise_xor_9_1 ,

    m31 **sub_component_inputs_verify_bitwise_xor_8,
    m31 **sub_component_inputs_verify_bitwise_xor_12,
    m31 **sub_component_inputs_verify_bitwise_xor_4,
    m31 **sub_component_inputs_verify_bitwise_xor_7,
    m31 **sub_component_inputs_verify_bitwise_xor_9,

    uint32_t **blake_g_input,

    unsigned trace_log_size
) {
    m31 **device_traces = clone_to_device<m31*>(traces, N_TRACE_COLUMNS);
    m31 **device_lookup_blake_g_0 = clone_to_device<m31 *>(lookup_blake_g_0, 20);
    m31 **device_lookup_verify_bitwise_xor_12_0 = clone_to_device<m31 *>(lookup_verify_bitwise_xor_12_0, 3);
    m31 **device_lookup_verify_bitwise_xor_12_1 = clone_to_device<m31 *>(lookup_verify_bitwise_xor_12_1, 3);
    m31 **device_lookup_verify_bitwise_xor_4_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_4_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_4_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_4_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_7_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_7_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_7_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_7_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_2  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_2 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_3  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_3 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_4  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_4 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_5  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_5 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_6  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_6 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_7  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_7 , 3);
    m31 **device_lookup_verify_bitwise_xor_9_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_9_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_9_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_9_1 , 3);

    m31 **device_sub_component_inputs_verify_bitwise_xor_8  = clone_to_device<m31 *>(sub_component_inputs_verify_bitwise_xor_8 , 3 * 8);
    m31 **device_sub_component_inputs_verify_bitwise_xor_12 = clone_to_device<m31 *>(sub_component_inputs_verify_bitwise_xor_12, 3 * 2);
    m31 **device_sub_component_inputs_verify_bitwise_xor_4  = clone_to_device<m31 *>(sub_component_inputs_verify_bitwise_xor_4 , 3 * 2);
    m31 **device_sub_component_inputs_verify_bitwise_xor_7  = clone_to_device<m31 *>(sub_component_inputs_verify_bitwise_xor_7 , 3 * 2);
    m31 **device_sub_component_inputs_verify_bitwise_xor_9  = clone_to_device<m31 *>(sub_component_inputs_verify_bitwise_xor_9 , 3 * 2);

    uint32_t **device_blake_g_input = clone_to_device<uint32_t *>(blake_g_input, 6);


    timer global_timer;
    global_timer.start("generate blake_g base trace");

    unsigned trace_size = 1 << trace_log_size;
    int block_dim = trace_size < GEN_TRACE_BLAKE_G_THREAD_COUNT_MAX ? trace_size : GEN_TRACE_BLAKE_G_THREAD_COUNT_MAX;
    int num_blocks = block_dim < GEN_TRACE_BLAKE_G_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_g_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        device_lookup_blake_g_0,
        device_lookup_verify_bitwise_xor_12_0,
        device_lookup_verify_bitwise_xor_12_1,
        device_lookup_verify_bitwise_xor_4_0,
        device_lookup_verify_bitwise_xor_4_1,
        device_lookup_verify_bitwise_xor_7_0,
        device_lookup_verify_bitwise_xor_7_1,
        device_lookup_verify_bitwise_xor_8_0,
        device_lookup_verify_bitwise_xor_8_1,
        device_lookup_verify_bitwise_xor_8_2,
        device_lookup_verify_bitwise_xor_8_3,
        device_lookup_verify_bitwise_xor_8_4,
        device_lookup_verify_bitwise_xor_8_5,
        device_lookup_verify_bitwise_xor_8_6,
        device_lookup_verify_bitwise_xor_8_7,
        device_lookup_verify_bitwise_xor_9_0,
        device_lookup_verify_bitwise_xor_9_1,

        device_sub_component_inputs_verify_bitwise_xor_8,
        device_sub_component_inputs_verify_bitwise_xor_12,
        device_sub_component_inputs_verify_bitwise_xor_4,
        device_sub_component_inputs_verify_bitwise_xor_7,
        device_sub_component_inputs_verify_bitwise_xor_9,

        device_blake_g_input,

        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate blake_g base trace");

    cuda_free_memory(device_blake_g_input);

    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_9  );
    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_7  );
    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_4  );
    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_12  );
    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_8  );

    cuda_free_memory(device_lookup_verify_bitwise_xor_9_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_9_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_7  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_6  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_5  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_4  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_3  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_2  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_7_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_7_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_4_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_4_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_12_1 );
    cuda_free_memory(device_lookup_verify_bitwise_xor_12_0 );
    cuda_free_memory(device_lookup_blake_g_0);

    cuda_free_memory(device_traces);
}

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_blake_g_interaction_trace_col_gen_kernel(
    LookupElementsBasic<N>  *lookup_elements_n,
    LookupElementsBasic<M>  *lookup_elements_m,
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

template <int N>
__launch_bounds__(256, 2)
__global__ void generate_blake_g_interaction_trace_col_single_gen_kernel(
    LookupElementsBasic<N>  *lookup_elements_n,
    m31 **lookup_state_0,
    unsigned n_rows,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Enabler column: 1 for real rows (vec_index < n_rows), 0 for padding rows
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
        // Apply -1 * enabler to mask padding rows
        logup_col_write_frac(vec_index, mul(qm31{P-1, 0, 0, 0}, enabler_col), denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}


__global__ void generate_blake_g_interaction_trace_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
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
            interaction_traces[0][vec_index] = 0;
            interaction_traces[1][vec_index] = 0;
            interaction_traces[2][vec_index] = 0;
            interaction_traces[3][vec_index] = 0;
            qm31 pre_value = qm31 {0};
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        } else {
            qm31 pre_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][vec_index], interaction_traces[pre_index * 4 + 1][vec_index]},
                cm31{interaction_traces[pre_index * 4 + 2][vec_index], interaction_traces[pre_index * 4 + 3][vec_index]}
            };
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        }

        interaction_traces[rep_index * 4 + 0][vec_index] = numerator0[vec_index];
        interaction_traces[rep_index * 4 + 1][vec_index] = numerator1[vec_index];
        interaction_traces[rep_index * 4 + 2][vec_index] = numerator2[vec_index];
        interaction_traces[rep_index * 4 + 3][vec_index] = numerator3[vec_index];
    }
}

__global__ void generate_blake_g_interaction_trace_cumsum_shift(
    unsigned last_index,
    unsigned trace_size,
    m31 **interactive_traces,
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
        sum0 = add(sum0, interactive_traces[idx0][i]);
        sum1 = add(sum1, interactive_traces[idx1][i]);
        sum2 = add(sum2, interactive_traces[idx2][i]);
        sum3 = add(sum3, interactive_traces[idx3][i]);
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

__global__ void generate_blake_g_interaction_trace_coord_prefix_sum(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    m31 **interactive_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interactive_traces[4 * last_index - 4][vec_index] = sub(interactive_traces[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interactive_traces[4 * last_index - 3][vec_index] = sub(interactive_traces[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interactive_traces[4 * last_index - 2][vec_index] = sub(interactive_traces[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interactive_traces[4 * last_index - 1][vec_index] = sub(interactive_traces[4 * last_index - 1][vec_index], cumsum_shift.b.b);

    }
}



void generate_blake_g_interaction_traces(
    void *blake_g,
    void *verify_bitwise_xor_12,
    void *verify_bitwise_xor_4 ,
    void *verify_bitwise_xor_7 ,
    void *verify_bitwise_xor_8 ,
    void *verify_bitwise_xor_8_b ,
    void *verify_bitwise_xor_9 ,

    m31 **lookup_blake_g_0,
    m31 **lookup_verify_bitwise_xor_12_0,
    m31 **lookup_verify_bitwise_xor_12_1,
    m31 **lookup_verify_bitwise_xor_4_0 ,
    m31 **lookup_verify_bitwise_xor_4_1 ,
    m31 **lookup_verify_bitwise_xor_7_0 ,
    m31 **lookup_verify_bitwise_xor_7_1 ,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_4 ,
    m31 **lookup_verify_bitwise_xor_8_5 ,
    m31 **lookup_verify_bitwise_xor_8_6 ,
    m31 **lookup_verify_bitwise_xor_8_7 ,
    m31 **lookup_verify_bitwise_xor_9_0 ,
    m31 **lookup_verify_bitwise_xor_9_1 ,

    unsigned n_rows,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    BlakeG *blake_g_lookup_elements = (BlakeG*)blake_g;
    VerifyBitwiseXor_12 *verify_bitwise_xor_12_lookup_elements = (VerifyBitwiseXor_12*)verify_bitwise_xor_12;
    VerifyBitwiseXor_4  *verify_bitwise_xor_4_lookup_elements  = (VerifyBitwiseXor_4 *)verify_bitwise_xor_4;
    VerifyBitwiseXor_7  *verify_bitwise_xor_7_lookup_elements  = (VerifyBitwiseXor_7 *)verify_bitwise_xor_7;
    VerifyBitwiseXor_8  *verify_bitwise_xor_8_lookup_elements  = (VerifyBitwiseXor_8 *)verify_bitwise_xor_8;
    VerifyBitwiseXor_8  *verify_bitwise_xor_8_b_lookup_elements  = (VerifyBitwiseXor_8 *)verify_bitwise_xor_8_b;
    VerifyBitwiseXor_9  *verify_bitwise_xor_9_lookup_elements  = (VerifyBitwiseXor_9 *)verify_bitwise_xor_9;

    BlakeG *device_blake_g_lookup_elements = cuda_malloc<BlakeG>(1);
    VerifyBitwiseXor_12 *device_verify_bitwise_xor_12_lookup_elements = cuda_malloc<VerifyBitwiseXor_12>(1);
    VerifyBitwiseXor_4 *device_verify_bitwise_xor_4_lookup_elements = cuda_malloc<VerifyBitwiseXor_4>(1);
    VerifyBitwiseXor_7 *device_verify_bitwise_xor_7_lookup_elements = cuda_malloc<VerifyBitwiseXor_7>(1);
    VerifyBitwiseXor_8 *device_verify_bitwise_xor_8_lookup_elements = cuda_malloc<VerifyBitwiseXor_8>(1);
    VerifyBitwiseXor_8 *device_verify_bitwise_xor_8_b_lookup_elements = cuda_malloc<VerifyBitwiseXor_8>(1);
    VerifyBitwiseXor_9 *device_verify_bitwise_xor_9_lookup_elements = cuda_malloc<VerifyBitwiseXor_9>(1);

    cuda_mem_copy_host_to_device<BlakeG>(blake_g_lookup_elements, device_blake_g_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_12>(verify_bitwise_xor_12_lookup_elements, device_verify_bitwise_xor_12_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_4> (verify_bitwise_xor_4_lookup_elements , device_verify_bitwise_xor_4_lookup_elements , 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_7> (verify_bitwise_xor_7_lookup_elements , device_verify_bitwise_xor_7_lookup_elements , 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_8> (verify_bitwise_xor_8_lookup_elements , device_verify_bitwise_xor_8_lookup_elements , 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_8> (verify_bitwise_xor_8_b_lookup_elements , device_verify_bitwise_xor_8_b_lookup_elements , 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_9> (verify_bitwise_xor_9_lookup_elements , device_verify_bitwise_xor_9_lookup_elements , 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_blake_g_0 = clone_to_device<m31 *>(lookup_blake_g_0, 20);
    m31 **device_lookup_verify_bitwise_xor_12_0 = clone_to_device<m31 *>(lookup_verify_bitwise_xor_12_0, 3);
    m31 **device_lookup_verify_bitwise_xor_12_1 = clone_to_device<m31 *>(lookup_verify_bitwise_xor_12_1, 3);
    m31 **device_lookup_verify_bitwise_xor_4_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_4_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_4_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_4_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_7_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_7_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_7_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_7_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_2  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_2 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_3  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_3 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_4  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_4 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_5  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_5 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_6  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_6 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_7  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_7 , 3);
    m31 **device_lookup_verify_bitwise_xor_9_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_9_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_9_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_9_1 , 3);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4 * N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    // dump_lookup_data(lookup_verify_bitwise_xor_8_0, 3, trace_size);
    // dump_lookup_data(lookup_verify_bitwise_xor_8_1, 3, trace_size);

    timer global_timer;
    global_timer.start("generate blake_g interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For verify_bitwise_xor_8_0 & verify_bitwise_xor_8_1
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_lookup_elements,
        device_verify_bitwise_xor_8_lookup_elements,

        device_lookup_verify_bitwise_xor_8_0,
        device_lookup_verify_bitwise_xor_8_1,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 0, trace_size);

    // #1 Interaction trace For verify_bitwise_xor_8_2 & verify_bitwise_xor_8_3 (uses vbx_8_b)
    // NOTE: lookup_8_2 and 8_3 correspond to SIMD vbx_8_b[0] and vbx_8_b[1]
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_b_lookup_elements,
        device_verify_bitwise_xor_8_b_lookup_elements,

        device_lookup_verify_bitwise_xor_8_2,
        device_lookup_verify_bitwise_xor_8_3,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 1, trace_size);

    // #2 Interaction trace For verify_bitwise_xor_12_0 & verify_bitwise_xor_4_0
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_12_lookup_elements,
        device_verify_bitwise_xor_4_lookup_elements,

        device_lookup_verify_bitwise_xor_12_0,
        device_lookup_verify_bitwise_xor_4_0,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 2, trace_size);
    // #3 Interaction trace For verify_bitwise_xor_12_1 & verify_bitwise_xor_4_1
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_12_lookup_elements,
        device_verify_bitwise_xor_4_lookup_elements,

        device_lookup_verify_bitwise_xor_12_1,
        device_lookup_verify_bitwise_xor_4_1,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 3, trace_size);
    // #4 Interaction trace For verify_bitwise_xor_8_4 & verify_bitwise_xor_8_5 (uses vbx_8)
    // NOTE: lookup_8_4 and 8_5 correspond to SIMD vbx_8[2] and vbx_8[3]
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_lookup_elements,
        device_verify_bitwise_xor_8_lookup_elements,

        device_lookup_verify_bitwise_xor_8_4,
        device_lookup_verify_bitwise_xor_8_5,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 4, trace_size);

    // #5 Interaction trace For verify_bitwise_xor_8_6 & verify_bitwise_xor_8_7 (uses vbx_8_b)
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_b_lookup_elements,
        device_verify_bitwise_xor_8_b_lookup_elements,

        device_lookup_verify_bitwise_xor_8_6,
        device_lookup_verify_bitwise_xor_8_7,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 5, trace_size);

    // #6 Interaction trace For verify_bitwise_xor_7_0 & verify_bitwise_xor_9_0
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_7_lookup_elements,
        device_verify_bitwise_xor_9_lookup_elements,

        device_lookup_verify_bitwise_xor_7_0,
        device_lookup_verify_bitwise_xor_9_0,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 6, trace_size);

    // #7 Interaction trace For verify_bitwise_xor_7_1 & verify_bitwise_xor_9_1
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_7_lookup_elements,
        device_verify_bitwise_xor_9_lookup_elements,

        device_lookup_verify_bitwise_xor_7_1,
        device_lookup_verify_bitwise_xor_9_1,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 7, trace_size);

    // #8 Interaction trace For blake_g_0 (provider: apply enabler to mask padding rows)
    block_dim = trace_size < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_col_single_gen_kernel<20><<<num_blocks, block_dim>>>(
        device_blake_g_lookup_elements,

        device_lookup_blake_g_0,

        n_rows,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // dump_interaction_traces(interaction_traces, 8, trace_size);

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_blake_g_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_g_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[4 * N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate blake_g interaction trace");

    cuda_free_memory(device_verify_bitwise_xor_12_lookup_elements);
    cuda_free_memory(device_verify_bitwise_xor_4_lookup_elements);
    cuda_free_memory(device_verify_bitwise_xor_7_lookup_elements);
    cuda_free_memory(device_verify_bitwise_xor_8_lookup_elements);
    cuda_free_memory(device_verify_bitwise_xor_8_b_lookup_elements);
    cuda_free_memory(device_verify_bitwise_xor_9_lookup_elements);

    // dump_interaction_traces(interaction_traces, 8, trace_size);

    cuda_free_memory(device_logup_denom);

    cuda_free_memory(device_lookup_verify_bitwise_xor_9_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_9_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_7  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_6  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_5  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_4  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_3  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_2  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_7_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_7_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_4_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_4_0  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_12_1 );
    cuda_free_memory(device_lookup_verify_bitwise_xor_12_0 );
    cuda_free_memory(device_lookup_blake_g_0);

    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}