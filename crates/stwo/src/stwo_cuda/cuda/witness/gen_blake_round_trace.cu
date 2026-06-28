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

#include "gen_blake_round_trace.cuh"
#include "gen_blake_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_blake_round_sigma_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"
#include "gen_blake.cuh"

#define N_TRACE_COLUMNS 212

#define GEN_TRACE_BLAKE_ROUND_THREAD_COUNT_MAX 256

#define BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS 30

__launch_bounds__(256, 2)
__global__ void generate_blake_round_trace_kernel(
    m31 **traces,

    m31 **lookup_blake_g_0,
    m31 **lookup_blake_g_1,
    m31 **lookup_blake_g_2,
    m31 **lookup_blake_g_3,
    m31 **lookup_blake_g_4,
    m31 **lookup_blake_g_5,
    m31 **lookup_blake_g_6,
    m31 **lookup_blake_g_7,
    m31 **lookup_blake_round_0,
    m31 **lookup_blake_round_1,
    m31 **lookup_blake_round_sigma_0,
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,
    m31 **lookup_memory_address_to_id_5,
    m31 **lookup_memory_address_to_id_6,
    m31 **lookup_memory_address_to_id_7,
    m31 **lookup_memory_address_to_id_8,
    m31 **lookup_memory_address_to_id_9,
    m31 **lookup_memory_address_to_id_10,
    m31 **lookup_memory_address_to_id_11,
    m31 **lookup_memory_address_to_id_12,
    m31 **lookup_memory_address_to_id_13,
    m31 **lookup_memory_address_to_id_14,
    m31 **lookup_memory_address_to_id_15,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,
    m31 **lookup_memory_id_to_big_5,
    m31 **lookup_memory_id_to_big_6,
    m31 **lookup_memory_id_to_big_7,
    m31 **lookup_memory_id_to_big_8,
    m31 **lookup_memory_id_to_big_9,
    m31 **lookup_memory_id_to_big_10,
    m31 **lookup_memory_id_to_big_11,
    m31 **lookup_memory_id_to_big_12,
    m31 **lookup_memory_id_to_big_13,
    m31 **lookup_memory_id_to_big_14,
    m31 **lookup_memory_id_to_big_15,
    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_7_2_5_1,
    m31 **lookup_range_check_7_2_5_2,
    m31 **lookup_range_check_7_2_5_3,
    m31 **lookup_range_check_7_2_5_4,
    m31 **lookup_range_check_7_2_5_5,
    m31 **lookup_range_check_7_2_5_6,
    m31 **lookup_range_check_7_2_5_7,
    m31 **lookup_range_check_7_2_5_8,
    m31 **lookup_range_check_7_2_5_9,
    m31 **lookup_range_check_7_2_5_10,
    m31 **lookup_range_check_7_2_5_11,
    m31 **lookup_range_check_7_2_5_12,
    m31 **lookup_range_check_7_2_5_13,
    m31 **lookup_range_check_7_2_5_14,
    m31 **lookup_range_check_7_2_5_15,

    m31 **sub_component_inputs_blake_round_sigma,
    m31 **sub_component_inputs_range_check_7_2_5,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_blake_g,

    m31 **blake_round_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0 = {0};
    const m31 M31_1 = {1};
    const m31 M31_128 = {128};
    const m31 M31_2048 = {2048};
    const m31 M31_4 = {4};
    const m31 M31_512 = {512};

    const uint16_t UInt16_2 = 2;
    const uint16_t UInt16_7 = 7;
    const uint16_t UInt16_9 = 9;

    if (row < trace_size) {
        m31 input_limb_0_col0 = blake_round_input[0][row];
        traces[0][row] = input_limb_0_col0;
        m31 input_limb_1_col1 = blake_round_input[1][row];
        traces[1][row] = input_limb_1_col1;
        m31 input_limb_2_col2 = low_as_m31(blake_round_input[2][row]);
        traces[2][row] = input_limb_2_col2;
        m31 input_limb_3_col3 = high_as_m31(blake_round_input[2][row]);
        traces[3][row] = input_limb_3_col3;
        m31 input_limb_4_col4 = low_as_m31(blake_round_input[3][row]);
        traces[4][row] = input_limb_4_col4;
        m31 input_limb_5_col5 = high_as_m31(blake_round_input[3][row]);
        traces[5][row] = input_limb_5_col5;
        m31 input_limb_6_col6 = low_as_m31(blake_round_input[4][row]);
        traces[6][row] = input_limb_6_col6;
        m31 input_limb_7_col7 = high_as_m31(blake_round_input[4][row]);
        traces[7][row] = input_limb_7_col7;
        m31 input_limb_8_col8 = low_as_m31(blake_round_input[5][row]);
        traces[8][row] = input_limb_8_col8;
        m31 input_limb_9_col9 = high_as_m31(blake_round_input[5][row]);
        traces[9][row] = input_limb_9_col9;
        m31 input_limb_10_col10 = low_as_m31(blake_round_input[6][row]);
        traces[10][row] = input_limb_10_col10;
        m31 input_limb_11_col11 = high_as_m31(blake_round_input[6][row]);
        traces[11][row] = input_limb_11_col11;
        m31 input_limb_12_col12 = low_as_m31(blake_round_input[7][row]);
        traces[12][row] = input_limb_12_col12;
        m31 input_limb_13_col13 = high_as_m31(blake_round_input[7][row]);
        traces[13][row] = input_limb_13_col13;
        m31 input_limb_14_col14 = low_as_m31(blake_round_input[8][row]);
        traces[14][row] = input_limb_14_col14;
        m31 input_limb_15_col15 = high_as_m31(blake_round_input[8][row]);
        traces[15][row] = input_limb_15_col15;
        m31 input_limb_16_col16 = low_as_m31(blake_round_input[9][row]);
        traces[16][row] = input_limb_16_col16;
        m31 input_limb_17_col17 = high_as_m31(blake_round_input[9][row]);
        traces[17][row] = input_limb_17_col17;
        m31 input_limb_18_col18 = low_as_m31(blake_round_input[10][row]);
        traces[18][row] = input_limb_18_col18;
        m31 input_limb_19_col19 = high_as_m31(blake_round_input[10][row]);
        traces[19][row] = input_limb_19_col19;
        m31 input_limb_20_col20 = low_as_m31(blake_round_input[11][row]);
        traces[20][row] = input_limb_20_col20;
        m31 input_limb_21_col21 = high_as_m31(blake_round_input[11][row]);
        traces[21][row] = input_limb_21_col21;
        m31 input_limb_22_col22 = low_as_m31(blake_round_input[12][row]);
        traces[22][row] = input_limb_22_col22;
        m31 input_limb_23_col23 = high_as_m31(blake_round_input[12][row]);
        traces[23][row] = input_limb_23_col23;
        m31 input_limb_24_col24 = low_as_m31(blake_round_input[13][row]);
        traces[24][row] = input_limb_24_col24;
        m31 input_limb_25_col25 = high_as_m31(blake_round_input[13][row]);
        traces[25][row] = input_limb_25_col25;
        m31 input_limb_26_col26 = low_as_m31(blake_round_input[14][row]);
        traces[26][row] = input_limb_26_col26;
        m31 input_limb_27_col27 = high_as_m31(blake_round_input[14][row]);
        traces[27][row] = input_limb_27_col27;
        m31 input_limb_28_col28 = low_as_m31(blake_round_input[15][row]);
        traces[28][row] = input_limb_28_col28;
        m31 input_limb_29_col29 = high_as_m31(blake_round_input[15][row]);
        traces[29][row] = input_limb_29_col29;
        m31 input_limb_30_col30 = low_as_m31(blake_round_input[16][row]);
        traces[30][row] = input_limb_30_col30;
        m31 input_limb_31_col31 = high_as_m31(blake_round_input[16][row]);
        traces[31][row] = input_limb_31_col31;
        m31 input_limb_32_col32 = low_as_m31(blake_round_input[17][row]);
        traces[32][row] = input_limb_32_col32;
        m31 input_limb_33_col33 = high_as_m31(blake_round_input[17][row]);
        traces[33][row] = input_limb_33_col33;
        m31 input_limb_34_col34 = blake_round_input[18][row];
        traces[34][row] = input_limb_34_col34;

        sub_component_inputs_blake_round_sigma[0][row] = input_limb_1_col1;
        m31 blake_round_sigma_output_tmp_92ff8_0_0 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_1 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_2 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_3 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_4 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_5 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_6 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_7 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_8 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_9 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_10 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_11 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_12 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_13 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_14 = {0};
        m31 blake_round_sigma_output_tmp_92ff8_0_15 = {0};
        blake_round_sigma_deduce_output(
            input_limb_1_col1,
            &blake_round_sigma_output_tmp_92ff8_0_0,
            &blake_round_sigma_output_tmp_92ff8_0_1,
            &blake_round_sigma_output_tmp_92ff8_0_2,
            &blake_round_sigma_output_tmp_92ff8_0_3,
            &blake_round_sigma_output_tmp_92ff8_0_4,
            &blake_round_sigma_output_tmp_92ff8_0_5,
            &blake_round_sigma_output_tmp_92ff8_0_6,
            &blake_round_sigma_output_tmp_92ff8_0_7,
            &blake_round_sigma_output_tmp_92ff8_0_8,
            &blake_round_sigma_output_tmp_92ff8_0_9,
            &blake_round_sigma_output_tmp_92ff8_0_10,
            &blake_round_sigma_output_tmp_92ff8_0_11,
            &blake_round_sigma_output_tmp_92ff8_0_12,
            &blake_round_sigma_output_tmp_92ff8_0_13,
            &blake_round_sigma_output_tmp_92ff8_0_14,
            &blake_round_sigma_output_tmp_92ff8_0_15
        );
        m31 blake_round_sigma_output_limb_0_col35  = blake_round_sigma_output_tmp_92ff8_0_0;
        traces[35][row] = blake_round_sigma_output_limb_0_col35;
        m31 blake_round_sigma_output_limb_1_col36  = blake_round_sigma_output_tmp_92ff8_0_1;
        traces[36][row] = blake_round_sigma_output_limb_1_col36;
        m31 blake_round_sigma_output_limb_2_col37  = blake_round_sigma_output_tmp_92ff8_0_2;
        traces[37][row] = blake_round_sigma_output_limb_2_col37;
        m31 blake_round_sigma_output_limb_3_col38  = blake_round_sigma_output_tmp_92ff8_0_3;
        traces[38][row] = blake_round_sigma_output_limb_3_col38;
        m31 blake_round_sigma_output_limb_4_col39  = blake_round_sigma_output_tmp_92ff8_0_4;
        traces[39][row] = blake_round_sigma_output_limb_4_col39;
        m31 blake_round_sigma_output_limb_5_col40  = blake_round_sigma_output_tmp_92ff8_0_5;
        traces[40][row] = blake_round_sigma_output_limb_5_col40;
        m31 blake_round_sigma_output_limb_6_col41  = blake_round_sigma_output_tmp_92ff8_0_6;
        traces[41][row] = blake_round_sigma_output_limb_6_col41;
        m31 blake_round_sigma_output_limb_7_col42  = blake_round_sigma_output_tmp_92ff8_0_7;
        traces[42][row] = blake_round_sigma_output_limb_7_col42;
        m31 blake_round_sigma_output_limb_8_col43  = blake_round_sigma_output_tmp_92ff8_0_8;
        traces[43][row] = blake_round_sigma_output_limb_8_col43;
        m31 blake_round_sigma_output_limb_9_col44  = blake_round_sigma_output_tmp_92ff8_0_9;
        traces[44][row] = blake_round_sigma_output_limb_9_col44;
        m31 blake_round_sigma_output_limb_10_col45 = blake_round_sigma_output_tmp_92ff8_0_10;
        traces[45][row] = blake_round_sigma_output_limb_10_col45;
        m31 blake_round_sigma_output_limb_11_col46 = blake_round_sigma_output_tmp_92ff8_0_11;
        traces[46][row] = blake_round_sigma_output_limb_11_col46;
        m31 blake_round_sigma_output_limb_12_col47 = blake_round_sigma_output_tmp_92ff8_0_12;
        traces[47][row] = blake_round_sigma_output_limb_12_col47;
        m31 blake_round_sigma_output_limb_13_col48 = blake_round_sigma_output_tmp_92ff8_0_13;
        traces[48][row] = blake_round_sigma_output_limb_13_col48;
        m31 blake_round_sigma_output_limb_14_col49 = blake_round_sigma_output_tmp_92ff8_0_14;
        traces[49][row] = blake_round_sigma_output_limb_14_col49;
        m31 blake_round_sigma_output_limb_15_col50 = blake_round_sigma_output_tmp_92ff8_0_15;
        traces[50][row] = blake_round_sigma_output_limb_15_col50;

        lookup_blake_round_sigma_0[0][row]  = input_limb_1_col1;
        lookup_blake_round_sigma_0[1][row]  = blake_round_sigma_output_limb_0_col35;
        lookup_blake_round_sigma_0[2][row]  = blake_round_sigma_output_limb_1_col36;
        lookup_blake_round_sigma_0[3][row]  = blake_round_sigma_output_limb_2_col37;
        lookup_blake_round_sigma_0[4][row]  = blake_round_sigma_output_limb_3_col38;
        lookup_blake_round_sigma_0[5][row]  = blake_round_sigma_output_limb_4_col39;
        lookup_blake_round_sigma_0[6][row]  = blake_round_sigma_output_limb_5_col40;
        lookup_blake_round_sigma_0[7][row]  = blake_round_sigma_output_limb_6_col41;
        lookup_blake_round_sigma_0[8][row]  = blake_round_sigma_output_limb_7_col42;
        lookup_blake_round_sigma_0[9][row]  = blake_round_sigma_output_limb_8_col43;
        lookup_blake_round_sigma_0[10][row] = blake_round_sigma_output_limb_9_col44;
        lookup_blake_round_sigma_0[11][row] = blake_round_sigma_output_limb_10_col45;
        lookup_blake_round_sigma_0[12][row] = blake_round_sigma_output_limb_11_col46;
        lookup_blake_round_sigma_0[13][row] = blake_round_sigma_output_limb_12_col47;
        lookup_blake_round_sigma_0[14][row] = blake_round_sigma_output_limb_13_col48;
        lookup_blake_round_sigma_0[15][row] = blake_round_sigma_output_limb_14_col49;
        lookup_blake_round_sigma_0[16][row] = blake_round_sigma_output_limb_15_col50;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_1 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_0_col35)),
            &memory_address_to_id_value_tmp_92ff8_1
        );

        m31 memory_id_to_big_value_tmp_92ff8_2[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_1,
            memory_id_to_big_value_tmp_92ff8_2
        );
        uint16_t tmp_92ff8_3 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_2[1])) >> (UInt16_7));
        m31 low_16_bits_col51 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_2[1]), mul((m31{tmp_92ff8_3}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_2[0]));
        traces[51][row] = low_16_bits_col51;
        m31 high_16_bits_col52 = add(add(mul((memory_id_to_big_value_tmp_92ff8_2[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_2[2]) * (M31_4))), (m31{tmp_92ff8_3}));
        traces[52][row] = high_16_bits_col52;
        uint32_t expected_word_tmp_92ff8_4 = low_16_bits_col51 + (high_16_bits_col52 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_5 = ((expected_word_tmp_92ff8_4 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col53 = m31{low_7_ms_bits_tmp_92ff8_5};
        traces[53][row] = low_7_ms_bits_col53;
        uint16_t high_14_ms_bits_tmp_92ff8_6 = ((expected_word_tmp_92ff8_4 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col54 = m31{high_14_ms_bits_tmp_92ff8_6};
        traces[54][row] = high_14_ms_bits_col54;
        uint16_t high_5_ms_bits_tmp_92ff8_7 = ((high_14_ms_bits_tmp_92ff8_6) >> (UInt16_9));
        m31 high_5_ms_bits_col55 = m31{high_5_ms_bits_tmp_92ff8_7};
        traces[55][row] = high_5_ms_bits_col55;
        sub_component_inputs_range_check_7_2_5[0 * 3 + 0][row] = low_7_ms_bits_col53;
        sub_component_inputs_range_check_7_2_5[0 * 3 + 1][row] = sub((high_16_bits_col52), mul((high_14_ms_bits_col54), (M31_4)));
        sub_component_inputs_range_check_7_2_5[0 * 3 + 2][row] = high_5_ms_bits_col55;

        lookup_range_check_7_2_5_0[0][row] = low_7_ms_bits_col53;
        lookup_range_check_7_2_5_0[1][row] = sub((high_16_bits_col52), mul((high_14_ms_bits_col54), (M31_4)));;
        lookup_range_check_7_2_5_0[2][row] = high_5_ms_bits_col55;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_8 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_0_col35)),
            &memory_address_to_id_value_tmp_92ff8_8
        );
        m31 message_word_0_id_col56 = memory_address_to_id_value_tmp_92ff8_8;
        traces[56][row] = message_word_0_id_col56;
        sub_component_inputs_memory_address_to_id[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_0_col35));
        lookup_memory_address_to_id_0[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_0_col35));
        lookup_memory_address_to_id_0[1][row] = message_word_0_id_col56;
        sub_component_inputs_memory_id_to_big[0][row] = message_word_0_id_col56;
        lookup_memory_id_to_big_0[0][row] = message_word_0_id_col56;
        lookup_memory_id_to_big_0[1][row] = sub((low_16_bits_col51), mul((low_7_ms_bits_col53), (M31_512)));
        lookup_memory_id_to_big_0[2][row] = add((low_7_ms_bits_col53), mul(sub((high_16_bits_col52), mul((high_14_ms_bits_col54), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_0[3][row] = sub((high_14_ms_bits_col54), mul((high_5_ms_bits_col55), (M31_512)));
        lookup_memory_id_to_big_0[4][row] = high_5_ms_bits_col55;
        lookup_memory_id_to_big_0[5][row] = M31_0;
        lookup_memory_id_to_big_0[6][row] = M31_0;
        lookup_memory_id_to_big_0[7][row] = M31_0;
        lookup_memory_id_to_big_0[8][row] = M31_0;
        lookup_memory_id_to_big_0[9][row] = M31_0;
        lookup_memory_id_to_big_0[10][row] = M31_0;
        lookup_memory_id_to_big_0[11][row] = M31_0;
        lookup_memory_id_to_big_0[12][row] = M31_0;
        lookup_memory_id_to_big_0[13][row] = M31_0;
        lookup_memory_id_to_big_0[14][row] = M31_0;
        lookup_memory_id_to_big_0[15][row] = M31_0;
        lookup_memory_id_to_big_0[16][row] = M31_0;
        lookup_memory_id_to_big_0[17][row] = M31_0;
        lookup_memory_id_to_big_0[18][row] = M31_0;
        lookup_memory_id_to_big_0[19][row] = M31_0;
        lookup_memory_id_to_big_0[20][row] = M31_0;
        lookup_memory_id_to_big_0[21][row] = M31_0;
        lookup_memory_id_to_big_0[22][row] = M31_0;
        lookup_memory_id_to_big_0[23][row] = M31_0;
        lookup_memory_id_to_big_0[24][row] = M31_0;
        lookup_memory_id_to_big_0[25][row] = M31_0;
        lookup_memory_id_to_big_0[26][row] = M31_0;
        lookup_memory_id_to_big_0[27][row] = M31_0;
        lookup_memory_id_to_big_0[28][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_9 = expected_word_tmp_92ff8_4;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_10 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_1_col36)),
            &memory_address_to_id_value_tmp_92ff8_10
        );

        m31 memory_id_to_big_value_tmp_92ff8_11[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_10,
            memory_id_to_big_value_tmp_92ff8_11
        );
        uint16_t tmp_92ff8_12 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_11[1])) >> (UInt16_7));
        m31 low_16_bits_col57 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_11[1]), mul((m31{tmp_92ff8_12}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_11[0]));
        traces[57][row] = low_16_bits_col57;
        m31 high_16_bits_col58 = add(add(mul((memory_id_to_big_value_tmp_92ff8_11[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_11[2]) * (M31_4))), (m31{tmp_92ff8_12}));
        traces[58][row] = high_16_bits_col58;
        uint32_t expected_word_tmp_92ff8_13 = low_16_bits_col57 + (high_16_bits_col58 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_14 = ((expected_word_tmp_92ff8_13 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col59 = m31{low_7_ms_bits_tmp_92ff8_14};
        traces[59][row] = low_7_ms_bits_col59;
        uint16_t high_14_ms_bits_tmp_92ff8_15 = ((expected_word_tmp_92ff8_13 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col60 = m31{high_14_ms_bits_tmp_92ff8_15};
        traces[60][row] = high_14_ms_bits_col60;
        uint16_t high_5_ms_bits_tmp_92ff8_16 = ((high_14_ms_bits_tmp_92ff8_15) >> (UInt16_9));
        m31 high_5_ms_bits_col61 = m31{high_5_ms_bits_tmp_92ff8_16};
        traces[61][row] = high_5_ms_bits_col61;
        sub_component_inputs_range_check_7_2_5[1 * 3 + 0][row] = low_7_ms_bits_col59;
        sub_component_inputs_range_check_7_2_5[1 * 3 + 1][row] = sub((high_16_bits_col58), mul((high_14_ms_bits_col60), (M31_4)));
        sub_component_inputs_range_check_7_2_5[1 * 3 + 2][row] = high_5_ms_bits_col61;
        lookup_range_check_7_2_5_1[0][row] = low_7_ms_bits_col59;
        lookup_range_check_7_2_5_1[1][row] = sub((high_16_bits_col58), mul((high_14_ms_bits_col60), (M31_4)));
        lookup_range_check_7_2_5_1[2][row] = high_5_ms_bits_col61;


        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_17 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_1_col36)),
            &memory_address_to_id_value_tmp_92ff8_17
        );
        m31 message_word_1_id_col62 = memory_address_to_id_value_tmp_92ff8_17;
        traces[62][row] = message_word_1_id_col62;
        sub_component_inputs_memory_address_to_id[1][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_1_col36));
        lookup_memory_address_to_id_1[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_1_col36));
        lookup_memory_address_to_id_1[1][row] = message_word_1_id_col62;
        sub_component_inputs_memory_id_to_big[1][row] = message_word_1_id_col62;
        lookup_memory_id_to_big_1[0][row] = message_word_1_id_col62;
        lookup_memory_id_to_big_1[1][row] = sub((low_16_bits_col57), mul((low_7_ms_bits_col59), (M31_512)));
        lookup_memory_id_to_big_1[2][row] = add((low_7_ms_bits_col59), mul(sub((high_16_bits_col58), mul((high_14_ms_bits_col60), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_1[3][row] = sub((high_14_ms_bits_col60), mul((high_5_ms_bits_col61), (M31_512)));
        lookup_memory_id_to_big_1[4][row] = high_5_ms_bits_col61;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_1[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_18 = expected_word_tmp_92ff8_13;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_19 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_2_col37)),
            &memory_address_to_id_value_tmp_92ff8_19
        );
        m31 memory_id_to_big_value_tmp_92ff8_20[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_19,
            memory_id_to_big_value_tmp_92ff8_20
        );
        uint16_t tmp_92ff8_21 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_20[1])) >> (UInt16_7));
        m31 low_16_bits_col63 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_20[1]), mul((m31{tmp_92ff8_21}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_20[0]));
        traces[63][row] = low_16_bits_col63;
        m31 high_16_bits_col64 = add(add(mul((memory_id_to_big_value_tmp_92ff8_20[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_20[2]) * (M31_4))), (m31{tmp_92ff8_21}));
        traces[64][row] = high_16_bits_col64;
        uint32_t expected_word_tmp_92ff8_22 = low_16_bits_col63 + (high_16_bits_col64 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_23 = ((expected_word_tmp_92ff8_22 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col65 = m31{low_7_ms_bits_tmp_92ff8_23};
        traces[65][row] = low_7_ms_bits_col65;
        uint16_t high_14_ms_bits_tmp_92ff8_24 = ((expected_word_tmp_92ff8_22 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col66 = m31{high_14_ms_bits_tmp_92ff8_24};
        traces[66][row] = high_14_ms_bits_col66;
        uint16_t high_5_ms_bits_tmp_92ff8_25 = ((high_14_ms_bits_tmp_92ff8_24) >> (UInt16_9));
        m31 high_5_ms_bits_col67 = m31{high_5_ms_bits_tmp_92ff8_25};
        traces[67][row] = high_5_ms_bits_col67;
        sub_component_inputs_range_check_7_2_5[2 * 3 + 0][row] = low_7_ms_bits_col65;
        sub_component_inputs_range_check_7_2_5[2 * 3 + 1][row] = sub((high_16_bits_col64), mul((high_14_ms_bits_col66), (M31_4)));
        sub_component_inputs_range_check_7_2_5[2 * 3 + 2][row] = high_5_ms_bits_col67;
        lookup_range_check_7_2_5_2[0][row] = low_7_ms_bits_col65;
        lookup_range_check_7_2_5_2[1][row] = sub((high_16_bits_col64), mul((high_14_ms_bits_col66), (M31_4)));
        lookup_range_check_7_2_5_2[2][row] = high_5_ms_bits_col67;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_26 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_2_col37)),
            &memory_address_to_id_value_tmp_92ff8_26
        );
        m31 message_word_2_id_col68 = memory_address_to_id_value_tmp_92ff8_26;
        traces[68][row] = message_word_2_id_col68;
        sub_component_inputs_memory_address_to_id[2][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_2_col37));
        lookup_memory_address_to_id_2[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_2_col37));
        lookup_memory_address_to_id_2[1][row] = message_word_2_id_col68;
        sub_component_inputs_memory_id_to_big[2][row] = message_word_2_id_col68;
        lookup_memory_id_to_big_2[0][row] = message_word_2_id_col68;
        lookup_memory_id_to_big_2[1][row] = sub((low_16_bits_col63), mul((low_7_ms_bits_col65), (M31_512)));
        lookup_memory_id_to_big_2[2][row] = add((low_7_ms_bits_col65), mul(sub((high_16_bits_col64), mul((high_14_ms_bits_col66), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_2[3][row] = sub((high_14_ms_bits_col66), mul((high_5_ms_bits_col67), (M31_512)));
        lookup_memory_id_to_big_2[4][row] = high_5_ms_bits_col67;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_2[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_27 = expected_word_tmp_92ff8_22;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_28 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_3_col38)),
            &memory_address_to_id_value_tmp_92ff8_28
        );
        m31 memory_id_to_big_value_tmp_92ff8_29[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_28,
            memory_id_to_big_value_tmp_92ff8_29
        );
        uint16_t tmp_92ff8_30 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_29[1])) >> (UInt16_7));
        m31 low_16_bits_col69 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_29[1]), mul((m31{tmp_92ff8_30}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_29[0]));
        traces[69][row] = low_16_bits_col69;
        m31 high_16_bits_col70 = add(add(mul((memory_id_to_big_value_tmp_92ff8_29[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_29[2]) * (M31_4))), (m31{tmp_92ff8_30}));
        traces[70][row] = high_16_bits_col70;
        uint32_t expected_word_tmp_92ff8_31 = low_16_bits_col69 + (high_16_bits_col70 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_32 = ((expected_word_tmp_92ff8_31 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col71 = m31{low_7_ms_bits_tmp_92ff8_32};
        traces[71][row] = low_7_ms_bits_col71;
        uint16_t high_14_ms_bits_tmp_92ff8_33 = ((expected_word_tmp_92ff8_31 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col72 = m31{high_14_ms_bits_tmp_92ff8_33};
        traces[72][row] = high_14_ms_bits_col72;
        uint16_t high_5_ms_bits_tmp_92ff8_34 = ((high_14_ms_bits_tmp_92ff8_33) >> (UInt16_9));
        m31 high_5_ms_bits_col73 = m31{high_5_ms_bits_tmp_92ff8_34};
        traces[73][row] = high_5_ms_bits_col73;
        sub_component_inputs_range_check_7_2_5[3 * 3 + 0][row] = low_7_ms_bits_col71;
        sub_component_inputs_range_check_7_2_5[3 * 3 + 1][row] = sub((high_16_bits_col70), mul((high_14_ms_bits_col72), (M31_4)));
        sub_component_inputs_range_check_7_2_5[3 * 3 + 2][row] = high_5_ms_bits_col73;
        lookup_range_check_7_2_5_3[0][row] = low_7_ms_bits_col71;
        lookup_range_check_7_2_5_3[1][row] = sub((high_16_bits_col70), mul((high_14_ms_bits_col72), (M31_4)));
        lookup_range_check_7_2_5_3[2][row] = high_5_ms_bits_col73;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_35 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_3_col38)),
            &memory_address_to_id_value_tmp_92ff8_35
        );
        m31 message_word_3_id_col74 = memory_address_to_id_value_tmp_92ff8_35;
        traces[74][row] = message_word_3_id_col74;
        sub_component_inputs_memory_address_to_id[3][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_3_col38));
        lookup_memory_address_to_id_3[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_3_col38));
        lookup_memory_address_to_id_3[1][row] = message_word_3_id_col74;
        sub_component_inputs_memory_id_to_big[3][row] = message_word_3_id_col74;
        lookup_memory_id_to_big_3[0][row] = message_word_3_id_col74;
        lookup_memory_id_to_big_3[1][row] = sub((low_16_bits_col69), mul((low_7_ms_bits_col71), (M31_512)));
        lookup_memory_id_to_big_3[2][row] = add((low_7_ms_bits_col71), mul(sub((high_16_bits_col70), mul((high_14_ms_bits_col72), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_3[3][row] = sub((high_14_ms_bits_col72), mul((high_5_ms_bits_col73), (M31_512)));
        lookup_memory_id_to_big_3[4][row] = high_5_ms_bits_col73;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_3[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_36 = expected_word_tmp_92ff8_31;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_37 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_4_col39)),
            &memory_address_to_id_value_tmp_92ff8_37
        );
        m31 memory_id_to_big_value_tmp_92ff8_38[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_37,
            memory_id_to_big_value_tmp_92ff8_38
        );
        uint16_t tmp_92ff8_39 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_38[1])) >> (UInt16_7));
        m31 low_16_bits_col75 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_38[1]), mul((m31{tmp_92ff8_39}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_38[0]));
        traces[75][row] = low_16_bits_col75;
        m31 high_16_bits_col76 = add(add(mul((memory_id_to_big_value_tmp_92ff8_38[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_38[2]) * (M31_4))), (m31{tmp_92ff8_39}));
        traces[76][row] = high_16_bits_col76;
        uint32_t expected_word_tmp_92ff8_40 = low_16_bits_col75 + (high_16_bits_col76 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_41 = ((expected_word_tmp_92ff8_40 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col77 = m31{low_7_ms_bits_tmp_92ff8_41};
        traces[77][row] = low_7_ms_bits_col77;
        uint16_t high_14_ms_bits_tmp_92ff8_42 = ((expected_word_tmp_92ff8_40 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col78 = m31{high_14_ms_bits_tmp_92ff8_42};
        traces[78][row] = high_14_ms_bits_col78;
        uint16_t high_5_ms_bits_tmp_92ff8_43 = ((high_14_ms_bits_tmp_92ff8_42) >> (UInt16_9));
        m31 high_5_ms_bits_col79 = m31{high_5_ms_bits_tmp_92ff8_43};
        traces[79][row] = high_5_ms_bits_col79;
        sub_component_inputs_range_check_7_2_5[4 * 3 + 0][row] = low_7_ms_bits_col77;
        sub_component_inputs_range_check_7_2_5[4 * 3 + 1][row] = sub((high_16_bits_col76), mul((high_14_ms_bits_col78), (M31_4)));
        sub_component_inputs_range_check_7_2_5[4 * 3 + 2][row] = high_5_ms_bits_col79;
        lookup_range_check_7_2_5_4[0][row] = low_7_ms_bits_col77;
        lookup_range_check_7_2_5_4[1][row] = sub((high_16_bits_col76), mul((high_14_ms_bits_col78), (M31_4)));
        lookup_range_check_7_2_5_4[2][row] = high_5_ms_bits_col79;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_44 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_4_col39)),
            &memory_address_to_id_value_tmp_92ff8_44
        );
        m31 message_word_4_id_col80 = memory_address_to_id_value_tmp_92ff8_44;
        traces[80][row] = message_word_4_id_col80;
        sub_component_inputs_memory_address_to_id[4][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_4_col39));
        lookup_memory_address_to_id_4[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_4_col39));
        lookup_memory_address_to_id_4[1][row] = message_word_4_id_col80;
        sub_component_inputs_memory_id_to_big[4][row] = message_word_4_id_col80;
        lookup_memory_id_to_big_4[0][row] = message_word_4_id_col80;
        lookup_memory_id_to_big_4[1][row] = sub((low_16_bits_col75), mul((low_7_ms_bits_col77), (M31_512)));
        lookup_memory_id_to_big_4[2][row] = add((low_7_ms_bits_col77), mul(sub((high_16_bits_col76), mul((high_14_ms_bits_col78), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_4[3][row] = sub((high_14_ms_bits_col78), mul((high_5_ms_bits_col79), (M31_512)));
        lookup_memory_id_to_big_4[4][row] = high_5_ms_bits_col79;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_4[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_45 = expected_word_tmp_92ff8_40;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_46 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_5_col40)),
            &memory_address_to_id_value_tmp_92ff8_46
        );
        m31 memory_id_to_big_value_tmp_92ff8_47[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_46,
            memory_id_to_big_value_tmp_92ff8_47
        );
        uint16_t tmp_92ff8_48 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_47[1])) >> (UInt16_7));
        m31 low_16_bits_col81 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_47[1]), mul((m31{tmp_92ff8_48}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_47[0]));
        traces[81][row] = low_16_bits_col81;
        m31 high_16_bits_col82 = add(add(mul((memory_id_to_big_value_tmp_92ff8_47[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_47[2]) * (M31_4))), (m31{tmp_92ff8_48}));
        traces[82][row] = high_16_bits_col82;
        uint32_t expected_word_tmp_92ff8_49 = low_16_bits_col81 + (high_16_bits_col82 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_50 = ((expected_word_tmp_92ff8_49 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col83 = m31{low_7_ms_bits_tmp_92ff8_50};
        traces[83][row] = low_7_ms_bits_col83;
        uint16_t high_14_ms_bits_tmp_92ff8_51 = ((expected_word_tmp_92ff8_49 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col84 = m31{high_14_ms_bits_tmp_92ff8_51};
        traces[84][row] = high_14_ms_bits_col84;
        uint16_t high_5_ms_bits_tmp_92ff8_52 = ((high_14_ms_bits_tmp_92ff8_51) >> (UInt16_9));
        m31 high_5_ms_bits_col85 = m31{high_5_ms_bits_tmp_92ff8_52};
        traces[85][row] = high_5_ms_bits_col85;
        sub_component_inputs_range_check_7_2_5[5 * 3 + 0][row] = low_7_ms_bits_col83;
        sub_component_inputs_range_check_7_2_5[5 * 3 + 1][row] = sub((high_16_bits_col82), mul((high_14_ms_bits_col84), (M31_4)));
        sub_component_inputs_range_check_7_2_5[5 * 3 + 2][row] = high_5_ms_bits_col85;
        lookup_range_check_7_2_5_5[0][row] = low_7_ms_bits_col83;
        lookup_range_check_7_2_5_5[1][row] = sub((high_16_bits_col82), mul((high_14_ms_bits_col84), (M31_4)));
        lookup_range_check_7_2_5_5[2][row] = high_5_ms_bits_col85;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_53 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_5_col40)),
            &memory_address_to_id_value_tmp_92ff8_53
        );
        m31 message_word_5_id_col86 = memory_address_to_id_value_tmp_92ff8_53;
        traces[86][row] = message_word_5_id_col86;
        sub_component_inputs_memory_address_to_id[5][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_5_col40));
        lookup_memory_address_to_id_5[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_5_col40));
        lookup_memory_address_to_id_5[1][row] = message_word_5_id_col86;
        sub_component_inputs_memory_id_to_big[5][row] = message_word_5_id_col86;
        lookup_memory_id_to_big_5[0][row] = message_word_5_id_col86;
        lookup_memory_id_to_big_5[1][row] = sub((low_16_bits_col81), mul((low_7_ms_bits_col83), (M31_512)));
        lookup_memory_id_to_big_5[2][row] = add((low_7_ms_bits_col83), mul(sub((high_16_bits_col82), mul((high_14_ms_bits_col84), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_5[3][row] = sub((high_14_ms_bits_col84), mul((high_5_ms_bits_col85), (M31_512)));
        lookup_memory_id_to_big_5[4][row] = high_5_ms_bits_col85;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_5[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_54 = expected_word_tmp_92ff8_49;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_55 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_6_col41)),
            &memory_address_to_id_value_tmp_92ff8_55
        );
        m31 memory_id_to_big_value_tmp_92ff8_56[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_55,
            memory_id_to_big_value_tmp_92ff8_56
        );
        uint16_t tmp_92ff8_57 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_56[1])) >> (UInt16_7));
        m31 low_16_bits_col87 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_56[1]), mul((m31{tmp_92ff8_57}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_56[0]));
        traces[87][row] = low_16_bits_col87;
        m31 high_16_bits_col88 = add(add(mul((memory_id_to_big_value_tmp_92ff8_56[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_56[2]) * (M31_4))), (m31{tmp_92ff8_57}));
        traces[88][row] = high_16_bits_col88;
        uint32_t expected_word_tmp_92ff8_58 = low_16_bits_col87 + (high_16_bits_col88 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_59 = ((expected_word_tmp_92ff8_58 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col89 = m31{low_7_ms_bits_tmp_92ff8_59};
        traces[89][row] = low_7_ms_bits_col89;
        uint16_t high_14_ms_bits_tmp_92ff8_60 = ((expected_word_tmp_92ff8_58 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col90 = m31{high_14_ms_bits_tmp_92ff8_60};
        traces[90][row] = high_14_ms_bits_col90;
        uint16_t high_5_ms_bits_tmp_92ff8_61 = ((high_14_ms_bits_tmp_92ff8_60) >> (UInt16_9));
        m31 high_5_ms_bits_col91 = m31{high_5_ms_bits_tmp_92ff8_61};
        traces[91][row] = high_5_ms_bits_col91;
        sub_component_inputs_range_check_7_2_5[6 * 3 + 0][row] = low_7_ms_bits_col89;
        sub_component_inputs_range_check_7_2_5[6 * 3 + 1][row] = sub((high_16_bits_col88), mul((high_14_ms_bits_col90), (M31_4)));
        sub_component_inputs_range_check_7_2_5[6 * 3 + 2][row] = high_5_ms_bits_col91;
        lookup_range_check_7_2_5_6[0][row] = low_7_ms_bits_col89;
        lookup_range_check_7_2_5_6[1][row] = sub((high_16_bits_col88), mul((high_14_ms_bits_col90), (M31_4)));
        lookup_range_check_7_2_5_6[2][row] = high_5_ms_bits_col91;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_62 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_6_col41)),
            &memory_address_to_id_value_tmp_92ff8_62
        );
        m31 message_word_6_id_col92 = memory_address_to_id_value_tmp_92ff8_62;
        traces[92][row] = message_word_6_id_col92;
        sub_component_inputs_memory_address_to_id[6][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_6_col41));
        lookup_memory_address_to_id_6[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_6_col41));
        lookup_memory_address_to_id_6[1][row] = message_word_6_id_col92;
        sub_component_inputs_memory_id_to_big[6][row] = message_word_6_id_col92;
        lookup_memory_id_to_big_6[0][row] = message_word_6_id_col92;
        lookup_memory_id_to_big_6[1][row] = sub((low_16_bits_col87), mul((low_7_ms_bits_col89), (M31_512)));
        lookup_memory_id_to_big_6[2][row] = add((low_7_ms_bits_col89), mul(sub((high_16_bits_col88), mul((high_14_ms_bits_col90), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_6[3][row] = sub((high_14_ms_bits_col90), mul((high_5_ms_bits_col91), (M31_512)));
        lookup_memory_id_to_big_6[4][row] = high_5_ms_bits_col91;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_6[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_63 = expected_word_tmp_92ff8_58;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_64 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_7_col42)),
            &memory_address_to_id_value_tmp_92ff8_64
        );
        m31 memory_id_to_big_value_tmp_92ff8_65[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_64,
            memory_id_to_big_value_tmp_92ff8_65
        );
        uint16_t tmp_92ff8_66 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_65[1])) >> (UInt16_7));
        m31 low_16_bits_col93 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_65[1]), mul((m31{tmp_92ff8_66}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_65[0]));
        traces[93][row] = low_16_bits_col93;
        m31 high_16_bits_col94 = add(add(mul((memory_id_to_big_value_tmp_92ff8_65[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_65[2]) * (M31_4))), (m31{tmp_92ff8_66}));
        traces[94][row] = high_16_bits_col94;
        uint32_t expected_word_tmp_92ff8_67 = low_16_bits_col93 + (high_16_bits_col94 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_68 = ((expected_word_tmp_92ff8_67 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col95 = m31{low_7_ms_bits_tmp_92ff8_68};
        traces[95][row] = low_7_ms_bits_col95;
        uint16_t high_14_ms_bits_tmp_92ff8_69 = ((expected_word_tmp_92ff8_67 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col96 = m31{high_14_ms_bits_tmp_92ff8_69};
        traces[96][row] = high_14_ms_bits_col96;
        uint16_t high_5_ms_bits_tmp_92ff8_70 = ((high_14_ms_bits_tmp_92ff8_69) >> (UInt16_9));
        m31 high_5_ms_bits_col97 = m31{high_5_ms_bits_tmp_92ff8_70};
        traces[97][row] = high_5_ms_bits_col97;
        sub_component_inputs_range_check_7_2_5[7 * 3 + 0][row] = low_7_ms_bits_col95;
        sub_component_inputs_range_check_7_2_5[7 * 3 + 1][row] = sub((high_16_bits_col94), mul((high_14_ms_bits_col96), (M31_4)));
        sub_component_inputs_range_check_7_2_5[7 * 3 + 2][row] = high_5_ms_bits_col97;
        lookup_range_check_7_2_5_7[0][row] = low_7_ms_bits_col95;
        lookup_range_check_7_2_5_7[1][row] = sub((high_16_bits_col94), mul((high_14_ms_bits_col96), (M31_4)));
        lookup_range_check_7_2_5_7[2][row] = high_5_ms_bits_col97;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_71 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_7_col42)),
            &memory_address_to_id_value_tmp_92ff8_71
        );
        m31 message_word_7_id_col98 = memory_address_to_id_value_tmp_92ff8_71;
        traces[98][row] = message_word_7_id_col98;
        sub_component_inputs_memory_address_to_id[7][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_7_col42));
        lookup_memory_address_to_id_7[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_7_col42));
        lookup_memory_address_to_id_7[1][row] = message_word_7_id_col98;
        sub_component_inputs_memory_id_to_big[7][row] = message_word_7_id_col98;
        lookup_memory_id_to_big_7[0][row] = message_word_7_id_col98;
        lookup_memory_id_to_big_7[1][row] = sub((low_16_bits_col93), mul((low_7_ms_bits_col95), (M31_512)));
        lookup_memory_id_to_big_7[2][row] = add((low_7_ms_bits_col95), mul(sub((high_16_bits_col94), mul((high_14_ms_bits_col96), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_7[3][row] = sub((high_14_ms_bits_col96), mul((high_5_ms_bits_col97), (M31_512)));
        lookup_memory_id_to_big_7[4][row] = high_5_ms_bits_col97;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_7[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_72 = expected_word_tmp_92ff8_67;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_73 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_8_col43)),
            &memory_address_to_id_value_tmp_92ff8_73
        );
        m31 memory_id_to_big_value_tmp_92ff8_74[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_73,
            memory_id_to_big_value_tmp_92ff8_74
        );
        uint16_t tmp_92ff8_75 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_74[1])) >> (UInt16_7));
        m31 low_16_bits_col99 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_74[1]), mul((m31{tmp_92ff8_75}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_74[0]));
        traces[99][row] = low_16_bits_col99;
        m31 high_16_bits_col100 = add(add(mul((memory_id_to_big_value_tmp_92ff8_74[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_74[2]) * (M31_4))), (m31{tmp_92ff8_75}));
        traces[100][row] = high_16_bits_col100;
        uint32_t expected_word_tmp_92ff8_76 = low_16_bits_col99 + (high_16_bits_col100 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_77 = ((expected_word_tmp_92ff8_76 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col101 = m31{low_7_ms_bits_tmp_92ff8_77};
        traces[101][row] = low_7_ms_bits_col101;
        uint16_t high_14_ms_bits_tmp_92ff8_78 = ((expected_word_tmp_92ff8_76 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col102 = m31{high_14_ms_bits_tmp_92ff8_78};
        traces[102][row] = high_14_ms_bits_col102;
        uint16_t high_5_ms_bits_tmp_92ff8_79 = ((high_14_ms_bits_tmp_92ff8_78) >> (UInt16_9));
        m31 high_5_ms_bits_col103 = m31{high_5_ms_bits_tmp_92ff8_79};
        traces[103][row] = high_5_ms_bits_col103;
        sub_component_inputs_range_check_7_2_5[8 * 3 + 0][row] = low_7_ms_bits_col101;
        sub_component_inputs_range_check_7_2_5[8 * 3 + 1][row] = sub((high_16_bits_col100), mul((high_14_ms_bits_col102), (M31_4)));
        sub_component_inputs_range_check_7_2_5[8 * 3 + 2][row] = high_5_ms_bits_col103;
        lookup_range_check_7_2_5_8[0][row] = low_7_ms_bits_col101;
        lookup_range_check_7_2_5_8[1][row] = sub((high_16_bits_col100), mul((high_14_ms_bits_col102), (M31_4)));
        lookup_range_check_7_2_5_8[2][row] = high_5_ms_bits_col103;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_80 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_8_col43)),
            &memory_address_to_id_value_tmp_92ff8_80
        );
        m31 message_word_8_id_col104 = memory_address_to_id_value_tmp_92ff8_80;
        traces[104][row] = message_word_8_id_col104;
        sub_component_inputs_memory_address_to_id[8][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_8_col43));
        lookup_memory_address_to_id_8[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_8_col43));
        lookup_memory_address_to_id_8[1][row] = message_word_8_id_col104;
        sub_component_inputs_memory_id_to_big[8][row] = message_word_8_id_col104;
        lookup_memory_id_to_big_8[0][row] = message_word_8_id_col104;
        lookup_memory_id_to_big_8[1][row] = sub((low_16_bits_col99), mul((low_7_ms_bits_col101), (M31_512)));
        lookup_memory_id_to_big_8[2][row] = add((low_7_ms_bits_col101), mul(sub((high_16_bits_col100), mul((high_14_ms_bits_col102), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_8[3][row] = sub((high_14_ms_bits_col102), mul((high_5_ms_bits_col103), (M31_512)));
        lookup_memory_id_to_big_8[4][row] = high_5_ms_bits_col103;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_8[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_81 = expected_word_tmp_92ff8_76;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_82 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_9_col44)),
            &memory_address_to_id_value_tmp_92ff8_82
        );
        m31 memory_id_to_big_value_tmp_92ff8_83[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_82,
            memory_id_to_big_value_tmp_92ff8_83
        );
        uint16_t tmp_92ff8_84 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_83[1])) >> (UInt16_7));
        m31 low_16_bits_col105 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_83[1]), mul((m31{tmp_92ff8_84}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_83[0]));
        traces[105][row] = low_16_bits_col105;
        m31 high_16_bits_col106 = add(add(mul((memory_id_to_big_value_tmp_92ff8_83[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_83[2]) * (M31_4))), (m31{tmp_92ff8_84}));
        traces[106][row] = high_16_bits_col106;
        uint32_t expected_word_tmp_92ff8_85 = low_16_bits_col105 + (high_16_bits_col106 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_86 = ((expected_word_tmp_92ff8_85 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col107 = m31{low_7_ms_bits_tmp_92ff8_86};
        traces[107][row] = low_7_ms_bits_col107;
        uint16_t high_14_ms_bits_tmp_92ff8_87 = ((expected_word_tmp_92ff8_85 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col108 = m31{high_14_ms_bits_tmp_92ff8_87};
        traces[108][row] = high_14_ms_bits_col108;
        uint16_t high_5_ms_bits_tmp_92ff8_88 = ((high_14_ms_bits_tmp_92ff8_87) >> (UInt16_9));
        m31 high_5_ms_bits_col109 = m31{high_5_ms_bits_tmp_92ff8_88};
        traces[109][row] = high_5_ms_bits_col109;
        sub_component_inputs_range_check_7_2_5[9 * 3 + 0][row] = low_7_ms_bits_col107;
        sub_component_inputs_range_check_7_2_5[9 * 3 + 1][row] = sub((high_16_bits_col106), mul((high_14_ms_bits_col108), (M31_4)));
        sub_component_inputs_range_check_7_2_5[9 * 3 + 2][row] = high_5_ms_bits_col109;
        lookup_range_check_7_2_5_9[0][row] = low_7_ms_bits_col107;
        lookup_range_check_7_2_5_9[1][row] = sub((high_16_bits_col106), mul((high_14_ms_bits_col108), (M31_4)));
        lookup_range_check_7_2_5_9[2][row] = high_5_ms_bits_col109;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_89 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_9_col44)),
            &memory_address_to_id_value_tmp_92ff8_89
        );
        m31 message_word_9_id_col110 = memory_address_to_id_value_tmp_92ff8_89;
        traces[110][row] = message_word_9_id_col110;
        sub_component_inputs_memory_address_to_id[9][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_9_col44));
        lookup_memory_address_to_id_9[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_9_col44));
        lookup_memory_address_to_id_9[1][row] = message_word_9_id_col110;
        sub_component_inputs_memory_id_to_big[9][row] = message_word_9_id_col110;
        lookup_memory_id_to_big_9[0][row] = message_word_9_id_col110;
        lookup_memory_id_to_big_9[1][row] = sub((low_16_bits_col105), mul((low_7_ms_bits_col107), (M31_512)));
        lookup_memory_id_to_big_9[2][row] = add((low_7_ms_bits_col107), mul(sub((high_16_bits_col106), mul((high_14_ms_bits_col108), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_9[3][row] = sub((high_14_ms_bits_col108), mul((high_5_ms_bits_col109), (M31_512)));
        lookup_memory_id_to_big_9[4][row] = high_5_ms_bits_col109;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_9[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_90 = expected_word_tmp_92ff8_85;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_91 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_10_col45)),
            &memory_address_to_id_value_tmp_92ff8_91
        );
        m31 memory_id_to_big_value_tmp_92ff8_92[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_91,
            memory_id_to_big_value_tmp_92ff8_92
        );
        uint16_t tmp_92ff8_93 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_92[1])) >> (UInt16_7));
        m31 low_16_bits_col111 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_92[1]), mul((m31{tmp_92ff8_93}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_92[0]));
        traces[111][row] = low_16_bits_col111;
        m31 high_16_bits_col112 = add(add(mul((memory_id_to_big_value_tmp_92ff8_92[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_92[2]) * (M31_4))), (m31{tmp_92ff8_93}));
        traces[112][row] = high_16_bits_col112;
        uint32_t expected_word_tmp_92ff8_94 = low_16_bits_col111 + (high_16_bits_col112 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_95 = ((expected_word_tmp_92ff8_94 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col113 = m31{low_7_ms_bits_tmp_92ff8_95};
        traces[113][row] = low_7_ms_bits_col113;
        uint16_t high_14_ms_bits_tmp_92ff8_96 = ((expected_word_tmp_92ff8_94 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col114 = m31{high_14_ms_bits_tmp_92ff8_96};
        traces[114][row] = high_14_ms_bits_col114;
        uint16_t high_5_ms_bits_tmp_92ff8_97 = ((high_14_ms_bits_tmp_92ff8_96) >> (UInt16_9));
        m31 high_5_ms_bits_col115 = m31{high_5_ms_bits_tmp_92ff8_97};
        traces[115][row] = high_5_ms_bits_col115;
        sub_component_inputs_range_check_7_2_5[10 * 3 + 0][row] = low_7_ms_bits_col113;
        sub_component_inputs_range_check_7_2_5[10 * 3 + 1][row] = sub((high_16_bits_col112), mul((high_14_ms_bits_col114), (M31_4)));
        sub_component_inputs_range_check_7_2_5[10 * 3 + 2][row] = high_5_ms_bits_col115;
        lookup_range_check_7_2_5_10[0][row] = low_7_ms_bits_col113;
        lookup_range_check_7_2_5_10[1][row] = sub((high_16_bits_col112), mul((high_14_ms_bits_col114), (M31_4)));
        lookup_range_check_7_2_5_10[2][row] = high_5_ms_bits_col115;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_98 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_10_col45)),
            &memory_address_to_id_value_tmp_92ff8_98
        );
        m31 message_word_10_id_col116 = memory_address_to_id_value_tmp_92ff8_98;
        traces[116][row] = message_word_10_id_col116;
        sub_component_inputs_memory_address_to_id[10][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_10_col45));
        lookup_memory_address_to_id_10[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_10_col45));
        lookup_memory_address_to_id_10[1][row] = message_word_10_id_col116;
        sub_component_inputs_memory_id_to_big[10][row] = message_word_10_id_col116;
        lookup_memory_id_to_big_10[0][row] = message_word_10_id_col116;
        lookup_memory_id_to_big_10[1][row] = sub((low_16_bits_col111), mul((low_7_ms_bits_col113), (M31_512)));
        lookup_memory_id_to_big_10[2][row] = add((low_7_ms_bits_col113), mul(sub((high_16_bits_col112), mul((high_14_ms_bits_col114), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_10[3][row] = sub((high_14_ms_bits_col114), mul((high_5_ms_bits_col115), (M31_512)));
        lookup_memory_id_to_big_10[4][row] = high_5_ms_bits_col115;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_10[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_99 = expected_word_tmp_92ff8_94;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_100 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_11_col46)),
            &memory_address_to_id_value_tmp_92ff8_100
        );
        m31 memory_id_to_big_value_tmp_92ff8_101[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_100,
            memory_id_to_big_value_tmp_92ff8_101
        );
        uint16_t tmp_92ff8_102 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_101[1])) >> (UInt16_7));
        m31 low_16_bits_col117 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_101[1]), mul((m31{tmp_92ff8_102}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_101[0]));
        traces[117][row] = low_16_bits_col117;
        m31 high_16_bits_col118 = add(add(mul((memory_id_to_big_value_tmp_92ff8_101[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_101[2]) * (M31_4))), (m31{tmp_92ff8_102}));
        traces[118][row] = high_16_bits_col118;
        uint32_t expected_word_tmp_92ff8_103 = low_16_bits_col117 + (high_16_bits_col118 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_104 = ((expected_word_tmp_92ff8_103 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col119 = m31{low_7_ms_bits_tmp_92ff8_104};
        traces[119][row] = low_7_ms_bits_col119;
        uint16_t high_14_ms_bits_tmp_92ff8_105 = ((expected_word_tmp_92ff8_103 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col120 = m31{high_14_ms_bits_tmp_92ff8_105};
        traces[120][row] = high_14_ms_bits_col120;
        uint16_t high_5_ms_bits_tmp_92ff8_106 = ((high_14_ms_bits_tmp_92ff8_105) >> (UInt16_9));
        m31 high_5_ms_bits_col121 = m31{high_5_ms_bits_tmp_92ff8_106};
        traces[121][row] = high_5_ms_bits_col121;
        sub_component_inputs_range_check_7_2_5[11 * 3 + 0][row] = low_7_ms_bits_col119;
        sub_component_inputs_range_check_7_2_5[11 * 3 + 1][row] = sub((high_16_bits_col118), mul((high_14_ms_bits_col120), (M31_4)));
        sub_component_inputs_range_check_7_2_5[11 * 3 + 2][row] = high_5_ms_bits_col121;
        lookup_range_check_7_2_5_11[0][row] = low_7_ms_bits_col119;
        lookup_range_check_7_2_5_11[1][row] = sub((high_16_bits_col118), mul((high_14_ms_bits_col120), (M31_4)));
        lookup_range_check_7_2_5_11[2][row] = high_5_ms_bits_col121;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_107 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_11_col46)),
            &memory_address_to_id_value_tmp_92ff8_107
        );
        m31 message_word_11_id_col122 = memory_address_to_id_value_tmp_92ff8_107;
        traces[122][row] = message_word_11_id_col122;
        sub_component_inputs_memory_address_to_id[11][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_11_col46));
        lookup_memory_address_to_id_11[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_11_col46));
        lookup_memory_address_to_id_11[1][row] = message_word_11_id_col122;
        sub_component_inputs_memory_id_to_big[11][row] = message_word_11_id_col122;
        lookup_memory_id_to_big_11[0][row] = message_word_11_id_col122;
        lookup_memory_id_to_big_11[1][row] = sub((low_16_bits_col117), mul((low_7_ms_bits_col119), (M31_512)));
        lookup_memory_id_to_big_11[2][row] = add((low_7_ms_bits_col119), mul(sub((high_16_bits_col118), mul((high_14_ms_bits_col120), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_11[3][row] = sub((high_14_ms_bits_col120), mul((high_5_ms_bits_col121), (M31_512)));
        lookup_memory_id_to_big_11[4][row] = high_5_ms_bits_col121;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_11[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_108 = expected_word_tmp_92ff8_103;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_109 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_12_col47)),
            &memory_address_to_id_value_tmp_92ff8_109
        );
        m31 memory_id_to_big_value_tmp_92ff8_110[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_109,
            memory_id_to_big_value_tmp_92ff8_110
        );
        uint16_t tmp_92ff8_111 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_110[1])) >> (UInt16_7));
        m31 low_16_bits_col123 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_110[1]), mul((m31{tmp_92ff8_111}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_110[0]));
        traces[123][row] = low_16_bits_col123;
        m31 high_16_bits_col124 = add(add(mul((memory_id_to_big_value_tmp_92ff8_110[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_110[2]) * (M31_4))), (m31{tmp_92ff8_111}));
        traces[124][row] = high_16_bits_col124;
        uint32_t expected_word_tmp_92ff8_112 = low_16_bits_col123 + (high_16_bits_col124 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_113 = ((expected_word_tmp_92ff8_112 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col125 = m31{low_7_ms_bits_tmp_92ff8_113};
        traces[125][row] = low_7_ms_bits_col125;
        uint16_t high_14_ms_bits_tmp_92ff8_114 = ((expected_word_tmp_92ff8_112 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col126 = m31{high_14_ms_bits_tmp_92ff8_114};
        traces[126][row] = high_14_ms_bits_col126;
        uint16_t high_5_ms_bits_tmp_92ff8_115 = ((high_14_ms_bits_tmp_92ff8_114) >> (UInt16_9));
        m31 high_5_ms_bits_col127 = m31{high_5_ms_bits_tmp_92ff8_115};
        traces[127][row] = high_5_ms_bits_col127;
        sub_component_inputs_range_check_7_2_5[12 * 3 + 0][row] = low_7_ms_bits_col125;
        sub_component_inputs_range_check_7_2_5[12 * 3 + 1][row] = sub((high_16_bits_col124), mul((high_14_ms_bits_col126), (M31_4)));
        sub_component_inputs_range_check_7_2_5[12 * 3 + 2][row] = high_5_ms_bits_col127;
        lookup_range_check_7_2_5_12[0][row] = low_7_ms_bits_col125;
        lookup_range_check_7_2_5_12[1][row] = sub((high_16_bits_col124), mul((high_14_ms_bits_col126), (M31_4)));
        lookup_range_check_7_2_5_12[2][row] = high_5_ms_bits_col127;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_116 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_12_col47)),
            &memory_address_to_id_value_tmp_92ff8_116
        );
        m31 message_word_12_id_col128 = memory_address_to_id_value_tmp_92ff8_116;
        traces[128][row] = message_word_12_id_col128;
        sub_component_inputs_memory_address_to_id[12][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_12_col47));
        lookup_memory_address_to_id_12[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_12_col47));
        lookup_memory_address_to_id_12[1][row] = message_word_12_id_col128;
        sub_component_inputs_memory_id_to_big[12][row] = message_word_12_id_col128;
        lookup_memory_id_to_big_12[0][row] = message_word_12_id_col128;
        lookup_memory_id_to_big_12[1][row] = sub((low_16_bits_col123), mul((low_7_ms_bits_col125), (M31_512)));
        lookup_memory_id_to_big_12[2][row] = add((low_7_ms_bits_col125), mul(sub((high_16_bits_col124), mul((high_14_ms_bits_col126), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_12[3][row] = sub((high_14_ms_bits_col126), mul((high_5_ms_bits_col127), (M31_512)));
        lookup_memory_id_to_big_12[4][row] = high_5_ms_bits_col127;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_12[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_117 = expected_word_tmp_92ff8_112;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_118 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_13_col48)),
            &memory_address_to_id_value_tmp_92ff8_118
        );
        m31 memory_id_to_big_value_tmp_92ff8_119[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_118,
            memory_id_to_big_value_tmp_92ff8_119
        );
        uint16_t tmp_92ff8_120 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_119[1])) >> (UInt16_7));
        m31 low_16_bits_col129 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_119[1]), mul((m31{tmp_92ff8_120}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_119[0]));
        traces[129][row] = low_16_bits_col129;
        m31 high_16_bits_col130 = add(add(mul((memory_id_to_big_value_tmp_92ff8_119[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_119[2]) * (M31_4))), (m31{tmp_92ff8_120}));
        traces[130][row] = high_16_bits_col130;
        uint32_t expected_word_tmp_92ff8_121 = low_16_bits_col129 + (high_16_bits_col130 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_122 = ((expected_word_tmp_92ff8_121 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col131 = m31{low_7_ms_bits_tmp_92ff8_122};
        traces[131][row] = low_7_ms_bits_col131;
        uint16_t high_14_ms_bits_tmp_92ff8_123 = ((expected_word_tmp_92ff8_121 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col132 = m31{high_14_ms_bits_tmp_92ff8_123};
        traces[132][row] = high_14_ms_bits_col132;
        uint16_t high_5_ms_bits_tmp_92ff8_124 = ((high_14_ms_bits_tmp_92ff8_123) >> (UInt16_9));
        m31 high_5_ms_bits_col133 = m31{high_5_ms_bits_tmp_92ff8_124};
        traces[133][row] = high_5_ms_bits_col133;
        sub_component_inputs_range_check_7_2_5[13 * 3 + 0][row] = low_7_ms_bits_col131;
        sub_component_inputs_range_check_7_2_5[13 * 3 + 1][row] = sub((high_16_bits_col130), mul((high_14_ms_bits_col132), (M31_4)));
        sub_component_inputs_range_check_7_2_5[13 * 3 + 2][row] = high_5_ms_bits_col133;
        lookup_range_check_7_2_5_13[0][row] = low_7_ms_bits_col131;
        lookup_range_check_7_2_5_13[1][row] = sub((high_16_bits_col130), mul((high_14_ms_bits_col132), (M31_4)));
        lookup_range_check_7_2_5_13[2][row] = high_5_ms_bits_col133;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_125 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_13_col48)),
            &memory_address_to_id_value_tmp_92ff8_125
        );
        m31 message_word_13_id_col134 = memory_address_to_id_value_tmp_92ff8_125;
        traces[134][row] = message_word_13_id_col134;
        sub_component_inputs_memory_address_to_id[13][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_13_col48));
        lookup_memory_address_to_id_13[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_13_col48));
        lookup_memory_address_to_id_13[1][row] = message_word_13_id_col134;
        sub_component_inputs_memory_id_to_big[13][row] = message_word_13_id_col134;
        lookup_memory_id_to_big_13[0][row] = message_word_13_id_col134;
        lookup_memory_id_to_big_13[1][row] = sub((low_16_bits_col129), mul((low_7_ms_bits_col131), (M31_512)));
        lookup_memory_id_to_big_13[2][row] = add((low_7_ms_bits_col131), mul(sub((high_16_bits_col130), mul((high_14_ms_bits_col132), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_13[3][row] = sub((high_14_ms_bits_col132), mul((high_5_ms_bits_col133), (M31_512)));
        lookup_memory_id_to_big_13[4][row] = high_5_ms_bits_col133;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_13[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_126 = expected_word_tmp_92ff8_121;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_127 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_14_col49)),
            &memory_address_to_id_value_tmp_92ff8_127
        );
        m31 memory_id_to_big_value_tmp_92ff8_128[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_127,
            memory_id_to_big_value_tmp_92ff8_128
        );
        uint16_t tmp_92ff8_129 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_128[1])) >> (UInt16_7));
        m31 low_16_bits_col135 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_128[1]), mul((m31{tmp_92ff8_129}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_128[0]));
        traces[135][row] = low_16_bits_col135;
        m31 high_16_bits_col136 = add(add(mul((memory_id_to_big_value_tmp_92ff8_128[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_128[2]) * (M31_4))), (m31{tmp_92ff8_129}));
        traces[136][row] = high_16_bits_col136;
        uint32_t expected_word_tmp_92ff8_130 = low_16_bits_col135 + (high_16_bits_col136 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_131 = ((expected_word_tmp_92ff8_130 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col137 = m31{low_7_ms_bits_tmp_92ff8_131};
        traces[137][row] = low_7_ms_bits_col137;
        uint16_t high_14_ms_bits_tmp_92ff8_132 = ((expected_word_tmp_92ff8_130 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col138 = m31{high_14_ms_bits_tmp_92ff8_132};
        traces[138][row] = high_14_ms_bits_col138;
        uint16_t high_5_ms_bits_tmp_92ff8_133 = ((high_14_ms_bits_tmp_92ff8_132) >> (UInt16_9));
        m31 high_5_ms_bits_col139 = m31{high_5_ms_bits_tmp_92ff8_133};
        traces[139][row] = high_5_ms_bits_col139;
        sub_component_inputs_range_check_7_2_5[14 * 3 + 0][row] = low_7_ms_bits_col137;
        sub_component_inputs_range_check_7_2_5[14 * 3 + 1][row] = sub((high_16_bits_col136), mul((high_14_ms_bits_col138), (M31_4)));
        sub_component_inputs_range_check_7_2_5[14 * 3 + 2][row] = high_5_ms_bits_col139;
        lookup_range_check_7_2_5_14[0][row] = low_7_ms_bits_col137;
        lookup_range_check_7_2_5_14[1][row] = sub((high_16_bits_col136), mul((high_14_ms_bits_col138), (M31_4)));
        lookup_range_check_7_2_5_14[2][row] = high_5_ms_bits_col139;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_134 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_14_col49)),
            &memory_address_to_id_value_tmp_92ff8_134
        );
        m31 message_word_14_id_col140 = memory_address_to_id_value_tmp_92ff8_134;
        traces[140][row] = message_word_14_id_col140;
        sub_component_inputs_memory_address_to_id[14][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_14_col49));
        lookup_memory_address_to_id_14[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_14_col49));
        lookup_memory_address_to_id_14[1][row] = message_word_14_id_col140;
        sub_component_inputs_memory_id_to_big[14][row] = message_word_14_id_col140;
        lookup_memory_id_to_big_14[0][row] = message_word_14_id_col140;
        lookup_memory_id_to_big_14[1][row] = sub((low_16_bits_col135), mul((low_7_ms_bits_col137), (M31_512)));
        lookup_memory_id_to_big_14[2][row] = add((low_7_ms_bits_col137), mul(sub((high_16_bits_col136), mul((high_14_ms_bits_col138), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_14[3][row] = sub((high_14_ms_bits_col138), mul((high_5_ms_bits_col139), (M31_512)));
        lookup_memory_id_to_big_14[4][row] = high_5_ms_bits_col139;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_14[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_135 = expected_word_tmp_92ff8_130;

        // Read Blake Word.
        m31 memory_address_to_id_value_tmp_92ff8_136 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_15_col50)),
            &memory_address_to_id_value_tmp_92ff8_136
        );
        m31 memory_id_to_big_value_tmp_92ff8_137[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_92ff8_136,
            memory_id_to_big_value_tmp_92ff8_137
        );
        uint16_t tmp_92ff8_138 = (((uint16_t)(memory_id_to_big_value_tmp_92ff8_137[1])) >> (UInt16_7));
        m31 low_16_bits_col141 = add(mul(sub((memory_id_to_big_value_tmp_92ff8_137[1]), mul((m31{tmp_92ff8_138}), (M31_128))), (M31_512)), (memory_id_to_big_value_tmp_92ff8_137[0]));
        traces[141][row] = low_16_bits_col141;
        m31 high_16_bits_col142 = add(add(mul((memory_id_to_big_value_tmp_92ff8_137[3]), (M31_2048)), ((memory_id_to_big_value_tmp_92ff8_137[2]) * (M31_4))), (m31{tmp_92ff8_138}));
        traces[142][row] = high_16_bits_col142;
        uint32_t expected_word_tmp_92ff8_139 = low_16_bits_col141 + (high_16_bits_col142 << 16);

        // Verify Blake Word.
        uint16_t low_7_ms_bits_tmp_92ff8_140 = ((expected_word_tmp_92ff8_139 & 0xFFFF) >> (UInt16_9));
        m31 low_7_ms_bits_col143 = m31{low_7_ms_bits_tmp_92ff8_140};
        traces[143][row] = low_7_ms_bits_col143;
        uint16_t high_14_ms_bits_tmp_92ff8_141 = ((expected_word_tmp_92ff8_139 >> 16) >> (UInt16_2));
        m31 high_14_ms_bits_col144 = m31{high_14_ms_bits_tmp_92ff8_141};
        traces[144][row] = high_14_ms_bits_col144;
        uint16_t high_5_ms_bits_tmp_92ff8_142 = ((high_14_ms_bits_tmp_92ff8_141) >> (UInt16_9));
        m31 high_5_ms_bits_col145 = m31{high_5_ms_bits_tmp_92ff8_142};
        traces[145][row] = high_5_ms_bits_col145;
        sub_component_inputs_range_check_7_2_5[15 * 3 + 0][row] = low_7_ms_bits_col143;
        sub_component_inputs_range_check_7_2_5[15 * 3 + 1][row] = sub((high_16_bits_col142), mul((high_14_ms_bits_col144), (M31_4)));
        sub_component_inputs_range_check_7_2_5[15 * 3 + 2][row] = high_5_ms_bits_col145;
        lookup_range_check_7_2_5_15[0][row] = low_7_ms_bits_col143;
        lookup_range_check_7_2_5_15[1][row] = sub((high_16_bits_col142), mul((high_14_ms_bits_col144), (M31_4)));
        lookup_range_check_7_2_5_15[2][row] = high_5_ms_bits_col145;

        // Mem Verify.
        m31 memory_address_to_id_value_tmp_92ff8_143 = m31{0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((input_limb_34_col34), (blake_round_sigma_output_limb_15_col50)),
            &memory_address_to_id_value_tmp_92ff8_143
        );
        m31 message_word_15_id_col146 = memory_address_to_id_value_tmp_92ff8_143;
        traces[146][row] = message_word_15_id_col146;
        sub_component_inputs_memory_address_to_id[15][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_15_col50));
        lookup_memory_address_to_id_15[0][row] = add((input_limb_34_col34), (blake_round_sigma_output_limb_15_col50));
        lookup_memory_address_to_id_15[1][row] = message_word_15_id_col146;
        sub_component_inputs_memory_id_to_big[15][row] = message_word_15_id_col146;
        lookup_memory_id_to_big_15[0][row] = message_word_15_id_col146;
        lookup_memory_id_to_big_15[1][row] = sub((low_16_bits_col141), mul((low_7_ms_bits_col143), (M31_512)));
        lookup_memory_id_to_big_15[2][row] = add((low_7_ms_bits_col143), mul(sub((high_16_bits_col142), mul((high_14_ms_bits_col144), (M31_4))), (M31_128)));
        lookup_memory_id_to_big_15[3][row] = sub((high_14_ms_bits_col144), mul((high_5_ms_bits_col145), (M31_512)));
        lookup_memory_id_to_big_15[4][row] = high_5_ms_bits_col145;
        for (int i = 5; i < 29; ++i) lookup_memory_id_to_big_15[i][row] = M31_0;

        uint32_t read_blake_word_output_tmp_92ff8_144 = expected_word_tmp_92ff8_139;

        sub_component_inputs_blake_g[0 * 6 + 0][row] = blake_round_input[2 + 0][row];
        sub_component_inputs_blake_g[0 * 6 + 1][row] = blake_round_input[2 + 4][row];
        sub_component_inputs_blake_g[0 * 6 + 2][row] = blake_round_input[2 + 8][row];
        sub_component_inputs_blake_g[0 * 6 + 3][row] = blake_round_input[2 + 12][row];
        sub_component_inputs_blake_g[0 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_9;
        sub_component_inputs_blake_g[0 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_18;

        uint32_t blake_g_output_tmp_92ff8_145[4] = {0};
        uint32_t blake_g_input_tmp[6] = {blake_round_input[2 + 0][row], blake_round_input[2 + 4][row], blake_round_input[2 + 8][row], blake_round_input[2 + 12][row], read_blake_word_output_tmp_92ff8_9, read_blake_word_output_tmp_92ff8_18};
        blake_deduce_output(blake_g_input_tmp, blake_g_output_tmp_92ff8_145);
        m31 blake_g_output_limb_0_col147 = m31 {low_as_m31(blake_g_output_tmp_92ff8_145[0])};
        traces[147][row] = blake_g_output_limb_0_col147;
        m31 blake_g_output_limb_1_col148 = m31 {high_as_m31(blake_g_output_tmp_92ff8_145[0])};
        traces[148][row] = blake_g_output_limb_1_col148;
        m31 blake_g_output_limb_2_col149 = m31 {low_as_m31(blake_g_output_tmp_92ff8_145[1])};
        traces[149][row] = blake_g_output_limb_2_col149;
        m31 blake_g_output_limb_3_col150 = m31 {high_as_m31(blake_g_output_tmp_92ff8_145[1])};
        traces[150][row] = blake_g_output_limb_3_col150;
        m31 blake_g_output_limb_4_col151 = m31 {low_as_m31(blake_g_output_tmp_92ff8_145[2])};
        traces[151][row] = blake_g_output_limb_4_col151;
        m31 blake_g_output_limb_5_col152 = m31 {high_as_m31(blake_g_output_tmp_92ff8_145[2])};
        traces[152][row] = blake_g_output_limb_5_col152;
        m31 blake_g_output_limb_6_col153 = m31 {low_as_m31(blake_g_output_tmp_92ff8_145[3])};
        traces[153][row] = blake_g_output_limb_6_col153;
        m31 blake_g_output_limb_7_col154 = m31 {high_as_m31(blake_g_output_tmp_92ff8_145[3])};
        traces[154][row] = blake_g_output_limb_7_col154;

        lookup_blake_g_0[0][row] = input_limb_2_col2;
        lookup_blake_g_0[1][row] = input_limb_3_col3;
        lookup_blake_g_0[2][row] = input_limb_10_col10;
        lookup_blake_g_0[3][row] = input_limb_11_col11;
        lookup_blake_g_0[4][row] = input_limb_18_col18;
        lookup_blake_g_0[5][row] = input_limb_19_col19;
        lookup_blake_g_0[6][row] = input_limb_26_col26;
        lookup_blake_g_0[7][row] = input_limb_27_col27;
        lookup_blake_g_0[8][row] = low_16_bits_col51;
        lookup_blake_g_0[9][row] = high_16_bits_col52;
        lookup_blake_g_0[10][row] = low_16_bits_col57;
        lookup_blake_g_0[11][row] = high_16_bits_col58;
        lookup_blake_g_0[12][row] = blake_g_output_limb_0_col147;
        lookup_blake_g_0[13][row] = blake_g_output_limb_1_col148;
        lookup_blake_g_0[14][row] = blake_g_output_limb_2_col149;
        lookup_blake_g_0[15][row] = blake_g_output_limb_3_col150;
        lookup_blake_g_0[16][row] = blake_g_output_limb_4_col151;
        lookup_blake_g_0[17][row] = blake_g_output_limb_5_col152;
        lookup_blake_g_0[18][row] = blake_g_output_limb_6_col153;
        lookup_blake_g_0[19][row] = blake_g_output_limb_7_col154;

        sub_component_inputs_blake_g[1 * 6 + 0][row] = blake_round_input[2 + 1][row];
        sub_component_inputs_blake_g[1 * 6 + 1][row] = blake_round_input[2 + 5][row];
        sub_component_inputs_blake_g[1 * 6 + 2][row] = blake_round_input[2 + 9][row];
        sub_component_inputs_blake_g[1 * 6 + 3][row] = blake_round_input[2 + 13][row];
        sub_component_inputs_blake_g[1 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_27;
        sub_component_inputs_blake_g[1 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_36;

        uint32_t blake_g_output_tmp_92ff8_146[4] = {0};
        uint32_t blake_g_input_tmp_1[6] = {
            blake_round_input[2 + 1][row], blake_round_input[2 + 5][row], blake_round_input[2 + 9][row], blake_round_input[2 + 13][row],
            read_blake_word_output_tmp_92ff8_27, read_blake_word_output_tmp_92ff8_36
        };
        blake_deduce_output(blake_g_input_tmp_1, blake_g_output_tmp_92ff8_146);

        m31 blake_g_output_limb_0_col155 = m31{low_as_m31(blake_g_output_tmp_92ff8_146[0])};
        traces[155][row] = blake_g_output_limb_0_col155;
        m31 blake_g_output_limb_1_col156 = m31{high_as_m31(blake_g_output_tmp_92ff8_146[0])};
        traces[156][row] = blake_g_output_limb_1_col156;
        m31 blake_g_output_limb_2_col157 = m31{low_as_m31(blake_g_output_tmp_92ff8_146[1])};
        traces[157][row] = blake_g_output_limb_2_col157;
        m31 blake_g_output_limb_3_col158 = m31{high_as_m31(blake_g_output_tmp_92ff8_146[1])};
        traces[158][row] = blake_g_output_limb_3_col158;
        m31 blake_g_output_limb_4_col159 = m31{low_as_m31(blake_g_output_tmp_92ff8_146[2])};
        traces[159][row] = blake_g_output_limb_4_col159;
        m31 blake_g_output_limb_5_col160 = m31{high_as_m31(blake_g_output_tmp_92ff8_146[2])};
        traces[160][row] = blake_g_output_limb_5_col160;
        m31 blake_g_output_limb_6_col161 = m31{low_as_m31(blake_g_output_tmp_92ff8_146[3])};
        traces[161][row] = blake_g_output_limb_6_col161;
        m31 blake_g_output_limb_7_col162 = m31{high_as_m31(blake_g_output_tmp_92ff8_146[3])};
        traces[162][row] = blake_g_output_limb_7_col162;

        lookup_blake_g_1[0][row] = input_limb_4_col4;
        lookup_blake_g_1[1][row] = input_limb_5_col5;
        lookup_blake_g_1[2][row] = input_limb_12_col12;
        lookup_blake_g_1[3][row] = input_limb_13_col13;
        lookup_blake_g_1[4][row] = input_limb_20_col20;
        lookup_blake_g_1[5][row] = input_limb_21_col21;
        lookup_blake_g_1[6][row] = input_limb_28_col28;
        lookup_blake_g_1[7][row] = input_limb_29_col29;
        lookup_blake_g_1[8][row] = low_16_bits_col63;
        lookup_blake_g_1[9][row] = high_16_bits_col64;
        lookup_blake_g_1[10][row] = low_16_bits_col69;
        lookup_blake_g_1[11][row] = high_16_bits_col70;
        lookup_blake_g_1[12][row] = blake_g_output_limb_0_col155;
        lookup_blake_g_1[13][row] = blake_g_output_limb_1_col156;
        lookup_blake_g_1[14][row] = blake_g_output_limb_2_col157;
        lookup_blake_g_1[15][row] = blake_g_output_limb_3_col158;
        lookup_blake_g_1[16][row] = blake_g_output_limb_4_col159;
        lookup_blake_g_1[17][row] = blake_g_output_limb_5_col160;
        lookup_blake_g_1[18][row] = blake_g_output_limb_6_col161;
        lookup_blake_g_1[19][row] = blake_g_output_limb_7_col162;

        sub_component_inputs_blake_g[2 * 6 + 0][row] = blake_round_input[2 + 2][row];
        sub_component_inputs_blake_g[2 * 6 + 1][row] = blake_round_input[2 + 6][row];
        sub_component_inputs_blake_g[2 * 6 + 2][row] = blake_round_input[2 + 10][row];
        sub_component_inputs_blake_g[2 * 6 + 3][row] = blake_round_input[2 + 14][row];
        sub_component_inputs_blake_g[2 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_45;
        sub_component_inputs_blake_g[2 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_54;

        uint32_t blake_g_output_tmp_92ff8_147[4] = {0};
        uint32_t blake_g_input_tmp_2[6] = {
            blake_round_input[2 + 2][row], blake_round_input[2 + 6][row], blake_round_input[2 + 10][row], blake_round_input[2 + 14][row],
            read_blake_word_output_tmp_92ff8_45, read_blake_word_output_tmp_92ff8_54
        };
        blake_deduce_output(blake_g_input_tmp_2, blake_g_output_tmp_92ff8_147);

        m31 blake_g_output_limb_0_col163 = m31{low_as_m31(blake_g_output_tmp_92ff8_147[0])};
        traces[163][row] = blake_g_output_limb_0_col163;
        m31 blake_g_output_limb_1_col164 = m31{high_as_m31(blake_g_output_tmp_92ff8_147[0])};
        traces[164][row] = blake_g_output_limb_1_col164;
        m31 blake_g_output_limb_2_col165 = m31{low_as_m31(blake_g_output_tmp_92ff8_147[1])};
        traces[165][row] = blake_g_output_limb_2_col165;
        m31 blake_g_output_limb_3_col166 = m31{high_as_m31(blake_g_output_tmp_92ff8_147[1])};
        traces[166][row] = blake_g_output_limb_3_col166;
        m31 blake_g_output_limb_4_col167 = m31{low_as_m31(blake_g_output_tmp_92ff8_147[2])};
        traces[167][row] = blake_g_output_limb_4_col167;
        m31 blake_g_output_limb_5_col168 = m31{high_as_m31(blake_g_output_tmp_92ff8_147[2])};
        traces[168][row] = blake_g_output_limb_5_col168;
        m31 blake_g_output_limb_6_col169 = m31{low_as_m31(blake_g_output_tmp_92ff8_147[3])};
        traces[169][row] = blake_g_output_limb_6_col169;
        m31 blake_g_output_limb_7_col170 = m31{high_as_m31(blake_g_output_tmp_92ff8_147[3])};
        traces[170][row] = blake_g_output_limb_7_col170;

        lookup_blake_g_2[0][row] = input_limb_6_col6;
        lookup_blake_g_2[1][row] = input_limb_7_col7;
        lookup_blake_g_2[2][row] = input_limb_14_col14;
        lookup_blake_g_2[3][row] = input_limb_15_col15;
        lookup_blake_g_2[4][row] = input_limb_22_col22;
        lookup_blake_g_2[5][row] = input_limb_23_col23;
        lookup_blake_g_2[6][row] = input_limb_30_col30;
        lookup_blake_g_2[7][row] = input_limb_31_col31;
        lookup_blake_g_2[8][row] = low_16_bits_col75;
        lookup_blake_g_2[9][row] = high_16_bits_col76;
        lookup_blake_g_2[10][row] = low_16_bits_col81;
        lookup_blake_g_2[11][row] = high_16_bits_col82;
        lookup_blake_g_2[12][row] = blake_g_output_limb_0_col163;
        lookup_blake_g_2[13][row] = blake_g_output_limb_1_col164;
        lookup_blake_g_2[14][row] = blake_g_output_limb_2_col165;
        lookup_blake_g_2[15][row] = blake_g_output_limb_3_col166;
        lookup_blake_g_2[16][row] = blake_g_output_limb_4_col167;
        lookup_blake_g_2[17][row] = blake_g_output_limb_5_col168;
        lookup_blake_g_2[18][row] = blake_g_output_limb_6_col169;
        lookup_blake_g_2[19][row] = blake_g_output_limb_7_col170;

        sub_component_inputs_blake_g[3 * 6 + 0][row] = blake_round_input[2 + 3][row];
        sub_component_inputs_blake_g[3 * 6 + 1][row] = blake_round_input[2 + 7][row];
        sub_component_inputs_blake_g[3 * 6 + 2][row] = blake_round_input[2 + 11][row];
        sub_component_inputs_blake_g[3 * 6 + 3][row] = blake_round_input[2 + 15][row];
        sub_component_inputs_blake_g[3 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_63;
        sub_component_inputs_blake_g[3 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_72;

        uint32_t blake_g_output_tmp_92ff8_148[4] = {0};
        uint32_t blake_g_input_tmp_3[6] = {
            blake_round_input[2 + 3][row], blake_round_input[2 + 7][row], blake_round_input[2 + 11][row], blake_round_input[2 + 15][row],
            read_blake_word_output_tmp_92ff8_63, read_blake_word_output_tmp_92ff8_72
        };
        blake_deduce_output(blake_g_input_tmp_3, blake_g_output_tmp_92ff8_148);

        m31 blake_g_output_limb_0_col171 = m31{low_as_m31(blake_g_output_tmp_92ff8_148[0])};
        traces[171][row] = blake_g_output_limb_0_col171;
        m31 blake_g_output_limb_1_col172 = m31{high_as_m31(blake_g_output_tmp_92ff8_148[0])};
        traces[172][row] = blake_g_output_limb_1_col172;
        m31 blake_g_output_limb_2_col173 = m31{low_as_m31(blake_g_output_tmp_92ff8_148[1])};
        traces[173][row] = blake_g_output_limb_2_col173;
        m31 blake_g_output_limb_3_col174 = m31{high_as_m31(blake_g_output_tmp_92ff8_148[1])};
        traces[174][row] = blake_g_output_limb_3_col174;
        m31 blake_g_output_limb_4_col175 = m31{low_as_m31(blake_g_output_tmp_92ff8_148[2])};
        traces[175][row] = blake_g_output_limb_4_col175;
        m31 blake_g_output_limb_5_col176 = m31{high_as_m31(blake_g_output_tmp_92ff8_148[2])};
        traces[176][row] = blake_g_output_limb_5_col176;
        m31 blake_g_output_limb_6_col177 = m31{low_as_m31(blake_g_output_tmp_92ff8_148[3])};
        traces[177][row] = blake_g_output_limb_6_col177;
        m31 blake_g_output_limb_7_col178 = m31{high_as_m31(blake_g_output_tmp_92ff8_148[3])};
        traces[178][row] = blake_g_output_limb_7_col178;

        lookup_blake_g_3[0][row] = input_limb_8_col8;
        lookup_blake_g_3[1][row] = input_limb_9_col9;
        lookup_blake_g_3[2][row] = input_limb_16_col16;
        lookup_blake_g_3[3][row] = input_limb_17_col17;
        lookup_blake_g_3[4][row] = input_limb_24_col24;
        lookup_blake_g_3[5][row] = input_limb_25_col25;
        lookup_blake_g_3[6][row] = input_limb_32_col32;
        lookup_blake_g_3[7][row] = input_limb_33_col33;
        lookup_blake_g_3[8][row] = low_16_bits_col87;
        lookup_blake_g_3[9][row] = high_16_bits_col88;
        lookup_blake_g_3[10][row] = low_16_bits_col93;
        lookup_blake_g_3[11][row] = high_16_bits_col94;
        lookup_blake_g_3[12][row] = blake_g_output_limb_0_col171;
        lookup_blake_g_3[13][row] = blake_g_output_limb_1_col172;
        lookup_blake_g_3[14][row] = blake_g_output_limb_2_col173;
        lookup_blake_g_3[15][row] = blake_g_output_limb_3_col174;
        lookup_blake_g_3[16][row] = blake_g_output_limb_4_col175;
        lookup_blake_g_3[17][row] = blake_g_output_limb_5_col176;
        lookup_blake_g_3[18][row] = blake_g_output_limb_6_col177;
        lookup_blake_g_3[19][row] = blake_g_output_limb_7_col178;

        sub_component_inputs_blake_g[4 * 6 + 0][row] = blake_g_output_tmp_92ff8_145[0];
        sub_component_inputs_blake_g[4 * 6 + 1][row] = blake_g_output_tmp_92ff8_146[1];
        sub_component_inputs_blake_g[4 * 6 + 2][row] = blake_g_output_tmp_92ff8_147[2];
        sub_component_inputs_blake_g[4 * 6 + 3][row] = blake_g_output_tmp_92ff8_148[3];
        sub_component_inputs_blake_g[4 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_81;
        sub_component_inputs_blake_g[4 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_90;

        uint32_t blake_g_output_tmp_92ff8_149[4] = {0};
        uint32_t blake_g_input_tmp_4[6] = {
            blake_g_output_tmp_92ff8_145[0], blake_g_output_tmp_92ff8_146[1], blake_g_output_tmp_92ff8_147[2], blake_g_output_tmp_92ff8_148[3],
            read_blake_word_output_tmp_92ff8_81, read_blake_word_output_tmp_92ff8_90
        };
        blake_deduce_output(blake_g_input_tmp_4, blake_g_output_tmp_92ff8_149);

        m31 blake_g_output_limb_0_col179 = m31{low_as_m31(blake_g_output_tmp_92ff8_149[0])};
        traces[179][row] = blake_g_output_limb_0_col179;
        m31 blake_g_output_limb_1_col180 = m31{high_as_m31(blake_g_output_tmp_92ff8_149[0])};
        traces[180][row] = blake_g_output_limb_1_col180;
        m31 blake_g_output_limb_2_col181 = m31{low_as_m31(blake_g_output_tmp_92ff8_149[1])};
        traces[181][row] = blake_g_output_limb_2_col181;
        m31 blake_g_output_limb_3_col182 = m31{high_as_m31(blake_g_output_tmp_92ff8_149[1])};
        traces[182][row] = blake_g_output_limb_3_col182;
        m31 blake_g_output_limb_4_col183 = m31{low_as_m31(blake_g_output_tmp_92ff8_149[2])};
        traces[183][row] = blake_g_output_limb_4_col183;
        m31 blake_g_output_limb_5_col184 = m31{high_as_m31(blake_g_output_tmp_92ff8_149[2])};
        traces[184][row] = blake_g_output_limb_5_col184;
        m31 blake_g_output_limb_6_col185 = m31{low_as_m31(blake_g_output_tmp_92ff8_149[3])};
        traces[185][row] = blake_g_output_limb_6_col185;
        m31 blake_g_output_limb_7_col186 = m31{high_as_m31(blake_g_output_tmp_92ff8_149[3])};
        traces[186][row] = blake_g_output_limb_7_col186;

        lookup_blake_g_4[0][row] = blake_g_output_limb_0_col147;
        lookup_blake_g_4[1][row] = blake_g_output_limb_1_col148;
        lookup_blake_g_4[2][row] = blake_g_output_limb_2_col157;
        lookup_blake_g_4[3][row] = blake_g_output_limb_3_col158;
        lookup_blake_g_4[4][row] = blake_g_output_limb_4_col167;
        lookup_blake_g_4[5][row] = blake_g_output_limb_5_col168;
        lookup_blake_g_4[6][row] = blake_g_output_limb_6_col177;
        lookup_blake_g_4[7][row] = blake_g_output_limb_7_col178;
        lookup_blake_g_4[8][row] = low_16_bits_col99;
        lookup_blake_g_4[9][row] = high_16_bits_col100;
        lookup_blake_g_4[10][row] = low_16_bits_col105;
        lookup_blake_g_4[11][row] = high_16_bits_col106;
        lookup_blake_g_4[12][row] = blake_g_output_limb_0_col179;
        lookup_blake_g_4[13][row] = blake_g_output_limb_1_col180;
        lookup_blake_g_4[14][row] = blake_g_output_limb_2_col181;
        lookup_blake_g_4[15][row] = blake_g_output_limb_3_col182;
        lookup_blake_g_4[16][row] = blake_g_output_limb_4_col183;
        lookup_blake_g_4[17][row] = blake_g_output_limb_5_col184;
        lookup_blake_g_4[18][row] = blake_g_output_limb_6_col185;
        lookup_blake_g_4[19][row] = blake_g_output_limb_7_col186;

        sub_component_inputs_blake_g[5 * 6 + 0][row] = blake_g_output_tmp_92ff8_146[0];
        sub_component_inputs_blake_g[5 * 6 + 1][row] = blake_g_output_tmp_92ff8_147[1];
        sub_component_inputs_blake_g[5 * 6 + 2][row] = blake_g_output_tmp_92ff8_148[2];
        sub_component_inputs_blake_g[5 * 6 + 3][row] = blake_g_output_tmp_92ff8_145[3];
        sub_component_inputs_blake_g[5 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_99;
        sub_component_inputs_blake_g[5 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_108;

        uint32_t blake_g_output_tmp_92ff8_150[4] = {0};
        uint32_t blake_g_input_tmp_5[6] = {
            blake_g_output_tmp_92ff8_146[0], blake_g_output_tmp_92ff8_147[1], blake_g_output_tmp_92ff8_148[2], blake_g_output_tmp_92ff8_145[3],
            read_blake_word_output_tmp_92ff8_99, read_blake_word_output_tmp_92ff8_108
        };
        blake_deduce_output(blake_g_input_tmp_5, blake_g_output_tmp_92ff8_150);

        m31 blake_g_output_limb_0_col187 = m31{low_as_m31(blake_g_output_tmp_92ff8_150[0])};
        traces[187][row] = blake_g_output_limb_0_col187;
        m31 blake_g_output_limb_1_col188 = m31{high_as_m31(blake_g_output_tmp_92ff8_150[0])};
        traces[188][row] = blake_g_output_limb_1_col188;
        m31 blake_g_output_limb_2_col189 = m31{low_as_m31(blake_g_output_tmp_92ff8_150[1])};
        traces[189][row] = blake_g_output_limb_2_col189;
        m31 blake_g_output_limb_3_col190 = m31{high_as_m31(blake_g_output_tmp_92ff8_150[1])};
        traces[190][row] = blake_g_output_limb_3_col190;
        m31 blake_g_output_limb_4_col191 = m31{low_as_m31(blake_g_output_tmp_92ff8_150[2])};
        traces[191][row] = blake_g_output_limb_4_col191;
        m31 blake_g_output_limb_5_col192 = m31{high_as_m31(blake_g_output_tmp_92ff8_150[2])};
        traces[192][row] = blake_g_output_limb_5_col192;
        m31 blake_g_output_limb_6_col193 = m31{low_as_m31(blake_g_output_tmp_92ff8_150[3])};
        traces[193][row] = blake_g_output_limb_6_col193;
        m31 blake_g_output_limb_7_col194 = m31{high_as_m31(blake_g_output_tmp_92ff8_150[3])};
        traces[194][row] = blake_g_output_limb_7_col194;

        lookup_blake_g_5[0][row] = blake_g_output_limb_0_col155;
        lookup_blake_g_5[1][row] = blake_g_output_limb_1_col156;
        lookup_blake_g_5[2][row] = blake_g_output_limb_2_col165;
        lookup_blake_g_5[3][row] = blake_g_output_limb_3_col166;
        lookup_blake_g_5[4][row] = blake_g_output_limb_4_col175;
        lookup_blake_g_5[5][row] = blake_g_output_limb_5_col176;
        lookup_blake_g_5[6][row] = blake_g_output_limb_6_col153;
        lookup_blake_g_5[7][row] = blake_g_output_limb_7_col154;
        lookup_blake_g_5[8][row] = low_16_bits_col111;
        lookup_blake_g_5[9][row] = high_16_bits_col112;
        lookup_blake_g_5[10][row] = low_16_bits_col117;
        lookup_blake_g_5[11][row] = high_16_bits_col118;
        lookup_blake_g_5[12][row] = blake_g_output_limb_0_col187;
        lookup_blake_g_5[13][row] = blake_g_output_limb_1_col188;
        lookup_blake_g_5[14][row] = blake_g_output_limb_2_col189;
        lookup_blake_g_5[15][row] = blake_g_output_limb_3_col190;
        lookup_blake_g_5[16][row] = blake_g_output_limb_4_col191;
        lookup_blake_g_5[17][row] = blake_g_output_limb_5_col192;
        lookup_blake_g_5[18][row] = blake_g_output_limb_6_col193;
        lookup_blake_g_5[19][row] = blake_g_output_limb_7_col194;

        sub_component_inputs_blake_g[6 * 6 + 0][row] = blake_g_output_tmp_92ff8_147[0];
        sub_component_inputs_blake_g[6 * 6 + 1][row] = blake_g_output_tmp_92ff8_148[1];
        sub_component_inputs_blake_g[6 * 6 + 2][row] = blake_g_output_tmp_92ff8_145[2];
        sub_component_inputs_blake_g[6 * 6 + 3][row] = blake_g_output_tmp_92ff8_146[3];
        sub_component_inputs_blake_g[6 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_117;
        sub_component_inputs_blake_g[6 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_126;

        uint32_t blake_g_output_tmp_92ff8_151[4] = {0};
        uint32_t blake_g_input_tmp_6[6] = {
            blake_g_output_tmp_92ff8_147[0], blake_g_output_tmp_92ff8_148[1], blake_g_output_tmp_92ff8_145[2], blake_g_output_tmp_92ff8_146[3],
            read_blake_word_output_tmp_92ff8_117, read_blake_word_output_tmp_92ff8_126
        };
        blake_deduce_output(blake_g_input_tmp_6, blake_g_output_tmp_92ff8_151);

        m31 blake_g_output_limb_0_col195 = m31{low_as_m31(blake_g_output_tmp_92ff8_151[0])};
        traces[195][row] = blake_g_output_limb_0_col195;
        m31 blake_g_output_limb_1_col196 = m31{high_as_m31(blake_g_output_tmp_92ff8_151[0])};
        traces[196][row] = blake_g_output_limb_1_col196;
        m31 blake_g_output_limb_2_col197 = m31{low_as_m31(blake_g_output_tmp_92ff8_151[1])};
        traces[197][row] = blake_g_output_limb_2_col197;
        m31 blake_g_output_limb_3_col198 = m31{high_as_m31(blake_g_output_tmp_92ff8_151[1])};
        traces[198][row] = blake_g_output_limb_3_col198;
        m31 blake_g_output_limb_4_col199 = m31{low_as_m31(blake_g_output_tmp_92ff8_151[2])};
        traces[199][row] = blake_g_output_limb_4_col199;
        m31 blake_g_output_limb_5_col200 = m31{high_as_m31(blake_g_output_tmp_92ff8_151[2])};
        traces[200][row] = blake_g_output_limb_5_col200;
        m31 blake_g_output_limb_6_col201 = m31{low_as_m31(blake_g_output_tmp_92ff8_151[3])};
        traces[201][row] = blake_g_output_limb_6_col201;
        m31 blake_g_output_limb_7_col202 = m31{high_as_m31(blake_g_output_tmp_92ff8_151[3])};
        traces[202][row] = blake_g_output_limb_7_col202;

        lookup_blake_g_6[0][row] = blake_g_output_limb_0_col163;
        lookup_blake_g_6[1][row] = blake_g_output_limb_1_col164;
        lookup_blake_g_6[2][row] = blake_g_output_limb_2_col173;
        lookup_blake_g_6[3][row] = blake_g_output_limb_3_col174;
        lookup_blake_g_6[4][row] = blake_g_output_limb_4_col151;
        lookup_blake_g_6[5][row] = blake_g_output_limb_5_col152;
        lookup_blake_g_6[6][row] = blake_g_output_limb_6_col161;
        lookup_blake_g_6[7][row] = blake_g_output_limb_7_col162;
        lookup_blake_g_6[8][row] = low_16_bits_col123;
        lookup_blake_g_6[9][row] = high_16_bits_col124;
        lookup_blake_g_6[10][row] = low_16_bits_col129;
        lookup_blake_g_6[11][row] = high_16_bits_col130;
        lookup_blake_g_6[12][row] = blake_g_output_limb_0_col195;
        lookup_blake_g_6[13][row] = blake_g_output_limb_1_col196;
        lookup_blake_g_6[14][row] = blake_g_output_limb_2_col197;
        lookup_blake_g_6[15][row] = blake_g_output_limb_3_col198;
        lookup_blake_g_6[16][row] = blake_g_output_limb_4_col199;
        lookup_blake_g_6[17][row] = blake_g_output_limb_5_col200;
        lookup_blake_g_6[18][row] = blake_g_output_limb_6_col201;
        lookup_blake_g_6[19][row] = blake_g_output_limb_7_col202;


        sub_component_inputs_blake_g[7 * 6 + 0][row] = blake_g_output_tmp_92ff8_148[0];
        sub_component_inputs_blake_g[7 * 6 + 1][row] = blake_g_output_tmp_92ff8_145[1];
        sub_component_inputs_blake_g[7 * 6 + 2][row] = blake_g_output_tmp_92ff8_146[2];
        sub_component_inputs_blake_g[7 * 6 + 3][row] = blake_g_output_tmp_92ff8_147[3];
        sub_component_inputs_blake_g[7 * 6 + 4][row] = read_blake_word_output_tmp_92ff8_135;
        sub_component_inputs_blake_g[7 * 6 + 5][row] = read_blake_word_output_tmp_92ff8_144;

        uint32_t blake_g_output_tmp_92ff8_152[4] = {0};
        uint32_t blake_g_input_tmp_7[6] = {
            blake_g_output_tmp_92ff8_148[0], blake_g_output_tmp_92ff8_145[1], blake_g_output_tmp_92ff8_146[2], blake_g_output_tmp_92ff8_147[3],
            read_blake_word_output_tmp_92ff8_135, read_blake_word_output_tmp_92ff8_144
        };
        blake_deduce_output(blake_g_input_tmp_7, blake_g_output_tmp_92ff8_152);

        m31 blake_g_output_limb_0_col203 = m31{low_as_m31(blake_g_output_tmp_92ff8_152[0])};
        traces[203][row] = blake_g_output_limb_0_col203;
        m31 blake_g_output_limb_1_col204 = m31{high_as_m31(blake_g_output_tmp_92ff8_152[0])};
        traces[204][row] = blake_g_output_limb_1_col204;
        m31 blake_g_output_limb_2_col205 = m31{low_as_m31(blake_g_output_tmp_92ff8_152[1])};
        traces[205][row] = blake_g_output_limb_2_col205;
        m31 blake_g_output_limb_3_col206 = m31{high_as_m31(blake_g_output_tmp_92ff8_152[1])};
        traces[206][row] = blake_g_output_limb_3_col206;
        m31 blake_g_output_limb_4_col207 = m31{low_as_m31(blake_g_output_tmp_92ff8_152[2])};
        traces[207][row] = blake_g_output_limb_4_col207;
        m31 blake_g_output_limb_5_col208 = m31{high_as_m31(blake_g_output_tmp_92ff8_152[2])};
        traces[208][row] = blake_g_output_limb_5_col208;
        m31 blake_g_output_limb_6_col209 = m31{low_as_m31(blake_g_output_tmp_92ff8_152[3])};
        traces[209][row] = blake_g_output_limb_6_col209;
        m31 blake_g_output_limb_7_col210 = m31{high_as_m31(blake_g_output_tmp_92ff8_152[3])};
        traces[210][row] = blake_g_output_limb_7_col210;

        lookup_blake_g_7[0][row] = blake_g_output_limb_0_col171;
        lookup_blake_g_7[1][row] = blake_g_output_limb_1_col172;
        lookup_blake_g_7[2][row] = blake_g_output_limb_2_col149;
        lookup_blake_g_7[3][row] = blake_g_output_limb_3_col150;
        lookup_blake_g_7[4][row] = blake_g_output_limb_4_col159;
        lookup_blake_g_7[5][row] = blake_g_output_limb_5_col160;
        lookup_blake_g_7[6][row] = blake_g_output_limb_6_col169;
        lookup_blake_g_7[7][row] = blake_g_output_limb_7_col170;
        lookup_blake_g_7[8][row] = low_16_bits_col135;
        lookup_blake_g_7[9][row] = high_16_bits_col136;
        lookup_blake_g_7[10][row] = low_16_bits_col141;
        lookup_blake_g_7[11][row] = high_16_bits_col142;
        lookup_blake_g_7[12][row] = blake_g_output_limb_0_col203;
        lookup_blake_g_7[13][row] = blake_g_output_limb_1_col204;
        lookup_blake_g_7[14][row] = blake_g_output_limb_2_col205;
        lookup_blake_g_7[15][row] = blake_g_output_limb_3_col206;
        lookup_blake_g_7[16][row] = blake_g_output_limb_4_col207;
        lookup_blake_g_7[17][row] = blake_g_output_limb_5_col208;
        lookup_blake_g_7[18][row] = blake_g_output_limb_6_col209;
        lookup_blake_g_7[19][row] = blake_g_output_limb_7_col210;

        // lookup_blake_round_0
        lookup_blake_round_0[0][row]  = input_limb_0_col0;
        lookup_blake_round_0[1][row]  = input_limb_1_col1;
        lookup_blake_round_0[2][row]  = input_limb_2_col2;
        lookup_blake_round_0[3][row]  = input_limb_3_col3;
        lookup_blake_round_0[4][row]  = input_limb_4_col4;
        lookup_blake_round_0[5][row]  = input_limb_5_col5;
        lookup_blake_round_0[6][row]  = input_limb_6_col6;
        lookup_blake_round_0[7][row]  = input_limb_7_col7;
        lookup_blake_round_0[8][row]  = input_limb_8_col8;
        lookup_blake_round_0[9][row]  = input_limb_9_col9;
        lookup_blake_round_0[10][row] = input_limb_10_col10;
        lookup_blake_round_0[11][row] = input_limb_11_col11;
        lookup_blake_round_0[12][row] = input_limb_12_col12;
        lookup_blake_round_0[13][row] = input_limb_13_col13;
        lookup_blake_round_0[14][row] = input_limb_14_col14;
        lookup_blake_round_0[15][row] = input_limb_15_col15;
        lookup_blake_round_0[16][row] = input_limb_16_col16;
        lookup_blake_round_0[17][row] = input_limb_17_col17;
        lookup_blake_round_0[18][row] = input_limb_18_col18;
        lookup_blake_round_0[19][row] = input_limb_19_col19;
        lookup_blake_round_0[20][row] = input_limb_20_col20;
        lookup_blake_round_0[21][row] = input_limb_21_col21;
        lookup_blake_round_0[22][row] = input_limb_22_col22;
        lookup_blake_round_0[23][row] = input_limb_23_col23;
        lookup_blake_round_0[24][row] = input_limb_24_col24;
        lookup_blake_round_0[25][row] = input_limb_25_col25;
        lookup_blake_round_0[26][row] = input_limb_26_col26;
        lookup_blake_round_0[27][row] = input_limb_27_col27;
        lookup_blake_round_0[28][row] = input_limb_28_col28;
        lookup_blake_round_0[29][row] = input_limb_29_col29;
        lookup_blake_round_0[30][row] = input_limb_30_col30;
        lookup_blake_round_0[31][row] = input_limb_31_col31;
        lookup_blake_round_0[32][row] = input_limb_32_col32;
        lookup_blake_round_0[33][row] = input_limb_33_col33;
        lookup_blake_round_0[34][row] = input_limb_34_col34;

        // lookup_blake_round_1
        lookup_blake_round_1[0][row]  = input_limb_0_col0;
        lookup_blake_round_1[1][row]  = add(input_limb_1_col1, M31_1);
        lookup_blake_round_1[2][row]  = blake_g_output_limb_0_col179;
        lookup_blake_round_1[3][row]  = blake_g_output_limb_1_col180;
        lookup_blake_round_1[4][row]  = blake_g_output_limb_0_col187;
        lookup_blake_round_1[5][row]  = blake_g_output_limb_1_col188;
        lookup_blake_round_1[6][row]  = blake_g_output_limb_0_col195;
        lookup_blake_round_1[7][row]  = blake_g_output_limb_1_col196;
        lookup_blake_round_1[8][row]  = blake_g_output_limb_0_col203;
        lookup_blake_round_1[9][row]  = blake_g_output_limb_1_col204;
        lookup_blake_round_1[10][row] = blake_g_output_limb_2_col205;
        lookup_blake_round_1[11][row] = blake_g_output_limb_3_col206;
        lookup_blake_round_1[12][row] = blake_g_output_limb_2_col181;
        lookup_blake_round_1[13][row] = blake_g_output_limb_3_col182;
        lookup_blake_round_1[14][row] = blake_g_output_limb_2_col189;
        lookup_blake_round_1[15][row] = blake_g_output_limb_3_col190;
        lookup_blake_round_1[16][row] = blake_g_output_limb_2_col197;
        lookup_blake_round_1[17][row] = blake_g_output_limb_3_col198;
        lookup_blake_round_1[18][row] = blake_g_output_limb_4_col199;
        lookup_blake_round_1[19][row] = blake_g_output_limb_5_col200;
        lookup_blake_round_1[20][row] = blake_g_output_limb_4_col207;
        lookup_blake_round_1[21][row] = blake_g_output_limb_5_col208;
        lookup_blake_round_1[22][row] = blake_g_output_limb_4_col183;
        lookup_blake_round_1[23][row] = blake_g_output_limb_5_col184;
        lookup_blake_round_1[24][row] = blake_g_output_limb_4_col191;
        lookup_blake_round_1[25][row] = blake_g_output_limb_5_col192;
        lookup_blake_round_1[26][row] = blake_g_output_limb_6_col193;
        lookup_blake_round_1[27][row] = blake_g_output_limb_7_col194;
        lookup_blake_round_1[28][row] = blake_g_output_limb_6_col201;
        lookup_blake_round_1[29][row] = blake_g_output_limb_7_col202;
        lookup_blake_round_1[30][row] = blake_g_output_limb_6_col209;
        lookup_blake_round_1[31][row] = blake_g_output_limb_7_col210;
        lookup_blake_round_1[32][row] = blake_g_output_limb_6_col185;
        lookup_blake_round_1[33][row] = blake_g_output_limb_7_col186;
        lookup_blake_round_1[34][row] = input_limb_34_col34;

        // Enabler column: 1 for real rows (row < n_rows), 0 for padding rows
        m31 enabler_col = (row < n_rows) ? M31_1 : M31_0;
        traces[211][row] = enabler_col;

    }
}


void generate_blake_round_traces(
    m31 **traces,

    m31 **lookup_blake_g_0,
    m31 **lookup_blake_g_1,
    m31 **lookup_blake_g_2,
    m31 **lookup_blake_g_3,
    m31 **lookup_blake_g_4,
    m31 **lookup_blake_g_5,
    m31 **lookup_blake_g_6,
    m31 **lookup_blake_g_7,
    m31 **lookup_blake_round_0,
    m31 **lookup_blake_round_1,
    m31 **lookup_blake_round_sigma_0,
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,
    m31 **lookup_memory_address_to_id_5,
    m31 **lookup_memory_address_to_id_6,
    m31 **lookup_memory_address_to_id_7,
    m31 **lookup_memory_address_to_id_8,
    m31 **lookup_memory_address_to_id_9,
    m31 **lookup_memory_address_to_id_10,
    m31 **lookup_memory_address_to_id_11,
    m31 **lookup_memory_address_to_id_12,
    m31 **lookup_memory_address_to_id_13,
    m31 **lookup_memory_address_to_id_14,
    m31 **lookup_memory_address_to_id_15,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,
    m31 **lookup_memory_id_to_big_5,
    m31 **lookup_memory_id_to_big_6,
    m31 **lookup_memory_id_to_big_7,
    m31 **lookup_memory_id_to_big_8,
    m31 **lookup_memory_id_to_big_9,
    m31 **lookup_memory_id_to_big_10,
    m31 **lookup_memory_id_to_big_11,
    m31 **lookup_memory_id_to_big_12,
    m31 **lookup_memory_id_to_big_13,
    m31 **lookup_memory_id_to_big_14,
    m31 **lookup_memory_id_to_big_15,
    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_7_2_5_1,
    m31 **lookup_range_check_7_2_5_2,
    m31 **lookup_range_check_7_2_5_3,
    m31 **lookup_range_check_7_2_5_4,
    m31 **lookup_range_check_7_2_5_5,
    m31 **lookup_range_check_7_2_5_6,
    m31 **lookup_range_check_7_2_5_7,
    m31 **lookup_range_check_7_2_5_8,
    m31 **lookup_range_check_7_2_5_9,
    m31 **lookup_range_check_7_2_5_10,
    m31 **lookup_range_check_7_2_5_11,
    m31 **lookup_range_check_7_2_5_12,
    m31 **lookup_range_check_7_2_5_13,
    m31 **lookup_range_check_7_2_5_14,
    m31 **lookup_range_check_7_2_5_15,

    m31 **sub_component_inputs_blake_round_sigma,
    m31 **sub_component_inputs_range_check_7_2_5,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_blake_g,

    m31 **blake_round_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned n_rows,
    unsigned trace_log_len
) {
    init_blake_round_sigma_constants_only_once();

    // 1. clone_to_device
    m31 **device_traces = clone_to_device<m31 *>(traces, N_TRACE_COLUMNS);

    m31 **device_lookup_blake_g_0 = clone_to_device<m31 *>(lookup_blake_g_0, 20);
    m31 **device_lookup_blake_g_1 = clone_to_device<m31 *>(lookup_blake_g_1, 20);
    m31 **device_lookup_blake_g_2 = clone_to_device<m31 *>(lookup_blake_g_2, 20);
    m31 **device_lookup_blake_g_3 = clone_to_device<m31 *>(lookup_blake_g_3, 20);
    m31 **device_lookup_blake_g_4 = clone_to_device<m31 *>(lookup_blake_g_4, 20);
    m31 **device_lookup_blake_g_5 = clone_to_device<m31 *>(lookup_blake_g_5, 20);
    m31 **device_lookup_blake_g_6 = clone_to_device<m31 *>(lookup_blake_g_6, 20);
    m31 **device_lookup_blake_g_7 = clone_to_device<m31 *>(lookup_blake_g_7, 20);

    m31 **device_lookup_blake_round_0 = clone_to_device<m31 *>(lookup_blake_round_0, 35);
    m31 **device_lookup_blake_round_1 = clone_to_device<m31 *>(lookup_blake_round_1, 35);
    m31 **device_lookup_blake_round_sigma_0 = clone_to_device<m31 *>(lookup_blake_round_sigma_0, 17);

    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31 *>(lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_address_to_id_3 = clone_to_device<m31 *>(lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_memory_address_to_id_4 = clone_to_device<m31 *>(lookup_memory_address_to_id_4, 2);
    m31 **device_lookup_memory_address_to_id_5 = clone_to_device<m31 *>(lookup_memory_address_to_id_5, 2);
    m31 **device_lookup_memory_address_to_id_6 = clone_to_device<m31 *>(lookup_memory_address_to_id_6, 2);
    m31 **device_lookup_memory_address_to_id_7 = clone_to_device<m31 *>(lookup_memory_address_to_id_7, 2);
    m31 **device_lookup_memory_address_to_id_8 = clone_to_device<m31 *>(lookup_memory_address_to_id_8, 2);
    m31 **device_lookup_memory_address_to_id_9 = clone_to_device<m31 *>(lookup_memory_address_to_id_9, 2);
    m31 **device_lookup_memory_address_to_id_10 = clone_to_device<m31 *>(lookup_memory_address_to_id_10, 2);
    m31 **device_lookup_memory_address_to_id_11 = clone_to_device<m31 *>(lookup_memory_address_to_id_11, 2);
    m31 **device_lookup_memory_address_to_id_12 = clone_to_device<m31 *>(lookup_memory_address_to_id_12, 2);
    m31 **device_lookup_memory_address_to_id_13 = clone_to_device<m31 *>(lookup_memory_address_to_id_13, 2);
    m31 **device_lookup_memory_address_to_id_14 = clone_to_device<m31 *>(lookup_memory_address_to_id_14, 2);
    m31 **device_lookup_memory_address_to_id_15 = clone_to_device<m31 *>(lookup_memory_address_to_id_15, 2);

    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>(lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31 *>(lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_memory_id_to_big_3 = clone_to_device<m31 *>(lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_memory_id_to_big_4 = clone_to_device<m31 *>(lookup_memory_id_to_big_4, 29);
    m31 **device_lookup_memory_id_to_big_5 = clone_to_device<m31 *>(lookup_memory_id_to_big_5, 29);
    m31 **device_lookup_memory_id_to_big_6 = clone_to_device<m31 *>(lookup_memory_id_to_big_6, 29);
    m31 **device_lookup_memory_id_to_big_7 = clone_to_device<m31 *>(lookup_memory_id_to_big_7, 29);
    m31 **device_lookup_memory_id_to_big_8 = clone_to_device<m31 *>(lookup_memory_id_to_big_8, 29);
    m31 **device_lookup_memory_id_to_big_9 = clone_to_device<m31 *>(lookup_memory_id_to_big_9, 29);
    m31 **device_lookup_memory_id_to_big_10 = clone_to_device<m31 *>(lookup_memory_id_to_big_10, 29);
    m31 **device_lookup_memory_id_to_big_11 = clone_to_device<m31 *>(lookup_memory_id_to_big_11, 29);
    m31 **device_lookup_memory_id_to_big_12 = clone_to_device<m31 *>(lookup_memory_id_to_big_12, 29);
    m31 **device_lookup_memory_id_to_big_13 = clone_to_device<m31 *>(lookup_memory_id_to_big_13, 29);
    m31 **device_lookup_memory_id_to_big_14 = clone_to_device<m31 *>(lookup_memory_id_to_big_14, 29);
    m31 **device_lookup_memory_id_to_big_15 = clone_to_device<m31 *>(lookup_memory_id_to_big_15, 29);

    m31 **device_lookup_range_check_7_2_5_0 = clone_to_device<m31 *>(lookup_range_check_7_2_5_0, 3);
    m31 **device_lookup_range_check_7_2_5_1 = clone_to_device<m31 *>(lookup_range_check_7_2_5_1, 3);
    m31 **device_lookup_range_check_7_2_5_2 = clone_to_device<m31 *>(lookup_range_check_7_2_5_2, 3);
    m31 **device_lookup_range_check_7_2_5_3 = clone_to_device<m31 *>(lookup_range_check_7_2_5_3, 3);
    m31 **device_lookup_range_check_7_2_5_4 = clone_to_device<m31 *>(lookup_range_check_7_2_5_4, 3);
    m31 **device_lookup_range_check_7_2_5_5 = clone_to_device<m31 *>(lookup_range_check_7_2_5_5, 3);
    m31 **device_lookup_range_check_7_2_5_6 = clone_to_device<m31 *>(lookup_range_check_7_2_5_6, 3);
    m31 **device_lookup_range_check_7_2_5_7 = clone_to_device<m31 *>(lookup_range_check_7_2_5_7, 3);
    m31 **device_lookup_range_check_7_2_5_8 = clone_to_device<m31 *>(lookup_range_check_7_2_5_8, 3);
    m31 **device_lookup_range_check_7_2_5_9 = clone_to_device<m31 *>(lookup_range_check_7_2_5_9, 3);
    m31 **device_lookup_range_check_7_2_5_10 = clone_to_device<m31 *>(lookup_range_check_7_2_5_10, 3);
    m31 **device_lookup_range_check_7_2_5_11 = clone_to_device<m31 *>(lookup_range_check_7_2_5_11, 3);
    m31 **device_lookup_range_check_7_2_5_12 = clone_to_device<m31 *>(lookup_range_check_7_2_5_12, 3);
    m31 **device_lookup_range_check_7_2_5_13 = clone_to_device<m31 *>(lookup_range_check_7_2_5_13, 3);
    m31 **device_lookup_range_check_7_2_5_14 = clone_to_device<m31 *>(lookup_range_check_7_2_5_14, 3);
    m31 **device_lookup_range_check_7_2_5_15 = clone_to_device<m31 *>(lookup_range_check_7_2_5_15, 3);

    m31 **device_sub_component_inputs_blake_round_sigma = clone_to_device<m31 *>(sub_component_inputs_blake_round_sigma, 1);
    m31 **device_sub_component_inputs_range_check_7_2_5 = clone_to_device<m31 *>(sub_component_inputs_range_check_7_2_5, 3 * 16);
    m31 **device_sub_component_inputs_memory_address_to_id = clone_to_device<m31 *>(sub_component_inputs_memory_address_to_id, 1 * 16);
    m31 **device_sub_component_inputs_memory_id_to_big = clone_to_device<m31 *>(sub_component_inputs_memory_id_to_big, 1 * 16);
    m31 **device_sub_component_inputs_blake_g = clone_to_device<m31 *>(sub_component_inputs_blake_g, 6 * 8);

    m31 **device_blake_round_input = clone_to_device<m31 *>(blake_round_input, (1 + 1 + 16 + 1));

    unsigned **device_memory_id_to_big_transpose_big_value_ptr = clone_to_device<m31 *>(memory_id_to_big_transpose_big_value_ptr, 8);


    timer global_timer;
    global_timer.start("generate blake_round base trace");

    unsigned trace_size = 1 << trace_log_len;
    int block_dim = trace_size < GEN_TRACE_BLAKE_ROUND_THREAD_COUNT_MAX ? trace_size : GEN_TRACE_BLAKE_ROUND_THREAD_COUNT_MAX;
    int num_blocks = block_dim < GEN_TRACE_BLAKE_ROUND_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // 2. start kernel
    generate_blake_round_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,

        device_lookup_blake_g_0,
        device_lookup_blake_g_1,
        device_lookup_blake_g_2,
        device_lookup_blake_g_3,
        device_lookup_blake_g_4,
        device_lookup_blake_g_5,
        device_lookup_blake_g_6,
        device_lookup_blake_g_7,
        device_lookup_blake_round_0,
        device_lookup_blake_round_1,
        device_lookup_blake_round_sigma_0,
        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_address_to_id_3,
        device_lookup_memory_address_to_id_4,
        device_lookup_memory_address_to_id_5,
        device_lookup_memory_address_to_id_6,
        device_lookup_memory_address_to_id_7,
        device_lookup_memory_address_to_id_8,
        device_lookup_memory_address_to_id_9,
        device_lookup_memory_address_to_id_10,
        device_lookup_memory_address_to_id_11,
        device_lookup_memory_address_to_id_12,
        device_lookup_memory_address_to_id_13,
        device_lookup_memory_address_to_id_14,
        device_lookup_memory_address_to_id_15,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_id_to_big_1,
        device_lookup_memory_id_to_big_2,
        device_lookup_memory_id_to_big_3,
        device_lookup_memory_id_to_big_4,
        device_lookup_memory_id_to_big_5,
        device_lookup_memory_id_to_big_6,
        device_lookup_memory_id_to_big_7,
        device_lookup_memory_id_to_big_8,
        device_lookup_memory_id_to_big_9,
        device_lookup_memory_id_to_big_10,
        device_lookup_memory_id_to_big_11,
        device_lookup_memory_id_to_big_12,
        device_lookup_memory_id_to_big_13,
        device_lookup_memory_id_to_big_14,
        device_lookup_memory_id_to_big_15,
        device_lookup_range_check_7_2_5_0,
        device_lookup_range_check_7_2_5_1,
        device_lookup_range_check_7_2_5_2,
        device_lookup_range_check_7_2_5_3,
        device_lookup_range_check_7_2_5_4,
        device_lookup_range_check_7_2_5_5,
        device_lookup_range_check_7_2_5_6,
        device_lookup_range_check_7_2_5_7,
        device_lookup_range_check_7_2_5_8,
        device_lookup_range_check_7_2_5_9,
        device_lookup_range_check_7_2_5_10,
        device_lookup_range_check_7_2_5_11,
        device_lookup_range_check_7_2_5_12,
        device_lookup_range_check_7_2_5_13,
        device_lookup_range_check_7_2_5_14,
        device_lookup_range_check_7_2_5_15,

        device_sub_component_inputs_blake_round_sigma,
        device_sub_component_inputs_range_check_7_2_5,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        device_sub_component_inputs_blake_g,

        device_blake_round_input,

        memory_address_to_id_address_to_raw_id,

        device_memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,

        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate blake_round base trace");

    cuda_free_memory(device_blake_round_input);

    cuda_free_memory(device_sub_component_inputs_blake_g);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_range_check_7_2_5);
    cuda_free_memory(device_sub_component_inputs_blake_round_sigma);

    cuda_free_memory(device_lookup_range_check_7_2_5_15);
    cuda_free_memory(device_lookup_range_check_7_2_5_14);
    cuda_free_memory(device_lookup_range_check_7_2_5_13);
    cuda_free_memory(device_lookup_range_check_7_2_5_12);
    cuda_free_memory(device_lookup_range_check_7_2_5_11);
    cuda_free_memory(device_lookup_range_check_7_2_5_10);
    cuda_free_memory(device_lookup_range_check_7_2_5_9);
    cuda_free_memory(device_lookup_range_check_7_2_5_8);
    cuda_free_memory(device_lookup_range_check_7_2_5_7);
    cuda_free_memory(device_lookup_range_check_7_2_5_6);
    cuda_free_memory(device_lookup_range_check_7_2_5_5);
    cuda_free_memory(device_lookup_range_check_7_2_5_4);
    cuda_free_memory(device_lookup_range_check_7_2_5_3);
    cuda_free_memory(device_lookup_range_check_7_2_5_2);
    cuda_free_memory(device_lookup_range_check_7_2_5_1);
    cuda_free_memory(device_lookup_range_check_7_2_5_0);

    cuda_free_memory(device_lookup_memory_id_to_big_15);
    cuda_free_memory(device_lookup_memory_id_to_big_14);
    cuda_free_memory(device_lookup_memory_id_to_big_13);
    cuda_free_memory(device_lookup_memory_id_to_big_12);
    cuda_free_memory(device_lookup_memory_id_to_big_11);
    cuda_free_memory(device_lookup_memory_id_to_big_10);
    cuda_free_memory(device_lookup_memory_id_to_big_9);
    cuda_free_memory(device_lookup_memory_id_to_big_8);
    cuda_free_memory(device_lookup_memory_id_to_big_7);
    cuda_free_memory(device_lookup_memory_id_to_big_6);
    cuda_free_memory(device_lookup_memory_id_to_big_5);
    cuda_free_memory(device_lookup_memory_id_to_big_4);
    cuda_free_memory(device_lookup_memory_id_to_big_3);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_0);

    cuda_free_memory(device_lookup_memory_address_to_id_15);
    cuda_free_memory(device_lookup_memory_address_to_id_14);
    cuda_free_memory(device_lookup_memory_address_to_id_13);
    cuda_free_memory(device_lookup_memory_address_to_id_12);
    cuda_free_memory(device_lookup_memory_address_to_id_11);
    cuda_free_memory(device_lookup_memory_address_to_id_10);
    cuda_free_memory(device_lookup_memory_address_to_id_9);
    cuda_free_memory(device_lookup_memory_address_to_id_8);
    cuda_free_memory(device_lookup_memory_address_to_id_7);
    cuda_free_memory(device_lookup_memory_address_to_id_6);
    cuda_free_memory(device_lookup_memory_address_to_id_5);
    cuda_free_memory(device_lookup_memory_address_to_id_4);
    cuda_free_memory(device_lookup_memory_address_to_id_3);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_0);

    cuda_free_memory(device_lookup_blake_round_sigma_0);
    cuda_free_memory(device_lookup_blake_round_1);
    cuda_free_memory(device_lookup_blake_round_0);

    cuda_free_memory(device_lookup_blake_g_7);
    cuda_free_memory(device_lookup_blake_g_6);
    cuda_free_memory(device_lookup_blake_g_5);
    cuda_free_memory(device_lookup_blake_g_4);
    cuda_free_memory(device_lookup_blake_g_3);
    cuda_free_memory(device_lookup_blake_g_2);
    cuda_free_memory(device_lookup_blake_g_1);
    cuda_free_memory(device_lookup_blake_g_0);

    cuda_free_memory(device_traces);
    cuda_free_memory(device_memory_id_to_big_transpose_big_value_ptr);
}


template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_blake_round_interaction_trace_col_gen_kernel(
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

// Kernel for second-to-last column: first lookup with enabler, second with +1
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_blake_round_interaction_trace_col_gen_kernel_second2last(
    LookupElementsBasic<N>  *lookup_elements_n,
    LookupElementsBasic<M>  *lookup_elements_m,
    m31 **lookup_state_0,
    m31 **lookup_state_1,
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
        // Apply enabler to denom0 (first lookup), add denom1 (second lookup)
        logup_col_write_frac(vec_index, add(denom1, mul(denom0, enabler_col)), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_blake_round_sigma_interaction_trace_col_gen_kernel(
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
    m31 init_combine_reg[17] = {};
    m31 final_combine_reg[M] = {};

    for (int i = 0; i < 17; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }
    for (int i = 0; i < M; i++) {
        final_combine_reg[i] = lookup_state_1[i][vec_index];
    }
    if (vec_index < trace_size) {
        qm31 denom0 = lookup_elements_n->combine(init_combine_reg, 17);
        qm31 denom1 = lookup_elements_m->combine(final_combine_reg, M);
        logup_col_write_frac(vec_index, add(denom1, denom0), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

template <int N>
__launch_bounds__(256, 2)
__global__ void generate_blake_round_interaction_trace_col_single_gen_kernel(
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


__global__ void generate_blake_round_interaction_trace_finalize_col_kernel(
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

__global__ void generate_blake_round_interaction_trace_cumsum_shift(
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

__global__ void generate_blake_round_interaction_trace_coord_prefix_sum(
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

void generate_blake_round_interaction_traces(
    void *blake_g,
    void *blake_round,
    void *blake_round_sigma ,
    void *memory_address_to_id ,
    void *memory_id_to_big ,
    void *range_check_7_2_5 ,

    m31 **lookup_blake_g_0,
    m31 **lookup_blake_g_1,
    m31 **lookup_blake_g_2,
    m31 **lookup_blake_g_3,
    m31 **lookup_blake_g_4,
    m31 **lookup_blake_g_5,
    m31 **lookup_blake_g_6,
    m31 **lookup_blake_g_7,

    m31 **lookup_blake_round_0,
    m31 **lookup_blake_round_1,

    m31 **lookup_blake_round_sigma_0,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,
    m31 **lookup_memory_address_to_id_5,
    m31 **lookup_memory_address_to_id_6,
    m31 **lookup_memory_address_to_id_7,
    m31 **lookup_memory_address_to_id_8,
    m31 **lookup_memory_address_to_id_9,
    m31 **lookup_memory_address_to_id_10,
    m31 **lookup_memory_address_to_id_11,
    m31 **lookup_memory_address_to_id_12,
    m31 **lookup_memory_address_to_id_13,
    m31 **lookup_memory_address_to_id_14,
    m31 **lookup_memory_address_to_id_15,

    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,
    m31 **lookup_memory_id_to_big_5,
    m31 **lookup_memory_id_to_big_6,
    m31 **lookup_memory_id_to_big_7,
    m31 **lookup_memory_id_to_big_8,
    m31 **lookup_memory_id_to_big_9,
    m31 **lookup_memory_id_to_big_10,
    m31 **lookup_memory_id_to_big_11,
    m31 **lookup_memory_id_to_big_12,
    m31 **lookup_memory_id_to_big_13,
    m31 **lookup_memory_id_to_big_14,
    m31 **lookup_memory_id_to_big_15,

    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_7_2_5_1,
    m31 **lookup_range_check_7_2_5_2,
    m31 **lookup_range_check_7_2_5_3,
    m31 **lookup_range_check_7_2_5_4,
    m31 **lookup_range_check_7_2_5_5,
    m31 **lookup_range_check_7_2_5_6,
    m31 **lookup_range_check_7_2_5_7,
    m31 **lookup_range_check_7_2_5_8,
    m31 **lookup_range_check_7_2_5_9,
    m31 **lookup_range_check_7_2_5_10,
    m31 **lookup_range_check_7_2_5_11,
    m31 **lookup_range_check_7_2_5_12,
    m31 **lookup_range_check_7_2_5_13,
    m31 **lookup_range_check_7_2_5_14,
    m31 **lookup_range_check_7_2_5_15,

    unsigned n_rows,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    BlakeG *blake_g_lookup_elements = (BlakeG*)blake_g;
    BlakeG *device_blake_g_lookup_elements = cuda_malloc<BlakeG>(1);
    cuda_mem_copy_host_to_device<BlakeG>(blake_g_lookup_elements, device_blake_g_lookup_elements, 1);

    BlakeRound *blake_round_lookup_elements = (BlakeRound *)blake_round;
    BlakeRound *device_blake_round_lookup_elements = cuda_malloc<BlakeRound>(1);
    cuda_mem_copy_host_to_device<BlakeRound>(blake_round_lookup_elements, device_blake_round_lookup_elements, 1);

    BlakeRoundSigma *blake_round_sigma_lookup_elements = (BlakeRoundSigma *)blake_round_sigma;
    BlakeRoundSigma *device_blake_round_sigma_lookup_elements = cuda_malloc<BlakeRoundSigma>(1);
    cuda_mem_copy_host_to_device<BlakeRoundSigma>(blake_round_sigma_lookup_elements, device_blake_round_sigma_lookup_elements, 1);

    MemoryAddressToId *memory_address_to_id_lookup_elements = (MemoryAddressToId *)memory_address_to_id;
    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id_lookup_elements, 1);

    MemoryIdToBig *memory_id_to_big_lookup_elements = (MemoryIdToBig *)memory_id_to_big;
    MemoryIdToBig *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big_lookup_elements, 1);

    RangeCheck_7_2_5 *range_check_7_2_5_lookup_elements = (RangeCheck_7_2_5 *)range_check_7_2_5;
    RangeCheck_7_2_5 *device_range_check_7_2_5_lookup_elements = cuda_malloc<RangeCheck_7_2_5>(1);
    cuda_mem_copy_host_to_device<RangeCheck_7_2_5>(range_check_7_2_5_lookup_elements, device_range_check_7_2_5_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_blake_g_0 = clone_to_device<m31 *>(lookup_blake_g_0, 20);
    m31 **device_lookup_blake_g_1 = clone_to_device<m31 *>(lookup_blake_g_1, 20);
    m31 **device_lookup_blake_g_2 = clone_to_device<m31 *>(lookup_blake_g_2, 20);
    m31 **device_lookup_blake_g_3 = clone_to_device<m31 *>(lookup_blake_g_3, 20);
    m31 **device_lookup_blake_g_4 = clone_to_device<m31 *>(lookup_blake_g_4, 20);
    m31 **device_lookup_blake_g_5 = clone_to_device<m31 *>(lookup_blake_g_5, 20);
    m31 **device_lookup_blake_g_6 = clone_to_device<m31 *>(lookup_blake_g_6, 20);
    m31 **device_lookup_blake_g_7 = clone_to_device<m31 *>(lookup_blake_g_7, 20);

    m31 **device_lookup_blake_round_0 = clone_to_device<m31 *>(lookup_blake_round_0, 35);
    m31 **device_lookup_blake_round_1 = clone_to_device<m31 *>(lookup_blake_round_1, 35);

    m31 **device_lookup_blake_round_sigma_0 = clone_to_device<m31 *>(lookup_blake_round_sigma_0, 17);

    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31 *>(lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_address_to_id_3 = clone_to_device<m31 *>(lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_memory_address_to_id_4 = clone_to_device<m31 *>(lookup_memory_address_to_id_4, 2);
    m31 **device_lookup_memory_address_to_id_5 = clone_to_device<m31 *>(lookup_memory_address_to_id_5, 2);
    m31 **device_lookup_memory_address_to_id_6 = clone_to_device<m31 *>(lookup_memory_address_to_id_6, 2);
    m31 **device_lookup_memory_address_to_id_7 = clone_to_device<m31 *>(lookup_memory_address_to_id_7, 2);
    m31 **device_lookup_memory_address_to_id_8 = clone_to_device<m31 *>(lookup_memory_address_to_id_8, 2);
    m31 **device_lookup_memory_address_to_id_9 = clone_to_device<m31 *>(lookup_memory_address_to_id_9, 2);
    m31 **device_lookup_memory_address_to_id_10 = clone_to_device<m31 *>(lookup_memory_address_to_id_10, 2);
    m31 **device_lookup_memory_address_to_id_11 = clone_to_device<m31 *>(lookup_memory_address_to_id_11, 2);
    m31 **device_lookup_memory_address_to_id_12 = clone_to_device<m31 *>(lookup_memory_address_to_id_12, 2);
    m31 **device_lookup_memory_address_to_id_13 = clone_to_device<m31 *>(lookup_memory_address_to_id_13, 2);
    m31 **device_lookup_memory_address_to_id_14 = clone_to_device<m31 *>(lookup_memory_address_to_id_14, 2);
    m31 **device_lookup_memory_address_to_id_15 = clone_to_device<m31 *>(lookup_memory_address_to_id_15, 2);

    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>(lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31 *>(lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_memory_id_to_big_3 = clone_to_device<m31 *>(lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_memory_id_to_big_4 = clone_to_device<m31 *>(lookup_memory_id_to_big_4, 29);
    m31 **device_lookup_memory_id_to_big_5 = clone_to_device<m31 *>(lookup_memory_id_to_big_5, 29);
    m31 **device_lookup_memory_id_to_big_6 = clone_to_device<m31 *>(lookup_memory_id_to_big_6, 29);
    m31 **device_lookup_memory_id_to_big_7 = clone_to_device<m31 *>(lookup_memory_id_to_big_7, 29);
    m31 **device_lookup_memory_id_to_big_8 = clone_to_device<m31 *>(lookup_memory_id_to_big_8, 29);
    m31 **device_lookup_memory_id_to_big_9 = clone_to_device<m31 *>(lookup_memory_id_to_big_9, 29);
    m31 **device_lookup_memory_id_to_big_10 = clone_to_device<m31 *>(lookup_memory_id_to_big_10, 29);
    m31 **device_lookup_memory_id_to_big_11 = clone_to_device<m31 *>(lookup_memory_id_to_big_11, 29);
    m31 **device_lookup_memory_id_to_big_12 = clone_to_device<m31 *>(lookup_memory_id_to_big_12, 29);
    m31 **device_lookup_memory_id_to_big_13 = clone_to_device<m31 *>(lookup_memory_id_to_big_13, 29);
    m31 **device_lookup_memory_id_to_big_14 = clone_to_device<m31 *>(lookup_memory_id_to_big_14, 29);
    m31 **device_lookup_memory_id_to_big_15 = clone_to_device<m31 *>(lookup_memory_id_to_big_15, 29);

    m31 **device_lookup_range_check_7_2_5_0 = clone_to_device<m31 *>(lookup_range_check_7_2_5_0, 3);
    m31 **device_lookup_range_check_7_2_5_1 = clone_to_device<m31 *>(lookup_range_check_7_2_5_1, 3);
    m31 **device_lookup_range_check_7_2_5_2 = clone_to_device<m31 *>(lookup_range_check_7_2_5_2, 3);
    m31 **device_lookup_range_check_7_2_5_3 = clone_to_device<m31 *>(lookup_range_check_7_2_5_3, 3);
    m31 **device_lookup_range_check_7_2_5_4 = clone_to_device<m31 *>(lookup_range_check_7_2_5_4, 3);
    m31 **device_lookup_range_check_7_2_5_5 = clone_to_device<m31 *>(lookup_range_check_7_2_5_5, 3);
    m31 **device_lookup_range_check_7_2_5_6 = clone_to_device<m31 *>(lookup_range_check_7_2_5_6, 3);
    m31 **device_lookup_range_check_7_2_5_7 = clone_to_device<m31 *>(lookup_range_check_7_2_5_7, 3);
    m31 **device_lookup_range_check_7_2_5_8 = clone_to_device<m31 *>(lookup_range_check_7_2_5_8, 3);
    m31 **device_lookup_range_check_7_2_5_9 = clone_to_device<m31 *>(lookup_range_check_7_2_5_9, 3);
    m31 **device_lookup_range_check_7_2_5_10 = clone_to_device<m31 *>(lookup_range_check_7_2_5_10, 3);
    m31 **device_lookup_range_check_7_2_5_11 = clone_to_device<m31 *>(lookup_range_check_7_2_5_11, 3);
    m31 **device_lookup_range_check_7_2_5_12 = clone_to_device<m31 *>(lookup_range_check_7_2_5_12, 3);
    m31 **device_lookup_range_check_7_2_5_13 = clone_to_device<m31 *>(lookup_range_check_7_2_5_13, 3);
    m31 **device_lookup_range_check_7_2_5_14 = clone_to_device<m31 *>(lookup_range_check_7_2_5_14, 3);
    m31 **device_lookup_range_check_7_2_5_15 = clone_to_device<m31 *>(lookup_range_check_7_2_5_15, 3);


    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4 * BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    // dump_lookup_data(lookup_verify_bitwise_xor_8_0, 3, trace_size);
    // dump_lookup_data(lookup_verify_bitwise_xor_8_1, 3, trace_size);

    timer global_timer;
    global_timer.start("generate blake_round interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For blake_round_sigma_0 & range_check_7_2_5_0
    generate_blake_round_sigma_interaction_trace_col_gen_kernel<17, 3><<<num_blocks, block_dim>>>(
        device_blake_round_sigma_lookup_elements,
        device_range_check_7_2_5_lookup_elements,

        device_lookup_blake_round_sigma_0,
        device_lookup_range_check_7_2_5_0,

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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #1 Interaction trace For memory_address_to_id_0 & memory_id_to_big_0
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,

        device_lookup_memory_address_to_id_0,
        device_lookup_memory_id_to_big_0,

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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #2. range_check_7_2_5_1 & memory_address_to_id_1
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_1,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #3. memory_id_to_big_1 & range_check_7_2_5_2
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_7_2_5_lookup_elements,
        device_lookup_memory_id_to_big_1,
        device_lookup_range_check_7_2_5_2,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #4. memory_address_to_id_2 & memory_id_to_big_2
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_id_to_big_2,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #5. range_check_7_2_5_3 & memory_address_to_id_3
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_3,
        device_lookup_memory_address_to_id_3,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #6. memory_id_to_big_3 & range_check_7_2_5_4
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_7_2_5_lookup_elements,
        device_lookup_memory_id_to_big_3,
        device_lookup_range_check_7_2_5_4,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #7. memory_address_to_id_4 & memory_id_to_big_4
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_4,
        device_lookup_memory_id_to_big_4,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #8. range_check_7_2_5_5 & memory_address_to_id_5
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_5,
        device_lookup_memory_address_to_id_5,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #9. memory_id_to_big_5 & range_check_7_2_5_6
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_7_2_5_lookup_elements,
        device_lookup_memory_id_to_big_5,
        device_lookup_range_check_7_2_5_6,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #10. memory_address_to_id_6 & memory_id_to_big_6
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_6,
        device_lookup_memory_id_to_big_6,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #11. range_check_7_2_5_7 & memory_address_to_id_7
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_7,
        device_lookup_memory_address_to_id_7,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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


    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_round_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #12. memory_id_to_big_7 & range_check_7_2_5_8
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_7_2_5_lookup_elements,
        device_lookup_memory_id_to_big_7,
        device_lookup_range_check_7_2_5_8,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #13. memory_address_to_id_8 & memory_id_to_big_8
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_8,
        device_lookup_memory_id_to_big_8,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #14. range_check_7_2_5_9 & memory_address_to_id_9
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_9,
        device_lookup_memory_address_to_id_9,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #15. memory_id_to_big_9 & range_check_7_2_5_10
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_7_2_5_lookup_elements,
        device_lookup_memory_id_to_big_9,
        device_lookup_range_check_7_2_5_10,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #16. memory_address_to_id_10 & memory_id_to_big_10
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_10,
        device_lookup_memory_id_to_big_10,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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


    // #17. range_check_7_2_5_11 & memory_address_to_id_11
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_11,
        device_lookup_memory_address_to_id_11,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #18. memory_id_to_big_11 & range_check_7_2_5_12
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_7_2_5_lookup_elements,
        device_lookup_memory_id_to_big_11,
        device_lookup_range_check_7_2_5_12,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #19. memory_address_to_id_12 & memory_id_to_big_12
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_12,
        device_lookup_memory_id_to_big_12,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        19,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #20. range_check_7_2_5_13 & memory_address_to_id_13
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_13,
        device_lookup_memory_address_to_id_13,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        20,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #21. memory_id_to_big_13 & range_check_7_2_5_14
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_7_2_5_lookup_elements,
        device_lookup_memory_id_to_big_13,
        device_lookup_range_check_7_2_5_14,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        21,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #22. memory_address_to_id_14 & memory_id_to_big_14
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<2, 29><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_memory_id_to_big_lookup_elements,
        device_lookup_memory_address_to_id_14,
        device_lookup_memory_id_to_big_14,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        22,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #23. range_check_7_2_5_15 & memory_address_to_id_15
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<3, 2><<<num_blocks, block_dim>>>(
        device_range_check_7_2_5_lookup_elements,
        device_memory_address_to_id_lookup_elements,
        device_lookup_range_check_7_2_5_15,
        device_lookup_memory_address_to_id_15,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        23,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #24. memory_id_to_big_15 & blake_g_0
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<29, 20><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_blake_g_lookup_elements,
        device_lookup_memory_id_to_big_15,
        device_lookup_blake_g_0,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        24,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #25. blake_g_1 & blake_g_2
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<20, 20><<<num_blocks, block_dim>>>(
        device_blake_g_lookup_elements,
        device_blake_g_lookup_elements,
        device_lookup_blake_g_1,
        device_lookup_blake_g_2,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        25,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #26. blake_g_3 & blake_g_4
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<20, 20><<<num_blocks, block_dim>>>(
        device_blake_g_lookup_elements,
        device_blake_g_lookup_elements,
        device_lookup_blake_g_3,
        device_lookup_blake_g_4,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        26,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #27. blake_g_5 & blake_g_6
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel<20, 20><<<num_blocks, block_dim>>>(
        device_blake_g_lookup_elements,
        device_blake_g_lookup_elements,
        device_lookup_blake_g_5,
        device_lookup_blake_g_6,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        27,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #28. blake_g_7 & blake_round_0 (second-to-last column: apply enabler to first lookup)
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_round_interaction_trace_col_gen_kernel_second2last<20, 35><<<num_blocks, block_dim>>>(
        device_blake_g_lookup_elements,
        device_blake_round_lookup_elements,
        device_lookup_blake_g_7,
        device_lookup_blake_round_0,
        n_rows,
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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        28,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // #29  Interaction trace For device_lookup_blake_round_1
    block_dim = trace_size < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_round_interaction_trace_col_single_gen_kernel<35><<<num_blocks, block_dim>>>(
        device_blake_round_lookup_elements,

        device_lookup_blake_round_1,

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
    generate_blake_round_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        29,
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
    generate_blake_round_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_round_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate blake_round interaction trace");

    // dump_interaction_traces(interaction_traces, 29, trace_size);

    // free denom_inv, device_numerator3, device_numerator2, device_numerator1, device_numerator0
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator0);

    // free device_interaction_traces
    cuda_free_memory(device_interaction_traces);

    // free device_lookup_range_check_7_2_5_15 ~ device_lookup_range_check_7_2_5_0
    cuda_free_memory(device_lookup_range_check_7_2_5_15);
    cuda_free_memory(device_lookup_range_check_7_2_5_14);
    cuda_free_memory(device_lookup_range_check_7_2_5_13);
    cuda_free_memory(device_lookup_range_check_7_2_5_12);
    cuda_free_memory(device_lookup_range_check_7_2_5_11);
    cuda_free_memory(device_lookup_range_check_7_2_5_10);
    cuda_free_memory(device_lookup_range_check_7_2_5_9);
    cuda_free_memory(device_lookup_range_check_7_2_5_8);
    cuda_free_memory(device_lookup_range_check_7_2_5_7);
    cuda_free_memory(device_lookup_range_check_7_2_5_6);
    cuda_free_memory(device_lookup_range_check_7_2_5_5);
    cuda_free_memory(device_lookup_range_check_7_2_5_4);
    cuda_free_memory(device_lookup_range_check_7_2_5_3);
    cuda_free_memory(device_lookup_range_check_7_2_5_2);
    cuda_free_memory(device_lookup_range_check_7_2_5_1);
    cuda_free_memory(device_lookup_range_check_7_2_5_0);

    // free device_lookup_memory_id_to_big_15 ~ device_lookup_memory_id_to_big_0
    cuda_free_memory(device_lookup_memory_id_to_big_15);
    cuda_free_memory(device_lookup_memory_id_to_big_14);
    cuda_free_memory(device_lookup_memory_id_to_big_13);
    cuda_free_memory(device_lookup_memory_id_to_big_12);
    cuda_free_memory(device_lookup_memory_id_to_big_11);
    cuda_free_memory(device_lookup_memory_id_to_big_10);
    cuda_free_memory(device_lookup_memory_id_to_big_9);
    cuda_free_memory(device_lookup_memory_id_to_big_8);
    cuda_free_memory(device_lookup_memory_id_to_big_7);
    cuda_free_memory(device_lookup_memory_id_to_big_6);
    cuda_free_memory(device_lookup_memory_id_to_big_5);
    cuda_free_memory(device_lookup_memory_id_to_big_4);
    cuda_free_memory(device_lookup_memory_id_to_big_3);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_0);

    // free device_lookup_memory_address_to_id_15 ~ device_lookup_memory_address_to_id_0
    cuda_free_memory(device_lookup_memory_address_to_id_15);
    cuda_free_memory(device_lookup_memory_address_to_id_14);
    cuda_free_memory(device_lookup_memory_address_to_id_13);
    cuda_free_memory(device_lookup_memory_address_to_id_12);
    cuda_free_memory(device_lookup_memory_address_to_id_11);
    cuda_free_memory(device_lookup_memory_address_to_id_10);
    cuda_free_memory(device_lookup_memory_address_to_id_9);
    cuda_free_memory(device_lookup_memory_address_to_id_8);
    cuda_free_memory(device_lookup_memory_address_to_id_7);
    cuda_free_memory(device_lookup_memory_address_to_id_6);
    cuda_free_memory(device_lookup_memory_address_to_id_5);
    cuda_free_memory(device_lookup_memory_address_to_id_4);
    cuda_free_memory(device_lookup_memory_address_to_id_3);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_0);

    // free device_lookup_blake_round_sigma_0
    cuda_free_memory(device_lookup_blake_round_sigma_0);

    // free device_lookup_blake_round_1, device_lookup_blake_round_0
    cuda_free_memory(device_lookup_blake_round_1);
    cuda_free_memory(device_lookup_blake_round_0);

    // free device_lookup_blake_g_7 ~ device_lookup_blake_g_0
    cuda_free_memory(device_lookup_blake_g_7);
    cuda_free_memory(device_lookup_blake_g_6);
    cuda_free_memory(device_lookup_blake_g_5);
    cuda_free_memory(device_lookup_blake_g_4);
    cuda_free_memory(device_lookup_blake_g_3);
    cuda_free_memory(device_lookup_blake_g_2);
    cuda_free_memory(device_lookup_blake_g_1);
    cuda_free_memory(device_lookup_blake_g_0);

    // free device_logup_denom
    cuda_free_memory(device_logup_denom);

    cuda_free_memory(device_range_check_7_2_5_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_blake_round_sigma_lookup_elements);
    cuda_free_memory(device_blake_round_lookup_elements);
    cuda_free_memory(device_blake_g_lookup_elements);
}