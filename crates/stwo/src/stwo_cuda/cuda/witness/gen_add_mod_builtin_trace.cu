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

#include "gen_add_mod_builtin_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

// Base trace generation kernel for add_mod_builtin
// 267 trace columns, 29 memory_address_to_id lookups, 24 memory_id_to_big lookups
__launch_bounds__(ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_add_mod_builtin_trace_kernel(
    m31 **traces,

    // Lookup data arrays - 29 MemoryAddressToId lookups (2 elements each)
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
    m31 **lookup_memory_address_to_id_16,
    m31 **lookup_memory_address_to_id_17,
    m31 **lookup_memory_address_to_id_18,
    m31 **lookup_memory_address_to_id_19,
    m31 **lookup_memory_address_to_id_20,
    m31 **lookup_memory_address_to_id_21,
    m31 **lookup_memory_address_to_id_22,
    m31 **lookup_memory_address_to_id_23,
    m31 **lookup_memory_address_to_id_24,
    m31 **lookup_memory_address_to_id_25,
    m31 **lookup_memory_address_to_id_26,
    m31 **lookup_memory_address_to_id_27,
    m31 **lookup_memory_address_to_id_28,

    // Lookup data arrays - 24 MemoryIdToBig lookups (29 elements each)
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
    m31 **lookup_memory_id_to_big_16,
    m31 **lookup_memory_id_to_big_17,
    m31 **lookup_memory_id_to_big_18,
    m31 **lookup_memory_id_to_big_19,
    m31 **lookup_memory_id_to_big_20,
    m31 **lookup_memory_id_to_big_21,
    m31 **lookup_memory_id_to_big_22,
    m31 **lookup_memory_id_to_big_23,

    // Sub-component inputs
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    // Constants
    const m31 M31_0         = {0};
    const m31 M31_1         = {1};
    const m31 M31_2         = {2};
    const m31 M31_3         = {3};
    const m31 M31_4         = {4};
    const m31 M31_5         = {5};
    const m31 M31_6         = {6};
    const m31 M31_7         = {7};
    const m31 M31_16        = {16};
    const m31 M31_64        = {64};
    const m31 M31_128       = {128};
    const m31 M31_136       = {136};
    const m31 M31_256       = {256};
    const m31 M31_508       = {508};
    const m31 M31_511       = {511};
    const m31 M31_512       = {512};
    const m31 M31_262144    = {262144};
    const m31 M31_32768     = {32768};
    const m31 M31_134217728 = {134217728};
    const m31 M31_536870912 = {536870912};

    const uint16_t UInt16_1 = 1;
    const uint16_t UInt16_2 = 2;
    const uint16_t UInt16_3 = 3;

    if (row < trace_size) {
        // seq = row index
        m31 seq = {row};

        // is_instance_0 = (seq == 0)
        m31 is_instance_0_col0 = (row == 0) ? M31_1 : M31_0;
        traces[0][row] = is_instance_0_col0;

        // prev_instance_addr = segment_start + 7 * ((seq - 1) + is_instance_0)
        // When row=0: prev_instance_addr = segment_start + 7 * (0-1+1) = segment_start
        // When row>0: prev_instance_addr = segment_start + 7 * (seq - 1)
        m31 segment_start_m31 = {segment_start};
        m31 prev_seq = (row == 0) ? M31_0 : sub(seq, M31_1);
        m31 prev_instance_addr = add(segment_start_m31, mul(M31_7, prev_seq));

        // instance_addr = segment_start + 7 * seq
        m31 instance_addr = add(segment_start_m31, mul(M31_7, seq));

        // ============ Read p0 (instance_addr + 0) ============
        // memory_address_to_id_0
        m31 p0_id_col1 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            instance_addr,
            &p0_id_col1
        );
        traces[1][row] = p0_id_col1;
        sub_component_inputs_memory_address_to_id[0][row] = instance_addr;
        lookup_memory_address_to_id_0[0][row] = instance_addr;
        lookup_memory_address_to_id_0[1][row] = p0_id_col1;

        // memory_id_to_big_0 - Read 11 limbs for p0 (99 bits)
        m31 p0_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            p0_id_col1,
            p0_value
        );

        // p0 limbs (columns 2-12)
        m31 p0_limb_0_col2 = p0_value[0]; traces[2][row] = p0_limb_0_col2;
        m31 p0_limb_1_col3 = p0_value[1]; traces[3][row] = p0_limb_1_col3;
        m31 p0_limb_2_col4 = p0_value[2]; traces[4][row] = p0_limb_2_col4;
        m31 p0_limb_3_col5 = p0_value[3]; traces[5][row] = p0_limb_3_col5;
        m31 p0_limb_4_col6 = p0_value[4]; traces[6][row] = p0_limb_4_col6;
        m31 p0_limb_5_col7 = p0_value[5]; traces[7][row] = p0_limb_5_col7;
        m31 p0_limb_6_col8 = p0_value[6]; traces[8][row] = p0_limb_6_col8;
        m31 p0_limb_7_col9 = p0_value[7]; traces[9][row] = p0_limb_7_col9;
        m31 p0_limb_8_col10 = p0_value[8]; traces[10][row] = p0_limb_8_col10;
        m31 p0_limb_9_col11 = p0_value[9]; traces[11][row] = p0_limb_9_col11;
        m31 p0_limb_10_col12 = p0_value[10]; traces[12][row] = p0_limb_10_col12;

        sub_component_inputs_memory_id_to_big[0][row] = p0_id_col1;
        lookup_memory_id_to_big_0[0][row] = p0_id_col1;
        lookup_memory_id_to_big_0[1][row] = p0_limb_0_col2;
        lookup_memory_id_to_big_0[2][row] = p0_limb_1_col3;
        lookup_memory_id_to_big_0[3][row] = p0_limb_2_col4;
        lookup_memory_id_to_big_0[4][row] = p0_limb_3_col5;
        lookup_memory_id_to_big_0[5][row] = p0_limb_4_col6;
        lookup_memory_id_to_big_0[6][row] = p0_limb_5_col7;
        lookup_memory_id_to_big_0[7][row] = p0_limb_6_col8;
        lookup_memory_id_to_big_0[8][row] = p0_limb_7_col9;
        lookup_memory_id_to_big_0[9][row] = p0_limb_8_col10;
        lookup_memory_id_to_big_0[10][row] = p0_limb_9_col11;
        lookup_memory_id_to_big_0[11][row] = p0_limb_10_col12;
        for (int i = 12; i < 29; i++) {
            lookup_memory_id_to_big_0[i][row] = M31_0;
        }

        // ============ Read p1 (instance_addr + 1) ============
        // memory_address_to_id_1
        m31 addr_p1 = add(instance_addr, M31_1);
        m31 p1_id_col13 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_p1,
            &p1_id_col13
        );
        traces[13][row] = p1_id_col13;
        sub_component_inputs_memory_address_to_id[1][row] = addr_p1;
        lookup_memory_address_to_id_1[0][row] = addr_p1;
        lookup_memory_address_to_id_1[1][row] = p1_id_col13;

        // memory_id_to_big_1 - Read 11 limbs for p1
        m31 p1_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            p1_id_col13,
            p1_value
        );

        // p1 limbs (columns 14-24)
        m31 p1_limb_0_col14 = p1_value[0]; traces[14][row] = p1_limb_0_col14;
        m31 p1_limb_1_col15 = p1_value[1]; traces[15][row] = p1_limb_1_col15;
        m31 p1_limb_2_col16 = p1_value[2]; traces[16][row] = p1_limb_2_col16;
        m31 p1_limb_3_col17 = p1_value[3]; traces[17][row] = p1_limb_3_col17;
        m31 p1_limb_4_col18 = p1_value[4]; traces[18][row] = p1_limb_4_col18;
        m31 p1_limb_5_col19 = p1_value[5]; traces[19][row] = p1_limb_5_col19;
        m31 p1_limb_6_col20 = p1_value[6]; traces[20][row] = p1_limb_6_col20;
        m31 p1_limb_7_col21 = p1_value[7]; traces[21][row] = p1_limb_7_col21;
        m31 p1_limb_8_col22 = p1_value[8]; traces[22][row] = p1_limb_8_col22;
        m31 p1_limb_9_col23 = p1_value[9]; traces[23][row] = p1_limb_9_col23;
        m31 p1_limb_10_col24 = p1_value[10]; traces[24][row] = p1_limb_10_col24;

        sub_component_inputs_memory_id_to_big[1][row] = p1_id_col13;
        lookup_memory_id_to_big_1[0][row] = p1_id_col13;
        lookup_memory_id_to_big_1[1][row] = p1_limb_0_col14;
        lookup_memory_id_to_big_1[2][row] = p1_limb_1_col15;
        lookup_memory_id_to_big_1[3][row] = p1_limb_2_col16;
        lookup_memory_id_to_big_1[4][row] = p1_limb_3_col17;
        lookup_memory_id_to_big_1[5][row] = p1_limb_4_col18;
        lookup_memory_id_to_big_1[6][row] = p1_limb_5_col19;
        lookup_memory_id_to_big_1[7][row] = p1_limb_6_col20;
        lookup_memory_id_to_big_1[8][row] = p1_limb_7_col21;
        lookup_memory_id_to_big_1[9][row] = p1_limb_8_col22;
        lookup_memory_id_to_big_1[10][row] = p1_limb_9_col23;
        lookup_memory_id_to_big_1[11][row] = p1_limb_10_col24;
        for (int i = 12; i < 29; i++) {
            lookup_memory_id_to_big_1[i][row] = M31_0;
        }

        // ============ Read p2 (instance_addr + 2) ============
        // memory_address_to_id_2
        m31 addr_p2 = add(instance_addr, M31_2);
        m31 p2_id_col25 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_p2,
            &p2_id_col25
        );
        traces[25][row] = p2_id_col25;
        sub_component_inputs_memory_address_to_id[2][row] = addr_p2;
        lookup_memory_address_to_id_2[0][row] = addr_p2;
        lookup_memory_address_to_id_2[1][row] = p2_id_col25;

        // memory_id_to_big_2 - Read 11 limbs for p2
        m31 p2_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            p2_id_col25,
            p2_value
        );

        // p2 limbs (columns 26-36)
        m31 p2_limb_0_col26 = p2_value[0]; traces[26][row] = p2_limb_0_col26;
        m31 p2_limb_1_col27 = p2_value[1]; traces[27][row] = p2_limb_1_col27;
        m31 p2_limb_2_col28 = p2_value[2]; traces[28][row] = p2_limb_2_col28;
        m31 p2_limb_3_col29 = p2_value[3]; traces[29][row] = p2_limb_3_col29;
        m31 p2_limb_4_col30 = p2_value[4]; traces[30][row] = p2_limb_4_col30;
        m31 p2_limb_5_col31 = p2_value[5]; traces[31][row] = p2_limb_5_col31;
        m31 p2_limb_6_col32 = p2_value[6]; traces[32][row] = p2_limb_6_col32;
        m31 p2_limb_7_col33 = p2_value[7]; traces[33][row] = p2_limb_7_col33;
        m31 p2_limb_8_col34 = p2_value[8]; traces[34][row] = p2_limb_8_col34;
        m31 p2_limb_9_col35 = p2_value[9]; traces[35][row] = p2_limb_9_col35;
        m31 p2_limb_10_col36 = p2_value[10]; traces[36][row] = p2_limb_10_col36;

        sub_component_inputs_memory_id_to_big[2][row] = p2_id_col25;
        lookup_memory_id_to_big_2[0][row] = p2_id_col25;
        lookup_memory_id_to_big_2[1][row] = p2_limb_0_col26;
        lookup_memory_id_to_big_2[2][row] = p2_limb_1_col27;
        lookup_memory_id_to_big_2[3][row] = p2_limb_2_col28;
        lookup_memory_id_to_big_2[4][row] = p2_limb_3_col29;
        lookup_memory_id_to_big_2[5][row] = p2_limb_4_col30;
        lookup_memory_id_to_big_2[6][row] = p2_limb_5_col31;
        lookup_memory_id_to_big_2[7][row] = p2_limb_6_col32;
        lookup_memory_id_to_big_2[8][row] = p2_limb_7_col33;
        lookup_memory_id_to_big_2[9][row] = p2_limb_8_col34;
        lookup_memory_id_to_big_2[10][row] = p2_limb_9_col35;
        lookup_memory_id_to_big_2[11][row] = p2_limb_10_col36;
        for (int i = 12; i < 29; i++) {
            lookup_memory_id_to_big_2[i][row] = M31_0;
        }

        // ============ Read p3 (instance_addr + 3) ============
        // memory_address_to_id_3
        m31 addr_p3 = add(instance_addr, M31_3);
        m31 p3_id_col37 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_p3,
            &p3_id_col37
        );
        traces[37][row] = p3_id_col37;
        sub_component_inputs_memory_address_to_id[3][row] = addr_p3;
        lookup_memory_address_to_id_3[0][row] = addr_p3;
        lookup_memory_address_to_id_3[1][row] = p3_id_col37;

        // memory_id_to_big_3 - Read 11 limbs for p3
        m31 p3_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            p3_id_col37,
            p3_value
        );

        // p3 limbs (columns 38-48)
        m31 p3_limb_0_col38 = p3_value[0]; traces[38][row] = p3_limb_0_col38;
        m31 p3_limb_1_col39 = p3_value[1]; traces[39][row] = p3_limb_1_col39;
        m31 p3_limb_2_col40 = p3_value[2]; traces[40][row] = p3_limb_2_col40;
        m31 p3_limb_3_col41 = p3_value[3]; traces[41][row] = p3_limb_3_col41;
        m31 p3_limb_4_col42 = p3_value[4]; traces[42][row] = p3_limb_4_col42;
        m31 p3_limb_5_col43 = p3_value[5]; traces[43][row] = p3_limb_5_col43;
        m31 p3_limb_6_col44 = p3_value[6]; traces[44][row] = p3_limb_6_col44;
        m31 p3_limb_7_col45 = p3_value[7]; traces[45][row] = p3_limb_7_col45;
        m31 p3_limb_8_col46 = p3_value[8]; traces[46][row] = p3_limb_8_col46;
        m31 p3_limb_9_col47 = p3_value[9]; traces[47][row] = p3_limb_9_col47;
        m31 p3_limb_10_col48 = p3_value[10]; traces[48][row] = p3_limb_10_col48;

        sub_component_inputs_memory_id_to_big[3][row] = p3_id_col37;
        lookup_memory_id_to_big_3[0][row] = p3_id_col37;
        lookup_memory_id_to_big_3[1][row] = p3_limb_0_col38;
        lookup_memory_id_to_big_3[2][row] = p3_limb_1_col39;
        lookup_memory_id_to_big_3[3][row] = p3_limb_2_col40;
        lookup_memory_id_to_big_3[4][row] = p3_limb_3_col41;
        lookup_memory_id_to_big_3[5][row] = p3_limb_4_col42;
        lookup_memory_id_to_big_3[6][row] = p3_limb_5_col43;
        lookup_memory_id_to_big_3[7][row] = p3_limb_6_col44;
        lookup_memory_id_to_big_3[8][row] = p3_limb_7_col45;
        lookup_memory_id_to_big_3[9][row] = p3_limb_8_col46;
        lookup_memory_id_to_big_3[10][row] = p3_limb_9_col47;
        lookup_memory_id_to_big_3[11][row] = p3_limb_10_col48;
        for (int i = 12; i < 29; i++) {
            lookup_memory_id_to_big_3[i][row] = M31_0;
        }

        // ============ Read values_ptr (instance_addr + 4) ============
        // memory_address_to_id_4
        m31 addr_values_ptr = add(instance_addr, M31_4);
        m31 values_ptr_id_col49 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_values_ptr,
            &values_ptr_id_col49
        );
        traces[49][row] = values_ptr_id_col49;
        sub_component_inputs_memory_address_to_id[4][row] = addr_values_ptr;
        lookup_memory_address_to_id_4[0][row] = addr_values_ptr;
        lookup_memory_address_to_id_4[1][row] = values_ptr_id_col49;

        // memory_id_to_big_4 - Read 4 limbs for values_ptr (29 bits)
        m31 values_ptr_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            values_ptr_id_col49,
            values_ptr_value
        );

        m31 values_ptr_limb_0_col50 = values_ptr_value[0]; traces[50][row] = values_ptr_limb_0_col50;
        m31 values_ptr_limb_1_col51 = values_ptr_value[1]; traces[51][row] = values_ptr_limb_1_col51;
        m31 values_ptr_limb_2_col52 = values_ptr_value[2]; traces[52][row] = values_ptr_limb_2_col52;
        m31 values_ptr_limb_3_col53 = values_ptr_value[3]; traces[53][row] = values_ptr_limb_3_col53;

        // Range check MSB for values_ptr
        uint16_t partial_limb_msb_tmp_0 = (((uint16_t)(values_ptr_limb_3_col53)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col54 = m31{partial_limb_msb_tmp_0};
        traces[54][row] = partial_limb_msb_col54;

        sub_component_inputs_memory_id_to_big[4][row] = values_ptr_id_col49;
        lookup_memory_id_to_big_4[0][row] = values_ptr_id_col49;
        lookup_memory_id_to_big_4[1][row] = values_ptr_limb_0_col50;
        lookup_memory_id_to_big_4[2][row] = values_ptr_limb_1_col51;
        lookup_memory_id_to_big_4[3][row] = values_ptr_limb_2_col52;
        lookup_memory_id_to_big_4[4][row] = values_ptr_limb_3_col53;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_4[i][row] = M31_0;
        }

        // Compute values_ptr as combined value
        m31 values_ptr = add(
            add(
                add(values_ptr_limb_0_col50, mul(values_ptr_limb_1_col51, M31_512)),
                mul(values_ptr_limb_2_col52, M31_262144)
            ),
            mul(values_ptr_limb_3_col53, M31_134217728)
        );

        // ============ Read offsets_ptr (instance_addr + 5) ============
        // memory_address_to_id_5
        m31 addr_offsets_ptr = add(instance_addr, M31_5);
        m31 offsets_ptr_id_col55 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_offsets_ptr,
            &offsets_ptr_id_col55
        );
        traces[55][row] = offsets_ptr_id_col55;
        sub_component_inputs_memory_address_to_id[5][row] = addr_offsets_ptr;
        lookup_memory_address_to_id_5[0][row] = addr_offsets_ptr;
        lookup_memory_address_to_id_5[1][row] = offsets_ptr_id_col55;

        // memory_id_to_big_5 - Read 4 limbs for offsets_ptr
        m31 offsets_ptr_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            offsets_ptr_id_col55,
            offsets_ptr_value
        );

        m31 offsets_ptr_limb_0_col56 = offsets_ptr_value[0]; traces[56][row] = offsets_ptr_limb_0_col56;
        m31 offsets_ptr_limb_1_col57 = offsets_ptr_value[1]; traces[57][row] = offsets_ptr_limb_1_col57;
        m31 offsets_ptr_limb_2_col58 = offsets_ptr_value[2]; traces[58][row] = offsets_ptr_limb_2_col58;
        m31 offsets_ptr_limb_3_col59 = offsets_ptr_value[3]; traces[59][row] = offsets_ptr_limb_3_col59;

        // Range check MSB for offsets_ptr
        uint16_t partial_limb_msb_tmp_1 = (((uint16_t)(offsets_ptr_limb_3_col59)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col60 = m31{partial_limb_msb_tmp_1};
        traces[60][row] = partial_limb_msb_col60;

        sub_component_inputs_memory_id_to_big[5][row] = offsets_ptr_id_col55;
        lookup_memory_id_to_big_5[0][row] = offsets_ptr_id_col55;
        lookup_memory_id_to_big_5[1][row] = offsets_ptr_limb_0_col56;
        lookup_memory_id_to_big_5[2][row] = offsets_ptr_limb_1_col57;
        lookup_memory_id_to_big_5[3][row] = offsets_ptr_limb_2_col58;
        lookup_memory_id_to_big_5[4][row] = offsets_ptr_limb_3_col59;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_5[i][row] = M31_0;
        }

        // Compute offsets_ptr as combined value
        m31 offsets_ptr_combined = add(
            add(
                add(offsets_ptr_limb_0_col56, mul(offsets_ptr_limb_1_col57, M31_512)),
                mul(offsets_ptr_limb_2_col58, M31_262144)
            ),
            mul(offsets_ptr_limb_3_col59, M31_134217728)
        );

        // ============ Read offsets_ptr_prev (prev_instance_addr + 5) ============
        // memory_address_to_id_6
        m31 addr_offsets_ptr_prev = add(prev_instance_addr, M31_5);
        m31 offsets_ptr_prev_id_col61 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_offsets_ptr_prev,
            &offsets_ptr_prev_id_col61
        );
        traces[61][row] = offsets_ptr_prev_id_col61;
        sub_component_inputs_memory_address_to_id[6][row] = addr_offsets_ptr_prev;
        lookup_memory_address_to_id_6[0][row] = addr_offsets_ptr_prev;
        lookup_memory_address_to_id_6[1][row] = offsets_ptr_prev_id_col61;

        // memory_id_to_big_6 - Read 4 limbs for offsets_ptr_prev
        m31 offsets_ptr_prev_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            offsets_ptr_prev_id_col61,
            offsets_ptr_prev_value
        );

        m31 offsets_ptr_prev_limb_0_col62 = offsets_ptr_prev_value[0]; traces[62][row] = offsets_ptr_prev_limb_0_col62;
        m31 offsets_ptr_prev_limb_1_col63 = offsets_ptr_prev_value[1]; traces[63][row] = offsets_ptr_prev_limb_1_col63;
        m31 offsets_ptr_prev_limb_2_col64 = offsets_ptr_prev_value[2]; traces[64][row] = offsets_ptr_prev_limb_2_col64;
        m31 offsets_ptr_prev_limb_3_col65 = offsets_ptr_prev_value[3]; traces[65][row] = offsets_ptr_prev_limb_3_col65;

        // Range check MSB
        uint16_t partial_limb_msb_tmp_2 = (((uint16_t)(offsets_ptr_prev_limb_3_col65)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col66 = m31{partial_limb_msb_tmp_2};
        traces[66][row] = partial_limb_msb_col66;

        sub_component_inputs_memory_id_to_big[6][row] = offsets_ptr_prev_id_col61;
        lookup_memory_id_to_big_6[0][row] = offsets_ptr_prev_id_col61;
        lookup_memory_id_to_big_6[1][row] = offsets_ptr_prev_limb_0_col62;
        lookup_memory_id_to_big_6[2][row] = offsets_ptr_prev_limb_1_col63;
        lookup_memory_id_to_big_6[3][row] = offsets_ptr_prev_limb_2_col64;
        lookup_memory_id_to_big_6[4][row] = offsets_ptr_prev_limb_3_col65;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_6[i][row] = M31_0;
        }

        // ============ Read n (instance_addr + 6) ============
        // memory_address_to_id_7
        m31 addr_n = add(instance_addr, M31_6);
        m31 n_id_col67 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_n,
            &n_id_col67
        );
        traces[67][row] = n_id_col67;
        sub_component_inputs_memory_address_to_id[7][row] = addr_n;
        lookup_memory_address_to_id_7[0][row] = addr_n;
        lookup_memory_address_to_id_7[1][row] = n_id_col67;

        // memory_id_to_big_7 - Read 4 limbs for n
        m31 n_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            n_id_col67,
            n_value
        );

        m31 n_limb_0_col68 = n_value[0]; traces[68][row] = n_limb_0_col68;
        m31 n_limb_1_col69 = n_value[1]; traces[69][row] = n_limb_1_col69;
        m31 n_limb_2_col70 = n_value[2]; traces[70][row] = n_limb_2_col70;
        m31 n_limb_3_col71 = n_value[3]; traces[71][row] = n_limb_3_col71;

        // Range check MSB
        uint16_t partial_limb_msb_tmp_3 = (((uint16_t)(n_limb_3_col71)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col72 = m31{partial_limb_msb_tmp_3};
        traces[72][row] = partial_limb_msb_col72;

        sub_component_inputs_memory_id_to_big[7][row] = n_id_col67;
        lookup_memory_id_to_big_7[0][row] = n_id_col67;
        lookup_memory_id_to_big_7[1][row] = n_limb_0_col68;
        lookup_memory_id_to_big_7[2][row] = n_limb_1_col69;
        lookup_memory_id_to_big_7[3][row] = n_limb_2_col70;
        lookup_memory_id_to_big_7[4][row] = n_limb_3_col71;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_7[i][row] = M31_0;
        }

        // ============ Read n_prev (prev_instance_addr + 6) ============
        // memory_address_to_id_8
        m31 addr_n_prev = add(prev_instance_addr, M31_6);
        m31 n_prev_id_col73 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_n_prev,
            &n_prev_id_col73
        );
        traces[73][row] = n_prev_id_col73;
        sub_component_inputs_memory_address_to_id[8][row] = addr_n_prev;
        lookup_memory_address_to_id_8[0][row] = addr_n_prev;
        lookup_memory_address_to_id_8[1][row] = n_prev_id_col73;

        // memory_id_to_big_8 - Read 4 limbs for n_prev
        m31 n_prev_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            n_prev_id_col73,
            n_prev_value
        );

        m31 n_prev_limb_0_col74 = n_prev_value[0]; traces[74][row] = n_prev_limb_0_col74;
        m31 n_prev_limb_1_col75 = n_prev_value[1]; traces[75][row] = n_prev_limb_1_col75;
        m31 n_prev_limb_2_col76 = n_prev_value[2]; traces[76][row] = n_prev_limb_2_col76;
        m31 n_prev_limb_3_col77 = n_prev_value[3]; traces[77][row] = n_prev_limb_3_col77;

        // Range check MSB
        uint16_t partial_limb_msb_tmp_4 = (((uint16_t)(n_prev_limb_3_col77)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col78 = m31{partial_limb_msb_tmp_4};
        traces[78][row] = partial_limb_msb_col78;

        sub_component_inputs_memory_id_to_big[8][row] = n_prev_id_col73;
        lookup_memory_id_to_big_8[0][row] = n_prev_id_col73;
        lookup_memory_id_to_big_8[1][row] = n_prev_limb_0_col74;
        lookup_memory_id_to_big_8[2][row] = n_prev_limb_1_col75;
        lookup_memory_id_to_big_8[3][row] = n_prev_limb_2_col76;
        lookup_memory_id_to_big_8[4][row] = n_prev_limb_3_col77;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_8[i][row] = M31_0;
        }

        // ============ Read values_ptr_prev (prev_instance_addr + 4) ============
        // memory_address_to_id_9
        m31 addr_values_ptr_prev = add(prev_instance_addr, M31_4);
        m31 values_ptr_prev_id_col79 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_values_ptr_prev,
            &values_ptr_prev_id_col79
        );
        traces[79][row] = values_ptr_prev_id_col79;
        sub_component_inputs_memory_address_to_id[9][row] = addr_values_ptr_prev;
        lookup_memory_address_to_id_9[0][row] = addr_values_ptr_prev;
        lookup_memory_address_to_id_9[1][row] = values_ptr_prev_id_col79;

        // ============ Read p_prev0 (prev_instance_addr + 0) ============
        // memory_address_to_id_10
        m31 p_prev0_id_col80 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            prev_instance_addr,
            &p_prev0_id_col80
        );
        traces[80][row] = p_prev0_id_col80;
        sub_component_inputs_memory_address_to_id[10][row] = prev_instance_addr;
        lookup_memory_address_to_id_10[0][row] = prev_instance_addr;
        lookup_memory_address_to_id_10[1][row] = p_prev0_id_col80;

        // ============ Read p_prev1 (prev_instance_addr + 1) ============
        // memory_address_to_id_11
        m31 addr_p_prev1 = add(prev_instance_addr, M31_1);
        m31 p_prev1_id_col81 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_p_prev1,
            &p_prev1_id_col81
        );
        traces[81][row] = p_prev1_id_col81;
        sub_component_inputs_memory_address_to_id[11][row] = addr_p_prev1;
        lookup_memory_address_to_id_11[0][row] = addr_p_prev1;
        lookup_memory_address_to_id_11[1][row] = p_prev1_id_col81;

        // ============ Read p_prev2 (prev_instance_addr + 2) ============
        // memory_address_to_id_12
        m31 addr_p_prev2 = add(prev_instance_addr, M31_2);
        m31 p_prev2_id_col82 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_p_prev2,
            &p_prev2_id_col82
        );
        traces[82][row] = p_prev2_id_col82;
        sub_component_inputs_memory_address_to_id[12][row] = addr_p_prev2;
        lookup_memory_address_to_id_12[0][row] = addr_p_prev2;
        lookup_memory_address_to_id_12[1][row] = p_prev2_id_col82;

        // ============ Read p_prev3 (prev_instance_addr + 3) ============
        // memory_address_to_id_13
        m31 addr_p_prev3 = add(prev_instance_addr, M31_3);
        m31 p_prev3_id_col83 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_p_prev3,
            &p_prev3_id_col83
        );
        traces[83][row] = p_prev3_id_col83;
        sub_component_inputs_memory_address_to_id[13][row] = addr_p_prev3;
        lookup_memory_address_to_id_13[0][row] = addr_p_prev3;
        lookup_memory_address_to_id_13[1][row] = p_prev3_id_col83;

        // ============ Read offsets_a (offsets_ptr_combined) ============
        // memory_address_to_id_14
        m31 offsets_a_id_col84 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            offsets_ptr_combined,
            &offsets_a_id_col84
        );
        traces[84][row] = offsets_a_id_col84;
        sub_component_inputs_memory_address_to_id[14][row] = offsets_ptr_combined;
        lookup_memory_address_to_id_14[0][row] = offsets_ptr_combined;
        lookup_memory_address_to_id_14[1][row] = offsets_a_id_col84;

        // Read small value for offsets_a with signed decoding
        m31 offsets_a_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            offsets_a_id_col84,
            offsets_a_value
        );

        // Cond Decode Small Sign for offsets_a
        m31 msb_col85 = (offsets_a_value[27] == M31_256) ? M31_1 : M31_0;
        traces[85][row] = msb_col85;
        m31 mid_limbs_set_col86 = (offsets_a_value[20] == M31_511) ? M31_1 : M31_0;
        traces[86][row] = mid_limbs_set_col86;

        m31 offsets_a_limb_0_col87 = offsets_a_value[0]; traces[87][row] = offsets_a_limb_0_col87;
        m31 offsets_a_limb_1_col88 = offsets_a_value[1]; traces[88][row] = offsets_a_limb_1_col88;
        m31 offsets_a_limb_2_col89 = offsets_a_value[2]; traces[89][row] = offsets_a_limb_2_col89;
        m31 remainder_bits_col90 = m31{((uint16_t)offsets_a_value[3]) & UInt16_3};
        traces[90][row] = remainder_bits_col90;
        m31 partial_limb_msb_col91 = m31{(((uint16_t)remainder_bits_col90) & UInt16_2) >> UInt16_1};
        traces[91][row] = partial_limb_msb_col91;

        // memory_id_to_big_9 with signed encoding
        sub_component_inputs_memory_id_to_big[9][row] = offsets_a_id_col84;
        lookup_memory_id_to_big_9[0][row] = offsets_a_id_col84;
        lookup_memory_id_to_big_9[1][row] = offsets_a_limb_0_col87;
        lookup_memory_id_to_big_9[2][row] = offsets_a_limb_1_col88;
        lookup_memory_id_to_big_9[3][row] = offsets_a_limb_2_col89;
        lookup_memory_id_to_big_9[4][row] = add(remainder_bits_col90, mul(mid_limbs_set_col86, M31_508));
        for (int i = 5; i < 21; i++) {
            lookup_memory_id_to_big_9[i][row] = mul(mid_limbs_set_col86, M31_511);
        }
        lookup_memory_id_to_big_9[21][row] = sub(mul(M31_136, msb_col85), mid_limbs_set_col86);
        for (int i = 22; i < 28; i++) {
            lookup_memory_id_to_big_9[i][row] = M31_0;
        }
        lookup_memory_id_to_big_9[28][row] = mul(msb_col85, M31_256);

        // Compute offset_a value (signed)
        m31 offset_a = sub(sub(add(add(add(offsets_a_limb_0_col87,
            mul(offsets_a_limb_1_col88, M31_512)),
            mul(offsets_a_limb_2_col89, M31_262144)),
            mul(remainder_bits_col90, M31_134217728)),
            msb_col85),
            mul(M31_536870912, mid_limbs_set_col86));

        // ============ Read offsets_b (offsets_ptr_combined + 1) ============
        // memory_address_to_id_15
        m31 addr_offsets_b = add(offsets_ptr_combined, M31_1);
        m31 offsets_b_id_col92 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_offsets_b,
            &offsets_b_id_col92
        );
        traces[92][row] = offsets_b_id_col92;
        sub_component_inputs_memory_address_to_id[15][row] = addr_offsets_b;
        lookup_memory_address_to_id_15[0][row] = addr_offsets_b;
        lookup_memory_address_to_id_15[1][row] = offsets_b_id_col92;

        // Read small value for offsets_b with signed decoding
        m31 offsets_b_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            offsets_b_id_col92,
            offsets_b_value
        );

        // Cond Decode Small Sign for offsets_b
        m31 msb_col93 = (offsets_b_value[27] == M31_256) ? M31_1 : M31_0;
        traces[93][row] = msb_col93;
        m31 mid_limbs_set_col94 = (offsets_b_value[20] == M31_511) ? M31_1 : M31_0;
        traces[94][row] = mid_limbs_set_col94;

        m31 offsets_b_limb_0_col95 = offsets_b_value[0]; traces[95][row] = offsets_b_limb_0_col95;
        m31 offsets_b_limb_1_col96 = offsets_b_value[1]; traces[96][row] = offsets_b_limb_1_col96;
        m31 offsets_b_limb_2_col97 = offsets_b_value[2]; traces[97][row] = offsets_b_limb_2_col97;
        m31 remainder_bits_col98 = m31{((uint16_t)offsets_b_value[3]) & UInt16_3};
        traces[98][row] = remainder_bits_col98;
        m31 partial_limb_msb_col99 = m31{(((uint16_t)remainder_bits_col98) & UInt16_2) >> UInt16_1};
        traces[99][row] = partial_limb_msb_col99;

        // memory_id_to_big_10 with signed encoding
        sub_component_inputs_memory_id_to_big[10][row] = offsets_b_id_col92;
        lookup_memory_id_to_big_10[0][row] = offsets_b_id_col92;
        lookup_memory_id_to_big_10[1][row] = offsets_b_limb_0_col95;
        lookup_memory_id_to_big_10[2][row] = offsets_b_limb_1_col96;
        lookup_memory_id_to_big_10[3][row] = offsets_b_limb_2_col97;
        lookup_memory_id_to_big_10[4][row] = add(remainder_bits_col98, mul(mid_limbs_set_col94, M31_508));
        for (int i = 5; i < 21; i++) {
            lookup_memory_id_to_big_10[i][row] = mul(mid_limbs_set_col94, M31_511);
        }
        lookup_memory_id_to_big_10[21][row] = sub(mul(M31_136, msb_col93), mid_limbs_set_col94);
        for (int i = 22; i < 28; i++) {
            lookup_memory_id_to_big_10[i][row] = M31_0;
        }
        lookup_memory_id_to_big_10[28][row] = mul(msb_col93, M31_256);

        // Compute offset_b value (signed)
        m31 offset_b = sub(sub(add(add(add(offsets_b_limb_0_col95,
            mul(offsets_b_limb_1_col96, M31_512)),
            mul(offsets_b_limb_2_col97, M31_262144)),
            mul(remainder_bits_col98, M31_134217728)),
            msb_col93),
            mul(M31_536870912, mid_limbs_set_col94));

        // ============ Read offsets_c (offsets_ptr_combined + 2) ============
        // memory_address_to_id_16
        m31 addr_offsets_c = add(offsets_ptr_combined, M31_2);
        m31 offsets_c_id_col100 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr_offsets_c,
            &offsets_c_id_col100
        );
        traces[100][row] = offsets_c_id_col100;
        sub_component_inputs_memory_address_to_id[16][row] = addr_offsets_c;
        lookup_memory_address_to_id_16[0][row] = addr_offsets_c;
        lookup_memory_address_to_id_16[1][row] = offsets_c_id_col100;

        // Read small value for offsets_c with signed decoding
        m31 offsets_c_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            offsets_c_id_col100,
            offsets_c_value
        );

        // Cond Decode Small Sign for offsets_c
        m31 msb_col101 = (offsets_c_value[27] == M31_256) ? M31_1 : M31_0;
        traces[101][row] = msb_col101;
        m31 mid_limbs_set_col102 = (offsets_c_value[20] == M31_511) ? M31_1 : M31_0;
        traces[102][row] = mid_limbs_set_col102;

        m31 offsets_c_limb_0_col103 = offsets_c_value[0]; traces[103][row] = offsets_c_limb_0_col103;
        m31 offsets_c_limb_1_col104 = offsets_c_value[1]; traces[104][row] = offsets_c_limb_1_col104;
        m31 offsets_c_limb_2_col105 = offsets_c_value[2]; traces[105][row] = offsets_c_limb_2_col105;
        m31 remainder_bits_col106 = m31{((uint16_t)offsets_c_value[3]) & UInt16_3};
        traces[106][row] = remainder_bits_col106;
        m31 partial_limb_msb_col107 = m31{(((uint16_t)remainder_bits_col106) & UInt16_2) >> UInt16_1};
        traces[107][row] = partial_limb_msb_col107;

        // memory_id_to_big_11 with signed encoding
        sub_component_inputs_memory_id_to_big[11][row] = offsets_c_id_col100;
        lookup_memory_id_to_big_11[0][row] = offsets_c_id_col100;
        lookup_memory_id_to_big_11[1][row] = offsets_c_limb_0_col103;
        lookup_memory_id_to_big_11[2][row] = offsets_c_limb_1_col104;
        lookup_memory_id_to_big_11[3][row] = offsets_c_limb_2_col105;
        lookup_memory_id_to_big_11[4][row] = add(remainder_bits_col106, mul(mid_limbs_set_col102, M31_508));
        for (int i = 5; i < 21; i++) {
            lookup_memory_id_to_big_11[i][row] = mul(mid_limbs_set_col102, M31_511);
        }
        lookup_memory_id_to_big_11[21][row] = sub(mul(M31_136, msb_col101), mid_limbs_set_col102);
        for (int i = 22; i < 28; i++) {
            lookup_memory_id_to_big_11[i][row] = M31_0;
        }
        lookup_memory_id_to_big_11[28][row] = mul(msb_col101, M31_256);

        // Compute offset_c value (signed)
        m31 offset_c = sub(sub(add(add(add(offsets_c_limb_0_col103,
            mul(offsets_c_limb_1_col104, M31_512)),
            mul(offsets_c_limb_2_col105, M31_262144)),
            mul(remainder_bits_col106, M31_134217728)),
            msb_col101),
            mul(M31_536870912, mid_limbs_set_col102));

        // Compute addresses for a, b, c operands
        // a_addr = values_ptr + offset_a (no multiplication - offset is direct address offset)
        m31 a_addr_base = add(values_ptr, offset_a);
        // b_addr = values_ptr + offset_b
        m31 b_addr_base = add(values_ptr, offset_b);
        // c_addr = values_ptr + offset_c
        m31 c_addr_base = add(values_ptr, offset_c);

        // Read a0 (values_ptr + offset_a)
        // memory_address_to_id_17
        m31 a0_id_col108 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            a_addr_base,
            &a0_id_col108
        );
        traces[108][row] = a0_id_col108;
        sub_component_inputs_memory_address_to_id[17][row] = a_addr_base;
        lookup_memory_address_to_id_17[0][row] = a_addr_base;
        lookup_memory_address_to_id_17[1][row] = a0_id_col108;

        // memory_id_to_big_12 - Read 11 limbs for a0
        m31 a0_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            a0_id_col108,
            a0_value
        );

        m31 a0_limb_0_col109 = a0_value[0]; traces[109][row] = a0_limb_0_col109;
        m31 a0_limb_1_col110 = a0_value[1]; traces[110][row] = a0_limb_1_col110;
        m31 a0_limb_2_col111 = a0_value[2]; traces[111][row] = a0_limb_2_col111;
        m31 a0_limb_3_col112 = a0_value[3]; traces[112][row] = a0_limb_3_col112;
        m31 a0_limb_4_col113 = a0_value[4]; traces[113][row] = a0_limb_4_col113;
        m31 a0_limb_5_col114 = a0_value[5]; traces[114][row] = a0_limb_5_col114;
        m31 a0_limb_6_col115 = a0_value[6]; traces[115][row] = a0_limb_6_col115;
        m31 a0_limb_7_col116 = a0_value[7]; traces[116][row] = a0_limb_7_col116;
        m31 a0_limb_8_col117 = a0_value[8]; traces[117][row] = a0_limb_8_col117;
        m31 a0_limb_9_col118 = a0_value[9]; traces[118][row] = a0_limb_9_col118;
        m31 a0_limb_10_col119 = a0_value[10]; traces[119][row] = a0_limb_10_col119;

        sub_component_inputs_memory_id_to_big[12][row] = a0_id_col108;
        lookup_memory_id_to_big_12[0][row] = a0_id_col108;
        lookup_memory_id_to_big_12[1][row] = a0_limb_0_col109;
        lookup_memory_id_to_big_12[2][row] = a0_limb_1_col110;
        lookup_memory_id_to_big_12[3][row] = a0_limb_2_col111;
        lookup_memory_id_to_big_12[4][row] = a0_limb_3_col112;
        lookup_memory_id_to_big_12[5][row] = a0_limb_4_col113;
        lookup_memory_id_to_big_12[6][row] = a0_limb_5_col114;
        lookup_memory_id_to_big_12[7][row] = a0_limb_6_col115;
        lookup_memory_id_to_big_12[8][row] = a0_limb_7_col116;
        lookup_memory_id_to_big_12[9][row] = a0_limb_8_col117;
        lookup_memory_id_to_big_12[10][row] = a0_limb_9_col118;
        lookup_memory_id_to_big_12[11][row] = a0_limb_10_col119;
        for (int i = 12; i < 29; i++) {
            lookup_memory_id_to_big_12[i][row] = M31_0;
        }

        // Read a1 (a_addr_base + 1)
        m31 a1_addr = add(a_addr_base, M31_1);
        m31 a1_id_col120 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, a1_addr, &a1_id_col120);
        traces[120][row] = a1_id_col120;
        sub_component_inputs_memory_address_to_id[18][row] = a1_addr;
        lookup_memory_address_to_id_18[0][row] = a1_addr;
        lookup_memory_address_to_id_18[1][row] = a1_id_col120;

        m31 a1_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, a1_id_col120, a1_value);

        m31 a1_limb_0_col121 = a1_value[0]; traces[121][row] = a1_limb_0_col121;
        m31 a1_limb_1_col122 = a1_value[1]; traces[122][row] = a1_limb_1_col122;
        m31 a1_limb_2_col123 = a1_value[2]; traces[123][row] = a1_limb_2_col123;
        m31 a1_limb_3_col124 = a1_value[3]; traces[124][row] = a1_limb_3_col124;
        m31 a1_limb_4_col125 = a1_value[4]; traces[125][row] = a1_limb_4_col125;
        m31 a1_limb_5_col126 = a1_value[5]; traces[126][row] = a1_limb_5_col126;
        m31 a1_limb_6_col127 = a1_value[6]; traces[127][row] = a1_limb_6_col127;
        m31 a1_limb_7_col128 = a1_value[7]; traces[128][row] = a1_limb_7_col128;
        m31 a1_limb_8_col129 = a1_value[8]; traces[129][row] = a1_limb_8_col129;
        m31 a1_limb_9_col130 = a1_value[9]; traces[130][row] = a1_limb_9_col130;
        m31 a1_limb_10_col131 = a1_value[10]; traces[131][row] = a1_limb_10_col131;

        sub_component_inputs_memory_id_to_big[13][row] = a1_id_col120;
        lookup_memory_id_to_big_13[0][row] = a1_id_col120;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_13[i][row] = a1_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_13[i][row] = M31_0;

        // Read a2 (a_addr_base + 2)
        m31 a2_addr = add(a_addr_base, M31_2);
        m31 a2_id_col132 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, a2_addr, &a2_id_col132);
        traces[132][row] = a2_id_col132;
        sub_component_inputs_memory_address_to_id[19][row] = a2_addr;
        lookup_memory_address_to_id_19[0][row] = a2_addr;
        lookup_memory_address_to_id_19[1][row] = a2_id_col132;

        m31 a2_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, a2_id_col132, a2_value);

        for (int i = 0; i < 11; i++) traces[133 + i][row] = a2_value[i];
        // Define explicit variables for a2 limbs used in carry computation
        m31 a2_limb_0_col133 = a2_value[0];
        m31 a2_limb_1_col134 = a2_value[1];
        m31 a2_limb_2_col135 = a2_value[2];
        m31 a2_limb_3_col136 = a2_value[3];
        m31 a2_limb_4_col137 = a2_value[4];
        m31 a2_limb_5_col138 = a2_value[5];
        m31 a2_limb_6_col139 = a2_value[6];
        m31 a2_limb_7_col140 = a2_value[7];
        m31 a2_limb_8_col141 = a2_value[8];
        m31 a2_limb_9_col142 = a2_value[9];
        m31 a2_limb_10_col143 = a2_value[10];

        sub_component_inputs_memory_id_to_big[14][row] = a2_id_col132;
        lookup_memory_id_to_big_14[0][row] = a2_id_col132;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_14[i][row] = a2_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_14[i][row] = M31_0;

        // Read a3 (a_addr_base + 3)
        m31 a3_addr = add(a_addr_base, M31_3);
        m31 a3_id_col144 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, a3_addr, &a3_id_col144);
        traces[144][row] = a3_id_col144;
        sub_component_inputs_memory_address_to_id[20][row] = a3_addr;
        lookup_memory_address_to_id_20[0][row] = a3_addr;
        lookup_memory_address_to_id_20[1][row] = a3_id_col144;

        m31 a3_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, a3_id_col144, a3_value);

        for (int i = 0; i < 11; i++) traces[145 + i][row] = a3_value[i];
        // Define explicit variables for a3 limbs used in carry computation
        m31 a3_limb_0_col145 = a3_value[0];
        m31 a3_limb_1_col146 = a3_value[1];
        m31 a3_limb_2_col147 = a3_value[2];
        m31 a3_limb_3_col148 = a3_value[3];
        m31 a3_limb_4_col149 = a3_value[4];
        m31 a3_limb_5_col150 = a3_value[5];
        m31 a3_limb_6_col151 = a3_value[6];
        m31 a3_limb_7_col152 = a3_value[7];
        m31 a3_limb_8_col153 = a3_value[8];

        sub_component_inputs_memory_id_to_big[15][row] = a3_id_col144;
        lookup_memory_id_to_big_15[0][row] = a3_id_col144;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_15[i][row] = a3_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_15[i][row] = M31_0;

        // Read b0 (b_addr_base)
        m31 b0_id_col156 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, b_addr_base, &b0_id_col156);
        traces[156][row] = b0_id_col156;
        sub_component_inputs_memory_address_to_id[21][row] = b_addr_base;
        lookup_memory_address_to_id_21[0][row] = b_addr_base;
        lookup_memory_address_to_id_21[1][row] = b0_id_col156;

        m31 b0_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, b0_id_col156, b0_value);

        m31 b0_limb_0_col157 = b0_value[0]; traces[157][row] = b0_limb_0_col157;
        m31 b0_limb_1_col158 = b0_value[1]; traces[158][row] = b0_limb_1_col158;
        m31 b0_limb_2_col159 = b0_value[2]; traces[159][row] = b0_limb_2_col159;
        m31 b0_limb_3_col160 = b0_value[3]; traces[160][row] = b0_limb_3_col160;
        m31 b0_limb_4_col161 = b0_value[4]; traces[161][row] = b0_limb_4_col161;
        m31 b0_limb_5_col162 = b0_value[5]; traces[162][row] = b0_limb_5_col162;
        m31 b0_limb_6_col163 = b0_value[6]; traces[163][row] = b0_limb_6_col163;
        m31 b0_limb_7_col164 = b0_value[7]; traces[164][row] = b0_limb_7_col164;
        m31 b0_limb_8_col165 = b0_value[8]; traces[165][row] = b0_limb_8_col165;
        m31 b0_limb_9_col166 = b0_value[9]; traces[166][row] = b0_limb_9_col166;
        m31 b0_limb_10_col167 = b0_value[10]; traces[167][row] = b0_limb_10_col167;

        sub_component_inputs_memory_id_to_big[16][row] = b0_id_col156;
        lookup_memory_id_to_big_16[0][row] = b0_id_col156;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_16[i][row] = b0_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_16[i][row] = M31_0;

        // Read b1, b2, b3 (similar pattern)
        m31 b1_addr = add(b_addr_base, M31_1);
        m31 b1_id_col168 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, b1_addr, &b1_id_col168);
        traces[168][row] = b1_id_col168;
        sub_component_inputs_memory_address_to_id[22][row] = b1_addr;
        lookup_memory_address_to_id_22[0][row] = b1_addr;
        lookup_memory_address_to_id_22[1][row] = b1_id_col168;

        m31 b1_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, b1_id_col168, b1_value);
        for (int i = 0; i < 11; i++) traces[169 + i][row] = b1_value[i];
        // Define explicit variables for b1 limbs used in carry computation
        m31 b1_limb_0_col169 = b1_value[0];
        m31 b1_limb_1_col170 = b1_value[1];
        m31 b1_limb_2_col171 = b1_value[2];
        m31 b1_limb_3_col172 = b1_value[3];
        m31 b1_limb_4_col173 = b1_value[4];
        m31 b1_limb_5_col174 = b1_value[5];
        m31 b1_limb_6_col175 = b1_value[6];
        m31 b1_limb_7_col176 = b1_value[7];
        m31 b1_limb_8_col177 = b1_value[8];
        m31 b1_limb_9_col178 = b1_value[9];
        m31 b1_limb_10_col179 = b1_value[10];

        sub_component_inputs_memory_id_to_big[17][row] = b1_id_col168;
        lookup_memory_id_to_big_17[0][row] = b1_id_col168;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_17[i][row] = b1_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_17[i][row] = M31_0;

        // Read b2
        m31 b2_addr = add(b_addr_base, M31_2);
        m31 b2_id_col180 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, b2_addr, &b2_id_col180);
        traces[180][row] = b2_id_col180;
        sub_component_inputs_memory_address_to_id[23][row] = b2_addr;
        lookup_memory_address_to_id_23[0][row] = b2_addr;
        lookup_memory_address_to_id_23[1][row] = b2_id_col180;

        m31 b2_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, b2_id_col180, b2_value);
        for (int i = 0; i < 11; i++) traces[181 + i][row] = b2_value[i];
        // Define explicit variables for b2 limbs used in carry computation
        m31 b2_limb_0_col181 = b2_value[0];
        m31 b2_limb_1_col182 = b2_value[1];
        m31 b2_limb_2_col183 = b2_value[2];
        m31 b2_limb_3_col184 = b2_value[3];
        m31 b2_limb_4_col185 = b2_value[4];
        m31 b2_limb_5_col186 = b2_value[5];
        m31 b2_limb_6_col187 = b2_value[6];
        m31 b2_limb_7_col188 = b2_value[7];
        m31 b2_limb_8_col189 = b2_value[8];
        m31 b2_limb_9_col190 = b2_value[9];
        m31 b2_limb_10_col191 = b2_value[10];

        sub_component_inputs_memory_id_to_big[18][row] = b2_id_col180;
        lookup_memory_id_to_big_18[0][row] = b2_id_col180;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_18[i][row] = b2_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_18[i][row] = M31_0;

        // Read b3
        m31 b3_addr = add(b_addr_base, M31_3);
        m31 b3_id_col192 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, b3_addr, &b3_id_col192);
        traces[192][row] = b3_id_col192;
        sub_component_inputs_memory_address_to_id[24][row] = b3_addr;
        lookup_memory_address_to_id_24[0][row] = b3_addr;
        lookup_memory_address_to_id_24[1][row] = b3_id_col192;

        m31 b3_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, b3_id_col192, b3_value);
        for (int i = 0; i < 11; i++) traces[193 + i][row] = b3_value[i];
        // Define explicit variables for b3 limbs used in carry computation
        m31 b3_limb_0_col193 = b3_value[0];
        m31 b3_limb_1_col194 = b3_value[1];
        m31 b3_limb_2_col195 = b3_value[2];
        m31 b3_limb_3_col196 = b3_value[3];
        m31 b3_limb_4_col197 = b3_value[4];
        m31 b3_limb_5_col198 = b3_value[5];
        m31 b3_limb_6_col199 = b3_value[6];
        m31 b3_limb_7_col200 = b3_value[7];
        m31 b3_limb_8_col201 = b3_value[8];

        sub_component_inputs_memory_id_to_big[19][row] = b3_id_col192;
        lookup_memory_id_to_big_19[0][row] = b3_id_col192;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_19[i][row] = b3_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_19[i][row] = M31_0;

        // Read c0
        m31 c0_id_col204 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, c_addr_base, &c0_id_col204);
        traces[204][row] = c0_id_col204;
        sub_component_inputs_memory_address_to_id[25][row] = c_addr_base;
        lookup_memory_address_to_id_25[0][row] = c_addr_base;
        lookup_memory_address_to_id_25[1][row] = c0_id_col204;

        m31 c0_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, c0_id_col204, c0_value);

        m31 c0_limb_0_col205 = c0_value[0]; traces[205][row] = c0_limb_0_col205;
        m31 c0_limb_1_col206 = c0_value[1]; traces[206][row] = c0_limb_1_col206;
        m31 c0_limb_2_col207 = c0_value[2]; traces[207][row] = c0_limb_2_col207;
        m31 c0_limb_3_col208 = c0_value[3]; traces[208][row] = c0_limb_3_col208;
        m31 c0_limb_4_col209 = c0_value[4]; traces[209][row] = c0_limb_4_col209;
        m31 c0_limb_5_col210 = c0_value[5]; traces[210][row] = c0_limb_5_col210;
        m31 c0_limb_6_col211 = c0_value[6]; traces[211][row] = c0_limb_6_col211;
        m31 c0_limb_7_col212 = c0_value[7]; traces[212][row] = c0_limb_7_col212;
        m31 c0_limb_8_col213 = c0_value[8]; traces[213][row] = c0_limb_8_col213;
        m31 c0_limb_9_col214 = c0_value[9]; traces[214][row] = c0_limb_9_col214;
        m31 c0_limb_10_col215 = c0_value[10]; traces[215][row] = c0_limb_10_col215;

        sub_component_inputs_memory_id_to_big[20][row] = c0_id_col204;
        lookup_memory_id_to_big_20[0][row] = c0_id_col204;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_20[i][row] = c0_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_20[i][row] = M31_0;

        // Read c1, c2, c3 (similar pattern)
        m31 c1_addr = add(c_addr_base, M31_1);
        m31 c1_id_col216 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, c1_addr, &c1_id_col216);
        traces[216][row] = c1_id_col216;
        sub_component_inputs_memory_address_to_id[26][row] = c1_addr;
        lookup_memory_address_to_id_26[0][row] = c1_addr;
        lookup_memory_address_to_id_26[1][row] = c1_id_col216;

        m31 c1_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, c1_id_col216, c1_value);
        for (int i = 0; i < 11; i++) traces[217 + i][row] = c1_value[i];
        // Define explicit variables for c1 limbs used in carry computation
        m31 c1_limb_0_col217 = c1_value[0];
        m31 c1_limb_1_col218 = c1_value[1];
        m31 c1_limb_2_col219 = c1_value[2];
        m31 c1_limb_3_col220 = c1_value[3];
        m31 c1_limb_4_col221 = c1_value[4];
        m31 c1_limb_5_col222 = c1_value[5];
        m31 c1_limb_6_col223 = c1_value[6];
        m31 c1_limb_7_col224 = c1_value[7];
        m31 c1_limb_8_col225 = c1_value[8];
        m31 c1_limb_9_col226 = c1_value[9];
        m31 c1_limb_10_col227 = c1_value[10];

        sub_component_inputs_memory_id_to_big[21][row] = c1_id_col216;
        lookup_memory_id_to_big_21[0][row] = c1_id_col216;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_21[i][row] = c1_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_21[i][row] = M31_0;

        // Read c2
        m31 c2_addr = add(c_addr_base, M31_2);
        m31 c2_id_col228 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, c2_addr, &c2_id_col228);
        traces[228][row] = c2_id_col228;
        sub_component_inputs_memory_address_to_id[27][row] = c2_addr;
        lookup_memory_address_to_id_27[0][row] = c2_addr;
        lookup_memory_address_to_id_27[1][row] = c2_id_col228;

        m31 c2_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, c2_id_col228, c2_value);
        for (int i = 0; i < 11; i++) traces[229 + i][row] = c2_value[i];
        // Define explicit variables for c2 limbs used in carry computation
        m31 c2_limb_0_col229 = c2_value[0];
        m31 c2_limb_1_col230 = c2_value[1];
        m31 c2_limb_2_col231 = c2_value[2];
        m31 c2_limb_3_col232 = c2_value[3];
        m31 c2_limb_4_col233 = c2_value[4];
        m31 c2_limb_5_col234 = c2_value[5];
        m31 c2_limb_6_col235 = c2_value[6];
        m31 c2_limb_7_col236 = c2_value[7];
        m31 c2_limb_8_col237 = c2_value[8];
        m31 c2_limb_9_col238 = c2_value[9];
        m31 c2_limb_10_col239 = c2_value[10];

        sub_component_inputs_memory_id_to_big[22][row] = c2_id_col228;
        lookup_memory_id_to_big_22[0][row] = c2_id_col228;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_22[i][row] = c2_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_22[i][row] = M31_0;

        // Read c3
        m31 c3_addr = add(c_addr_base, M31_3);
        m31 c3_id_col240 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, c3_addr, &c3_id_col240);
        traces[240][row] = c3_id_col240;
        sub_component_inputs_memory_address_to_id[28][row] = c3_addr;
        lookup_memory_address_to_id_28[0][row] = c3_addr;
        lookup_memory_address_to_id_28[1][row] = c3_id_col240;

        m31 c3_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, c3_id_col240, c3_value);
        for (int i = 0; i < 11; i++) traces[241 + i][row] = c3_value[i];
        // Define explicit variables for c3 limbs used in carry computation
        m31 c3_limb_0_col241 = c3_value[0];
        m31 c3_limb_1_col242 = c3_value[1];
        m31 c3_limb_2_col243 = c3_value[2];
        m31 c3_limb_3_col244 = c3_value[3];
        m31 c3_limb_4_col245 = c3_value[4];
        m31 c3_limb_5_col246 = c3_value[5];
        m31 c3_limb_6_col247 = c3_value[6];
        m31 c3_limb_7_col248 = c3_value[7];
        m31 c3_limb_8_col249 = c3_value[8];

        sub_component_inputs_memory_id_to_big[23][row] = c3_id_col240;
        lookup_memory_id_to_big_23[0][row] = c3_id_col240;
        for (int i = 1; i <= 11; i++) lookup_memory_id_to_big_23[i][row] = c3_value[i-1];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_23[i][row] = M31_0;

        // Column 252: sub_p_bit - whether to subtract p from a + b
        // Compute diff = (a + b) - c as 384-bit number, check if diff == 0
        // If diff == 0, then sub_p_bit = 0 (no subtraction of p needed)
        // If diff != 0, then sub_p_bit = 1 (we subtracted p, so a + b >= p)

        // ============== sub_p_bit computation ==============
        // Check if (a + b) == c by comparing all 44 limbs with carry propagation.
        // Each limb is 9 bits. We compute diff = a + b - c with carry propagation
        // and check if the result is all zeros.

        // Build a and b and c as 64-bit limb arrays for easier arithmetic
        int64_t a_limbs[44], b_limbs[44], c_limbs[44];
        // a0 limbs (indices 0-10)
        a_limbs[0] = a0_limb_0_col109; a_limbs[1] = a0_limb_1_col110; a_limbs[2] = a0_limb_2_col111;
        a_limbs[3] = a0_limb_3_col112; a_limbs[4] = a0_limb_4_col113; a_limbs[5] = a0_limb_5_col114;
        a_limbs[6] = a0_limb_6_col115; a_limbs[7] = a0_limb_7_col116; a_limbs[8] = a0_limb_8_col117;
        a_limbs[9] = a0_limb_9_col118; a_limbs[10] = a0_limb_10_col119;
        // a1 limbs (indices 11-21)
        a_limbs[11] = a1_limb_0_col121; a_limbs[12] = a1_limb_1_col122; a_limbs[13] = a1_limb_2_col123;
        a_limbs[14] = a1_limb_3_col124; a_limbs[15] = a1_limb_4_col125; a_limbs[16] = a1_limb_5_col126;
        a_limbs[17] = a1_limb_6_col127; a_limbs[18] = a1_limb_7_col128; a_limbs[19] = a1_limb_8_col129;
        a_limbs[20] = a1_limb_9_col130; a_limbs[21] = a1_limb_10_col131;
        // a2 limbs (indices 22-32)
        a_limbs[22] = a2_limb_0_col133; a_limbs[23] = a2_limb_1_col134; a_limbs[24] = a2_limb_2_col135;
        a_limbs[25] = a2_limb_3_col136; a_limbs[26] = a2_limb_4_col137; a_limbs[27] = a2_limb_5_col138;
        a_limbs[28] = a2_limb_6_col139; a_limbs[29] = a2_limb_7_col140; a_limbs[30] = a2_limb_8_col141;
        a_limbs[31] = a2_limb_9_col142; a_limbs[32] = a2_limb_10_col143;
        // a3 limbs (indices 33-43)
        a_limbs[33] = a3_limb_0_col145; a_limbs[34] = a3_limb_1_col146; a_limbs[35] = a3_limb_2_col147;
        a_limbs[36] = a3_limb_3_col148; a_limbs[37] = a3_limb_4_col149; a_limbs[38] = a3_limb_5_col150;
        a_limbs[39] = a3_limb_6_col151; a_limbs[40] = a3_limb_7_col152; a_limbs[41] = a3_limb_8_col153;
        a_limbs[42] = a3_value[9]; a_limbs[43] = a3_value[10];

        // b limbs
        b_limbs[0] = b0_limb_0_col157; b_limbs[1] = b0_limb_1_col158; b_limbs[2] = b0_limb_2_col159;
        b_limbs[3] = b0_limb_3_col160; b_limbs[4] = b0_limb_4_col161; b_limbs[5] = b0_limb_5_col162;
        b_limbs[6] = b0_limb_6_col163; b_limbs[7] = b0_limb_7_col164; b_limbs[8] = b0_limb_8_col165;
        b_limbs[9] = b0_limb_9_col166; b_limbs[10] = b0_limb_10_col167;
        b_limbs[11] = b1_limb_0_col169; b_limbs[12] = b1_limb_1_col170; b_limbs[13] = b1_limb_2_col171;
        b_limbs[14] = b1_limb_3_col172; b_limbs[15] = b1_limb_4_col173; b_limbs[16] = b1_limb_5_col174;
        b_limbs[17] = b1_limb_6_col175; b_limbs[18] = b1_limb_7_col176; b_limbs[19] = b1_limb_8_col177;
        b_limbs[20] = b1_limb_9_col178; b_limbs[21] = b1_limb_10_col179;
        b_limbs[22] = b2_limb_0_col181; b_limbs[23] = b2_limb_1_col182; b_limbs[24] = b2_limb_2_col183;
        b_limbs[25] = b2_limb_3_col184; b_limbs[26] = b2_limb_4_col185; b_limbs[27] = b2_limb_5_col186;
        b_limbs[28] = b2_limb_6_col187; b_limbs[29] = b2_limb_7_col188; b_limbs[30] = b2_limb_8_col189;
        b_limbs[31] = b2_limb_9_col190; b_limbs[32] = b2_limb_10_col191;
        b_limbs[33] = b3_limb_0_col193; b_limbs[34] = b3_limb_1_col194; b_limbs[35] = b3_limb_2_col195;
        b_limbs[36] = b3_limb_3_col196; b_limbs[37] = b3_limb_4_col197; b_limbs[38] = b3_limb_5_col198;
        b_limbs[39] = b3_limb_6_col199; b_limbs[40] = b3_limb_7_col200; b_limbs[41] = b3_limb_8_col201;
        b_limbs[42] = b3_value[9]; b_limbs[43] = b3_value[10];

        // c limbs
        c_limbs[0] = c0_limb_0_col205; c_limbs[1] = c0_limb_1_col206; c_limbs[2] = c0_limb_2_col207;
        c_limbs[3] = c0_limb_3_col208; c_limbs[4] = c0_limb_4_col209; c_limbs[5] = c0_limb_5_col210;
        c_limbs[6] = c0_limb_6_col211; c_limbs[7] = c0_limb_7_col212; c_limbs[8] = c0_limb_8_col213;
        c_limbs[9] = c0_limb_9_col214; c_limbs[10] = c0_limb_10_col215;
        c_limbs[11] = c1_limb_0_col217; c_limbs[12] = c1_limb_1_col218; c_limbs[13] = c1_limb_2_col219;
        c_limbs[14] = c1_limb_3_col220; c_limbs[15] = c1_limb_4_col221; c_limbs[16] = c1_limb_5_col222;
        c_limbs[17] = c1_limb_6_col223; c_limbs[18] = c1_limb_7_col224; c_limbs[19] = c1_limb_8_col225;
        c_limbs[20] = c1_limb_9_col226; c_limbs[21] = c1_limb_10_col227;
        c_limbs[22] = c2_limb_0_col229; c_limbs[23] = c2_limb_1_col230; c_limbs[24] = c2_limb_2_col231;
        c_limbs[25] = c2_limb_3_col232; c_limbs[26] = c2_limb_4_col233; c_limbs[27] = c2_limb_5_col234;
        c_limbs[28] = c2_limb_6_col235; c_limbs[29] = c2_limb_7_col236; c_limbs[30] = c2_limb_8_col237;
        c_limbs[31] = c2_limb_9_col238; c_limbs[32] = c2_limb_10_col239;
        c_limbs[33] = c3_limb_0_col241; c_limbs[34] = c3_limb_1_col242; c_limbs[35] = c3_limb_2_col243;
        c_limbs[36] = c3_limb_3_col244; c_limbs[37] = c3_limb_4_col245; c_limbs[38] = c3_limb_5_col246;
        c_limbs[39] = c3_limb_6_col247; c_limbs[40] = c3_limb_7_col248; c_limbs[41] = c3_limb_8_col249;
        c_limbs[42] = c3_value[9]; c_limbs[43] = c3_value[10];

        // Compute (a + b) as 384-bit and compare with c
        // IMPORTANT: Each felt252 contributes only 96 bits to the 384-bit BigUInt (4 × 96 = 384).
        // But 11 9-bit limbs = 99 bits. So limb index 10 of each felt252 only has 6 significant bits
        // (96 mod 9 = 6). The 96-bit boundaries are at limb array indices 10, 21, 32.
        // At these boundaries, we wrap at 6 bits (64) instead of 9 bits (512).
        uint64_t ab_limbs[44];
        uint64_t carry_ab = 0;
        for (int i = 0; i < 44; i++) {
            uint64_t sum = (uint64_t)a_limbs[i] + (uint64_t)b_limbs[i] + carry_ab;
            // Check if this is a 96-bit boundary (limb index 10, 21, or 32)
            // At these boundaries, only 6 bits are significant (96 bits = 10×9 + 6)
            if (i == 10 || i == 21 || i == 32) {
                ab_limbs[i] = sum & 0x3F; // Lower 6 bits (mask 0x3F = 63)
                carry_ab = sum >> 6;
            } else {
                ab_limbs[i] = sum & 0x1FF; // Lower 9 bits
                carry_ab = sum >> 9;
            }
        }

        // Compare (a + b) with c limb by limb, respecting the 96-bit boundaries
        bool is_ab_equal_c = true;
        for (int i = 0; i < 44; i++) {
            uint64_t c_val = (uint64_t)c_limbs[i];
            // At 96-bit boundaries, only compare lower 6 bits
            if (i == 10 || i == 21 || i == 32) {
                if (ab_limbs[i] != (c_val & 0x3F)) {
                    is_ab_equal_c = false;
                    break;
                }
            } else {
                if (ab_limbs[i] != c_val) {
                    is_ab_equal_c = false;
                    break;
                }
            }
        }
        // Also check if there's overflow carry from a+b (should be checked but normally zero for valid input)
        if (carry_ab != 0) is_ab_equal_c = false;

        // sub_p_bit = 0 if (a+b) == c (no reduction), 1 if (a+b) != c (reduction by p happened)
        m31 sub_p_bit_col252 = is_ab_equal_c ? M31_0 : M31_1;

        traces[252][row] = sub_p_bit_col252;

        // Columns 253-266: Carry values - using explicit variable names matching CPU code
        // carry_0: ((term0 + 512*term1 + 262144*term2)) * 16
        // where term[i] = a[i] + b[i] - c[i] - p[i] * sub_p_bit
        m31 carry_0_col253 = mul(
            add(add(
                sub(sub(add(a0_limb_0_col109, b0_limb_0_col157), c0_limb_0_col205), mul(p0_limb_0_col2, sub_p_bit_col252)),
                mul(M31_512, sub(sub(add(a0_limb_1_col110, b0_limb_1_col158), c0_limb_1_col206), mul(p0_limb_1_col3, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a0_limb_2_col111, b0_limb_2_col159), c0_limb_2_col207), mul(p0_limb_2_col4, sub_p_bit_col252)))),
            M31_16);
        traces[253][row] = carry_0_col253;

        // carry_1
        m31 carry_1_col254 = mul(
            add(add(add(carry_0_col253,
                sub(sub(add(a0_limb_3_col112, b0_limb_3_col160), c0_limb_3_col208), mul(p0_limb_3_col5, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a0_limb_4_col113, b0_limb_4_col161), c0_limb_4_col209), mul(p0_limb_4_col6, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a0_limb_5_col114, b0_limb_5_col162), c0_limb_5_col210), mul(p0_limb_5_col7, sub_p_bit_col252)))),
            M31_16);
        traces[254][row] = carry_1_col254;

        // carry_2
        m31 carry_2_col255 = mul(
            add(add(add(carry_1_col254,
                sub(sub(add(a0_limb_6_col115, b0_limb_6_col163), c0_limb_6_col211), mul(p0_limb_6_col8, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a0_limb_7_col116, b0_limb_7_col164), c0_limb_7_col212), mul(p0_limb_7_col9, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a0_limb_8_col117, b0_limb_8_col165), c0_limb_8_col213), mul(p0_limb_8_col10, sub_p_bit_col252)))),
            M31_16);
        traces[255][row] = carry_2_col255;

        // carry_3 (boundary a0->a1)
        m31 carry_3_col256 = mul(
            add(add(add(carry_2_col255,
                sub(sub(add(a0_limb_9_col118, b0_limb_9_col166), c0_limb_9_col214), mul(p0_limb_9_col11, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a0_limb_10_col119, b0_limb_10_col167), c0_limb_10_col215), mul(p0_limb_10_col12, sub_p_bit_col252)))),
                mul(M31_32768, sub(sub(add(a1_limb_0_col121, b1_limb_0_col169), c1_limb_0_col217), mul(p1_limb_0_col14, sub_p_bit_col252)))),
            M31_128);
        traces[256][row] = carry_3_col256;

        // carry_4
        m31 carry_4_col257 = mul(
            add(add(add(carry_3_col256,
                sub(sub(add(a1_limb_1_col122, b1_limb_1_col170), c1_limb_1_col218), mul(p1_limb_1_col15, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a1_limb_2_col123, b1_limb_2_col171), c1_limb_2_col219), mul(p1_limb_2_col16, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a1_limb_3_col124, b1_limb_3_col172), c1_limb_3_col220), mul(p1_limb_3_col17, sub_p_bit_col252)))),
            M31_16);
        traces[257][row] = carry_4_col257;

        // carry_5
        m31 carry_5_col258 = mul(
            add(add(add(carry_4_col257,
                sub(sub(add(a1_limb_4_col125, b1_limb_4_col173), c1_limb_4_col221), mul(p1_limb_4_col18, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a1_limb_5_col126, b1_limb_5_col174), c1_limb_5_col222), mul(p1_limb_5_col19, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a1_limb_6_col127, b1_limb_6_col175), c1_limb_6_col223), mul(p1_limb_6_col20, sub_p_bit_col252)))),
            M31_16);
        traces[258][row] = carry_5_col258;

        // carry_6
        m31 carry_6_col259 = mul(
            add(add(add(carry_5_col258,
                sub(sub(add(a1_limb_7_col128, b1_limb_7_col176), c1_limb_7_col224), mul(p1_limb_7_col21, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a1_limb_8_col129, b1_limb_8_col177), c1_limb_8_col225), mul(p1_limb_8_col22, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a1_limb_9_col130, b1_limb_9_col178), c1_limb_9_col226), mul(p1_limb_9_col23, sub_p_bit_col252)))),
            M31_16);
        traces[259][row] = carry_6_col259;

        // carry_7 (boundary a1->a2)
        m31 carry_7_col260 = mul(
            add(add(add(carry_6_col259,
                sub(sub(add(a1_limb_10_col131, b1_limb_10_col179), c1_limb_10_col227), mul(p1_limb_10_col24, sub_p_bit_col252))),
                mul(M31_64, sub(sub(add(a2_limb_0_col133, b2_limb_0_col181), c2_limb_0_col229), mul(p2_limb_0_col26, sub_p_bit_col252)))),
                mul(M31_32768, sub(sub(add(a2_limb_1_col134, b2_limb_1_col182), c2_limb_1_col230), mul(p2_limb_1_col27, sub_p_bit_col252)))),
            M31_128);
        traces[260][row] = carry_7_col260;

        // carry_8
        m31 carry_8_col261 = mul(
            add(add(add(carry_7_col260,
                sub(sub(add(a2_limb_2_col135, b2_limb_2_col183), c2_limb_2_col231), mul(p2_limb_2_col28, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a2_limb_3_col136, b2_limb_3_col184), c2_limb_3_col232), mul(p2_limb_3_col29, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a2_limb_4_col137, b2_limb_4_col185), c2_limb_4_col233), mul(p2_limb_4_col30, sub_p_bit_col252)))),
            M31_16);
        traces[261][row] = carry_8_col261;

        // carry_9
        m31 carry_9_col262 = mul(
            add(add(add(carry_8_col261,
                sub(sub(add(a2_limb_5_col138, b2_limb_5_col186), c2_limb_5_col234), mul(p2_limb_5_col31, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a2_limb_6_col139, b2_limb_6_col187), c2_limb_6_col235), mul(p2_limb_6_col32, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a2_limb_7_col140, b2_limb_7_col188), c2_limb_7_col236), mul(p2_limb_7_col33, sub_p_bit_col252)))),
            M31_16);
        traces[262][row] = carry_9_col262;

        // carry_10 (boundary a2->a3)
        m31 carry_10_col263 = mul(
            add(add(add(carry_9_col262,
                sub(sub(add(a2_limb_8_col141, b2_limb_8_col189), c2_limb_8_col237), mul(p2_limb_8_col34, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a2_limb_9_col142, b2_limb_9_col190), c2_limb_9_col238), mul(p2_limb_9_col35, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a2_limb_10_col143, b2_limb_10_col191), c2_limb_10_col239), mul(p2_limb_10_col36, sub_p_bit_col252)))),
            M31_128);
        traces[263][row] = carry_10_col263;

        // carry_11
        m31 carry_11_col264 = mul(
            add(add(add(carry_10_col263,
                sub(sub(add(a3_limb_0_col145, b3_limb_0_col193), c3_limb_0_col241), mul(p3_limb_0_col38, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a3_limb_1_col146, b3_limb_1_col194), c3_limb_1_col242), mul(p3_limb_1_col39, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a3_limb_2_col147, b3_limb_2_col195), c3_limb_2_col243), mul(p3_limb_2_col40, sub_p_bit_col252)))),
            M31_16);
        traces[264][row] = carry_11_col264;

        // carry_12: a3_limb_3-5, b3_limb_3-5, c3_limb_3-5, p3_limb_3-5
        m31 carry_12_col265 = mul(
            add(add(add(carry_11_col264,
                sub(sub(add(a3_limb_3_col148, b3_limb_3_col196), c3_limb_3_col244), mul(p3_limb_3_col41, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a3_limb_4_col149, b3_limb_4_col197), c3_limb_4_col245), mul(p3_limb_4_col42, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a3_limb_5_col150, b3_limb_5_col198), c3_limb_5_col246), mul(p3_limb_5_col43, sub_p_bit_col252)))),
            M31_16);
        traces[265][row] = carry_12_col265;

        // carry_13: a3_limb_6-8, b3_limb_6-8, c3_limb_6-8, p3_limb_6-8
        m31 carry_13_col266 = mul(
            add(add(add(carry_12_col265,
                sub(sub(add(a3_limb_6_col151, b3_limb_6_col199), c3_limb_6_col247), mul(p3_limb_6_col44, sub_p_bit_col252))),
                mul(M31_512, sub(sub(add(a3_limb_7_col152, b3_limb_7_col200), c3_limb_7_col248), mul(p3_limb_7_col45, sub_p_bit_col252)))),
                mul(M31_262144, sub(sub(add(a3_limb_8_col153, b3_limb_8_col201), c3_limb_8_col249), mul(p3_limb_8_col46, sub_p_bit_col252)))),
            M31_16);
        traces[266][row] = carry_13_col266;
    }
}

// Wrapper function to launch base trace kernel
extern "C"
void generate_add_mod_builtin_traces(
    unsigned **traces,

    // Lookup data arrays - 29 MemoryAddressToId lookups
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_address_to_id_3,
    unsigned **lookup_memory_address_to_id_4,
    unsigned **lookup_memory_address_to_id_5,
    unsigned **lookup_memory_address_to_id_6,
    unsigned **lookup_memory_address_to_id_7,
    unsigned **lookup_memory_address_to_id_8,
    unsigned **lookup_memory_address_to_id_9,
    unsigned **lookup_memory_address_to_id_10,
    unsigned **lookup_memory_address_to_id_11,
    unsigned **lookup_memory_address_to_id_12,
    unsigned **lookup_memory_address_to_id_13,
    unsigned **lookup_memory_address_to_id_14,
    unsigned **lookup_memory_address_to_id_15,
    unsigned **lookup_memory_address_to_id_16,
    unsigned **lookup_memory_address_to_id_17,
    unsigned **lookup_memory_address_to_id_18,
    unsigned **lookup_memory_address_to_id_19,
    unsigned **lookup_memory_address_to_id_20,
    unsigned **lookup_memory_address_to_id_21,
    unsigned **lookup_memory_address_to_id_22,
    unsigned **lookup_memory_address_to_id_23,
    unsigned **lookup_memory_address_to_id_24,
    unsigned **lookup_memory_address_to_id_25,
    unsigned **lookup_memory_address_to_id_26,
    unsigned **lookup_memory_address_to_id_27,
    unsigned **lookup_memory_address_to_id_28,

    // Lookup data arrays - 24 MemoryIdToBig lookups
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_memory_id_to_big_3,
    unsigned **lookup_memory_id_to_big_4,
    unsigned **lookup_memory_id_to_big_5,
    unsigned **lookup_memory_id_to_big_6,
    unsigned **lookup_memory_id_to_big_7,
    unsigned **lookup_memory_id_to_big_8,
    unsigned **lookup_memory_id_to_big_9,
    unsigned **lookup_memory_id_to_big_10,
    unsigned **lookup_memory_id_to_big_11,
    unsigned **lookup_memory_id_to_big_12,
    unsigned **lookup_memory_id_to_big_13,
    unsigned **lookup_memory_id_to_big_14,
    unsigned **lookup_memory_id_to_big_15,
    unsigned **lookup_memory_id_to_big_16,
    unsigned **lookup_memory_id_to_big_17,
    unsigned **lookup_memory_id_to_big_18,
    unsigned **lookup_memory_id_to_big_19,
    unsigned **lookup_memory_id_to_big_20,
    unsigned **lookup_memory_id_to_big_21,
    unsigned **lookup_memory_id_to_big_22,
    unsigned **lookup_memory_id_to_big_23,

    // Sub-component inputs
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    timer global_timer;
    global_timer.start("generate add_mod_builtin base trace");
    unsigned trace_size = 1 << log_size;

    // Clone all pointer arrays to device memory
    m31 **device_traces = clone_to_device<m31*>(traces, ADD_MOD_BUILTIN_N_TRACE_COLUMNS);

    // 29 memory_address_to_id lookups (2 elements each)
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31*>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31*>(lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_address_to_id_3 = clone_to_device<m31*>(lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_memory_address_to_id_4 = clone_to_device<m31*>(lookup_memory_address_to_id_4, 2);
    m31 **device_lookup_memory_address_to_id_5 = clone_to_device<m31*>(lookup_memory_address_to_id_5, 2);
    m31 **device_lookup_memory_address_to_id_6 = clone_to_device<m31*>(lookup_memory_address_to_id_6, 2);
    m31 **device_lookup_memory_address_to_id_7 = clone_to_device<m31*>(lookup_memory_address_to_id_7, 2);
    m31 **device_lookup_memory_address_to_id_8 = clone_to_device<m31*>(lookup_memory_address_to_id_8, 2);
    m31 **device_lookup_memory_address_to_id_9 = clone_to_device<m31*>(lookup_memory_address_to_id_9, 2);
    m31 **device_lookup_memory_address_to_id_10 = clone_to_device<m31*>(lookup_memory_address_to_id_10, 2);
    m31 **device_lookup_memory_address_to_id_11 = clone_to_device<m31*>(lookup_memory_address_to_id_11, 2);
    m31 **device_lookup_memory_address_to_id_12 = clone_to_device<m31*>(lookup_memory_address_to_id_12, 2);
    m31 **device_lookup_memory_address_to_id_13 = clone_to_device<m31*>(lookup_memory_address_to_id_13, 2);
    m31 **device_lookup_memory_address_to_id_14 = clone_to_device<m31*>(lookup_memory_address_to_id_14, 2);
    m31 **device_lookup_memory_address_to_id_15 = clone_to_device<m31*>(lookup_memory_address_to_id_15, 2);
    m31 **device_lookup_memory_address_to_id_16 = clone_to_device<m31*>(lookup_memory_address_to_id_16, 2);
    m31 **device_lookup_memory_address_to_id_17 = clone_to_device<m31*>(lookup_memory_address_to_id_17, 2);
    m31 **device_lookup_memory_address_to_id_18 = clone_to_device<m31*>(lookup_memory_address_to_id_18, 2);
    m31 **device_lookup_memory_address_to_id_19 = clone_to_device<m31*>(lookup_memory_address_to_id_19, 2);
    m31 **device_lookup_memory_address_to_id_20 = clone_to_device<m31*>(lookup_memory_address_to_id_20, 2);
    m31 **device_lookup_memory_address_to_id_21 = clone_to_device<m31*>(lookup_memory_address_to_id_21, 2);
    m31 **device_lookup_memory_address_to_id_22 = clone_to_device<m31*>(lookup_memory_address_to_id_22, 2);
    m31 **device_lookup_memory_address_to_id_23 = clone_to_device<m31*>(lookup_memory_address_to_id_23, 2);
    m31 **device_lookup_memory_address_to_id_24 = clone_to_device<m31*>(lookup_memory_address_to_id_24, 2);
    m31 **device_lookup_memory_address_to_id_25 = clone_to_device<m31*>(lookup_memory_address_to_id_25, 2);
    m31 **device_lookup_memory_address_to_id_26 = clone_to_device<m31*>(lookup_memory_address_to_id_26, 2);
    m31 **device_lookup_memory_address_to_id_27 = clone_to_device<m31*>(lookup_memory_address_to_id_27, 2);
    m31 **device_lookup_memory_address_to_id_28 = clone_to_device<m31*>(lookup_memory_address_to_id_28, 2);

    // 24 memory_id_to_big lookups (29 elements each)
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31*>(lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31*>(lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_memory_id_to_big_3 = clone_to_device<m31*>(lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_memory_id_to_big_4 = clone_to_device<m31*>(lookup_memory_id_to_big_4, 29);
    m31 **device_lookup_memory_id_to_big_5 = clone_to_device<m31*>(lookup_memory_id_to_big_5, 29);
    m31 **device_lookup_memory_id_to_big_6 = clone_to_device<m31*>(lookup_memory_id_to_big_6, 29);
    m31 **device_lookup_memory_id_to_big_7 = clone_to_device<m31*>(lookup_memory_id_to_big_7, 29);
    m31 **device_lookup_memory_id_to_big_8 = clone_to_device<m31*>(lookup_memory_id_to_big_8, 29);
    m31 **device_lookup_memory_id_to_big_9 = clone_to_device<m31*>(lookup_memory_id_to_big_9, 29);
    m31 **device_lookup_memory_id_to_big_10 = clone_to_device<m31*>(lookup_memory_id_to_big_10, 29);
    m31 **device_lookup_memory_id_to_big_11 = clone_to_device<m31*>(lookup_memory_id_to_big_11, 29);
    m31 **device_lookup_memory_id_to_big_12 = clone_to_device<m31*>(lookup_memory_id_to_big_12, 29);
    m31 **device_lookup_memory_id_to_big_13 = clone_to_device<m31*>(lookup_memory_id_to_big_13, 29);
    m31 **device_lookup_memory_id_to_big_14 = clone_to_device<m31*>(lookup_memory_id_to_big_14, 29);
    m31 **device_lookup_memory_id_to_big_15 = clone_to_device<m31*>(lookup_memory_id_to_big_15, 29);
    m31 **device_lookup_memory_id_to_big_16 = clone_to_device<m31*>(lookup_memory_id_to_big_16, 29);
    m31 **device_lookup_memory_id_to_big_17 = clone_to_device<m31*>(lookup_memory_id_to_big_17, 29);
    m31 **device_lookup_memory_id_to_big_18 = clone_to_device<m31*>(lookup_memory_id_to_big_18, 29);
    m31 **device_lookup_memory_id_to_big_19 = clone_to_device<m31*>(lookup_memory_id_to_big_19, 29);
    m31 **device_lookup_memory_id_to_big_20 = clone_to_device<m31*>(lookup_memory_id_to_big_20, 29);
    m31 **device_lookup_memory_id_to_big_21 = clone_to_device<m31*>(lookup_memory_id_to_big_21, 29);
    m31 **device_lookup_memory_id_to_big_22 = clone_to_device<m31*>(lookup_memory_id_to_big_22, 29);
    m31 **device_lookup_memory_id_to_big_23 = clone_to_device<m31*>(lookup_memory_id_to_big_23, 29);

    // Sub-component inputs (29 memory_address_to_id, 24 memory_id_to_big)
    m31 **device_sub_component_inputs_memory_address_to_id = clone_to_device<m31*>(sub_component_inputs_memory_address_to_id, 29);
    m31 **device_sub_component_inputs_memory_id_to_big = clone_to_device<m31*>(sub_component_inputs_memory_id_to_big, 24);

    // memory_id_to_big transposed big values (8 elements)
    m31 **device_memory_id_to_big_transposed_big_values = clone_to_device<m31*>(memory_id_to_big_transposed_big_values, 8);

    dim3 block_dim(ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 1, 1);
    dim3 grid_dim((trace_size + block_dim.x - 1) / block_dim.x, 1, 1);

    generate_add_mod_builtin_trace_kernel<<<grid_dim, block_dim>>>(
        device_traces,
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
        device_lookup_memory_address_to_id_16,
        device_lookup_memory_address_to_id_17,
        device_lookup_memory_address_to_id_18,
        device_lookup_memory_address_to_id_19,
        device_lookup_memory_address_to_id_20,
        device_lookup_memory_address_to_id_21,
        device_lookup_memory_address_to_id_22,
        device_lookup_memory_address_to_id_23,
        device_lookup_memory_address_to_id_24,
        device_lookup_memory_address_to_id_25,
        device_lookup_memory_address_to_id_26,
        device_lookup_memory_address_to_id_27,
        device_lookup_memory_address_to_id_28,
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
        device_lookup_memory_id_to_big_16,
        device_lookup_memory_id_to_big_17,
        device_lookup_memory_id_to_big_18,
        device_lookup_memory_id_to_big_19,
        device_lookup_memory_id_to_big_20,
        device_lookup_memory_id_to_big_21,
        device_lookup_memory_id_to_big_22,
        device_lookup_memory_id_to_big_23,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        segment_start,
        memory_address_to_id_address_to_raw_id,
        device_memory_id_to_big_transposed_big_values,
        memory_id_to_big_small_values,
        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Free device memory for pointer arrays
    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_address_to_id_3);
    cuda_free_memory(device_lookup_memory_address_to_id_4);
    cuda_free_memory(device_lookup_memory_address_to_id_5);
    cuda_free_memory(device_lookup_memory_address_to_id_6);
    cuda_free_memory(device_lookup_memory_address_to_id_7);
    cuda_free_memory(device_lookup_memory_address_to_id_8);
    cuda_free_memory(device_lookup_memory_address_to_id_9);
    cuda_free_memory(device_lookup_memory_address_to_id_10);
    cuda_free_memory(device_lookup_memory_address_to_id_11);
    cuda_free_memory(device_lookup_memory_address_to_id_12);
    cuda_free_memory(device_lookup_memory_address_to_id_13);
    cuda_free_memory(device_lookup_memory_address_to_id_14);
    cuda_free_memory(device_lookup_memory_address_to_id_15);
    cuda_free_memory(device_lookup_memory_address_to_id_16);
    cuda_free_memory(device_lookup_memory_address_to_id_17);
    cuda_free_memory(device_lookup_memory_address_to_id_18);
    cuda_free_memory(device_lookup_memory_address_to_id_19);
    cuda_free_memory(device_lookup_memory_address_to_id_20);
    cuda_free_memory(device_lookup_memory_address_to_id_21);
    cuda_free_memory(device_lookup_memory_address_to_id_22);
    cuda_free_memory(device_lookup_memory_address_to_id_23);
    cuda_free_memory(device_lookup_memory_address_to_id_24);
    cuda_free_memory(device_lookup_memory_address_to_id_25);
    cuda_free_memory(device_lookup_memory_address_to_id_26);
    cuda_free_memory(device_lookup_memory_address_to_id_27);
    cuda_free_memory(device_lookup_memory_address_to_id_28);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_memory_id_to_big_3);
    cuda_free_memory(device_lookup_memory_id_to_big_4);
    cuda_free_memory(device_lookup_memory_id_to_big_5);
    cuda_free_memory(device_lookup_memory_id_to_big_6);
    cuda_free_memory(device_lookup_memory_id_to_big_7);
    cuda_free_memory(device_lookup_memory_id_to_big_8);
    cuda_free_memory(device_lookup_memory_id_to_big_9);
    cuda_free_memory(device_lookup_memory_id_to_big_10);
    cuda_free_memory(device_lookup_memory_id_to_big_11);
    cuda_free_memory(device_lookup_memory_id_to_big_12);
    cuda_free_memory(device_lookup_memory_id_to_big_13);
    cuda_free_memory(device_lookup_memory_id_to_big_14);
    cuda_free_memory(device_lookup_memory_id_to_big_15);
    cuda_free_memory(device_lookup_memory_id_to_big_16);
    cuda_free_memory(device_lookup_memory_id_to_big_17);
    cuda_free_memory(device_lookup_memory_id_to_big_18);
    cuda_free_memory(device_lookup_memory_id_to_big_19);
    cuda_free_memory(device_lookup_memory_id_to_big_20);
    cuda_free_memory(device_lookup_memory_id_to_big_21);
    cuda_free_memory(device_lookup_memory_id_to_big_22);
    cuda_free_memory(device_lookup_memory_id_to_big_23);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);
    cuda_free_memory(device_memory_id_to_big_transposed_big_values);

    global_timer.end("generate add_mod_builtin base trace");
}

// ============================================================================
// Interaction trace generation
// ============================================================================

// Interaction trace column generation kernel template
// Combines two lookup denominators: denom0 + denom1 / (denom0 * denom1)
template <int N, int M>
__launch_bounds__(ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_add_mod_builtin_interaction_col_gen_kernel(
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

// Last column kernel (single lookup)
template <int N>
__launch_bounds__(ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_add_mod_builtin_interaction_last_col_kernel(
    LookupElementsBasic<N> *lookup_elements,
    m31 **lookup_state,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 combine_reg[N] = {};

    for (int i = 0; i < N; i++) {
        combine_reg[i] = lookup_state[i][vec_index];
    }
    if (vec_index < trace_size) {
        qm31 denom = lookup_elements->combine(combine_reg, N);
        qm31 one = {1, 0, 0, 0};
        logup_col_write_frac(vec_index, one, denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize column kernel - accumulates interaction trace values
__global__ void generate_add_mod_builtin_interaction_finalize_col_kernel(
    unsigned col_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = (col_index == 0) ? -1 : static_cast<int>(col_index - 1);

    if (vec_index < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[vec_index], numerator1[vec_index]},
                cm31{numerator2[vec_index], numerator3[vec_index]}
            },
            denom_inv_ptr[vec_index]
        );

        if (pre_index == -1) {
            interaction_traces[0][vec_index] = {0};
            interaction_traces[1][vec_index] = {0};
            interaction_traces[2][vec_index] = {0};
            interaction_traces[3][vec_index] = {0};
            qm31 tmp = value;
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        } else {
            qm31 prev_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][vec_index],
                     interaction_traces[pre_index * 4 + 1][vec_index]},
                cm31{interaction_traces[pre_index * 4 + 2][vec_index],
                     interaction_traces[pre_index * 4 + 3][vec_index]}
            };
            qm31 tmp = add(value, prev_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        }

        interaction_traces[col_index * 4 + 0][vec_index] = numerator0[vec_index];
        interaction_traces[col_index * 4 + 1][vec_index] = numerator1[vec_index];
        interaction_traces[col_index * 4 + 2][vec_index] = numerator2[vec_index];
        interaction_traces[col_index * 4 + 3][vec_index] = numerator3[vec_index];
    }
}

// Cumsum shift kernel - computes the sum for shifting
__global__ void generate_add_mod_builtin_interaction_cumsum_shift_kernel(
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

// Coord prefix sum kernel - applies the shift
__global__ void generate_add_mod_builtin_interaction_coord_prefix_sum_kernel(
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

extern "C"
void generate_add_mod_builtin_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,

    // Lookup data arrays - 29 MemoryAddressToId lookups
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_address_to_id_3,
    unsigned **lookup_memory_address_to_id_4,
    unsigned **lookup_memory_address_to_id_5,
    unsigned **lookup_memory_address_to_id_6,
    unsigned **lookup_memory_address_to_id_7,
    unsigned **lookup_memory_address_to_id_8,
    unsigned **lookup_memory_address_to_id_9,
    unsigned **lookup_memory_address_to_id_10,
    unsigned **lookup_memory_address_to_id_11,
    unsigned **lookup_memory_address_to_id_12,
    unsigned **lookup_memory_address_to_id_13,
    unsigned **lookup_memory_address_to_id_14,
    unsigned **lookup_memory_address_to_id_15,
    unsigned **lookup_memory_address_to_id_16,
    unsigned **lookup_memory_address_to_id_17,
    unsigned **lookup_memory_address_to_id_18,
    unsigned **lookup_memory_address_to_id_19,
    unsigned **lookup_memory_address_to_id_20,
    unsigned **lookup_memory_address_to_id_21,
    unsigned **lookup_memory_address_to_id_22,
    unsigned **lookup_memory_address_to_id_23,
    unsigned **lookup_memory_address_to_id_24,
    unsigned **lookup_memory_address_to_id_25,
    unsigned **lookup_memory_address_to_id_26,
    unsigned **lookup_memory_address_to_id_27,
    unsigned **lookup_memory_address_to_id_28,

    // Lookup data arrays - 24 MemoryIdToBig lookups
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_memory_id_to_big_3,
    unsigned **lookup_memory_id_to_big_4,
    unsigned **lookup_memory_id_to_big_5,
    unsigned **lookup_memory_id_to_big_6,
    unsigned **lookup_memory_id_to_big_7,
    unsigned **lookup_memory_id_to_big_8,
    unsigned **lookup_memory_id_to_big_9,
    unsigned **lookup_memory_id_to_big_10,
    unsigned **lookup_memory_id_to_big_11,
    unsigned **lookup_memory_id_to_big_12,
    unsigned **lookup_memory_id_to_big_13,
    unsigned **lookup_memory_id_to_big_14,
    unsigned **lookup_memory_id_to_big_15,
    unsigned **lookup_memory_id_to_big_16,
    unsigned **lookup_memory_id_to_big_17,
    unsigned **lookup_memory_id_to_big_18,
    unsigned **lookup_memory_id_to_big_19,
    unsigned **lookup_memory_id_to_big_20,
    unsigned **lookup_memory_id_to_big_21,
    unsigned **lookup_memory_id_to_big_22,
    unsigned **lookup_memory_id_to_big_23,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate add_mod_builtin interaction trace");
    unsigned trace_size = 1 << log_size;

    int block_dim_val = trace_size < ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim_val < ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim_val - 1) / block_dim_val;

    // Allocate temporary storage
    qm31 *denom_ptr = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    // Clone lookup elements to device
    LookupElementsBasic<2> *mem_addr_to_id_host = (LookupElementsBasic<2>*)memory_address_to_id;
    LookupElementsBasic<29> *mem_id_to_big_host = (LookupElementsBasic<29>*)memory_id_to_big;
    LookupElementsBasic<2> *device_mem_addr_to_id = cuda_malloc<LookupElementsBasic<2>>(1);
    LookupElementsBasic<29> *device_mem_id_to_big = cuda_malloc<LookupElementsBasic<29>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<2>>(mem_addr_to_id_host, device_mem_addr_to_id, 1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<29>>(mem_id_to_big_host, device_mem_id_to_big, 1);

    m31 **device_interaction_traces = clone_to_device<m31*>((m31**)interaction_trace, 4 * ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS);

    // Clone lookup arrays to device
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_address_to_id_3 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_memory_address_to_id_4 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_4, 2);
    m31 **device_lookup_memory_address_to_id_5 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_5, 2);
    m31 **device_lookup_memory_address_to_id_6 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_6, 2);
    m31 **device_lookup_memory_address_to_id_7 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_7, 2);
    m31 **device_lookup_memory_address_to_id_8 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_8, 2);
    m31 **device_lookup_memory_address_to_id_9 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_9, 2);
    m31 **device_lookup_memory_address_to_id_10 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_10, 2);
    m31 **device_lookup_memory_address_to_id_11 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_11, 2);
    m31 **device_lookup_memory_address_to_id_12 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_12, 2);
    m31 **device_lookup_memory_address_to_id_13 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_13, 2);
    m31 **device_lookup_memory_address_to_id_14 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_14, 2);
    m31 **device_lookup_memory_address_to_id_15 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_15, 2);
    m31 **device_lookup_memory_address_to_id_16 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_16, 2);
    m31 **device_lookup_memory_address_to_id_17 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_17, 2);
    m31 **device_lookup_memory_address_to_id_18 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_18, 2);
    m31 **device_lookup_memory_address_to_id_19 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_19, 2);
    m31 **device_lookup_memory_address_to_id_20 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_20, 2);
    m31 **device_lookup_memory_address_to_id_21 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_21, 2);
    m31 **device_lookup_memory_address_to_id_22 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_22, 2);
    m31 **device_lookup_memory_address_to_id_23 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_23, 2);
    m31 **device_lookup_memory_address_to_id_24 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_24, 2);
    m31 **device_lookup_memory_address_to_id_25 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_25, 2);
    m31 **device_lookup_memory_address_to_id_26 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_26, 2);
    m31 **device_lookup_memory_address_to_id_27 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_27, 2);
    m31 **device_lookup_memory_address_to_id_28 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_28, 2);

    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_memory_id_to_big_3 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_memory_id_to_big_4 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_4, 29);
    m31 **device_lookup_memory_id_to_big_5 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_5, 29);
    m31 **device_lookup_memory_id_to_big_6 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_6, 29);
    m31 **device_lookup_memory_id_to_big_7 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_7, 29);
    m31 **device_lookup_memory_id_to_big_8 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_8, 29);
    m31 **device_lookup_memory_id_to_big_9 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_9, 29);
    m31 **device_lookup_memory_id_to_big_10 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_10, 29);
    m31 **device_lookup_memory_id_to_big_11 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_11, 29);
    m31 **device_lookup_memory_id_to_big_12 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_12, 29);
    m31 **device_lookup_memory_id_to_big_13 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_13, 29);
    m31 **device_lookup_memory_id_to_big_14 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_14, 29);
    m31 **device_lookup_memory_id_to_big_15 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_15, 29);
    m31 **device_lookup_memory_id_to_big_16 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_16, 29);
    m31 **device_lookup_memory_id_to_big_17 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_17, 29);
    m31 **device_lookup_memory_id_to_big_18 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_18, 29);
    m31 **device_lookup_memory_id_to_big_19 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_19, 29);
    m31 **device_lookup_memory_id_to_big_20 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_20, 29);
    m31 **device_lookup_memory_id_to_big_21 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_21, 29);
    m31 **device_lookup_memory_id_to_big_22 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_22, 29);
    m31 **device_lookup_memory_id_to_big_23 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_23, 29);

    // Lambda for finalizing each column
    auto launch_finalize = [&](unsigned col_index) {
        batch_inverse_secure_field(denom_ptr, denom_inv, trace_size);
        generate_add_mod_builtin_interaction_finalize_col_kernel<<<num_blocks, block_dim_val>>>(
            col_index, trace_size, denom_inv,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3,
            device_interaction_traces
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    };

    // 27 interaction trace columns (27 * 4 = 108 M31 columns)
    // Columns 0-8: pair(memory_address_to_id_n, memory_id_to_big_n) for n=0..8
    // Column 9: pair(memory_address_to_id_9, memory_address_to_id_10)
    // Column 10: pair(memory_address_to_id_11, memory_address_to_id_12)
    // Column 11: pair(memory_address_to_id_13, memory_address_to_id_14)
    // Columns 12-25: pair(memory_id_to_big_n, memory_address_to_id_m)
    // Column 26: single memory_id_to_big_23

    // Column 0: pair(memory_address_to_id_0, memory_id_to_big_0)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_0, device_lookup_memory_id_to_big_0,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(0);

    // Column 1: pair(memory_address_to_id_1, memory_id_to_big_1)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_1, device_lookup_memory_id_to_big_1,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(1);

    // Column 2: pair(memory_address_to_id_2, memory_id_to_big_2)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_2, device_lookup_memory_id_to_big_2,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(2);

    // Column 3: pair(memory_address_to_id_3, memory_id_to_big_3)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_3, device_lookup_memory_id_to_big_3,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(3);

    // Column 4: pair(memory_address_to_id_4, memory_id_to_big_4)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_4, device_lookup_memory_id_to_big_4,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(4);

    // Column 5: pair(memory_address_to_id_5, memory_id_to_big_5)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_5, device_lookup_memory_id_to_big_5,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(5);

    // Column 6: pair(memory_address_to_id_6, memory_id_to_big_6)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_6, device_lookup_memory_id_to_big_6,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(6);

    // Column 7: pair(memory_address_to_id_7, memory_id_to_big_7)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_7, device_lookup_memory_id_to_big_7,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(7);

    // Column 8: pair(memory_address_to_id_8, memory_id_to_big_8)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_memory_address_to_id_8, device_lookup_memory_id_to_big_8,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(8);

    // Column 9: pair(memory_address_to_id_9, memory_address_to_id_10)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 2><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_addr_to_id,
        device_lookup_memory_address_to_id_9, device_lookup_memory_address_to_id_10,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(9);

    // Column 10: pair(memory_address_to_id_11, memory_address_to_id_12)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 2><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_addr_to_id,
        device_lookup_memory_address_to_id_11, device_lookup_memory_address_to_id_12,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(10);

    // Column 11: pair(memory_address_to_id_13, memory_address_to_id_14)
    generate_add_mod_builtin_interaction_col_gen_kernel<2, 2><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_addr_to_id,
        device_lookup_memory_address_to_id_13, device_lookup_memory_address_to_id_14,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(11);

    // Column 12: pair(memory_id_to_big_9, memory_address_to_id_15)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_9, device_lookup_memory_address_to_id_15,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(12);

    // Column 13: pair(memory_id_to_big_10, memory_address_to_id_16)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_10, device_lookup_memory_address_to_id_16,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(13);

    // Column 14: pair(memory_id_to_big_11, memory_address_to_id_17)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_11, device_lookup_memory_address_to_id_17,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(14);

    // Column 15: pair(memory_id_to_big_12, memory_address_to_id_18)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_12, device_lookup_memory_address_to_id_18,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(15);

    // Column 16: pair(memory_id_to_big_13, memory_address_to_id_19)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_13, device_lookup_memory_address_to_id_19,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(16);

    // Column 17: pair(memory_id_to_big_14, memory_address_to_id_20)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_14, device_lookup_memory_address_to_id_20,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(17);

    // Column 18: pair(memory_id_to_big_15, memory_address_to_id_21)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_15, device_lookup_memory_address_to_id_21,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(18);

    // Column 19: pair(memory_id_to_big_16, memory_address_to_id_22)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_16, device_lookup_memory_address_to_id_22,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(19);

    // Column 20: pair(memory_id_to_big_17, memory_address_to_id_23)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_17, device_lookup_memory_address_to_id_23,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(20);

    // Column 21: pair(memory_id_to_big_18, memory_address_to_id_24)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_18, device_lookup_memory_address_to_id_24,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(21);

    // Column 22: pair(memory_id_to_big_19, memory_address_to_id_25)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_19, device_lookup_memory_address_to_id_25,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(22);

    // Column 23: pair(memory_id_to_big_20, memory_address_to_id_26)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_20, device_lookup_memory_address_to_id_26,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(23);

    // Column 24: pair(memory_id_to_big_21, memory_address_to_id_27)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_21, device_lookup_memory_address_to_id_27,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(24);

    // Column 25: pair(memory_id_to_big_22, memory_address_to_id_28)
    generate_add_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_mem_addr_to_id,
        device_lookup_memory_id_to_big_22, device_lookup_memory_address_to_id_28,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(25);

    // Column 26 (last): single memory_id_to_big_23
    generate_add_mod_builtin_interaction_last_col_kernel<29><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big,
        device_lookup_memory_id_to_big_23,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(26);

    // Compute cumsum_shift and apply coord_prefix_sum
    {
        size_t shared_size = 4 * block_dim_val * sizeof(m31);
        generate_add_mod_builtin_interaction_cumsum_shift_kernel<<<num_blocks, block_dim_val, shared_size>>>(
            ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS,
            trace_size,
            device_interaction_traces,
            (m31*)claimed_sum
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        generate_add_mod_builtin_interaction_coord_prefix_sum_kernel<<<num_blocks, block_dim_val>>>(
            (m31*)claimed_sum,
            ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS,
            trace_size,
            device_interaction_traces
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // Apply inclusive_prefix_sum only to the last 4 columns
        inclusive_prefix_sum(interaction_trace[4 * ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);
    }

    // Free device memory
    cuda_free_memory(denom_ptr);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_mem_addr_to_id);
    cuda_free_memory(device_mem_id_to_big);
    cuda_free_memory(device_interaction_traces);

    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_address_to_id_3);
    cuda_free_memory(device_lookup_memory_address_to_id_4);
    cuda_free_memory(device_lookup_memory_address_to_id_5);
    cuda_free_memory(device_lookup_memory_address_to_id_6);
    cuda_free_memory(device_lookup_memory_address_to_id_7);
    cuda_free_memory(device_lookup_memory_address_to_id_8);
    cuda_free_memory(device_lookup_memory_address_to_id_9);
    cuda_free_memory(device_lookup_memory_address_to_id_10);
    cuda_free_memory(device_lookup_memory_address_to_id_11);
    cuda_free_memory(device_lookup_memory_address_to_id_12);
    cuda_free_memory(device_lookup_memory_address_to_id_13);
    cuda_free_memory(device_lookup_memory_address_to_id_14);
    cuda_free_memory(device_lookup_memory_address_to_id_15);
    cuda_free_memory(device_lookup_memory_address_to_id_16);
    cuda_free_memory(device_lookup_memory_address_to_id_17);
    cuda_free_memory(device_lookup_memory_address_to_id_18);
    cuda_free_memory(device_lookup_memory_address_to_id_19);
    cuda_free_memory(device_lookup_memory_address_to_id_20);
    cuda_free_memory(device_lookup_memory_address_to_id_21);
    cuda_free_memory(device_lookup_memory_address_to_id_22);
    cuda_free_memory(device_lookup_memory_address_to_id_23);
    cuda_free_memory(device_lookup_memory_address_to_id_24);
    cuda_free_memory(device_lookup_memory_address_to_id_25);
    cuda_free_memory(device_lookup_memory_address_to_id_26);
    cuda_free_memory(device_lookup_memory_address_to_id_27);
    cuda_free_memory(device_lookup_memory_address_to_id_28);

    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_memory_id_to_big_3);
    cuda_free_memory(device_lookup_memory_id_to_big_4);
    cuda_free_memory(device_lookup_memory_id_to_big_5);
    cuda_free_memory(device_lookup_memory_id_to_big_6);
    cuda_free_memory(device_lookup_memory_id_to_big_7);
    cuda_free_memory(device_lookup_memory_id_to_big_8);
    cuda_free_memory(device_lookup_memory_id_to_big_9);
    cuda_free_memory(device_lookup_memory_id_to_big_10);
    cuda_free_memory(device_lookup_memory_id_to_big_11);
    cuda_free_memory(device_lookup_memory_id_to_big_12);
    cuda_free_memory(device_lookup_memory_id_to_big_13);
    cuda_free_memory(device_lookup_memory_id_to_big_14);
    cuda_free_memory(device_lookup_memory_id_to_big_15);
    cuda_free_memory(device_lookup_memory_id_to_big_16);
    cuda_free_memory(device_lookup_memory_id_to_big_17);
    cuda_free_memory(device_lookup_memory_id_to_big_18);
    cuda_free_memory(device_lookup_memory_id_to_big_19);
    cuda_free_memory(device_lookup_memory_id_to_big_20);
    cuda_free_memory(device_lookup_memory_id_to_big_21);
    cuda_free_memory(device_lookup_memory_id_to_big_22);
    cuda_free_memory(device_lookup_memory_id_to_big_23);

    global_timer.end("generate add_mod_builtin interaction trace");
}
