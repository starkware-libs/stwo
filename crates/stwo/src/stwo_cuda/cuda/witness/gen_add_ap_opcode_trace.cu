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

#include "gen_add_ap_opcode_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

#define ADD_AP_OPCODE_N_TRACE_COLUMNS 17

#define ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS 4

__launch_bounds__(256, 2)
__global__ void generate_add_ap_opcode_trace_kernel(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_range_check_11_0,
    m31 **lookup_range_check_18_0,
    m31 **lookup_verify_instruction_0,


    m31 **sub_component_inputss_verify_instruction,
    m31 **sub_component_inputss_memory_address_to_id,
    m31 **sub_component_inputss_memory_id_to_big,

    m31 **opcodes_inputs,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0         = {0};
    const m31 M31_1         = {1};
    const m31 M31_128       = {128};
    const m31 M31_134217728 = {134217728};
    const m31 M31_136       = {136};
    const m31 M31_16        = {16};
    const m31 M31_2147483646= {2147483646};
    const m31 M31_24        = {24};
    const m31 M31_256       = {256};
    const m31 M31_262144    = {262144};
    const m31 M31_32        = {32};
    const m31 M31_32767     = {32767};
    const m31 M31_32768     = {32768};
    const m31 M31_511       = {511};
    const m31 M31_512       = {512};
    const m31 M31_64        = {64};

    const uint16_t UInt16_1 = 1;
    const uint16_t UInt16_13= 13;
    const uint16_t UInt16_2 = 2;
    const uint16_t UInt16_3 = 3;
    const uint16_t UInt16_4 = 4;
    const uint16_t UInt16_5 = 5;
    const uint16_t UInt16_6 = 6;
    const uint16_t UInt16_7 = 7;

    const uint16_t UInt16_8 = 8;

    if (row < trace_size) {
        m31 input_pc_col0 = opcodes_inputs[0][row];
        traces[0][row] = input_pc_col0;
        m31 input_ap_col1 = opcodes_inputs[1][row];
        traces[1][row] = input_ap_col1;
        m31 input_fp_col2 = opcodes_inputs[2][row];
        traces[2][row] = input_fp_col2;
        Enabler enabler_col = Enabler(trace_size);

        m31 memory_address_to_id_value_tmp_c921e_0 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_pc_col0,
            &memory_address_to_id_value_tmp_c921e_0
        );
        m31 memory_id_to_big_value_tmp_c921e_1[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            memory_address_to_id_value_tmp_c921e_0,
            memory_id_to_big_value_tmp_c921e_1
        );
        uint16_t offset2_tmp_c921e_2 =
            (((((uint16_t)(memory_id_to_big_value_tmp_c921e_1[3])) >> (UInt16_5))
                + (((uint16_t)(memory_id_to_big_value_tmp_c921e_1[4])) << (UInt16_4)))
                + ((((uint16_t)(memory_id_to_big_value_tmp_c921e_1[5])) & (UInt16_7)) << (UInt16_13)));

        m31 offset2_col3 = m31 {offset2_tmp_c921e_2};
        traces[3][row] = offset2_col3;

        // op1_imm_tmp_c921e_3
        uint16_t op1_imm_tmp_c921e_3 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_c921e_1[5])) >> UInt16_3)
                + (((uint16_t)(memory_id_to_big_value_tmp_c921e_1[6])) << UInt16_6))
                >> UInt16_2)
                & UInt16_1);

        m31 op1_imm_col4 = m31{op1_imm_tmp_c921e_3};
        traces[4][row] = op1_imm_col4;

        // op1_base_fp_tmp_c921e_4
        uint16_t op1_base_fp_tmp_c921e_4 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_c921e_1[5])) >> UInt16_3)
                + (((uint16_t)(memory_id_to_big_value_tmp_c921e_1[6])) << UInt16_6))
                >> UInt16_3)
                & UInt16_1);

        m31 op1_base_fp_col5 = m31{op1_base_fp_tmp_c921e_4};
        traces[5][row] = op1_base_fp_col5;

        sub_component_inputss_verify_instruction[0 * 7 + 0][row] = input_pc_col0;
        sub_component_inputss_verify_instruction[0 * 7 + 1][row] = M31_32767;
        sub_component_inputss_verify_instruction[0 * 7 + 2][row] = M31_32767;
        sub_component_inputss_verify_instruction[0 * 7 + 3][row] = offset2_col3;
        sub_component_inputss_verify_instruction[0 * 7 + 4][row] = add(add(add((M31_24), mul((op1_imm_col4), (M31_32)))
                            , ((op1_base_fp_col5) * (M31_64)))
                            , mul(sub(sub((M31_1), (op1_imm_col4)), (op1_base_fp_col5)), (M31_128)));
        sub_component_inputss_verify_instruction[0 * 7 + 5][row] = M31_16;
        sub_component_inputss_verify_instruction[0 * 7 + 6][row] = M31_0;

        lookup_verify_instruction_0[0 * 7 + 0][row] = input_pc_col0;
        lookup_verify_instruction_0[0 * 7 + 1][row] = M31_32767;
        lookup_verify_instruction_0[0 * 7 + 2][row] = M31_32767;
        lookup_verify_instruction_0[0 * 7 + 3][row] = offset2_col3;
        lookup_verify_instruction_0[0 * 7 + 4][row] = add(add(add((M31_24), mul((op1_imm_col4), (M31_32)))
                            , ((op1_base_fp_col5) * (M31_64)))
                            , mul(sub(sub((M31_1), (op1_imm_col4)), (op1_base_fp_col5)), (M31_128)));
        lookup_verify_instruction_0[0 * 7 + 5][row] = M31_16;
        lookup_verify_instruction_0[0 * 7 + 6][row] = M31_0;

        m31 decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[19] = {0};

        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[0]  = M31_2147483646;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[1]  = M31_2147483646;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[2]  = sub(offset2_col3, M31_32768);
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[3]  = M31_1;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[4]  = M31_1;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[5]  = op1_imm_col4;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[6]  = op1_base_fp_col5;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[7]  = sub(sub(M31_1, op1_imm_col4), op1_base_fp_col5);
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[8]  = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[9]  = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[10] = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[11] = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[12] = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[13] = M31_1;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[14] = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[15] = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[16] = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[17] = M31_0;
        decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[18] = M31_0;

        m31 mem1_base_col6 = add(
            add(
                mul(op1_imm_col4, input_pc_col0),
                mul(op1_base_fp_col5, input_fp_col2)
            ),
            mul(decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[7], input_ap_col1)
        );
        traces[6][row] = mem1_base_col6;

        // Read Small.
        m31 memory_address_to_id_value_tmp_c921e_6 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add((mem1_base_col6), (decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[2])),
            &memory_address_to_id_value_tmp_c921e_6);

        m31 memory_id_to_big_value_tmp_c921e_7[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            memory_address_to_id_value_tmp_c921e_6,
            memory_id_to_big_value_tmp_c921e_7
        );

        m31 op1_id_col7 = memory_address_to_id_value_tmp_c921e_6;
        traces[7][row] = op1_id_col7;

        sub_component_inputss_memory_address_to_id[0][row] = add((mem1_base_col6), (decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[2]));
        lookup_memory_address_to_id_0[0 * 2 + 0][row] = add((mem1_base_col6), (decode_instruction_6a31a33b12bf4afe_output_tmp_c921e_5[2]));
        lookup_memory_address_to_id_0[0 * 2 + 1][row] = op1_id_col7;

        // Cond Decode Small Sign.
        bool msb_tmp_c921e_8 = (memory_id_to_big_value_tmp_c921e_7[27] == (M31_256));
        m31 msb_col8 = {msb_tmp_c921e_8};
        traces[8][row] = msb_col8;

        bool mid_limbs_set_tmp_c921e_9 = (memory_id_to_big_value_tmp_c921e_7[20] == (M31_511));
        m31 mid_limbs_set_col9 = {mid_limbs_set_tmp_c921e_9};
        traces[9][row] = mid_limbs_set_col9;

        m31 cond_decode_small_sign_output_tmp_c921e_10[2] = {0};
        cond_decode_small_sign_output_tmp_c921e_10[0] = msb_col8;
        cond_decode_small_sign_output_tmp_c921e_10[1] = mid_limbs_set_col9;

        m31 op1_limb_0_col10 = memory_id_to_big_value_tmp_c921e_7[0];
        traces[10][row] = op1_limb_0_col10;
        m31 op1_limb_1_col11 = memory_id_to_big_value_tmp_c921e_7[1];
        traces[11][row] = op1_limb_1_col11;
        m31 op1_limb_2_col12 = memory_id_to_big_value_tmp_c921e_7[2];
        traces[12][row] = op1_limb_2_col12;

        sub_component_inputss_memory_id_to_big[0 * 1 + 0][row] = op1_id_col7;
        lookup_memory_id_to_big_0[0 * 29 + 0][row]  = op1_id_col7;
        lookup_memory_id_to_big_0[0 * 29 + 1][row]  = op1_limb_0_col10;
        lookup_memory_id_to_big_0[0 * 29 + 2][row]  = op1_limb_1_col11;
        lookup_memory_id_to_big_0[0 * 29 + 3][row]  = op1_limb_2_col12;
        lookup_memory_id_to_big_0[0 * 29 + 4][row]  = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 5][row]  = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 6][row]  = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 7][row]  = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 8][row]  = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 9][row]  = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 10][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 11][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 12][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 13][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 14][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 15][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 16][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 17][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 18][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 19][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 20][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 21][row] = mul(mid_limbs_set_col9, M31_511);
        lookup_memory_id_to_big_0[0 * 29 + 22][row] = sub(mul(M31_136, msb_col8), mid_limbs_set_col9);
        lookup_memory_id_to_big_0[0 * 29 + 23][row] = M31_0;
        lookup_memory_id_to_big_0[0 * 29 + 24][row] = M31_0;
        lookup_memory_id_to_big_0[0 * 29 + 25][row] = M31_0;
        lookup_memory_id_to_big_0[0 * 29 + 26][row] = M31_0;
        lookup_memory_id_to_big_0[0 * 29 + 27][row] = M31_0;
        lookup_memory_id_to_big_0[0 * 29 + 28][row] = mul(msb_col8, M31_256);

        m31 read_small_output_tmp_c921e_11[2] = {
        sub(sub(add(add((op1_limb_0_col10), mul((op1_limb_1_col11), (M31_512)))
                , mul((op1_limb_2_col12), (M31_262144)))
                , (msb_col8))
                , mul((M31_134217728), (mid_limbs_set_col9))),
            op1_id_col7,
        };
        // Column 13: remainder_bits_col13 = memory_id_to_big_value[3] & 3
        uint16_t remainder_bits_tmp_c921e_12 = ((uint16_t)(memory_id_to_big_value_tmp_c921e_7[3])) & 3;
        m31 remainder_bits_col13 = {remainder_bits_tmp_c921e_12};
        traces[13][row] = remainder_bits_col13;

        // Column 14: partial_limb_msb_col14 = (remainder_bits_col13 & 2) >> 1
        uint16_t partial_limb_msb_tmp_c921e_13 = (remainder_bits_tmp_c921e_12 & 2) >> 1;
        m31 partial_limb_msb_col14 = {partial_limb_msb_tmp_c921e_13};
        traces[14][row] = partial_limb_msb_col14;

        // Compute next_ap for range checks
        m31 next_ap_tmp_c921e_16 = add(input_ap_col1, read_small_output_tmp_c921e_11[0]);

        // Column 15: range_check_ap_bot11bits_col15 = next_ap & 2047
        uint16_t range_check_ap_bot11bits_tmp_c921e_17 = ((uint16_t)(next_ap_tmp_c921e_16)) & 2047;
        m31 range_check_ap_bot11bits_col15 = {range_check_ap_bot11bits_tmp_c921e_17};
        traces[15][row] = range_check_ap_bot11bits_col15;

        // Range check lookups
        lookup_range_check_18_0[0 * 1 + 0][row] = mul(sub(next_ap_tmp_c921e_16, range_check_ap_bot11bits_col15), {1048576});
        lookup_range_check_11_0[0 * 1 + 0][row] = range_check_ap_bot11bits_col15;

        lookup_opcodes_0[0 * 3 + 0][row] = input_pc_col0;
        lookup_opcodes_0[0 * 3 + 1][row] = input_ap_col1;
        lookup_opcodes_0[0 * 3 + 2][row] = input_fp_col2;

        lookup_opcodes_1[0 * 3 + 0][row] = add((input_pc_col0), add((M31_1), (op1_imm_col4)));
        lookup_opcodes_1[0 * 3 + 1][row] = next_ap_tmp_c921e_16;
        lookup_opcodes_1[0 * 3 + 2][row] = input_fp_col2;

        // Column 16: enabler_col
        if (row < n_rows)
            traces[16][row] = 1;
        else
            traces[16][row] = 0;
    }
}


void generate_add_ap_opcode_traces(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_range_check_11_0,
    m31 **lookup_range_check_18_0,
    m31 **lookup_verify_instruction_0,


    m31 **sub_component_inputs_verify_instruction,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,


    m31 **opcodes_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    m31 **device_traces = clone_to_device<m31*>(traces, ADD_AP_OPCODE_N_TRACE_COLUMNS);
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_id_to_big_0  = clone_to_device<m31 *>(lookup_memory_id_to_big_0 , 29);
    m31 **device_lookup_opcodes_0  = clone_to_device<m31 *>(lookup_opcodes_0 , 3);
    m31 **device_lookup_opcodes_1  = clone_to_device<m31 *>(lookup_opcodes_1 , 3);
    m31 **device_lookup_range_check_11_0  = clone_to_device<m31 *>(lookup_range_check_11_0 , 1);
    m31 **device_lookup_range_check_18_0  = clone_to_device<m31 *>(lookup_range_check_18_0 , 1);
    m31 **device_lookup_verify_instruction_0  = clone_to_device<m31 *>(lookup_verify_instruction_0 , 7);

    m31 **device_sub_component_inputs_verify_instruction  = clone_to_device<m31 *>(sub_component_inputs_verify_instruction , 1 * 7);
    m31 **device_sub_component_inputs_memory_address_to_id  = clone_to_device<m31 *>(sub_component_inputs_memory_address_to_id , 1 * 1);
    m31 **device_sub_component_inputs_memory_id_to_big  = clone_to_device<m31 *>(sub_component_inputs_memory_id_to_big , 1 * 1);

    m31 **device_opcodes_input = clone_to_device<m31 *>(opcodes_input, 3);
    unsigned **device_memory_id_to_big_transposed_big_values = clone_to_device<m31 *>(memory_id_to_big_transposed_big_values, 8);


    timer global_timer;
    global_timer.start("generate add_ap_opcode base trace");

    unsigned trace_size = 1 << log_size;
    int block_dim = trace_size < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_add_ap_opcode_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,

        device_lookup_memory_address_to_id_0,
        device_lookup_memory_id_to_big_0,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        device_lookup_range_check_11_0,
        device_lookup_range_check_18_0,
        device_lookup_verify_instruction_0,

        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,

        device_opcodes_input,

        memory_address_to_id_address_to_raw_id,

        device_memory_id_to_big_transposed_big_values,
        memory_id_to_big_small_values,

        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate add_ap_opcode base trace");

    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_11_0);
    cuda_free_memory(device_lookup_range_check_18_0);
    cuda_free_memory(device_lookup_verify_instruction_0);

    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);

    cuda_free_memory(device_opcodes_input);
    cuda_free_memory(device_memory_id_to_big_transposed_big_values);
}

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_add_ap_opcode_interaction_trace_col_gen_kernel_round0(
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
__global__ void generate_add_ap_opcode_interaction_trace_col_gen_kernel_second2last(
    LookupElementsBasic<N>  *lookup_elements_n,
    LookupElementsBasic<M>  *lookup_elements_m,
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

template <int N>
__launch_bounds__(256, 2)
__global__ void generate_add_ap_opcode_interaction_trace_col_gen_kernel_last(
    LookupElementsBasic<N>  *lookup_elements_n,
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


__global__ void generate_add_ap_opcode_interaction_trace_finalize_col_kernel(
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

__global__ void generate_add_ap_opcode_interaction_trace_cumsum_shift(
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

__global__ void generate_add_ap_opcode_interaction_trace_coord_prefix_sum(
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

void generate_add_ap_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *range_check_11,
    void *range_check_18,
    void *verify_instruction,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_id_to_big_0 ,
    m31 **lookup_opcodes_0 ,
    m31 **lookup_opcodes_1 ,
    m31 **lookup_range_check_11_0 ,
    m31 **lookup_range_check_18_0 ,
    m31 **lookup_verify_instruction_0 ,

    unsigned n_rows,
    unsigned log_size,
    m31 **interaction_trace,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    MemoryAddressToId *memory_address_to_id_lookup_elements = (MemoryAddressToId *)memory_address_to_id;
    MemoryIdToBig  *memory_id_to_big_lookup_elements  = (MemoryIdToBig *)memory_id_to_big;
    Opcodes *opcodes_lookup_elements = (Opcodes *)opcodes;
    RangeCheck_11 *range_check_11_lookup_elements = (RangeCheck_11 *)range_check_11;
    RangeCheck_18 *range_check_18_lookup_elements = (RangeCheck_18 *)range_check_18;
    VerifyInstruction *verify_instruction_lookup_elements = (VerifyInstruction *)verify_instruction;

    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig  *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes_lookup_elements = cuda_malloc<Opcodes>(1);
    RangeCheck_11 *device_range_check_11_lookup_elements = cuda_malloc<RangeCheck_11>(1);
    RangeCheck_18 *device_range_check_18_lookup_elements = cuda_malloc<RangeCheck_18>(1);
    VerifyInstruction *device_verify_instruction_lookup_elements = cuda_malloc<VerifyInstruction>(1);

    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id_lookup_elements, 1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big_lookup_elements, 1);
    cuda_mem_copy_host_to_device<Opcodes>(opcodes_lookup_elements, device_opcodes_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_11>(range_check_11_lookup_elements, device_range_check_11_lookup_elements, 1);
    cuda_mem_copy_host_to_device<RangeCheck_18>(range_check_18_lookup_elements, device_range_check_18_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyInstruction>(verify_instruction_lookup_elements, device_verify_instruction_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>(lookup_opcodes_1, 3);
    m31 **device_lookup_range_check_11_0 = clone_to_device<m31 *>(lookup_range_check_11_0, 1);
    m31 **device_lookup_range_check_18_0 = clone_to_device<m31 *>(lookup_range_check_18_0, 1);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>(lookup_verify_instruction_0, 7);


    m31 **device_interaction_trace = clone_to_device<m31*>(interaction_trace, 4 * ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    // dump_lookup_data(lookup_verify_bitwise_xor_8_0, 3, trace_size);
    // dump_lookup_data(lookup_verify_bitwise_xor_8_1, 3, trace_size);

    timer global_timer;
    global_timer.start("generate add_ap_opcode interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For verify_instruction_0 & memory_address_to_id_0
    generate_add_ap_opcode_interaction_trace_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
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

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_add_ap_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        0,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_trace
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_trace, 0, trace_size);

    // #1 Interaction trace For memory_id_to_big_0 & range_check_18_0
    block_dim = trace_size < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_add_ap_opcode_interaction_trace_col_gen_kernel_round0<29, 1><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_range_check_18_lookup_elements,

        device_lookup_memory_id_to_big_0,
        device_lookup_range_check_18_0,

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
    generate_add_ap_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        1,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_trace
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_trace, 1, trace_size);

    // #2 Interaction trace For range_check_11_0 & opcodes_0
    block_dim = trace_size < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_add_ap_opcode_interaction_trace_col_gen_kernel_second2last<1, 3><<<num_blocks, block_dim>>>(
        device_range_check_11_lookup_elements,
        device_opcodes_lookup_elements,

        device_lookup_range_check_11_0,
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
    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_add_ap_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        2,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_trace
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_trace, 2, trace_size);

    // #3 Interaction trace For opcodes_1
    block_dim = trace_size < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_add_ap_opcode_interaction_trace_col_gen_kernel_last<3><<<num_blocks, block_dim>>>(
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
    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_add_ap_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        3,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_trace
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_trace, 3, trace_size);

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_add_ap_opcode_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_trace,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_add_ap_opcode_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_trace[4 * ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * ADD_AP_OPCODE_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate add_ap_opcode interaction trace");

    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_opcodes_lookup_elements);
    cuda_free_memory(device_range_check_11_lookup_elements);
    cuda_free_memory(device_range_check_18_lookup_elements);
    cuda_free_memory(device_verify_instruction_lookup_elements);

    cuda_free_memory(device_logup_denom);

    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_range_check_11_0);
    cuda_free_memory(device_lookup_range_check_18_0);
    cuda_free_memory(device_lookup_verify_instruction_0);

    cuda_free_memory(device_interaction_trace);

    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}