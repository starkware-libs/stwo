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

#include "gen_assert_eq_opcode_double_deref_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

#define ASSERT_EQ_OPCODE_DOUBLE_REF_N_TRACE_COLUMNS 19

#define ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS 4

__launch_bounds__(256, 2)
__global__ void generate_assert_eq_opcode_double_deref_trace_kernel(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0,

    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_verify_instruction_0,

    m31 **sub_component_inputs_verify_instruction,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,

    m31 **assert_eq_opcode_double_deref_inputs,

    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned row_offset,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0 = {0};
    const m31 M31_1 = {1};
    const m31 M31_8 = {8};
    const m31 M31_16 = {16};
    const m31 M31_32 = {32};
    const m31 M31_256 = {256};
    const m31 M31_32768 = {32768};
    const m31 M31_512 = {512};
    const m31 M31_262144 = {262144};

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
        m31 input_pc_col0 = assert_eq_opcode_double_deref_inputs[0][row];
        traces[0][row] = input_pc_col0;
        m31 input_ap_col1 = assert_eq_opcode_double_deref_inputs[1][row];
        traces[1][row] = input_ap_col1;
        m31 input_fp_col2 = assert_eq_opcode_double_deref_inputs[2][row];
        traces[2][row] = input_fp_col2;
        Enabler enabler_col = Enabler(trace_size);

        // === Decode Instruction ===
        m31 memory_address_to_id_value_tmp_26616_0 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_pc_col0,
            &memory_address_to_id_value_tmp_26616_0
        );
        m31 memory_id_to_big_value_tmp_26616_1[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_26616_0,
            memory_id_to_big_value_tmp_26616_1
        );
        // offset0
        uint16_t offset0_tmp_26616_2 =
            ((uint16_t)(memory_id_to_big_value_tmp_26616_1[0]))
            + ((((uint16_t)(memory_id_to_big_value_tmp_26616_1[1]) & UInt16_127)) << UInt16_9);
        m31 offset0_col3 = m31{offset0_tmp_26616_2};
        traces[3][row] = offset0_col3;
        // offset1
        uint16_t offset1_tmp_26616_3 =
            (((((uint16_t)(memory_id_to_big_value_tmp_26616_1[1])) >> UInt16_7)
            + (((uint16_t)(memory_id_to_big_value_tmp_26616_1[2])) << UInt16_2))
            + ((((uint16_t)(memory_id_to_big_value_tmp_26616_1[3])) & UInt16_31) << UInt16_11));
        m31 offset1_col4 = m31{offset1_tmp_26616_3};
        traces[4][row] = offset1_col4;
        // offset2
        uint16_t offset2_tmp_26616_4 =
            (((((uint16_t)(memory_id_to_big_value_tmp_26616_1[3])) >> UInt16_5)
            + (((uint16_t)(memory_id_to_big_value_tmp_26616_1[4])) << UInt16_4))
            + ((((uint16_t)(memory_id_to_big_value_tmp_26616_1[5])) & UInt16_7) << UInt16_13));
        m31 offset2_col5 = m31{offset2_tmp_26616_4};
        traces[5][row] = offset2_col5;
        // dst_base_fp
        uint16_t dst_base_fp_tmp_26616_5 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_26616_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_26616_1[6])) << UInt16_6))
            >> UInt16_0) & UInt16_1);
        m31 dst_base_fp_col6 = m31{dst_base_fp_tmp_26616_5};
        traces[6][row] = dst_base_fp_col6;
        // op0_base_fp
        uint16_t op0_base_fp_tmp_26616_6 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_26616_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_26616_1[6])) << UInt16_6))
            >> UInt16_1) & UInt16_1);
        m31 op0_base_fp_col7 = m31{op0_base_fp_tmp_26616_6};
        traces[7][row] = op0_base_fp_col7;
        // ap_update_add_1
        uint16_t ap_update_add_1_tmp_26616_7 =
            ((((((uint16_t)(memory_id_to_big_value_tmp_26616_1[5])) >> UInt16_3)
            + (((uint16_t)(memory_id_to_big_value_tmp_26616_1[6])) << UInt16_6))
            >> UInt16_11) & UInt16_1);
        m31 ap_update_add_1_col8 = m31{ap_update_add_1_tmp_26616_7};
        traces[8][row] = ap_update_add_1_col8;
        // === sub_component_inputs.verify_instruction  ===
        sub_component_inputs_verify_instruction[0 * 7 + 0][row] = input_pc_col0;
        sub_component_inputs_verify_instruction[0 * 7 + 1][row] = offset0_col3;
        sub_component_inputs_verify_instruction[0 * 7 + 2][row] = offset1_col4;
        sub_component_inputs_verify_instruction[0 * 7 + 3][row] = offset2_col5;
        sub_component_inputs_verify_instruction[0 * 7 + 4][row] =
            add(
                mul(dst_base_fp_col6, M31_8),
                mul(op0_base_fp_col7, M31_16)
            );
        sub_component_inputs_verify_instruction[0 * 7 + 5][row] =
            add(mul(ap_update_add_1_col8, M31_32), M31_256);
        sub_component_inputs_verify_instruction[0 * 7 + 6][row] = M31_0;
        // === lookup_verify_instruction_0  ===
        lookup_verify_instruction_0[0 * 7 + 0][row] = input_pc_col0;
        lookup_verify_instruction_0[0 * 7 + 1][row] = offset0_col3;
        lookup_verify_instruction_0[0 * 7 + 2][row] = offset1_col4;
        lookup_verify_instruction_0[0 * 7 + 3][row] = offset2_col5;
        lookup_verify_instruction_0[0 * 7 + 4][row] =
            add(
                mul(dst_base_fp_col6, M31_8),
                mul(op0_base_fp_col7, M31_16)
            );
        lookup_verify_instruction_0[0 * 7 + 5][row] =
            add(mul(ap_update_add_1_col8, M31_32), M31_256);
        lookup_verify_instruction_0[0 * 7 + 6][row] = M31_0;
        // === decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8 ===
        m31 decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[3] = {
            sub(offset0_col3, M31_32768),
            sub(offset1_col4, M31_32768),
            sub(offset2_col5, M31_32768)
        };
        // === mem_dst_base_col9 ===
        m31 mem_dst_base_col9 = add(
            mul(dst_base_fp_col6, input_fp_col2),
            mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)
        );
        traces[9][row] = mem_dst_base_col9;

        // === mem0_base_col10 ===
        m31 mem0_base_col10 = add(
            mul(op0_base_fp_col7, input_fp_col2),
            mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)
        );
        traces[10][row] = mem0_base_col10;

        // === Read Positive Num Bits 27 (mem1_base) ===
        m31 memory_address_to_id_value_tmp_26616_9 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add(mem0_base_col10, decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[1]),
            &memory_address_to_id_value_tmp_26616_9
        );
        m31 memory_id_to_big_value_tmp_26616_10[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr,
            memory_address_to_id_value_tmp_26616_9,
            memory_id_to_big_value_tmp_26616_10
        );
        m31 mem1_base_id_col11 = memory_address_to_id_value_tmp_26616_9;
        traces[11][row] = mem1_base_id_col11;
        sub_component_inputs_memory_address_to_id[0][row] =
            add(mem0_base_col10, decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[1]);
        lookup_memory_address_to_id_0[0 * 2 + 0][row] =
            add(mem0_base_col10, decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[1]);
        lookup_memory_address_to_id_0[0 * 2 + 1][row] = mem1_base_id_col11;

        m31 mem1_base_limb_0_col12 = memory_id_to_big_value_tmp_26616_10[0];
        traces[12][row] = mem1_base_limb_0_col12;
        m31 mem1_base_limb_1_col13 = memory_id_to_big_value_tmp_26616_10[1];
        traces[13][row] = mem1_base_limb_1_col13;
        m31 mem1_base_limb_2_col14 = memory_id_to_big_value_tmp_26616_10[2];
        traces[14][row] = mem1_base_limb_2_col14;
        m31 mem1_base_limb_3_col15 = memory_id_to_big_value_tmp_26616_10[3];
        traces[15][row] = mem1_base_limb_3_col15;
        // For 29 bits with 9-bit limbs: limb_3 uses only 2 bits, partial_limb_msb is the upper 7 bits
        m31 partial_limb_msb_col16 = m31{(uint32_t)mem1_base_limb_3_col15 >> 2};
        traces[16][row] = partial_limb_msb_col16;
        sub_component_inputs_memory_id_to_big[0][row] = mem1_base_id_col11;
        lookup_memory_id_to_big_0[0 * 29 + 0][row] = mem1_base_id_col11;
        lookup_memory_id_to_big_0[0 * 29 + 1][row] = mem1_base_limb_0_col12;
        lookup_memory_id_to_big_0[0 * 29 + 2][row] = mem1_base_limb_1_col13;
        lookup_memory_id_to_big_0[0 * 29 + 3][row] = mem1_base_limb_2_col14;
        lookup_memory_id_to_big_0[0 * 29 + 4][row] = mem1_base_limb_3_col15;
        for (int i = 0; i < 24; ++i) {
            lookup_memory_id_to_big_0[0 * 29 + 5 + i][row] = M31_0;
        }

        // === lookup_data.memory_address_to_id_1 (mem_dst_base) ===
        m31 memory_address_to_id_value_tmp_26616_12 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            add(mem_dst_base_col9, decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[0]),
            &memory_address_to_id_value_tmp_26616_12
        );
        m31 dst_id_col17 = memory_address_to_id_value_tmp_26616_12;
        traces[17][row] = dst_id_col17;
        sub_component_inputs_memory_address_to_id[1][row] =
            add(mem_dst_base_col9, decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[0]);
        lookup_memory_address_to_id_1[0 * 2 + 0][row] =
            add(mem_dst_base_col9, decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[0]);
        lookup_memory_address_to_id_1[0 * 2 + 1][row] = dst_id_col17;

        // === lookup_data.memory_address_to_id_2 (mem1_base_value+offset2) ===
        const m31 M31_134217728 = {134217728};
        m31 mem1_base_value_plus_offset2 = add(
            add(
                add(
                    mem1_base_limb_0_col12,
                    mul(mem1_base_limb_1_col13, M31_512)
                ),
                mul(mem1_base_limb_2_col14, M31_262144)
            ),
            add(
                mul(mem1_base_limb_3_col15, M31_134217728),
                decode_instruction_13e4f04f153b7d8c_output_tmp_26616_8_0[2]
            )
        );
        sub_component_inputs_memory_address_to_id[2][row] = mem1_base_value_plus_offset2;
        lookup_memory_address_to_id_2[0 * 2 + 0][row] = mem1_base_value_plus_offset2;
        lookup_memory_address_to_id_2[0 * 2 + 1][row] = dst_id_col17;

        // === lookup_data.opcodes_0/1 ===
        lookup_opcodes_0[0 * 3 + 0][row] = input_pc_col0;
        lookup_opcodes_0[0 * 3 + 1][row] = input_ap_col1;
        lookup_opcodes_0[0 * 3 + 2][row] = input_fp_col2;

        lookup_opcodes_1[0 * 3 + 0][row] = add(input_pc_col0, M31_1);
        lookup_opcodes_1[0 * 3 + 1][row] = add(input_ap_col1, ap_update_add_1_col8);
        lookup_opcodes_1[0 * 3 + 2][row] = input_fp_col2;

        // === Enable ===
        if (row < row_offset)
            traces[18][row] = 1;
        else
            traces[18][row] = 0;
    }

}

void generate_assert_eq_opcode_double_deref_traces(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_verify_instruction_0,


    m31 **sub_componet_input_verify_instruction,
    m31 **sub_componet_input_memory_address_to_id,
    m31 **sub_componet_input_memory_id_to_big,

    m31 **assert_eq_opcode_double_deref_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned row_offset,
    unsigned trace_log_size
) {
    m31 **device_traces = clone_to_device<m31*>(traces, ASSERT_EQ_OPCODE_DOUBLE_REF_N_TRACE_COLUMNS);
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31 *>(lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_id_to_big_0  = clone_to_device<m31 *>(lookup_memory_id_to_big_0 , 29);
    m31 **device_lookup_opcodes_0  = clone_to_device<m31 *>(lookup_opcodes_0 , 3);
    m31 **device_lookup_opcodes_1  = clone_to_device<m31 *>(lookup_opcodes_1 , 3);
    m31 **device_lookup_verify_instruction_0  = clone_to_device<m31 *>(lookup_verify_instruction_0 , 7);

    m31 **device_sub_componet_input_verify_instruction  = clone_to_device<m31 *>(sub_componet_input_verify_instruction , 1 * 7);
    m31 **device_sub_componet_input_memory_address_to_id  = clone_to_device<m31 *>(sub_componet_input_memory_address_to_id , 3 * 1);
    m31 **device_sub_componet_input_memory_id_to_big  = clone_to_device<m31 *>(sub_componet_input_memory_id_to_big , 1 * 1);

    m31 **device_assert_eq_opcode_double_deref_input = clone_to_device<m31 *>(assert_eq_opcode_double_deref_input, 3);
    unsigned **device_memory_id_to_big_transpose_big_value_ptr = clone_to_device<m31 *>(memory_id_to_big_transpose_big_value_ptr, 8);


    timer global_timer;
    global_timer.start("generate assert_eq_opcode_double_deref base trace");

    unsigned trace_size = 1 << trace_log_size;
    int block_dim = trace_size < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_assert_eq_opcode_double_deref_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,

        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_id_to_big_0,

        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        device_lookup_verify_instruction_0,

        device_sub_componet_input_verify_instruction,
        device_sub_componet_input_memory_address_to_id,
        device_sub_componet_input_memory_id_to_big,

        device_assert_eq_opcode_double_deref_input,

        memory_address_to_id_address_to_raw_id,

        device_memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,

        row_offset,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate assert_eq_opcode_double_deref base trace");

    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_verify_instruction_0);

    cuda_free_memory(device_sub_componet_input_verify_instruction);
    cuda_free_memory(device_sub_componet_input_memory_address_to_id);
    cuda_free_memory(device_sub_componet_input_memory_id_to_big);

    cuda_free_memory(device_assert_eq_opcode_double_deref_input);
    cuda_free_memory(device_memory_id_to_big_transpose_big_value_ptr);
}

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel_round0(
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
    m31 init_combine_reg[N] = {0};
    m31 final_combine_reg[M] = {0};

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

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel(
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
    m31 init_combine_reg[N] = {0};
    m31 final_combine_reg[M] = {0};

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

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel_second2last(
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
    unsigned row_offset
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    qm31 enabler_col = {0};
    if (vec_index < row_offset) {
        enabler_col = {1};
    }

    m31 init_combine_reg[N] = {0};
    m31 final_combine_reg[M] = {0};

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
__global__ void generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel_last(
    LookupElementsBasic<N>  *lookup_elements_n,
    m31 **lookup_state_0,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned row_offset
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    qm31 enabler_col = {0};
    if (vec_index < row_offset) {
        enabler_col = {1};
    }

    m31 init_combine_reg[N] = {0};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }

    if (vec_index < trace_size) {
        qm31 denom = lookup_elements_n->combine(init_combine_reg, N);
       logup_col_write_frac(vec_index, mul(qm31{P-1, 0, 0, 0}, enabler_col), denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}


__global__ void generate_assert_eq_opcode_double_deref_interaction_trace_finalize_col_kernel(
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

__global__ void generate_assert_eq_opcode_double_deref_interaction_trace_cumsum_shift(
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

__global__ void generate_assert_eq_opcode_double_deref_interaction_trace_coord_prefix_sum(
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

void generate_assert_eq_opcode_double_deref_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0 ,

    m31 **lookup_opcodes_0 ,
    m31 **lookup_opcodes_1 ,
    m31 **lookup_verify_instruction_0 ,

    unsigned row_offset,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    MemoryAddressToId *memory_address_to_id_lookup_elements = (MemoryAddressToId *)memory_address_to_id;
    MemoryIdToBig  *memory_id_to_big_lookup_elements  = (MemoryIdToBig *)memory_id_to_big;
    Opcodes *opcodes_lookup_elements = (Opcodes *)opcodes;
    VerifyInstruction *verify_instruction_lookup_elements = (VerifyInstruction *)verify_instruction;

    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig  *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes_lookup_elements = cuda_malloc<Opcodes>(1);
    VerifyInstruction *device_verify_instruction_lookup_elements = cuda_malloc<VerifyInstruction>(1);

    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id_lookup_elements, 1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big_lookup_elements, 1);
    cuda_mem_copy_host_to_device<Opcodes>(opcodes_lookup_elements, device_opcodes_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyInstruction>(verify_instruction_lookup_elements, device_verify_instruction_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31 *>(lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>(lookup_opcodes_1, 3);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>(lookup_verify_instruction_0, 7);


    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4 * ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    // dump_lookup_data(lookup_memory_id_to_big_0, 29, trace_size);
    // dump_lookup_data(lookup_memory_address_to_id_1, 2, trace_size);

    timer global_timer;
    global_timer.start("generate assert_eq_opcode_double_deref interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For verify_instruction_0 & memory_address_to_id_0
    generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
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
    generate_assert_eq_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #1 Interaction trace For memory_id_to_big_0 & memory_address_to_id_1
    block_dim = trace_size < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel<29, 2><<<num_blocks, block_dim>>>(
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

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_assert_eq_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #2 Interaction trace For memory_address_to_id_2 & opcodes_0
    block_dim = trace_size < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel_second2last<2, 3><<<num_blocks, block_dim>>>(
        device_memory_address_to_id_lookup_elements,
        device_opcodes_lookup_elements,

        device_lookup_memory_address_to_id_2,
        device_lookup_opcodes_0,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        row_offset
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_assert_eq_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #3 Interaction trace For opcodes_1
    block_dim = trace_size < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < ASSERT_EQ_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_assert_eq_opcode_double_deref_interaction_trace_col_gen_kernel_last<3><<<num_blocks, block_dim>>>(
        device_opcodes_lookup_elements,

        device_lookup_opcodes_1,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        row_offset
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_assert_eq_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_assert_eq_opcode_double_deref_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_assert_eq_opcode_double_deref_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[4 * ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * ASSERT_EQ_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate assert_eq_opcode_double_deref interaction trace");

    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_opcodes_lookup_elements);
    cuda_free_memory(device_verify_instruction_lookup_elements);

    cuda_free_memory(device_logup_denom);

    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_verify_instruction_0);

    cuda_free_memory(device_interaction_traces);

    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}