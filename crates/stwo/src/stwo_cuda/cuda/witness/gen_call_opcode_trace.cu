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

#include "gen_call_opcode_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

#define CALL_OPCODE_N_TRACE_COLUMNS 25
#define CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS 5
#define CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

__launch_bounds__(256, 2)
__global__ void generate_call_opcode_trace_kernel(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_verify_instruction_0,

    m31 **sub_component_inputss_verify_instruction,
    m31 **sub_component_inputss_memory_address_to_id,
    m31 **sub_component_inputss_memory_id_to_big,

    m31 **call_opcode_inputs,

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
    const m31 M31_64        = {64};
    const m31 M31_66        = {66};
    const m31 M31_128       = {128};
    const m31 M31_512       = {512};
    const m31 M31_262144    = {262144};
    const m31 M31_134217728 = {134217728};
    const m31 M31_32768     = {32768};
    const m31 M31_32769     = {32769};

    const uint16_t UInt16_1 = 1;
    const uint16_t UInt16_2 = 2;
    const uint16_t UInt16_3 = 3;
    const uint16_t UInt16_4 = 4;
    const uint16_t UInt16_5 = 5;
    const uint16_t UInt16_6 = 6;
    const uint16_t UInt16_7 = 7;
    const uint16_t UInt16_13 = 13;

    if (row < trace_size) {
        // Input columns
        m31 input_pc_col0 = call_opcode_inputs[0][row];
        traces[0][row] = input_pc_col0;
        m31 input_ap_col1 = call_opcode_inputs[1][row];
        traces[1][row] = input_ap_col1;
        m31 input_fp_col2 = call_opcode_inputs[2][row];
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

        // offset2 calculation
        uint16_t offset2_tmp =
            (((((uint16_t)(memory_id_to_big_value_tmp_1[3])) >> (UInt16_5))
                + (((uint16_t)(memory_id_to_big_value_tmp_1[4])) << (UInt16_4)))
                + ((((uint16_t)(memory_id_to_big_value_tmp_1[5])) & (UInt16_7)) << (UInt16_13)));
        m31 offset2_col3 = m31{offset2_tmp};
        traces[3][row] = offset2_col3;

        // op1_base_fp calculation
        uint16_t op1_base_fp_tmp =
            ((((((uint16_t)(memory_id_to_big_value_tmp_1[5])) >> UInt16_3)
                + (((uint16_t)(memory_id_to_big_value_tmp_1[6])) << UInt16_6))
                >> UInt16_3)
                & UInt16_1);
        m31 op1_base_fp_col4 = m31{op1_base_fp_tmp};
        traces[4][row] = op1_base_fp_col4;

        // verify_instruction sub_component_inputs and lookup
        sub_component_inputss_verify_instruction[0 * 7 + 0][row] = input_pc_col0;
        sub_component_inputss_verify_instruction[0 * 7 + 1][row] = M31_32768;
        sub_component_inputss_verify_instruction[0 * 7 + 2][row] = M31_32769;
        sub_component_inputss_verify_instruction[0 * 7 + 3][row] = offset2_col3;
        sub_component_inputss_verify_instruction[0 * 7 + 4][row] = add(
            mul(op1_base_fp_col4, M31_64),
            mul(sub(M31_1, op1_base_fp_col4), M31_128)
        );
        sub_component_inputss_verify_instruction[0 * 7 + 5][row] = M31_66;
        sub_component_inputss_verify_instruction[0 * 7 + 6][row] = M31_0;

        lookup_verify_instruction_0[0][row] = input_pc_col0;
        lookup_verify_instruction_0[1][row] = M31_32768;
        lookup_verify_instruction_0[2][row] = M31_32769;
        lookup_verify_instruction_0[3][row] = offset2_col3;
        lookup_verify_instruction_0[4][row] = add(
            mul(op1_base_fp_col4, M31_64),
            mul(sub(M31_1, op1_base_fp_col4), M31_128)
        );
        lookup_verify_instruction_0[5][row] = M31_66;
        lookup_verify_instruction_0[6][row] = M31_0;

        // decode_instruction output: offset2 - 32768
        m31 offset2_adjusted = sub(offset2_col3, M31_32768);

        // Read stored_fp at ap (memory_address_to_id_0)
        m31 memory_address_to_id_value_tmp_5 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            input_ap_col1,
            &memory_address_to_id_value_tmp_5
        );
        m31 stored_fp_id_col5 = memory_address_to_id_value_tmp_5;
        traces[5][row] = stored_fp_id_col5;

        sub_component_inputss_memory_address_to_id[0][row] = input_ap_col1;
        lookup_memory_address_to_id_0[0][row] = input_ap_col1;
        lookup_memory_address_to_id_0[1][row] = stored_fp_id_col5;

        // Read stored_fp limbs (memory_id_to_big_0)
        m31 stored_fp_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            stored_fp_id_col5,
            stored_fp_value
        );

        m31 stored_fp_limb_0_col6 = stored_fp_value[0];
        traces[6][row] = stored_fp_limb_0_col6;
        m31 stored_fp_limb_1_col7 = stored_fp_value[1];
        traces[7][row] = stored_fp_limb_1_col7;
        m31 stored_fp_limb_2_col8 = stored_fp_value[2];
        traces[8][row] = stored_fp_limb_2_col8;
        m31 stored_fp_limb_3_col9 = stored_fp_value[3];
        traces[9][row] = stored_fp_limb_3_col9;

        // Range Check Last Limb Bits In Ms Limb 2
        uint16_t partial_limb_msb_tmp_0 = (((uint16_t)(stored_fp_limb_3_col9)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col10 = m31{partial_limb_msb_tmp_0};
        traces[10][row] = partial_limb_msb_col10;

        sub_component_inputss_memory_id_to_big[0][row] = stored_fp_id_col5;
        lookup_memory_id_to_big_0[0][row] = stored_fp_id_col5;
        lookup_memory_id_to_big_0[1][row] = stored_fp_limb_0_col6;
        lookup_memory_id_to_big_0[2][row] = stored_fp_limb_1_col7;
        lookup_memory_id_to_big_0[3][row] = stored_fp_limb_2_col8;
        lookup_memory_id_to_big_0[4][row] = stored_fp_limb_3_col9;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_0[i][row] = M31_0;
        }

        // Read stored_ret_pc at ap+1 (memory_address_to_id_1)
        m31 ap_plus_1 = add(input_ap_col1, M31_1);
        m31 memory_address_to_id_value_tmp_12 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            ap_plus_1,
            &memory_address_to_id_value_tmp_12
        );
        m31 stored_ret_pc_id_col11 = memory_address_to_id_value_tmp_12;
        traces[11][row] = stored_ret_pc_id_col11;

        sub_component_inputss_memory_address_to_id[1][row] = ap_plus_1;
        lookup_memory_address_to_id_1[0][row] = ap_plus_1;
        lookup_memory_address_to_id_1[1][row] = stored_ret_pc_id_col11;

        // Read stored_ret_pc limbs (memory_id_to_big_1)
        m31 stored_ret_pc_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            stored_ret_pc_id_col11,
            stored_ret_pc_value
        );

        m31 stored_ret_pc_limb_0_col12 = stored_ret_pc_value[0];
        traces[12][row] = stored_ret_pc_limb_0_col12;
        m31 stored_ret_pc_limb_1_col13 = stored_ret_pc_value[1];
        traces[13][row] = stored_ret_pc_limb_1_col13;
        m31 stored_ret_pc_limb_2_col14 = stored_ret_pc_value[2];
        traces[14][row] = stored_ret_pc_limb_2_col14;
        m31 stored_ret_pc_limb_3_col15 = stored_ret_pc_value[3];
        traces[15][row] = stored_ret_pc_limb_3_col15;

        // Range Check Last Limb Bits In Ms Limb 2
        uint16_t partial_limb_msb_tmp_1 = (((uint16_t)(stored_ret_pc_limb_3_col15)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col16 = m31{partial_limb_msb_tmp_1};
        traces[16][row] = partial_limb_msb_col16;

        sub_component_inputss_memory_id_to_big[1][row] = stored_ret_pc_id_col11;
        lookup_memory_id_to_big_1[0][row] = stored_ret_pc_id_col11;
        lookup_memory_id_to_big_1[1][row] = stored_ret_pc_limb_0_col12;
        lookup_memory_id_to_big_1[2][row] = stored_ret_pc_limb_1_col13;
        lookup_memory_id_to_big_1[3][row] = stored_ret_pc_limb_2_col14;
        lookup_memory_id_to_big_1[4][row] = stored_ret_pc_limb_3_col15;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_1[i][row] = M31_0;
        }

        // mem1_base = op1_base_fp * fp + (1 - op1_base_fp) * ap
        m31 mem1_base_col17 = add(
            mul(op1_base_fp_col4, input_fp_col2),
            mul(sub(M31_1, op1_base_fp_col4), input_ap_col1)
        );
        traces[17][row] = mem1_base_col17;

        // Read next_pc at mem1_base + offset2_adjusted (memory_address_to_id_2)
        m31 next_pc_addr = add(mem1_base_col17, offset2_adjusted);
        m31 memory_address_to_id_value_tmp_19 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            next_pc_addr,
            &memory_address_to_id_value_tmp_19
        );
        m31 next_pc_id_col18 = memory_address_to_id_value_tmp_19;
        traces[18][row] = next_pc_id_col18;

        sub_component_inputss_memory_address_to_id[2][row] = next_pc_addr;
        lookup_memory_address_to_id_2[0][row] = next_pc_addr;
        lookup_memory_address_to_id_2[1][row] = next_pc_id_col18;

        // Read next_pc limbs (memory_id_to_big_2)
        m31 next_pc_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            next_pc_id_col18,
            next_pc_value
        );

        m31 next_pc_limb_0_col19 = next_pc_value[0];
        traces[19][row] = next_pc_limb_0_col19;
        m31 next_pc_limb_1_col20 = next_pc_value[1];
        traces[20][row] = next_pc_limb_1_col20;
        m31 next_pc_limb_2_col21 = next_pc_value[2];
        traces[21][row] = next_pc_limb_2_col21;
        m31 next_pc_limb_3_col22 = next_pc_value[3];
        traces[22][row] = next_pc_limb_3_col22;

        // Range Check Last Limb Bits In Ms Limb 2
        uint16_t partial_limb_msb_tmp_2 = (((uint16_t)(next_pc_limb_3_col22)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col23 = m31{partial_limb_msb_tmp_2};
        traces[23][row] = partial_limb_msb_col23;

        sub_component_inputss_memory_id_to_big[2][row] = next_pc_id_col18;
        lookup_memory_id_to_big_2[0][row] = next_pc_id_col18;
        lookup_memory_id_to_big_2[1][row] = next_pc_limb_0_col19;
        lookup_memory_id_to_big_2[2][row] = next_pc_limb_1_col20;
        lookup_memory_id_to_big_2[3][row] = next_pc_limb_2_col21;
        lookup_memory_id_to_big_2[4][row] = next_pc_limb_3_col22;
        for (int i = 5; i < 29; i++) {
            lookup_memory_id_to_big_2[i][row] = M31_0;
        }

        // opcodes_0: [pc, ap, fp]
        lookup_opcodes_0[0][row] = input_pc_col0;
        lookup_opcodes_0[1][row] = input_ap_col1;
        lookup_opcodes_0[2][row] = input_fp_col2;

        // opcodes_1: [next_pc_value, ap+2, ap+2]
        // next_pc_value = limb0 + limb1*512 + limb2*262144 + limb3*134217728
        m31 next_pc_combined = add(
            add(
                add(next_pc_limb_0_col19, mul(next_pc_limb_1_col20, M31_512)),
                mul(next_pc_limb_2_col21, M31_262144)
            ),
            mul(next_pc_limb_3_col22, M31_134217728)
        );
        m31 ap_plus_2 = add(input_ap_col1, M31_2);
        lookup_opcodes_1[0][row] = next_pc_combined;
        lookup_opcodes_1[1][row] = ap_plus_2;
        lookup_opcodes_1[2][row] = ap_plus_2;

        // Enabler column
        if (row < n_rows)
            traces[24][row] = 1;
        else
            traces[24][row] = 0;
    }
}

void generate_call_opcode_traces(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_verify_instruction_0,

    m31 **sub_component_inputs_verify_instruction,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,

    m31 **call_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    m31 **device_traces = clone_to_device<m31*>(traces, CALL_OPCODE_N_TRACE_COLUMNS);
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>(lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31 *>(lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>(lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31 *>(lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>(lookup_opcodes_1, 3);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>(lookup_verify_instruction_0, 7);

    m31 **device_sub_component_inputs_verify_instruction = clone_to_device<m31 *>(sub_component_inputs_verify_instruction, 1 * 7);
    m31 **device_sub_component_inputs_memory_address_to_id = clone_to_device<m31 *>(sub_component_inputs_memory_address_to_id, 3 * 1);
    m31 **device_sub_component_inputs_memory_id_to_big = clone_to_device<m31 *>(sub_component_inputs_memory_id_to_big, 3 * 1);

    m31 **device_call_opcode_input = clone_to_device<m31 *>(call_opcode_input, 3);
    unsigned **device_memory_id_to_big_transposed_big_values = clone_to_device<m31 *>(memory_id_to_big_transposed_big_values, 8);


    timer global_timer;
    global_timer.start("generate call_opcode base trace");

    unsigned trace_size = 1 << log_size;
    int block_dim = trace_size < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_call_opcode_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,

        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_address_to_id_2,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_id_to_big_1,
        device_lookup_memory_id_to_big_2,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        device_lookup_verify_instruction_0,

        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,

        device_call_opcode_input,

        memory_address_to_id_address_to_raw_id,

        device_memory_id_to_big_transposed_big_values,
        memory_id_to_big_small_values,

        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate call_opcode base trace");

    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_verify_instruction_0);

    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);

    cuda_free_memory(device_call_opcode_input);
    cuda_free_memory(device_memory_id_to_big_transposed_big_values);
}

// ============================================================================
// Interaction Trace Generation
// ============================================================================

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_call_opcode_interaction_trace_col_gen_kernel_round0(
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

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_call_opcode_interaction_trace_col_gen_kernel_second2last(
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

template <int N>
__launch_bounds__(256, 2)
__global__ void generate_call_opcode_interaction_trace_col_gen_kernel_last(
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

__global__ void generate_call_opcode_interaction_trace_finalize_col_kernel(
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

__global__ void generate_call_opcode_interaction_trace_cumsum_shift(
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

__global__ void generate_call_opcode_interaction_trace_coord_prefix_sum(
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

void generate_call_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    m31 **interaction_trace,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    MemoryAddressToId *memory_address_to_id_lookup_elements = (MemoryAddressToId *)memory_address_to_id;
    MemoryIdToBig *memory_id_to_big_lookup_elements = (MemoryIdToBig *)memory_id_to_big;
    Opcodes *opcodes_lookup_elements = (Opcodes *)opcodes;
    VerifyInstruction *verify_instruction_lookup_elements = (VerifyInstruction *)verify_instruction;

    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
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
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>(lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31 *>(lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>(lookup_opcodes_1, 3);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>(lookup_verify_instruction_0, 7);

    m31 **device_interaction_trace = clone_to_device<m31*>(interaction_trace, 4 * CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);

    timer global_timer;
    global_timer.start("generate call_opcode interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    // #0 Interaction trace: verify_instruction_0 + memory_address_to_id_0
    block_dim = trace_size < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_call_opcode_interaction_trace_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
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
    generate_call_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #1 Interaction trace: memory_id_to_big_0 + memory_address_to_id_1
    block_dim = trace_size < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_call_opcode_interaction_trace_col_gen_kernel_round0<29, 2><<<num_blocks, block_dim>>>(
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
    generate_call_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #2 Interaction trace: memory_id_to_big_1 + memory_address_to_id_2
    block_dim = trace_size < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_call_opcode_interaction_trace_col_gen_kernel_round0<29, 2><<<num_blocks, block_dim>>>(
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
    generate_call_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #3 Interaction trace: memory_id_to_big_2 * enabler + opcodes_0
    block_dim = trace_size < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_call_opcode_interaction_trace_col_gen_kernel_second2last<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_opcodes_lookup_elements,

        device_lookup_memory_id_to_big_2,
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
    generate_call_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #4 Interaction trace: -1 * enabler / opcodes_1
    block_dim = trace_size < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < CALL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_call_opcode_interaction_trace_col_gen_kernel_last<3><<<num_blocks, block_dim>>>(
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
    generate_call_opcode_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        4,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_trace
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Compute cumsum_shift
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_call_opcode_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_trace,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_call_opcode_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_trace[4 * CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * CALL_OPCODE_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate call_opcode interaction trace");

    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_opcodes_lookup_elements);
    cuda_free_memory(device_verify_instruction_lookup_elements);

    cuda_free_memory(device_logup_denom);

    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_verify_instruction_0);

    cuda_free_memory(device_interaction_trace);

    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}
