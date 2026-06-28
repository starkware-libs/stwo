#include "gen_jump_opcode_double_deref_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"
#include "cuda_mem_pool.cuh"
#include "prefix_sum.cuh"
#include "batch_inverse.cuh"
#include <cstdio>

// Constants used in trace generation
#define M31_0 0
#define M31_1 1
#define M31_2 2
#define M31_8 8
#define M31_16 16
#define M31_32 32
#define M31_512 512
#define M31_262144 262144
#define M31_134217728 134217728
#define M31_32767 32767
#define M31_32768 32768

__global__ void generate_jump_opcode_double_deref_trace_kernel(
    m31 **traces,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,
    m31 **lookup_verify_instruction_0,

    m31 **sub_component_inputs_verify_instruction,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,

    m31 **jump_opcode_double_deref_input,

    m31 *memory_address_to_id_address_to_raw_id,

    m31 **memory_id_to_big_transposed_big_values,
    m31 *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_rows = 1u << log_size;
    if (idx >= total_rows) return;

    // For padding rows (idx >= n_rows), use the first valid row's data
    unsigned data_idx = (idx < n_rows) ? idx : 0;

    // Input columns
    m31 input_pc = jump_opcode_double_deref_input[0][data_idx];
    m31 input_ap = jump_opcode_double_deref_input[1][data_idx];
    m31 input_fp = jump_opcode_double_deref_input[2][data_idx];

    // Decode instruction
    // Note: address_to_raw_id array is 0-indexed for 1-indexed addresses
    // So for address N, we access address_to_raw_id[N-1]
    m31 mem_id_0 = memory_address_to_id_address_to_raw_id[input_pc - 1];

    // Get memory_id_to_big value using the helper function
    m31 instr_limbs[28];
    memory_id_to_big_state_deduce_output(
        (unsigned **)memory_id_to_big_transposed_big_values,
        (unsigned *)memory_id_to_big_small_values,
        mem_id_0,
        instr_limbs
    );
    m31 limb0 = instr_limbs[0];
    m31 limb1 = instr_limbs[1];
    m31 limb2 = instr_limbs[2];
    m31 limb3 = instr_limbs[3];
    m31 limb4 = instr_limbs[4];
    m31 limb5 = instr_limbs[5];
    m31 limb6 = instr_limbs[6];

    // offset1 = (limb1 >> 7) + (limb2 << 2) + ((limb3 & 31) << 11)
    m31 offset1 = ((limb1 >> 7) + (limb2 << 2) + ((limb3 & 31) << 11)) & 0xFFFF;

    // offset2 = (limb3 >> 5) + (limb4 << 4) + ((limb5 & 7) << 13)
    m31 offset2 = ((limb3 >> 5) + (limb4 << 4) + ((limb5 & 7) << 13)) & 0xFFFF;

    // op0_base_fp = ((limb5 >> 3) + (limb6 << 6)) >> 1 & 1
    m31 flags_val = (limb5 >> 3) + (limb6 << 6);
    m31 op0_base_fp = (flags_val >> 1) & 1;

    // ap_update_add_1 = (flags_val >> 11) & 1
    m31 ap_update_add_1 = (flags_val >> 11) & 1;

    // mem0_base = op0_base_fp * fp + (1 - op0_base_fp) * ap (plain arithmetic)
    m31 mem0_base = {op0_base_fp * input_fp + (1 - op0_base_fp) * input_ap};

    // offset1_signed = offset1 - 32768 (use plain arithmetic for address calculation)
    m31 offset1_signed = {offset1 - M31_32768};

    // First memory read: mem0_base + offset1_signed (plain arithmetic)
    m31 mem1_base_addr = {mem0_base + offset1_signed};
    m31 mem1_base_id = memory_address_to_id_address_to_raw_id[mem1_base_addr - 1];

    // Get mem1_base value from memory_id_to_big
    m31 mem1_limbs[28];
    memory_id_to_big_state_deduce_output(
        (unsigned **)memory_id_to_big_transposed_big_values,
        (unsigned *)memory_id_to_big_small_values,
        mem1_base_id,
        mem1_limbs
    );
    m31 mem1_limb0 = mem1_limbs[0];
    m31 mem1_limb1 = mem1_limbs[1];
    m31 mem1_limb2 = mem1_limbs[2];
    m31 mem1_limb3 = mem1_limbs[3];

    // partial_limb_msb for mem1_base = (mem1_limb3 & 2) >> 1
    m31 partial_limb_msb_0 = (mem1_limb3 & 2) >> 1;

    // Compute mem1_base value (plain arithmetic for address calculation)
    m31 mem1_base_val = {mem1_limb0 + mem1_limb1 * M31_512 +
                         mem1_limb2 * M31_262144 + mem1_limb3 * M31_134217728};

    // offset2_signed = offset2 - 32768 (use plain arithmetic for address calculation)
    m31 offset2_signed = {offset2 - M31_32768};

    // Second memory read: mem1_base_val + offset2_signed (plain arithmetic)
    m31 next_pc_addr = {mem1_base_val + offset2_signed};
    m31 next_pc_id = memory_address_to_id_address_to_raw_id[next_pc_addr - 1];

    // Get next_pc value from memory_id_to_big
    m31 next_pc_limbs[28];
    memory_id_to_big_state_deduce_output(
        (unsigned **)memory_id_to_big_transposed_big_values,
        (unsigned *)memory_id_to_big_small_values,
        next_pc_id,
        next_pc_limbs
    );
    m31 next_pc_limb0 = next_pc_limbs[0];
    m31 next_pc_limb1 = next_pc_limbs[1];
    m31 next_pc_limb2 = next_pc_limbs[2];
    m31 next_pc_limb3 = next_pc_limbs[3];

    // partial_limb_msb for next_pc = (next_pc_limb3 & 2) >> 1
    m31 partial_limb_msb_1 = (next_pc_limb3 & 2) >> 1;

    // Compute next_pc value
    m31 next_pc = add(next_pc_limb0, mul(next_pc_limb1, M31_512));
    next_pc = add(next_pc, mul(next_pc_limb2, M31_262144));
    next_pc = add(next_pc, mul(next_pc_limb3, M31_134217728));

    // Enabler column
    m31 enabler = (idx < n_rows) ? M31_1 : M31_0;

    // Write trace columns
    traces[0][idx] = input_pc;
    traces[1][idx] = input_ap;
    traces[2][idx] = input_fp;
    traces[3][idx] = offset1;
    traces[4][idx] = offset2;
    traces[5][idx] = op0_base_fp;
    traces[6][idx] = ap_update_add_1;
    traces[7][idx] = mem0_base;
    traces[8][idx] = mem1_base_id;
    traces[9][idx] = mem1_limb0;
    traces[10][idx] = mem1_limb1;
    traces[11][idx] = mem1_limb2;
    traces[12][idx] = mem1_limb3;
    traces[13][idx] = partial_limb_msb_0;
    traces[14][idx] = next_pc_id;
    traces[15][idx] = next_pc_limb0;
    traces[16][idx] = next_pc_limb1;
    traces[17][idx] = next_pc_limb2;
    traces[18][idx] = next_pc_limb3;
    traces[19][idx] = partial_limb_msb_1;
    traces[20][idx] = enabler;

    // Write lookup data for verify_instruction_0 [7 fields]
    lookup_verify_instruction_0[0][idx] = input_pc;
    lookup_verify_instruction_0[1][idx] = M31_32767;
    lookup_verify_instruction_0[2][idx] = offset1;
    lookup_verify_instruction_0[3][idx] = offset2;
    lookup_verify_instruction_0[4][idx] = add(M31_8, mul(op0_base_fp, M31_16));
    lookup_verify_instruction_0[5][idx] = add(M31_2, mul(ap_update_add_1, M31_32));
    lookup_verify_instruction_0[6][idx] = M31_0;

    // Write lookup data for memory_address_to_id_0 [2 fields]
    lookup_memory_address_to_id_0[0][idx] = mem1_base_addr;
    lookup_memory_address_to_id_0[1][idx] = mem1_base_id;

    // Write lookup data for memory_address_to_id_1 [2 fields]
    lookup_memory_address_to_id_1[0][idx] = next_pc_addr;
    lookup_memory_address_to_id_1[1][idx] = next_pc_id;

    // Write lookup data for memory_id_to_big_0 [29 fields: 1 id + 4 limbs + 24 zeros]
    lookup_memory_id_to_big_0[0][idx] = mem1_base_id;
    lookup_memory_id_to_big_0[1][idx] = mem1_limbs[0];
    lookup_memory_id_to_big_0[2][idx] = mem1_limbs[1];
    lookup_memory_id_to_big_0[3][idx] = mem1_limbs[2];
    lookup_memory_id_to_big_0[4][idx] = mem1_limbs[3];
    for (int i = 5; i < 29; i++) {
        lookup_memory_id_to_big_0[i][idx] = M31_0;
    }

    // Write lookup data for memory_id_to_big_1 [29 fields: 1 id + 4 limbs + 24 zeros]
    lookup_memory_id_to_big_1[0][idx] = next_pc_id;
    lookup_memory_id_to_big_1[1][idx] = next_pc_limbs[0];
    lookup_memory_id_to_big_1[2][idx] = next_pc_limbs[1];
    lookup_memory_id_to_big_1[3][idx] = next_pc_limbs[2];
    lookup_memory_id_to_big_1[4][idx] = next_pc_limbs[3];
    for (int i = 5; i < 29; i++) {
        lookup_memory_id_to_big_1[i][idx] = M31_0;
    }

    // Write lookup data for opcodes_0 [3 fields]
    lookup_opcodes_0[0][idx] = input_pc;
    lookup_opcodes_0[1][idx] = input_ap;
    lookup_opcodes_0[2][idx] = input_fp;

    // Write lookup data for opcodes_1 [3 fields]
    lookup_opcodes_1[0][idx] = next_pc;
    lookup_opcodes_1[1][idx] = add(input_ap, ap_update_add_1);
    lookup_opcodes_1[2][idx] = input_fp;

    // Write sub_component_inputs for verify_instruction [1 input, 7 fields each]
    sub_component_inputs_verify_instruction[0][idx] = input_pc;
    sub_component_inputs_verify_instruction[1][idx] = M31_32767;
    sub_component_inputs_verify_instruction[2][idx] = offset1;
    sub_component_inputs_verify_instruction[3][idx] = offset2;
    sub_component_inputs_verify_instruction[4][idx] = add(M31_8, mul(op0_base_fp, M31_16));
    sub_component_inputs_verify_instruction[5][idx] = add(M31_2, mul(ap_update_add_1, M31_32));
    sub_component_inputs_verify_instruction[6][idx] = M31_0;

    // Write sub_component_inputs for memory_address_to_id [2 inputs, 1 field each]
    sub_component_inputs_memory_address_to_id[0][idx] = mem1_base_addr;
    sub_component_inputs_memory_address_to_id[1][idx] = next_pc_addr;

    // Write sub_component_inputs for memory_id_to_big [2 inputs, 1 field each]
    sub_component_inputs_memory_id_to_big[0][idx] = mem1_base_id;
    sub_component_inputs_memory_id_to_big[1][idx] = next_pc_id;
}

void generate_jump_opcode_double_deref_traces(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,

    unsigned **jump_opcode_double_deref_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {

    // Early return if n_rows is 0 - nothing to generate
    if (n_rows == 0) {
        printf("jump_opcode_double_deref: n_rows=0, skipping kernel\n");
        return;
    }

    unsigned total_rows = 1u << log_size;
    unsigned block_size = JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    unsigned grid_size = (total_rows + block_size - 1) / block_size;

    // Copy arrays to device
    m31 **device_traces = clone_to_device<m31 *>((m31 **)traces, JUMP_OPCODE_DOUBLE_DEREF_N_TRACE_COLUMNS);
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>((m31 **)lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>((m31 **)lookup_opcodes_1, 3);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>((m31 **)lookup_verify_instruction_0, 7);
    m31 **device_sub_component_inputs_verify_instruction = clone_to_device<m31 *>((m31 **)sub_component_inputs_verify_instruction, 7);
    m31 **device_sub_component_inputs_memory_address_to_id = clone_to_device<m31 *>((m31 **)sub_component_inputs_memory_address_to_id, 2);
    m31 **device_sub_component_inputs_memory_id_to_big = clone_to_device<m31 *>((m31 **)sub_component_inputs_memory_id_to_big, 2);
    m31 **device_jump_opcode_double_deref_input = clone_to_device<m31 *>((m31 **)jump_opcode_double_deref_input, 3);
    m31 **device_memory_id_to_big_transposed_big_values = clone_to_device<m31 *>((m31 **)memory_id_to_big_transposed_big_values, 28);

    generate_jump_opcode_double_deref_trace_kernel<<<grid_size, block_size>>>(
        device_traces,
        device_lookup_memory_address_to_id_0,
        device_lookup_memory_address_to_id_1,
        device_lookup_memory_id_to_big_0,
        device_lookup_memory_id_to_big_1,
        device_lookup_opcodes_0,
        device_lookup_opcodes_1,
        device_lookup_verify_instruction_0,
        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        device_jump_opcode_double_deref_input,
        (m31 *)memory_address_to_id_address_to_raw_id,
        device_memory_id_to_big_transposed_big_values,
        (m31 *)memory_id_to_big_small_values,
        n_rows,
        log_size
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);
    cuda_free_memory(device_lookup_verify_instruction_0);
    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);
    cuda_free_memory(device_jump_opcode_double_deref_input);
    cuda_free_memory(device_memory_id_to_big_transposed_big_values);
}

// ============================================================================
// Interaction Trace Generation
// ============================================================================

// Round 0: verify_instruction + memory_address_to_id_0
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_round0(
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

// Round 1: memory_id_to_big_0 + memory_address_to_id_1
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_round1(
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

// Second2last: memory_id_to_big_1*enabler + opcodes_0
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_second2last(
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

// Last: -enabler * opcodes_1
template <int N>
__launch_bounds__(256, 2)
__global__ void generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_last(
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

__global__ void generate_jump_opcode_double_deref_interaction_trace_finalize_col_kernel(
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

__global__ void generate_jump_opcode_double_deref_interaction_trace_cumsum_shift(
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

__global__ void generate_jump_opcode_double_deref_interaction_trace_coord_prefix_sum(
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

void generate_jump_opcode_double_deref_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
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

    MemoryAddressToId *device_memory_address_to_id_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig *device_memory_id_to_big_lookup_elements = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes_lookup_elements = cuda_malloc<Opcodes>(1);
    VerifyInstruction *device_verify_instruction_lookup_elements = cuda_malloc<VerifyInstruction>(1);

    cuda_mem_copy_host_to_device<MemoryAddressToId>(memory_address_to_id_lookup_elements, device_memory_address_to_id_lookup_elements, 1);
    cuda_mem_copy_host_to_device<MemoryIdToBig>(memory_id_to_big_lookup_elements, device_memory_id_to_big_lookup_elements, 1);
    cuda_mem_copy_host_to_device<Opcodes>(opcodes_lookup_elements, device_opcodes_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyInstruction>(verify_instruction_lookup_elements, device_verify_instruction_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31 *>((m31 **)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31 *>((m31 **)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_opcodes_0 = clone_to_device<m31 *>((m31 **)lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31 *>((m31 **)lookup_opcodes_1, 3);
    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31 *>((m31 **)lookup_verify_instruction_0, 7);

    m31 **device_interaction_traces = clone_to_device<m31*>((m31 **)interaction_traces, 4 * JUMP_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For verify_instruction_0 & memory_address_to_id_0
    generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_round0<7, 2><<<num_blocks, block_dim>>>(
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
    generate_jump_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #1 Interaction trace For memory_id_to_big_0 & memory_address_to_id_1
    block_dim = trace_size < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_round1<29, 2><<<num_blocks, block_dim>>>(
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
    generate_jump_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #2 Interaction trace For memory_id_to_big_1 * enabler & opcodes_0
    block_dim = trace_size < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_second2last<29, 3><<<num_blocks, block_dim>>>(
        device_memory_id_to_big_lookup_elements,
        device_opcodes_lookup_elements,
        device_lookup_memory_id_to_big_1,
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
    generate_jump_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #3 Interaction trace For opcodes_1 (with -enabler)
    block_dim = trace_size < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < JUMP_OPCODE_DOUBLE_DEREF_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_jump_opcode_double_deref_interaction_trace_col_gen_kernel_last<3><<<num_blocks, block_dim>>>(
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
    generate_jump_opcode_double_deref_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_jump_opcode_double_deref_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        JUMP_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        (m31 *)claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_jump_opcode_double_deref_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        (m31 *)claimed_sum,
        JUMP_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum((m31 *)interaction_traces[4 * JUMP_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum((m31 *)interaction_traces[4 * JUMP_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum((m31 *)interaction_traces[4 * JUMP_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum((m31 *)interaction_traces[4 * JUMP_OPCODE_DOUBLE_DEREF_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    cuda_free_memory(device_memory_address_to_id_lookup_elements);
    cuda_free_memory(device_memory_id_to_big_lookup_elements);
    cuda_free_memory(device_opcodes_lookup_elements);
    cuda_free_memory(device_verify_instruction_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
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
