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

#include "gen_bitwise_builtin_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

// Base trace generation kernel for bitwise_builtin
// 89 trace columns, 5 memory_address_to_id lookups, 5 memory_id_to_big lookups,
// 27 verify_bitwise_xor_9 lookups, 1 verify_bitwise_xor_8 lookup
__launch_bounds__(BITWISE_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_bitwise_builtin_trace_kernel(
    m31 **traces,

    // Lookup data arrays - 5 MemoryAddressToId lookups (2 elements each)
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,

    // Lookup data arrays - 5 MemoryIdToBig lookups (29 elements each)
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,

    // Lookup data arrays - 27 VerifyBitwiseXor9 lookups (3 elements each)
    m31 **lookup_verify_bitwise_xor_9_0,
    m31 **lookup_verify_bitwise_xor_9_1,
    m31 **lookup_verify_bitwise_xor_9_2,
    m31 **lookup_verify_bitwise_xor_9_3,
    m31 **lookup_verify_bitwise_xor_9_4,
    m31 **lookup_verify_bitwise_xor_9_5,
    m31 **lookup_verify_bitwise_xor_9_6,
    m31 **lookup_verify_bitwise_xor_9_7,
    m31 **lookup_verify_bitwise_xor_9_8,
    m31 **lookup_verify_bitwise_xor_9_9,
    m31 **lookup_verify_bitwise_xor_9_10,
    m31 **lookup_verify_bitwise_xor_9_11,
    m31 **lookup_verify_bitwise_xor_9_12,
    m31 **lookup_verify_bitwise_xor_9_13,
    m31 **lookup_verify_bitwise_xor_9_14,
    m31 **lookup_verify_bitwise_xor_9_15,
    m31 **lookup_verify_bitwise_xor_9_16,
    m31 **lookup_verify_bitwise_xor_9_17,
    m31 **lookup_verify_bitwise_xor_9_18,
    m31 **lookup_verify_bitwise_xor_9_19,
    m31 **lookup_verify_bitwise_xor_9_20,
    m31 **lookup_verify_bitwise_xor_9_21,
    m31 **lookup_verify_bitwise_xor_9_22,
    m31 **lookup_verify_bitwise_xor_9_23,
    m31 **lookup_verify_bitwise_xor_9_24,
    m31 **lookup_verify_bitwise_xor_9_25,
    m31 **lookup_verify_bitwise_xor_9_26,

    // Lookup data arrays - 1 VerifyBitwiseXor8 lookup (3 elements each)
    m31 **lookup_verify_bitwise_xor_8_0,

    // Sub-component inputs
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_verify_bitwise_xor_9,
    m31 **sub_component_inputs_verify_bitwise_xor_8,

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
    // 1073741824 = 2^30, which is the multiplicative inverse of 2 in M31
    // (2 * 2^30 = 2^31 = 1 mod (2^31-1))
    const m31 M31_1073741824 = {1073741824};

    if (row < trace_size) {
        // seq = row index
        m31 seq = {row};
        m31 segment_start_m31 = {segment_start};

        // instance_addr = segment_start + 5 * seq
        // Memory layout: [op0, op1, and, xor, or] per instance
        m31 instance_addr = add(segment_start_m31, mul(M31_5, seq));

        // ============ Read op0 (instance_addr + 0) ============
        m31 op0_addr = instance_addr;
        m31 op0_id_col0 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op0_addr,
            &op0_id_col0
        );
        traces[0][row] = op0_id_col0;
        sub_component_inputs_memory_address_to_id[0][row] = op0_addr;
        lookup_memory_address_to_id_0[0][row] = op0_addr;
        lookup_memory_address_to_id_0[1][row] = op0_id_col0;

        // Read 28 limbs for op0 (252-bit value: 27 x 9-bit + 1 x 8-bit)
        m31 op0_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            op0_id_col0,
            op0_value
        );

        // op0 limbs (columns 1-28)
        m31 op0_limb[28];
        for (int i = 0; i < 28; i++) {
            op0_limb[i] = op0_value[i];
            traces[1 + i][row] = op0_limb[i];
        }

        sub_component_inputs_memory_id_to_big[0][row] = op0_id_col0;
        lookup_memory_id_to_big_0[0][row] = op0_id_col0;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_0[1 + i][row] = op0_limb[i];
        }

        // ============ Read op1 (instance_addr + 1) ============
        m31 op1_addr = add(instance_addr, M31_1);
        m31 op1_id_col29 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            op1_addr,
            &op1_id_col29
        );
        traces[29][row] = op1_id_col29;
        sub_component_inputs_memory_address_to_id[1][row] = op1_addr;
        lookup_memory_address_to_id_1[0][row] = op1_addr;
        lookup_memory_address_to_id_1[1][row] = op1_id_col29;

        // Read 28 limbs for op1 (252-bit value)
        m31 op1_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            op1_id_col29,
            op1_value
        );

        // op1 limbs (columns 30-57)
        m31 op1_limb[28];
        for (int i = 0; i < 28; i++) {
            op1_limb[i] = op1_value[i];
            traces[30 + i][row] = op1_limb[i];
        }

        sub_component_inputs_memory_id_to_big[1][row] = op1_id_col29;
        lookup_memory_id_to_big_1[0][row] = op1_id_col29;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_1[1 + i][row] = op1_limb[i];
        }

        // ============ Compute XOR and AND for each limb ============
        // XOR is computed directly, AND = (op0 + op1 - xor) * 2^30 mod M31
        m31 xor_result[28];
        m31 and_result[28];

        // Array of lookup pointers for verify_bitwise_xor_9
        m31 **xor9_lookups[27] = {
            lookup_verify_bitwise_xor_9_0, lookup_verify_bitwise_xor_9_1,
            lookup_verify_bitwise_xor_9_2, lookup_verify_bitwise_xor_9_3,
            lookup_verify_bitwise_xor_9_4, lookup_verify_bitwise_xor_9_5,
            lookup_verify_bitwise_xor_9_6, lookup_verify_bitwise_xor_9_7,
            lookup_verify_bitwise_xor_9_8, lookup_verify_bitwise_xor_9_9,
            lookup_verify_bitwise_xor_9_10, lookup_verify_bitwise_xor_9_11,
            lookup_verify_bitwise_xor_9_12, lookup_verify_bitwise_xor_9_13,
            lookup_verify_bitwise_xor_9_14, lookup_verify_bitwise_xor_9_15,
            lookup_verify_bitwise_xor_9_16, lookup_verify_bitwise_xor_9_17,
            lookup_verify_bitwise_xor_9_18, lookup_verify_bitwise_xor_9_19,
            lookup_verify_bitwise_xor_9_20, lookup_verify_bitwise_xor_9_21,
            lookup_verify_bitwise_xor_9_22, lookup_verify_bitwise_xor_9_23,
            lookup_verify_bitwise_xor_9_24, lookup_verify_bitwise_xor_9_25,
            lookup_verify_bitwise_xor_9_26
        };

        // Compute XOR for limbs 0-26 (9-bit XOR)
        for (int i = 0; i < 27; i++) {
            // XOR is computed as bitwise XOR of the two limbs
            uint16_t xor_val = ((uint16_t)op0_limb[i]) ^ ((uint16_t)op1_limb[i]);
            xor_result[i] = {xor_val};
            traces[58 + i][row] = xor_result[i];

            // AND = (op0 + op1 - xor) * 2^30 mod M31
            and_result[i] = mul(M31_1073741824, sub(add(op0_limb[i], op1_limb[i]), xor_result[i]));

            // Fill lookup data for verify_bitwise_xor_9
            xor9_lookups[i][0][row] = op0_limb[i];
            xor9_lookups[i][1][row] = op1_limb[i];
            xor9_lookups[i][2][row] = xor_result[i];

            // Sub-component inputs for verify_bitwise_xor_9
            sub_component_inputs_verify_bitwise_xor_9[i * 3][row] = op0_limb[i];
            sub_component_inputs_verify_bitwise_xor_9[i * 3 + 1][row] = op1_limb[i];
            sub_component_inputs_verify_bitwise_xor_9[i * 3 + 2][row] = xor_result[i];
        }

        // Compute XOR for limb 27 (8-bit XOR)
        {
            uint16_t xor_val = ((uint16_t)op0_limb[27]) ^ ((uint16_t)op1_limb[27]);
            xor_result[27] = {xor_val};
            traces[85][row] = xor_result[27];

            // AND = (op0 + op1 - xor) * 2^30 mod M31
            and_result[27] = mul(M31_1073741824, sub(add(op0_limb[27], op1_limb[27]), xor_result[27]));

            // Fill lookup data for verify_bitwise_xor_8
            lookup_verify_bitwise_xor_8_0[0][row] = op0_limb[27];
            lookup_verify_bitwise_xor_8_0[1][row] = op1_limb[27];
            lookup_verify_bitwise_xor_8_0[2][row] = xor_result[27];

            // Sub-component inputs for verify_bitwise_xor_8
            sub_component_inputs_verify_bitwise_xor_8[0][row] = op0_limb[27];
            sub_component_inputs_verify_bitwise_xor_8[1][row] = op1_limb[27];
            sub_component_inputs_verify_bitwise_xor_8[2][row] = xor_result[27];
        }

        // ============ Read and_id (instance_addr + 2) ============
        m31 and_addr = add(instance_addr, M31_2);
        m31 and_id_col86 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            and_addr,
            &and_id_col86
        );
        traces[86][row] = and_id_col86;
        sub_component_inputs_memory_address_to_id[2][row] = and_addr;
        lookup_memory_address_to_id_2[0][row] = and_addr;
        lookup_memory_address_to_id_2[1][row] = and_id_col86;

        // memory_id_to_big_2 - Store computed AND result
        sub_component_inputs_memory_id_to_big[2][row] = and_id_col86;
        lookup_memory_id_to_big_2[0][row] = and_id_col86;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_2[1 + i][row] = and_result[i];
        }

        // ============ Read xor_id (instance_addr + 3) ============
        m31 xor_addr = add(instance_addr, M31_3);
        m31 xor_id_col87 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            xor_addr,
            &xor_id_col87
        );
        traces[87][row] = xor_id_col87;
        sub_component_inputs_memory_address_to_id[3][row] = xor_addr;
        lookup_memory_address_to_id_3[0][row] = xor_addr;
        lookup_memory_address_to_id_3[1][row] = xor_id_col87;

        // memory_id_to_big_3 - Store XOR result
        sub_component_inputs_memory_id_to_big[3][row] = xor_id_col87;
        lookup_memory_id_to_big_3[0][row] = xor_id_col87;
        for (int i = 0; i < 28; i++) {
            lookup_memory_id_to_big_3[1 + i][row] = xor_result[i];
        }

        // ============ Read or_id (instance_addr + 4) ============
        m31 or_addr = add(instance_addr, M31_4);
        m31 or_id_col88 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            or_addr,
            &or_id_col88
        );
        traces[88][row] = or_id_col88;
        sub_component_inputs_memory_address_to_id[4][row] = or_addr;
        lookup_memory_address_to_id_4[0][row] = or_addr;
        lookup_memory_address_to_id_4[1][row] = or_id_col88;

        // memory_id_to_big_4 - Store OR result (and + xor)
        sub_component_inputs_memory_id_to_big[4][row] = or_id_col88;
        lookup_memory_id_to_big_4[0][row] = or_id_col88;
        for (int i = 0; i < 28; i++) {
            // OR = AND + XOR
            m31 or_limb = add(and_result[i], xor_result[i]);
            lookup_memory_id_to_big_4[1 + i][row] = or_limb;
        }
    }
}

// Host wrapper function
extern "C"
void generate_bitwise_builtin_traces(
    unsigned **traces,

    // Lookup data arrays - 5 MemoryAddressToId lookups
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_address_to_id_3,
    unsigned **lookup_memory_address_to_id_4,

    // Lookup data arrays - 5 MemoryIdToBig lookups
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_memory_id_to_big_3,
    unsigned **lookup_memory_id_to_big_4,

    // Lookup data arrays - 27 VerifyBitwiseXor9 lookups
    unsigned **lookup_verify_bitwise_xor_9_0,
    unsigned **lookup_verify_bitwise_xor_9_1,
    unsigned **lookup_verify_bitwise_xor_9_2,
    unsigned **lookup_verify_bitwise_xor_9_3,
    unsigned **lookup_verify_bitwise_xor_9_4,
    unsigned **lookup_verify_bitwise_xor_9_5,
    unsigned **lookup_verify_bitwise_xor_9_6,
    unsigned **lookup_verify_bitwise_xor_9_7,
    unsigned **lookup_verify_bitwise_xor_9_8,
    unsigned **lookup_verify_bitwise_xor_9_9,
    unsigned **lookup_verify_bitwise_xor_9_10,
    unsigned **lookup_verify_bitwise_xor_9_11,
    unsigned **lookup_verify_bitwise_xor_9_12,
    unsigned **lookup_verify_bitwise_xor_9_13,
    unsigned **lookup_verify_bitwise_xor_9_14,
    unsigned **lookup_verify_bitwise_xor_9_15,
    unsigned **lookup_verify_bitwise_xor_9_16,
    unsigned **lookup_verify_bitwise_xor_9_17,
    unsigned **lookup_verify_bitwise_xor_9_18,
    unsigned **lookup_verify_bitwise_xor_9_19,
    unsigned **lookup_verify_bitwise_xor_9_20,
    unsigned **lookup_verify_bitwise_xor_9_21,
    unsigned **lookup_verify_bitwise_xor_9_22,
    unsigned **lookup_verify_bitwise_xor_9_23,
    unsigned **lookup_verify_bitwise_xor_9_24,
    unsigned **lookup_verify_bitwise_xor_9_25,
    unsigned **lookup_verify_bitwise_xor_9_26,

    // Lookup data arrays - 1 VerifyBitwiseXor8 lookup
    unsigned **lookup_verify_bitwise_xor_8_0,

    // Sub-component inputs
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_verify_bitwise_xor_9,
    unsigned **sub_component_inputs_verify_bitwise_xor_8,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    unsigned trace_size = 1 << log_size;

    dim3 blockDim(BITWISE_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX);
    dim3 gridDim((trace_size + blockDim.x - 1) / blockDim.x);

    // Copy pointer arrays to device memory
    m31 **device_traces = clone_to_device<m31*>((m31**)traces, BITWISE_BUILTIN_N_TRACE_COLUMNS);

    // Memory address to id lookups (2 elements each)
    m31 **device_lookup_addr2id_0 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_addr2id_1 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_addr2id_2 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_addr2id_3 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_addr2id_4 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_4, 2);

    // Memory id to big lookups (29 elements each)
    m31 **device_lookup_id2big_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_id2big_1 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_id2big_2 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_id2big_3 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_id2big_4 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_4, 29);

    // Verify bitwise xor 9 lookups (3 elements each)
    m31 **device_lookup_xor9_0 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_0, 3);
    m31 **device_lookup_xor9_1 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_1, 3);
    m31 **device_lookup_xor9_2 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_2, 3);
    m31 **device_lookup_xor9_3 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_3, 3);
    m31 **device_lookup_xor9_4 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_4, 3);
    m31 **device_lookup_xor9_5 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_5, 3);
    m31 **device_lookup_xor9_6 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_6, 3);
    m31 **device_lookup_xor9_7 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_7, 3);
    m31 **device_lookup_xor9_8 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_8, 3);
    m31 **device_lookup_xor9_9 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_9, 3);
    m31 **device_lookup_xor9_10 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_10, 3);
    m31 **device_lookup_xor9_11 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_11, 3);
    m31 **device_lookup_xor9_12 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_12, 3);
    m31 **device_lookup_xor9_13 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_13, 3);
    m31 **device_lookup_xor9_14 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_14, 3);
    m31 **device_lookup_xor9_15 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_15, 3);
    m31 **device_lookup_xor9_16 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_16, 3);
    m31 **device_lookup_xor9_17 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_17, 3);
    m31 **device_lookup_xor9_18 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_18, 3);
    m31 **device_lookup_xor9_19 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_19, 3);
    m31 **device_lookup_xor9_20 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_20, 3);
    m31 **device_lookup_xor9_21 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_21, 3);
    m31 **device_lookup_xor9_22 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_22, 3);
    m31 **device_lookup_xor9_23 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_23, 3);
    m31 **device_lookup_xor9_24 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_24, 3);
    m31 **device_lookup_xor9_25 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_25, 3);
    m31 **device_lookup_xor9_26 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_26, 3);

    // Verify bitwise xor 8 lookup (3 elements)
    m31 **device_lookup_xor8_0 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_8_0, 3);

    // Sub-component inputs
    m31 **device_sub_addr2id = clone_to_device<m31*>((m31**)sub_component_inputs_memory_address_to_id, 5);
    m31 **device_sub_id2big = clone_to_device<m31*>((m31**)sub_component_inputs_memory_id_to_big, 5);
    m31 **device_sub_xor9 = clone_to_device<m31*>((m31**)sub_component_inputs_verify_bitwise_xor_9, 81);
    m31 **device_sub_xor8 = clone_to_device<m31*>((m31**)sub_component_inputs_verify_bitwise_xor_8, 3);

    // Clone memory_id_to_big transposed_big_values pointer array to device
    // (8 device pointers stored in host memory — must be on device for kernel access)
    m31 **device_id2big_transposed = clone_to_device<m31*>((m31**)memory_id_to_big_transposed_big_values, 8);

    generate_bitwise_builtin_trace_kernel<<<gridDim, blockDim>>>(
        device_traces,

        device_lookup_addr2id_0,
        device_lookup_addr2id_1,
        device_lookup_addr2id_2,
        device_lookup_addr2id_3,
        device_lookup_addr2id_4,

        device_lookup_id2big_0,
        device_lookup_id2big_1,
        device_lookup_id2big_2,
        device_lookup_id2big_3,
        device_lookup_id2big_4,

        device_lookup_xor9_0,
        device_lookup_xor9_1,
        device_lookup_xor9_2,
        device_lookup_xor9_3,
        device_lookup_xor9_4,
        device_lookup_xor9_5,
        device_lookup_xor9_6,
        device_lookup_xor9_7,
        device_lookup_xor9_8,
        device_lookup_xor9_9,
        device_lookup_xor9_10,
        device_lookup_xor9_11,
        device_lookup_xor9_12,
        device_lookup_xor9_13,
        device_lookup_xor9_14,
        device_lookup_xor9_15,
        device_lookup_xor9_16,
        device_lookup_xor9_17,
        device_lookup_xor9_18,
        device_lookup_xor9_19,
        device_lookup_xor9_20,
        device_lookup_xor9_21,
        device_lookup_xor9_22,
        device_lookup_xor9_23,
        device_lookup_xor9_24,
        device_lookup_xor9_25,
        device_lookup_xor9_26,

        device_lookup_xor8_0,

        device_sub_addr2id,
        device_sub_id2big,
        device_sub_xor9,
        device_sub_xor8,

        segment_start,

        (m31*)memory_address_to_id_address_to_raw_id,
        device_id2big_transposed,
        (m31*)memory_id_to_big_small_values,

        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Free device memory
    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_addr2id_1);
    cuda_free_memory(device_lookup_addr2id_2);
    cuda_free_memory(device_lookup_addr2id_3);
    cuda_free_memory(device_lookup_addr2id_4);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_lookup_id2big_1);
    cuda_free_memory(device_lookup_id2big_2);
    cuda_free_memory(device_lookup_id2big_3);
    cuda_free_memory(device_lookup_id2big_4);
    cuda_free_memory(device_lookup_xor9_0);
    cuda_free_memory(device_lookup_xor9_1);
    cuda_free_memory(device_lookup_xor9_2);
    cuda_free_memory(device_lookup_xor9_3);
    cuda_free_memory(device_lookup_xor9_4);
    cuda_free_memory(device_lookup_xor9_5);
    cuda_free_memory(device_lookup_xor9_6);
    cuda_free_memory(device_lookup_xor9_7);
    cuda_free_memory(device_lookup_xor9_8);
    cuda_free_memory(device_lookup_xor9_9);
    cuda_free_memory(device_lookup_xor9_10);
    cuda_free_memory(device_lookup_xor9_11);
    cuda_free_memory(device_lookup_xor9_12);
    cuda_free_memory(device_lookup_xor9_13);
    cuda_free_memory(device_lookup_xor9_14);
    cuda_free_memory(device_lookup_xor9_15);
    cuda_free_memory(device_lookup_xor9_16);
    cuda_free_memory(device_lookup_xor9_17);
    cuda_free_memory(device_lookup_xor9_18);
    cuda_free_memory(device_lookup_xor9_19);
    cuda_free_memory(device_lookup_xor9_20);
    cuda_free_memory(device_lookup_xor9_21);
    cuda_free_memory(device_lookup_xor9_22);
    cuda_free_memory(device_lookup_xor9_23);
    cuda_free_memory(device_lookup_xor9_24);
    cuda_free_memory(device_lookup_xor9_25);
    cuda_free_memory(device_lookup_xor9_26);
    cuda_free_memory(device_lookup_xor8_0);
    cuda_free_memory(device_sub_addr2id);
    cuda_free_memory(device_sub_id2big);
    cuda_free_memory(device_sub_xor9);
    cuda_free_memory(device_sub_xor8);
    cuda_free_memory(device_id2big_transposed);
}

// Column generation kernel for pairs of lookups (N elements + M elements)
template <int N, int M>
__launch_bounds__(BITWISE_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_bitwise_builtin_interaction_col_gen_kernel(
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

// Finalize column kernel - accumulates interaction trace values
__global__ void generate_bitwise_builtin_interaction_finalize_col_kernel(
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
__global__ void generate_bitwise_builtin_interaction_cumsum_shift_kernel(
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
__global__ void generate_bitwise_builtin_interaction_coord_prefix_sum_kernel(
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

// Legacy interaction trace generation kernel for bitwise_builtin (unused - kept for reference)
__launch_bounds__(BITWISE_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_bitwise_builtin_interaction_trace_kernel_legacy(
    MemoryAddressToId *memory_address_to_id,
    MemoryIdToBig *memory_id_to_big,
    VerifyBitwiseXor_9 *verify_bitwise_xor_9,
    VerifyBitwiseXor_8 *verify_bitwise_xor_8,

    // Lookup data arrays - 5 MemoryAddressToId lookups
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,

    // Lookup data arrays - 5 MemoryIdToBig lookups
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,

    // Lookup data arrays - 27 VerifyBitwiseXor9 lookups
    m31 **lookup_verify_bitwise_xor_9_0,
    m31 **lookup_verify_bitwise_xor_9_1,
    m31 **lookup_verify_bitwise_xor_9_2,
    m31 **lookup_verify_bitwise_xor_9_3,
    m31 **lookup_verify_bitwise_xor_9_4,
    m31 **lookup_verify_bitwise_xor_9_5,
    m31 **lookup_verify_bitwise_xor_9_6,
    m31 **lookup_verify_bitwise_xor_9_7,
    m31 **lookup_verify_bitwise_xor_9_8,
    m31 **lookup_verify_bitwise_xor_9_9,
    m31 **lookup_verify_bitwise_xor_9_10,
    m31 **lookup_verify_bitwise_xor_9_11,
    m31 **lookup_verify_bitwise_xor_9_12,
    m31 **lookup_verify_bitwise_xor_9_13,
    m31 **lookup_verify_bitwise_xor_9_14,
    m31 **lookup_verify_bitwise_xor_9_15,
    m31 **lookup_verify_bitwise_xor_9_16,
    m31 **lookup_verify_bitwise_xor_9_17,
    m31 **lookup_verify_bitwise_xor_9_18,
    m31 **lookup_verify_bitwise_xor_9_19,
    m31 **lookup_verify_bitwise_xor_9_20,
    m31 **lookup_verify_bitwise_xor_9_21,
    m31 **lookup_verify_bitwise_xor_9_22,
    m31 **lookup_verify_bitwise_xor_9_23,
    m31 **lookup_verify_bitwise_xor_9_24,
    m31 **lookup_verify_bitwise_xor_9_25,
    m31 **lookup_verify_bitwise_xor_9_26,

    // Lookup data arrays - 1 VerifyBitwiseXor8 lookup
    m31 **lookup_verify_bitwise_xor_8_0,

    unsigned n_rows,
    m31 **interaction_trace
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows) {
        // Helper lambda to write a qm31 fraction (num, denom) to 8 m31 columns
        // pair_idx is the logical pair index (0-18)
        // Each pair occupies 8 m31 columns: 4 for num, 4 for denom
        #define WRITE_FRAC(pair_idx, num, denom) do { \
            unsigned base_col = (pair_idx) * 8; \
            interaction_trace[base_col + 0][row] = (num).a.a; \
            interaction_trace[base_col + 1][row] = (num).a.b; \
            interaction_trace[base_col + 2][row] = (num).b.a; \
            interaction_trace[base_col + 3][row] = (num).b.b; \
            interaction_trace[base_col + 4][row] = (denom).a.a; \
            interaction_trace[base_col + 5][row] = (denom).a.b; \
            interaction_trace[base_col + 6][row] = (denom).b.a; \
            interaction_trace[base_col + 7][row] = (denom).b.b; \
        } while(0)

        // Pair 0: memory_address_to_id_0 + memory_id_to_big_0
        {
            m31 lookup_addr2id_0[2] = {
                lookup_memory_address_to_id_0[0][row],
                lookup_memory_address_to_id_0[1][row]
            };
            m31 lookup_id2big_0[29];
            for (int i = 0; i < 29; i++) {
                lookup_id2big_0[i] = lookup_memory_id_to_big_0[i][row];
            }
            qm31 denom0 = memory_address_to_id->combine(lookup_addr2id_0, 2);
            qm31 denom1 = memory_id_to_big->combine(lookup_id2big_0, 29);
            qm31 num = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            WRITE_FRAC(0, num, denom);
        }

        // Pair 1: memory_address_to_id_1 + memory_id_to_big_1
        {
            m31 lookup_addr2id_1[2] = {
                lookup_memory_address_to_id_1[0][row],
                lookup_memory_address_to_id_1[1][row]
            };
            m31 lookup_id2big_1[29];
            for (int i = 0; i < 29; i++) {
                lookup_id2big_1[i] = lookup_memory_id_to_big_1[i][row];
            }
            qm31 denom0 = memory_address_to_id->combine(lookup_addr2id_1, 2);
            qm31 denom1 = memory_id_to_big->combine(lookup_id2big_1, 29);
            qm31 num = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            WRITE_FRAC(1, num, denom);
        }

        // Pairs 2-14: verify_bitwise_xor_9 pairs (0,1), (2,3), ... (24,25)
        m31 **xor9_lookups[27] = {
            lookup_verify_bitwise_xor_9_0, lookup_verify_bitwise_xor_9_1,
            lookup_verify_bitwise_xor_9_2, lookup_verify_bitwise_xor_9_3,
            lookup_verify_bitwise_xor_9_4, lookup_verify_bitwise_xor_9_5,
            lookup_verify_bitwise_xor_9_6, lookup_verify_bitwise_xor_9_7,
            lookup_verify_bitwise_xor_9_8, lookup_verify_bitwise_xor_9_9,
            lookup_verify_bitwise_xor_9_10, lookup_verify_bitwise_xor_9_11,
            lookup_verify_bitwise_xor_9_12, lookup_verify_bitwise_xor_9_13,
            lookup_verify_bitwise_xor_9_14, lookup_verify_bitwise_xor_9_15,
            lookup_verify_bitwise_xor_9_16, lookup_verify_bitwise_xor_9_17,
            lookup_verify_bitwise_xor_9_18, lookup_verify_bitwise_xor_9_19,
            lookup_verify_bitwise_xor_9_20, lookup_verify_bitwise_xor_9_21,
            lookup_verify_bitwise_xor_9_22, lookup_verify_bitwise_xor_9_23,
            lookup_verify_bitwise_xor_9_24, lookup_verify_bitwise_xor_9_25,
            lookup_verify_bitwise_xor_9_26
        };

        // 13 pairs of xor_9 lookups
        for (int p = 0; p < 13; p++) {
            int idx0 = p * 2;
            int idx1 = p * 2 + 1;
            m31 lookup0[3] = {
                xor9_lookups[idx0][0][row],
                xor9_lookups[idx0][1][row],
                xor9_lookups[idx0][2][row]
            };
            m31 lookup1[3] = {
                xor9_lookups[idx1][0][row],
                xor9_lookups[idx1][1][row],
                xor9_lookups[idx1][2][row]
            };
            qm31 denom0 = verify_bitwise_xor_9->combine(lookup0, 3);
            qm31 denom1 = verify_bitwise_xor_9->combine(lookup1, 3);
            qm31 num = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            WRITE_FRAC(2 + p, num, denom);
        }

        // Pair 15: xor_9[26] + xor_8[0]
        {
            m31 lookup_xor9[3] = {
                lookup_verify_bitwise_xor_9_26[0][row],
                lookup_verify_bitwise_xor_9_26[1][row],
                lookup_verify_bitwise_xor_9_26[2][row]
            };
            m31 lookup_xor8[3] = {
                lookup_verify_bitwise_xor_8_0[0][row],
                lookup_verify_bitwise_xor_8_0[1][row],
                lookup_verify_bitwise_xor_8_0[2][row]
            };
            qm31 denom0 = verify_bitwise_xor_9->combine(lookup_xor9, 3);
            qm31 denom1 = verify_bitwise_xor_8->combine(lookup_xor8, 3);
            qm31 num = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            WRITE_FRAC(15, num, denom);
        }

        // Pair 16: memory_address_to_id_2 + memory_id_to_big_2
        {
            m31 lookup_addr2id_2[2] = {
                lookup_memory_address_to_id_2[0][row],
                lookup_memory_address_to_id_2[1][row]
            };
            m31 lookup_id2big_2[29];
            for (int i = 0; i < 29; i++) {
                lookup_id2big_2[i] = lookup_memory_id_to_big_2[i][row];
            }
            qm31 denom0 = memory_address_to_id->combine(lookup_addr2id_2, 2);
            qm31 denom1 = memory_id_to_big->combine(lookup_id2big_2, 29);
            qm31 num = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            WRITE_FRAC(16, num, denom);
        }

        // Pair 17: memory_address_to_id_3 + memory_id_to_big_3
        {
            m31 lookup_addr2id_3[2] = {
                lookup_memory_address_to_id_3[0][row],
                lookup_memory_address_to_id_3[1][row]
            };
            m31 lookup_id2big_3[29];
            for (int i = 0; i < 29; i++) {
                lookup_id2big_3[i] = lookup_memory_id_to_big_3[i][row];
            }
            qm31 denom0 = memory_address_to_id->combine(lookup_addr2id_3, 2);
            qm31 denom1 = memory_id_to_big->combine(lookup_id2big_3, 29);
            qm31 num = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            WRITE_FRAC(17, num, denom);
        }

        // Pair 18: memory_address_to_id_4 + memory_id_to_big_4
        {
            m31 lookup_addr2id_4[2] = {
                lookup_memory_address_to_id_4[0][row],
                lookup_memory_address_to_id_4[1][row]
            };
            m31 lookup_id2big_4[29];
            for (int i = 0; i < 29; i++) {
                lookup_id2big_4[i] = lookup_memory_id_to_big_4[i][row];
            }
            qm31 denom0 = memory_address_to_id->combine(lookup_addr2id_4, 2);
            qm31 denom1 = memory_id_to_big->combine(lookup_id2big_4, 29);
            qm31 num = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            WRITE_FRAC(18, num, denom);
        }

        #undef WRITE_FRAC
    }
}

// Host wrapper function for interaction trace
extern "C"
void generate_bitwise_builtin_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *verify_bitwise_xor_9,
    void *verify_bitwise_xor_8,

    // Lookup data arrays - 5 MemoryAddressToId lookups
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_address_to_id_3,
    unsigned **lookup_memory_address_to_id_4,

    // Lookup data arrays - 5 MemoryIdToBig lookups
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_memory_id_to_big_3,
    unsigned **lookup_memory_id_to_big_4,

    // Lookup data arrays - 27 VerifyBitwiseXor9 lookups
    unsigned **lookup_verify_bitwise_xor_9_0,
    unsigned **lookup_verify_bitwise_xor_9_1,
    unsigned **lookup_verify_bitwise_xor_9_2,
    unsigned **lookup_verify_bitwise_xor_9_3,
    unsigned **lookup_verify_bitwise_xor_9_4,
    unsigned **lookup_verify_bitwise_xor_9_5,
    unsigned **lookup_verify_bitwise_xor_9_6,
    unsigned **lookup_verify_bitwise_xor_9_7,
    unsigned **lookup_verify_bitwise_xor_9_8,
    unsigned **lookup_verify_bitwise_xor_9_9,
    unsigned **lookup_verify_bitwise_xor_9_10,
    unsigned **lookup_verify_bitwise_xor_9_11,
    unsigned **lookup_verify_bitwise_xor_9_12,
    unsigned **lookup_verify_bitwise_xor_9_13,
    unsigned **lookup_verify_bitwise_xor_9_14,
    unsigned **lookup_verify_bitwise_xor_9_15,
    unsigned **lookup_verify_bitwise_xor_9_16,
    unsigned **lookup_verify_bitwise_xor_9_17,
    unsigned **lookup_verify_bitwise_xor_9_18,
    unsigned **lookup_verify_bitwise_xor_9_19,
    unsigned **lookup_verify_bitwise_xor_9_20,
    unsigned **lookup_verify_bitwise_xor_9_21,
    unsigned **lookup_verify_bitwise_xor_9_22,
    unsigned **lookup_verify_bitwise_xor_9_23,
    unsigned **lookup_verify_bitwise_xor_9_24,
    unsigned **lookup_verify_bitwise_xor_9_25,
    unsigned **lookup_verify_bitwise_xor_9_26,

    // Lookup data arrays - 1 VerifyBitwiseXor8 lookup
    unsigned **lookup_verify_bitwise_xor_8_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
) {
    unsigned trace_size = 1 << log_size;
    unsigned block_dim_val = BITWISE_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX;
    unsigned num_blocks = (trace_size + block_dim_val - 1) / block_dim_val;

    // Allocate intermediate buffers for logup accumulation
    m31 *device_numerator0, *device_numerator1, *device_numerator2, *device_numerator3;
    qm31 *denom_ptr, *denom_inv;
    device_numerator0 = cuda_malloc<m31>(trace_size);
    device_numerator1 = cuda_malloc<m31>(trace_size);
    device_numerator2 = cuda_malloc<m31>(trace_size);
    device_numerator3 = cuda_malloc<m31>(trace_size);
    denom_ptr = cuda_malloc<qm31>(trace_size);
    denom_inv = cuda_malloc<qm31>(trace_size);

    // Allocate coordinate sums buffer
    m31 *device_coordinate_sums = cuda_malloc<m31>(4);
    ASSERT_CUDA_SUCCESS(cudaMemsetAsync(device_coordinate_sums, 0, 4 * sizeof(m31), 0));

    // Copy lookup element structs to device memory
    MemoryAddressToId *device_mem_addr_to_id = cuda_malloc<MemoryAddressToId>(1);
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(device_mem_addr_to_id, memory_address_to_id, sizeof(MemoryAddressToId), cudaMemcpyHostToDevice, 0));

    MemoryIdToBig *device_mem_id_to_big = cuda_malloc<MemoryIdToBig>(1);
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(device_mem_id_to_big, memory_id_to_big, sizeof(MemoryIdToBig), cudaMemcpyHostToDevice, 0));

    VerifyBitwiseXor_9 *device_xor9 = cuda_malloc<VerifyBitwiseXor_9>(1);
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(device_xor9, verify_bitwise_xor_9, sizeof(VerifyBitwiseXor_9), cudaMemcpyHostToDevice, 0));

    VerifyBitwiseXor_8 *device_xor8 = cuda_malloc<VerifyBitwiseXor_8>(1);
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(device_xor8, verify_bitwise_xor_8, sizeof(VerifyBitwiseXor_8), cudaMemcpyHostToDevice, 0));

    // Copy pointer arrays to device memory
    // Memory address to id lookups (2 elements each)
    m31 **device_lookup_addr2id_0 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_addr2id_1 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_addr2id_2 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_addr2id_3 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_addr2id_4 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_4, 2);

    // Memory id to big lookups (29 elements each)
    m31 **device_lookup_id2big_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_id2big_1 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_id2big_2 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_id2big_3 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_id2big_4 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_4, 29);

    // Verify bitwise xor 9 lookups (3 elements each)
    m31 **device_lookup_xor9_0 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_0, 3);
    m31 **device_lookup_xor9_1 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_1, 3);
    m31 **device_lookup_xor9_2 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_2, 3);
    m31 **device_lookup_xor9_3 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_3, 3);
    m31 **device_lookup_xor9_4 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_4, 3);
    m31 **device_lookup_xor9_5 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_5, 3);
    m31 **device_lookup_xor9_6 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_6, 3);
    m31 **device_lookup_xor9_7 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_7, 3);
    m31 **device_lookup_xor9_8 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_8, 3);
    m31 **device_lookup_xor9_9 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_9, 3);
    m31 **device_lookup_xor9_10 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_10, 3);
    m31 **device_lookup_xor9_11 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_11, 3);
    m31 **device_lookup_xor9_12 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_12, 3);
    m31 **device_lookup_xor9_13 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_13, 3);
    m31 **device_lookup_xor9_14 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_14, 3);
    m31 **device_lookup_xor9_15 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_15, 3);
    m31 **device_lookup_xor9_16 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_16, 3);
    m31 **device_lookup_xor9_17 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_17, 3);
    m31 **device_lookup_xor9_18 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_18, 3);
    m31 **device_lookup_xor9_19 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_19, 3);
    m31 **device_lookup_xor9_20 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_20, 3);
    m31 **device_lookup_xor9_21 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_21, 3);
    m31 **device_lookup_xor9_22 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_22, 3);
    m31 **device_lookup_xor9_23 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_23, 3);
    m31 **device_lookup_xor9_24 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_24, 3);
    m31 **device_lookup_xor9_25 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_25, 3);
    m31 **device_lookup_xor9_26 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_9_26, 3);

    // Verify bitwise xor 8 lookup (3 elements)
    m31 **device_lookup_xor8_0 = clone_to_device<m31*>((m31**)lookup_verify_bitwise_xor_8_0, 3);

    // Interaction trace (19 pairs × 4 m31 columns = 76 m31 columns total)
    // Each pair stores: 4 m31 for accumulated value
    m31 **device_interaction_trace = clone_to_device<m31*>((m31**)interaction_trace, BITWISE_BUILTIN_N_INTERACTION_TRACE_COLUMNS * 4);

    // Lambda for finalizing each column
    auto launch_finalize = [&](unsigned col_index) {
        batch_inverse_secure_field(denom_ptr, denom_inv, trace_size);
        generate_bitwise_builtin_interaction_finalize_col_kernel<<<num_blocks, block_dim_val>>>(
            col_index, trace_size, denom_inv,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3,
            device_interaction_trace
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    };

    // 19 interaction trace columns (19 * 4 = 76 M31 columns)
    // Bitwise builtin layout:
    // Column 0: pair(memory_address_to_id_0, memory_id_to_big_0)
    // Column 1: pair(memory_address_to_id_1, memory_id_to_big_1)
    // Columns 2-14: xor_9 pairs (0,1), (2,3), ..., (24,25)
    // Column 15: pair(xor_9[26], xor_8[0])
    // Column 16: pair(memory_address_to_id_2, memory_id_to_big_2)
    // Column 17: pair(memory_address_to_id_3, memory_id_to_big_3)
    // Column 18: pair(memory_address_to_id_4, memory_id_to_big_4)

    // Column 0: pair(memory_address_to_id_0, memory_id_to_big_0)
    generate_bitwise_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_addr2id_0, device_lookup_id2big_0,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    launch_finalize(0);

    // Column 1: pair(memory_address_to_id_1, memory_id_to_big_1)
    generate_bitwise_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_addr2id_1, device_lookup_id2big_1,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    launch_finalize(1);

    // Columns 2-14: xor_9 pairs
    m31 **xor9_lookups[27] = {
        device_lookup_xor9_0, device_lookup_xor9_1, device_lookup_xor9_2, device_lookup_xor9_3,
        device_lookup_xor9_4, device_lookup_xor9_5, device_lookup_xor9_6, device_lookup_xor9_7,
        device_lookup_xor9_8, device_lookup_xor9_9, device_lookup_xor9_10, device_lookup_xor9_11,
        device_lookup_xor9_12, device_lookup_xor9_13, device_lookup_xor9_14, device_lookup_xor9_15,
        device_lookup_xor9_16, device_lookup_xor9_17, device_lookup_xor9_18, device_lookup_xor9_19,
        device_lookup_xor9_20, device_lookup_xor9_21, device_lookup_xor9_22, device_lookup_xor9_23,
        device_lookup_xor9_24, device_lookup_xor9_25, device_lookup_xor9_26
    };

    for (int p = 0; p < 13; p++) {
        generate_bitwise_builtin_interaction_col_gen_kernel<3, 3><<<num_blocks, block_dim_val>>>(
            device_xor9, device_xor9,
            xor9_lookups[p * 2], xor9_lookups[p * 2 + 1],
            trace_size, denom_ptr,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        launch_finalize(2 + p);
    }

    // Column 15: pair(xor_9[26], xor_8[0])
    generate_bitwise_builtin_interaction_col_gen_kernel<3, 3><<<num_blocks, block_dim_val>>>(
        device_xor9, device_xor8,
        device_lookup_xor9_26, device_lookup_xor8_0,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    launch_finalize(15);

    // Column 16: pair(memory_address_to_id_2, memory_id_to_big_2)
    generate_bitwise_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_addr2id_2, device_lookup_id2big_2,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    launch_finalize(16);

    // Column 17: pair(memory_address_to_id_3, memory_id_to_big_3)
    generate_bitwise_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_addr2id_3, device_lookup_id2big_3,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    launch_finalize(17);

    // Column 18: pair(memory_address_to_id_4, memory_id_to_big_4)
    generate_bitwise_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_addr2id_4, device_lookup_id2big_4,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    launch_finalize(18);

    // Compute cumsum_shift and apply coord_prefix_sum
    unsigned last_index = BITWISE_BUILTIN_N_INTERACTION_TRACE_COLUMNS;
    size_t shared_mem_size = 4 * block_dim_val * sizeof(m31);
    generate_bitwise_builtin_interaction_cumsum_shift_kernel<<<num_blocks, block_dim_val, shared_mem_size>>>(
        last_index, trace_size, device_interaction_trace, device_coordinate_sums
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generate_bitwise_builtin_interaction_coord_prefix_sum_kernel<<<num_blocks, block_dim_val>>>(
        device_coordinate_sums, last_index, trace_size, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Apply inclusive_prefix_sum only to the last 4 columns
    inclusive_prefix_sum(interaction_trace[4 * BITWISE_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * BITWISE_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * BITWISE_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * BITWISE_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    // Copy coordinate sums to claimed_sum output
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(claimed_sum, device_coordinate_sums, 4 * sizeof(m31), cudaMemcpyDeviceToHost, 0));

    // Free intermediate buffers
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_ptr);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_coordinate_sums);

    // Free device memory - lookup element structs
    cuda_free_memory(device_mem_addr_to_id);
    cuda_free_memory(device_mem_id_to_big);
    cuda_free_memory(device_xor9);
    cuda_free_memory(device_xor8);

    // Free device memory - pointer arrays
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_addr2id_1);
    cuda_free_memory(device_lookup_addr2id_2);
    cuda_free_memory(device_lookup_addr2id_3);
    cuda_free_memory(device_lookup_addr2id_4);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_lookup_id2big_1);
    cuda_free_memory(device_lookup_id2big_2);
    cuda_free_memory(device_lookup_id2big_3);
    cuda_free_memory(device_lookup_id2big_4);
    cuda_free_memory(device_lookup_xor9_0);
    cuda_free_memory(device_lookup_xor9_1);
    cuda_free_memory(device_lookup_xor9_2);
    cuda_free_memory(device_lookup_xor9_3);
    cuda_free_memory(device_lookup_xor9_4);
    cuda_free_memory(device_lookup_xor9_5);
    cuda_free_memory(device_lookup_xor9_6);
    cuda_free_memory(device_lookup_xor9_7);
    cuda_free_memory(device_lookup_xor9_8);
    cuda_free_memory(device_lookup_xor9_9);
    cuda_free_memory(device_lookup_xor9_10);
    cuda_free_memory(device_lookup_xor9_11);
    cuda_free_memory(device_lookup_xor9_12);
    cuda_free_memory(device_lookup_xor9_13);
    cuda_free_memory(device_lookup_xor9_14);
    cuda_free_memory(device_lookup_xor9_15);
    cuda_free_memory(device_lookup_xor9_16);
    cuda_free_memory(device_lookup_xor9_17);
    cuda_free_memory(device_lookup_xor9_18);
    cuda_free_memory(device_lookup_xor9_19);
    cuda_free_memory(device_lookup_xor9_20);
    cuda_free_memory(device_lookup_xor9_21);
    cuda_free_memory(device_lookup_xor9_22);
    cuda_free_memory(device_lookup_xor9_23);
    cuda_free_memory(device_lookup_xor9_24);
    cuda_free_memory(device_lookup_xor9_25);
    cuda_free_memory(device_lookup_xor9_26);
    cuda_free_memory(device_lookup_xor8_0);
    cuda_free_memory(device_interaction_trace);
}
