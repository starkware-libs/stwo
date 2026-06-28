/**
 * CUDA implementation for blake_compress_opcode trace generation.
 *
 * This component handles the Blake2s compress opcode with:
 * - 174 trace columns
 * - 20 memory_address_to_id lookups
 * - 20 memory_id_to_big lookups
 * - 17 range_check_7_2_5 lookups
 * - 4 verify_bitwise_xor_8 lookups
 * - 2 blake_round lookups
 * - 8 triple_xor_32 lookups
 * - 1 verify_instruction lookup
 * - 2 opcodes lookups
 * - 37 interaction trace columns
 */

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

#include "gen_blake_compress_opcode_trace.cuh"
#include "gen_blake_round_sigma_trace.cuh"
#include "gen_blake.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

// Constants for Blake compress opcode
#define M31_0 ((m31)0)
#define M31_1 ((m31)1)
#define M31_2 ((m31)2)
#define M31_3 ((m31)3)
#define M31_4 ((m31)4)
#define M31_5 ((m31)5)
#define M31_6 ((m31)6)
#define M31_7 ((m31)7)
#define M31_8 ((m31)8)
#define M31_9 ((m31)9)
#define M31_10 ((m31)10)
#define M31_14 ((m31)14)
#define M31_16 ((m31)16)
#define M31_32 ((m31)32)
#define M31_64 ((m31)64)
#define M31_81 ((m31)81)
#define M31_82 ((m31)82)
#define M31_127 ((m31)127)
#define M31_128 ((m31)128)
#define M31_256 ((m31)256)
#define M31_512 ((m31)512)
#define M31_2048 ((m31)2048)
#define M31_32768 ((m31)32768)
#define M31_262144 ((m31)262144)
#define M31_134217728 ((m31)134217728)

// Blake2s IV constants (correct values)
// IV[0] = 0x6A09E667 = 1779033703
// IV[1] = 0xBB67AE85 = 3144134277
// IV[2] = 0x3C6EF372 = 1013904242
// IV[3] = 0xA54FF53A = 2773480762
// IV[4] = 0x510E527F = 1359893119
// IV[5] = 0x9B05688C = 2600822924
// IV[6] = 0x1F83D9AB = 528734635
// IV[7] = 0x5BE0CD19 = 1541459225

__device__ __constant__ uint32_t BLAKE2S_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__device__ __constant__ uint8_t BLAKE2S_SIGMA[10][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 }
};

// UInt16 constants
#define UINT16_0 0
#define UINT16_1 1
#define UINT16_2 2
#define UINT16_3 3
#define UINT16_4 4
#define UINT16_5 5
#define UINT16_6 6
#define UINT16_7 7
#define UINT16_8 8
#define UINT16_9 9
#define UINT16_11 11
#define UINT16_13 13
#define UINT16_14 14
#define UINT16_31 31
#define UINT16_81 81
#define UINT16_82 82
#define UINT16_127 127

// Helper function: Extract low 16 bits from Blake word
__device__ inline m31 blake_word_low_16_bits(m31 limb0, m31 limb1) {
    uint16_t tmp = ((uint16_t)limb1) >> UINT16_7;
    return (m31)((((limb1 - tmp * 128) * 512) + limb0) % P);
}

// Helper function: Extract high 16 bits from Blake word
__device__ inline m31 blake_word_high_16_bits(m31 limb1, m31 limb2, m31 limb3) {
    uint16_t tmp = ((uint16_t)limb1) >> UINT16_7;
    return (m31)(((limb3 * 2048) + (limb2 * 4) + tmp) % P);
}

// Helper function: Range check 7_2_5 extraction
__device__ inline void extract_range_check_7_2_5(
    m31 low_16, m31 high_16,
    m31 *low_7_ms_bits, m31 *high_14_ms_bits, m31 *high_5_ms_bits,
    m31 *rc_0, m31 *rc_1, m31 *rc_2
) {
    uint16_t low_7 = ((uint16_t)low_16) >> UINT16_9;
    *low_7_ms_bits = (m31)(low_7);

    uint16_t high_14 = ((uint16_t)high_16) >> UINT16_2;
    *high_14_ms_bits = (m31)(high_14);

    uint16_t high_5 = high_14 >> UINT16_9;
    *high_5_ms_bits = (m31)(high_5);

    *rc_0 = *low_7_ms_bits;
    *rc_1 = sub(high_16, mul(*high_14_ms_bits, M31_4));
    *rc_2 = *high_5_ms_bits;
}

// Helper function: Memory ID to big lookup data (29 fields with many zeros)
__device__ inline void fill_memory_id_to_big_lookup(
    m31 *lookup, m31 id, m31 limb0, m31 limb1, m31 limb2, m31 limb3
) {
    lookup[0] = id;
    lookup[1] = limb0;
    lookup[2] = limb1;
    lookup[3] = limb2;
    lookup[4] = limb3;
    for (int i = 5; i < 29; i++) {
        lookup[i] = M31_0;
    }
}

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return rotate_right(x, n);
}

__device__ inline void blake_g(
    uint32_t *v, int a, int b, int c, int d, uint32_t x, uint32_t y
) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7);
}

__device__ inline void blake_round_cuda(uint32_t v[16], const uint32_t m[16], int round) {
    const uint8_t *s = BLAKE2S_SIGMA[round];
    blake_g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
    blake_g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
    blake_g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
    blake_g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
    blake_g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
    blake_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
    blake_g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
    blake_g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
}

// Helper to read a 32-bit word (little-endian) from memory without populating lookups.
__device__ inline uint32_t read_word32(
    m31 address,
    unsigned *memory_address_to_id_table,
    unsigned **memory_id_to_big_table,
    unsigned *memory_id_to_big_small
) {
    m31 memory_id = M31_0;
    memory_address_to_id_deduce_output(memory_address_to_id_table, address, &memory_id);

    m31 value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_table,
        memory_id_to_big_small,
        memory_id,
        value
    );

    uint16_t tmp = ((uint16_t)value[1]) >> UINT16_7;
    // Note: Don't use % P here - the formula naturally produces 16-bit values
    // and % P with uint16_t cast would truncate incorrectly
    uint16_t low = (uint16_t)(((value[1] - tmp * 128) * 512) + value[0]);
    uint16_t high = (uint16_t)((value[3] * 2048) + (value[2] * 4) + tmp);
    return ((uint32_t)high << 16) | low;
}

/**
 * Main kernel for blake_compress_opcode trace generation.
 */
__launch_bounds__(256, 2)
__global__ void generate_blake_compress_opcode_trace_kernel(
    m31 **traces,

    // Lookup data pointers - blake_round (2 × 35 fields)
    m31 **lookup_blake_round_0,
    m31 **lookup_blake_round_1,

    // Lookup data pointers - memory_address_to_id (20 × 2 fields)
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

    // Lookup data pointers - memory_id_to_big (20 × 29 fields)
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

    // Lookup data pointers - opcodes (2 × 3 fields)
    m31 **lookup_opcodes_0,
    m31 **lookup_opcodes_1,

    // Lookup data pointers - range_check_7_2_5 (17 × 3 fields)
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
    m31 **lookup_range_check_7_2_5_16,

    // Lookup data pointers - triple_xor_32 (8 × 8 fields)
    m31 **lookup_triple_xor_32_0,
    m31 **lookup_triple_xor_32_1,
    m31 **lookup_triple_xor_32_2,
    m31 **lookup_triple_xor_32_3,
    m31 **lookup_triple_xor_32_4,
    m31 **lookup_triple_xor_32_5,
    m31 **lookup_triple_xor_32_6,
    m31 **lookup_triple_xor_32_7,

    // Lookup data pointers - verify_bitwise_xor_8 (4 × 3 fields)
    m31 **lookup_verify_bitwise_xor_8_0,
    m31 **lookup_verify_bitwise_xor_8_1,
    m31 **lookup_verify_bitwise_xor_8_2,
    m31 **lookup_verify_bitwise_xor_8_3,

    // Lookup data pointers - verify_instruction (1 × 7 fields)
    m31 **lookup_verify_instruction_0,

    // Sub-component inputs
    m31 **sub_component_inputs_verify_instruction,    // 1 × 7 fields
    m31 **sub_component_inputs_memory_address_to_id,  // 20 × 1 field
    m31 **sub_component_inputs_memory_id_to_big,      // 20 × 1 field
    m31 **sub_component_inputs_range_check_7_2_5,     // 17 × 3 fields
    m31 **sub_component_inputs_verify_bitwise_xor_8,  // 4 × 3 fields
    m31 **sub_component_inputs_blake_round,           // 10 × 19 fields
    m31 **sub_component_inputs_triple_xor_32,         // 8 × 3 fields

    // Opcode inputs (pc, ap, fp)
    m31 **blake_compress_opcode_inputs,

    // Memory lookup tables
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned row_offset,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= trace_size) return;

    // ========== Input columns (col0-col2) ==========
    m31 input_pc_col0 = blake_compress_opcode_inputs[0][row];
    traces[0][row] = input_pc_col0;
    m31 input_ap_col1 = blake_compress_opcode_inputs[1][row];
    traces[1][row] = input_ap_col1;
    m31 input_fp_col2 = blake_compress_opcode_inputs[2][row];
    traces[2][row] = input_fp_col2;

    // ========== Decode Instruction (col3-col10) ==========

    // Memory lookup for instruction decode
    m31 memory_address_to_id_value_0 = {0};
    memory_address_to_id_deduce_output(
        memory_address_to_id_address_to_raw_id,
        input_pc_col0,
        &memory_address_to_id_value_0
    );

    m31 memory_id_to_big_value_0[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        memory_address_to_id_value_0,
        memory_id_to_big_value_0
    );

    // Decode offset0, offset1, offset2
    uint16_t offset0 = (uint16_t)(memory_id_to_big_value_0[0])
        + ((((uint16_t)(memory_id_to_big_value_0[1])) & UINT16_127) << UINT16_9);
    m31 offset0_col3 = (m31)(offset0);
    traces[3][row] = offset0_col3;

    uint16_t tmp1 = ((uint16_t)(memory_id_to_big_value_0[1])) >> UINT16_7;
    uint16_t tmp2 = ((uint16_t)(memory_id_to_big_value_0[2])) << UINT16_2;
    uint16_t tmp3 = ((uint16_t)(memory_id_to_big_value_0[3])) & UINT16_31;
    uint16_t offset1 = tmp1 + tmp2 + (tmp3 << UINT16_11);
    m31 offset1_col4 = (m31)(offset1);
    traces[4][row] = offset1_col4;

    uint16_t tmp4 = ((uint16_t)(memory_id_to_big_value_0[3])) >> UINT16_5;
    uint16_t tmp5 = ((uint16_t)(memory_id_to_big_value_0[4])) << UINT16_4;
    uint16_t tmp6 = ((uint16_t)(memory_id_to_big_value_0[5])) & UINT16_7;
    uint16_t offset2 = tmp4 + tmp5 + (tmp6 << UINT16_13);
    m31 offset2_col5 = (m31)(offset2);
    traces[5][row] = offset2_col5;

    // Decode flags
    uint16_t flags_tmp = (((uint16_t)(memory_id_to_big_value_0[5])) >> UINT16_3)
        + (((uint16_t)(memory_id_to_big_value_0[6])) << UINT16_6);

    uint16_t dst_base_fp = (m31)(flags_tmp >> UINT16_0) & UINT16_1;
    m31 dst_base_fp_col6 = (m31)(dst_base_fp);
    traces[6][row] = dst_base_fp_col6;

    uint16_t op0_base_fp = (m31)(flags_tmp >> UINT16_1) & UINT16_1;
    m31 op0_base_fp_col7 = (m31)(op0_base_fp);
    traces[7][row] = op0_base_fp_col7;

    uint16_t op1_base_fp = (m31)(flags_tmp >> UINT16_3) & UINT16_1;
    m31 op1_base_fp_col8 = (m31)(op1_base_fp);
    traces[8][row] = op1_base_fp_col8;

    uint16_t ap_update_add_1 = (m31)(flags_tmp >> UINT16_11) & UINT16_1;
    m31 ap_update_add_1_col9 = (m31)(ap_update_add_1);
    traces[9][row] = ap_update_add_1_col9;

    m31 opcode_extension_col10 = memory_id_to_big_value_0[7];
    traces[10][row] = opcode_extension_col10;

    // ========== Sub-component inputs: verify_instruction ==========
    sub_component_inputs_verify_instruction[0][row] = input_pc_col0;
    sub_component_inputs_verify_instruction[1][row] = offset0_col3;
    sub_component_inputs_verify_instruction[2][row] = offset1_col4;
    sub_component_inputs_verify_instruction[3][row] = offset2_col5;
    m31 verify_instr_4 = add(add(add(mul(dst_base_fp_col6, M31_8), mul(op0_base_fp_col7, M31_16)),
                                 mul(op1_base_fp_col8, M31_64)),
                             mul(sub(M31_1, op1_base_fp_col8), M31_128));
    sub_component_inputs_verify_instruction[4][row] = verify_instr_4;
    m31 verify_instr_5 = mul(ap_update_add_1_col9, M31_32);
    sub_component_inputs_verify_instruction[5][row] = verify_instr_5;
    sub_component_inputs_verify_instruction[6][row] = opcode_extension_col10;

    // ========== Lookup data: verify_instruction_0 ==========
    lookup_verify_instruction_0[0][row] = input_pc_col0;
    lookup_verify_instruction_0[1][row] = offset0_col3;
    lookup_verify_instruction_0[2][row] = offset1_col4;
    lookup_verify_instruction_0[3][row] = offset2_col5;
    lookup_verify_instruction_0[4][row] = verify_instr_4;
    lookup_verify_instruction_0[5][row] = verify_instr_5;
    lookup_verify_instruction_0[6][row] = opcode_extension_col10;

    // Decoded instruction outputs
    m31 offset0_minus_32768 = sub(offset0_col3, M31_32768);
    m31 offset1_minus_32768 = sub(offset1_col4, M31_32768);
    m31 offset2_minus_32768 = sub(offset2_col5, M31_32768);

    // ========== mem0_base_col11 ==========
    m31 mem0_base_col11 = add(mul(op0_base_fp_col7, input_fp_col2),
                              mul(sub(M31_1, op0_base_fp_col7), input_ap_col1));
    traces[11][row] = mem0_base_col11;

    // ========== Read op0 (Read Positive Num Bits 29) ==========
    m31 op0_addr = add(mem0_base_col11, offset1_minus_32768);
    m31 op0_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, op0_addr, &op0_id);
    m31 op0_id_col12 = op0_id;
    traces[12][row] = op0_id_col12;

    m31 op0_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        op0_id,
        op0_value
    );

    m31 op0_limb_0_col13 = op0_value[0];
    m31 op0_limb_1_col14 = op0_value[1];
    m31 op0_limb_2_col15 = op0_value[2];
    m31 op0_limb_3_col16 = op0_value[3];
    traces[13][row] = op0_limb_0_col13;
    traces[14][row] = op0_limb_1_col14;
    traces[15][row] = op0_limb_2_col15;
    traces[16][row] = op0_limb_3_col16;

    // partial_limb_msb_col17
    uint16_t partial_msb_0 = (((uint16_t)(op0_limb_3_col16)) & UINT16_2) >> UINT16_1;
    m31 partial_limb_msb_col17 = (m31)(partial_msb_0);
    traces[17][row] = partial_limb_msb_col17;

    // Sub-component inputs & lookup data for memory_address_to_id_0
    sub_component_inputs_memory_address_to_id[0][row] = op0_addr;
    lookup_memory_address_to_id_0[0][row] = op0_addr;
    lookup_memory_address_to_id_0[1][row] = op0_id_col12;

    // Lookup data for memory_id_to_big_0 (29 fields)
    lookup_memory_id_to_big_0[0][row] = op0_id_col12;
    lookup_memory_id_to_big_0[1][row] = op0_limb_0_col13;
    lookup_memory_id_to_big_0[2][row] = op0_limb_1_col14;
    lookup_memory_id_to_big_0[3][row] = op0_limb_2_col15;
    lookup_memory_id_to_big_0[4][row] = op0_limb_3_col16;
    for (int i = 5; i < 29; i++) {
        lookup_memory_id_to_big_0[i][row] = M31_0;
    }
    sub_component_inputs_memory_id_to_big[0][row] = op0_id_col12;

    // ========== mem1_base_col18 ==========
    m31 op1_not_base_fp = sub(M31_1, op1_base_fp_col8);
    m31 mem1_base_col18 = add(mul(op1_base_fp_col8, input_fp_col2),
                              mul(op1_not_base_fp, input_ap_col1));
    traces[18][row] = mem1_base_col18;

    // ========== Read op1 (Read Positive Num Bits 29) ==========
    m31 op1_addr = add(mem1_base_col18, offset2_minus_32768);
    m31 op1_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, op1_addr, &op1_id);
    m31 op1_id_col19 = op1_id;
    traces[19][row] = op1_id_col19;

    m31 op1_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        op1_id,
        op1_value
    );

    m31 op1_limb_0_col20 = op1_value[0];
    m31 op1_limb_1_col21 = op1_value[1];
    m31 op1_limb_2_col22 = op1_value[2];
    m31 op1_limb_3_col23 = op1_value[3];
    traces[20][row] = op1_limb_0_col20;
    traces[21][row] = op1_limb_1_col21;
    traces[22][row] = op1_limb_2_col22;
    traces[23][row] = op1_limb_3_col23;

    // partial_limb_msb_col24
    uint16_t partial_msb_1 = (((uint16_t)(op1_limb_3_col23)) & UINT16_2) >> UINT16_1;
    m31 partial_limb_msb_col24 = (m31)(partial_msb_1);
    traces[24][row] = partial_limb_msb_col24;

    // Sub-component inputs & lookup data for memory_address_to_id_1
    sub_component_inputs_memory_address_to_id[1][row] = op1_addr;
    lookup_memory_address_to_id_1[0][row] = op1_addr;
    lookup_memory_address_to_id_1[1][row] = op1_id_col19;

    // Lookup data for memory_id_to_big_1
    lookup_memory_id_to_big_1[0][row] = op1_id_col19;
    lookup_memory_id_to_big_1[1][row] = op1_limb_0_col20;
    lookup_memory_id_to_big_1[2][row] = op1_limb_1_col21;
    lookup_memory_id_to_big_1[3][row] = op1_limb_2_col22;
    lookup_memory_id_to_big_1[4][row] = op1_limb_3_col23;
    for (int i = 5; i < 29; i++) {
        lookup_memory_id_to_big_1[i][row] = M31_0;
    }
    sub_component_inputs_memory_id_to_big[1][row] = op1_id_col19;

    // ========== Read ap value (col25-col30) ==========
    m31 ap_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, input_ap_col1, &ap_id);
    m31 ap_id_col25 = ap_id;
    traces[25][row] = ap_id_col25;

    m31 ap_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        ap_id,
        ap_value
    );

    m31 ap_limb_0_col26 = ap_value[0];
    m31 ap_limb_1_col27 = ap_value[1];
    m31 ap_limb_2_col28 = ap_value[2];
    m31 ap_limb_3_col29 = ap_value[3];
    traces[26][row] = ap_limb_0_col26;
    traces[27][row] = ap_limb_1_col27;
    traces[28][row] = ap_limb_2_col28;
    traces[29][row] = ap_limb_3_col29;

    uint16_t partial_msb_2 = (((uint16_t)(ap_limb_3_col29)) & UINT16_2) >> UINT16_1;
    m31 partial_limb_msb_col30 = (m31)(partial_msb_2);
    traces[30][row] = partial_limb_msb_col30;

    // Sub-component inputs & lookup data for memory_address_to_id_2
    sub_component_inputs_memory_address_to_id[2][row] = input_ap_col1;
    lookup_memory_address_to_id_2[0][row] = input_ap_col1;
    lookup_memory_address_to_id_2[1][row] = ap_id_col25;

    // Lookup data for memory_id_to_big_2
    lookup_memory_id_to_big_2[0][row] = ap_id_col25;
    lookup_memory_id_to_big_2[1][row] = ap_limb_0_col26;
    lookup_memory_id_to_big_2[2][row] = ap_limb_1_col27;
    lookup_memory_id_to_big_2[3][row] = ap_limb_2_col28;
    lookup_memory_id_to_big_2[4][row] = ap_limb_3_col29;
    for (int i = 5; i < 29; i++) {
        lookup_memory_id_to_big_2[i][row] = M31_0;
    }
    sub_component_inputs_memory_id_to_big[2][row] = ap_id_col25;

    // ========== mem_dst_base_col31 ==========
    m31 mem_dst_base_col31 = add(mul(dst_base_fp_col6, input_fp_col2),
                                  mul(sub(M31_1, dst_base_fp_col6), input_ap_col1));
    traces[31][row] = mem_dst_base_col31;

    // Compute decode_blake_opcode output addresses
    m31 state_ptr = add(add(add(op0_limb_0_col13, mul(op0_limb_1_col14, M31_512)),
                           mul(op0_limb_2_col15, M31_262144)),
                       mul(op0_limb_3_col16, M31_134217728));
    m31 message_ptr = add(add(add(op1_limb_0_col20, mul(op1_limb_1_col21, M31_512)),
                             mul(op1_limb_2_col22, M31_262144)),
                         mul(op1_limb_3_col23, M31_134217728));
    m31 output_ptr = add(add(add(ap_limb_0_col26, mul(ap_limb_1_col27, M31_512)),
                            mul(ap_limb_2_col28, M31_262144)),
                        mul(ap_limb_3_col29, M31_134217728));

    // ========== Read Blake Word: dst/counter (col32-col37) ==========
    // decode_instruction_472fe_output_tmp_53f39_9.0[0] is offset0_minus_32768
    m31 dst_addr = add(mem_dst_base_col31, offset0_minus_32768);

    m31 dst_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, dst_addr, &dst_memory_id);

    m31 dst_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        dst_memory_id,
        dst_value
    );

    uint16_t dst_tmp = ((uint16_t)(dst_value[1])) >> UINT16_7;
    m31 low_16_bits_col32 = (m31)((((dst_value[1] - dst_tmp * 128) * 512) + dst_value[0]) % P);
    traces[32][row] = low_16_bits_col32;
    m31 high_16_bits_col33 = (m31)(((dst_value[3] * 2048) + (dst_value[2] * 4) + dst_tmp) % P);
    traces[33][row] = high_16_bits_col33;

    // Range check extraction for dst
    uint16_t low_7_ms_bits_34 = (m31)((uint16_t)(low_16_bits_col32)) >> UINT16_9;
    m31 low_7_ms_bits_col34 = (m31)(low_7_ms_bits_34);
    traces[34][row] = low_7_ms_bits_col34;
    uint16_t high_14_ms_bits_35 = (m31)((uint16_t)(high_16_bits_col33)) >> UINT16_2;
    m31 high_14_ms_bits_col35 = (m31)(high_14_ms_bits_35);
    traces[35][row] = high_14_ms_bits_col35;
    uint16_t high_5_ms_bits_36 = high_14_ms_bits_35 >> UINT16_9;
    m31 high_5_ms_bits_col36 = (m31)(high_5_ms_bits_36);
    traces[36][row] = high_5_ms_bits_col36;

    m31 dst_id_col37 = dst_memory_id;
    traces[37][row] = dst_id_col37;

    // Range check 7_2_5 lookup data for dst (index 0)
    m31 rc_7_2_5_0_1 = sub(high_16_bits_col33, mul(high_14_ms_bits_col35, M31_4));
    lookup_range_check_7_2_5_0[0][row] = low_7_ms_bits_col34;
    lookup_range_check_7_2_5_0[1][row] = rc_7_2_5_0_1;
    lookup_range_check_7_2_5_0[2][row] = high_5_ms_bits_col36;
    sub_component_inputs_range_check_7_2_5[0][row] = low_7_ms_bits_col34;
    sub_component_inputs_range_check_7_2_5[1][row] = rc_7_2_5_0_1;
    sub_component_inputs_range_check_7_2_5[2][row] = high_5_ms_bits_col36;

    // Memory lookups for dst (index 3)
    sub_component_inputs_memory_address_to_id[3][row] = dst_addr;
    lookup_memory_address_to_id_3[0][row] = dst_addr;
    lookup_memory_address_to_id_3[1][row] = dst_id_col37;

    // memory_id_to_big lookup for dst (index 3)
    m31 dst_limb0 = sub(low_16_bits_col32, mul(low_7_ms_bits_col34, M31_512));
    m31 dst_limb1 = add(low_7_ms_bits_col34, mul(rc_7_2_5_0_1, M31_128));
    m31 dst_limb2 = sub(high_14_ms_bits_col35, mul(high_5_ms_bits_col36, M31_512));
    m31 dst_limb3 = high_5_ms_bits_col36;
    lookup_memory_id_to_big_3[0][row] = dst_id_col37;
    lookup_memory_id_to_big_3[1][row] = dst_limb0;
    lookup_memory_id_to_big_3[2][row] = dst_limb1;
    lookup_memory_id_to_big_3[3][row] = dst_limb2;
    lookup_memory_id_to_big_3[4][row] = dst_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_3[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[3][row] = dst_id_col37;

    // ========== Read Blake Word: state_0 (col38-col43) ==========
    m31 state_0_addr = state_ptr;
    m31 state_0_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_0_addr, &state_0_memory_id);

    m31 state_0_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_0_memory_id,
        state_0_value
    );

    uint16_t state_0_tmp = (m31)((uint16_t)(state_0_value[1])) >> UINT16_7;
    m31 low_16_bits_col38 = (m31)((((state_0_value[1] - state_0_tmp * 128) * 512) + state_0_value[0]) % P);
    traces[38][row] = low_16_bits_col38;
    m31 high_16_bits_col39 = (m31)(((state_0_value[3] * 2048) + (state_0_value[2] * 4) + state_0_tmp) % P);
    traces[39][row] = high_16_bits_col39;

    uint16_t low_7_ms_bits_40 = (m31)((uint16_t)(low_16_bits_col38)) >> UINT16_9;
    m31 low_7_ms_bits_col40 = (m31)(low_7_ms_bits_40);
    traces[40][row] = low_7_ms_bits_col40;
    uint16_t high_14_ms_bits_41 = (m31)((uint16_t)(high_16_bits_col39)) >> UINT16_2;
    m31 high_14_ms_bits_col41 = (m31)(high_14_ms_bits_41);
    traces[41][row] = high_14_ms_bits_col41;
    uint16_t high_5_ms_bits_42 = high_14_ms_bits_41 >> UINT16_9;
    m31 high_5_ms_bits_col42 = (m31)(high_5_ms_bits_42);
    traces[42][row] = high_5_ms_bits_col42;
    m31 state_0_id_col43 = state_0_memory_id;
    traces[43][row] = state_0_id_col43;

    m31 rc_7_2_5_1_1 = sub(high_16_bits_col39, mul(high_14_ms_bits_col41, M31_4));
    lookup_range_check_7_2_5_1[0][row] = low_7_ms_bits_col40;
    lookup_range_check_7_2_5_1[1][row] = rc_7_2_5_1_1;
    lookup_range_check_7_2_5_1[2][row] = high_5_ms_bits_col42;
    sub_component_inputs_range_check_7_2_5[3][row] = low_7_ms_bits_col40;
    sub_component_inputs_range_check_7_2_5[4][row] = rc_7_2_5_1_1;
    sub_component_inputs_range_check_7_2_5[5][row] = high_5_ms_bits_col42;

    sub_component_inputs_memory_address_to_id[4][row] = state_0_addr;
    lookup_memory_address_to_id_4[0][row] = state_0_addr;
    lookup_memory_address_to_id_4[1][row] = state_0_id_col43;

    m31 s0_limb0 = sub(low_16_bits_col38, mul(low_7_ms_bits_col40, M31_512));
    m31 s0_limb1 = add(low_7_ms_bits_col40, mul(rc_7_2_5_1_1, M31_128));
    m31 s0_limb2 = sub(high_14_ms_bits_col41, mul(high_5_ms_bits_col42, M31_512));
    m31 s0_limb3 = high_5_ms_bits_col42;
    lookup_memory_id_to_big_4[0][row] = state_0_id_col43;
    lookup_memory_id_to_big_4[1][row] = s0_limb0;
    lookup_memory_id_to_big_4[2][row] = s0_limb1;
    lookup_memory_id_to_big_4[3][row] = s0_limb2;
    lookup_memory_id_to_big_4[4][row] = s0_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_4[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[4][row] = state_0_id_col43;

    // ========== Read Blake Word: state_1 (col44-col49) ==========
    m31 state_1_addr = add(state_ptr, M31_1);
    m31 state_1_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_1_addr, &state_1_memory_id);

    m31 state_1_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_1_memory_id,
        state_1_value
    );

    uint16_t state_1_tmp = (m31)((uint16_t)(state_1_value[1])) >> UINT16_7;
    m31 low_16_bits_col44 = (m31)((((state_1_value[1] - state_1_tmp * 128) * 512) + state_1_value[0]) % P);
    traces[44][row] = low_16_bits_col44;
    m31 high_16_bits_col45 = (m31)(((state_1_value[3] * 2048) + (state_1_value[2] * 4) + state_1_tmp) % P);
    traces[45][row] = high_16_bits_col45;

    uint16_t low_7_ms_bits_46 = (m31)((uint16_t)(low_16_bits_col44)) >> UINT16_9;
    m31 low_7_ms_bits_col46 = (m31)(low_7_ms_bits_46);
    traces[46][row] = low_7_ms_bits_col46;
    uint16_t high_14_ms_bits_47 = (m31)((uint16_t)(high_16_bits_col45)) >> UINT16_2;
    m31 high_14_ms_bits_col47 = (m31)(high_14_ms_bits_47);
    traces[47][row] = high_14_ms_bits_col47;
    uint16_t high_5_ms_bits_48 = high_14_ms_bits_47 >> UINT16_9;
    m31 high_5_ms_bits_col48 = (m31)(high_5_ms_bits_48);
    traces[48][row] = high_5_ms_bits_col48;
    m31 state_1_id_col49 = state_1_memory_id;
    traces[49][row] = state_1_id_col49;

    m31 rc_7_2_5_2_1 = sub(high_16_bits_col45, mul(high_14_ms_bits_col47, M31_4));
    lookup_range_check_7_2_5_2[0][row] = low_7_ms_bits_col46;
    lookup_range_check_7_2_5_2[1][row] = rc_7_2_5_2_1;
    lookup_range_check_7_2_5_2[2][row] = high_5_ms_bits_col48;
    sub_component_inputs_range_check_7_2_5[6][row] = low_7_ms_bits_col46;
    sub_component_inputs_range_check_7_2_5[7][row] = rc_7_2_5_2_1;
    sub_component_inputs_range_check_7_2_5[8][row] = high_5_ms_bits_col48;

    sub_component_inputs_memory_address_to_id[5][row] = state_1_addr;
    lookup_memory_address_to_id_5[0][row] = state_1_addr;
    lookup_memory_address_to_id_5[1][row] = state_1_id_col49;

    m31 s1_limb0 = sub(low_16_bits_col44, mul(low_7_ms_bits_col46, M31_512));
    m31 s1_limb1 = add(low_7_ms_bits_col46, mul(rc_7_2_5_2_1, M31_128));
    m31 s1_limb2 = sub(high_14_ms_bits_col47, mul(high_5_ms_bits_col48, M31_512));
    m31 s1_limb3 = high_5_ms_bits_col48;
    lookup_memory_id_to_big_5[0][row] = state_1_id_col49;
    lookup_memory_id_to_big_5[1][row] = s1_limb0;
    lookup_memory_id_to_big_5[2][row] = s1_limb1;
    lookup_memory_id_to_big_5[3][row] = s1_limb2;
    lookup_memory_id_to_big_5[4][row] = s1_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_5[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[5][row] = state_1_id_col49;

    // ========== Read Blake Word: state_2 (col50-col55) ==========
    m31 state_2_addr = add(state_ptr, M31_2);
    m31 state_2_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_2_addr, &state_2_memory_id);

    m31 state_2_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_2_memory_id,
        state_2_value
    );

    uint16_t state_2_tmp = (m31)((uint16_t)(state_2_value[1])) >> UINT16_7;
    m31 low_16_bits_col50 = (m31)((((state_2_value[1] - state_2_tmp * 128) * 512) + state_2_value[0]) % P);
    traces[50][row] = low_16_bits_col50;
    m31 high_16_bits_col51 = (m31)(((state_2_value[3] * 2048) + (state_2_value[2] * 4) + state_2_tmp) % P);
    traces[51][row] = high_16_bits_col51;

    uint16_t low_7_ms_bits_52 = (m31)((uint16_t)(low_16_bits_col50)) >> UINT16_9;
    m31 low_7_ms_bits_col52 = (m31)(low_7_ms_bits_52);
    traces[52][row] = low_7_ms_bits_col52;
    uint16_t high_14_ms_bits_53 = (m31)((uint16_t)(high_16_bits_col51)) >> UINT16_2;
    m31 high_14_ms_bits_col53 = (m31)(high_14_ms_bits_53);
    traces[53][row] = high_14_ms_bits_col53;
    uint16_t high_5_ms_bits_54 = high_14_ms_bits_53 >> UINT16_9;
    m31 high_5_ms_bits_col54 = (m31)(high_5_ms_bits_54);
    traces[54][row] = high_5_ms_bits_col54;
    m31 state_2_id_col55 = state_2_memory_id;
    traces[55][row] = state_2_id_col55;

    m31 rc_7_2_5_3_1 = sub(high_16_bits_col51, mul(high_14_ms_bits_col53, M31_4));
    lookup_range_check_7_2_5_3[0][row] = low_7_ms_bits_col52;
    lookup_range_check_7_2_5_3[1][row] = rc_7_2_5_3_1;
    lookup_range_check_7_2_5_3[2][row] = high_5_ms_bits_col54;
    sub_component_inputs_range_check_7_2_5[9][row] = low_7_ms_bits_col52;
    sub_component_inputs_range_check_7_2_5[10][row] = rc_7_2_5_3_1;
    sub_component_inputs_range_check_7_2_5[11][row] = high_5_ms_bits_col54;

    sub_component_inputs_memory_address_to_id[6][row] = state_2_addr;
    lookup_memory_address_to_id_6[0][row] = state_2_addr;
    lookup_memory_address_to_id_6[1][row] = state_2_id_col55;

    m31 s2_limb0 = sub(low_16_bits_col50, mul(low_7_ms_bits_col52, M31_512));
    m31 s2_limb1 = add(low_7_ms_bits_col52, mul(rc_7_2_5_3_1, M31_128));
    m31 s2_limb2 = sub(high_14_ms_bits_col53, mul(high_5_ms_bits_col54, M31_512));
    m31 s2_limb3 = high_5_ms_bits_col54;
    lookup_memory_id_to_big_6[0][row] = state_2_id_col55;
    lookup_memory_id_to_big_6[1][row] = s2_limb0;
    lookup_memory_id_to_big_6[2][row] = s2_limb1;
    lookup_memory_id_to_big_6[3][row] = s2_limb2;
    lookup_memory_id_to_big_6[4][row] = s2_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_6[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[6][row] = state_2_id_col55;

    // ========== Read Blake Word: state_3 (col56-col61) ==========
    m31 state_3_addr = add(state_ptr, M31_3);
    m31 state_3_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_3_addr, &state_3_memory_id);

    m31 state_3_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_3_memory_id,
        state_3_value
    );

    uint16_t state_3_tmp = (m31)((uint16_t)(state_3_value[1])) >> UINT16_7;
    m31 low_16_bits_col56 = (m31)((((state_3_value[1] - state_3_tmp * 128) * 512) + state_3_value[0]) % P);
    traces[56][row] = low_16_bits_col56;
    m31 high_16_bits_col57 = (m31)(((state_3_value[3] * 2048) + (state_3_value[2] * 4) + state_3_tmp) % P);
    traces[57][row] = high_16_bits_col57;

    uint16_t low_7_ms_bits_58 = (m31)((uint16_t)(low_16_bits_col56)) >> UINT16_9;
    m31 low_7_ms_bits_col58 = (m31)(low_7_ms_bits_58);
    traces[58][row] = low_7_ms_bits_col58;
    uint16_t high_14_ms_bits_59 = (m31)((uint16_t)(high_16_bits_col57)) >> UINT16_2;
    m31 high_14_ms_bits_col59 = (m31)(high_14_ms_bits_59);
    traces[59][row] = high_14_ms_bits_col59;
    uint16_t high_5_ms_bits_60 = high_14_ms_bits_59 >> UINT16_9;
    m31 high_5_ms_bits_col60 = (m31)(high_5_ms_bits_60);
    traces[60][row] = high_5_ms_bits_col60;
    m31 state_3_id_col61 = state_3_memory_id;
    traces[61][row] = state_3_id_col61;

    m31 rc_7_2_5_4_1 = sub(high_16_bits_col57, mul(high_14_ms_bits_col59, M31_4));
    lookup_range_check_7_2_5_4[0][row] = low_7_ms_bits_col58;
    lookup_range_check_7_2_5_4[1][row] = rc_7_2_5_4_1;
    lookup_range_check_7_2_5_4[2][row] = high_5_ms_bits_col60;
    sub_component_inputs_range_check_7_2_5[12][row] = low_7_ms_bits_col58;
    sub_component_inputs_range_check_7_2_5[13][row] = rc_7_2_5_4_1;
    sub_component_inputs_range_check_7_2_5[14][row] = high_5_ms_bits_col60;

    sub_component_inputs_memory_address_to_id[7][row] = state_3_addr;
    lookup_memory_address_to_id_7[0][row] = state_3_addr;
    lookup_memory_address_to_id_7[1][row] = state_3_id_col61;

    m31 s3_limb0 = sub(low_16_bits_col56, mul(low_7_ms_bits_col58, M31_512));
    m31 s3_limb1 = add(low_7_ms_bits_col58, mul(rc_7_2_5_4_1, M31_128));
    m31 s3_limb2 = sub(high_14_ms_bits_col59, mul(high_5_ms_bits_col60, M31_512));
    m31 s3_limb3 = high_5_ms_bits_col60;
    lookup_memory_id_to_big_7[0][row] = state_3_id_col61;
    lookup_memory_id_to_big_7[1][row] = s3_limb0;
    lookup_memory_id_to_big_7[2][row] = s3_limb1;
    lookup_memory_id_to_big_7[3][row] = s3_limb2;
    lookup_memory_id_to_big_7[4][row] = s3_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_7[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[7][row] = state_3_id_col61;

    // ========== Read Blake Word: state_4 (col62-col67) ==========
    m31 state_4_addr = add(state_ptr, M31_4);
    m31 state_4_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_4_addr, &state_4_memory_id);

    m31 state_4_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_4_memory_id,
        state_4_value
    );

    uint16_t state_4_tmp = (m31)((uint16_t)(state_4_value[1])) >> UINT16_7;
    m31 low_16_bits_col62 = (m31)((((state_4_value[1] - state_4_tmp * 128) * 512) + state_4_value[0]) % P);
    traces[62][row] = low_16_bits_col62;
    m31 high_16_bits_col63 = (m31)(((state_4_value[3] * 2048) + (state_4_value[2] * 4) + state_4_tmp) % P);
    traces[63][row] = high_16_bits_col63;

    uint16_t low_7_ms_bits_64 = (m31)((uint16_t)(low_16_bits_col62)) >> UINT16_9;
    m31 low_7_ms_bits_col64 = (m31)(low_7_ms_bits_64);
    traces[64][row] = low_7_ms_bits_col64;
    uint16_t high_14_ms_bits_65 = (m31)((uint16_t)(high_16_bits_col63)) >> UINT16_2;
    m31 high_14_ms_bits_col65 = (m31)(high_14_ms_bits_65);
    traces[65][row] = high_14_ms_bits_col65;
    uint16_t high_5_ms_bits_66 = high_14_ms_bits_65 >> UINT16_9;
    m31 high_5_ms_bits_col66 = (m31)(high_5_ms_bits_66);
    traces[66][row] = high_5_ms_bits_col66;
    m31 state_4_id_col67 = state_4_memory_id;
    traces[67][row] = state_4_id_col67;

    m31 rc_7_2_5_5_1 = sub(high_16_bits_col63, mul(high_14_ms_bits_col65, M31_4));
    lookup_range_check_7_2_5_5[0][row] = low_7_ms_bits_col64;
    lookup_range_check_7_2_5_5[1][row] = rc_7_2_5_5_1;
    lookup_range_check_7_2_5_5[2][row] = high_5_ms_bits_col66;
    sub_component_inputs_range_check_7_2_5[15][row] = low_7_ms_bits_col64;
    sub_component_inputs_range_check_7_2_5[16][row] = rc_7_2_5_5_1;
    sub_component_inputs_range_check_7_2_5[17][row] = high_5_ms_bits_col66;

    sub_component_inputs_memory_address_to_id[8][row] = state_4_addr;
    lookup_memory_address_to_id_8[0][row] = state_4_addr;
    lookup_memory_address_to_id_8[1][row] = state_4_id_col67;

    m31 s4_limb0 = sub(low_16_bits_col62, mul(low_7_ms_bits_col64, M31_512));
    m31 s4_limb1 = add(low_7_ms_bits_col64, mul(rc_7_2_5_5_1, M31_128));
    m31 s4_limb2 = sub(high_14_ms_bits_col65, mul(high_5_ms_bits_col66, M31_512));
    m31 s4_limb3 = high_5_ms_bits_col66;
    lookup_memory_id_to_big_8[0][row] = state_4_id_col67;
    lookup_memory_id_to_big_8[1][row] = s4_limb0;
    lookup_memory_id_to_big_8[2][row] = s4_limb1;
    lookup_memory_id_to_big_8[3][row] = s4_limb2;
    lookup_memory_id_to_big_8[4][row] = s4_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_8[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[8][row] = state_4_id_col67;

    // ========== Read Blake Word: state_5 (col68-col73) ==========
    m31 state_5_addr = add(state_ptr, M31_5);
    m31 state_5_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_5_addr, &state_5_memory_id);

    m31 state_5_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_5_memory_id,
        state_5_value
    );

    uint16_t state_5_tmp = (m31)((uint16_t)(state_5_value[1])) >> UINT16_7;
    m31 low_16_bits_col68 = (m31)((((state_5_value[1] - state_5_tmp * 128) * 512) + state_5_value[0]) % P);
    traces[68][row] = low_16_bits_col68;
    m31 high_16_bits_col69 = (m31)(((state_5_value[3] * 2048) + (state_5_value[2] * 4) + state_5_tmp) % P);
    traces[69][row] = high_16_bits_col69;

    uint16_t low_7_ms_bits_70 = (m31)((uint16_t)(low_16_bits_col68)) >> UINT16_9;
    m31 low_7_ms_bits_col70 = (m31)(low_7_ms_bits_70);
    traces[70][row] = low_7_ms_bits_col70;
    uint16_t high_14_ms_bits_71 = (m31)((uint16_t)(high_16_bits_col69)) >> UINT16_2;
    m31 high_14_ms_bits_col71 = (m31)(high_14_ms_bits_71);
    traces[71][row] = high_14_ms_bits_col71;
    uint16_t high_5_ms_bits_72 = high_14_ms_bits_71 >> UINT16_9;
    m31 high_5_ms_bits_col72 = (m31)(high_5_ms_bits_72);
    traces[72][row] = high_5_ms_bits_col72;
    m31 state_5_id_col73 = state_5_memory_id;
    traces[73][row] = state_5_id_col73;

    m31 rc_7_2_5_6_1 = sub(high_16_bits_col69, mul(high_14_ms_bits_col71, M31_4));
    lookup_range_check_7_2_5_6[0][row] = low_7_ms_bits_col70;
    lookup_range_check_7_2_5_6[1][row] = rc_7_2_5_6_1;
    lookup_range_check_7_2_5_6[2][row] = high_5_ms_bits_col72;
    sub_component_inputs_range_check_7_2_5[18][row] = low_7_ms_bits_col70;
    sub_component_inputs_range_check_7_2_5[19][row] = rc_7_2_5_6_1;
    sub_component_inputs_range_check_7_2_5[20][row] = high_5_ms_bits_col72;

    sub_component_inputs_memory_address_to_id[9][row] = state_5_addr;
    lookup_memory_address_to_id_9[0][row] = state_5_addr;
    lookup_memory_address_to_id_9[1][row] = state_5_id_col73;

    m31 s5_limb0 = sub(low_16_bits_col68, mul(low_7_ms_bits_col70, M31_512));
    m31 s5_limb1 = add(low_7_ms_bits_col70, mul(rc_7_2_5_6_1, M31_128));
    m31 s5_limb2 = sub(high_14_ms_bits_col71, mul(high_5_ms_bits_col72, M31_512));
    m31 s5_limb3 = high_5_ms_bits_col72;
    lookup_memory_id_to_big_9[0][row] = state_5_id_col73;
    lookup_memory_id_to_big_9[1][row] = s5_limb0;
    lookup_memory_id_to_big_9[2][row] = s5_limb1;
    lookup_memory_id_to_big_9[3][row] = s5_limb2;
    lookup_memory_id_to_big_9[4][row] = s5_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_9[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[9][row] = state_5_id_col73;

    // ========== Read Blake Word: state_6 (col74-col79) ==========
    m31 state_6_addr = add(state_ptr, M31_6);
    m31 state_6_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_6_addr, &state_6_memory_id);

    m31 state_6_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_6_memory_id,
        state_6_value
    );

    uint16_t state_6_tmp = (m31)((uint16_t)(state_6_value[1])) >> UINT16_7;
    m31 low_16_bits_col74 = (m31)((((state_6_value[1] - state_6_tmp * 128) * 512) + state_6_value[0]) % P);
    traces[74][row] = low_16_bits_col74;
    m31 high_16_bits_col75 = (m31)(((state_6_value[3] * 2048) + (state_6_value[2] * 4) + state_6_tmp) % P);
    traces[75][row] = high_16_bits_col75;

    uint16_t low_7_ms_bits_76 = (m31)((uint16_t)(low_16_bits_col74)) >> UINT16_9;
    m31 low_7_ms_bits_col76 = (m31)(low_7_ms_bits_76);
    traces[76][row] = low_7_ms_bits_col76;
    uint16_t high_14_ms_bits_77 = (m31)((uint16_t)(high_16_bits_col75)) >> UINT16_2;
    m31 high_14_ms_bits_col77 = (m31)(high_14_ms_bits_77);
    traces[77][row] = high_14_ms_bits_col77;
    uint16_t high_5_ms_bits_78 = high_14_ms_bits_77 >> UINT16_9;
    m31 high_5_ms_bits_col78 = (m31)(high_5_ms_bits_78);
    traces[78][row] = high_5_ms_bits_col78;
    m31 state_6_id_col79 = state_6_memory_id;
    traces[79][row] = state_6_id_col79;

    m31 rc_7_2_5_7_1 = sub(high_16_bits_col75, mul(high_14_ms_bits_col77, M31_4));
    lookup_range_check_7_2_5_7[0][row] = low_7_ms_bits_col76;
    lookup_range_check_7_2_5_7[1][row] = rc_7_2_5_7_1;
    lookup_range_check_7_2_5_7[2][row] = high_5_ms_bits_col78;
    sub_component_inputs_range_check_7_2_5[21][row] = low_7_ms_bits_col76;
    sub_component_inputs_range_check_7_2_5[22][row] = rc_7_2_5_7_1;
    sub_component_inputs_range_check_7_2_5[23][row] = high_5_ms_bits_col78;

    sub_component_inputs_memory_address_to_id[10][row] = state_6_addr;
    lookup_memory_address_to_id_10[0][row] = state_6_addr;
    lookup_memory_address_to_id_10[1][row] = state_6_id_col79;

    m31 s6_limb0 = sub(low_16_bits_col74, mul(low_7_ms_bits_col76, M31_512));
    m31 s6_limb1 = add(low_7_ms_bits_col76, mul(rc_7_2_5_7_1, M31_128));
    m31 s6_limb2 = sub(high_14_ms_bits_col77, mul(high_5_ms_bits_col78, M31_512));
    m31 s6_limb3 = high_5_ms_bits_col78;
    lookup_memory_id_to_big_10[0][row] = state_6_id_col79;
    lookup_memory_id_to_big_10[1][row] = s6_limb0;
    lookup_memory_id_to_big_10[2][row] = s6_limb1;
    lookup_memory_id_to_big_10[3][row] = s6_limb2;
    lookup_memory_id_to_big_10[4][row] = s6_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_10[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[10][row] = state_6_id_col79;

    // ========== Read Blake Word: state_7 (col80-col85) ==========
    m31 state_7_addr = add(state_ptr, M31_7);
    m31 state_7_memory_id = {0};
    memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, state_7_addr, &state_7_memory_id);

    m31 state_7_value[N_M31_IN_FELT252] = {0};
    memory_id_to_big_state_deduce_output(
        memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_value_ptr,
        state_7_memory_id,
        state_7_value
    );

    uint16_t state_7_tmp = (m31)((uint16_t)(state_7_value[1])) >> UINT16_7;
    m31 low_16_bits_col80 = (m31)((((state_7_value[1] - state_7_tmp * 128) * 512) + state_7_value[0]) % P);
    traces[80][row] = low_16_bits_col80;
    m31 high_16_bits_col81 = (m31)(((state_7_value[3] * 2048) + (state_7_value[2] * 4) + state_7_tmp) % P);
    traces[81][row] = high_16_bits_col81;

    uint16_t low_7_ms_bits_82 = (m31)((uint16_t)(low_16_bits_col80)) >> UINT16_9;
    m31 low_7_ms_bits_col82 = (m31)(low_7_ms_bits_82);
    traces[82][row] = low_7_ms_bits_col82;
    uint16_t high_14_ms_bits_83 = (m31)((uint16_t)(high_16_bits_col81)) >> UINT16_2;
    m31 high_14_ms_bits_col83 = (m31)(high_14_ms_bits_83);
    traces[83][row] = high_14_ms_bits_col83;
    uint16_t high_5_ms_bits_84 = high_14_ms_bits_83 >> UINT16_9;
    m31 high_5_ms_bits_col84 = (m31)(high_5_ms_bits_84);
    traces[84][row] = high_5_ms_bits_col84;
    m31 state_7_id_col85 = state_7_memory_id;
    traces[85][row] = state_7_id_col85;

    m31 rc_7_2_5_8_1 = sub(high_16_bits_col81, mul(high_14_ms_bits_col83, M31_4));
    lookup_range_check_7_2_5_8[0][row] = low_7_ms_bits_col82;
    lookup_range_check_7_2_5_8[1][row] = rc_7_2_5_8_1;
    lookup_range_check_7_2_5_8[2][row] = high_5_ms_bits_col84;
    sub_component_inputs_range_check_7_2_5[24][row] = low_7_ms_bits_col82;
    sub_component_inputs_range_check_7_2_5[25][row] = rc_7_2_5_8_1;
    sub_component_inputs_range_check_7_2_5[26][row] = high_5_ms_bits_col84;

    sub_component_inputs_memory_address_to_id[11][row] = state_7_addr;
    lookup_memory_address_to_id_11[0][row] = state_7_addr;
    lookup_memory_address_to_id_11[1][row] = state_7_id_col85;

    m31 s7_limb0 = sub(low_16_bits_col80, mul(low_7_ms_bits_col82, M31_512));
    m31 s7_limb1 = add(low_7_ms_bits_col82, mul(rc_7_2_5_8_1, M31_128));
    m31 s7_limb2 = sub(high_14_ms_bits_col83, mul(high_5_ms_bits_col84, M31_512));
    m31 s7_limb3 = high_5_ms_bits_col84;
    lookup_memory_id_to_big_11[0][row] = state_7_id_col85;
    lookup_memory_id_to_big_11[1][row] = s7_limb0;
    lookup_memory_id_to_big_11[2][row] = s7_limb1;
    lookup_memory_id_to_big_11[3][row] = s7_limb2;
    lookup_memory_id_to_big_11[4][row] = s7_limb3;
    for (int i = 5; i < 29; i++) lookup_memory_id_to_big_11[i][row] = M31_0;
    sub_component_inputs_memory_id_to_big[11][row] = state_7_id_col85;

    // ========== Split 16 and Bitwise XOR operations (col86-col91) ==========
    // Split 16 Low Part Size 8 on dst/counter low word
    uint16_t ms_8_bits_86 = (m31)((uint16_t)(low_16_bits_col32)) >> UINT16_8;
    m31 ms_8_bits_col86 = (m31)(ms_8_bits_86);
    traces[86][row] = ms_8_bits_col86;

    // Split 16 Low Part Size 8 on dst/counter high word
    uint16_t ms_8_bits_87 = (m31)((uint16_t)(high_16_bits_col33)) >> UINT16_8;
    m31 ms_8_bits_col87 = (m31)(ms_8_bits_87);
    traces[87][row] = ms_8_bits_col87;

    // Bitwise XOR operations for IV[4] computation
    m31 split_low_0 = sub(low_16_bits_col32, mul(ms_8_bits_col86, M31_256));
    m31 split_low_1 = sub(high_16_bits_col33, mul(ms_8_bits_col87, M31_256));

    // XOR with 127 (0x7F), 82 (0x52), 14 (0x0E), 81 (0x51)
    uint16_t xor_88 = (m31)((uint16_t)(split_low_0)) ^ UINT16_127;
    m31 xor_col88 = (m31)(xor_88);
    traces[88][row] = xor_col88;

    uint16_t xor_89 = ms_8_bits_86 ^ UINT16_82;
    m31 xor_col89 = (m31)(xor_89);
    traces[89][row] = xor_col89;

    uint16_t xor_90 = (m31)((uint16_t)(split_low_1)) ^ UINT16_14;
    m31 xor_col90 = (m31)(xor_90);
    traces[90][row] = xor_col90;

    uint16_t xor_91 = ms_8_bits_87 ^ UINT16_81;
    m31 xor_col91 = (m31)(xor_91);
    traces[91][row] = xor_col91;

    // Verify bitwise XOR lookups
    lookup_verify_bitwise_xor_8_0[0][row] = split_low_0;
    lookup_verify_bitwise_xor_8_0[1][row] = M31_127;
    lookup_verify_bitwise_xor_8_0[2][row] = xor_col88;
    sub_component_inputs_verify_bitwise_xor_8[0][row] = split_low_0;
    sub_component_inputs_verify_bitwise_xor_8[1][row] = M31_127;
    sub_component_inputs_verify_bitwise_xor_8[2][row] = xor_col88;

    lookup_verify_bitwise_xor_8_1[0][row] = ms_8_bits_col86;
    lookup_verify_bitwise_xor_8_1[1][row] = M31_82;
    lookup_verify_bitwise_xor_8_1[2][row] = xor_col89;
    sub_component_inputs_verify_bitwise_xor_8[3][row] = ms_8_bits_col86;
    sub_component_inputs_verify_bitwise_xor_8[4][row] = M31_82;
    sub_component_inputs_verify_bitwise_xor_8[5][row] = xor_col89;

    lookup_verify_bitwise_xor_8_2[0][row] = split_low_1;
    lookup_verify_bitwise_xor_8_2[1][row] = M31_14;
    lookup_verify_bitwise_xor_8_2[2][row] = xor_col90;
    sub_component_inputs_verify_bitwise_xor_8[6][row] = split_low_1;
    sub_component_inputs_verify_bitwise_xor_8[7][row] = M31_14;
    sub_component_inputs_verify_bitwise_xor_8[8][row] = xor_col90;

    lookup_verify_bitwise_xor_8_3[0][row] = ms_8_bits_col87;
    lookup_verify_bitwise_xor_8_3[1][row] = M31_81;
    lookup_verify_bitwise_xor_8_3[2][row] = xor_col91;
    sub_component_inputs_verify_bitwise_xor_8[9][row] = ms_8_bits_col87;
    sub_component_inputs_verify_bitwise_xor_8[10][row] = M31_81;
    sub_component_inputs_verify_bitwise_xor_8[11][row] = xor_col91;

    // ========== Blake2s rounds (col92-col124) ==========
    m31 seq = (m31)row;

    uint32_t h_words[8] = {
        ((uint32_t)high_16_bits_col39 << 16) | (uint32_t)low_16_bits_col38,
        ((uint32_t)high_16_bits_col45 << 16) | (uint32_t)low_16_bits_col44,
        ((uint32_t)high_16_bits_col51 << 16) | (uint32_t)low_16_bits_col50,
        ((uint32_t)high_16_bits_col57 << 16) | (uint32_t)low_16_bits_col56,
        ((uint32_t)high_16_bits_col63 << 16) | (uint32_t)low_16_bits_col62,
        ((uint32_t)high_16_bits_col69 << 16) | (uint32_t)low_16_bits_col68,
        ((uint32_t)high_16_bits_col75 << 16) | (uint32_t)low_16_bits_col74,
        ((uint32_t)high_16_bits_col81 << 16) | (uint32_t)low_16_bits_col80
    };


    // state[12]: Computed from xor columns (NOT IV[4]^t0)
    // In Rust: ((xor_col88) + ((xor_col89) * 256)), ((xor_col90) + ((xor_col91) * 256))
    uint32_t state_12_low = ((uint32_t)(uint16_t)xor_col88) + (((uint32_t)(uint16_t)xor_col89) << 8);
    uint32_t state_12_high = ((uint32_t)(uint16_t)xor_col90) + (((uint32_t)(uint16_t)xor_col91) << 8);
    uint32_t state_12 = state_12_low | (state_12_high << 16);

    // state[13]: Just IV[5] = 2600822924 (NOT IV[5]^t1)
    uint32_t state_13 = BLAKE2S_IV[5];  // 2600822924

    // state[14]: Computed based on opcode_flag (NOT IV[6]^f0)
    // In Rust: flag = opcode_extension_col10 - 1
    // flag * 9812 + (1 - flag) * 55723 for low, flag * 57468 + (1 - flag) * 8067 for high
    // opcode_extension_col10 = 1 means flag = 0 (not last block)
    // opcode_extension_col10 = 2 means flag = 1 (last block)
    uint32_t opcode_flag = (opcode_extension_col10 > M31_1) ? (opcode_extension_col10 - M31_1) : 0u;
    uint32_t state_14_low = opcode_flag ? 9812u : 55723u;
    uint32_t state_14_high = opcode_flag ? 57468u : 8067u;
    uint32_t state_14 = state_14_low | (state_14_high << 16);

    // state[15]: Cairo uses IV[7] = 0x5BE0CD19 = 1541459225 (not standard Blake2s v[15] which would be IV[7]^f1)
    uint32_t state_15 = BLAKE2S_IV[7];  // IV[7] = 0x5BE0CD19 = 1541459225

    uint32_t v_state[16] = {
        h_words[0], h_words[1], h_words[2], h_words[3],
        h_words[4], h_words[5], h_words[6], h_words[7],
        BLAKE2S_IV[0], BLAKE2S_IV[1], BLAKE2S_IV[2], BLAKE2S_IV[3],
        state_12,   // Counter value (computed from xor columns)
        state_13,   // IV[5] directly
        state_14,   // Flag value (computed based on opcode_flag)
        state_15    // IV[4] directly
    };
    // Preload message block
    uint32_t message_block[16];
    for (int i = 0; i < 16; i++) {
        message_block[i] = read_word32(
            add(message_ptr, (m31)i),
            memory_address_to_id_address_to_raw_id,
            memory_id_to_big_transpose_big_value_ptr,
            memory_id_to_big_small_value_ptr
        );
    }

    // Lookup data for blake_round_0 (initial state)
    // Structure matches the constraint code in blake_compress_opcode.rs:419-458
    // [0]: seq
    // [1]: M31_0 (round = 0)
    // [2-17]: h_words[0-7] low/high (from trace columns 38,39,44,45,50,51,56,57,62,63,68,69,74,75,80,81)
    // [18-25]: IV[0-3] low/high (constants)
    // [26-27]: state_12 low/high (computed)
    // [28-29]: IV[5] low/high (constant)
    // [30-31]: state_14 low/high (computed)
    // [32-33]: IV[7] low/high (constant)
    // [34]: message_ptr
    lookup_blake_round_0[0][row] = seq;
    lookup_blake_round_0[1][row] = M31_0;
    // h_words[0-7] from trace columns
    lookup_blake_round_0[2][row] = traces[38][row];   // h_words[0] low
    lookup_blake_round_0[3][row] = traces[39][row];   // h_words[0] high
    lookup_blake_round_0[4][row] = traces[44][row];   // h_words[1] low
    lookup_blake_round_0[5][row] = traces[45][row];   // h_words[1] high
    lookup_blake_round_0[6][row] = traces[50][row];   // h_words[2] low
    lookup_blake_round_0[7][row] = traces[51][row];   // h_words[2] high
    lookup_blake_round_0[8][row] = traces[56][row];   // h_words[3] low
    lookup_blake_round_0[9][row] = traces[57][row];   // h_words[3] high
    lookup_blake_round_0[10][row] = traces[62][row];  // h_words[4] low
    lookup_blake_round_0[11][row] = traces[63][row];  // h_words[4] high
    lookup_blake_round_0[12][row] = traces[68][row];  // h_words[5] low
    lookup_blake_round_0[13][row] = traces[69][row];  // h_words[5] high
    lookup_blake_round_0[14][row] = traces[74][row];  // h_words[6] low
    lookup_blake_round_0[15][row] = traces[75][row];  // h_words[6] high
    lookup_blake_round_0[16][row] = traces[80][row];  // h_words[7] low
    lookup_blake_round_0[17][row] = traces[81][row];  // h_words[7] high
    // IV[0-3] constants (M31_58983, M31_27145, M31_44677, M31_47975, M31_62322, M31_15470, M31_62778, M31_42319)
    lookup_blake_round_0[18][row] = 58983;  // IV[0] low = 0x6A09E667 & 0xFFFF = 0xE667 = 58983
    lookup_blake_round_0[19][row] = 27145;  // IV[0] high = 0x6A09E667 >> 16 = 0x6A09 = 27145
    lookup_blake_round_0[20][row] = 44677;  // IV[1] low = 0xBB67AE85 & 0xFFFF = 0xAE85 = 44677
    lookup_blake_round_0[21][row] = 47975;  // IV[1] high = 0xBB67AE85 >> 16 = 0xBB67 = 47975
    lookup_blake_round_0[22][row] = 62322;  // IV[2] low = 0x3C6EF372 & 0xFFFF = 0xF372 = 62322
    lookup_blake_round_0[23][row] = 15470;  // IV[2] high = 0x3C6EF372 >> 16 = 0x3C6E = 15470
    lookup_blake_round_0[24][row] = 62778;  // IV[3] low = 0xA54FF53A & 0xFFFF = 0xF53A = 62778
    lookup_blake_round_0[25][row] = 42319;  // IV[3] high = 0xA54FF53A >> 16 = 0xA54F = 42319
    // state_12 low/high (computed from xor values)
    lookup_blake_round_0[26][row] = state_12_low;
    lookup_blake_round_0[27][row] = state_12_high;
    // IV[5] constant (M31_26764, M31_39685)
    lookup_blake_round_0[28][row] = 26764;  // IV[5] low = 0x9B05688C & 0xFFFF = 0x688C = 26764
    lookup_blake_round_0[29][row] = 39685;  // IV[5] high = 0x9B05688C >> 16 = 0x9B05 = 39685
    // state_14 low/high (computed from opcode_flag)
    lookup_blake_round_0[30][row] = state_14_low;
    lookup_blake_round_0[31][row] = state_14_high;
    // IV[7] constant (M31_52505, M31_23520)
    lookup_blake_round_0[32][row] = 52505;  // IV[7] low = 0x5BE0CD19 & 0xFFFF = 0xCD19 = 52505
    lookup_blake_round_0[33][row] = 23520;  // IV[7] high = 0x5BE0CD19 >> 16 = 0x5BE0 = 23520
    // message_ptr
    lookup_blake_round_0[34][row] = message_ptr;


    // Run 10 rounds, capturing sub-component inputs
    uint32_t v_round[16];
    for (int i = 0; i < 16; i++) v_round[i] = v_state[i];

    for (int round_idx = 0; round_idx < 10; round_idx++) {

        int base = round_idx * 19;
        sub_component_inputs_blake_round[base + 0][row] = seq;
        sub_component_inputs_blake_round[base + 1][row] = (m31)round_idx;
        for (int i = 0; i < 16; i++) {
            sub_component_inputs_blake_round[base + 2 + i][row] = (m31)v_round[i];
        }
        sub_component_inputs_blake_round[base + 18][row] = message_ptr;

        blake_round_cuda(v_round, message_block, round_idx);

    }


    // Final state trace columns (col92-col124)
    for (int i = 0; i < 16; i++) {
        m31 low = (m31)(v_round[i] & 0xFFFF);
        m31 high = (m31)(v_round[i] >> 16);
        traces[92 + i * 2][row] = low;
        traces[92 + i * 2 + 1][row] = high;
    }

    // Extra debug: verify what was written for row 0
    traces[124][row] = message_ptr;

    // Lookup data for blake_round_1 (final state)
    // Structure matches the constraint code in blake_compress_opcode.rs:461-500
    // [0]: seq
    // [1]: M31_10 (round = 10)
    // [2-33]: v_round[0-15] low/high from trace columns 92-123
    // [34]: message_ptr (trace column 124)
    lookup_blake_round_1[0][row] = seq;
    lookup_blake_round_1[1][row] = M31_10;
    // v_round[0-15] from trace columns 92-123
    for (int i = 0; i < 16; i++) {
        lookup_blake_round_1[2 + i * 2][row] = traces[92 + i * 2][row];
        lookup_blake_round_1[3 + i * 2][row] = traces[92 + i * 2 + 1][row];
    }
    lookup_blake_round_1[34][row] = traces[124][row];

    // ========== Triple XOR 32 (col125-col140) ==========
    uint32_t blake_output_words[8];
    for (int i = 0; i < 8; i++) {
        blake_output_words[i] = v_round[i] ^ v_round[i + 8] ^ h_words[i];
        m31 out_low = (m31)(blake_output_words[i] & 0xFFFF);
        m31 out_high = (m31)(blake_output_words[i] >> 16);
        traces[125 + i * 2][row] = out_low;
        traces[125 + i * 2 + 1][row] = out_high;

        // lookup_triple_xor_32_i
        m31 in0_low = (m31)(v_round[i] & 0xFFFF);
        m31 in0_high = (m31)(v_round[i] >> 16);
        m31 in1_low = (m31)(v_round[i + 8] & 0xFFFF);
        m31 in1_high = (m31)(v_round[i + 8] >> 16);
        m31 h_low = (m31)(h_words[i] & 0xFFFF);
        m31 h_high = (m31)(h_words[i] >> 16);

        m31 **lookup_arr = nullptr;
        switch (i) {
            case 0: lookup_arr = lookup_triple_xor_32_0; break;
            case 1: lookup_arr = lookup_triple_xor_32_1; break;
            case 2: lookup_arr = lookup_triple_xor_32_2; break;
            case 3: lookup_arr = lookup_triple_xor_32_3; break;
            case 4: lookup_arr = lookup_triple_xor_32_4; break;
            case 5: lookup_arr = lookup_triple_xor_32_5; break;
            case 6: lookup_arr = lookup_triple_xor_32_6; break;
            default: lookup_arr = lookup_triple_xor_32_7; break;
        }
        lookup_arr[0][row] = in0_low;
        lookup_arr[1][row] = in0_high;
        lookup_arr[2][row] = in1_low;
        lookup_arr[3][row] = in1_high;
        lookup_arr[4][row] = h_low;
        lookup_arr[5][row] = h_high;
        lookup_arr[6][row] = out_low;
        lookup_arr[7][row] = out_high;

        // sub-component inputs (flattened)
        int sc_base = i * 3;
        sub_component_inputs_triple_xor_32[sc_base + 0][row] = (m31)v_round[i];
        sub_component_inputs_triple_xor_32[sc_base + 1][row] = (m31)v_round[i + 8];
        sub_component_inputs_triple_xor_32[sc_base + 2][row] = (m31)h_words[i];
    }

    // ========== Output verification (col141-col172) ==========
    for (int i = 0; i < 8; i++) {
        uint32_t word = blake_output_words[i];
        uint16_t out_low = (uint16_t)(word & 0xFFFF);
        uint16_t out_high = (uint16_t)(word >> 16);
        m31 out_low_m31 = (m31)out_low;
        m31 out_high_m31 = (m31)out_high;

        uint16_t low_7 = out_low >> 9;
        uint16_t high_14 = out_high >> 2;
        uint16_t high_5 = high_14 >> 9;

        int col_base = 141 + i * 4;
        traces[col_base + 0][row] = (m31)low_7;
        traces[col_base + 1][row] = (m31)high_14;
        traces[col_base + 2][row] = (m31)high_5;

        m31 rc1 = sub(out_high_m31, mul((m31)high_14, M31_4));

        // range_check lookups
        m31 **lookup_rc = nullptr;
        switch (i) {
            case 0: lookup_rc = lookup_range_check_7_2_5_9; break;
            case 1: lookup_rc = lookup_range_check_7_2_5_10; break;
            case 2: lookup_rc = lookup_range_check_7_2_5_11; break;
            case 3: lookup_rc = lookup_range_check_7_2_5_12; break;
            case 4: lookup_rc = lookup_range_check_7_2_5_13; break;
            case 5: lookup_rc = lookup_range_check_7_2_5_14; break;
            case 6: lookup_rc = lookup_range_check_7_2_5_15; break;
            default: lookup_rc = lookup_range_check_7_2_5_16; break;
        }
        lookup_rc[0][row] = (m31)low_7;
        lookup_rc[1][row] = rc1;
        lookup_rc[2][row] = (m31)high_5;

        int rc_base = (9 + i) * 3;
        sub_component_inputs_range_check_7_2_5[rc_base + 0][row] = (m31)low_7;
        sub_component_inputs_range_check_7_2_5[rc_base + 1][row] = rc1;
        sub_component_inputs_range_check_7_2_5[rc_base + 2][row] = (m31)high_5;

        // Memory write verification
        m31 out_addr = add(output_ptr, (m31)i);
        m31 out_id = M31_0;
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, out_addr, &out_id);
        traces[col_base + 3][row] = out_id;

        m31 **lookup_mem_addr = nullptr;
        m31 **lookup_mem_big = nullptr;
        switch (i) {
            case 0: lookup_mem_addr = lookup_memory_address_to_id_12; lookup_mem_big = lookup_memory_id_to_big_12; break;
            case 1: lookup_mem_addr = lookup_memory_address_to_id_13; lookup_mem_big = lookup_memory_id_to_big_13; break;
            case 2: lookup_mem_addr = lookup_memory_address_to_id_14; lookup_mem_big = lookup_memory_id_to_big_14; break;
            case 3: lookup_mem_addr = lookup_memory_address_to_id_15; lookup_mem_big = lookup_memory_id_to_big_15; break;
            case 4: lookup_mem_addr = lookup_memory_address_to_id_16; lookup_mem_big = lookup_memory_id_to_big_16; break;
            case 5: lookup_mem_addr = lookup_memory_address_to_id_17; lookup_mem_big = lookup_memory_id_to_big_17; break;
            case 6: lookup_mem_addr = lookup_memory_address_to_id_18; lookup_mem_big = lookup_memory_id_to_big_18; break;
            default: lookup_mem_addr = lookup_memory_address_to_id_19; lookup_mem_big = lookup_memory_id_to_big_19; break;
        }

        lookup_mem_addr[0][row] = out_addr;
        lookup_mem_addr[1][row] = out_id;
        sub_component_inputs_memory_address_to_id[12 + i][row] = out_addr;

        m31 limb0 = sub(out_low_m31, mul((m31)low_7, M31_512));
        m31 limb1 = add((m31)low_7, mul(rc1, M31_128));
        m31 limb2 = sub((m31)high_14, mul((m31)high_5, M31_512));
        m31 limb3 = (m31)high_5;

        lookup_mem_big[0][row] = out_id;
        lookup_mem_big[1][row] = limb0;
        lookup_mem_big[2][row] = limb1;
        lookup_mem_big[3][row] = limb2;
        lookup_mem_big[4][row] = limb3;
        for (int k = 5; k < 29; k++) {
            lookup_mem_big[k][row] = M31_0;
        }
        sub_component_inputs_memory_id_to_big[12 + i][row] = out_id;
    }

    // ========== Lookup data: opcodes_0 and opcodes_1 ==========
    lookup_opcodes_0[0][row] = input_pc_col0;
    lookup_opcodes_0[1][row] = input_ap_col1;
    lookup_opcodes_0[2][row] = input_fp_col2;

    lookup_opcodes_1[0][row] = add(input_pc_col0, M31_1);
    lookup_opcodes_1[1][row] = add(input_ap_col1, ap_update_add_1_col9);
    lookup_opcodes_1[2][row] = input_fp_col2;

    // ========== Enabler column (col173) ==========
    // enabler = 1 for actual rows (row < row_offset), 0 for padding
    if (row < row_offset) {
        traces[BLAKE_COMPRESS_OPCODE_N_TRACE_COLUMNS - 1][row] = M31_1;
    } else {
        traces[BLAKE_COMPRESS_OPCODE_N_TRACE_COLUMNS - 1][row] = M31_0;
    }
}

/**
 * FFI function to generate blake_compress_opcode traces.
 */
void generate_blake_compress_opcode_traces(
    unsigned **traces,

    unsigned **lookup_blake_round_0,
    unsigned **lookup_blake_round_1,

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

    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    unsigned **lookup_range_check_7_2_5_0,
    unsigned **lookup_range_check_7_2_5_1,
    unsigned **lookup_range_check_7_2_5_2,
    unsigned **lookup_range_check_7_2_5_3,
    unsigned **lookup_range_check_7_2_5_4,
    unsigned **lookup_range_check_7_2_5_5,
    unsigned **lookup_range_check_7_2_5_6,
    unsigned **lookup_range_check_7_2_5_7,
    unsigned **lookup_range_check_7_2_5_8,
    unsigned **lookup_range_check_7_2_5_9,
    unsigned **lookup_range_check_7_2_5_10,
    unsigned **lookup_range_check_7_2_5_11,
    unsigned **lookup_range_check_7_2_5_12,
    unsigned **lookup_range_check_7_2_5_13,
    unsigned **lookup_range_check_7_2_5_14,
    unsigned **lookup_range_check_7_2_5_15,
    unsigned **lookup_range_check_7_2_5_16,

    unsigned **lookup_triple_xor_32_0,
    unsigned **lookup_triple_xor_32_1,
    unsigned **lookup_triple_xor_32_2,
    unsigned **lookup_triple_xor_32_3,
    unsigned **lookup_triple_xor_32_4,
    unsigned **lookup_triple_xor_32_5,
    unsigned **lookup_triple_xor_32_6,
    unsigned **lookup_triple_xor_32_7,

    unsigned **lookup_verify_bitwise_xor_8_0,
    unsigned **lookup_verify_bitwise_xor_8_1,
    unsigned **lookup_verify_bitwise_xor_8_2,
    unsigned **lookup_verify_bitwise_xor_8_3,

    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_7_2_5,
    unsigned **sub_component_inputs_verify_bitwise_xor_8,
    unsigned **sub_component_inputs_blake_round,
    unsigned **sub_component_inputs_triple_xor_32,

    unsigned **blake_compress_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {

    timer global_timer;
    global_timer.start("generate blake_compress_opcode base trace");

    unsigned trace_size = 1 << log_size;

    // Clone pointers to device
    m31 **device_traces = clone_to_device<m31*>(traces, BLAKE_COMPRESS_OPCODE_N_TRACE_COLUMNS);
    m31 **device_blake_compress_opcode_input = clone_to_device<m31*>(blake_compress_opcode_input, 3);
    unsigned **device_memory_id_to_big_transpose_big_value_ptr =
        clone_to_device<unsigned*>(memory_id_to_big_transposed_big_values, 8);

    // Clone lookup data pointers
    m31 **device_lookup_blake_round_0 = clone_to_device<m31*>(lookup_blake_round_0, 35);
    m31 **device_lookup_blake_round_1 = clone_to_device<m31*>(lookup_blake_round_1, 35);

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

    m31 **device_lookup_opcodes_0 = clone_to_device<m31*>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31*>(lookup_opcodes_1, 3);

    m31 **device_lookup_range_check_7_2_5_0 = clone_to_device<m31*>(lookup_range_check_7_2_5_0, 3);
    m31 **device_lookup_range_check_7_2_5_1 = clone_to_device<m31*>(lookup_range_check_7_2_5_1, 3);
    m31 **device_lookup_range_check_7_2_5_2 = clone_to_device<m31*>(lookup_range_check_7_2_5_2, 3);
    m31 **device_lookup_range_check_7_2_5_3 = clone_to_device<m31*>(lookup_range_check_7_2_5_3, 3);
    m31 **device_lookup_range_check_7_2_5_4 = clone_to_device<m31*>(lookup_range_check_7_2_5_4, 3);
    m31 **device_lookup_range_check_7_2_5_5 = clone_to_device<m31*>(lookup_range_check_7_2_5_5, 3);
    m31 **device_lookup_range_check_7_2_5_6 = clone_to_device<m31*>(lookup_range_check_7_2_5_6, 3);
    m31 **device_lookup_range_check_7_2_5_7 = clone_to_device<m31*>(lookup_range_check_7_2_5_7, 3);
    m31 **device_lookup_range_check_7_2_5_8 = clone_to_device<m31*>(lookup_range_check_7_2_5_8, 3);
    m31 **device_lookup_range_check_7_2_5_9 = clone_to_device<m31*>(lookup_range_check_7_2_5_9, 3);
    m31 **device_lookup_range_check_7_2_5_10 = clone_to_device<m31*>(lookup_range_check_7_2_5_10, 3);
    m31 **device_lookup_range_check_7_2_5_11 = clone_to_device<m31*>(lookup_range_check_7_2_5_11, 3);
    m31 **device_lookup_range_check_7_2_5_12 = clone_to_device<m31*>(lookup_range_check_7_2_5_12, 3);
    m31 **device_lookup_range_check_7_2_5_13 = clone_to_device<m31*>(lookup_range_check_7_2_5_13, 3);
    m31 **device_lookup_range_check_7_2_5_14 = clone_to_device<m31*>(lookup_range_check_7_2_5_14, 3);
    m31 **device_lookup_range_check_7_2_5_15 = clone_to_device<m31*>(lookup_range_check_7_2_5_15, 3);
    m31 **device_lookup_range_check_7_2_5_16 = clone_to_device<m31*>(lookup_range_check_7_2_5_16, 3);

    m31 **device_lookup_triple_xor_32_0 = clone_to_device<m31*>(lookup_triple_xor_32_0, 8);
    m31 **device_lookup_triple_xor_32_1 = clone_to_device<m31*>(lookup_triple_xor_32_1, 8);
    m31 **device_lookup_triple_xor_32_2 = clone_to_device<m31*>(lookup_triple_xor_32_2, 8);
    m31 **device_lookup_triple_xor_32_3 = clone_to_device<m31*>(lookup_triple_xor_32_3, 8);
    m31 **device_lookup_triple_xor_32_4 = clone_to_device<m31*>(lookup_triple_xor_32_4, 8);
    m31 **device_lookup_triple_xor_32_5 = clone_to_device<m31*>(lookup_triple_xor_32_5, 8);
    m31 **device_lookup_triple_xor_32_6 = clone_to_device<m31*>(lookup_triple_xor_32_6, 8);
    m31 **device_lookup_triple_xor_32_7 = clone_to_device<m31*>(lookup_triple_xor_32_7, 8);

    m31 **device_lookup_verify_bitwise_xor_8_0 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_0, 3);
    m31 **device_lookup_verify_bitwise_xor_8_1 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_1, 3);
    m31 **device_lookup_verify_bitwise_xor_8_2 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_2, 3);
    m31 **device_lookup_verify_bitwise_xor_8_3 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_3, 3);

    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31*>(lookup_verify_instruction_0, 7);

    // Clone sub-component input pointers
    m31 **device_sub_component_inputs_verify_instruction =
        clone_to_device<m31*>(sub_component_inputs_verify_instruction, 7);
    m31 **device_sub_component_inputs_memory_address_to_id =
        clone_to_device<m31*>(sub_component_inputs_memory_address_to_id, 20);
    m31 **device_sub_component_inputs_memory_id_to_big =
        clone_to_device<m31*>(sub_component_inputs_memory_id_to_big, 20);
    m31 **device_sub_component_inputs_range_check_7_2_5 =
        clone_to_device<m31*>(sub_component_inputs_range_check_7_2_5, 17 * 3);
    m31 **device_sub_component_inputs_verify_bitwise_xor_8 =
        clone_to_device<m31*>(sub_component_inputs_verify_bitwise_xor_8, 4 * 3);
    m31 **device_sub_component_inputs_blake_round =
        clone_to_device<m31*>(sub_component_inputs_blake_round, 10 * 19);
    m31 **device_sub_component_inputs_triple_xor_32 =
        clone_to_device<m31*>(sub_component_inputs_triple_xor_32, 8 * 3);

    int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX
        ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX
        ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_blake_compress_opcode_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,

        device_lookup_blake_round_0,
        device_lookup_blake_round_1,

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

        device_lookup_opcodes_0,
        device_lookup_opcodes_1,

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
        device_lookup_range_check_7_2_5_16,

        device_lookup_triple_xor_32_0,
        device_lookup_triple_xor_32_1,
        device_lookup_triple_xor_32_2,
        device_lookup_triple_xor_32_3,
        device_lookup_triple_xor_32_4,
        device_lookup_triple_xor_32_5,
        device_lookup_triple_xor_32_6,
        device_lookup_triple_xor_32_7,

        device_lookup_verify_bitwise_xor_8_0,
        device_lookup_verify_bitwise_xor_8_1,
        device_lookup_verify_bitwise_xor_8_2,
        device_lookup_verify_bitwise_xor_8_3,

        device_lookup_verify_instruction_0,

        device_sub_component_inputs_verify_instruction,
        device_sub_component_inputs_memory_address_to_id,
        device_sub_component_inputs_memory_id_to_big,
        device_sub_component_inputs_range_check_7_2_5,
        device_sub_component_inputs_verify_bitwise_xor_8,
        device_sub_component_inputs_blake_round,
        device_sub_component_inputs_triple_xor_32,

        device_blake_compress_opcode_input,

        memory_address_to_id_address_to_raw_id,
        device_memory_id_to_big_transpose_big_value_ptr,
        memory_id_to_big_small_values,

        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate blake_compress_opcode base trace");

    // Free device memory
    cuda_free_memory(device_traces);
    cuda_free_memory(device_blake_compress_opcode_input);
    cuda_free_memory(device_memory_id_to_big_transpose_big_value_ptr);

    cuda_free_memory(device_lookup_blake_round_0);
    cuda_free_memory(device_lookup_blake_round_1);

    // Free all lookup memory_address_to_id pointers
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

    // Free all lookup memory_id_to_big pointers
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

    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);

    // Free range check lookup pointers
    cuda_free_memory(device_lookup_range_check_7_2_5_0);
    cuda_free_memory(device_lookup_range_check_7_2_5_1);
    cuda_free_memory(device_lookup_range_check_7_2_5_2);
    cuda_free_memory(device_lookup_range_check_7_2_5_3);
    cuda_free_memory(device_lookup_range_check_7_2_5_4);
    cuda_free_memory(device_lookup_range_check_7_2_5_5);
    cuda_free_memory(device_lookup_range_check_7_2_5_6);
    cuda_free_memory(device_lookup_range_check_7_2_5_7);
    cuda_free_memory(device_lookup_range_check_7_2_5_8);
    cuda_free_memory(device_lookup_range_check_7_2_5_9);
    cuda_free_memory(device_lookup_range_check_7_2_5_10);
    cuda_free_memory(device_lookup_range_check_7_2_5_11);
    cuda_free_memory(device_lookup_range_check_7_2_5_12);
    cuda_free_memory(device_lookup_range_check_7_2_5_13);
    cuda_free_memory(device_lookup_range_check_7_2_5_14);
    cuda_free_memory(device_lookup_range_check_7_2_5_15);
    cuda_free_memory(device_lookup_range_check_7_2_5_16);

    // Free triple xor lookup pointers
    cuda_free_memory(device_lookup_triple_xor_32_0);
    cuda_free_memory(device_lookup_triple_xor_32_1);
    cuda_free_memory(device_lookup_triple_xor_32_2);
    cuda_free_memory(device_lookup_triple_xor_32_3);
    cuda_free_memory(device_lookup_triple_xor_32_4);
    cuda_free_memory(device_lookup_triple_xor_32_5);
    cuda_free_memory(device_lookup_triple_xor_32_6);
    cuda_free_memory(device_lookup_triple_xor_32_7);

    cuda_free_memory(device_lookup_verify_bitwise_xor_8_0);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_1);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_2);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_3);

    cuda_free_memory(device_lookup_verify_instruction_0);

    // Free sub-component input pointers
    cuda_free_memory(device_sub_component_inputs_verify_instruction);
    cuda_free_memory(device_sub_component_inputs_memory_address_to_id);
    cuda_free_memory(device_sub_component_inputs_memory_id_to_big);
    cuda_free_memory(device_sub_component_inputs_range_check_7_2_5);
    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_8);
    cuda_free_memory(device_sub_component_inputs_blake_round);
    cuda_free_memory(device_sub_component_inputs_triple_xor_32);
}

/**
 * Interaction trace generation kernels for blake_compress_opcode.
 *
 * The interaction trace has 37 columns (N_INTERACTION_TRACE_COLUMNS).
 * Each column is a QM31 value, so we have 37 × 4 = 148 M31 columns.
 */

// Template kernel for generating logup column with two lookups (add)
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void blake_compress_interaction_col_gen_add_kernel(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    m31 **lookup_data_0,
    m31 **lookup_data_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_index < trace_size) {
        m31 combine_reg_0[N] = {0};
        m31 combine_reg_1[M] = {0};

        for (int i = 0; i < N; i++) {
            combine_reg_0[i] = lookup_data_0[i][vec_index];
        }
        for (int i = 0; i < M; i++) {
            combine_reg_1[i] = lookup_data_1[i][vec_index];
        }

        qm31 denom0 = lookup_elements_n->combine(combine_reg_0, N);
        qm31 denom1 = lookup_elements_m->combine(combine_reg_1, M);
        logup_col_write_frac(vec_index, add(denom1, denom0), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Template kernel for generating logup column with two lookups (subtract - for blake_round)
template <int N, int M>
__launch_bounds__(256, 2)
__global__ void blake_compress_interaction_col_gen_sub_kernel(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    m31 **lookup_data_0,
    m31 **lookup_data_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_index < trace_size) {
        m31 combine_reg_0[N] = {0};
        m31 combine_reg_1[M] = {0};

        for (int i = 0; i < N; i++) {
            combine_reg_0[i] = lookup_data_0[i][vec_index];
        }
        for (int i = 0; i < M; i++) {
            combine_reg_1[i] = lookup_data_1[i][vec_index];
        }

        qm31 denom0 = lookup_elements_n->combine(combine_reg_0, N);
        qm31 denom1 = lookup_elements_m->combine(combine_reg_1, M);
        // Subtract instead of add for blake_round
        logup_col_write_frac(vec_index, sub(denom0, denom1), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Single lookup with enabler
template <int N>
__launch_bounds__(256, 2)
__global__ void blake_compress_interaction_col_gen_single_kernel(
    LookupElementsBasic<N> *lookup_elements_n,
    m31 **lookup_data_0,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned row_offset
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    qm31 enabler = {0};
    if (vec_index < row_offset) {
        enabler = qm31{cm31{(m31)(P - 1), (m31)0}, cm31{(m31)0, (m31)0}};  // -1 in M31
    }

    if (vec_index < trace_size) {
        m31 combine_reg[N] = {0};
        for (int i = 0; i < N; i++) {
            combine_reg[i] = lookup_data_0[i][vec_index];
        }

        qm31 denom = lookup_elements_n->combine(combine_reg, N);
        logup_col_write_frac(vec_index, enabler, denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Opcodes column with enabler (denom1 - denom0) * enabler
template <int N>
__launch_bounds__(256, 2)
__global__ void blake_compress_interaction_opcodes_kernel(
    LookupElementsBasic<N> *lookup_elements_n,
    m31 **lookup_data_0,
    m31 **lookup_data_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    unsigned row_offset
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        m31 combine_reg_0[N] = {0};
        m31 combine_reg_1[N] = {0};
        for (int i = 0; i < N; i++) {
            combine_reg_0[i] = lookup_data_0[i][vec_index];
            combine_reg_1[i] = lookup_data_1[i][vec_index];
        }
        qm31 denom0 = lookup_elements_n->combine(combine_reg_0, N);
        qm31 denom1 = lookup_elements_n->combine(combine_reg_1, N);
        qm31 enabler = {0};
        if (vec_index < row_offset) {
            enabler = qm31{cm31{(m31)1, (m31)0}, cm31{(m31)0, (m31)0}};
        }
        qm31 numer = mul(sub(denom1, denom0), enabler);
        logup_col_write_frac(vec_index, numer, mul(denom0, denom1),
                             denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize column kernel
__global__ void blake_compress_interaction_finalize_col_kernel(
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

// Cumsum shift kernel
__global__ void blake_compress_interaction_cumsum_shift_kernel(
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

    m31 sum0 = {0};
    m31 sum1 = {0};
    m31 sum2 = {0};
    m31 sum3 = {0};

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

// Coordinate prefix sum kernel
__global__ void blake_compress_interaction_coord_prefix_sum_kernel(
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
        qm31 cumsum_shift = div(claimed_sum, (m31)trace_size);

        interactive_traces[4 * last_index - 4][vec_index] =
            sub(interactive_traces[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interactive_traces[4 * last_index - 3][vec_index] =
            sub(interactive_traces[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interactive_traces[4 * last_index - 2][vec_index] =
            sub(interactive_traces[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interactive_traces[4 * last_index - 1][vec_index] =
            sub(interactive_traces[4 * last_index - 1][vec_index], cumsum_shift.b.b);
    }
}

/**
 * Debug helper function to dump numerator data from device to host and print.
 * (No-op: debug output disabled)
 */
void dump_numerator_data(m31 *device_numerator0, m31 *device_numerator1, m31 *device_numerator2, m31 *device_numerator3, unsigned trace_size) {
    (void)device_numerator0; (void)device_numerator1; (void)device_numerator2; (void)device_numerator3; (void)trace_size;
}

/**
 * Debug helper function to dump denominator and denom_inv data from device to host and print.
 * (No-op: debug output disabled)
 */
void dump_denom_data(qm31 *device_denom, qm31 *device_denom_inv, unsigned trace_size, unsigned col_index) {
    (void)device_denom; (void)device_denom_inv; (void)trace_size; (void)col_index;
}

/**
 * Debug helper function to dump interaction trace data from device to host and print.
 * (No-op: debug output disabled)
 */
void dump_interaction_traces(unsigned **interaction_trace, unsigned col_index, unsigned trace_size) {
    (void)interaction_trace; (void)col_index; (void)trace_size;
}

/**
 * FFI function to generate blake_compress_opcode interaction traces.
 */
void generate_blake_compress_opcode_interaction_traces(
    void *blake_round,
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *range_check_7_2_5,
    void *triple_xor_32,
    void *verify_bitwise_xor_8,
    void *verify_instruction,

    unsigned **lookup_blake_round_0,
    unsigned **lookup_blake_round_1,

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

    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    unsigned **lookup_range_check_7_2_5_0,
    unsigned **lookup_range_check_7_2_5_1,
    unsigned **lookup_range_check_7_2_5_2,
    unsigned **lookup_range_check_7_2_5_3,
    unsigned **lookup_range_check_7_2_5_4,
    unsigned **lookup_range_check_7_2_5_5,
    unsigned **lookup_range_check_7_2_5_6,
    unsigned **lookup_range_check_7_2_5_7,
    unsigned **lookup_range_check_7_2_5_8,
    unsigned **lookup_range_check_7_2_5_9,
    unsigned **lookup_range_check_7_2_5_10,
    unsigned **lookup_range_check_7_2_5_11,
    unsigned **lookup_range_check_7_2_5_12,
    unsigned **lookup_range_check_7_2_5_13,
    unsigned **lookup_range_check_7_2_5_14,
    unsigned **lookup_range_check_7_2_5_15,
    unsigned **lookup_range_check_7_2_5_16,

    unsigned **lookup_triple_xor_32_0,
    unsigned **lookup_triple_xor_32_1,
    unsigned **lookup_triple_xor_32_2,
    unsigned **lookup_triple_xor_32_3,
    unsigned **lookup_triple_xor_32_4,
    unsigned **lookup_triple_xor_32_5,
    unsigned **lookup_triple_xor_32_6,
    unsigned **lookup_triple_xor_32_7,

    unsigned **lookup_verify_bitwise_xor_8_0,
    unsigned **lookup_verify_bitwise_xor_8_1,
    unsigned **lookup_verify_bitwise_xor_8_2,
    unsigned **lookup_verify_bitwise_xor_8_3,

    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
) {

    unsigned trace_size = 1u << log_size;
    unsigned row_offset = n_rows;

    // Copy lookup element structs to device
    BlakeRound *host_blake_round = (BlakeRound *)blake_round;
    MemoryAddressToId *host_memory_address_to_id = (MemoryAddressToId *)memory_address_to_id;
    MemoryIdToBig *host_memory_id_to_big = (MemoryIdToBig *)memory_id_to_big;
    Opcodes *host_opcodes = (Opcodes *)opcodes;
    RangeCheck_7_2_5 *host_range_check_7_2_5 = (RangeCheck_7_2_5 *)range_check_7_2_5;
    TripleXor32 *host_triple_xor_32 = (TripleXor32 *)triple_xor_32;
    VerifyBitwiseXor_8 *host_verify_bitwise_xor_8 = (VerifyBitwiseXor_8 *)verify_bitwise_xor_8;
    VerifyInstruction *host_verify_instruction = (VerifyInstruction *)verify_instruction;

    BlakeRound *device_blake_round = cuda_malloc<BlakeRound>(1);
    MemoryAddressToId *device_memory_address_to_id = cuda_malloc<MemoryAddressToId>(1);
    MemoryIdToBig *device_memory_id_to_big = cuda_malloc<MemoryIdToBig>(1);
    Opcodes *device_opcodes = cuda_malloc<Opcodes>(1);
    RangeCheck_7_2_5 *device_range_check_7_2_5 = cuda_malloc<RangeCheck_7_2_5>(1);
    TripleXor32 *device_triple_xor_32 = cuda_malloc<TripleXor32>(1);
    VerifyBitwiseXor_8 *device_verify_bitwise_xor_8 = cuda_malloc<VerifyBitwiseXor_8>(1);
    VerifyInstruction *device_verify_instruction = cuda_malloc<VerifyInstruction>(1);

    cuda_mem_copy_host_to_device(host_blake_round, device_blake_round, 1);
    cuda_mem_copy_host_to_device(host_memory_address_to_id, device_memory_address_to_id, 1);
    cuda_mem_copy_host_to_device(host_memory_id_to_big, device_memory_id_to_big, 1);
    cuda_mem_copy_host_to_device(host_opcodes, device_opcodes, 1);
    cuda_mem_copy_host_to_device(host_range_check_7_2_5, device_range_check_7_2_5, 1);
    cuda_mem_copy_host_to_device(host_triple_xor_32, device_triple_xor_32, 1);
    cuda_mem_copy_host_to_device(host_verify_bitwise_xor_8, device_verify_bitwise_xor_8, 1);
    cuda_mem_copy_host_to_device(host_verify_instruction, device_verify_instruction, 1);

    // Clone lookup data pointers
    m31 **device_lookup_blake_round_0 = clone_to_device<m31*>(lookup_blake_round_0, 35);
    m31 **device_lookup_blake_round_1 = clone_to_device<m31*>(lookup_blake_round_1, 35);

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

    m31 **device_lookup_opcodes_0 = clone_to_device<m31*>(lookup_opcodes_0, 3);
    m31 **device_lookup_opcodes_1 = clone_to_device<m31*>(lookup_opcodes_1, 3);

    m31 **device_lookup_range_check_7_2_5_0 = clone_to_device<m31*>(lookup_range_check_7_2_5_0, 3);
    m31 **device_lookup_range_check_7_2_5_1 = clone_to_device<m31*>(lookup_range_check_7_2_5_1, 3);
    m31 **device_lookup_range_check_7_2_5_2 = clone_to_device<m31*>(lookup_range_check_7_2_5_2, 3);
    m31 **device_lookup_range_check_7_2_5_3 = clone_to_device<m31*>(lookup_range_check_7_2_5_3, 3);
    m31 **device_lookup_range_check_7_2_5_4 = clone_to_device<m31*>(lookup_range_check_7_2_5_4, 3);
    m31 **device_lookup_range_check_7_2_5_5 = clone_to_device<m31*>(lookup_range_check_7_2_5_5, 3);
    m31 **device_lookup_range_check_7_2_5_6 = clone_to_device<m31*>(lookup_range_check_7_2_5_6, 3);
    m31 **device_lookup_range_check_7_2_5_7 = clone_to_device<m31*>(lookup_range_check_7_2_5_7, 3);
    m31 **device_lookup_range_check_7_2_5_8 = clone_to_device<m31*>(lookup_range_check_7_2_5_8, 3);
    m31 **device_lookup_range_check_7_2_5_9 = clone_to_device<m31*>(lookup_range_check_7_2_5_9, 3);
    m31 **device_lookup_range_check_7_2_5_10 = clone_to_device<m31*>(lookup_range_check_7_2_5_10, 3);
    m31 **device_lookup_range_check_7_2_5_11 = clone_to_device<m31*>(lookup_range_check_7_2_5_11, 3);
    m31 **device_lookup_range_check_7_2_5_12 = clone_to_device<m31*>(lookup_range_check_7_2_5_12, 3);
    m31 **device_lookup_range_check_7_2_5_13 = clone_to_device<m31*>(lookup_range_check_7_2_5_13, 3);
    m31 **device_lookup_range_check_7_2_5_14 = clone_to_device<m31*>(lookup_range_check_7_2_5_14, 3);
    m31 **device_lookup_range_check_7_2_5_15 = clone_to_device<m31*>(lookup_range_check_7_2_5_15, 3);
    m31 **device_lookup_range_check_7_2_5_16 = clone_to_device<m31*>(lookup_range_check_7_2_5_16, 3);

    m31 **device_lookup_triple_xor_32_0 = clone_to_device<m31*>(lookup_triple_xor_32_0, 8);
    m31 **device_lookup_triple_xor_32_1 = clone_to_device<m31*>(lookup_triple_xor_32_1, 8);
    m31 **device_lookup_triple_xor_32_2 = clone_to_device<m31*>(lookup_triple_xor_32_2, 8);
    m31 **device_lookup_triple_xor_32_3 = clone_to_device<m31*>(lookup_triple_xor_32_3, 8);
    m31 **device_lookup_triple_xor_32_4 = clone_to_device<m31*>(lookup_triple_xor_32_4, 8);
    m31 **device_lookup_triple_xor_32_5 = clone_to_device<m31*>(lookup_triple_xor_32_5, 8);
    m31 **device_lookup_triple_xor_32_6 = clone_to_device<m31*>(lookup_triple_xor_32_6, 8);
    m31 **device_lookup_triple_xor_32_7 = clone_to_device<m31*>(lookup_triple_xor_32_7, 8);

    m31 **device_lookup_verify_bitwise_xor_8_0 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_0, 3);
    m31 **device_lookup_verify_bitwise_xor_8_1 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_1, 3);
    m31 **device_lookup_verify_bitwise_xor_8_2 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_2, 3);
    m31 **device_lookup_verify_bitwise_xor_8_3 = clone_to_device<m31*>(lookup_verify_bitwise_xor_8_3, 3);

    m31 **device_lookup_verify_instruction_0 = clone_to_device<m31*>(lookup_verify_instruction_0, 7);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_trace, 4 * BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    timer global_timer;
    global_timer.start("generate blake_compress_opcode interaction trace");

    auto launch_finalize = [&](unsigned col_index) {
        int block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
        int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
        // Debug: dump denom and denom_inv after batch inverse (only for column 0)
        if (col_index == 0) {
            dump_denom_data(device_logup_denom, denom_inv, trace_size, col_index);
        }
        blake_compress_interaction_finalize_col_kernel<<<num_blocks, block_dim>>>(
            col_index, trace_size, denom_inv,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3,
            device_interaction_traces
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    };

    // Column 0: verify_instruction_0 + memory_address_to_id_0
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<7, 2><<<num_blocks, block_dim>>>(
            device_verify_instruction, device_memory_address_to_id,
            device_lookup_verify_instruction_0, device_lookup_memory_address_to_id_0,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(0);

        dump_interaction_traces(interaction_trace, 0, trace_size);
    }

    // Column 1: memory_id_to_big_0 + memory_address_to_id_1
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 2><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_memory_address_to_id,
            device_lookup_memory_id_to_big_0, device_lookup_memory_address_to_id_1,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(1);

        dump_interaction_traces(interaction_trace, 1, trace_size);
    }

    // Column 2: memory_id_to_big_1 + memory_address_to_id_2
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 2><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_memory_address_to_id,
            device_lookup_memory_id_to_big_1, device_lookup_memory_address_to_id_2,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(2);

        dump_interaction_traces(interaction_trace, 2, trace_size);
    }

    // Column 3: memory_id_to_big_2 + range_check_7_2_5_0
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_2, device_lookup_range_check_7_2_5_0,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(3);

        dump_interaction_traces(interaction_trace, 3, trace_size);
    }

    // Column 4: memory_address_to_id_3 + memory_id_to_big_3
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_3, device_lookup_memory_id_to_big_3,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(4);

        dump_interaction_traces(interaction_trace, 4, trace_size);
    }

    // Column 5: range_check_7_2_5_1 + memory_address_to_id_4
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_1, device_lookup_memory_address_to_id_4,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(5);

        dump_interaction_traces(interaction_trace, 5, trace_size);
    }

    // Column 6: memory_id_to_big_4 + range_check_7_2_5_2
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_4, device_lookup_range_check_7_2_5_2,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(6);

        dump_interaction_traces(interaction_trace, 6, trace_size);
    }

    // Column 7: memory_address_to_id_5 + memory_id_to_big_5
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_5, device_lookup_memory_id_to_big_5,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(7);

        dump_interaction_traces(interaction_trace, 7, trace_size);
    }

    // Column 8: range_check_7_2_5_3 + memory_address_to_id_6
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_3, device_lookup_memory_address_to_id_6,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(8);

        dump_interaction_traces(interaction_trace, 8, trace_size);
    }

    // Column 9: memory_id_to_big_6 + range_check_7_2_5_4
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_6, device_lookup_range_check_7_2_5_4,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(9);

        dump_interaction_traces(interaction_trace, 9, trace_size);
    }

    // Column 10: memory_address_to_id_7 + memory_id_to_big_7
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_7, device_lookup_memory_id_to_big_7,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(10);

        dump_interaction_traces(interaction_trace, 10, trace_size);
    }

    // Column 11: range_check_7_2_5_5 + memory_address_to_id_8
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_5, device_lookup_memory_address_to_id_8,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(11);

        dump_interaction_traces(interaction_trace, 11, trace_size);
    }

    // Column 12: memory_id_to_big_8 + range_check_7_2_5_6
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_8, device_lookup_range_check_7_2_5_6,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(12);

        dump_interaction_traces(interaction_trace, 12, trace_size);
    }

    // Column 13: memory_address_to_id_9 + memory_id_to_big_9
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_9, device_lookup_memory_id_to_big_9,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(13);

        dump_interaction_traces(interaction_trace, 13, trace_size);
    }

    // Column 14: range_check_7_2_5_7 + memory_address_to_id_10
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_7, device_lookup_memory_address_to_id_10,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(14);

        dump_interaction_traces(interaction_trace, 14, trace_size);
    }

    // Column 15: memory_id_to_big_10 + range_check_7_2_5_8
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_10, device_lookup_range_check_7_2_5_8,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(15);

        dump_interaction_traces(interaction_trace, 15, trace_size);
    }

    // Column 16: memory_address_to_id_11 + memory_id_to_big_11
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_11, device_lookup_memory_id_to_big_11,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(16);

        dump_interaction_traces(interaction_trace, 16, trace_size);
    }

    // Column 17: verify_bitwise_xor_8_0 + verify_bitwise_xor_8_1
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 3><<<num_blocks, block_dim>>>(
            device_verify_bitwise_xor_8, device_verify_bitwise_xor_8,
            device_lookup_verify_bitwise_xor_8_0, device_lookup_verify_bitwise_xor_8_1,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(17);

        dump_interaction_traces(interaction_trace, 17, trace_size);
    }

    // Column 18: verify_bitwise_xor_8_2 + verify_bitwise_xor_8_3
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 3><<<num_blocks, block_dim>>>(
            device_verify_bitwise_xor_8, device_verify_bitwise_xor_8,
            device_lookup_verify_bitwise_xor_8_2, device_lookup_verify_bitwise_xor_8_3,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(18);

        dump_interaction_traces(interaction_trace, 18, trace_size);
    }

    // Column 19: blake_round_0 - blake_round_1 (uses subtraction)
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_sub_kernel<35, 35><<<num_blocks, block_dim>>>(
            device_blake_round, device_blake_round,
            device_lookup_blake_round_0, device_lookup_blake_round_1,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(19);

        dump_interaction_traces(interaction_trace, 19, trace_size);
    }

    // Column 20: triple_xor_32_0 + triple_xor_32_1
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<8, 8><<<num_blocks, block_dim>>>(
            device_triple_xor_32, device_triple_xor_32,
            device_lookup_triple_xor_32_0, device_lookup_triple_xor_32_1,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(20);

        dump_interaction_traces(interaction_trace, 20, trace_size);
    }

    // Column 21: triple_xor_32_2 + triple_xor_32_3
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<8, 8><<<num_blocks, block_dim>>>(
            device_triple_xor_32, device_triple_xor_32,
            device_lookup_triple_xor_32_2, device_lookup_triple_xor_32_3,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(21);

        dump_interaction_traces(interaction_trace, 21, trace_size);
    }

    // Column 22: triple_xor_32_4 + triple_xor_32_5
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<8, 8><<<num_blocks, block_dim>>>(
            device_triple_xor_32, device_triple_xor_32,
            device_lookup_triple_xor_32_4, device_lookup_triple_xor_32_5,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(22);

        dump_interaction_traces(interaction_trace, 22, trace_size);
    }

    // Column 23: triple_xor_32_6 + triple_xor_32_7
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<8, 8><<<num_blocks, block_dim>>>(
            device_triple_xor_32, device_triple_xor_32,
            device_lookup_triple_xor_32_6, device_lookup_triple_xor_32_7,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(23);

        dump_interaction_traces(interaction_trace, 23, trace_size);
    }

    // Column 24: range_check_7_2_5_9 + memory_address_to_id_12
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_9, device_lookup_memory_address_to_id_12,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(24);

        dump_interaction_traces(interaction_trace, 24, trace_size);
    }

    // Column 25: memory_id_to_big_12 + range_check_7_2_5_10
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_12, device_lookup_range_check_7_2_5_10,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(25);

        dump_interaction_traces(interaction_trace, 25, trace_size);
    }

    // Column 26: memory_address_to_id_13 + memory_id_to_big_13
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_13, device_lookup_memory_id_to_big_13,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(26);

        dump_interaction_traces(interaction_trace, 26, trace_size);
    }

    // Column 27: range_check_7_2_5_11 + memory_address_to_id_14
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_11, device_lookup_memory_address_to_id_14,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(27);

        dump_interaction_traces(interaction_trace, 27, trace_size);
    }

    // Column 28: memory_id_to_big_14 + range_check_7_2_5_12
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_14, device_lookup_range_check_7_2_5_12,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(28);

        dump_interaction_traces(interaction_trace, 28, trace_size);
    }

    // Column 29: memory_address_to_id_15 + memory_id_to_big_15
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_15, device_lookup_memory_id_to_big_15,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(29);

        dump_interaction_traces(interaction_trace, 29, trace_size);
    }

    // Column 30: range_check_7_2_5_13 + memory_address_to_id_16
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_13, device_lookup_memory_address_to_id_16,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(30);

        dump_interaction_traces(interaction_trace, 30, trace_size);
    }

    // Column 31: memory_id_to_big_16 + range_check_7_2_5_14
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_16, device_lookup_range_check_7_2_5_14,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(31);

        dump_interaction_traces(interaction_trace, 31, trace_size);
    }

    // Column 32: memory_address_to_id_17 + memory_id_to_big_17
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_17, device_lookup_memory_id_to_big_17,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(32);

        dump_interaction_traces(interaction_trace, 32, trace_size);
    }

    // Column 33: range_check_7_2_5_15 + memory_address_to_id_18
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<3, 2><<<num_blocks, block_dim>>>(
            device_range_check_7_2_5, device_memory_address_to_id,
            device_lookup_range_check_7_2_5_15, device_lookup_memory_address_to_id_18,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(33);

        dump_interaction_traces(interaction_trace, 33, trace_size);
    }

    // Column 34: memory_id_to_big_18 + range_check_7_2_5_16
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<29, 3><<<num_blocks, block_dim>>>(
            device_memory_id_to_big, device_range_check_7_2_5,
            device_lookup_memory_id_to_big_18, device_lookup_range_check_7_2_5_16,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(34);

        dump_interaction_traces(interaction_trace, 34, trace_size);
    }

    // Column 35: memory_address_to_id_19 + memory_id_to_big_19
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_col_gen_add_kernel<2, 29><<<num_blocks, block_dim>>>(
            device_memory_address_to_id, device_memory_id_to_big,
            device_lookup_memory_address_to_id_19, device_lookup_memory_id_to_big_19,
            trace_size, device_logup_denom,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(35);

        dump_interaction_traces(interaction_trace, 35, trace_size);
    }

    // Opcodes column (with enabler)
    {
        int block_dim = trace_size < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX;
        int num_blocks = block_dim < BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        blake_compress_interaction_opcodes_kernel<3><<<num_blocks, block_dim>>>(
            device_opcodes,
            device_lookup_opcodes_0,
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

        dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

        launch_finalize(36);

        dump_interaction_traces(interaction_trace, 36, trace_size);
    }

    // Compute claimed_sum and shift (reuse existing kernels)
    {
        int block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
        int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        size_t shared_size = 4 * block_dim * sizeof(m31);
        blake_compress_interaction_cumsum_shift_kernel<<<num_blocks, block_dim, shared_size>>>(
            BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS,
            trace_size,
            device_interaction_traces,
            (m31 *)claimed_sum
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        blake_compress_interaction_coord_prefix_sum_kernel<<<num_blocks, block_dim>>>(
            (m31 *)claimed_sum,
            BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS,
            trace_size,
            device_interaction_traces
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        inclusive_prefix_sum(interaction_trace[4 * BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);
    }

    // Free device memory
    cuda_free_memory(device_blake_round);
    cuda_free_memory(device_memory_address_to_id);
    cuda_free_memory(device_memory_id_to_big);
    cuda_free_memory(device_opcodes);
    cuda_free_memory(device_range_check_7_2_5);
    cuda_free_memory(device_triple_xor_32);
    cuda_free_memory(device_verify_bitwise_xor_8);
    cuda_free_memory(device_verify_instruction);

    cuda_free_memory(device_lookup_blake_round_0);
    cuda_free_memory(device_lookup_blake_round_1);

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

    cuda_free_memory(device_lookup_opcodes_0);
    cuda_free_memory(device_lookup_opcodes_1);

    cuda_free_memory(device_lookup_range_check_7_2_5_0);
    cuda_free_memory(device_lookup_range_check_7_2_5_1);
    cuda_free_memory(device_lookup_range_check_7_2_5_2);
    cuda_free_memory(device_lookup_range_check_7_2_5_3);
    cuda_free_memory(device_lookup_range_check_7_2_5_4);
    cuda_free_memory(device_lookup_range_check_7_2_5_5);
    cuda_free_memory(device_lookup_range_check_7_2_5_6);
    cuda_free_memory(device_lookup_range_check_7_2_5_7);
    cuda_free_memory(device_lookup_range_check_7_2_5_8);
    cuda_free_memory(device_lookup_range_check_7_2_5_9);
    cuda_free_memory(device_lookup_range_check_7_2_5_10);
    cuda_free_memory(device_lookup_range_check_7_2_5_11);
    cuda_free_memory(device_lookup_range_check_7_2_5_12);
    cuda_free_memory(device_lookup_range_check_7_2_5_13);
    cuda_free_memory(device_lookup_range_check_7_2_5_14);
    cuda_free_memory(device_lookup_range_check_7_2_5_15);
    cuda_free_memory(device_lookup_range_check_7_2_5_16);

    cuda_free_memory(device_lookup_triple_xor_32_0);
    cuda_free_memory(device_lookup_triple_xor_32_1);
    cuda_free_memory(device_lookup_triple_xor_32_2);
    cuda_free_memory(device_lookup_triple_xor_32_3);
    cuda_free_memory(device_lookup_triple_xor_32_4);
    cuda_free_memory(device_lookup_triple_xor_32_5);
    cuda_free_memory(device_lookup_triple_xor_32_6);
    cuda_free_memory(device_lookup_triple_xor_32_7);

    cuda_free_memory(device_lookup_verify_bitwise_xor_8_0);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_1);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_2);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_3);

    cuda_free_memory(device_lookup_verify_instruction_0);

    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate blake_compress_opcode interaction traces");
}
