
#ifndef GEN_BLAKE_COMPRESS_OPCODE_TRACE_H
#define GEN_BLAKE_COMPRESS_OPCODE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define BLAKE_COMPRESS_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

#define BLAKE_COMPRESS_OPCODE_N_TRACE_COLUMNS 174
#define BLAKE_COMPRESS_OPCODE_N_INTERACTION_TRACE_COLUMNS 37

/**
 * CUDA kernel wrapper for blake_compress_opcode trace generation.
 *
 * This opcode is significantly more complex than other opcodes:
 * - 174 trace columns (vs 9 for assert_eq_opcode_imm)
 * - 73 lookup data groups (vs 5)
 * - 80 sub-component lookups across 8 types (vs 3)
 * - 37 interaction columns (vs 3)
 *
 * Sub-components:
 * - verify_instruction: 1 lookup × 7 fields
 * - memory_address_to_id: 20 lookups × 2 fields
 * - memory_id_to_big: 20 lookups × 29 fields
 * - range_check_7_2_5: 17 lookups × 3 fields
 * - verify_bitwise_xor_8: 4 lookups × 3 fields
 * - blake_round: 10 lookups × 35 fields
 * - triple_xor_32: 8 lookups × 8 fields
 * - opcodes: 2 lookups × 3 fields
 */

extern "C"
void generate_blake_compress_opcode_traces(
    // Output: 174 trace columns
    unsigned **traces,

    // Lookup data - blake_round (2 lookups × 35 fields = 70 BaseFieldVec)
    unsigned **lookup_blake_round_0,
    unsigned **lookup_blake_round_1,

    // Lookup data - memory_address_to_id (20 lookups × 2 fields = 40 BaseFieldVec)
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

    // Lookup data - memory_id_to_big (20 lookups × 29 fields = 580 BaseFieldVec)
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

    // Lookup data - opcodes (2 lookups × 3 fields = 6 BaseFieldVec)
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    // Lookup data - range_check_7_2_5 (17 lookups × 3 fields = 51 BaseFieldVec)
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

    // Lookup data - triple_xor_32 (8 lookups × 8 fields = 64 BaseFieldVec)
    unsigned **lookup_triple_xor_32_0,
    unsigned **lookup_triple_xor_32_1,
    unsigned **lookup_triple_xor_32_2,
    unsigned **lookup_triple_xor_32_3,
    unsigned **lookup_triple_xor_32_4,
    unsigned **lookup_triple_xor_32_5,
    unsigned **lookup_triple_xor_32_6,
    unsigned **lookup_triple_xor_32_7,

    // Lookup data - verify_bitwise_xor_8 (4 lookups × 3 fields = 12 BaseFieldVec)
    unsigned **lookup_verify_bitwise_xor_8_0,
    unsigned **lookup_verify_bitwise_xor_8_1,
    unsigned **lookup_verify_bitwise_xor_8_2,
    unsigned **lookup_verify_bitwise_xor_8_3,

    // Lookup data - verify_instruction (1 lookup × 7 fields = 7 BaseFieldVec)
    unsigned **lookup_verify_instruction_0,

    // Sub-component inputs - verify_instruction (1 × 7 fields)
    unsigned **sub_component_inputs_verify_instruction,
    // Sub-component inputs - memory_address_to_id (20 × 1 field)
    unsigned **sub_component_inputs_memory_address_to_id,
    // Sub-component inputs - memory_id_to_big (20 × 1 field)
    unsigned **sub_component_inputs_memory_id_to_big,
    // Sub-component inputs - range_check_7_2_5 (17 × 3 fields)
    unsigned **sub_component_inputs_range_check_7_2_5,
    // Sub-component inputs - verify_bitwise_xor_8 (4 × 3 fields)
    unsigned **sub_component_inputs_verify_bitwise_xor_8,
    // Sub-component inputs - blake_round (10 × 19 fields)
    unsigned **sub_component_inputs_blake_round,
    // Sub-component inputs - triple_xor_32 (8 × 3 fields)
    unsigned **sub_component_inputs_triple_xor_32,

    // Opcode inputs (pc, ap, fp)
    unsigned **blake_compress_opcode_input,

    // Memory lookup tables
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_blake_compress_opcode_interaction_traces(
    // Relation pointers (cast from Rust relation structs)
    void *blake_round,
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *range_check_7_2_5,
    void *triple_xor_32,
    void *verify_bitwise_xor_8,
    void *verify_instruction,

    // Lookup data - blake_round (2 lookups)
    unsigned **lookup_blake_round_0,
    unsigned **lookup_blake_round_1,

    // Lookup data - memory_address_to_id (20 lookups)
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

    // Lookup data - memory_id_to_big (20 lookups)
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

    // Lookup data - opcodes (2 lookups)
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    // Lookup data - range_check_7_2_5 (17 lookups)
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

    // Lookup data - triple_xor_32 (8 lookups)
    unsigned **lookup_triple_xor_32_0,
    unsigned **lookup_triple_xor_32_1,
    unsigned **lookup_triple_xor_32_2,
    unsigned **lookup_triple_xor_32_3,
    unsigned **lookup_triple_xor_32_4,
    unsigned **lookup_triple_xor_32_5,
    unsigned **lookup_triple_xor_32_6,
    unsigned **lookup_triple_xor_32_7,

    // Lookup data - verify_bitwise_xor_8 (4 lookups)
    unsigned **lookup_verify_bitwise_xor_8_0,
    unsigned **lookup_verify_bitwise_xor_8_1,
    unsigned **lookup_verify_bitwise_xor_8_2,
    unsigned **lookup_verify_bitwise_xor_8_3,

    // Lookup data - verify_instruction (1 lookup)
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    // Output: 37 × 4 interaction trace columns (QM31)
    unsigned **interaction_trace,
    // Output: claimed_sum as 4 × M31
    unsigned *claimed_sum
);

#endif // GEN_BLAKE_COMPRESS_OPCODE_TRACE_H
