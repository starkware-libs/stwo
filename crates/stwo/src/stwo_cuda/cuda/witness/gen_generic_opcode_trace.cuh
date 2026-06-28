
#ifndef GEN_GENERIC_OPCODE_TRACE_H
#define GEN_GENERIC_OPCODE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define GENERIC_OPCODE_N_TRACE_COLUMNS 244
#define GENERIC_OPCODE_N_INTERACTION_TRACE_COLUMNS 34
#define GENERIC_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

/**
 * CUDA kernel wrapper for generic_opcode trace generation.
 *
 * This is the most complex opcode in the Cairo VM:
 * - 244 trace columns
 * - 67 lookup data groups
 * - 34 interaction columns
 *
 * Sub-components:
 * - verify_instruction: 1 lookup × 7 fields
 * - memory_address_to_id: 3 lookups × 2 fields
 * - memory_id_to_big: 3 lookups × 29 fields
 * - opcodes: 2 lookups × 3 fields
 * - range_check_9_9: 4 lookups × 2 fields
 * - range_check_9_9_b: 4 lookups × 2 fields
 * - range_check_9_9_c: 4 lookups × 2 fields
 * - range_check_9_9_d: 4 lookups × 2 fields
 * - range_check_9_9_e: 4 lookups × 2 fields
 * - range_check_9_9_f: 4 lookups × 2 fields
 * - range_check_9_9_g: 2 lookups × 2 fields
 * - range_check_9_9_h: 2 lookups × 2 fields
 * - range_check_19: 4 lookups × 1 field
 * - range_check_19_b: 4 lookups × 1 field
 * - range_check_19_c: 4 lookups × 1 field
 * - range_check_19_d: 3 lookups × 1 field
 * - range_check_19_e: 3 lookups × 1 field
 * - range_check_19_f: 3 lookups × 1 field
 * - range_check_19_g: 3 lookups × 1 field
 * - range_check_19_h: 4 lookups × 1 field
 * - range_check_18: 1 lookup × 1 field
 * - range_check_11: 1 lookup × 1 field
 */

extern "C"
void generate_generic_opcode_traces(
    // Output: 244 trace columns
    unsigned **traces,

    // Lookup data - memory_address_to_id (3 lookups × 2 fields)
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,

    // Lookup data - memory_id_to_big (3 lookups × 29 fields)
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,

    // Lookup data - opcodes (2 lookups × 3 fields)
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    // Lookup data - range_check_9_9 (4 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_0,
    unsigned **lookup_range_check_9_9_1,
    unsigned **lookup_range_check_9_9_2,
    unsigned **lookup_range_check_9_9_3,

    // Lookup data - range_check_9_9_b (4 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_b_0,
    unsigned **lookup_range_check_9_9_b_1,
    unsigned **lookup_range_check_9_9_b_2,
    unsigned **lookup_range_check_9_9_b_3,

    // Lookup data - range_check_9_9_c (4 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_c_0,
    unsigned **lookup_range_check_9_9_c_1,
    unsigned **lookup_range_check_9_9_c_2,
    unsigned **lookup_range_check_9_9_c_3,

    // Lookup data - range_check_9_9_d (4 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_d_0,
    unsigned **lookup_range_check_9_9_d_1,
    unsigned **lookup_range_check_9_9_d_2,
    unsigned **lookup_range_check_9_9_d_3,

    // Lookup data - range_check_9_9_e (4 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_e_0,
    unsigned **lookup_range_check_9_9_e_1,
    unsigned **lookup_range_check_9_9_e_2,
    unsigned **lookup_range_check_9_9_e_3,

    // Lookup data - range_check_9_9_f (4 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_f_0,
    unsigned **lookup_range_check_9_9_f_1,
    unsigned **lookup_range_check_9_9_f_2,
    unsigned **lookup_range_check_9_9_f_3,

    // Lookup data - range_check_9_9_g (2 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_g_0,
    unsigned **lookup_range_check_9_9_g_1,

    // Lookup data - range_check_9_9_h (2 lookups × 2 fields)
    unsigned **lookup_range_check_9_9_h_0,
    unsigned **lookup_range_check_9_9_h_1,

    // Lookup data - range_check_19 (4 lookups × 1 field)
    unsigned **lookup_range_check_19_0,
    unsigned **lookup_range_check_19_1,
    unsigned **lookup_range_check_19_2,
    unsigned **lookup_range_check_19_3,

    // Lookup data - range_check_19_b (4 lookups × 1 field)
    unsigned **lookup_range_check_19_b_0,
    unsigned **lookup_range_check_19_b_1,
    unsigned **lookup_range_check_19_b_2,
    unsigned **lookup_range_check_19_b_3,

    // Lookup data - range_check_19_c (4 lookups × 1 field)
    unsigned **lookup_range_check_19_c_0,
    unsigned **lookup_range_check_19_c_1,
    unsigned **lookup_range_check_19_c_2,
    unsigned **lookup_range_check_19_c_3,

    // Lookup data - range_check_19_d (3 lookups × 1 field)
    unsigned **lookup_range_check_19_d_0,
    unsigned **lookup_range_check_19_d_1,
    unsigned **lookup_range_check_19_d_2,

    // Lookup data - range_check_19_e (3 lookups × 1 field)
    unsigned **lookup_range_check_19_e_0,
    unsigned **lookup_range_check_19_e_1,
    unsigned **lookup_range_check_19_e_2,

    // Lookup data - range_check_19_f (3 lookups × 1 field)
    unsigned **lookup_range_check_19_f_0,
    unsigned **lookup_range_check_19_f_1,
    unsigned **lookup_range_check_19_f_2,

    // Lookup data - range_check_19_g (3 lookups × 1 field)
    unsigned **lookup_range_check_19_g_0,
    unsigned **lookup_range_check_19_g_1,
    unsigned **lookup_range_check_19_g_2,

    // Lookup data - range_check_19_h (4 lookups × 1 field)
    unsigned **lookup_range_check_19_h_0,
    unsigned **lookup_range_check_19_h_1,
    unsigned **lookup_range_check_19_h_2,
    unsigned **lookup_range_check_19_h_3,

    // Lookup data - range_check_18 (1 lookup × 1 field)
    unsigned **lookup_range_check_18_0,

    // Lookup data - range_check_11 (1 lookup × 1 field)
    unsigned **lookup_range_check_11_0,

    // Lookup data - verify_instruction (1 lookup × 7 fields)
    unsigned **lookup_verify_instruction_0,

    // Sub-component inputs - verify_instruction (1 × 7 fields)
    unsigned **sub_component_inputs_verify_instruction,
    // Sub-component inputs - memory_address_to_id (3 × 1 field)
    unsigned **sub_component_inputs_memory_address_to_id,
    // Sub-component inputs - memory_id_to_big (3 × 1 field)
    unsigned **sub_component_inputs_memory_id_to_big,

    // Opcode inputs (pc, ap, fp)
    unsigned **generic_opcode_input,

    // Memory lookup tables
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_generic_opcode_interaction_traces(
    // Relation pointers
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *range_check_9_9,
    void *range_check_9_9_b,
    void *range_check_9_9_c,
    void *range_check_9_9_d,
    void *range_check_9_9_e,
    void *range_check_9_9_f,
    void *range_check_9_9_g,
    void *range_check_9_9_h,
    void *range_check_19,
    void *range_check_19_b,
    void *range_check_19_c,
    void *range_check_19_d,
    void *range_check_19_e,
    void *range_check_19_f,
    void *range_check_19_g,
    void *range_check_19_h,
    void *range_check_18,
    void *range_check_11,
    void *verify_instruction,

    // Lookup data - memory_address_to_id (3 lookups)
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,

    // Lookup data - memory_id_to_big (3 lookups)
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,

    // Lookup data - opcodes (2 lookups)
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,

    // Lookup data - range_check_9_9 (4 lookups)
    unsigned **lookup_range_check_9_9_0,
    unsigned **lookup_range_check_9_9_1,
    unsigned **lookup_range_check_9_9_2,
    unsigned **lookup_range_check_9_9_3,

    // Lookup data - range_check_9_9_b (4 lookups)
    unsigned **lookup_range_check_9_9_b_0,
    unsigned **lookup_range_check_9_9_b_1,
    unsigned **lookup_range_check_9_9_b_2,
    unsigned **lookup_range_check_9_9_b_3,

    // Lookup data - range_check_9_9_c (4 lookups)
    unsigned **lookup_range_check_9_9_c_0,
    unsigned **lookup_range_check_9_9_c_1,
    unsigned **lookup_range_check_9_9_c_2,
    unsigned **lookup_range_check_9_9_c_3,

    // Lookup data - range_check_9_9_d (4 lookups)
    unsigned **lookup_range_check_9_9_d_0,
    unsigned **lookup_range_check_9_9_d_1,
    unsigned **lookup_range_check_9_9_d_2,
    unsigned **lookup_range_check_9_9_d_3,

    // Lookup data - range_check_9_9_e (4 lookups)
    unsigned **lookup_range_check_9_9_e_0,
    unsigned **lookup_range_check_9_9_e_1,
    unsigned **lookup_range_check_9_9_e_2,
    unsigned **lookup_range_check_9_9_e_3,

    // Lookup data - range_check_9_9_f (4 lookups)
    unsigned **lookup_range_check_9_9_f_0,
    unsigned **lookup_range_check_9_9_f_1,
    unsigned **lookup_range_check_9_9_f_2,
    unsigned **lookup_range_check_9_9_f_3,

    // Lookup data - range_check_9_9_g (2 lookups)
    unsigned **lookup_range_check_9_9_g_0,
    unsigned **lookup_range_check_9_9_g_1,

    // Lookup data - range_check_9_9_h (2 lookups)
    unsigned **lookup_range_check_9_9_h_0,
    unsigned **lookup_range_check_9_9_h_1,

    // Lookup data - range_check_19 (4 lookups)
    unsigned **lookup_range_check_19_0,
    unsigned **lookup_range_check_19_1,
    unsigned **lookup_range_check_19_2,
    unsigned **lookup_range_check_19_3,

    // Lookup data - range_check_19_b (4 lookups)
    unsigned **lookup_range_check_19_b_0,
    unsigned **lookup_range_check_19_b_1,
    unsigned **lookup_range_check_19_b_2,
    unsigned **lookup_range_check_19_b_3,

    // Lookup data - range_check_19_c (4 lookups)
    unsigned **lookup_range_check_19_c_0,
    unsigned **lookup_range_check_19_c_1,
    unsigned **lookup_range_check_19_c_2,
    unsigned **lookup_range_check_19_c_3,

    // Lookup data - range_check_19_d (3 lookups)
    unsigned **lookup_range_check_19_d_0,
    unsigned **lookup_range_check_19_d_1,
    unsigned **lookup_range_check_19_d_2,

    // Lookup data - range_check_19_e (3 lookups)
    unsigned **lookup_range_check_19_e_0,
    unsigned **lookup_range_check_19_e_1,
    unsigned **lookup_range_check_19_e_2,

    // Lookup data - range_check_19_f (3 lookups)
    unsigned **lookup_range_check_19_f_0,
    unsigned **lookup_range_check_19_f_1,
    unsigned **lookup_range_check_19_f_2,

    // Lookup data - range_check_19_g (3 lookups)
    unsigned **lookup_range_check_19_g_0,
    unsigned **lookup_range_check_19_g_1,
    unsigned **lookup_range_check_19_g_2,

    // Lookup data - range_check_19_h (4 lookups)
    unsigned **lookup_range_check_19_h_0,
    unsigned **lookup_range_check_19_h_1,
    unsigned **lookup_range_check_19_h_2,
    unsigned **lookup_range_check_19_h_3,

    // Lookup data - range_check_18 (1 lookup)
    unsigned **lookup_range_check_18_0,

    // Lookup data - range_check_11 (1 lookup)
    unsigned **lookup_range_check_11_0,

    // Lookup data - verify_instruction (1 lookup)
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    // Output: 34 × 4 interaction trace columns (QM31)
    unsigned **interaction_trace,
    // Output: claimed_sum as 4 × M31
    unsigned *claimed_sum
);

#endif // GEN_GENERIC_OPCODE_TRACE_H
