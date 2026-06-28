
#ifndef GEN_MUL_OPCODE_TRACE_H
#define GEN_MUL_OPCODE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define MUL_OPCODE_N_TRACE_COLUMNS 130
#define MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS 19
#define MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

extern "C"
void generate_mul_opcode_traces(
    unsigned **traces,

    // Lookup data arrays
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_19_0,
    unsigned **lookup_range_check_19_1,
    unsigned **lookup_range_check_19_2,
    unsigned **lookup_range_check_19_3,
    unsigned **lookup_range_check_19_b_0,
    unsigned **lookup_range_check_19_b_1,
    unsigned **lookup_range_check_19_b_2,
    unsigned **lookup_range_check_19_b_3,
    unsigned **lookup_range_check_19_c_0,
    unsigned **lookup_range_check_19_c_1,
    unsigned **lookup_range_check_19_c_2,
    unsigned **lookup_range_check_19_c_3,
    unsigned **lookup_range_check_19_d_0,
    unsigned **lookup_range_check_19_d_1,
    unsigned **lookup_range_check_19_d_2,
    unsigned **lookup_range_check_19_e_0,
    unsigned **lookup_range_check_19_e_1,
    unsigned **lookup_range_check_19_e_2,
    unsigned **lookup_range_check_19_f_0,
    unsigned **lookup_range_check_19_f_1,
    unsigned **lookup_range_check_19_f_2,
    unsigned **lookup_range_check_19_g_0,
    unsigned **lookup_range_check_19_g_1,
    unsigned **lookup_range_check_19_g_2,
    unsigned **lookup_range_check_19_h_0,
    unsigned **lookup_range_check_19_h_1,
    unsigned **lookup_range_check_19_h_2,
    unsigned **lookup_range_check_19_h_3,
    unsigned **lookup_verify_instruction_0,

    // Sub-component inputs
    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_19,
    unsigned **sub_component_inputs_range_check_19_b,
    unsigned **sub_component_inputs_range_check_19_c,
    unsigned **sub_component_inputs_range_check_19_d,
    unsigned **sub_component_inputs_range_check_19_e,
    unsigned **sub_component_inputs_range_check_19_f,
    unsigned **sub_component_inputs_range_check_19_g,
    unsigned **sub_component_inputs_range_check_19_h,

    unsigned **mul_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_mul_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,
    void *range_check_19,
    void *range_check_19_b,
    void *range_check_19_c,
    void *range_check_19_d,
    void *range_check_19_e,
    void *range_check_19_f,
    void *range_check_19_g,
    void *range_check_19_h,

    // Lookup data arrays
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_19_0,
    unsigned **lookup_range_check_19_1,
    unsigned **lookup_range_check_19_2,
    unsigned **lookup_range_check_19_3,
    unsigned **lookup_range_check_19_b_0,
    unsigned **lookup_range_check_19_b_1,
    unsigned **lookup_range_check_19_b_2,
    unsigned **lookup_range_check_19_b_3,
    unsigned **lookup_range_check_19_c_0,
    unsigned **lookup_range_check_19_c_1,
    unsigned **lookup_range_check_19_c_2,
    unsigned **lookup_range_check_19_c_3,
    unsigned **lookup_range_check_19_d_0,
    unsigned **lookup_range_check_19_d_1,
    unsigned **lookup_range_check_19_d_2,
    unsigned **lookup_range_check_19_e_0,
    unsigned **lookup_range_check_19_e_1,
    unsigned **lookup_range_check_19_e_2,
    unsigned **lookup_range_check_19_f_0,
    unsigned **lookup_range_check_19_f_1,
    unsigned **lookup_range_check_19_f_2,
    unsigned **lookup_range_check_19_g_0,
    unsigned **lookup_range_check_19_g_1,
    unsigned **lookup_range_check_19_g_2,
    unsigned **lookup_range_check_19_h_0,
    unsigned **lookup_range_check_19_h_1,
    unsigned **lookup_range_check_19_h_2,
    unsigned **lookup_range_check_19_h_3,
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
);

#endif // GEN_MUL_OPCODE_TRACE_H
