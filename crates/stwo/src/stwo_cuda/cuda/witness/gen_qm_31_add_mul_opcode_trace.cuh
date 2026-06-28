
#ifndef GEN_QM_31_ADD_MUL_OPCODE_TRACE_H
#define GEN_QM_31_ADD_MUL_OPCODE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define QM_31_ADD_MUL_OPCODE_N_TRACE_COLUMNS 73
#define QM_31_ADD_MUL_OPCODE_N_INTERACTION_TRACE_COLUMNS 6
#define QM_31_ADD_MUL_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

extern "C"
void generate_qm_31_add_mul_opcode_traces(
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
    unsigned **lookup_range_check_4_4_4_4_0,
    unsigned **lookup_range_check_4_4_4_4_1,
    unsigned **lookup_range_check_4_4_4_4_2,
    unsigned **lookup_verify_instruction_0,

    // Sub-component inputs
    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_4_4_4_4,

    unsigned **qm_31_add_mul_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_qm_31_add_mul_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,
    void *range_check_4_4_4_4,

    // Lookup data arrays
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_4_4_4_4_0,
    unsigned **lookup_range_check_4_4_4_4_1,
    unsigned **lookup_range_check_4_4_4_4_2,
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
);

#endif // GEN_QM_31_ADD_MUL_OPCODE_TRACE_H
