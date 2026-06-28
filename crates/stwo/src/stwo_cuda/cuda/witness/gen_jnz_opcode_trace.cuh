
#ifndef GEN_JNZ_OPCODE_TRACE_H
#define GEN_JNZ_OPCODE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define JNZ_OPCODE_N_TRACE_COLUMNS 37
#define JNZ_OPCODE_N_INTERACTION_TRACE_COLUMNS 3
#define JNZ_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

extern "C"
void generate_jnz_opcode_traces(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_verify_instruction_0,

    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,

    unsigned **jnz_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_jnz_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *verify_instruction,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
);

#endif // GEN_JNZ_OPCODE_TRACE_H
