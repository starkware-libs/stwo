
#ifndef GEN_ADD_AP_OPCODE_TRACE_H
#define GEN_ADD_AP_OPCODE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define ADD_AP_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

typedef struct Enabler {
    unsigned padding_offset;

    HOST_DEVICE_FORCEINLINE Enabler(unsigned padding_offset) : padding_offset(padding_offset) {}


}Enabler;


extern "C"
void generate_add_ap_opcode_traces(
    unsigned **traces,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_11_0,
    unsigned **lookup_range_check_18_0,
    unsigned **lookup_verify_instruction_0,


    unsigned **sub_component_inputs_verify_instruction,
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,


    unsigned **opcodes_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_add_ap_opcode_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *opcodes,
    void *range_check_11,
    void *range_check_18,
    void *verify_instruction,

    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_opcodes_0,
    unsigned **lookup_opcodes_1,
    unsigned **lookup_range_check_11_0,
    unsigned **lookup_range_check_18_0,
    unsigned **lookup_verify_instruction_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
);
#endif // GEN_ADD_AP_OPCODE_TRACE_H