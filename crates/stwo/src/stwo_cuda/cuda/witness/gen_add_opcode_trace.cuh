
#ifndef GEN_ADD_OPCODE_TRACE_H
#define GEN_ADD_OPCODE_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define ADD_OPCODE_TRACE_GEN_THREAD_COUNT_MAX 256

typedef struct Enabler {
    unsigned padding_offset;

    HOST_DEVICE_FORCEINLINE Enabler(unsigned padding_offset) : padding_offset(padding_offset) {}


}Enabler;


extern "C"
void generate_add_opcode_traces(
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


    m31 **sub_componet_input_verify_instruction,
    m31 **sub_componet_input_memory_address_to_id,
    m31 **sub_componet_input_memory_id_to_big,


    m31 **add_opcode_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned row_offset,
    unsigned trace_log_size
);

extern "C"
void generate_add_opcode_interaction_traces(
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

    unsigned row_offset,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);
#endif // GEN_ADD_OPCODE_TRACE_H