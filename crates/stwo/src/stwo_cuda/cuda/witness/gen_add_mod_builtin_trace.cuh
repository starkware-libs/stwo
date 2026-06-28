
#ifndef GEN_ADD_MOD_BUILTIN_TRACE_H
#define GEN_ADD_MOD_BUILTIN_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define ADD_MOD_BUILTIN_N_TRACE_COLUMNS 267
#define ADD_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS 27
#define ADD_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX 256

extern "C"
void generate_add_mod_builtin_traces(
    unsigned **traces,

    // Lookup data arrays - 29 MemoryAddressToId lookups (2 elements each)
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
    unsigned **lookup_memory_address_to_id_20,
    unsigned **lookup_memory_address_to_id_21,
    unsigned **lookup_memory_address_to_id_22,
    unsigned **lookup_memory_address_to_id_23,
    unsigned **lookup_memory_address_to_id_24,
    unsigned **lookup_memory_address_to_id_25,
    unsigned **lookup_memory_address_to_id_26,
    unsigned **lookup_memory_address_to_id_27,
    unsigned **lookup_memory_address_to_id_28,

    // Lookup data arrays - 24 MemoryIdToBig lookups (29 elements each)
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
    unsigned **lookup_memory_id_to_big_20,
    unsigned **lookup_memory_id_to_big_21,
    unsigned **lookup_memory_id_to_big_22,
    unsigned **lookup_memory_id_to_big_23,

    // Sub-component inputs
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_add_mod_builtin_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,

    // Lookup data arrays - 29 MemoryAddressToId lookups
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
    unsigned **lookup_memory_address_to_id_20,
    unsigned **lookup_memory_address_to_id_21,
    unsigned **lookup_memory_address_to_id_22,
    unsigned **lookup_memory_address_to_id_23,
    unsigned **lookup_memory_address_to_id_24,
    unsigned **lookup_memory_address_to_id_25,
    unsigned **lookup_memory_address_to_id_26,
    unsigned **lookup_memory_address_to_id_27,
    unsigned **lookup_memory_address_to_id_28,

    // Lookup data arrays - 24 MemoryIdToBig lookups
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
    unsigned **lookup_memory_id_to_big_20,
    unsigned **lookup_memory_id_to_big_21,
    unsigned **lookup_memory_id_to_big_22,
    unsigned **lookup_memory_id_to_big_23,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
);

#endif // GEN_ADD_MOD_BUILTIN_TRACE_H
