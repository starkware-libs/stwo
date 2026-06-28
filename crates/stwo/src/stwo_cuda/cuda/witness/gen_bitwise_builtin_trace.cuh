
#ifndef GEN_BITWISE_BUILTIN_TRACE_H
#define GEN_BITWISE_BUILTIN_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define BITWISE_BUILTIN_N_TRACE_COLUMNS 89
// 19 logical pairs, each stores 4 m31 values (one QM31 accumulated value)
// Total m31 columns = 19 * 4 = 76
#define BITWISE_BUILTIN_N_INTERACTION_TRACE_COLUMNS 19
#define BITWISE_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX 256

extern "C"
void generate_bitwise_builtin_traces(
    unsigned **traces,

    // Lookup data arrays - 5 MemoryAddressToId lookups (2 elements each)
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_address_to_id_3,
    unsigned **lookup_memory_address_to_id_4,

    // Lookup data arrays - 5 MemoryIdToBig lookups (29 elements each)
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_memory_id_to_big_3,
    unsigned **lookup_memory_id_to_big_4,

    // Lookup data arrays - 27 VerifyBitwiseXor9 lookups (3 elements each)
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

    // Lookup data arrays - 1 VerifyBitwiseXor8 lookup (3 elements each)
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
);

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
);

#endif // GEN_BITWISE_BUILTIN_TRACE_H
