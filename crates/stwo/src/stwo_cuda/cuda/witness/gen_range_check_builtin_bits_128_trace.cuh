
#ifndef GEN_RANGE_CHECK_BUILTIN_BITS_128_TRACE_H
#define GEN_RANGE_CHECK_BUILTIN_BITS_128_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// range_check_builtin_bits_128 trace structure:
// - 17 trace columns (1 id + 15 limbs + 1 partial_limb_msb for 128-bit value)
// - 1 MemoryAddressToId lookup (2 elements)
// - 1 MemoryIdToBig lookup (29 elements)
// - NO RangeCheck_6 lookup (uses partial_limb_msb instead)
// - 1 interaction trace logical column (4 M31 columns total)

#define RANGE_CHECK_128_BUILTIN_N_TRACE_COLUMNS 17
// 1 logical column, stores 4 m31 values (one QM31 accumulated value)
// Total m31 columns = 1 * 4 = 4
#define RANGE_CHECK_128_BUILTIN_N_INTERACTION_TRACE_COLUMNS 1
#define RANGE_CHECK_128_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX 256

extern "C"
void generate_range_check_builtin_bits_128_traces(
    unsigned **traces,

    // Lookup data arrays - 1 MemoryAddressToId lookup (2 elements)
    unsigned **lookup_memory_address_to_id_0,

    // Lookup data arrays - 1 MemoryIdToBig lookup (29 elements)
    unsigned **lookup_memory_id_to_big_0,

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
void generate_range_check_builtin_bits_128_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,

    // Lookup data arrays - 1 MemoryAddressToId lookup
    unsigned **lookup_memory_address_to_id_0,

    // Lookup data arrays - 1 MemoryIdToBig lookup
    unsigned **lookup_memory_id_to_big_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
);

#endif // GEN_RANGE_CHECK_BUILTIN_BITS_128_TRACE_H
