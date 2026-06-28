
#ifndef GEN_RANGE_CHECK_BUILTIN_BITS_96_TRACE_H
#define GEN_RANGE_CHECK_BUILTIN_BITS_96_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// range_check_builtin_bits_96 trace structure:
// - 12 trace columns (1 id + 11 limbs for 96-bit value)
// - 1 MemoryAddressToId lookup (2 elements)
// - 1 MemoryIdToBig lookup (29 elements)
// - 1 RangeCheck_6 lookup (1 element)
// - 2 interaction trace logical columns (8 M31 columns total)

#define RANGE_CHECK_96_BUILTIN_N_TRACE_COLUMNS 12
// 2 logical pairs, each stores 4 m31 values (one QM31 accumulated value)
// Total m31 columns = 2 * 4 = 8
#define RANGE_CHECK_96_BUILTIN_N_INTERACTION_TRACE_COLUMNS 2
#define RANGE_CHECK_96_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX 256

extern "C"
void generate_range_check_builtin_bits_96_traces(
    unsigned **traces,

    // Lookup data arrays - 1 MemoryAddressToId lookup (2 elements)
    unsigned **lookup_memory_address_to_id_0,

    // Lookup data arrays - 1 MemoryIdToBig lookup (29 elements)
    unsigned **lookup_memory_id_to_big_0,

    // Lookup data arrays - 1 RangeCheck_6 lookup (1 element)
    unsigned **lookup_range_check_6_0,

    // Sub-component inputs
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_6,

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
void generate_range_check_builtin_bits_96_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *range_check_6,

    // Lookup data arrays - 1 MemoryAddressToId lookup
    unsigned **lookup_memory_address_to_id_0,

    // Lookup data arrays - 1 MemoryIdToBig lookup
    unsigned **lookup_memory_id_to_big_0,

    // Lookup data arrays - 1 RangeCheck_6 lookup
    unsigned **lookup_range_check_6_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
);

#endif // GEN_RANGE_CHECK_BUILTIN_BITS_96_TRACE_H
