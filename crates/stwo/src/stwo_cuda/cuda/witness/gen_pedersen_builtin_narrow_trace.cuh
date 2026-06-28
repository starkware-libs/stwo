#ifndef GEN_PEDERSEN_BUILTIN_NARROW_TRACE_CUH
#define GEN_PEDERSEN_BUILTIN_NARROW_TRACE_CUH

#include "../fields.cuh"
#include <cstdint>

// Base trace generation: 3 output columns (narrow variant, window_bits_9)
extern "C" void gen_pedersen_builtin_narrow_trace(
    m31** traces,            // 3 output columns
    m31** lk_mem_0,          // 3 arrays
    m31** lk_mem_1,          // 3 arrays
    m31** lk_mem_2,          // 3 arrays
    m31** lk_agg_0,          // 4 arrays
    m31** sub_mem,           // 3 arrays
    m31** sub_agg,           // 3 arrays
    unsigned* address_to_raw_id,
    uint32_t segment_start,
    uint32_t n_rows,
    uint32_t log_size
);

// Interaction trace generation: 2 logup columns (8 M31 columns)
extern "C" void gen_pedersen_builtin_narrow_interaction_trace(
    void* lookup_elements,
    m31** lk_mem_0,          // 3 arrays
    m31** lk_mem_1,          // 3 arrays
    m31** lk_mem_2,          // 3 arrays
    m31** lk_agg_0,          // 4 arrays
    uint32_t log_size,
    m31** interaction_trace_columns,   // 8 columns (4*2)
    m31* claimed_sum                   // 4 m31s for qm31
);

#endif // GEN_PEDERSEN_BUILTIN_NARROW_TRACE_CUH
