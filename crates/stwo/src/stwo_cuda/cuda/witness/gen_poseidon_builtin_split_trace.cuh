#ifndef GEN_POSEIDON_BUILTIN_SPLIT_TRACE_CUH
#define GEN_POSEIDON_BUILTIN_SPLIT_TRACE_CUH

#include "../fields.cuh"
#include <cstdint>

// Base trace generation: 6 output columns (split poseidon_builtin)
extern "C" void gen_poseidon_builtin_split_trace(
    m31** traces,            // 6 output columns
    m31** lk_mem_0,          // 3 arrays
    m31** lk_mem_1,          // 3 arrays
    m31** lk_mem_2,          // 3 arrays
    m31** lk_mem_3,          // 3 arrays
    m31** lk_mem_4,          // 3 arrays
    m31** lk_mem_5,          // 3 arrays
    m31** lk_agg_0,          // 7 arrays
    m31** sub_mem,           // 6 arrays
    m31** sub_agg,           // 6 arrays (3 input IDs + 3 output IDs)
    unsigned* address_to_raw_id,
    uint32_t segment_start,
    uint32_t n_rows,
    uint32_t log_size
);

// Interaction trace generation: 4 logup columns (16 M31 columns)
extern "C" void gen_poseidon_builtin_split_interaction_trace(
    void* lookup_elements,
    m31** lk_mem_0,          // 3 arrays
    m31** lk_mem_1,          // 3 arrays
    m31** lk_mem_2,          // 3 arrays
    m31** lk_mem_3,          // 3 arrays
    m31** lk_mem_4,          // 3 arrays
    m31** lk_mem_5,          // 3 arrays
    m31** lk_agg_0,          // 7 arrays
    uint32_t log_size,
    m31** interaction_trace_columns,   // 16 columns (4*4)
    m31* claimed_sum                   // 4 m31s for qm31
);

#endif // GEN_POSEIDON_BUILTIN_SPLIT_TRACE_CUH
