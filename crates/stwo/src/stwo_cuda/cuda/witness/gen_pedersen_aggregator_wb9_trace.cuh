#ifndef GEN_PEDERSEN_AGGREGATOR_WB9_TRACE_CUH
#define GEN_PEDERSEN_AGGREGATOR_WB9_TRACE_CUH

/**
 * CUDA trace generation for pedersen_aggregator_window_bits_9 (234-col trace).
 *
 * Generates the pedersen aggregator base trace directly on GPU, using the
 * GPU-resident pedersen table. This eliminates the need for the CPU
 * PEDERSEN_TABLE_9 in the CUDA proving path.
 *
 * Column layout:
 *   0-2:     Input limbs (3 cols)
 *   3-30:    memory_id_to_big(input_0) -> value_a (28 M31 limbs)
 *   31-58:   memory_id_to_big(input_1) -> value_b (28 M31 limbs)
 *   59-61:   Verify Reduced 252 for value A (3 cols)
 *   62-64:   Verify Reduced 252 for value B (3 cols)
 *   65-148:  PEM chain 0 output (28 m_shifted_zeros + 28 result_x + 28 result_y)
 *   149-232: PEM chain 1 output (28 m_shifted_zeros + 28 result_x + 28 result_y)
 *   233:     Multiplicity
 *
 * Sub-component feeds:
 *   - memory_id_to_big: 3 feeds x 1 column
 *   - range_check_8: 4 feeds x 1 column
 *   - partial_ec_mul_window_bits_9: 56 feeds x 86 columns
 *
 * Lookup data:
 *   - memory_id_to_big: 3 x 30 elements
 *   - range_check_8: 4 x 2 elements
 *   - partial_ec_mul_window_bits_9: 4 x 87 elements
 *   - pedersen_aggregator_window_bits_9: 1 x 4 elements (self-lookup)
 */

#include <cstdint>
#include "../fields.cuh"

// Number of trace columns
#define AGG9_N_TRACE_COLUMNS 234

// Number of input columns
#define AGG9_N_INPUT_COLUMNS 3

// Number of logup columns for interaction trace (6 logup x 4 BaseField)
#define AGG9_N_LOGUP_COLUMNS 6

/**
 * Merged trace generation kernel.
 * Generates 234 trace columns, all lookup data arrays, and all sub-component
 * input arrays in a single pass.
 */
extern "C" void gen_pedersen_aggregator_wb9_trace(
    m31** traces,                    // 234 trace output columns
    // Lookup data
    m31** lk_mem_0,                  // 30 arrays (memory_id_to_big #0)
    m31** lk_mem_1,                  // 30 arrays (memory_id_to_big #1)
    m31** lk_mem_2,                  // 30 arrays (memory_id_to_big #2)
    m31** lk_rc8_0,                  // 2 arrays (range_check_8 #0)
    m31** lk_rc8_1,                  // 2 arrays (range_check_8 #1)
    m31** lk_rc8_2,                  // 2 arrays (range_check_8 #2)
    m31** lk_rc8_3,                  // 2 arrays (range_check_8 #3)
    m31** lk_pem_0,                  // 87 arrays (partial_ec_mul #0: chain 0 input)
    m31** lk_pem_1,                  // 87 arrays (partial_ec_mul #1: chain 0 output)
    m31** lk_pem_2,                  // 87 arrays (partial_ec_mul #2: chain 1 input)
    m31** lk_pem_3,                  // 87 arrays (partial_ec_mul #3: chain 1 output)
    m31** lk_agg_0,                  // 4 arrays (self-lookup)
    m31* mults,                      // multiplicity data (n_rows elements)
    // Sub-component inputs
    m31** sub_mem,                   // 3 arrays
    m31** sub_rc8,                   // 4 arrays
    m31** sub_pem,                   // 86 arrays, each 56*trace_size (flattened)
    // Inputs
    m31** inputs,                    // 3 input columns
    // Memory state pointers
    unsigned** transpose_big_value_ptr,
    unsigned* small_value_ptr,
    // Sizes
    uint32_t n_rows,                 // Number of valid (non-padding) rows
    uint32_t log_size                // Log2 of padded trace size
);

/**
 * Interaction trace generation kernel.
 * Uses lookup data arrays (already on GPU) to compute 6 logup columns.
 */
extern "C" void gen_pedersen_aggregator_wb9_interaction_trace(
    // CommonLookupElements (= LookupElements<128>)
    void* lookup_elements,
    // Lookup data (all device pointers)
    m31** lk_mem_0,         // 30 arrays
    m31** lk_mem_1,         // 30 arrays
    m31** lk_mem_2,         // 30 arrays
    m31** lk_rc8_0,         // 2 arrays
    m31** lk_rc8_1,         // 2 arrays
    m31** lk_rc8_2,         // 2 arrays
    m31** lk_rc8_3,         // 2 arrays
    m31** lk_pem_0,         // 87 arrays
    m31** lk_pem_1,         // 87 arrays
    m31** lk_pem_2,         // 87 arrays
    m31** lk_pem_3,         // 87 arrays
    m31** lk_agg_0,         // 4 arrays
    m31* mults,             // multiplicities (device pointer)
    // Sizes
    uint32_t log_size,
    // Output
    m31** interaction_trace_columns,   // 4 * 6 = 24 columns
    m31* claimed_sum                   // 4 m31s for qm31
);

#endif // GEN_PEDERSEN_AGGREGATOR_WB9_TRACE_CUH
