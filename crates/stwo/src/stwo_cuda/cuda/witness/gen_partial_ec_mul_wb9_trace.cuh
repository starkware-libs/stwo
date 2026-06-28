#ifndef GEN_PARTIAL_EC_MUL_WB9_TRACE_CUH
#define GEN_PARTIAL_EC_MUL_WB9_TRACE_CUH

/**
 * CUDA trace generation for partial_ec_mul_window_bits_9 (311-col "now" architecture).
 *
 * This is the AIR-compatible version with 311 trace columns, 65 logup columns,
 * and sub-component feeds to pedersen_points_table, rc_9_9, and rc_20.
 *
 * Column layout:
 *   0-85:    Input (86 cols: 2 + 28 + 28 + 28)
 *   86-113:  Pedersen points table output x (28 limbs)
 *   114-141: Pedersen points table output y (28 limbs)
 *   142-169: EC slope (28 limbs)
 *   170:     VerifyMul #1 k
 *   171-197: VerifyMul #1 carries (27)
 *   198-225: EC result_x (28 limbs)
 *   226:     VerifyMul #2 k
 *   227-253: VerifyMul #2 carries (27)
 *   254-281: EC result_y (28 limbs)
 *   282:     VerifyMul #3 k
 *   283-309: VerifyMul #3 carries (27)
 *   310:     Enabler
 */

#include <cstdint>
#include "../fields.cuh"

// Number of trace columns
#define PEM_WB9_N_TRACE_COLUMNS 311

// Number of input columns
#define PEM_WB9_N_INPUT_COLUMNS 86

// Number of logup columns for interaction trace
#define PEM_WB9_N_LOGUP_COLUMNS 65

// Sub-component feed counts per rc_20 variant [a,b,c,d,e,f,g,h]
// Total: 12+12+12+12+9+9+9+9 = 84
static const int RC_20_COUNTS[8] = {12, 12, 12, 12, 9, 9, 9, 9};

// Sub-component feed counts per rc_9_9 variant [a,b,c,d,e,f,g,h]
// Total: 6+6+6+6+6+6+3+3 = 42
static const int RC_9_9_COUNTS[8] = {6, 6, 6, 6, 6, 6, 3, 3};

/**
 * Merged trace generation kernel.
 * Generates 311 trace columns, all lookup data arrays, and all sub-component
 * input arrays in a single pass.
 */
extern "C" void gen_partial_ec_mul_wb9_trace(
    m31** traces,                       // 311 trace output columns
    // Lookup data - self-interaction
    m31** lookup_partial_ec_mul_0,      // 87 arrays
    m31** lookup_partial_ec_mul_1,      // 87 arrays
    // Lookup data - pedersen_points_table
    m31** lookup_ppt_0,                 // 58 arrays
    // Lookup data - rc_20 variants (2 elements per entry, flat)
    m31** lookup_rc_20,                 // 12*2=24
    m31** lookup_rc_20_b,               // 12*2=24
    m31** lookup_rc_20_c,               // 12*2=24
    m31** lookup_rc_20_d,               // 12*2=24
    m31** lookup_rc_20_e,               // 9*2=18
    m31** lookup_rc_20_f,               // 9*2=18
    m31** lookup_rc_20_g,               // 9*2=18
    m31** lookup_rc_20_h,               // 9*2=18
    // Lookup data - rc_9_9 variants (3 elements per entry, flat)
    m31** lookup_rc_9_9,                // 6*3=18
    m31** lookup_rc_9_9_b,              // 6*3=18
    m31** lookup_rc_9_9_c,              // 6*3=18
    m31** lookup_rc_9_9_d,              // 6*3=18
    m31** lookup_rc_9_9_e,              // 6*3=18
    m31** lookup_rc_9_9_f,              // 6*3=18
    m31** lookup_rc_9_9_g,              // 3*3=9
    m31** lookup_rc_9_9_h,              // 3*3=9
    // Sub-component inputs - PPT
    m31** sub_inputs_ppt,               // 1*1=1
    // Sub-component inputs - rc_9_9 variants (2 cols per feed)
    m31** sub_inputs_rc_9_9,            // 6*2=12
    m31** sub_inputs_rc_9_9_b,          // 6*2=12
    m31** sub_inputs_rc_9_9_c,          // 6*2=12
    m31** sub_inputs_rc_9_9_d,          // 6*2=12
    m31** sub_inputs_rc_9_9_e,          // 6*2=12
    m31** sub_inputs_rc_9_9_f,          // 6*2=12
    m31** sub_inputs_rc_9_9_g,          // 3*2=6
    m31** sub_inputs_rc_9_9_h,          // 3*2=6
    // Sub-component inputs - rc_20 variants (1 col per feed)
    m31** sub_inputs_rc_20,             // 12*1=12
    m31** sub_inputs_rc_20_b,           // 12*1=12
    m31** sub_inputs_rc_20_c,           // 12*1=12
    m31** sub_inputs_rc_20_d,           // 12*1=12
    m31** sub_inputs_rc_20_e,           // 9*1=9
    m31** sub_inputs_rc_20_f,           // 9*1=9
    m31** sub_inputs_rc_20_g,           // 9*1=9
    m31** sub_inputs_rc_20_h,           // 9*1=9
    // Inputs
    m31** inputs,                       // 86 input columns
    uint32_t n_rows,                    // Number of valid rows
    uint32_t log_size                   // Log2 of trace size
);

/**
 * Interaction trace generation kernel.
 * Uses lookup data arrays to compute 65 logup columns.
 */
extern "C" void gen_partial_ec_mul_wb9_interaction_trace(
    // Lookup elements (per-relation, modified for CUDA)
    void* partial_ec_mul_lookup_elements,
    void* pedersen_points_table_lookup_elements,
    void* rc_20_lookup_elements,
    void* rc_20_b_lookup_elements,
    void* rc_20_c_lookup_elements,
    void* rc_20_d_lookup_elements,
    void* rc_20_e_lookup_elements,
    void* rc_20_f_lookup_elements,
    void* rc_20_g_lookup_elements,
    void* rc_20_h_lookup_elements,
    void* rc_9_9_lookup_elements,
    void* rc_9_9_b_lookup_elements,
    void* rc_9_9_c_lookup_elements,
    void* rc_9_9_d_lookup_elements,
    void* rc_9_9_e_lookup_elements,
    void* rc_9_9_f_lookup_elements,
    void* rc_9_9_g_lookup_elements,
    void* rc_9_9_h_lookup_elements,
    // Lookup data pointers
    m31** lookup_partial_ec_mul_0,      // 87 arrays
    m31** lookup_partial_ec_mul_1,      // 87 arrays
    m31** lookup_ppt_0,                 // 58 arrays
    m31** lookup_rc_20,                 // 24 flat ptrs
    m31** lookup_rc_20_b,               // 24
    m31** lookup_rc_20_c,               // 24
    m31** lookup_rc_20_d,               // 24
    m31** lookup_rc_20_e,               // 18
    m31** lookup_rc_20_f,               // 18
    m31** lookup_rc_20_g,               // 18
    m31** lookup_rc_20_h,               // 18
    m31** lookup_rc_9_9,                // 18 flat ptrs
    m31** lookup_rc_9_9_b,              // 18
    m31** lookup_rc_9_9_c,              // 18
    m31** lookup_rc_9_9_d,              // 18
    m31** lookup_rc_9_9_e,              // 18
    m31** lookup_rc_9_9_f,              // 18
    m31** lookup_rc_9_9_g,              // 9
    m31** lookup_rc_9_9_h,              // 9
    // Sizes
    uint32_t n_rows,
    uint32_t log_size,
    // Output
    m31** interaction_trace_columns,    // 4*65 = 260 cols
    m31* claimed_sum                    // 4 u32s for qm31
);

#endif // GEN_PARTIAL_EC_MUL_WB9_TRACE_CUH
