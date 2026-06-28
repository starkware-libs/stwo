#ifndef GEN_PARTIAL_EC_MUL_TRACE_CUH
#define GEN_PARTIAL_EC_MUL_TRACE_CUH

#include "fields.cuh"

// Number of trace columns for partial_ec_mul
#define PARTIAL_EC_MUL_N_TRACE_COLUMNS 472

// Number of LogUp columns for interaction trace
// Based on relation usage: 1 PartialEcMul, 1 PedersenPointsTable,
// 8 RangeCheck_9_9 variants, 8 RangeCheck_19 variants = 18 relations
// Each relation needs fraction numerator/denominator pairs
#define PARTIAL_EC_MUL_N_LOGUP_COLS 107

// Input column layout:
// Col 0: index_in_table
// Col 1: round
// Col 2: window_value
// Cols 3-16: 14 limbs for pedersen table index
// Cols 17-44: acc_x (28 x 9-bit limbs)
// Cols 45-72: acc_y (28 x 9-bit limbs)
#define PARTIAL_EC_MUL_N_INPUT_COLUMNS 73

// pedersen_points_table output: 56 limbs (point_x 28 + point_y 28)
#define PARTIAL_EC_MUL_POINT_OUTPUT_START 73
#define PARTIAL_EC_MUL_POINT_OUTPUT_COLS 56

// EC Add intermediate values start at column 129
#define PARTIAL_EC_MUL_EC_ADD_START 129

extern "C" {

// Generate base trace for partial_ec_mul component
// Inputs are 73 columns: index, round, window_value, 14 limbs, acc_x[28], acc_y[28]
// Padding rows (n_rows <= row < trace_size) are filled with first-row copy
void partial_ec_mul_generate_trace(
    m31** input_columns,            // 73 input columns
    unsigned int n_rows,            // Number of valid (non-padding) rows
    unsigned int log_size,          // Log2 of padded trace size
    m31** trace_columns             // 472 output trace columns
);

// Add inputs to sub-component multiplicities
void partial_ec_mul_add_to_multiplicities(
    m31** trace_columns,
    unsigned int n_rows,
    unsigned int log_size,  // Log2 of trace size for proper padding handling
    // PedersenPointsTable multiplicities (1 lookup per row)
    m31* pedersen_points_table_mults,
    unsigned int pedersen_points_table_log_size,
    // RangeCheck_9_9 multiplicities (8 variants)
    m31* rc_9_9_mults,
    unsigned int rc_9_9_log_size,
    m31* rc_9_9_b_mults,
    unsigned int rc_9_9_b_log_size,
    m31* rc_9_9_c_mults,
    unsigned int rc_9_9_c_log_size,
    m31* rc_9_9_d_mults,
    unsigned int rc_9_9_d_log_size,
    m31* rc_9_9_e_mults,
    unsigned int rc_9_9_e_log_size,
    m31* rc_9_9_f_mults,
    unsigned int rc_9_9_f_log_size,
    m31* rc_9_9_g_mults,
    unsigned int rc_9_9_g_log_size,
    m31* rc_9_9_h_mults,
    unsigned int rc_9_9_h_log_size,
    // RangeCheck_19 multiplicities (8 variants)
    m31* rc_19_mults,
    unsigned int rc_19_log_size,
    m31* rc_19_b_mults,
    unsigned int rc_19_b_log_size,
    m31* rc_19_c_mults,
    unsigned int rc_19_c_log_size,
    m31* rc_19_d_mults,
    unsigned int rc_19_d_log_size,
    m31* rc_19_e_mults,
    unsigned int rc_19_e_log_size,
    m31* rc_19_f_mults,
    unsigned int rc_19_f_log_size,
    m31* rc_19_g_mults,
    unsigned int rc_19_g_log_size,
    m31* rc_19_h_mults,
    unsigned int rc_19_h_log_size
);

// Generate interaction trace for partial_ec_mul component
void partial_ec_mul_generate_interaction_trace(
    m31** trace_columns,                              // Base trace (472 columns)
    unsigned int n_rows,                              // Number of valid (non-padding) rows
    unsigned int log_size,                            // Log2 of padded trace size
    // Lookup elements for each relation
    void* pedersen_points_table_lookup_elements,
    void* range_check_9_9_lookup_elements,
    void* range_check_9_9_b_lookup_elements,
    void* range_check_9_9_c_lookup_elements,
    void* range_check_9_9_d_lookup_elements,
    void* range_check_9_9_e_lookup_elements,
    void* range_check_9_9_f_lookup_elements,
    void* range_check_9_9_g_lookup_elements,
    void* range_check_9_9_h_lookup_elements,
    void* range_check_19_lookup_elements,
    void* range_check_19_b_lookup_elements,
    void* range_check_19_c_lookup_elements,
    void* range_check_19_d_lookup_elements,
    void* range_check_19_e_lookup_elements,
    void* range_check_19_f_lookup_elements,
    void* range_check_19_g_lookup_elements,
    void* range_check_19_h_lookup_elements,
    void* partial_ec_mul_lookup_elements,
    m31** interaction_trace_columns,                  // Output interaction trace
    qm31* claimed_sum                                 // Output claimed sum
);

// Add inputs to pedersen_points_table multiplicities
// Takes table indices and atomically adds 1 to the multiplicity at each index
void pedersen_points_table_add_inputs(
    m31* indices,           // Table indices (one per row)
    unsigned int n_rows,    // Number of rows
    m31* mults,             // Output multiplicities (atomically updated)
    unsigned int mults_log_size  // Log2 of multiplicities table size
);

// ============================================================================
// Merged trace generation (following blake_g pattern)
// ============================================================================

// Merged CUDA trace generation for partial_ec_mul.
// Generates trace, lookup_data, and sub_component_inputs in a single kernel call.
void generate_partial_ec_mul_trace(
    m31** traces,                               // 472 trace output columns
    // Lookup data pointers
    m31** lookup_partial_ec_mul_0,              // 73 arrays
    m31** lookup_partial_ec_mul_1,              // 73 arrays
    m31** lookup_pedersen_points_table_0,       // 57 arrays
    // Range check 19 lookup pointers (all variants)
    m31** lookup_rc_19,                         // 12 arrays
    m31** lookup_rc_19_b,                       // 12 arrays
    m31** lookup_rc_19_c,                       // 12 arrays
    m31** lookup_rc_19_d,                       // 9 arrays
    m31** lookup_rc_19_e,                       // 9 arrays
    m31** lookup_rc_19_f,                       // 9 arrays
    m31** lookup_rc_19_g,                       // 9 arrays
    m31** lookup_rc_19_h,                       // 12 arrays
    // Range check 9_9 lookup pointers (all variants)
    m31** lookup_rc_9_9,                        // 18*2 = 36 arrays
    m31** lookup_rc_9_9_b,                      // 36 arrays
    m31** lookup_rc_9_9_c,                      // 36 arrays
    m31** lookup_rc_9_9_d,                      // 36 arrays
    m31** lookup_rc_9_9_e,                      // 36 arrays
    m31** lookup_rc_9_9_f,                      // 36 arrays
    m31** lookup_rc_9_9_g,                      // 9*2 = 18 arrays
    m31** lookup_rc_9_9_h,                      // 18 arrays
    // Sub component inputs pointers
    m31** sub_inputs_ppt,                       // 1*1 arrays (pedersen_points_table)
    m31** sub_inputs_rc_9_9,                    // 18*2 = 36 arrays
    m31** sub_inputs_rc_9_9_b,                  // 36 arrays
    m31** sub_inputs_rc_9_9_c,                  // 36 arrays
    m31** sub_inputs_rc_9_9_d,                  // 36 arrays
    m31** sub_inputs_rc_9_9_e,                  // 36 arrays
    m31** sub_inputs_rc_9_9_f,                  // 36 arrays
    m31** sub_inputs_rc_9_9_g,                  // 18 arrays
    m31** sub_inputs_rc_9_9_h,                  // 18 arrays
    m31** sub_inputs_rc_19_h,                   // 12*1 = 12 arrays
    m31** sub_inputs_rc_19,                     // 12 arrays
    m31** sub_inputs_rc_19_b,                   // 12 arrays
    m31** sub_inputs_rc_19_c,                   // 12 arrays
    m31** sub_inputs_rc_19_d,                   // 9 arrays
    m31** sub_inputs_rc_19_e,                   // 9 arrays
    m31** sub_inputs_rc_19_f,                   // 9 arrays
    m31** sub_inputs_rc_19_g,                   // 9 arrays
    // Inputs
    m31** inputs,                               // 73 input columns
    unsigned int n_rows,                        // Number of valid rows
    unsigned int log_size                       // Log2 of trace size
);

// Generate interaction trace from lookup_data (instead of trace columns)
// Uses all lookup data arrays to compute the 107 LogUp columns.
void generate_partial_ec_mul_interaction_traces(
    // Lookup elements for each relation (18 relations total)
    void* pedersen_points_table_lookup_elements,
    void* range_check_9_9_lookup_elements,
    void* range_check_9_9_b_lookup_elements,
    void* range_check_9_9_c_lookup_elements,
    void* range_check_9_9_d_lookup_elements,
    void* range_check_9_9_e_lookup_elements,
    void* range_check_9_9_f_lookup_elements,
    void* range_check_9_9_g_lookup_elements,
    void* range_check_9_9_h_lookup_elements,
    void* range_check_19_lookup_elements,
    void* range_check_19_b_lookup_elements,
    void* range_check_19_c_lookup_elements,
    void* range_check_19_d_lookup_elements,
    void* range_check_19_e_lookup_elements,
    void* range_check_19_f_lookup_elements,
    void* range_check_19_g_lookup_elements,
    void* range_check_19_h_lookup_elements,
    void* partial_ec_mul_lookup_elements,
    // Lookup data pointers - main relations
    m31** lookup_partial_ec_mul_0,              // 73 arrays
    m31** lookup_partial_ec_mul_1,              // 73 arrays
    m31** lookup_pedersen_points_table_0,       // 57 arrays
    // Lookup data pointers - range_check_19 variants (1 element each)
    m31** lookup_rc_19,                         // 12 arrays
    m31** lookup_rc_19_b,                       // 12 arrays
    m31** lookup_rc_19_c,                       // 12 arrays
    m31** lookup_rc_19_d,                       // 9 arrays
    m31** lookup_rc_19_e,                       // 9 arrays
    m31** lookup_rc_19_f,                       // 9 arrays
    m31** lookup_rc_19_g,                       // 9 arrays
    m31** lookup_rc_19_h,                       // 12 arrays
    // Lookup data pointers - range_check_9_9 variants (2 elements each)
    m31** lookup_rc_9_9,                        // 18*2 = 36 arrays
    m31** lookup_rc_9_9_b,                      // 36 arrays
    m31** lookup_rc_9_9_c,                      // 36 arrays
    m31** lookup_rc_9_9_d,                      // 36 arrays
    m31** lookup_rc_9_9_e,                      // 36 arrays
    m31** lookup_rc_9_9_f,                      // 36 arrays
    m31** lookup_rc_9_9_g,                      // 9*2 = 18 arrays
    m31** lookup_rc_9_9_h,                      // 18 arrays
    // Sizes
    unsigned int n_rows,                        // Number of valid (non-padding) rows
    unsigned int log_size,                      // Log2 of padded trace size
    // Output
    m31** interaction_trace_columns,            // Output interaction trace (4*107 = 428 cols)
    m31* claimed_sum                            // Output claimed sum (4 m31s for qm31)
);

}

#endif // GEN_PARTIAL_EC_MUL_TRACE_CUH
