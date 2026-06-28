#ifndef GEN_CUBE_252_TRACE_CUH
#define GEN_CUBE_252_TRACE_CUH

#include "fields.cuh"

// Number of trace columns for cube_252
#define CUBE_252_N_TRACE_COLUMNS 141

// Trace column layout:
// Columns 0-9: Input limbs (10 x 27-bit)
// Columns 10-27: Unpacked limbs (18 columns for positions 0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25)
// Columns 28-55: First multiplication result x² (28 x 9-bit limbs)
// Column 56: k for first mul
// Columns 57-83: Carry for first mul (27 values)
// Columns 84-111: Second multiplication result x³ (28 x 9-bit limbs)
// Column 112: k for second mul
// Columns 113-139: Carry for second mul (27 values)
// Column 140: Enabler

// LogUp interaction elements needed:
// - cube_252 (self lookup)
// - range_check_9_9 variants (a through h)
// - range_check_19 variants (a through h)

extern "C" {

// Generate base trace and lookup data for cube_252 component (following blake_g pattern)
// Output: trace columns (141 columns) + lookup data arrays
// Input: width27 inputs (10 limbs per row)
void generate_cube_252_trace(
    // Output: trace columns
    m31** trace_columns,                    // 141 output trace columns
    // Output: lookup data for cube_252 self-lookup (20 elements)
    m31** lookup_cube_252_0,
    // Output: lookup data for range_check_9_9 variants (2 elements each)
    m31** lookup_rc_9_9_0,
    m31** lookup_rc_9_9_1,
    m31** lookup_rc_9_9_2,
    m31** lookup_rc_9_9_3,
    m31** lookup_rc_9_9_4,
    m31** lookup_rc_9_9_5,
    m31** lookup_rc_9_9_b_0,
    m31** lookup_rc_9_9_b_1,
    m31** lookup_rc_9_9_b_2,
    m31** lookup_rc_9_9_b_3,
    m31** lookup_rc_9_9_b_4,
    m31** lookup_rc_9_9_b_5,
    m31** lookup_rc_9_9_c_0,
    m31** lookup_rc_9_9_c_1,
    m31** lookup_rc_9_9_c_2,
    m31** lookup_rc_9_9_c_3,
    m31** lookup_rc_9_9_c_4,
    m31** lookup_rc_9_9_c_5,
    m31** lookup_rc_9_9_d_0,
    m31** lookup_rc_9_9_d_1,
    m31** lookup_rc_9_9_d_2,
    m31** lookup_rc_9_9_d_3,
    m31** lookup_rc_9_9_d_4,
    m31** lookup_rc_9_9_d_5,
    m31** lookup_rc_9_9_e_0,
    m31** lookup_rc_9_9_e_1,
    m31** lookup_rc_9_9_e_2,
    m31** lookup_rc_9_9_e_3,
    m31** lookup_rc_9_9_e_4,
    m31** lookup_rc_9_9_e_5,
    m31** lookup_rc_9_9_f_0,
    m31** lookup_rc_9_9_f_1,
    m31** lookup_rc_9_9_f_2,
    m31** lookup_rc_9_9_f_3,
    m31** lookup_rc_9_9_f_4,
    m31** lookup_rc_9_9_f_5,
    m31** lookup_rc_9_9_g_0,
    m31** lookup_rc_9_9_g_1,
    m31** lookup_rc_9_9_g_2,
    m31** lookup_rc_9_9_h_0,
    m31** lookup_rc_9_9_h_1,
    m31** lookup_rc_9_9_h_2,
    // Output: lookup data for range_check_19 variants (1 element each)
    m31** lookup_rc_19_0,
    m31** lookup_rc_19_1,
    m31** lookup_rc_19_2,
    m31** lookup_rc_19_3,
    m31** lookup_rc_19_4,
    m31** lookup_rc_19_5,
    m31** lookup_rc_19_6,
    m31** lookup_rc_19_7,
    m31** lookup_rc_19_b_0,
    m31** lookup_rc_19_b_1,
    m31** lookup_rc_19_b_2,
    m31** lookup_rc_19_b_3,
    m31** lookup_rc_19_b_4,
    m31** lookup_rc_19_b_5,
    m31** lookup_rc_19_b_6,
    m31** lookup_rc_19_b_7,
    m31** lookup_rc_19_c_0,
    m31** lookup_rc_19_c_1,
    m31** lookup_rc_19_c_2,
    m31** lookup_rc_19_c_3,
    m31** lookup_rc_19_c_4,
    m31** lookup_rc_19_c_5,
    m31** lookup_rc_19_c_6,
    m31** lookup_rc_19_c_7,
    m31** lookup_rc_19_d_0,
    m31** lookup_rc_19_d_1,
    m31** lookup_rc_19_d_2,
    m31** lookup_rc_19_d_3,
    m31** lookup_rc_19_d_4,
    m31** lookup_rc_19_d_5,
    m31** lookup_rc_19_e_0,
    m31** lookup_rc_19_e_1,
    m31** lookup_rc_19_e_2,
    m31** lookup_rc_19_e_3,
    m31** lookup_rc_19_e_4,
    m31** lookup_rc_19_e_5,
    m31** lookup_rc_19_f_0,
    m31** lookup_rc_19_f_1,
    m31** lookup_rc_19_f_2,
    m31** lookup_rc_19_f_3,
    m31** lookup_rc_19_f_4,
    m31** lookup_rc_19_f_5,
    m31** lookup_rc_19_g_0,
    m31** lookup_rc_19_g_1,
    m31** lookup_rc_19_g_2,
    m31** lookup_rc_19_g_3,
    m31** lookup_rc_19_g_4,
    m31** lookup_rc_19_g_5,
    m31** lookup_rc_19_h_0,
    m31** lookup_rc_19_h_1,
    m31** lookup_rc_19_h_2,
    m31** lookup_rc_19_h_3,
    m31** lookup_rc_19_h_4,
    m31** lookup_rc_19_h_5,
    m31** lookup_rc_19_h_6,
    m31** lookup_rc_19_h_7,
    // Sub-component inputs for range_check_9_9 variants
    m31** sub_rc_9_9,                       // 6 * 2 = 12 pointers
    m31** sub_rc_9_9_b,                     // 6 * 2 = 12 pointers
    m31** sub_rc_9_9_c,                     // 6 * 2 = 12 pointers
    m31** sub_rc_9_9_d,                     // 6 * 2 = 12 pointers
    m31** sub_rc_9_9_e,                     // 6 * 2 = 12 pointers
    m31** sub_rc_9_9_f,                     // 6 * 2 = 12 pointers
    m31** sub_rc_9_9_g,                     // 3 * 2 = 6 pointers
    m31** sub_rc_9_9_h,                     // 3 * 2 = 6 pointers
    // Sub-component inputs for range_check_19 variants
    m31** sub_rc_19,                        // 8 * 1 = 8 pointers
    m31** sub_rc_19_b,                      // 8 * 1 = 8 pointers
    m31** sub_rc_19_c,                      // 8 * 1 = 8 pointers
    m31** sub_rc_19_d,                      // 6 * 1 = 6 pointers
    m31** sub_rc_19_e,                      // 6 * 1 = 6 pointers
    m31** sub_rc_19_f,                      // 6 * 1 = 6 pointers
    m31** sub_rc_19_g,                      // 6 * 1 = 6 pointers
    m31** sub_rc_19_h,                      // 8 * 1 = 8 pointers
    // Input
    m31** inputs,                           // 10 input columns (Width27 format)
    unsigned int trace_log_size             // Log size of trace
);

// Generate interaction trace for cube_252 component
// Parameter order matches SIMD and Rust bindings: cube_252, rc_19*, rc_9_9*
void generate_cube_252_interaction_trace(
    m31** trace_columns,                    // Base trace (141 columns)
    unsigned int trace_size,
    // Lookup elements for each relation (order: cube_252, rc_19*, rc_9_9*)
    void* cube_252_lookup_elements,
    void* rc_19_lookup_elements,
    void* rc_19_b_lookup_elements,
    void* rc_19_c_lookup_elements,
    void* rc_19_d_lookup_elements,
    void* rc_19_e_lookup_elements,
    void* rc_19_f_lookup_elements,
    void* rc_19_g_lookup_elements,
    void* rc_19_h_lookup_elements,
    void* rc_9_9_lookup_elements,
    void* rc_9_9_b_lookup_elements,
    void* rc_9_9_c_lookup_elements,
    void* rc_9_9_d_lookup_elements,
    void* rc_9_9_e_lookup_elements,
    void* rc_9_9_f_lookup_elements,
    void* rc_9_9_g_lookup_elements,
    void* rc_9_9_h_lookup_elements,
    m31** interaction_trace_columns,        // Output interaction trace
    qm31* claimed_sum                       // Output claimed sum
);

}

#endif // GEN_CUBE_252_TRACE_CUH
