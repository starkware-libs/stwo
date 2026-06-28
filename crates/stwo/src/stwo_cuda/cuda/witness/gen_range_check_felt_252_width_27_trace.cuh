#ifndef GEN_RANGE_CHECK_FELT_252_WIDTH_27_TRACE_CUH
#define GEN_RANGE_CHECK_FELT_252_WIDTH_27_TRACE_CUH

#include "fields.cuh"

// Number of trace columns for range_check_felt_252_width_27
#define RANGE_CHECK_FELT_252_WIDTH_27_N_TRACE_COLUMNS 20

// Number of LogUp columns (8 fraction pairs)
#define RANGE_CHECK_FELT_252_WIDTH_27_N_LOGUP_COLS 8

// Trace column layout:
// Columns 0-9: Input limbs (10 Width27 limbs)
// Columns 10-18: Extracted parts for range checks
//   - Column 10: limb_0_high_part (9 bits)
//   - Column 11: limb_1_low_part (9 bits)
//   - Column 12: limb_2_high_part (9 bits)
//   - Column 13: limb_3_low_part (9 bits)
//   - Column 14: limb_4_high_part (9 bits)
//   - Column 15: limb_5_low_part (9 bits)
//   - Column 16: limb_6_high_part (9 bits)
//   - Column 17: limb_7_low_part (9 bits)
//   - Column 18: limb_8_high_part (9 bits)
// Column 19: Enabler

// LogUp interaction structure (8 columns):
// LogUp 0: range_check_9_9 + range_check_18[0] (fractions paired)
// LogUp 1: range_check_18[1] + range_check_9_9_b (fractions paired)
// LogUp 2: range_check_18_b[0] + range_check_18[2] (fractions paired)
// LogUp 3: range_check_9_9_c + range_check_18[3] (fractions paired)
// LogUp 4: range_check_18[4] + range_check_9_9_d (fractions paired)
// LogUp 5: range_check_18_b[1] + range_check_18[5] (fractions paired)
// LogUp 6: range_check_9_9_e + range_check_18[6] (fractions paired)
// LogUp 7: range_check_felt_252_width_27 self-lookup with enabler

// Relations used:
// - RangeCheck_18: 7 lookups
// - RangeCheck_18_B: 2 lookups
// - RangeCheck_9_9: 1 lookup
// - RangeCheck_9_9_B: 1 lookup
// - RangeCheck_9_9_C: 1 lookup
// - RangeCheck_9_9_D: 1 lookup
// - RangeCheck_9_9_E: 1 lookup
// - RangeCheckFelt252Width27: 1 self-lookup

extern "C" {

// Generate base trace for range_check_felt_252_width_27 component
// Input: Felt252Width27 values (10 limbs per row)
void range_check_felt_252_width_27_generate_trace(
    m31** input_limbs,              // 10 input columns (Width27 format)
    unsigned int n_rows,            // Padded size (power of 2)
    unsigned int actual_n_rows,     // Actual data rows (before padding)
    m31** trace_columns             // 20 output trace columns
);

// Add inputs to sub-component multiplicities
void range_check_felt_252_width_27_add_to_multiplicities(
    m31** trace_columns,
    unsigned int n_rows,
    // RangeCheck_18 multiplicities (7 lookups per row)
    m31* rc_18_mults,
    unsigned int rc_18_log_size,
    // RangeCheck_18_B multiplicities (2 lookups per row)
    m31* rc_18_b_mults,
    unsigned int rc_18_b_log_size,
    // RangeCheck_9_9 multiplicities (1 lookup per row)
    m31* rc_9_9_mults,
    unsigned int rc_9_9_log_size,
    // RangeCheck_9_9_B multiplicities (1 lookup per row)
    m31* rc_9_9_b_mults,
    unsigned int rc_9_9_b_log_size,
    // RangeCheck_9_9_C multiplicities (1 lookup per row)
    m31* rc_9_9_c_mults,
    unsigned int rc_9_9_c_log_size,
    // RangeCheck_9_9_D multiplicities (1 lookup per row)
    m31* rc_9_9_d_mults,
    unsigned int rc_9_9_d_log_size,
    // RangeCheck_9_9_E multiplicities (1 lookup per row)
    m31* rc_9_9_e_mults,
    unsigned int rc_9_9_e_log_size
);

// Generate interaction trace for range_check_felt_252_width_27 component
void range_check_felt_252_width_27_generate_interaction_trace(
    m31** trace_columns,                              // Base trace (20 columns)
    unsigned int trace_size,
    // Lookup elements for each relation
    void* range_check_9_9_lookup_elements,
    void* range_check_18_lookup_elements,
    void* range_check_9_9_b_lookup_elements,
    void* range_check_18_b_lookup_elements,
    void* range_check_9_9_c_lookup_elements,
    void* range_check_9_9_d_lookup_elements,
    void* range_check_9_9_e_lookup_elements,
    void* range_check_felt_252_width_27_lookup_elements,
    m31** interaction_trace_columns,                  // Output interaction trace (32 columns)
    qm31* claimed_sum                                 // Output claimed sum
);

}

#endif // GEN_RANGE_CHECK_FELT_252_WIDTH_27_TRACE_CUH
