#ifndef GEN_POSEIDON_3_PARTIAL_ROUNDS_CHAIN_TRACE_CUH
#define GEN_POSEIDON_3_PARTIAL_ROUNDS_CHAIN_TRACE_CUH

#include "fields.cuh"

// Number of trace columns for poseidon_3_partial_rounds_chain
#define POSEIDON_3_PARTIAL_ROUNDS_CHAIN_N_TRACE_COLUMNS 169

// Number of LogUp columns (9 fraction pairs)
#define POSEIDON_3_PARTIAL_ROUNDS_CHAIN_N_LOGUP_COLS 9

// Trace column layout:
// Columns 0-1: Input limbs (index, round_number)
// Columns 2-11: State[0] - 10 limbs (Width27 format)
// Columns 12-21: State[1] - 10 limbs (Width27 format)
// Columns 22-31: State[2] - 10 limbs (Width27 format)
// Columns 32-41: State[3] - 10 limbs (Width27 format)
// Columns 42-71: Poseidon round keys output (30 limbs = 3 x 10)
// Columns 72-81: Cube252 output[0] - 10 limbs
// Columns 82-91: Combination[0] - 10 limbs
// Column 92: p_coef[0]
// Columns 93-102: Combination[1] - 10 limbs
// Column 103: p_coef[1]
// Columns 104-113: Cube252 output[1] - 10 limbs
// Columns 114-123: Combination[2] - 10 limbs
// Column 124: p_coef[2]
// Columns 125-134: Combination[3] - 10 limbs
// Column 135: p_coef[3]
// Columns 136-145: Cube252 output[2] - 10 limbs
// Columns 146-155: Combination[4] - 10 limbs
// Column 156: p_coef[4]
// Columns 157-166: Combination[5] - 10 limbs
// Column 167: p_coef[5]
// Column 168: Enabler

// LogUp interaction structure (9 columns):
// LogUp 0: cube_252_0 + cube_252_1 (fractions paired)
// LogUp 1: cube_252_2 + poseidon_round_keys (fractions paired)
// LogUp 2: range_check_felt_252_width_27_0 + range_check_felt_252_width_27_1 (fractions paired)
// LogUp 3: range_check_felt_252_width_27_2 + range_check_4_4_0 (fractions paired)
// LogUp 4: range_check_4_4_1 + range_check_4_4_2 (fractions paired)
// LogUp 5: range_check_4_4_4_4_0 + range_check_4_4_4_4_1 (fractions paired)
// LogUp 6: range_check_4_4_4_4_2 + range_check_4_4_4_4_3 (fractions paired)
// LogUp 7: range_check_4_4_4_4_4 + range_check_4_4_4_4_5 (fractions paired)
// LogUp 8: poseidon_3_partial_rounds_chain self-lookup (denom1 * enabler - denom0 * enabler)

// Relations used:
// - Cube252: 3 lookups (20 values each: 10 input + 10 output)
// - PoseidonRoundKeys: 1 lookup (31 values: 1 round_num + 30 key limbs)
// - RangeCheckFelt252Width27: 3 lookups (10 values each)
// - RangeCheck_4_4: 3 lookups (2 values each)
// - RangeCheck_4_4_4_4: 6 lookups (4 values each)
// - Poseidon3PartialRoundsChain: 2 self-lookups (42 values each: 2 limbs + 4x10 state limbs)

extern "C" {

// Generate base trace for poseidon_3_partial_rounds_chain component
// Inputs:
//   - input_limb_0: Index column (n_rows)
//   - input_limb_1: Round number column (n_rows)
//   - states: 4 state arrays, each with 10 columns (Width27 format)
// Output: trace columns (169 columns)
void poseidon_3_partial_rounds_chain_generate_trace(
    m31* input_limb_0,              // Index values
    m31* input_limb_1,              // Round number values
    m31** state_0,                  // State[0]: 10 input columns (Width27 format)
    m31** state_1,                  // State[1]: 10 input columns (Width27 format)
    m31** state_2,                  // State[2]: 10 input columns (Width27 format)
    m31** state_3,                  // State[3]: 10 input columns (Width27 format)
    unsigned int n_rows,
    unsigned int actual_n_rows,     // Number of actual (non-padding) rows
    m31** trace_columns,            // 169 output trace columns
    // Poseidon round keys lookup table (from preprocessed columns)
    m31** poseidon_round_keys_table // 30 columns of round keys
);

// Add inputs to sub-component multiplicities
void poseidon_3_partial_rounds_chain_add_to_multiplicities(
    m31** trace_columns,
    unsigned int n_rows,
    // Cube252 multiplicities (will add 3 lookups per row)
    m31* cube_252_mults,
    unsigned int cube_252_log_size,
    // PoseidonRoundKeys multiplicities (will add 1 lookup per row)
    m31* poseidon_round_keys_mults,
    // RangeCheckFelt252Width27 multiplicities (will add 3 lookups per row)
    m31* rc_felt_252_width_27_mults,
    unsigned int rc_felt_252_width_27_log_size,
    // RangeCheck_4_4 multiplicities (will add 3 lookups per row)
    m31* rc_4_4_mults,
    unsigned int rc_4_4_log_size,
    // RangeCheck_4_4_4_4 multiplicities (will add 6 lookups per row)
    m31* rc_4_4_4_4_mults,
    unsigned int rc_4_4_4_4_log_size
);

// Generate interaction trace for poseidon_3_partial_rounds_chain component
void poseidon_3_partial_rounds_chain_generate_interaction_trace(
    m31** trace_columns,                              // Base trace (169 columns)
    unsigned int trace_size,
    // Lookup elements for each relation
    void* cube_252_lookup_elements,
    void* poseidon_round_keys_lookup_elements,
    void* range_check_felt_252_width_27_lookup_elements,
    void* range_check_4_4_lookup_elements,
    void* range_check_4_4_4_4_lookup_elements,
    void* poseidon_3_partial_rounds_chain_lookup_elements,
    m31** interaction_trace_columns,                  // Output interaction trace (36 columns)
    qm31* claimed_sum                                 // Output claimed sum
);

}

#endif // GEN_POSEIDON_3_PARTIAL_ROUNDS_CHAIN_TRACE_CUH
