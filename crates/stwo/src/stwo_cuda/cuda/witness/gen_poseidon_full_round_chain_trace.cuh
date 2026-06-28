#ifndef GEN_POSEIDON_FULL_ROUND_CHAIN_TRACE_CUH
#define GEN_POSEIDON_FULL_ROUND_CHAIN_TRACE_CUH

#include "fields.cuh"

// Number of trace columns for poseidon_full_round_chain
#define POSEIDON_FULL_ROUND_CHAIN_N_TRACE_COLUMNS 126

// Number of LogUp columns (6 fraction pairs)
#define POSEIDON_FULL_ROUND_CHAIN_N_LOGUP_COLS 6

// Trace column layout:
// Columns 0-1: Input limbs (index, round_number)
// Columns 2-11: State[0] - 10 limbs (Width27 format)
// Columns 12-21: State[1] - 10 limbs (Width27 format)
// Columns 22-31: State[2] - 10 limbs (Width27 format)
// Columns 32-41: Cube252 output for state[0] - 10 limbs
// Columns 42-51: Cube252 output for state[1] - 10 limbs
// Columns 52-61: Cube252 output for state[2] - 10 limbs
// Columns 62-71: Poseidon round keys output[0] - 10 limbs
// Columns 72-81: Poseidon round keys output[1] - 10 limbs
// Columns 82-91: Poseidon round keys output[2] - 10 limbs
// Columns 92-101: Linear combination result 1 (3*cube0 + cube1 + cube2 + key0) - 10 limbs
// Column 102: p_coef for linear combination 1
// Columns 103-112: Linear combination result 2 (cube0 - cube1 + cube2 + key1) - 10 limbs
// Column 113: p_coef for linear combination 2
// Columns 114-123: Linear combination result 3 (cube0 + cube1 - 2*cube2 + key2) - 10 limbs
// Column 124: p_coef for linear combination 3
// Column 125: Enabler

// LogUp interaction structure:
// LogUp 0: cube_252_0 + cube_252_1 (fractions paired)
// LogUp 1: cube_252_2 + poseidon_round_keys_0 (fractions paired)
// LogUp 2: range_check_3_3_3_3_3_0 + range_check_3_3_3_3_3_1 (fractions paired)
// LogUp 3: range_check_3_3_3_3_3_2 + range_check_3_3_3_3_3_3 (fractions paired)
// LogUp 4: range_check_3_3_3_3_3_4 + range_check_3_3_3_3_3_5 (fractions paired)
// LogUp 5: poseidon_full_round_chain self-lookup with enabler (denom1 * enabler - denom0 * enabler)

// Relations used:
// - Cube252: 3 lookups (20 values each: 10 input + 10 output)
// - PoseidonRoundKeys: 1 lookup (31 values: 1 round_num + 30 key limbs)
// - RangeCheck_3_3_3_3_3: 6 lookups (5 values each)
// - PoseidonFullRoundChain: 2 self-lookups (32 values each: 2 limbs + 3x10 state limbs)

extern "C" {

// Generate base trace for poseidon_full_round_chain component
// Inputs:
//   - input_limb_0: Index column (n_rows)
//   - input_limb_1: Round number column (n_rows)
//   - states: 3 state arrays, each with 10 columns (Width27 format)
// Output: trace columns (126 columns)
void poseidon_full_round_chain_generate_trace(
    m31* input_limb_0,              // Index values
    m31* input_limb_1,              // Round number values
    m31** state_0,                  // State[0]: 10 input columns (Width27 format)
    m31** state_1,                  // State[1]: 10 input columns (Width27 format)
    m31** state_2,                  // State[2]: 10 input columns (Width27 format)
    unsigned int n_rows,
    m31** trace_columns,            // 126 output trace columns
    // Poseidon round keys lookup table (from preprocessed columns)
    m31** poseidon_round_keys_table // 30 columns of round keys
);

// Add inputs to sub-component multiplicities
void poseidon_full_round_chain_add_to_multiplicities(
    m31** trace_columns,
    unsigned int n_rows,
    // Cube252 multiplicities (will add 3 lookups per row)
    m31* cube_252_mults,
    unsigned int cube_252_log_size,
    // PoseidonRoundKeys multiplicities (will add 1 lookup per row)
    m31* poseidon_round_keys_mults,
    // RangeCheck_3_3_3_3_3 multiplicities (will add 6 lookups per row)
    m31* rc_3_3_3_3_3_mults,
    unsigned int rc_3_3_3_3_3_log_size
);

// Generate interaction trace for poseidon_full_round_chain component
void poseidon_full_round_chain_generate_interaction_trace(
    m31** trace_columns,                              // Base trace (126 columns)
    unsigned int trace_size,
    // Lookup elements for each relation
    void* cube_252_lookup_elements,
    void* poseidon_round_keys_lookup_elements,
    void* range_check_3_3_3_3_3_lookup_elements,
    void* poseidon_full_round_chain_lookup_elements,
    m31** interaction_trace_columns,                  // Output interaction trace (24 columns)
    qm31* claimed_sum                                 // Output claimed sum
);

}

#endif // GEN_POSEIDON_FULL_ROUND_CHAIN_TRACE_CUH
