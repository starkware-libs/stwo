// CUDA trace generation for poseidon_3_partial_rounds_chain component
// 169 trace columns, computes 3 partial Poseidon rounds
//
// Trace layout:
// Columns 0-1: Input (index, round_number)
// Columns 2-41: Input states (4 x 10 Width27 limbs)
// Columns 42-71: Poseidon round keys (30 limbs = 3 x 10)
// Columns 72-81: Cube252 output[0] (10 limbs)
// Columns 82-91: Combination[0] (10 limbs)
// Column 92: p_coef[0]
// Columns 93-102: Combination[1] (10 limbs)
// Column 103: p_coef[1]
// Columns 104-113: Cube252 output[1] (10 limbs)
// Columns 114-123: Combination[2] (10 limbs)
// Column 124: p_coef[2]
// Columns 125-134: Combination[3] (10 limbs)
// Column 135: p_coef[3]
// Columns 136-145: Cube252 output[2] (10 limbs)
// Columns 146-155: Combination[4] (10 limbs)
// Column 156: p_coef[4]
// Columns 157-166: Combination[5] (10 limbs)
// Column 167: p_coef[5]
// Column 168: Enabler

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "gen_poseidon_3_partial_rounds_chain_trace.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
#include <cstdint>
#include <cstdio>
#include "cuda_mem_pool.cuh"

#define POSEIDON_3PRC_BLOCK_SIZE 256

// ============================================================================
// Felt252Field type and operations
// ============================================================================

typedef ff_storage<8> Felt252Field;

__device__ __forceinline__ Felt252Field poseidon_3prc_felt_add(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

__device__ __forceinline__ Felt252Field poseidon_3prc_felt_sub(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

__device__ __forceinline__ Felt252Field poseidon_3prc_felt_mul(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

// Convert from Montgomery form to normal form (for extracting Width27 limbs)
__device__ __forceinline__ Felt252Field poseidon_3prc_felt_from_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::from_montgomery(a);
}

// Convert to Montgomery form for arithmetic
__device__ __forceinline__ Felt252Field poseidon_3prc_felt_to_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::to_montgomery(a);
}

// Full reduction to canonical form [0, p)
// The ff_dispatch_st::add may return values >= p after multiple additions.
// We need to fully reduce before extracting Width27 limbs.
__device__ __forceinline__ Felt252Field poseidon_3prc_felt_reduce(const Felt252Field& a) {
    // Apply reduce multiple times to handle cases where a >= 2p or more
    // The reduce<1> function subtracts p if a >= p
    // With 11 additions in the combination, we might need up to 11 reductions
    Felt252Field result = ff_dispatch_st<ff_config_starknet>::reduce<1>(a);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    return result;
}

// p_coef computation helper for combination (bias = 2^28 + 2 = 268435458, subtract = 2)
// Formula: p_coef = ((raw_diff + 268435458) & 0xFFFF) - 2
// Uses M31 field subtraction to handle underflow correctly
__device__ __forceinline__ m31 compute_p_coef_comb(int64_t raw_diff) {
    int64_t biased = raw_diff + 268435458LL;
    uint32_t biased_masked = (uint32_t)(biased & 0xFFFF);
    return sub((m31){biased_masked}, (m31){2});
}

// p_coef computation helper for doubled (bias = 2^27 + 1 = 134217729, subtract = 1)
// Formula: p_coef = ((raw_diff + 134217729) & 0xFFFF) - 1
// Uses M31 field subtraction to handle underflow correctly
__device__ __forceinline__ m31 compute_p_coef_doubled(int64_t raw_diff) {
    int64_t biased = raw_diff + 134217729LL;
    uint32_t biased_masked = (uint32_t)(biased & 0xFFFF);
    return sub((m31){biased_masked}, (m31){1});
}

// Montgomery cube factor for x^3 correction
__device__ __constant__ Felt252Field POSEIDON_3PRC_MONT_CUBE_FACTOR = {{
    0x406DF18E, 0xCC7177D1, 0x77FFCC06, 0x75457066,
    0x36300018, 0xF47D84F8, 0x873C0A6D, 0x038E5F79
}};

// ============================================================================
// Width27 to Felt252Field conversion
// ============================================================================

__device__ Felt252Field width27_to_felt252_3prc(const m31* limbs) {
    uint64_t val0 = 0, val1 = 0, val2 = 0, val3 = 0;

    val0 = (uint64_t)limbs[0];
    val0 |= ((uint64_t)limbs[1]) << 27;
    val0 |= ((uint64_t)limbs[2]) << 54;

    val1 = ((uint64_t)limbs[2]) >> 10;
    val1 |= ((uint64_t)limbs[3]) << 17;
    val1 |= ((uint64_t)limbs[4]) << 44;

    val2 = ((uint64_t)limbs[4]) >> 20;
    val2 |= ((uint64_t)limbs[5]) << 7;
    val2 |= ((uint64_t)limbs[6]) << 34;
    val2 |= ((uint64_t)limbs[7]) << 61;

    val3 = ((uint64_t)limbs[7]) >> 3;
    val3 |= ((uint64_t)limbs[8]) << 24;
    val3 |= ((uint64_t)limbs[9]) << 51;

    Felt252Field result;
    result.limbs[0] = (uint32_t)(val0 & 0xFFFFFFFF);
    result.limbs[1] = (uint32_t)((val0 >> 32) & 0xFFFFFFFF);
    result.limbs[2] = (uint32_t)(val1 & 0xFFFFFFFF);
    result.limbs[3] = (uint32_t)((val1 >> 32) & 0xFFFFFFFF);
    result.limbs[4] = (uint32_t)(val2 & 0xFFFFFFFF);
    result.limbs[5] = (uint32_t)((val2 >> 32) & 0xFFFFFFFF);
    result.limbs[6] = (uint32_t)(val3 & 0xFFFFFFFF);
    result.limbs[7] = (uint32_t)((val3 >> 32) & 0xFFFFFFFF);

    return result;
}

// Convert Felt252Field back to 10 Width27 limbs
__device__ void felt252_to_width27_3prc(const Felt252Field& felt, m31* limbs) {
    uint64_t val0 = ((uint64_t)felt.limbs[1] << 32) | felt.limbs[0];
    uint64_t val1 = ((uint64_t)felt.limbs[3] << 32) | felt.limbs[2];
    uint64_t val2 = ((uint64_t)felt.limbs[5] << 32) | felt.limbs[4];
    uint64_t val3 = ((uint64_t)felt.limbs[7] << 32) | felt.limbs[6];

    // Extract 27-bit limbs
    limbs[0] = (m31){(uint32_t)(val0 & 0x7FFFFFF)};
    limbs[1] = (m31){(uint32_t)((val0 >> 27) & 0x7FFFFFF)};
    uint64_t cross01 = (val0 >> 54) | (val1 << 10);
    limbs[2] = (m31){(uint32_t)(cross01 & 0x7FFFFFF)};
    limbs[3] = (m31){(uint32_t)((val1 >> 17) & 0x7FFFFFF)};
    uint64_t cross12 = (val1 >> 44) | (val2 << 20);
    limbs[4] = (m31){(uint32_t)(cross12 & 0x7FFFFFF)};
    limbs[5] = (m31){(uint32_t)((val2 >> 7) & 0x7FFFFFF)};
    limbs[6] = (m31){(uint32_t)((val2 >> 34) & 0x7FFFFFF)};
    uint64_t cross23 = (val2 >> 61) | (val3 << 3);
    limbs[7] = (m31){(uint32_t)(cross23 & 0x7FFFFFF)};
    limbs[8] = (m31){(uint32_t)((val3 >> 24) & 0x7FFFFFF)};
    limbs[9] = (m31){(uint32_t)((val3 >> 51) & 0x1FF)};
}

// Compute cube (x^3) of a Felt252 value
__device__ Felt252Field compute_cube_felt252_3prc(const Felt252Field& x) {
    Felt252Field x2 = poseidon_3prc_felt_mul(x, x);
    Felt252Field x3 = poseidon_3prc_felt_mul(x2, x);
    // Use local copy to avoid potential __constant__ memory issues
    Felt252Field local_factor = POSEIDON_3PRC_MONT_CUBE_FACTOR;
    return poseidon_3prc_felt_mul(x3, local_factor);
}

// ============================================================================
// Partial round computation
// In Poseidon partial rounds, state[3] (the last state) is cubed
// The linear combination formula from SIMD code:
//   combination = 4*s0 + 2*s1 + 3*s2 + 1*s3 - 1*cube(s3) + 1*key
// Then: second_combination = 2 * combination
// Output state = [cube(s3), second_combination]
// ============================================================================

// Compute 4*s0 + 2*s1 + 3*s2 + s3 - cube(s3) + key for first combination
__device__ Felt252Field compute_partial_round_combination1(
    const Felt252Field& s0,
    const Felt252Field& s1,
    const Felt252Field& s2,
    const Felt252Field& s3,
    const Felt252Field& cube_s3,
    const Felt252Field& key
) {
    Felt252Field result = key;

    // Add 4*s0
    result = poseidon_3prc_felt_add(result, s0);
    result = poseidon_3prc_felt_add(result, s0);
    result = poseidon_3prc_felt_add(result, s0);
    result = poseidon_3prc_felt_add(result, s0);

    // Add 2*s1
    result = poseidon_3prc_felt_add(result, s1);
    result = poseidon_3prc_felt_add(result, s1);

    // Add 3*s2
    result = poseidon_3prc_felt_add(result, s2);
    result = poseidon_3prc_felt_add(result, s2);
    result = poseidon_3prc_felt_add(result, s2);

    // Add 1*s3
    result = poseidon_3prc_felt_add(result, s3);

    // Subtract 1*cube(s3)
    result = poseidon_3prc_felt_sub(result, cube_s3);

    return result;
}

// Compute 2 * value
__device__ Felt252Field compute_double(const Felt252Field& val) {
    return poseidon_3prc_felt_add(val, val);
}

// ============================================================================
// Main trace generation kernel
// ============================================================================

__global__ void poseidon_3_partial_rounds_chain_trace_kernel(
    m31* input_limb_0,              // Index values
    m31* input_limb_1,              // Round number values
    m31** state_0,                  // State[0]: 10 input columns
    m31** state_1,                  // State[1]: 10 input columns
    m31** state_2,                  // State[2]: 10 input columns
    m31** state_3,                  // State[3]: 10 input columns
    unsigned int n_rows,
    unsigned int actual_n_rows,     // Number of actual (non-padding) rows
    m31** trace_columns,            // 169 output trace columns
    m31** poseidon_round_keys_table // 30 columns of round keys
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Read inputs
    m31 idx = input_limb_0[row];
    m31 round_num = input_limb_1[row];

    // Read 4 input states (each 10 Width27 limbs)
    m31 s0[10], s1[10], s2[10], s3[10];
    for (int i = 0; i < 10; i++) {
        s0[i] = state_0[i][row];
        s1[i] = state_1[i][row];
        s2[i] = state_2[i][row];
        s3[i] = state_3[i][row];
    }

    // Write input columns (0-41)
    trace_columns[0][row] = idx;
    trace_columns[1][row] = round_num;
    for (int i = 0; i < 10; i++) {
        trace_columns[2 + i][row] = s0[i];      // Columns 2-11
        trace_columns[12 + i][row] = s1[i];     // Columns 12-21
        trace_columns[22 + i][row] = s2[i];     // Columns 22-31
        trace_columns[32 + i][row] = s3[i];     // Columns 32-41
    }

    // Look up poseidon round keys (columns 42-71)
    m31 key0_limbs[10], key1_limbs[10], key2_limbs[10];
    unsigned int round_idx = round_num;
    for (int i = 0; i < 10; i++) {
        key0_limbs[i] = poseidon_round_keys_table[i][round_idx];
        key1_limbs[i] = poseidon_round_keys_table[10 + i][round_idx];
        key2_limbs[i] = poseidon_round_keys_table[20 + i][round_idx];
    }

    for (int i = 0; i < 10; i++) {
        trace_columns[42 + i][row] = key0_limbs[i];    // Columns 42-51
        trace_columns[52 + i][row] = key1_limbs[i];    // Columns 52-61
        trace_columns[62 + i][row] = key2_limbs[i];    // Columns 62-71
    }

    // Convert states to Felt252Field
    Felt252Field f0 = width27_to_felt252_3prc(s0);
    Felt252Field f1 = width27_to_felt252_3prc(s1);
    Felt252Field f2 = width27_to_felt252_3prc(s2);
    Felt252Field f3 = width27_to_felt252_3prc(s3);

    Felt252Field key0 = width27_to_felt252_3prc(key0_limbs);
    Felt252Field key1 = width27_to_felt252_3prc(key1_limbs);
    Felt252Field key2 = width27_to_felt252_3prc(key2_limbs);

    // ========================
    // Partial Round 1
    // ========================
    // Input: [s0, s1, s2, s3] = [f0, f1, f2, f3]
    // Cube s3 (the LAST state, not first)
    // comb1 = 4*s0 + 2*s1 + 3*s2 + s3 - cube + key0
    // doubled1 = 2 * comb1
    // Output: [cube0, doubled1]
    // Next round input: [s2, s3, cube0, doubled1]

    Felt252Field cube0 = compute_cube_felt252_3prc(f3);  // Cube state[3], NOT state[0]!
    m31 cube0_limbs[10];
    felt252_to_width27_3prc(cube0, cube0_limbs);

    // Write cube output (columns 72-81)
    for (int i = 0; i < 10; i++) {
        trace_columns[72 + i][row] = cube0_limbs[i];
    }

    // Compute combination: 4*s0 + 2*s1 + 3*s2 + s3 - cube + key0
    Felt252Field comb1 = compute_partial_round_combination1(f0, f1, f2, f3, cube0, key0);
    // Reduce to canonical form before extracting Width27 limbs
    comb1 = poseidon_3prc_felt_reduce(comb1);
    m31 comb1_limbs[10];
    felt252_to_width27_3prc(comb1, comb1_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[82 + i][row] = comb1_limbs[i];    // Columns 82-91
    }
    // Compute p_coef for comb1: 4*s0[0] + 2*s1[0] + 3*s2[0] + s3[0] - cube0[0] + key0[0] - comb1[0]
    int64_t raw_p_coef1 = 4LL * (int64_t)s0[0] + 2LL * (int64_t)s1[0]
                        + 3LL * (int64_t)s2[0] + (int64_t)s3[0]
                        - (int64_t)cube0_limbs[0] + (int64_t)key0_limbs[0]
                        - (int64_t)comb1_limbs[0];
    trace_columns[92][row] = compute_p_coef_comb(raw_p_coef1);

    // Double the combination
    Felt252Field doubled1 = compute_double(comb1);
    // Reduce to canonical form before extracting Width27 limbs
    doubled1 = poseidon_3prc_felt_reduce(doubled1);
    m31 doubled1_limbs[10];
    felt252_to_width27_3prc(doubled1, doubled1_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[93 + i][row] = doubled1_limbs[i]; // Columns 93-102
    }
    // Compute p_coef for doubled1: 2*comb1[0] - doubled1[0]
    int64_t raw_p_coef_doubled1 = 2LL * (int64_t)comb1_limbs[0] - (int64_t)doubled1_limbs[0];
    trace_columns[103][row] = compute_p_coef_doubled(raw_p_coef_doubled1);

    // ========================
    // Partial Round 2
    // ========================
    // Input: [s2, s3, cube0, doubled1] = [f2, f3, cube0, doubled1]
    // Cube doubled1 (last element of round 1 output)
    // comb2 = 4*s2 + 2*s3 + 3*cube0 + doubled1 - cube1 + key1
    // doubled2 = 2 * comb2
    // Output: [cube1, doubled2]

    Felt252Field cube1 = compute_cube_felt252_3prc(doubled1);  // Cube the doubled combination
    m31 cube1_limbs[10];
    felt252_to_width27_3prc(cube1, cube1_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[104 + i][row] = cube1_limbs[i];  // Columns 104-113
    }

    // Compute combination: 4*s2 + 2*s3 + 3*cube0 + doubled1 - cube1 + key1
    Felt252Field comb2 = compute_partial_round_combination1(f2, f3, cube0, doubled1, cube1, key1);
    // Reduce to canonical form before extracting Width27 limbs
    comb2 = poseidon_3prc_felt_reduce(comb2);
    m31 comb2_limbs[10];
    felt252_to_width27_3prc(comb2, comb2_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[114 + i][row] = comb2_limbs[i];  // Columns 114-123
    }
    // Compute p_coef for comb2: 4*s2[0] + 2*s3[0] + 3*cube0[0] + doubled1[0] - cube1[0] + key1[0] - comb2[0]
    int64_t raw_p_coef2 = 4LL * (int64_t)s2[0] + 2LL * (int64_t)s3[0]
                        + 3LL * (int64_t)cube0_limbs[0] + (int64_t)doubled1_limbs[0]
                        - (int64_t)cube1_limbs[0] + (int64_t)key1_limbs[0]
                        - (int64_t)comb2_limbs[0];
    trace_columns[124][row] = compute_p_coef_comb(raw_p_coef2);

    // Double the combination
    Felt252Field doubled2 = compute_double(comb2);
    // Reduce to canonical form before extracting Width27 limbs
    doubled2 = poseidon_3prc_felt_reduce(doubled2);
    m31 doubled2_limbs[10];
    felt252_to_width27_3prc(doubled2, doubled2_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[125 + i][row] = doubled2_limbs[i]; // Columns 125-134
    }
    // Compute p_coef for doubled2: 2*comb2[0] - doubled2[0]
    int64_t raw_p_coef_doubled2 = 2LL * (int64_t)comb2_limbs[0] - (int64_t)doubled2_limbs[0];
    trace_columns[135][row] = compute_p_coef_doubled(raw_p_coef_doubled2);

    // ========================
    // Partial Round 3
    // ========================
    // Input: [cube0, doubled1, cube1, doubled2]
    // Cube doubled2 (last element of round 2 output)
    // comb3 = 4*cube0 + 2*doubled1 + 3*cube1 + doubled2 - cube2 + key2
    // doubled3 = 2 * comb3
    // Output: [cube2, doubled3]

    Felt252Field cube2 = compute_cube_felt252_3prc(doubled2);
    m31 cube2_limbs[10];
    felt252_to_width27_3prc(cube2, cube2_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[136 + i][row] = cube2_limbs[i];  // Columns 136-145
    }

    // Compute combination: 4*cube0 + 2*doubled1 + 3*cube1 + doubled2 - cube2 + key2
    Felt252Field comb3 = compute_partial_round_combination1(cube0, doubled1, cube1, doubled2, cube2, key2);
    // Reduce to canonical form before extracting Width27 limbs
    comb3 = poseidon_3prc_felt_reduce(comb3);
    m31 comb3_limbs[10];
    felt252_to_width27_3prc(comb3, comb3_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[146 + i][row] = comb3_limbs[i];  // Columns 146-155
    }
    // Compute p_coef for comb3: 4*cube0[0] + 2*doubled1[0] + 3*cube1[0] + doubled2[0] - cube2[0] + key2[0] - comb3[0]
    int64_t raw_p_coef3 = 4LL * (int64_t)cube0_limbs[0] + 2LL * (int64_t)doubled1_limbs[0]
                        + 3LL * (int64_t)cube1_limbs[0] + (int64_t)doubled2_limbs[0]
                        - (int64_t)cube2_limbs[0] + (int64_t)key2_limbs[0]
                        - (int64_t)comb3_limbs[0];
    trace_columns[156][row] = compute_p_coef_comb(raw_p_coef3);

    // Double the combination
    Felt252Field doubled3 = compute_double(comb3);
    // Reduce to canonical form before extracting Width27 limbs
    doubled3 = poseidon_3prc_felt_reduce(doubled3);
    m31 doubled3_limbs[10];
    felt252_to_width27_3prc(doubled3, doubled3_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[157 + i][row] = doubled3_limbs[i]; // Columns 157-166
    }
    // Compute p_coef for doubled3: 2*comb3[0] - doubled3[0]
    int64_t raw_p_coef_doubled3 = 2LL * (int64_t)comb3_limbs[0] - (int64_t)doubled3_limbs[0];
    trace_columns[167][row] = compute_p_coef_doubled(raw_p_coef_doubled3);

    // Write enabler (column 168) - only 1 for actual rows, 0 for padding
    trace_columns[168][row] = (row < actual_n_rows) ? (m31){1} : (m31){0};
}

// ============================================================================
// Host functions
// ============================================================================

extern "C" void poseidon_3_partial_rounds_chain_generate_trace(
    m31* input_limb_0,
    m31* input_limb_1,
    m31** state_0,
    m31** state_1,
    m31** state_2,
    m31** state_3,
    unsigned int n_rows,
    unsigned int actual_n_rows,
    m31** trace_columns,
    m31** poseidon_round_keys_table
) {
    // Copy state pointers to device
    m31** d_state_0;
    m31** d_state_1;
    m31** d_state_2;
    m31** d_state_3;
    d_state_0 = cuda_mem_pool_allocate<m31*>(10);
    d_state_1 = cuda_mem_pool_allocate<m31*>(10);
    d_state_2 = cuda_mem_pool_allocate<m31*>(10);
    d_state_3 = cuda_mem_pool_allocate<m31*>(10);
    cudaMemcpyAsync(d_state_0, state_0, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_state_1, state_1, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_state_2, state_2, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_state_3, state_3, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy trace column pointers to device
    m31** d_trace_columns;
    d_trace_columns = cuda_mem_pool_allocate<m31*>(POSEIDON_3_PARTIAL_ROUNDS_CHAIN_N_TRACE_COLUMNS);
    cudaMemcpyAsync(d_trace_columns, trace_columns, POSEIDON_3_PARTIAL_ROUNDS_CHAIN_N_TRACE_COLUMNS * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy round keys table pointers to device
    m31** d_round_keys;
    d_round_keys = cuda_mem_pool_allocate<m31*>(30);
    cudaMemcpyAsync(d_round_keys, poseidon_round_keys_table, 30 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Launch kernel
    int block_size = POSEIDON_3PRC_BLOCK_SIZE;
    int num_blocks = (n_rows + block_size - 1) / block_size;

    poseidon_3_partial_rounds_chain_trace_kernel<<<num_blocks, block_size>>>(
        input_limb_0,
        input_limb_1,
        d_state_0,
        d_state_1,
        d_state_2,
        d_state_3,
        n_rows,
        actual_n_rows,
        d_trace_columns,
        d_round_keys
    );


    // Cleanup
    cuda_mem_pool_free(d_state_0);
    cuda_mem_pool_free(d_state_1);
    cuda_mem_pool_free(d_state_2);
    cuda_mem_pool_free(d_state_3);
    cuda_mem_pool_free(d_trace_columns);
    cuda_mem_pool_free(d_round_keys);
}

// Helper: compute 9 carries for one linear combination (one partial round).
// Formula per limb i:
//   val_i = 4*s0[i] + 2*s1[i] + 3*s2[i] + s3[i] - cube[i] + key[i] - combo[i]
//   carry_0 = (val_0 - p_coef) * 16
//   carry_i = (carry_{i-1} + val_i) * 16   for i in 1..6
//   carry_7 = (carry_6 + val_7 - p_coef * 136) * 16
//   carry_8 = (carry_7 + val_8) * 16
__device__ void compute_lc6_carries(
    m31* s0, m31* s1, m31* s2, m31* s3,
    m31* cube_out, m31* key, m31* combo,
    m31 p_coef,
    m31* carries  // output: 9 carries
) {
    m31 val0 = sub(sub(add(sub(add(add(add(mul((m31)4, s0[0]), mul((m31)2, s1[0])), mul((m31)3, s2[0])), s3[0]), cube_out[0]), key[0]), combo[0]), p_coef);
    carries[0] = mul(val0, (m31)16);

    for (int i = 1; i < 7; i++) {
        m31 val = sub(add(sub(add(add(add(add(carries[i-1], mul((m31)4, s0[i])), mul((m31)2, s1[i])), mul((m31)3, s2[i])), s3[i]), cube_out[i]), key[i]), combo[i]);
        carries[i] = mul(val, (m31)16);
    }

    m31 val7 = sub(sub(add(sub(add(add(add(add(carries[6], mul((m31)4, s0[7])), mul((m31)2, s1[7])), mul((m31)3, s2[7])), s3[7]), cube_out[7]), key[7]), combo[7]), mul(p_coef, (m31)136));
    carries[7] = mul(val7, (m31)16);

    m31 val8 = sub(add(sub(add(add(add(add(carries[7], mul((m31)4, s0[8])), mul((m31)2, s1[8])), mul((m31)3, s2[8])), s3[8]), cube_out[8]), key[8]), combo[8]);
    carries[8] = mul(val8, (m31)16);
}

__global__ void poseidon_3_partial_rounds_chain_add_to_multiplicities_kernel(
    m31** trace_columns,
    unsigned int n_rows,
    m31* rc_4_4_mults,
    unsigned int rc_4_4_log_size,
    m31* rc_4_4_4_4_mults,
    unsigned int rc_4_4_4_4_log_size
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Read input state limbs (cols 2-41)
    m31 state0[10], state1[10], state2[10], state3[10];
    for (int i = 0; i < 10; i++) {
        state0[i] = trace_columns[2 + i][row];
        state1[i] = trace_columns[12 + i][row];
        state2[i] = trace_columns[22 + i][row];
        state3[i] = trace_columns[32 + i][row];
    }

    // Read round keys (cols 42-71)
    m31 key0[10], key1[10], key2[10];
    for (int i = 0; i < 10; i++) {
        key0[i] = trace_columns[42 + i][row];
        key1[i] = trace_columns[52 + i][row];
        key2[i] = trace_columns[62 + i][row];
    }

    // Read cube outputs
    m31 cube0[10], cube1[10], cube2[10];
    for (int i = 0; i < 10; i++) {
        cube0[i] = trace_columns[72 + i][row];
        cube1[i] = trace_columns[104 + i][row];
        cube2[i] = trace_columns[136 + i][row];
    }

    // Read combinations
    m31 combo0[10], combo1[10], combo2[10];
    for (int i = 0; i < 10; i++) {
        combo0[i] = trace_columns[82 + i][row];
        combo1[i] = trace_columns[114 + i][row];
        combo2[i] = trace_columns[146 + i][row];
    }

    // Read scaled combinations (2x)
    m31 combo0_2x[10], combo1_2x[10], combo2_2x[10];
    for (int i = 0; i < 10; i++) {
        combo0_2x[i] = trace_columns[93 + i][row];
        combo1_2x[i] = trace_columns[125 + i][row];
        combo2_2x[i] = trace_columns[157 + i][row];
    }

    // Read p_coef values
    m31 p_coef0 = trace_columns[92][row];
    m31 p_coef1 = trace_columns[124][row];
    m31 p_coef2 = trace_columns[156][row];

    m31 carries[9];

    // ========== Round 0 ==========
    // LC6: 4*state0 + 2*state1 + 3*state2 + state3 - cube0 + key0 - combo0
    compute_lc6_carries(state0, state1, state2, state3, cube0, key0, combo0, p_coef0, carries);

    // rc_4_4_4_4[0]: [p_coef0+2, carry0+2, carry1+2, carry2+2]
    {
        uint32_t v0 = add(p_coef0, (m31)2);
        uint32_t v1 = add(carries[0], (m31)2);
        uint32_t v2 = add(carries[1], (m31)2);
        uint32_t v3 = add(carries[2], (m31)2);
        uint32_t idx = (v0 << 12) | (v1 << 8) | (v2 << 4) | v3;
        atomicAdd((unsigned int*)&rc_4_4_4_4_mults[idx], 1);
    }

    // rc_4_4_4_4[1]: [carry3+2, carry4+2, carry5+2, carry6+2]
    {
        uint32_t v0 = add(carries[3], (m31)2);
        uint32_t v1 = add(carries[4], (m31)2);
        uint32_t v2 = add(carries[5], (m31)2);
        uint32_t v3 = add(carries[6], (m31)2);
        uint32_t idx = (v0 << 12) | (v1 << 8) | (v2 << 4) | v3;
        atomicAdd((unsigned int*)&rc_4_4_4_4_mults[idx], 1);
    }

    // rc_4_4[0]: [carry7+2, carry8+2]
    {
        uint32_t v0 = add(carries[7], (m31)2);
        uint32_t v1 = add(carries[8], (m31)2);
        uint32_t idx = (v0 << 4) | v1;
        atomicAdd((unsigned int*)&rc_4_4_mults[idx], 1);
    }

    // ========== Round 1 ==========
    // LC6: 4*state2 + 2*state3 + 3*cube0 + combo0_2x - cube1 + key1 - combo1
    compute_lc6_carries(state2, state3, cube0, combo0_2x, cube1, key1, combo1, p_coef1, carries);

    // rc_4_4_4_4[2]: [p_coef1+2, carry0+2, carry1+2, carry2+2]
    {
        uint32_t v0 = add(p_coef1, (m31)2);
        uint32_t v1 = add(carries[0], (m31)2);
        uint32_t v2 = add(carries[1], (m31)2);
        uint32_t v3 = add(carries[2], (m31)2);
        uint32_t idx = (v0 << 12) | (v1 << 8) | (v2 << 4) | v3;
        atomicAdd((unsigned int*)&rc_4_4_4_4_mults[idx], 1);
    }

    // rc_4_4_4_4[3]: [carry3+2, carry4+2, carry5+2, carry6+2]
    {
        uint32_t v0 = add(carries[3], (m31)2);
        uint32_t v1 = add(carries[4], (m31)2);
        uint32_t v2 = add(carries[5], (m31)2);
        uint32_t v3 = add(carries[6], (m31)2);
        uint32_t idx = (v0 << 12) | (v1 << 8) | (v2 << 4) | v3;
        atomicAdd((unsigned int*)&rc_4_4_4_4_mults[idx], 1);
    }

    // rc_4_4[1]: [carry7+2, carry8+2]
    {
        uint32_t v0 = add(carries[7], (m31)2);
        uint32_t v1 = add(carries[8], (m31)2);
        uint32_t idx = (v0 << 4) | v1;
        atomicAdd((unsigned int*)&rc_4_4_mults[idx], 1);
    }

    // ========== Round 2 ==========
    // LC6: 4*cube0 + 2*combo0_2x + 3*cube1 + combo1_2x - cube2 + key2 - combo2
    compute_lc6_carries(cube0, combo0_2x, cube1, combo1_2x, cube2, key2, combo2, p_coef2, carries);

    // rc_4_4_4_4[4]: [p_coef2+2, carry0+2, carry1+2, carry2+2]
    {
        uint32_t v0 = add(p_coef2, (m31)2);
        uint32_t v1 = add(carries[0], (m31)2);
        uint32_t v2 = add(carries[1], (m31)2);
        uint32_t v3 = add(carries[2], (m31)2);
        uint32_t idx = (v0 << 12) | (v1 << 8) | (v2 << 4) | v3;
        atomicAdd((unsigned int*)&rc_4_4_4_4_mults[idx], 1);
    }

    // rc_4_4_4_4[5]: [carry3+2, carry4+2, carry5+2, carry6+2]
    {
        uint32_t v0 = add(carries[3], (m31)2);
        uint32_t v1 = add(carries[4], (m31)2);
        uint32_t v2 = add(carries[5], (m31)2);
        uint32_t v3 = add(carries[6], (m31)2);
        uint32_t idx = (v0 << 12) | (v1 << 8) | (v2 << 4) | v3;
        atomicAdd((unsigned int*)&rc_4_4_4_4_mults[idx], 1);
    }

    // rc_4_4[2]: [carry7+2, carry8+2]
    {
        uint32_t v0 = add(carries[7], (m31)2);
        uint32_t v1 = add(carries[8], (m31)2);
        uint32_t idx = (v0 << 4) | v1;
        atomicAdd((unsigned int*)&rc_4_4_mults[idx], 1);
    }
}

extern "C" void poseidon_3_partial_rounds_chain_add_to_multiplicities(
    m31** trace_columns,
    unsigned int n_rows,
    m31* cube_252_mults,
    unsigned int cube_252_log_size,
    m31* poseidon_round_keys_mults,
    m31* rc_felt_252_width_27_mults,
    unsigned int rc_felt_252_width_27_log_size,
    m31* rc_4_4_mults,
    unsigned int rc_4_4_log_size,
    m31* rc_4_4_4_4_mults,
    unsigned int rc_4_4_4_4_log_size
) {
    if (n_rows == 0) return;

    // Copy trace column pointers to device
    m31** d_trace_columns = clone_to_device<m31*>(trace_columns, 169);

    unsigned int block_size = 256;
    unsigned int grid_size = (n_rows + block_size - 1) / block_size;

    poseidon_3_partial_rounds_chain_add_to_multiplicities_kernel<<<grid_size, block_size>>>(
        d_trace_columns,
        n_rows,
        rc_4_4_mults,
        rc_4_4_log_size,
        rc_4_4_4_4_mults,
        rc_4_4_4_4_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(d_trace_columns);
}

// =============================================================================
// Phase 2: Compute LogUp Fractions
// =============================================================================

#define POSEIDON_3PRC_N_LOGUP_COLS 9

__global__ void poseidon_3_partial_rounds_chain_compute_fractions_kernel(
    m31** trace_columns,
    unsigned int trace_size,
    // Lookup elements for each relation
    LookupElementsBasic<31>* poseidon_round_keys, // 31 values per lookup
    LookupElementsBasic<20>* cube_252,            // 20 values per lookup
    LookupElementsBasic<4>* range_check_4_4_4_4,  // 4 values per lookup
    LookupElementsBasic<2>* range_check_4_4,      // 2 values per lookup
    LookupElementsBasic<10>* range_check_felt_252_width_27, // 10 values per lookup
    LookupElementsBasic<42>* poseidon_3_partial_rounds_chain, // 42 values per lookup
    // Output arrays for fractions
    qm31* denom_ptr,          // [9 * trace_size]
    m31* numerator0,          // [9 * trace_size]
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Trace layout from header:
    // Columns 0-1: Input limbs (index, round_number)
    // Columns 2-41: Input states (4 x 10 Width27 limbs)
    // Columns 42-71: Poseidon round keys output (30 limbs = 3 x 10)
    // Columns 72-81: Cube252 output[0] (10 limbs)
    // Columns 82-91: Combination[0] (10 limbs)
    // Column 92: p_coef[0]
    // Columns 93-102: Combination[1] (10 limbs)
    // Column 103: p_coef[1]
    // Columns 104-113: Cube252 output[1] (10 limbs)
    // Columns 114-123: Combination[2] (10 limbs)
    // Column 124: p_coef[2]
    // Columns 125-134: Combination[3] (10 limbs)
    // Column 135: p_coef[3]
    // Columns 136-145: Cube252 output[2] (10 limbs)
    // Columns 146-155: Combination[4] (10 limbs)
    // Column 156: p_coef[4]
    // Columns 157-166: Combination[5] (10 limbs)
    // Column 167: p_coef[5]
    // Column 168: Enabler

    m31 enabler = trace_columns[168][row];

    // LogUp column 0: poseidon_round_keys_0 + cube_252_0
    // poseidon_round_keys_0: round_num (col 1) + keys (cols 42-71)
    // cube_252_0: input state[3] (cols 32-41) + cube output[0] (cols 72-81)
    {
        m31 input0[31], input1[20];
        // poseidon_round_keys_0: round_num (col 1) + keys (cols 42-71)
        input0[0] = trace_columns[1][row]; // round_number
        for (int i = 0; i < 30; i++) {
            input0[1 + i] = trace_columns[42 + i][row]; // round keys
        }
        // cube_252_0: state[3] input (cols 32-41) + cube output (cols 72-81)
        for (int i = 0; i < 10; i++) {
            input1[i] = trace_columns[32 + i][row];     // state[3] input
            input1[10 + i] = trace_columns[72 + i][row]; // cube[0] output
        }
        qm31 denom0 = poseidon_round_keys->combine(input0, 31);
        qm31 denom1 = cube_252->combine(input1, 20);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 0 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 1: range_check_4_4_4_4_0 + range_check_4_4_4_4_1
    // These are from the first partial round's carry computation
    // We need to compute these from trace values
    {
        // Read values needed for computing carries from first combination
        m31 state0[10], state1[10], state2[10], state3[10];
        m31 cube0[10], key0[10], combo0[10];
        for (int i = 0; i < 10; i++) {
            state0[i] = trace_columns[2 + i][row];   // state[0]
            state1[i] = trace_columns[12 + i][row];  // state[1]
            state2[i] = trace_columns[22 + i][row];  // state[2]
            state3[i] = trace_columns[32 + i][row];  // state[3]
            cube0[i] = trace_columns[72 + i][row];   // cube[0] output
            key0[i] = trace_columns[42 + i][row];    // key[0]
            combo0[i] = trace_columns[82 + i][row];  // combination[0]
        }
        m31 p_coef0 = trace_columns[92][row];

        // First combination: 4*state0 + 2*state1 + 3*state2 + state3 - cube0 + key0
        // Compute carries: carry = (accumulated - combo - p_coef_adjustment) * 16
        m31 carry_0 = mul(sub(sub(add(sub(add(add(add(mul((m31)4, state0[0]), mul((m31)2, state1[0])), mul((m31)3, state2[0])), state3[0]), cube0[0]), key0[0]), combo0[0]), p_coef0), (m31)16);
        m31 carry_1 = mul(sub(add(sub(add(add(add(add(carry_0, mul((m31)4, state0[1])), mul((m31)2, state1[1])), mul((m31)3, state2[1])), state3[1]), cube0[1]), key0[1]), combo0[1]), (m31)16);
        m31 carry_2 = mul(sub(add(sub(add(add(add(add(carry_1, mul((m31)4, state0[2])), mul((m31)2, state1[2])), mul((m31)3, state2[2])), state3[2]), cube0[2]), key0[2]), combo0[2]), (m31)16);
        m31 carry_3 = mul(sub(add(sub(add(add(add(add(carry_2, mul((m31)4, state0[3])), mul((m31)2, state1[3])), mul((m31)3, state2[3])), state3[3]), cube0[3]), key0[3]), combo0[3]), (m31)16);
        m31 carry_4 = mul(sub(add(sub(add(add(add(add(carry_3, mul((m31)4, state0[4])), mul((m31)2, state1[4])), mul((m31)3, state2[4])), state3[4]), cube0[4]), key0[4]), combo0[4]), (m31)16);
        m31 carry_5 = mul(sub(add(sub(add(add(add(add(carry_4, mul((m31)4, state0[5])), mul((m31)2, state1[5])), mul((m31)3, state2[5])), state3[5]), cube0[5]), key0[5]), combo0[5]), (m31)16);
        m31 carry_6 = mul(sub(add(sub(add(add(add(add(carry_5, mul((m31)4, state0[6])), mul((m31)2, state1[6])), mul((m31)3, state2[6])), state3[6]), cube0[6]), key0[6]), combo0[6]), (m31)16);

        // Biased values for range_check_4_4_4_4_0 and _1 (bias +2)
        m31 input0[4] = {add(p_coef0, (m31)2), add(carry_0, (m31)2), add(carry_1, (m31)2), add(carry_2, (m31)2)};
        m31 input1[4] = {add(carry_3, (m31)2), add(carry_4, (m31)2), add(carry_5, (m31)2), add(carry_6, (m31)2)};

        qm31 denom0 = range_check_4_4_4_4->combine(input0, 4);
        qm31 denom1 = range_check_4_4_4_4->combine(input1, 4);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 1 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 2: range_check_4_4_0 + range_check_felt_252_width_27_0
    // range_check_4_4_0: [carry_7+2, carry_8+2] from first combination
    // range_check_felt_252_width_27_0: combination[0] (cols 82-91)
    {
        // Recompute carry_7 and carry_8 from first combination
        m31 state0[10], state1[10], state2[10], state3[10];
        m31 cube0[10], key0[10], combo0[10];
        for (int i = 0; i < 10; i++) {
            state0[i] = trace_columns[2 + i][row];
            state1[i] = trace_columns[12 + i][row];
            state2[i] = trace_columns[22 + i][row];
            state3[i] = trace_columns[32 + i][row];
            cube0[i] = trace_columns[72 + i][row];
            key0[i] = trace_columns[42 + i][row];
            combo0[i] = trace_columns[82 + i][row];
        }
        m31 p_coef0 = trace_columns[92][row];

        // Compute carries 0-6 to get to carry_7, carry_8
        m31 carry_0 = mul(sub(sub(add(sub(add(add(add(mul((m31)4, state0[0]), mul((m31)2, state1[0])), mul((m31)3, state2[0])), state3[0]), cube0[0]), key0[0]), combo0[0]), p_coef0), (m31)16);
        m31 carry_1 = mul(sub(add(sub(add(add(add(add(carry_0, mul((m31)4, state0[1])), mul((m31)2, state1[1])), mul((m31)3, state2[1])), state3[1]), cube0[1]), key0[1]), combo0[1]), (m31)16);
        m31 carry_2 = mul(sub(add(sub(add(add(add(add(carry_1, mul((m31)4, state0[2])), mul((m31)2, state1[2])), mul((m31)3, state2[2])), state3[2]), cube0[2]), key0[2]), combo0[2]), (m31)16);
        m31 carry_3 = mul(sub(add(sub(add(add(add(add(carry_2, mul((m31)4, state0[3])), mul((m31)2, state1[3])), mul((m31)3, state2[3])), state3[3]), cube0[3]), key0[3]), combo0[3]), (m31)16);
        m31 carry_4 = mul(sub(add(sub(add(add(add(add(carry_3, mul((m31)4, state0[4])), mul((m31)2, state1[4])), mul((m31)3, state2[4])), state3[4]), cube0[4]), key0[4]), combo0[4]), (m31)16);
        m31 carry_5 = mul(sub(add(sub(add(add(add(add(carry_4, mul((m31)4, state0[5])), mul((m31)2, state1[5])), mul((m31)3, state2[5])), state3[5]), cube0[5]), key0[5]), combo0[5]), (m31)16);
        m31 carry_6 = mul(sub(add(sub(add(add(add(add(carry_5, mul((m31)4, state0[6])), mul((m31)2, state1[6])), mul((m31)3, state2[6])), state3[6]), cube0[6]), key0[6]), combo0[6]), (m31)16);
        m31 carry_7 = mul(sub(sub(add(sub(add(add(add(add(carry_6, mul((m31)4, state0[7])), mul((m31)2, state1[7])), mul((m31)3, state2[7])), state3[7]), cube0[7]), key0[7]), combo0[7]), mul(p_coef0, (m31)136)), (m31)16);
        m31 carry_8 = mul(sub(add(sub(add(add(add(add(carry_7, mul((m31)4, state0[8])), mul((m31)2, state1[8])), mul((m31)3, state2[8])), state3[8]), cube0[8]), key0[8]), combo0[8]), (m31)16);

        m31 input0[2] = {add(carry_7, (m31)2), add(carry_8, (m31)2)};
        m31 input1[10];
        for (int i = 0; i < 10; i++) {
            input1[i] = combo0[i]; // range_check_felt_252_width_27_0 = combination[0]
        }

        qm31 denom0 = range_check_4_4->combine(input0, 2);
        qm31 denom1 = range_check_felt_252_width_27->combine(input1, 10);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 2 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 3: cube_252_1 + range_check_4_4_4_4_2
    // cube_252_1: doubled1 (cols 93-102) + cube output[1] (cols 104-113)
    // range_check_4_4_4_4_2: from THIRD combination (verifying cols 114-123)
    // Third combination: 4*input + 2*state3 + 3*cube0 + combo1 - cube1 + key1 - combo3
    {
        m31 cube_input[20];
        // cube_252_1: doubled1 (combination[1]) as input + cube output[1]
        for (int i = 0; i < 10; i++) {
            cube_input[i] = trace_columns[93 + i][row];      // doubled1 (cols 93-102) as cube input
            cube_input[10 + i] = trace_columns[104 + i][row]; // cube[1] output
        }

        // Compute range_check_4_4_4_4_2 from third combination (cols 114-123)
        // Third combination formula: 4*input + 2*state3 + 3*cube0 + combo1 - cube1 + key1 - combo3
        m31 input2[10], state3[10], cube0[10], combo1[10], cube1[10], key1[10], combo3[10];
        for (int i = 0; i < 10; i++) {
            input2[i] = trace_columns[22 + i][row];   // input state[2]
            state3[i] = trace_columns[32 + i][row];   // input state[3]
            cube0[i] = trace_columns[72 + i][row];    // cube[0] output
            combo1[i] = trace_columns[93 + i][row];   // doubled1
            cube1[i] = trace_columns[104 + i][row];   // cube[1] output
            key1[i] = trace_columns[52 + i][row];     // key[1]
            combo3[i] = trace_columns[114 + i][row];  // combination[3] = the output
        }
        m31 p_coef3 = trace_columns[124][row];

        // Compute carries for third combination
        // carry_0 = (4*input2[0] + 2*state3[0] + 3*cube0[0] + combo1[0] - cube1[0] + key1[0] - combo3[0] - p_coef3) * 16
        m31 carry_0 = mul(sub(sub(add(sub(add(add(add(mul((m31)4, input2[0]), mul((m31)2, state3[0])), mul((m31)3, cube0[0])), combo1[0]), cube1[0]), key1[0]), combo3[0]), p_coef3), (m31)16);
        m31 carry_1 = mul(sub(add(sub(add(add(add(add(carry_0, mul((m31)4, input2[1])), mul((m31)2, state3[1])), mul((m31)3, cube0[1])), combo1[1]), cube1[1]), key1[1]), combo3[1]), (m31)16);
        m31 carry_2 = mul(sub(add(sub(add(add(add(add(carry_1, mul((m31)4, input2[2])), mul((m31)2, state3[2])), mul((m31)3, cube0[2])), combo1[2]), cube1[2]), key1[2]), combo3[2]), (m31)16);

        m31 rc_input[4] = {add(p_coef3, (m31)2), add(carry_0, (m31)2), add(carry_1, (m31)2), add(carry_2, (m31)2)};

        qm31 denom0 = cube_252->combine(cube_input, 20);
        qm31 denom1 = range_check_4_4_4_4->combine(rc_input, 4);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 3 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 4: range_check_4_4_4_4_3 + range_check_4_4_1
    // Continue carries from THIRD combination (cols 114-123)
    {
        m31 input2[10], state3[10], cube0[10], combo1[10], cube1[10], key1[10], combo3[10];
        for (int i = 0; i < 10; i++) {
            input2[i] = trace_columns[22 + i][row];
            state3[i] = trace_columns[32 + i][row];
            cube0[i] = trace_columns[72 + i][row];
            combo1[i] = trace_columns[93 + i][row];
            cube1[i] = trace_columns[104 + i][row];
            key1[i] = trace_columns[52 + i][row];
            combo3[i] = trace_columns[114 + i][row];
        }
        m31 p_coef3 = trace_columns[124][row];

        // Compute all carries for third combination
        m31 carry_0 = mul(sub(sub(add(sub(add(add(add(mul((m31)4, input2[0]), mul((m31)2, state3[0])), mul((m31)3, cube0[0])), combo1[0]), cube1[0]), key1[0]), combo3[0]), p_coef3), (m31)16);
        m31 carry_1 = mul(sub(add(sub(add(add(add(add(carry_0, mul((m31)4, input2[1])), mul((m31)2, state3[1])), mul((m31)3, cube0[1])), combo1[1]), cube1[1]), key1[1]), combo3[1]), (m31)16);
        m31 carry_2 = mul(sub(add(sub(add(add(add(add(carry_1, mul((m31)4, input2[2])), mul((m31)2, state3[2])), mul((m31)3, cube0[2])), combo1[2]), cube1[2]), key1[2]), combo3[2]), (m31)16);
        m31 carry_3 = mul(sub(add(sub(add(add(add(add(carry_2, mul((m31)4, input2[3])), mul((m31)2, state3[3])), mul((m31)3, cube0[3])), combo1[3]), cube1[3]), key1[3]), combo3[3]), (m31)16);
        m31 carry_4 = mul(sub(add(sub(add(add(add(add(carry_3, mul((m31)4, input2[4])), mul((m31)2, state3[4])), mul((m31)3, cube0[4])), combo1[4]), cube1[4]), key1[4]), combo3[4]), (m31)16);
        m31 carry_5 = mul(sub(add(sub(add(add(add(add(carry_4, mul((m31)4, input2[5])), mul((m31)2, state3[5])), mul((m31)3, cube0[5])), combo1[5]), cube1[5]), key1[5]), combo3[5]), (m31)16);
        m31 carry_6 = mul(sub(add(sub(add(add(add(add(carry_5, mul((m31)4, input2[6])), mul((m31)2, state3[6])), mul((m31)3, cube0[6])), combo1[6]), cube1[6]), key1[6]), combo3[6]), (m31)16);
        m31 carry_7 = mul(sub(sub(add(sub(add(add(add(add(carry_6, mul((m31)4, input2[7])), mul((m31)2, state3[7])), mul((m31)3, cube0[7])), combo1[7]), cube1[7]), key1[7]), combo3[7]), mul(p_coef3, (m31)136)), (m31)16);
        m31 carry_8 = mul(sub(add(sub(add(add(add(add(carry_7, mul((m31)4, input2[8])), mul((m31)2, state3[8])), mul((m31)3, cube0[8])), combo1[8]), cube1[8]), key1[8]), combo3[8]), (m31)16);

        m31 rc_input0[4] = {add(carry_3, (m31)2), add(carry_4, (m31)2), add(carry_5, (m31)2), add(carry_6, (m31)2)};
        m31 rc_input1[2] = {add(carry_7, (m31)2), add(carry_8, (m31)2)};

        qm31 denom0 = range_check_4_4_4_4->combine(rc_input0, 4);
        qm31 denom1 = range_check_4_4->combine(rc_input1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 4 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 5: range_check_felt_252_width_27_1 + cube_252_2
    // range_check_felt_252_width_27_1: combination[3] (cols 114-123)
    // cube_252_2: combination[4] (cols 125-134) + cube output[2] (cols 136-145)
    {
        m31 input0[10], input1[20];
        // range_check_felt_252_width_27_1 = combination[3] (cols 114-123)
        for (int i = 0; i < 10; i++) {
            input0[i] = trace_columns[114 + i][row];
        }
        // cube_252_2: combination[4] (cols 125-134) + cube output[2] (cols 136-145)
        for (int i = 0; i < 10; i++) {
            input1[i] = trace_columns[125 + i][row];     // combination[4] as cube input
            input1[10 + i] = trace_columns[136 + i][row]; // cube[2] output
        }

        qm31 denom0 = range_check_felt_252_width_27->combine(input0, 10);
        qm31 denom1 = cube_252->combine(input1, 20);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 5 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 6: range_check_4_4_4_4_4 + range_check_4_4_4_4_5
    // Fourth combination verifying cols 146-155 (combination[5])
    // Formula: 4*cube0 + 2*combo1 + 3*cube1 + combo4 - cube2 + key2 - combo5
    {
        m31 cube0[10], combo1[10], cube1[10], combo4[10], cube2[10], key2[10], combo5[10];
        for (int i = 0; i < 10; i++) {
            cube0[i] = trace_columns[72 + i][row];     // cube[0] output
            combo1[i] = trace_columns[93 + i][row];    // doubled1 (cols 93-102)
            cube1[i] = trace_columns[104 + i][row];    // cube[1] output
            combo4[i] = trace_columns[125 + i][row];   // doubled3 (cols 125-134)
            cube2[i] = trace_columns[136 + i][row];    // cube[2] output
            key2[i] = trace_columns[62 + i][row];      // key[2]
            combo5[i] = trace_columns[146 + i][row];   // combination[5] = the output
        }
        m31 p_coef5 = trace_columns[156][row];

        // Formula: 4*cube0 + 2*combo1 + 3*cube1 + combo4 - cube2 + key2 - combo5 - p_coef5
        m31 carry_0 = mul(sub(sub(add(sub(add(add(add(mul((m31)4, cube0[0]), mul((m31)2, combo1[0])), mul((m31)3, cube1[0])), combo4[0]), cube2[0]), key2[0]), combo5[0]), p_coef5), (m31)16);
        m31 carry_1 = mul(sub(add(sub(add(add(add(add(carry_0, mul((m31)4, cube0[1])), mul((m31)2, combo1[1])), mul((m31)3, cube1[1])), combo4[1]), cube2[1]), key2[1]), combo5[1]), (m31)16);
        m31 carry_2 = mul(sub(add(sub(add(add(add(add(carry_1, mul((m31)4, cube0[2])), mul((m31)2, combo1[2])), mul((m31)3, cube1[2])), combo4[2]), cube2[2]), key2[2]), combo5[2]), (m31)16);
        m31 carry_3 = mul(sub(add(sub(add(add(add(add(carry_2, mul((m31)4, cube0[3])), mul((m31)2, combo1[3])), mul((m31)3, cube1[3])), combo4[3]), cube2[3]), key2[3]), combo5[3]), (m31)16);
        m31 carry_4 = mul(sub(add(sub(add(add(add(add(carry_3, mul((m31)4, cube0[4])), mul((m31)2, combo1[4])), mul((m31)3, cube1[4])), combo4[4]), cube2[4]), key2[4]), combo5[4]), (m31)16);
        m31 carry_5 = mul(sub(add(sub(add(add(add(add(carry_4, mul((m31)4, cube0[5])), mul((m31)2, combo1[5])), mul((m31)3, cube1[5])), combo4[5]), cube2[5]), key2[5]), combo5[5]), (m31)16);
        m31 carry_6 = mul(sub(add(sub(add(add(add(add(carry_5, mul((m31)4, cube0[6])), mul((m31)2, combo1[6])), mul((m31)3, cube1[6])), combo4[6]), cube2[6]), key2[6]), combo5[6]), (m31)16);

        m31 input0[4] = {add(p_coef5, (m31)2), add(carry_0, (m31)2), add(carry_1, (m31)2), add(carry_2, (m31)2)};
        m31 input1[4] = {add(carry_3, (m31)2), add(carry_4, (m31)2), add(carry_5, (m31)2), add(carry_6, (m31)2)};

        qm31 denom0 = range_check_4_4_4_4->combine(input0, 4);
        qm31 denom1 = range_check_4_4_4_4->combine(input1, 4);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 6 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 7: range_check_4_4_2 + range_check_felt_252_width_27_2
    // Continue carries from fourth combination (verifying cols 146-155)
    {
        m31 cube0[10], combo1[10], cube1[10], combo4[10], cube2[10], key2[10], combo5[10];
        for (int i = 0; i < 10; i++) {
            cube0[i] = trace_columns[72 + i][row];
            combo1[i] = trace_columns[93 + i][row];    // doubled1
            cube1[i] = trace_columns[104 + i][row];
            combo4[i] = trace_columns[125 + i][row];   // doubled3
            cube2[i] = trace_columns[136 + i][row];
            key2[i] = trace_columns[62 + i][row];
            combo5[i] = trace_columns[146 + i][row];
        }
        m31 p_coef5 = trace_columns[156][row];

        // Compute all carries for fourth combination
        m31 carry_0 = mul(sub(sub(add(sub(add(add(add(mul((m31)4, cube0[0]), mul((m31)2, combo1[0])), mul((m31)3, cube1[0])), combo4[0]), cube2[0]), key2[0]), combo5[0]), p_coef5), (m31)16);
        m31 carry_1 = mul(sub(add(sub(add(add(add(add(carry_0, mul((m31)4, cube0[1])), mul((m31)2, combo1[1])), mul((m31)3, cube1[1])), combo4[1]), cube2[1]), key2[1]), combo5[1]), (m31)16);
        m31 carry_2 = mul(sub(add(sub(add(add(add(add(carry_1, mul((m31)4, cube0[2])), mul((m31)2, combo1[2])), mul((m31)3, cube1[2])), combo4[2]), cube2[2]), key2[2]), combo5[2]), (m31)16);
        m31 carry_3 = mul(sub(add(sub(add(add(add(add(carry_2, mul((m31)4, cube0[3])), mul((m31)2, combo1[3])), mul((m31)3, cube1[3])), combo4[3]), cube2[3]), key2[3]), combo5[3]), (m31)16);
        m31 carry_4 = mul(sub(add(sub(add(add(add(add(carry_3, mul((m31)4, cube0[4])), mul((m31)2, combo1[4])), mul((m31)3, cube1[4])), combo4[4]), cube2[4]), key2[4]), combo5[4]), (m31)16);
        m31 carry_5 = mul(sub(add(sub(add(add(add(add(carry_4, mul((m31)4, cube0[5])), mul((m31)2, combo1[5])), mul((m31)3, cube1[5])), combo4[5]), cube2[5]), key2[5]), combo5[5]), (m31)16);
        m31 carry_6 = mul(sub(add(sub(add(add(add(add(carry_5, mul((m31)4, cube0[6])), mul((m31)2, combo1[6])), mul((m31)3, cube1[6])), combo4[6]), cube2[6]), key2[6]), combo5[6]), (m31)16);
        m31 carry_7 = mul(sub(sub(add(sub(add(add(add(add(carry_6, mul((m31)4, cube0[7])), mul((m31)2, combo1[7])), mul((m31)3, cube1[7])), combo4[7]), cube2[7]), key2[7]), combo5[7]), mul(p_coef5, (m31)136)), (m31)16);
        m31 carry_8 = mul(sub(add(sub(add(add(add(add(carry_7, mul((m31)4, cube0[8])), mul((m31)2, combo1[8])), mul((m31)3, cube1[8])), combo4[8]), cube2[8]), key2[8]), combo5[8]), (m31)16);

        m31 input0[2] = {add(carry_7, (m31)2), add(carry_8, (m31)2)};
        m31 input1[10];
        for (int i = 0; i < 10; i++) {
            input1[i] = combo5[i]; // range_check_felt_252_width_27_2 = combination[5] (cols 146-155)
        }

        qm31 denom0 = range_check_4_4->combine(input0, 2);
        qm31 denom1 = range_check_felt_252_width_27->combine(input1, 10);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 7 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 8: poseidon_3_partial_rounds_chain self-relation with enabler
    // Input: all 42 input limbs (cols 0-41)
    // Output: col0, col1+1, and specific output columns
    {
        m31 input0[42], input1[42];
        // poseidon_3_partial_rounds_chain_0: input state (cols 0-41)
        for (int i = 0; i < 42; i++) {
            input0[i] = trace_columns[i][row];
        }
        // poseidon_3_partial_rounds_chain_1: col0, col1+1, and output columns
        input1[0] = trace_columns[0][row];                // index unchanged
        input1[1] = add(trace_columns[1][row], (m31)1);   // round_num + 1
        // cube output[1] (cols 104-113)
        for (int i = 0; i < 10; i++) {
            input1[2 + i] = trace_columns[104 + i][row];
        }
        // combination[3] (cols 125-134)
        for (int i = 0; i < 10; i++) {
            input1[12 + i] = trace_columns[125 + i][row];
        }
        // cube output[2] (cols 136-145)
        for (int i = 0; i < 10; i++) {
            input1[22 + i] = trace_columns[136 + i][row];
        }
        // combination[5] (cols 157-166)
        for (int i = 0; i < 10; i++) {
            input1[32 + i] = trace_columns[157 + i][row];
        }

        qm31 denom0 = poseidon_3_partial_rounds_chain->combine(input0, 42);
        qm31 denom1 = poseidon_3_partial_rounds_chain->combine(input1, 42);
        // Numerator: (denom1 - denom0) * enabler
        qm31 numer = mul(sub(denom1, denom0), qm31{cm31{enabler, 0}, cm31{0, 0}});
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 8 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// =============================================================================
// Phase 3: Finalize interaction columns with accumulation
// =============================================================================

__global__ void poseidon_3_partial_rounds_chain_finalize_interaction_kernel(
    unsigned int trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces   // [36][trace_size] - 9 logup cols × 4 BaseField each
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Running sum of all fractions
    qm31 running_sum = qm31{cm31{0, 0}, cm31{0, 0}};

    // Write each of the 9 logup fractions with ACCUMULATION
    for (int i = 0; i < POSEIDON_3PRC_N_LOGUP_COLS; ++i) {
        unsigned int idx = i * trace_size + row;
        qm31 frac = mul(
            qm31 {
                cm31{numerator0[idx], numerator1[idx]},
                cm31{numerator2[idx], numerator3[idx]}
            },
            denom_inv_ptr[idx]
        );

        // Accumulate the fraction
        running_sum = add(running_sum, frac);

        // Each logup column occupies 4 consecutive columns
        int base_col = i * 4;
        interaction_traces[base_col + 0][row] = running_sum.a.a;
        interaction_traces[base_col + 1][row] = running_sum.a.b;
        interaction_traces[base_col + 2][row] = running_sum.b.a;
        interaction_traces[base_col + 3][row] = running_sum.b.b;
    }
}

// =============================================================================
// Phase 4: Compute cumulative sum (only sum last column)
// =============================================================================

__global__ void poseidon_3_partial_rounds_chain_cumsum_kernel(
    unsigned int trace_size,
    m31** interaction_traces,
    m31* coordinate_sums      // [4] - total sum
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = 0;
    m31 sum1 = 0;
    m31 sum2 = 0;
    m31 sum3 = 0;

    // Only sum the LAST column (index 8 = columns 32-35)
    int last_base_col = (POSEIDON_3PRC_N_LOGUP_COLS - 1) * 4;
    for (int i = tid; i < (int)trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_traces[last_base_col + 0][i]);
        sum1 = add(sum1, interaction_traces[last_base_col + 1][i]);
        sum2 = add(sum2, interaction_traces[last_base_col + 2][i]);
        sum3 = add(sum3, interaction_traces[last_base_col + 3][i]);
    }

    extern __shared__ m31 shared[];
    m31* sdata0 = &shared[0];
    m31* sdata1 = &shared[blockDim.x];
    m31* sdata2 = &shared[2 * blockDim.x];
    m31* sdata3 = &shared[3 * blockDim.x];

    sdata0[threadIdx.x] = sum0;
    sdata1[threadIdx.x] = sum1;
    sdata2[threadIdx.x] = sum2;
    sdata3[threadIdx.x] = sum3;

    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata0[threadIdx.x] = add(sdata0[threadIdx.x], sdata0[threadIdx.x + s]);
            sdata1[threadIdx.x] = add(sdata1[threadIdx.x], sdata1[threadIdx.x + s]);
            sdata2[threadIdx.x] = add(sdata2[threadIdx.x], sdata2[threadIdx.x + s]);
            sdata3[threadIdx.x] = add(sdata3[threadIdx.x], sdata3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sdata0[0]);
        atomic_add(&coordinate_sums[1], sdata1[0]);
        atomic_add(&coordinate_sums[2], sdata2[0]);
        atomic_add(&coordinate_sums[3], sdata3[0]);
    }
}

// =============================================================================
// Phase 5: Apply cumsum shift to last column
// =============================================================================

__global__ void poseidon_3_partial_rounds_chain_apply_cumsum_shift_kernel(
    m31* coordinate_sums,
    unsigned int trace_size,
    m31** interaction_traces
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    qm31 claimed_sum = qm31 {
        cm31{coordinate_sums[0], coordinate_sums[1]},
        cm31{coordinate_sums[2], coordinate_sums[3]}
    };
    qm31 cumsum_shift = div(claimed_sum, (m31)trace_size);

    // Only apply shift to the LAST column (32-35)
    int last_base_col = (POSEIDON_3PRC_N_LOGUP_COLS - 1) * 4;
    qm31 current = qm31 {
        cm31{interaction_traces[last_base_col + 0][row], interaction_traces[last_base_col + 1][row]},
        cm31{interaction_traces[last_base_col + 2][row], interaction_traces[last_base_col + 3][row]}
    };
    qm31 shifted = sub(current, cumsum_shift);
    interaction_traces[last_base_col + 0][row] = shifted.a.a;
    interaction_traces[last_base_col + 1][row] = shifted.a.b;
    interaction_traces[last_base_col + 2][row] = shifted.b.a;
    interaction_traces[last_base_col + 3][row] = shifted.b.b;
}

// =============================================================================
// Host wrapper function
// =============================================================================

extern "C" void poseidon_3_partial_rounds_chain_generate_interaction_trace(
    m31** trace_columns,
    unsigned int trace_size,
    void* cube_252_lookup_elements,
    void* poseidon_round_keys_lookup_elements,
    void* range_check_felt_252_width_27_lookup_elements,
    void* range_check_4_4_lookup_elements,
    void* range_check_4_4_4_4_lookup_elements,
    void* poseidon_3_partial_rounds_chain_lookup_elements,
    m31** interaction_trace_columns,
    qm31* claimed_sum
) {
    m31** device_trace_columns = clone_to_device<m31*>(trace_columns, POSEIDON_3_PARTIAL_ROUNDS_CHAIN_N_TRACE_COLUMNS);
    m31** device_interaction_traces = clone_to_device<m31*>(interaction_trace_columns, 4 * POSEIDON_3PRC_N_LOGUP_COLS);

    unsigned int n_fractions = POSEIDON_3PRC_N_LOGUP_COLS * trace_size;

    // Allocate temporary arrays for fraction computation
    qm31* denom_ptr;
    qm31* denom_inv;
    m31* numerator0;
    m31* numerator1;
    m31* numerator2;
    m31* numerator3;

    denom_ptr = cuda_mem_pool_allocate<qm31>(n_fractions);
    denom_inv = cuda_mem_pool_allocate<qm31>(n_fractions);
    numerator0 = cuda_mem_pool_allocate<m31>(n_fractions);
    numerator1 = cuda_mem_pool_allocate<m31>(n_fractions);
    numerator2 = cuda_mem_pool_allocate<m31>(n_fractions);
    numerator3 = cuda_mem_pool_allocate<m31>(n_fractions);

    // Clone lookup elements to device
    LookupElementsBasic<31>* d_poseidon_round_keys = clone_to_device<LookupElementsBasic<31>>((LookupElementsBasic<31>*)poseidon_round_keys_lookup_elements, 1);
    LookupElementsBasic<20>* d_cube_252 = clone_to_device<LookupElementsBasic<20>>((LookupElementsBasic<20>*)cube_252_lookup_elements, 1);
    LookupElementsBasic<4>* d_range_check_4_4_4_4 = clone_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)range_check_4_4_4_4_lookup_elements, 1);
    LookupElementsBasic<2>* d_range_check_4_4 = clone_to_device<LookupElementsBasic<2>>((LookupElementsBasic<2>*)range_check_4_4_lookup_elements, 1);
    LookupElementsBasic<10>* d_range_check_felt_252_width_27 = clone_to_device<LookupElementsBasic<10>>((LookupElementsBasic<10>*)range_check_felt_252_width_27_lookup_elements, 1);
    LookupElementsBasic<42>* d_poseidon_3_partial_rounds_chain = clone_to_device<LookupElementsBasic<42>>((LookupElementsBasic<42>*)poseidon_3_partial_rounds_chain_lookup_elements, 1);

    unsigned int block_size = 256;
    unsigned int grid_size = (trace_size + block_size - 1) / block_size;

    // Phase 1: Compute fractions
    poseidon_3_partial_rounds_chain_compute_fractions_kernel<<<grid_size, block_size>>>(
        device_trace_columns,
        trace_size,
        d_poseidon_round_keys, d_cube_252, d_range_check_4_4_4_4, d_range_check_4_4,
        d_range_check_felt_252_width_27, d_poseidon_3_partial_rounds_chain,
        denom_ptr, numerator0, numerator1, numerator2, numerator3
    );

    // Phase 2: Batch inverse on denominators
    batch_inverse_secure_field(denom_ptr, denom_inv, n_fractions);

    // Phase 3: Finalize with accumulation
    poseidon_3_partial_rounds_chain_finalize_interaction_kernel<<<grid_size, block_size>>>(
        trace_size,
        denom_inv,
        numerator0, numerator1, numerator2, numerator3,
        device_interaction_traces
    );

    // Phase 4: Compute cumulative sum (for claimed_sum)
    m31* d_coordinate_sums;
    d_coordinate_sums = cuda_mem_pool_allocate<m31>(4);
    cudaMemsetAsync(d_coordinate_sums, 0, 4 * sizeof(m31), 0);

    int cumsum_block_size = 256;
    int cumsum_grid_size = (trace_size + cumsum_block_size - 1) / cumsum_block_size;
    if (cumsum_grid_size > 256) cumsum_grid_size = 256;

    poseidon_3_partial_rounds_chain_cumsum_kernel<<<cumsum_grid_size, cumsum_block_size, 4 * cumsum_block_size * sizeof(m31)>>>(
        trace_size,
        device_interaction_traces,
        d_coordinate_sums
    );

    // Read claimed_sum from device and write to device output pointer
    m31 h_sums[4];
    cudaMemcpyAsync(h_sums, d_coordinate_sums, 4 * sizeof(m31), cudaMemcpyDeviceToHost, 0);
    qm31 h_claimed_sum = qm31{cm31{h_sums[0], h_sums[1]}, cm31{h_sums[2], h_sums[3]}};
    cudaMemcpyAsync(claimed_sum, &h_claimed_sum, sizeof(qm31), cudaMemcpyHostToDevice, 0);

    // Phase 5: Apply cumsum shift to last column
    poseidon_3_partial_rounds_chain_apply_cumsum_shift_kernel<<<grid_size, block_size>>>(
        d_coordinate_sums,
        trace_size,
        device_interaction_traces
    );

    // Phase 6: Inclusive prefix sum on last column (columns 32-35)
    int last_base_col = (POSEIDON_3PRC_N_LOGUP_COLS - 1) * 4;
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 0], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 1], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 2], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 3], trace_size);

    // Cleanup
    cuda_free_memory(device_trace_columns);
    cuda_free_memory(device_interaction_traces);
    cuda_mem_pool_free(denom_ptr);
    cuda_mem_pool_free(denom_inv);
    cuda_mem_pool_free(numerator0);
    cuda_mem_pool_free(numerator1);
    cuda_mem_pool_free(numerator2);
    cuda_mem_pool_free(numerator3);
    cuda_free_memory(d_poseidon_round_keys);
    cuda_free_memory(d_cube_252);
    cuda_free_memory(d_range_check_4_4_4_4);
    cuda_free_memory(d_range_check_4_4);
    cuda_free_memory(d_range_check_felt_252_width_27);
    cuda_free_memory(d_poseidon_3_partial_rounds_chain);
    cuda_mem_pool_free(d_coordinate_sums);
}
