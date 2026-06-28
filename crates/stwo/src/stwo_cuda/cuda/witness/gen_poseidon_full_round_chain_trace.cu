// CUDA trace generation for poseidon_full_round_chain component
// 126 trace columns, computes one full Poseidon round
//
// Trace layout:
// Columns 0-1: Input (index, round_number)
// Columns 2-31: Input states (3 x 10 Width27 limbs)
// Columns 32-61: Cube252 outputs (3 x 10 limbs)
// Columns 62-91: Poseidon round keys (3 x 10 limbs)
// Columns 92-101: Linear combination 1 result (10 limbs)
// Column 102: p_coef for combination 1
// Columns 103-112: Linear combination 2 result (10 limbs)
// Column 113: p_coef for combination 2
// Columns 114-123: Linear combination 3 result (10 limbs)
// Column 124: p_coef for combination 3
// Column 125: Enabler

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "gen_poseidon_full_round_chain_trace.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
#include <cstdint>
#include <cstdio>
#include "cuda_mem_pool.cuh"

#define POSEIDON_FRC_BLOCK_SIZE 256

// ============================================================================
// Felt252Field type and operations (same as cube_252)
// ============================================================================

typedef ff_storage<8> Felt252Field;

__device__ __forceinline__ Felt252Field poseidon_frc_felt_add(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

__device__ __forceinline__ Felt252Field poseidon_frc_felt_sub(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

__device__ __forceinline__ Felt252Field poseidon_frc_felt_mul(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

// Full reduction to canonical form [0, p)
// The ff_dispatch_st::add may return values >= p after multiple additions.
// We need to fully reduce before extracting Width27 limbs.
__device__ __forceinline__ Felt252Field poseidon_frc_felt_reduce(const Felt252Field& a) {
    // Apply reduce multiple times to handle cases where a >= 2p or more
    // The reduce<1> function subtracts p if a >= p
    // With 5 additions in the combination (3*cube0 + cube1 + cube2 + key), we need up to 5 reductions
    Felt252Field result = ff_dispatch_st<ff_config_starknet>::reduce<1>(a);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    result = ff_dispatch_st<ff_config_starknet>::reduce<1>(result);
    return result;
}

// Montgomery cube factor for x^3 correction
__device__ __constant__ Felt252Field POSEIDON_FRC_MONT_CUBE_FACTOR = {{
    0x406DF18E, 0xCC7177D1, 0x77FFCC06, 0x75457066,
    0x36300018, 0xF47D84F8, 0x873C0A6D, 0x038E5F79
}};

// Small constants for MDS matrix multiplication
__device__ __constant__ Felt252Field FELT252_ONE = {{1, 0, 0, 0, 0, 0, 0, 0}};
__device__ __constant__ Felt252Field FELT252_TWO = {{2, 0, 0, 0, 0, 0, 0, 0}};
__device__ __constant__ Felt252Field FELT252_THREE = {{3, 0, 0, 0, 0, 0, 0, 0}};

// ============================================================================
// Width27 to Felt252Field conversion (from cube_252)
// ============================================================================

__device__ Felt252Field width27_to_felt252_frc(const m31* limbs) {
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
__device__ void felt252_to_width27_frc(const Felt252Field& felt, m31* limbs) {
    uint64_t val0 = ((uint64_t)felt.limbs[1] << 32) | felt.limbs[0];
    uint64_t val1 = ((uint64_t)felt.limbs[3] << 32) | felt.limbs[2];
    uint64_t val2 = ((uint64_t)felt.limbs[5] << 32) | felt.limbs[4];
    uint64_t val3 = ((uint64_t)felt.limbs[7] << 32) | felt.limbs[6];

    // Extract 27-bit limbs
    limbs[0] = (m31){(uint32_t)(val0 & 0x7FFFFFF)};
    limbs[1] = (m31){(uint32_t)((val0 >> 27) & 0x7FFFFFF)};
    // limb 2 crosses val0/val1 boundary at bit 54
    uint64_t cross01 = (val0 >> 54) | (val1 << 10);
    limbs[2] = (m31){(uint32_t)(cross01 & 0x7FFFFFF)};
    limbs[3] = (m31){(uint32_t)((val1 >> 17) & 0x7FFFFFF)};
    // limb 4 crosses val1/val2 boundary at bit 44
    uint64_t cross12 = (val1 >> 44) | (val2 << 20);
    limbs[4] = (m31){(uint32_t)(cross12 & 0x7FFFFFF)};
    limbs[5] = (m31){(uint32_t)((val2 >> 7) & 0x7FFFFFF)};
    limbs[6] = (m31){(uint32_t)((val2 >> 34) & 0x7FFFFFF)};
    // limb 7 crosses val2/val3 boundary at bit 61
    uint64_t cross23 = (val2 >> 61) | (val3 << 3);
    limbs[7] = (m31){(uint32_t)(cross23 & 0x7FFFFFF)};
    limbs[8] = (m31){(uint32_t)((val3 >> 24) & 0x7FFFFFF)};
    limbs[9] = (m31){(uint32_t)((val3 >> 51) & 0x1FF)};  // Last limb is 9 bits for 252-bit total
}

// Compute cube (x^3) of a Felt252 value
__device__ Felt252Field compute_cube_felt252(const Felt252Field& x) {
    Felt252Field x2 = poseidon_frc_felt_mul(x, x);
    Felt252Field x3 = poseidon_frc_felt_mul(x2, x);
    // Apply Montgomery correction
    return poseidon_frc_felt_mul(x3, POSEIDON_FRC_MONT_CUBE_FACTOR);
}

// ============================================================================
// Linear combination with carry computation
// MDS matrix for Poseidon: out = M * cube_states + keys
// where M is:
//   [3  1  1]
//   [1 -1  1]
//   [1  1 -2]
// ============================================================================

// Compute linear combination result in Width27 format
// combo = c0 * cube0 + c1 * cube1 + c2 * cube2 + key
// where c0, c1, c2 are small integers (-2, -1, 1, 2, 3)
//
// Also computes p_coef using the formula:
//   diff = c0*cube0_limbs[0] + c1*cube1_limbs[0] + c2*cube2_limbs[0] + key_limbs[0] - combo_limbs[0]
//   p_coef = ((diff + bias) & 0xFFFF) - subtract
//
// Bias and subtract values for each round:
//   Round 0 (coefs 3,1,1):  bias = 134217729, subtract = 1
//   Round 1 (coefs 1,-1,1): bias = 268435458, subtract = 2
//   Round 2 (coefs 1,1,-2): bias = 402653187, subtract = 3
__device__ void compute_linear_combination(
    const Felt252Field& cube0,
    const Felt252Field& cube1,
    const Felt252Field& cube2,
    const Felt252Field& key,
    const m31* cube0_limbs,  // input: Width27 limbs of cube0
    const m31* cube1_limbs,  // input: Width27 limbs of cube1
    const m31* cube2_limbs,  // input: Width27 limbs of cube2
    const m31* key_limbs,    // input: Width27 limbs of key
    int c0, int c1, int c2,  // coefficients
    uint32_t bias,           // bias for p_coef computation
    uint32_t subtract,       // subtract value for p_coef
    m31* result_limbs,       // output: 10 Width27 limbs
    m31* p_coef              // output: p coefficient
) {
    // Apply coefficients using field operations
    Felt252Field result = key;  // Start with key

    // Add c0 * cube0
    if (c0 == 3) {
        result = poseidon_frc_felt_add(result, cube0);
        result = poseidon_frc_felt_add(result, cube0);
        result = poseidon_frc_felt_add(result, cube0);
    } else if (c0 == 1) {
        result = poseidon_frc_felt_add(result, cube0);
    } else if (c0 == -1) {
        result = poseidon_frc_felt_sub(result, cube0);
    } else if (c0 == -2) {
        result = poseidon_frc_felt_sub(result, cube0);
        result = poseidon_frc_felt_sub(result, cube0);
    }

    // Add c1 * cube1
    if (c1 == 1) {
        result = poseidon_frc_felt_add(result, cube1);
    } else if (c1 == -1) {
        result = poseidon_frc_felt_sub(result, cube1);
    }

    // Add c2 * cube2
    if (c2 == 1) {
        result = poseidon_frc_felt_add(result, cube2);
    } else if (c2 == -2) {
        result = poseidon_frc_felt_sub(result, cube2);
        result = poseidon_frc_felt_sub(result, cube2);
    }

    // Reduce to canonical form before extracting Width27 limbs
    result = poseidon_frc_felt_reduce(result);

    // Convert result to Width27 limbs
    felt252_to_width27_frc(result, result_limbs);

    // Compute p_coef for modular reduction verification
    // diff = c0*cube0[0] + c1*cube1[0] + c2*cube2[0] + key[0] - combo[0]
    // p_coef = ((diff + bias) & 0xFFFF) - subtract
    //
    // m31 is typedef'd to uint32_t, so we use the values directly.
    // We compute using signed arithmetic to handle negative coefficients.
    int64_t diff = 0;
    diff += (int64_t)c0 * (int64_t)cube0_limbs[0];
    diff += (int64_t)c1 * (int64_t)cube1_limbs[0];
    diff += (int64_t)c2 * (int64_t)cube2_limbs[0];
    diff += (int64_t)key_limbs[0];
    diff -= (int64_t)result_limbs[0];

    // Add bias and mask to 16 bits, then subtract using M31 field arithmetic
    // to handle underflow correctly (when biased < subtract)
    uint32_t biased = (uint32_t)((diff + (int64_t)bias) & 0xFFFF);
    *p_coef = sub((m31){biased}, (m31){subtract});
}

// ============================================================================
// Main trace generation kernel
// ============================================================================

__global__ void poseidon_full_round_chain_trace_kernel(
    m31* input_limb_0,              // Index values
    m31* input_limb_1,              // Round number values
    m31** state_0,                  // State[0]: 10 input columns
    m31** state_1,                  // State[1]: 10 input columns
    m31** state_2,                  // State[2]: 10 input columns
    unsigned int n_rows,
    m31** trace_columns,            // 126 output trace columns
    m31** poseidon_round_keys_table // 30 columns of round keys
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Read inputs
    m31 idx = input_limb_0[row];
    m31 round_num = input_limb_1[row];

    // Read 3 input states (each 10 Width27 limbs)
    m31 s0[10], s1[10], s2[10];
    for (int i = 0; i < 10; i++) {
        s0[i] = state_0[i][row];
        s1[i] = state_1[i][row];
        s2[i] = state_2[i][row];
    }

    // Write input columns (0-31)
    trace_columns[0][row] = idx;
    trace_columns[1][row] = round_num;
    for (int i = 0; i < 10; i++) {
        trace_columns[2 + i][row] = s0[i];      // Columns 2-11
        trace_columns[12 + i][row] = s1[i];     // Columns 12-21
        trace_columns[22 + i][row] = s2[i];     // Columns 22-31
    }

    // Convert states to Felt252Field
    Felt252Field f0 = width27_to_felt252_frc(s0);
    Felt252Field f1 = width27_to_felt252_frc(s1);
    Felt252Field f2 = width27_to_felt252_frc(s2);

    // Compute cubes (S-box application)
    Felt252Field cube0 = compute_cube_felt252(f0);
    Felt252Field cube1 = compute_cube_felt252(f1);
    Felt252Field cube2 = compute_cube_felt252(f2);

    // Convert cubes to Width27 and write to trace (columns 32-61)
    m31 cube0_limbs[10], cube1_limbs[10], cube2_limbs[10];
    felt252_to_width27_frc(cube0, cube0_limbs);
    felt252_to_width27_frc(cube1, cube1_limbs);
    felt252_to_width27_frc(cube2, cube2_limbs);

    for (int i = 0; i < 10; i++) {
        trace_columns[32 + i][row] = cube0_limbs[i];   // Columns 32-41
        trace_columns[42 + i][row] = cube1_limbs[i];   // Columns 42-51
        trace_columns[52 + i][row] = cube2_limbs[i];   // Columns 52-61
    }

    // Look up poseidon round keys (columns 62-91)
    // Round keys are indexed by round_num, output is 3 Felt252Width27 values (30 limbs)
    m31 key0_limbs[10], key1_limbs[10], key2_limbs[10];
    unsigned int round_idx = round_num;
    for (int i = 0; i < 10; i++) {
        key0_limbs[i] = poseidon_round_keys_table[i][round_idx];
        key1_limbs[i] = poseidon_round_keys_table[10 + i][round_idx];
        key2_limbs[i] = poseidon_round_keys_table[20 + i][round_idx];
    }

    // Write round keys to trace
    for (int i = 0; i < 10; i++) {
        trace_columns[62 + i][row] = key0_limbs[i];    // Columns 62-71
        trace_columns[72 + i][row] = key1_limbs[i];    // Columns 72-81
        trace_columns[82 + i][row] = key2_limbs[i];    // Columns 82-91
    }

    // Convert keys to Felt252Field
    Felt252Field key0 = width27_to_felt252_frc(key0_limbs);
    Felt252Field key1 = width27_to_felt252_frc(key1_limbs);
    Felt252Field key2 = width27_to_felt252_frc(key2_limbs);

    // Compute linear combinations (MDS matrix)
    // out0 = 3*cube0 + 1*cube1 + 1*cube2 + key0
    // out1 = 1*cube0 - 1*cube1 + 1*cube2 + key1
    // out2 = 1*cube0 + 1*cube1 - 2*cube2 + key2

    m31 combo1[10], combo2[10], combo3[10];
    m31 p_coef1, p_coef2, p_coef3;

    // Round 0: coefs (3,1,1), bias = 134217729, subtract = 1
    compute_linear_combination(cube0, cube1, cube2, key0,
                               cube0_limbs, cube1_limbs, cube2_limbs, key0_limbs,
                               3, 1, 1, 134217729u, 1u, combo1, &p_coef1);

    // Round 1: coefs (1,-1,1), bias = 268435458, subtract = 2
    compute_linear_combination(cube0, cube1, cube2, key1,
                               cube0_limbs, cube1_limbs, cube2_limbs, key1_limbs,
                               1, -1, 1, 268435458u, 2u, combo2, &p_coef2);

    // Round 2: coefs (1,1,-2), bias = 402653187, subtract = 3
    compute_linear_combination(cube0, cube1, cube2, key2,
                               cube0_limbs, cube1_limbs, cube2_limbs, key2_limbs,
                               1, 1, -2, 402653187u, 3u, combo3, &p_coef3);

    // Write linear combination results to trace
    for (int i = 0; i < 10; i++) {
        trace_columns[92 + i][row] = combo1[i];       // Columns 92-101
    }
    trace_columns[102][row] = p_coef1;                 // Column 102

    for (int i = 0; i < 10; i++) {
        trace_columns[103 + i][row] = combo2[i];      // Columns 103-112
    }
    trace_columns[113][row] = p_coef2;                 // Column 113

    for (int i = 0; i < 10; i++) {
        trace_columns[114 + i][row] = combo3[i];      // Columns 114-123
    }
    trace_columns[124][row] = p_coef3;                 // Column 124

    // Write enabler (column 125)
    trace_columns[125][row] = (m31){1};
}

// ============================================================================
// Host functions
// ============================================================================

extern "C" void poseidon_full_round_chain_generate_trace(
    m31* input_limb_0,
    m31* input_limb_1,
    m31** state_0,
    m31** state_1,
    m31** state_2,
    unsigned int n_rows,
    m31** trace_columns,
    m31** poseidon_round_keys_table
) {
    // Copy state pointers to device
    m31** d_state_0;
    m31** d_state_1;
    m31** d_state_2;
    d_state_0 = cuda_mem_pool_allocate<m31*>(10);
    d_state_1 = cuda_mem_pool_allocate<m31*>(10);
    d_state_2 = cuda_mem_pool_allocate<m31*>(10);
    cudaMemcpyAsync(d_state_0, state_0, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_state_1, state_1, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_state_2, state_2, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy trace column pointers to device
    m31** d_trace_columns;
    d_trace_columns = cuda_mem_pool_allocate<m31*>(POSEIDON_FULL_ROUND_CHAIN_N_TRACE_COLUMNS);
    cudaMemcpyAsync(d_trace_columns, trace_columns, POSEIDON_FULL_ROUND_CHAIN_N_TRACE_COLUMNS * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy round keys table pointers to device
    m31** d_round_keys;
    d_round_keys = cuda_mem_pool_allocate<m31*>(30);
    cudaMemcpyAsync(d_round_keys, poseidon_round_keys_table, 30 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Launch kernel
    int block_size = POSEIDON_FRC_BLOCK_SIZE;
    int num_blocks = (n_rows + block_size - 1) / block_size;

    poseidon_full_round_chain_trace_kernel<<<num_blocks, block_size>>>(
        input_limb_0,
        input_limb_1,
        d_state_0,
        d_state_1,
        d_state_2,
        n_rows,
        d_trace_columns,
        d_round_keys
    );


    // Cleanup
    cuda_mem_pool_free(d_state_0);
    cuda_mem_pool_free(d_state_1);
    cuda_mem_pool_free(d_state_2);
    cuda_mem_pool_free(d_trace_columns);
    cuda_mem_pool_free(d_round_keys);
}

// ============================================================================
// Carry computation kernel for add_to_multiplicities
// Reads trace columns from GPU, computes carries, writes 30 output arrays.
// ============================================================================

// Generic carry computation for one linear combination.
// val[i] = c0*cube0[i] + c1*cube1[i] + c2*cube2[i] + key[i] - combo[i]
// carry[0] = (val[0] - p_coef) * 16
// carry[i] = (carry[i-1] + val[i]) * 16   for i in 1..6
// carry[7] = (carry[6] + val[7] - p_coef * 136) * 16
// carry[8] = (carry[7] + val[8]) * 16
__device__ void compute_pfrc_carries(
    m31* cube0, m31* cube1, m31* cube2, m31* key, m31* combo,
    int c0, int c1, int c2,
    m31 p_coef,
    m31* carries  // output: 9 carries
) {
    // Helper lambda-like inline: compute val = c0*cube0[i] + c1*cube1[i] + c2*cube2[i] + key[i] - combo[i]
    #define PFRC_VAL(i) \
        sub(add(add(add( \
            (c0 == 3 ? add(add(mul((m31)3, cube0[(i)]), (m31)0), (m31)0) : \
             c0 == 1 ? cube0[(i)] : cube0[(i)]), \
            (c1 == 1 ? cube1[(i)] : (c1 == -1 ? sub((m31)0, cube1[(i)]) : cube1[(i)])), \
            (c2 == 1 ? cube2[(i)] : (c2 == -2 ? sub((m31)0, add(cube2[(i)], cube2[(i)])) : cube2[(i)]))), \
            key[(i)]), combo[(i)])

    // Actually, let's be more explicit to avoid macro complexity:
    // We compute val directly based on coefficients.
    auto compute_val = [&](int i) -> m31 {
        m31 result = key[i];
        if (c0 == 3) result = add(result, add(add(cube0[i], cube0[i]), cube0[i]));
        else if (c0 == 1) result = add(result, cube0[i]);
        if (c1 == 1) result = add(result, cube1[i]);
        else if (c1 == -1) result = sub(result, cube1[i]);
        if (c2 == 1) result = add(result, cube2[i]);
        else if (c2 == -2) result = sub(sub(result, cube2[i]), cube2[i]);
        result = sub(result, combo[i]);
        return result;
    };

    #undef PFRC_VAL

    // carry[0] = (val[0] - p_coef) * 16
    carries[0] = mul(sub(compute_val(0), p_coef), (m31)16);

    // carries 1-6: carry[i] = (carry[i-1] + val[i]) * 16
    for (int i = 1; i < 7; i++) {
        carries[i] = mul(add(carries[i-1], compute_val(i)), (m31)16);
    }

    // carry[7]: special case with p_coef * 136
    carries[7] = mul(sub(add(carries[6], compute_val(7)), mul(p_coef, (m31)136)), (m31)16);

    // carry[8]: final
    carries[8] = mul(add(carries[7], compute_val(8)), (m31)16);
}

__global__ void poseidon_full_round_chain_compute_rc_inputs_kernel(
    m31** trace_columns,
    unsigned int n_rows,
    m31** output_arrays  // 30 output arrays (6 sets × 5 columns)
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Read trace columns
    m31 cube0[10], cube1[10], cube2[10];
    m31 key0[10], key1[10], key2[10];
    m31 combo0[10], combo1[10], combo2[10];

    for (int i = 0; i < 10; i++) {
        cube0[i] = trace_columns[32 + i][row];
        cube1[i] = trace_columns[42 + i][row];
        cube2[i] = trace_columns[52 + i][row];
        key0[i] = trace_columns[62 + i][row];
        key1[i] = trace_columns[72 + i][row];
        key2[i] = trace_columns[82 + i][row];
        combo0[i] = trace_columns[92 + i][row];
        combo1[i] = trace_columns[103 + i][row];
        combo2[i] = trace_columns[114 + i][row];
    }

    m31 p_coef_0 = trace_columns[102][row];
    m31 p_coef_1 = trace_columns[113][row];
    m31 p_coef_2 = trace_columns[124][row];

    m31 carries[9];

    // ========== Linear Combination 0: Coefs (3, 1, 1) ==========
    compute_pfrc_carries(cube0, cube1, cube2, key0, combo0, 3, 1, 1, p_coef_0, carries);

    // Output set 0: [p_coef_0+1, carry0+1, carry1+1, carry2+1, carry3+1]
    output_arrays[0][row] = add(p_coef_0, (m31)1);
    output_arrays[1][row] = add(carries[0], (m31)1);
    output_arrays[2][row] = add(carries[1], (m31)1);
    output_arrays[3][row] = add(carries[2], (m31)1);
    output_arrays[4][row] = add(carries[3], (m31)1);

    // Output set 1: [carry4+1, carry5+1, carry6+1, carry7+1, carry8+1]
    output_arrays[5][row] = add(carries[4], (m31)1);
    output_arrays[6][row] = add(carries[5], (m31)1);
    output_arrays[7][row] = add(carries[6], (m31)1);
    output_arrays[8][row] = add(carries[7], (m31)1);
    output_arrays[9][row] = add(carries[8], (m31)1);

    // ========== Linear Combination 1: Coefs (1, -1, 1) ==========
    compute_pfrc_carries(cube0, cube1, cube2, key1, combo1, 1, -1, 1, p_coef_1, carries);

    // Output set 2: [p_coef_1+2, carry0+2, carry1+2, carry2+2, carry3+2]
    output_arrays[10][row] = add(p_coef_1, (m31)2);
    output_arrays[11][row] = add(carries[0], (m31)2);
    output_arrays[12][row] = add(carries[1], (m31)2);
    output_arrays[13][row] = add(carries[2], (m31)2);
    output_arrays[14][row] = add(carries[3], (m31)2);

    // Output set 3: [carry4+2, carry5+2, carry6+2, carry7+2, carry8+2]
    output_arrays[15][row] = add(carries[4], (m31)2);
    output_arrays[16][row] = add(carries[5], (m31)2);
    output_arrays[17][row] = add(carries[6], (m31)2);
    output_arrays[18][row] = add(carries[7], (m31)2);
    output_arrays[19][row] = add(carries[8], (m31)2);

    // ========== Linear Combination 2: Coefs (1, 1, -2) ==========
    compute_pfrc_carries(cube0, cube1, cube2, key2, combo2, 1, 1, -2, p_coef_2, carries);

    // Output set 4: [p_coef_2+3, carry0+3, carry1+3, carry2+3, carry3+3]
    output_arrays[20][row] = add(p_coef_2, (m31)3);
    output_arrays[21][row] = add(carries[0], (m31)3);
    output_arrays[22][row] = add(carries[1], (m31)3);
    output_arrays[23][row] = add(carries[2], (m31)3);
    output_arrays[24][row] = add(carries[3], (m31)3);

    // Output set 5: [carry4+3, carry5+3, carry6+3, carry7+3, carry8+3]
    output_arrays[25][row] = add(carries[4], (m31)3);
    output_arrays[26][row] = add(carries[5], (m31)3);
    output_arrays[27][row] = add(carries[6], (m31)3);
    output_arrays[28][row] = add(carries[7], (m31)3);
    output_arrays[29][row] = add(carries[8], (m31)3);
}

extern "C" void poseidon_full_round_chain_compute_rc_inputs(
    m31** trace_columns,
    unsigned int n_rows,
    m31** output_arrays  // 30 output device pointers
) {
    if (n_rows == 0) return;

    // Copy pointer arrays to device
    m31** d_trace_columns = clone_to_device<m31*>(trace_columns, POSEIDON_FULL_ROUND_CHAIN_N_TRACE_COLUMNS);
    m31** d_output_arrays = clone_to_device<m31*>(output_arrays, 30);

    unsigned int block_size = 256;
    unsigned int grid_size = (n_rows + block_size - 1) / block_size;

    poseidon_full_round_chain_compute_rc_inputs_kernel<<<grid_size, block_size>>>(
        d_trace_columns,
        n_rows,
        d_output_arrays
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(d_trace_columns);
    cuda_free_memory(d_output_arrays);
}

extern "C" void poseidon_full_round_chain_add_to_multiplicities(
    m31** trace_columns,
    unsigned int n_rows,
    m31* cube_252_mults,
    unsigned int cube_252_log_size,
    m31* poseidon_round_keys_mults,
    m31* rc_3_3_3_3_3_mults,
    unsigned int rc_3_3_3_3_3_log_size
) {
    // Stub kept for backward compatibility. Use poseidon_full_round_chain_compute_rc_inputs instead.
}

// =============================================================================
// Phase 2: Compute LogUp Fractions
// =============================================================================

#define POSEIDON_FRC_N_LOGUP_COLS 6

__global__ void poseidon_full_round_chain_compute_fractions_kernel(
    m31** trace_columns,
    unsigned int trace_size,
    // Lookup elements for each relation
    LookupElementsBasic<20>* cube_252,           // 20 values per lookup
    LookupElementsBasic<31>* poseidon_round_keys, // 31 values per lookup
    LookupElementsBasic<5>* range_check_3_3_3_3_3, // 5 values per lookup
    LookupElementsBasic<32>* poseidon_full_round_chain, // 32 values per lookup
    // Output arrays for fractions
    qm31* denom_ptr,          // [6 * trace_size]
    m31* numerator0,          // [6 * trace_size]
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Read trace values needed for lookups
    // Trace layout from header:
    // Columns 0-1: Input (index, round_number)
    // Columns 2-31: Input states (3 x 10 Width27 limbs)
    // Columns 32-61: Cube252 outputs (3 x 10 limbs)
    // Columns 62-91: Poseidon round keys (3 x 10 limbs)
    // Columns 92-101: Linear combination result 1 (10 limbs)
    // Column 102: p_coef for combination 1
    // Columns 103-112: Linear combination result 2 (10 limbs)
    // Column 113: p_coef for combination 2
    // Columns 114-123: Linear combination result 3 (10 limbs)
    // Column 124: p_coef for combination 3
    // Column 125: Enabler

    m31 enabler = trace_columns[125][row];

    // LogUp column 0: cube_252_0 + cube_252_1
    // cube_252_0: input state[0] (cols 2-11) + cube output[0] (cols 32-41)
    // cube_252_1: input state[1] (cols 12-21) + cube output[1] (cols 42-51)
    {
        m31 input0[20], input1[20];
        // cube_252_0: state[0] input (cols 2-11) + cube output (cols 32-41)
        for (int i = 0; i < 10; i++) {
            input0[i] = trace_columns[2 + i][row];      // state[0] input
            input0[10 + i] = trace_columns[32 + i][row]; // cube[0] output
        }
        // cube_252_1: state[1] input (cols 12-21) + cube output (cols 42-51)
        for (int i = 0; i < 10; i++) {
            input1[i] = trace_columns[12 + i][row];     // state[1] input
            input1[10 + i] = trace_columns[42 + i][row]; // cube[1] output
        }
        qm31 denom0 = cube_252->combine(input0, 20);
        qm31 denom1 = cube_252->combine(input1, 20);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 0 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 1: cube_252_2 + poseidon_round_keys_0
    // cube_252_2: input state[2] (cols 22-31) + cube output[2] (cols 52-61)
    // poseidon_round_keys_0: round_num (col 1) + keys (cols 62-91)
    {
        m31 input0[20], input1[31];
        // cube_252_2: state[2] input (cols 22-31) + cube output (cols 52-61)
        for (int i = 0; i < 10; i++) {
            input0[i] = trace_columns[22 + i][row];     // state[2] input
            input0[10 + i] = trace_columns[52 + i][row]; // cube[2] output
        }
        // poseidon_round_keys_0: round_num (col 1) + keys (cols 62-91)
        input1[0] = trace_columns[1][row]; // round_number
        for (int i = 0; i < 30; i++) {
            input1[1 + i] = trace_columns[62 + i][row]; // round keys
        }
        qm31 denom0 = cube_252->combine(input0, 20);
        qm31 denom1 = poseidon_round_keys->combine(input1, 31);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 1 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 2: range_check_3_3_3_3_3_0 + range_check_3_3_3_3_3_1
    // These are biased carry values from the first linear combination
    // rc_0: p_coef (col 102) + carries 0-3 (computed values, biased by +1)
    // rc_1: carries 4-8 (computed values, biased by +1)
    // The carry values need to be recomputed from the trace - but for the lookup,
    // they are stored in the lookup_data during trace generation.
    // However, the SIMD implementation computes them on-the-fly from the trace values.
    // For now, we extract them using the same formula as the SIMD code.
    {
        // First linear combination: 3*cube0 + cube1 + cube2 + key0
        // p_coef is at col 102
        m31 p_coef = trace_columns[102][row];

        // We need to compute carries from the trace values
        // carry_0 = (3*cube0[0] + cube1[0] + cube2[0] + key0[0] - combo1[0] - p_coef) * 16
        // This is complex; for the lookup we need the biased values (carry + bias)

        // The SIMD code uses biased values with +1 for the first combination
        // For simplicity and correctness, we read the values that would be stored
        // Actually, the range checks are applied to biased carry values
        // The carry values are NOT stored in the trace; they're computed transiently

        // In the SIMD implementation, range_check_3_3_3_3_3_0 and _1 are populated with:
        // rc_0 = [p_coef+1, carry_0+1, carry_1+1, carry_2+1, carry_3+1]
        // rc_1 = [carry_4+1, carry_5+1, carry_6+1, carry_7+1, carry_8+1]

        // We need to recompute the carries. Let's use M31_16 = 16 for carry extraction.
        m31 cube0[10], cube1[10], cube2[10], key0[10], combo1[10];
        for (int i = 0; i < 10; i++) {
            cube0[i] = trace_columns[32 + i][row];  // cube[0]
            cube1[i] = trace_columns[42 + i][row];  // cube[1]
            cube2[i] = trace_columns[52 + i][row];  // cube[2]
            key0[i] = trace_columns[62 + i][row];   // key[0]
            combo1[i] = trace_columns[92 + i][row]; // combination 1
        }

        // Compute carries for first combination (coefficients: 3, 1, 1, 1)
        // Following the SIMD pattern with M31_16 = 16 for carry extraction
        m31 carry_0 = mul(sub(sub(add(add(add(mul((m31)3, cube0[0]), cube1[0]), cube2[0]), key0[0]), combo1[0]), p_coef), (m31)16);
        m31 carry_1 = mul(sub(add(add(add(add(carry_0, mul((m31)3, cube0[1])), cube1[1]), cube2[1]), key0[1]), combo1[1]), (m31)16);
        m31 carry_2 = mul(sub(add(add(add(add(carry_1, mul((m31)3, cube0[2])), cube1[2]), cube2[2]), key0[2]), combo1[2]), (m31)16);
        m31 carry_3 = mul(sub(add(add(add(add(carry_2, mul((m31)3, cube0[3])), cube1[3]), cube2[3]), key0[3]), combo1[3]), (m31)16);
        m31 carry_4 = mul(sub(add(add(add(add(carry_3, mul((m31)3, cube0[4])), cube1[4]), cube2[4]), key0[4]), combo1[4]), (m31)16);
        m31 carry_5 = mul(sub(add(add(add(add(carry_4, mul((m31)3, cube0[5])), cube1[5]), cube2[5]), key0[5]), combo1[5]), (m31)16);
        m31 carry_6 = mul(sub(add(add(add(add(carry_5, mul((m31)3, cube0[6])), cube1[6]), cube2[6]), key0[6]), combo1[6]), (m31)16);
        // carry_7 has p_coef * 136 subtraction (for modular reduction)
        m31 carry_7 = mul(sub(sub(add(add(add(add(carry_6, mul((m31)3, cube0[7])), cube1[7]), cube2[7]), key0[7]), combo1[7]), mul(p_coef, (m31)136)), (m31)16);
        m31 carry_8 = mul(sub(add(add(add(add(carry_7, mul((m31)3, cube0[8])), cube1[8]), cube2[8]), key0[8]), combo1[8]), (m31)16);

        // Biased values (add 1 for first combination)
        m31 input0[5] = {add(p_coef, (m31)1), add(carry_0, (m31)1), add(carry_1, (m31)1), add(carry_2, (m31)1), add(carry_3, (m31)1)};
        m31 input1[5] = {add(carry_4, (m31)1), add(carry_5, (m31)1), add(carry_6, (m31)1), add(carry_7, (m31)1), add(carry_8, (m31)1)};

        qm31 denom0 = range_check_3_3_3_3_3->combine(input0, 5);
        qm31 denom1 = range_check_3_3_3_3_3->combine(input1, 5);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 2 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 3: range_check_3_3_3_3_3_2 + range_check_3_3_3_3_3_3
    // Second linear combination (coefficients: 1, -1, 1, 1)
    {
        m31 p_coef = trace_columns[113][row];

        m31 cube0[10], cube1[10], cube2[10], key1[10], combo2[10];
        for (int i = 0; i < 10; i++) {
            cube0[i] = trace_columns[32 + i][row];
            cube1[i] = trace_columns[42 + i][row];
            cube2[i] = trace_columns[52 + i][row];
            key1[i] = trace_columns[72 + i][row];   // key[1]
            combo2[i] = trace_columns[103 + i][row]; // combination 2
        }

        // Coefficients: 1, -1, 1, 1 (cube0 - cube1 + cube2 + key1)
        m31 carry_0 = mul(sub(sub(add(add(sub(cube0[0], cube1[0]), cube2[0]), key1[0]), combo2[0]), p_coef), (m31)16);
        m31 carry_1 = mul(sub(add(sub(add(carry_0, cube0[1]), cube1[1]), cube2[1]), sub(combo2[1], key1[1])), (m31)16);
        m31 carry_2 = mul(sub(add(sub(add(carry_1, cube0[2]), cube1[2]), cube2[2]), sub(combo2[2], key1[2])), (m31)16);
        m31 carry_3 = mul(sub(add(sub(add(carry_2, cube0[3]), cube1[3]), cube2[3]), sub(combo2[3], key1[3])), (m31)16);
        m31 carry_4 = mul(sub(add(sub(add(carry_3, cube0[4]), cube1[4]), cube2[4]), sub(combo2[4], key1[4])), (m31)16);
        m31 carry_5 = mul(sub(add(sub(add(carry_4, cube0[5]), cube1[5]), cube2[5]), sub(combo2[5], key1[5])), (m31)16);
        m31 carry_6 = mul(sub(add(sub(add(carry_5, cube0[6]), cube1[6]), cube2[6]), sub(combo2[6], key1[6])), (m31)16);
        m31 carry_7 = mul(sub(sub(add(sub(add(carry_6, cube0[7]), cube1[7]), cube2[7]), sub(combo2[7], key1[7])), mul(p_coef, (m31)136)), (m31)16);
        m31 carry_8 = mul(sub(add(sub(add(carry_7, cube0[8]), cube1[8]), cube2[8]), sub(combo2[8], key1[8])), (m31)16);

        // Biased values (add 2 for second combination)
        m31 input0[5] = {add(p_coef, (m31)2), add(carry_0, (m31)2), add(carry_1, (m31)2), add(carry_2, (m31)2), add(carry_3, (m31)2)};
        m31 input1[5] = {add(carry_4, (m31)2), add(carry_5, (m31)2), add(carry_6, (m31)2), add(carry_7, (m31)2), add(carry_8, (m31)2)};

        qm31 denom0 = range_check_3_3_3_3_3->combine(input0, 5);
        qm31 denom1 = range_check_3_3_3_3_3->combine(input1, 5);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 3 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 4: range_check_3_3_3_3_3_4 + range_check_3_3_3_3_3_5
    // Third linear combination (coefficients: 1, 1, -2, 1)
    {
        m31 p_coef = trace_columns[124][row];

        m31 cube0[10], cube1[10], cube2[10], key2[10], combo3[10];
        for (int i = 0; i < 10; i++) {
            cube0[i] = trace_columns[32 + i][row];
            cube1[i] = trace_columns[42 + i][row];
            cube2[i] = trace_columns[52 + i][row];
            key2[i] = trace_columns[82 + i][row];   // key[2]
            combo3[i] = trace_columns[114 + i][row]; // combination 3
        }

        // Coefficients: 1, 1, -2, 1 (cube0 + cube1 - 2*cube2 + key2)
        m31 carry_0 = mul(sub(sub(sub(add(add(cube0[0], cube1[0]), key2[0]), mul((m31)2, cube2[0])), combo3[0]), p_coef), (m31)16);
        m31 carry_1 = mul(sub(sub(add(add(add(carry_0, cube0[1]), cube1[1]), key2[1]), mul((m31)2, cube2[1])), combo3[1]), (m31)16);
        m31 carry_2 = mul(sub(sub(add(add(add(carry_1, cube0[2]), cube1[2]), key2[2]), mul((m31)2, cube2[2])), combo3[2]), (m31)16);
        m31 carry_3 = mul(sub(sub(add(add(add(carry_2, cube0[3]), cube1[3]), key2[3]), mul((m31)2, cube2[3])), combo3[3]), (m31)16);
        m31 carry_4 = mul(sub(sub(add(add(add(carry_3, cube0[4]), cube1[4]), key2[4]), mul((m31)2, cube2[4])), combo3[4]), (m31)16);
        m31 carry_5 = mul(sub(sub(add(add(add(carry_4, cube0[5]), cube1[5]), key2[5]), mul((m31)2, cube2[5])), combo3[5]), (m31)16);
        m31 carry_6 = mul(sub(sub(add(add(add(carry_5, cube0[6]), cube1[6]), key2[6]), mul((m31)2, cube2[6])), combo3[6]), (m31)16);
        m31 carry_7 = mul(sub(sub(sub(add(add(add(carry_6, cube0[7]), cube1[7]), key2[7]), mul((m31)2, cube2[7])), combo3[7]), mul(p_coef, (m31)136)), (m31)16);
        m31 carry_8 = mul(sub(sub(add(add(add(carry_7, cube0[8]), cube1[8]), key2[8]), mul((m31)2, cube2[8])), combo3[8]), (m31)16);

        // Biased values (add 3 for third combination)
        m31 input0[5] = {add(p_coef, (m31)3), add(carry_0, (m31)3), add(carry_1, (m31)3), add(carry_2, (m31)3), add(carry_3, (m31)3)};
        m31 input1[5] = {add(carry_4, (m31)3), add(carry_5, (m31)3), add(carry_6, (m31)3), add(carry_7, (m31)3), add(carry_8, (m31)3)};

        qm31 denom0 = range_check_3_3_3_3_3->combine(input0, 5);
        qm31 denom1 = range_check_3_3_3_3_3->combine(input1, 5);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 4 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 5: poseidon_full_round_chain self-relation with enabler
    // Input: all 32 input limbs (cols 0-31)
    // Output: col0, col1+1, and 30 combination limbs
    {
        m31 input0[32], input1[32];
        // poseidon_full_round_chain_0: input state (cols 0-31)
        for (int i = 0; i < 32; i++) {
            input0[i] = trace_columns[i][row];
        }
        // poseidon_full_round_chain_1: col0, col1+1, combination results
        input1[0] = trace_columns[0][row];                // index unchanged
        input1[1] = add(trace_columns[1][row], (m31)1);   // round_num + 1
        // combination 1 (cols 92-101)
        for (int i = 0; i < 10; i++) {
            input1[2 + i] = trace_columns[92 + i][row];
        }
        // combination 2 (cols 103-112)
        for (int i = 0; i < 10; i++) {
            input1[12 + i] = trace_columns[103 + i][row];
        }
        // combination 3 (cols 114-123)
        for (int i = 0; i < 10; i++) {
            input1[22 + i] = trace_columns[114 + i][row];
        }

        qm31 denom0 = poseidon_full_round_chain->combine(input0, 32);
        qm31 denom1 = poseidon_full_round_chain->combine(input1, 32);
        // Numerator: (denom1 - denom0) * enabler
        qm31 numer = mul(sub(denom1, denom0), qm31{cm31{enabler, 0}, cm31{0, 0}});
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 5 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// =============================================================================
// Phase 3: Finalize interaction columns with accumulation
// =============================================================================

__global__ void poseidon_full_round_chain_finalize_interaction_kernel(
    unsigned int trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces   // [24][trace_size] - 6 logup cols × 4 BaseField each
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Running sum of all fractions
    qm31 running_sum = qm31{cm31{0, 0}, cm31{0, 0}};

    // Write each of the 6 logup fractions with ACCUMULATION
    for (int i = 0; i < POSEIDON_FRC_N_LOGUP_COLS; ++i) {
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

__global__ void poseidon_full_round_chain_cumsum_kernel(
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

    // Only sum the LAST column (index 5 = columns 20-23)
    int last_base_col = (POSEIDON_FRC_N_LOGUP_COLS - 1) * 4;
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

__global__ void poseidon_full_round_chain_apply_cumsum_shift_kernel(
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

    // Only apply shift to the LAST column (20-23)
    int last_base_col = (POSEIDON_FRC_N_LOGUP_COLS - 1) * 4;
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

extern "C" void poseidon_full_round_chain_generate_interaction_trace(
    m31** trace_columns,
    unsigned int trace_size,
    void* cube_252_lookup_elements,
    void* poseidon_round_keys_lookup_elements,
    void* range_check_3_3_3_3_3_lookup_elements,
    void* poseidon_full_round_chain_lookup_elements,
    m31** interaction_trace_columns,
    qm31* claimed_sum
) {
    m31** device_trace_columns = clone_to_device<m31*>(trace_columns, POSEIDON_FULL_ROUND_CHAIN_N_TRACE_COLUMNS);
    m31** device_interaction_traces = clone_to_device<m31*>(interaction_trace_columns, 4 * POSEIDON_FRC_N_LOGUP_COLS);

    unsigned int n_fractions = POSEIDON_FRC_N_LOGUP_COLS * trace_size;

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
    LookupElementsBasic<20>* d_cube_252 = clone_to_device<LookupElementsBasic<20>>((LookupElementsBasic<20>*)cube_252_lookup_elements, 1);
    LookupElementsBasic<31>* d_poseidon_round_keys = clone_to_device<LookupElementsBasic<31>>((LookupElementsBasic<31>*)poseidon_round_keys_lookup_elements, 1);
    LookupElementsBasic<5>* d_range_check_3_3_3_3_3 = clone_to_device<LookupElementsBasic<5>>((LookupElementsBasic<5>*)range_check_3_3_3_3_3_lookup_elements, 1);
    LookupElementsBasic<32>* d_poseidon_full_round_chain = clone_to_device<LookupElementsBasic<32>>((LookupElementsBasic<32>*)poseidon_full_round_chain_lookup_elements, 1);

    unsigned int block_size = 256;
    unsigned int grid_size = (trace_size + block_size - 1) / block_size;

    // Phase 1: Compute fractions
    poseidon_full_round_chain_compute_fractions_kernel<<<grid_size, block_size>>>(
        device_trace_columns,
        trace_size,
        d_cube_252, d_poseidon_round_keys, d_range_check_3_3_3_3_3, d_poseidon_full_round_chain,
        denom_ptr, numerator0, numerator1, numerator2, numerator3
    );

    // Phase 2: Batch inverse on denominators
    batch_inverse_secure_field(denom_ptr, denom_inv, n_fractions);

    // Phase 3: Finalize with accumulation
    poseidon_full_round_chain_finalize_interaction_kernel<<<grid_size, block_size>>>(
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

    poseidon_full_round_chain_cumsum_kernel<<<cumsum_grid_size, cumsum_block_size, 4 * cumsum_block_size * sizeof(m31)>>>(
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
    poseidon_full_round_chain_apply_cumsum_shift_kernel<<<grid_size, block_size>>>(
        d_coordinate_sums,
        trace_size,
        device_interaction_traces
    );

    // Phase 6: Inclusive prefix sum on last column (columns 20-23)
    int last_base_col = (POSEIDON_FRC_N_LOGUP_COLS - 1) * 4;
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
    cuda_free_memory(d_cube_252);
    cuda_free_memory(d_poseidon_round_keys);
    cuda_free_memory(d_range_check_3_3_3_3_3);
    cuda_free_memory(d_poseidon_full_round_chain);
    cuda_mem_pool_free(d_coordinate_sums);
}
