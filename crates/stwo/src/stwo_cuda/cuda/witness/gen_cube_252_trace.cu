// CUDA trace generation for cube_252 component
// 141 trace columns, computes x^3 for Felt252 values
//
// Trace layout:
// Columns 0-9: Input limbs (10 x 27-bit)
// Columns 10-27: Unpacked limbs (18 columns)
// Columns 28-55: First mul result x² (28 limbs)
// Column 56: k1 (first mul)
// Columns 57-83: carry1 (27 values)
// Columns 84-111: Second mul result x³ (28 limbs)
// Column 112: k2 (second mul)
// Columns 113-139: carry2 (27 values)
// Column 140: enabler

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "gen_cube_252_trace.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
#include <cstdint>
#include <cstdio>
#include "cuda_mem_pool.cuh"

#define CUBE_252_BLOCK_SIZE 256

// ============================================================================
// Felt252Field type and operations (same as gen_poseidon_builtin_trace.cu)
// ============================================================================

typedef ff_storage<8> Felt252Field;

__device__ __forceinline__ Felt252Field cube252_felt_add(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

__device__ __forceinline__ Felt252Field cube252_felt_sub(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

__device__ __forceinline__ Felt252Field cube252_felt_mul(const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

// Montgomery multiplication factor: compensates for ONE Montgomery multiplication
// Equal to 2^512 % PRIME (from Rust FELT252_MONT_MUL_FACTOR)
// Used for x² = x * x
__device__ __constant__ Felt252Field CUBE252_MONT_MUL_FACTOR = {{
    0x7E000401, 0xFFFFFD73, 0x330FFFFF, 0x00000001,
    0xFF6F8000, 0xFFFFFFFF, 0x5E008810, 0x07FFD4AB
}};

// Montgomery cube factor: compensates for Montgomery form cubing
// Equal to 2^768 % PRIME (from Rust FELT252_MONT_CUBE_FACTOR)
// This is used for the full x³ computation, but we apply MUL_FACTOR to x² first,
// so for x³ we only need one more MUL_FACTOR (since x³ = x² * x)
__device__ __constant__ Felt252Field CUBE252_MONT_CUBE_FACTOR = {{
    0x406DF18E, 0xCC7177D1, 0x77FFCC06, 0x75457066,
    0x36300018, 0xF47D84F8, 0x873C0A6D, 0x038E5F79
}};

// ============================================================================
// Width27 to Felt252Field conversion
// ============================================================================

// Convert 10 M31 Width27 limbs to Felt252Field
__device__ Felt252Field width27_to_felt252(const m31* limbs) {
    uint64_t val0 = 0, val1 = 0, val2 = 0, val3 = 0;

    // Pack 27-bit limbs into 64-bit values
    val0 = (uint64_t)limbs[0];
    val0 |= ((uint64_t)limbs[1]) << 27;
    val0 |= ((uint64_t)limbs[2]) << 54;  // 10 bits overflow to val1

    val1 = ((uint64_t)limbs[2]) >> 10;   // remaining 17 bits of limb2
    val1 |= ((uint64_t)limbs[3]) << 17;
    val1 |= ((uint64_t)limbs[4]) << 44;  // 20 bits overflow to val2

    val2 = ((uint64_t)limbs[4]) >> 20;   // remaining 7 bits of limb4
    val2 |= ((uint64_t)limbs[5]) << 7;
    val2 |= ((uint64_t)limbs[6]) << 34;
    val2 |= ((uint64_t)limbs[7]) << 61;  // 3 bits overflow to val3

    val3 = ((uint64_t)limbs[7]) >> 3;    // remaining 24 bits of limb7
    val3 |= ((uint64_t)limbs[8]) << 24;
    val3 |= ((uint64_t)limbs[9]) << 51;  // last 9-bit limb

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

// Convert Felt252Field to 28 x 9-bit limbs
__device__ void felt252_to_limbs28(const Felt252Field& felt, m31* limbs) {
    uint64_t val0 = ((uint64_t)felt.limbs[1] << 32) | felt.limbs[0];
    uint64_t val1 = ((uint64_t)felt.limbs[3] << 32) | felt.limbs[2];
    uint64_t val2 = ((uint64_t)felt.limbs[5] << 32) | felt.limbs[4];
    uint64_t val3 = ((uint64_t)felt.limbs[7] << 32) | felt.limbs[6];

    // Extract 9-bit limbs
    for (int i = 0; i < 7; i++) {
        limbs[i] = (m31){(uint32_t)((val0 >> (i * 9)) & 0x1FF)};
    }
    // Cross boundary val0/val1 at bit 63
    uint64_t cross01 = (val0 >> 63) | (val1 << 1);
    limbs[7] = (m31){(uint32_t)(cross01 & 0x1FF)};

    for (int i = 0; i < 6; i++) {
        limbs[8 + i] = (m31){(uint32_t)((val1 >> (8 + i * 9)) & 0x1FF)};
    }
    // Cross boundary val1/val2 at bit 62
    uint64_t cross12 = (val1 >> 62) | (val2 << 2);
    limbs[14] = (m31){(uint32_t)(cross12 & 0x1FF)};

    for (int i = 0; i < 6; i++) {
        limbs[15 + i] = (m31){(uint32_t)((val2 >> (7 + i * 9)) & 0x1FF)};
    }
    // Cross boundary val2/val3 at bit 61
    uint64_t cross23 = (val2 >> 61) | (val3 << 3);
    limbs[21] = (m31){(uint32_t)(cross23 & 0x1FF)};

    for (int i = 0; i < 6; i++) {
        limbs[22 + i] = (m31){(uint32_t)((val3 >> (6 + i * 9)) & 0x1FF)};
    }
}

// Unpack Width27 to 28 x 9-bit limbs
// Each 27-bit limb becomes 3 x 9-bit limbs, except last limb is just 9 bits
// Note: The last input limb (input27[9]) should already be 9-bit from the constraint system
// We don't mask it here to match SIMD behavior exactly
__device__ void width27_to_limbs28(const m31* input27, m31* output28) {
    for (int i = 0; i < 9; i++) {
        uint32_t val = input27[i];
        output28[i * 3] = (m31){val & 0x1FF};
        output28[i * 3 + 1] = (m31){(val >> 9) & 0x1FF};
        output28[i * 3 + 2] = (m31){(val >> 18) & 0x1FF};
    }
    // Last limb is just 9 bits (from constraint system)
    output28[27] = input27[9];
}

// ============================================================================
// Double Karatsuba multiplication for 252-bit numbers
// Computes 55-limb product from two 28-limb inputs
// ============================================================================

// Single Karatsuba N=7: computes 13-limb output from two 7-limb inputs
__device__ void single_karatsuba_n7(const m31* a, const m31* b, int64_t* output) {
    // z0 = a_low * b_low (7x7 -> 13 limbs)
    // z2 = a_high * b_high (would be for higher parts)
    // For cube_252, we need the full convolution

    // Direct convolution for 7x7 -> 13 limbs
    for (int k = 0; k < 13; k++) {
        int64_t sum = 0;
        for (int i = 0; i <= k && i < 7; i++) {
            int j = k - i;
            if (j >= 0 && j < 7) {
                sum += (int64_t)a[i] * (int64_t)b[j];
            }
        }
        output[k] = sum;
    }
}

// Compute 55-limb product from two 28-limb inputs using double Karatsuba
__device__ void double_karatsuba_n7_product(const m31* a, const m31* b, int64_t* product) {
    // Split each 28-limb input into 4 x 7-limb parts
    // a = a0 + a1*B^7 + a2*B^14 + a3*B^21
    // b = b0 + b1*B^7 + b2*B^14 + b3*B^21
    // where B = 2^9

    // Initialize product to zero
    for (int i = 0; i < 55; i++) {
        product[i] = 0;
    }

    // Compute all 16 partial products and accumulate
    for (int ai = 0; ai < 4; ai++) {
        for (int bi = 0; bi < 4; bi++) {
            int64_t partial[13];
            single_karatsuba_n7(&a[ai * 7], &b[bi * 7], partial);

            int offset = (ai + bi) * 7;
            for (int k = 0; k < 13; k++) {
                if (offset + k < 55) {
                    product[offset + k] += partial[k];
                }
            }
        }
    }
}

// ============================================================================
// Convolution modular reduction for Felt252 - M31 arithmetic version
// p = 2^252 + 17*2^192 + 1
//
// This computes conv_mod using M31 field arithmetic to match the SIMD version.
// Input: conv array of M31 values
// Output: conv_mod array of M31 values
// ============================================================================

// M31 constants for conv_mod computation
#define M31_2 ((m31)2u)
#define M31_4 ((m31)4u)
#define M31_8 ((m31)8u)
#define M31_32 ((m31)32u)
#define M31_64 ((m31)64u)
#define M31_512 ((m31)512u)         // For packing limbs in cube_252_0 lookup
#define M31_524288 ((m31)524288u)   // For carry and k values in rc_19/rc_20 lookups (2^19)
#define M31_262144 ((m31)262144u)   // For limb packing: 27-bit width = 9+9+9, shift = 2^18

// Correct M31 multiplication using proper Mersenne prime reduction
// The standard mul() function in fields.cu has a bug for large products
// This uses iterative reduction for products up to 62 bits
__device__ __forceinline__ m31 mul_correct(m31 a, m31 b) {
    uint64_t v = ((uint64_t)a) * ((uint64_t)b);
    // For Mersenne prime M = 2^31 - 1:
    // v mod M = (v mod 2^31) + (v / 2^31), then reduce if >= M
    // We may need multiple iterations for large products

    // First iteration
    uint64_t lo = v & 0x7FFFFFFF;  // v mod 2^31 (lower 31 bits)
    uint64_t hi = v >> 31;          // v / 2^31 (upper bits)
    v = lo + hi;

    // Second iteration (needed when hi + lo >= 2^31)
    lo = v & 0x7FFFFFFF;
    hi = v >> 31;
    v = lo + hi;

    // Final check: if v >= P, subtract P
    if (v >= P) {
        v -= P;
    }
    return (m31)v;
}

__device__ void compute_conv_mod_m31(const m31* conv, m31* conv_mod) {
    // Modular reduction coefficients derived from p = 2^252 + 17*2^192 + 1
    // These match the Rust implementation exactly, using M31 arithmetic

    // limb 0: (32 * conv[0]) - (4 * conv[21]) + (8 * conv[49])
    conv_mod[0] = add(sub(mul(M31_32, conv[0]), mul(M31_4, conv[21])), mul(M31_8, conv[49]));

    // limbs 1-5: conv[i-1] + 32*conv[i] - 4*conv[i+21] + 8*conv[i+49]
    for (int i = 1; i <= 5; i++) {
        conv_mod[i] = add(add(sub(add(conv[i-1], mul(M31_32, conv[i])), mul(M31_4, conv[i+21])), mul(M31_8, conv[i+49])), (m31)0);
    }

    // limb 6: conv[5] + 32*conv[6] - 4*conv[27]
    conv_mod[6] = sub(add(conv[5], mul(M31_32, conv[6])), mul(M31_4, conv[27]));

    // limbs 7-20: 2*conv[i-7] + conv[i-1] + 32*conv[i] - 4*conv[i+21]
    for (int i = 7; i <= 20; i++) {
        conv_mod[i] = sub(add(add(mul(M31_2, conv[i-7]), conv[i-1]), mul(M31_32, conv[i])), mul(M31_4, conv[i+21]));
    }

    // limb 21: 2*conv[14] + conv[20] - 4*conv[42] + 64*conv[49]
    conv_mod[21] = add(sub(add(mul(M31_2, conv[14]), conv[20]), mul(M31_4, conv[42])), mul(M31_64, conv[49]));

    // limbs 22-26: 2*conv[i-7] - 4*conv[i+21] + 2*conv[i+27] + 64*conv[i+28]
    for (int i = 22; i <= 26; i++) {
        conv_mod[i] = add(add(sub(mul(M31_2, conv[i-7]), mul(M31_4, conv[i+21])), mul(M31_2, conv[i+27])), mul(M31_64, conv[i+28]));
    }

    // limb 27: 2*conv[20] - 4*conv[48] + 2*conv[54]
    conv_mod[27] = add(sub(mul(M31_2, conv[20]), mul(M31_4, conv[48])), mul(M31_2, conv[54]));
}

// Compute k and carry values for multiplication verification

// Helper: convert signed int64 to proper M31 field element
// For negative values, returns P + val (which is equivalent to P - |val|)
__device__ __forceinline__ m31 signed_to_m31(int64_t val) {
    // First reduce to the range (-P, P)
    int64_t reduced = val % (int64_t)P;
    if (reduced < 0) {
        reduced += P;  // Convert negative to positive M31 representation
    }
    return (m31)reduced;
}

// M31 constant for carry computation: 4194304 = 512^(-1) mod P
#define M31_4194304 ((m31)4194304u)
#define M31_136 ((m31)136u)
#define M31_65536 ((m31)65536u)
#define M31_134217728 ((m31)134217728u)  // 2^27

// M31 version of compute_k_and_carries
// Takes M31 conv_mod values directly (already in M31 field)
__device__ void compute_k_and_carries_m31(
    const m31* conv_mod,
    m31& k_out,
    m31* carry_out  // 27 carry values
) {
    // Compute k from conv_mod[0] and conv_mod[1]
    // k_mod_2_18_biased = ((conv_mod[0] + 2^27) + ((conv_mod[1] + 2^27) & 511) << 9 + 65536) & 262143
    // In SIMD: k = k_low.as_m31() + (k_high.as_m31() - M31_1) * M31_65536

    // First add the bias (2^27 = 134217728) to get into unsigned range
    // Note: In M31, adding 134217728 may wrap around, but we use the raw uint32 for bit operations
    uint32_t biased_0 = add(conv_mod[0], M31_134217728);
    uint32_t biased_1 = add(conv_mod[1], M31_134217728);

    // Compute k_biased using integer arithmetic (bit operations)
    uint32_t k_biased = (biased_0 + ((biased_1 & 511) << 9) + 65536) & 262143;

    // Extract k_low (lower 16 bits) and k_high (bits 16-17)
    uint32_t k_low = k_biased & 0xFFFF;
    uint32_t k_high = (k_biased >> 16) & 0x3;

    // k = k_low + (k_high - 1) * 65536 in M31 arithmetic
    // Note: Use mul_correct for large multiplications to avoid overflow bug
    m31 k = add((m31)k_low, mul_correct(sub((m31)k_high, (m31)1), M31_65536));
    k_out = k;

    // Compute carry chain using M31 field arithmetic
    // carry[i] = (conv_mod[i] + carry[i-1]) * 4194304 in M31
    // where 4194304 = 512^(-1) mod P
    // Use mul_correct because 4194304 is large and products can exceed 2^31

    // carry[0] = (conv_mod[0] - k) * 4194304
    m31 carry = mul_correct(sub(conv_mod[0], k), M31_4194304);
    carry_out[0] = carry;

    // carry[1..21] = (conv_mod[i] + carry[i-1]) * 4194304
    for (int i = 1; i < 21; i++) {
        carry = mul_correct(add(conv_mod[i], carry), M31_4194304);
        carry_out[i] = carry;
    }

    // carry[21] = (conv_mod[21] - 136*k + carry[20]) * 4194304
    carry = mul_correct(add(sub(conv_mod[21], mul_correct(M31_136, k)), carry), M31_4194304);
    carry_out[21] = carry;

    // carry[22..27] = (conv_mod[i] + carry[i-1]) * 4194304
    for (int i = 22; i < 27; i++) {
        carry = mul_correct(add(conv_mod[i], carry), M31_4194304);
        carry_out[i] = carry;
    }
}

// ============================================================================
// Main trace generation kernel with lookup data population
// ============================================================================

// Structure to hold all lookup data pointers for kernel
struct Cube252LookupPtrs {
    m31** cube_252_0;           // [20] elements
    // rc_9_9 variants (2 elements each) - 6 lookups each for a-f, 3 for g-h
    m31** rc_9_9_0; m31** rc_9_9_1; m31** rc_9_9_2;
    m31** rc_9_9_3; m31** rc_9_9_4; m31** rc_9_9_5;
    m31** rc_9_9_b_0; m31** rc_9_9_b_1; m31** rc_9_9_b_2;
    m31** rc_9_9_b_3; m31** rc_9_9_b_4; m31** rc_9_9_b_5;
    m31** rc_9_9_c_0; m31** rc_9_9_c_1; m31** rc_9_9_c_2;
    m31** rc_9_9_c_3; m31** rc_9_9_c_4; m31** rc_9_9_c_5;
    m31** rc_9_9_d_0; m31** rc_9_9_d_1; m31** rc_9_9_d_2;
    m31** rc_9_9_d_3; m31** rc_9_9_d_4; m31** rc_9_9_d_5;
    m31** rc_9_9_e_0; m31** rc_9_9_e_1; m31** rc_9_9_e_2;
    m31** rc_9_9_e_3; m31** rc_9_9_e_4; m31** rc_9_9_e_5;
    m31** rc_9_9_f_0; m31** rc_9_9_f_1; m31** rc_9_9_f_2;
    m31** rc_9_9_f_3; m31** rc_9_9_f_4; m31** rc_9_9_f_5;
    m31** rc_9_9_g_0; m31** rc_9_9_g_1; m31** rc_9_9_g_2;
    m31** rc_9_9_h_0; m31** rc_9_9_h_1; m31** rc_9_9_h_2;
    // rc_19 variants (1 element each)
    m31** rc_19_0; m31** rc_19_1; m31** rc_19_2; m31** rc_19_3;
    m31** rc_19_4; m31** rc_19_5; m31** rc_19_6; m31** rc_19_7;
    m31** rc_19_b_0; m31** rc_19_b_1; m31** rc_19_b_2; m31** rc_19_b_3;
    m31** rc_19_b_4; m31** rc_19_b_5; m31** rc_19_b_6; m31** rc_19_b_7;
    m31** rc_19_c_0; m31** rc_19_c_1; m31** rc_19_c_2; m31** rc_19_c_3;
    m31** rc_19_c_4; m31** rc_19_c_5; m31** rc_19_c_6; m31** rc_19_c_7;
    m31** rc_19_d_0; m31** rc_19_d_1; m31** rc_19_d_2;
    m31** rc_19_d_3; m31** rc_19_d_4; m31** rc_19_d_5;
    m31** rc_19_e_0; m31** rc_19_e_1; m31** rc_19_e_2;
    m31** rc_19_e_3; m31** rc_19_e_4; m31** rc_19_e_5;
    m31** rc_19_f_0; m31** rc_19_f_1; m31** rc_19_f_2;
    m31** rc_19_f_3; m31** rc_19_f_4; m31** rc_19_f_5;
    m31** rc_19_g_0; m31** rc_19_g_1; m31** rc_19_g_2;
    m31** rc_19_g_3; m31** rc_19_g_4; m31** rc_19_g_5;
    m31** rc_19_h_0; m31** rc_19_h_1; m31** rc_19_h_2; m31** rc_19_h_3;
    m31** rc_19_h_4; m31** rc_19_h_5; m31** rc_19_h_6; m31** rc_19_h_7;
};

// Structure to hold sub_component_inputs pointers
// Each array is flattened: [lookup_0_elem_0, lookup_0_elem_1, lookup_1_elem_0, ...]
struct Cube252SubComponentInputs {
    // rc_9_9 variants: 6 lookups × 2 elements = 12 pointers each (except g,h = 3×2=6)
    m31** rc_9_9;       // 12 pointers
    m31** rc_9_9_b;     // 12 pointers
    m31** rc_9_9_c;     // 12 pointers
    m31** rc_9_9_d;     // 12 pointers
    m31** rc_9_9_e;     // 12 pointers
    m31** rc_9_9_f;     // 12 pointers
    m31** rc_9_9_g;     // 6 pointers
    m31** rc_9_9_h;     // 6 pointers
    // rc_19 variants: N lookups × 1 element = N pointers
    m31** rc_19;        // 8 pointers
    m31** rc_19_b;      // 8 pointers
    m31** rc_19_c;      // 8 pointers
    m31** rc_19_d;      // 6 pointers
    m31** rc_19_e;      // 6 pointers
    m31** rc_19_f;      // 6 pointers
    m31** rc_19_g;      // 6 pointers
    m31** rc_19_h;      // 8 pointers
};

__global__ void cube_252_trace_kernel(
    m31** inputs,           // 10 input columns (Width27 format)
    unsigned int n_rows,
    m31** trace_columns,    // 141 output trace columns
    Cube252LookupPtrs lookup, // Lookup data pointers
    Cube252SubComponentInputs sub_inputs // Sub-component inputs pointers
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Read input (10 x 27-bit limbs)
    m31 input27[10];
    for (int i = 0; i < 10; i++) {
        input27[i] = inputs[i][row];
    }

    // Write input to trace columns 0-9
    for (int i = 0; i < 10; i++) {
        trace_columns[i][row] = input27[i];
    }

    // Unpack Width27 to 28 x 9-bit limbs
    m31 x[28];
    width27_to_limbs28(input27, x);

    // Write unpacked limbs to trace columns 10-27
    // Positions: 0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25
    trace_columns[10][row] = x[0];   // limb 0
    trace_columns[11][row] = x[1];   // limb 1
    trace_columns[12][row] = x[3];   // limb 3
    trace_columns[13][row] = x[4];   // limb 4
    trace_columns[14][row] = x[6];   // limb 6
    trace_columns[15][row] = x[7];   // limb 7
    trace_columns[16][row] = x[9];   // limb 9
    trace_columns[17][row] = x[10];  // limb 10
    trace_columns[18][row] = x[12];  // limb 12
    trace_columns[19][row] = x[13];  // limb 13
    trace_columns[20][row] = x[15];  // limb 15
    trace_columns[21][row] = x[16];  // limb 16
    trace_columns[22][row] = x[18];  // limb 18
    trace_columns[23][row] = x[19];  // limb 19
    trace_columns[24][row] = x[21];  // limb 21
    trace_columns[25][row] = x[22];  // limb 22
    trace_columns[26][row] = x[24];  // limb 24
    trace_columns[27][row] = x[25];  // limb 25

    // ========================================================================
    // Populate rc_9_9 lookups from unpacked limbs (following SIMD cube_252.rs)
    // ========================================================================
    // rc_9_9_0: [x[0], x[1]]
    lookup.rc_9_9_0[0][row] = x[0];
    lookup.rc_9_9_0[1][row] = x[1];
    // rc_9_9_b_0: [x[2], x[3]]
    lookup.rc_9_9_b_0[0][row] = x[2];
    lookup.rc_9_9_b_0[1][row] = x[3];
    // rc_9_9_c_0: [x[4], x[5]]
    lookup.rc_9_9_c_0[0][row] = x[4];
    lookup.rc_9_9_c_0[1][row] = x[5];
    // rc_9_9_d_0: [x[6], x[7]]
    lookup.rc_9_9_d_0[0][row] = x[6];
    lookup.rc_9_9_d_0[1][row] = x[7];
    // rc_9_9_e_0: [x[8], x[9]]
    lookup.rc_9_9_e_0[0][row] = x[8];
    lookup.rc_9_9_e_0[1][row] = x[9];
    // rc_9_9_f_0: [x[10], x[11]]
    lookup.rc_9_9_f_0[0][row] = x[10];
    lookup.rc_9_9_f_0[1][row] = x[11];
    // rc_9_9_g_0: [x[12], x[13]]
    lookup.rc_9_9_g_0[0][row] = x[12];
    lookup.rc_9_9_g_0[1][row] = x[13];
    // rc_9_9_h_0: [x[14], x[15]]
    lookup.rc_9_9_h_0[0][row] = x[14];
    lookup.rc_9_9_h_0[1][row] = x[15];
    // rc_9_9_1: [x[16], x[17]]
    lookup.rc_9_9_1[0][row] = x[16];
    lookup.rc_9_9_1[1][row] = x[17];
    // rc_9_9_b_1: [x[18], x[19]]
    lookup.rc_9_9_b_1[0][row] = x[18];
    lookup.rc_9_9_b_1[1][row] = x[19];
    // rc_9_9_c_1: [x[20], x[21]]
    lookup.rc_9_9_c_1[0][row] = x[20];
    lookup.rc_9_9_c_1[1][row] = x[21];
    // rc_9_9_d_1: [x[22], x[23]]
    lookup.rc_9_9_d_1[0][row] = x[22];
    lookup.rc_9_9_d_1[1][row] = x[23];
    // rc_9_9_e_1: [x[24], x[25]]
    lookup.rc_9_9_e_1[0][row] = x[24];
    lookup.rc_9_9_e_1[1][row] = x[25];
    // rc_9_9_f_1: [x[26], input_limb_9] (input_limb_9 is the raw 9-bit value)
    lookup.rc_9_9_f_1[0][row] = x[26];
    lookup.rc_9_9_f_1[1][row] = input27[9];

    // Convert to Felt252Field for multiplication
    // IMPORTANT: SIMD creates Felt252 from 28x9-bit unpacked limbs via from_limbs()
    // We must do the same to match the representation

    // First, construct Felt252Field from the 28x9-bit limbs (like SIMD does)
    // The x[] array already has the unpacked 9-bit limbs
    uint64_t val0 = 0, val1 = 0, val2 = 0, val3 = 0;
    for (int i = 0; i < 7; i++) {
        val0 |= ((uint64_t)(uint32_t)x[i]) << (9 * i);
    }
    // Bit position 63 is in limb 7 (9*7 = 63)
    val0 |= ((uint64_t)(uint32_t)x[7] & 0x1) << 63;
    val1 = ((uint32_t)x[7]) >> 1;
    for (int i = 8; i < 14; i++) {
        val1 |= ((uint64_t)(uint32_t)x[i]) << (8 + 9 * (i - 8));
    }
    // Cross into val2 at position 62+8=70, limb 14
    val1 |= ((uint64_t)(uint32_t)x[14] & 0x3) << 62;
    val2 = ((uint32_t)x[14]) >> 2;
    for (int i = 15; i < 21; i++) {
        val2 |= ((uint64_t)(uint32_t)x[i]) << (7 + 9 * (i - 15));
    }
    // Cross into val3 at position 61+7=68, limb 21
    val2 |= ((uint64_t)(uint32_t)x[21] & 0x7) << 61;
    val3 = ((uint32_t)x[21]) >> 3;
    for (int i = 22; i < 28; i++) {
        val3 |= ((uint64_t)(uint32_t)x[i]) << (6 + 9 * (i - 22));
    }

    Felt252Field x_felt;
    x_felt.limbs[0] = (uint32_t)(val0 & 0xFFFFFFFF);
    x_felt.limbs[1] = (uint32_t)((val0 >> 32) & 0xFFFFFFFF);
    x_felt.limbs[2] = (uint32_t)(val1 & 0xFFFFFFFF);
    x_felt.limbs[3] = (uint32_t)((val1 >> 32) & 0xFFFFFFFF);
    x_felt.limbs[4] = (uint32_t)(val2 & 0xFFFFFFFF);
    x_felt.limbs[5] = (uint32_t)((val2 >> 32) & 0xFFFFFFFF);
    x_felt.limbs[6] = (uint32_t)(val3 & 0xFFFFFFFF);
    x_felt.limbs[7] = (uint32_t)((val3 >> 32) & 0xFFFFFFFF);

    // Compute x² = x * x
    // Apply Montgomery multiplication factor to get correct limb representation
    Felt252Field x2_felt = cube252_felt_mul(x_felt, x_felt);
    x2_felt = cube252_felt_mul(x2_felt, CUBE252_MONT_MUL_FACTOR);

    // Compute x³ = x² * x
    // Apply Montgomery multiplication factor for this multiplication too
    Felt252Field x3_felt = cube252_felt_mul(x2_felt, x_felt);
    x3_felt = cube252_felt_mul(x3_felt, CUBE252_MONT_MUL_FACTOR);

    // Convert x² to 28 x 9-bit limbs (first mul result)
    m31 x2[28];
    felt252_to_limbs28(x2_felt, x2);

    // Write x² to trace columns 28-55
    for (int i = 0; i < 28; i++) {
        trace_columns[28 + i][row] = x2[i];
    }

    // ========================================================================
    // Populate rc_9_9 lookups from first mul result (x²) - columns 28-55
    // ========================================================================
    // rc_9_9_2: [x2[0], x2[1]]
    lookup.rc_9_9_2[0][row] = x2[0];
    lookup.rc_9_9_2[1][row] = x2[1];
    // rc_9_9_b_2: [x2[2], x2[3]]
    lookup.rc_9_9_b_2[0][row] = x2[2];
    lookup.rc_9_9_b_2[1][row] = x2[3];
    // rc_9_9_c_2: [x2[4], x2[5]]
    lookup.rc_9_9_c_2[0][row] = x2[4];
    lookup.rc_9_9_c_2[1][row] = x2[5];
    // rc_9_9_d_2: [x2[6], x2[7]]
    lookup.rc_9_9_d_2[0][row] = x2[6];
    lookup.rc_9_9_d_2[1][row] = x2[7];
    // rc_9_9_e_2: [x2[8], x2[9]]
    lookup.rc_9_9_e_2[0][row] = x2[8];
    lookup.rc_9_9_e_2[1][row] = x2[9];
    // rc_9_9_f_2: [x2[10], x2[11]]
    lookup.rc_9_9_f_2[0][row] = x2[10];
    lookup.rc_9_9_f_2[1][row] = x2[11];
    // rc_9_9_g_1: [x2[12], x2[13]]
    lookup.rc_9_9_g_1[0][row] = x2[12];
    lookup.rc_9_9_g_1[1][row] = x2[13];
    // rc_9_9_h_1: [x2[14], x2[15]]
    lookup.rc_9_9_h_1[0][row] = x2[14];
    lookup.rc_9_9_h_1[1][row] = x2[15];
    // rc_9_9_3: [x2[16], x2[17]]
    lookup.rc_9_9_3[0][row] = x2[16];
    lookup.rc_9_9_3[1][row] = x2[17];
    // rc_9_9_b_3: [x2[18], x2[19]]
    lookup.rc_9_9_b_3[0][row] = x2[18];
    lookup.rc_9_9_b_3[1][row] = x2[19];
    // rc_9_9_c_3: [x2[20], x2[21]]
    lookup.rc_9_9_c_3[0][row] = x2[20];
    lookup.rc_9_9_c_3[1][row] = x2[21];
    // rc_9_9_d_3: [x2[22], x2[23]]
    lookup.rc_9_9_d_3[0][row] = x2[22];
    lookup.rc_9_9_d_3[1][row] = x2[23];
    // rc_9_9_e_3: [x2[24], x2[25]]
    lookup.rc_9_9_e_3[0][row] = x2[24];
    lookup.rc_9_9_e_3[1][row] = x2[25];
    // rc_9_9_f_3: [x2[26], x2[27]]
    lookup.rc_9_9_f_3[0][row] = x2[26];
    lookup.rc_9_9_f_3[1][row] = x2[27];

    // Compute first multiplication verification (x * x = x²)
    // Use M31 arithmetic throughout to match SIMD version
    int64_t product1[55];
    double_karatsuba_n7_product(x, x, product1);

    // Convert product to M31 and compute convolution: product - result
    // In M31 arithmetic (subtraction wraps around P for negative results)
    m31 conv1[55];
    for (int i = 0; i < 28; i++) {
        // product1[i] is always < P (bounded by convolution sum)
        // x2[i] is 9-bit (< 512)
        // Use M31 subtraction
        conv1[i] = sub((m31)(product1[i] % P), x2[i]);
    }
    for (int i = 28; i < 55; i++) {
        conv1[i] = (m31)(product1[i] % P);
    }

    // Compute conv_mod for first multiplication using M31 arithmetic
    m31 conv_mod1[28];
    compute_conv_mod_m31(conv1, conv_mod1);

    // Compute k1 and carry1 using M31 arithmetic
    m31 k1;
    m31 carry1[27];
    compute_k_and_carries_m31(conv_mod1, k1, carry1);

    // Write k1 to column 56
    trace_columns[56][row] = k1;

    // Write carry1 to columns 57-83
    for (int i = 0; i < 27; i++) {
        trace_columns[57 + i][row] = carry1[i];
    }

    // ========================================================================
    // Populate rc_19 lookups from first mul (k1 and carry1)
    // Following SIMD pattern from cube_252.rs lines 1605-1740
    // ========================================================================
    // rc_19_h_0: k1 + 524288
    lookup.rc_19_h_0[0][row] = add(k1, M31_524288);
    // rc_19_0: carry1[0] + 131072
    lookup.rc_19_0[0][row] = add(carry1[0], M31_524288);
    // rc_19_b_0: carry1[1] + 131072
    lookup.rc_19_b_0[0][row] = add(carry1[1], M31_524288);
    // rc_19_c_0: carry1[2] + 131072
    lookup.rc_19_c_0[0][row] = add(carry1[2], M31_524288);
    // rc_19_d_0: carry1[3] + 131072
    lookup.rc_19_d_0[0][row] = add(carry1[3], M31_524288);
    // rc_19_e_0: carry1[4] + 131072
    lookup.rc_19_e_0[0][row] = add(carry1[4], M31_524288);
    // rc_19_f_0: carry1[5] + 131072
    lookup.rc_19_f_0[0][row] = add(carry1[5], M31_524288);
    // rc_19_g_0: carry1[6] + 131072
    lookup.rc_19_g_0[0][row] = add(carry1[6], M31_524288);
    // rc_19_h_1: carry1[7] + 131072
    lookup.rc_19_h_1[0][row] = add(carry1[7], M31_524288);
    // rc_19_1: carry1[8] + 131072
    lookup.rc_19_1[0][row] = add(carry1[8], M31_524288);
    // rc_19_b_1: carry1[9] + 131072
    lookup.rc_19_b_1[0][row] = add(carry1[9], M31_524288);
    // rc_19_c_1: carry1[10] + 131072
    lookup.rc_19_c_1[0][row] = add(carry1[10], M31_524288);
    // rc_19_d_1: carry1[11] + 131072
    lookup.rc_19_d_1[0][row] = add(carry1[11], M31_524288);
    // rc_19_e_1: carry1[12] + 131072
    lookup.rc_19_e_1[0][row] = add(carry1[12], M31_524288);
    // rc_19_f_1: carry1[13] + 131072
    lookup.rc_19_f_1[0][row] = add(carry1[13], M31_524288);
    // rc_19_g_1: carry1[14] + 131072
    lookup.rc_19_g_1[0][row] = add(carry1[14], M31_524288);
    // rc_19_h_2: carry1[15] + 131072
    lookup.rc_19_h_2[0][row] = add(carry1[15], M31_524288);
    // rc_19_2: carry1[16] + 131072
    lookup.rc_19_2[0][row] = add(carry1[16], M31_524288);
    // rc_19_b_2: carry1[17] + 131072
    lookup.rc_19_b_2[0][row] = add(carry1[17], M31_524288);
    // rc_19_c_2: carry1[18] + 131072
    lookup.rc_19_c_2[0][row] = add(carry1[18], M31_524288);
    // rc_19_d_2: carry1[19] + 131072
    lookup.rc_19_d_2[0][row] = add(carry1[19], M31_524288);
    // rc_19_e_2: carry1[20] + 131072
    lookup.rc_19_e_2[0][row] = add(carry1[20], M31_524288);
    // rc_19_f_2: carry1[21] + 131072
    lookup.rc_19_f_2[0][row] = add(carry1[21], M31_524288);
    // rc_19_g_2: carry1[22] + 131072
    lookup.rc_19_g_2[0][row] = add(carry1[22], M31_524288);
    // rc_19_h_3: carry1[23] + 131072
    lookup.rc_19_h_3[0][row] = add(carry1[23], M31_524288);
    // rc_19_3: carry1[24] + 131072
    lookup.rc_19_3[0][row] = add(carry1[24], M31_524288);
    // rc_19_b_3: carry1[25] + 131072
    lookup.rc_19_b_3[0][row] = add(carry1[25], M31_524288);
    // rc_19_c_3: carry1[26] + 131072
    lookup.rc_19_c_3[0][row] = add(carry1[26], M31_524288);

    // Convert x³ to 28 x 9-bit limbs (second mul result)
    m31 x3[28];
    felt252_to_limbs28(x3_felt, x3);

    // Write x³ to trace columns 84-111
    for (int i = 0; i < 28; i++) {
        trace_columns[84 + i][row] = x3[i];
    }

    // ========================================================================
    // Populate rc_9_9 lookups from second mul result (x³) - columns 84-111
    // ========================================================================
    // rc_9_9_4: [x3[0], x3[1]]
    lookup.rc_9_9_4[0][row] = x3[0];
    lookup.rc_9_9_4[1][row] = x3[1];
    // rc_9_9_b_4: [x3[2], x3[3]]
    lookup.rc_9_9_b_4[0][row] = x3[2];
    lookup.rc_9_9_b_4[1][row] = x3[3];
    // rc_9_9_c_4: [x3[4], x3[5]]
    lookup.rc_9_9_c_4[0][row] = x3[4];
    lookup.rc_9_9_c_4[1][row] = x3[5];
    // rc_9_9_d_4: [x3[6], x3[7]]
    lookup.rc_9_9_d_4[0][row] = x3[6];
    lookup.rc_9_9_d_4[1][row] = x3[7];
    // rc_9_9_e_4: [x3[8], x3[9]]
    lookup.rc_9_9_e_4[0][row] = x3[8];
    lookup.rc_9_9_e_4[1][row] = x3[9];
    // rc_9_9_f_4: [x3[10], x3[11]]
    lookup.rc_9_9_f_4[0][row] = x3[10];
    lookup.rc_9_9_f_4[1][row] = x3[11];
    // rc_9_9_g_2: [x3[12], x3[13]]
    lookup.rc_9_9_g_2[0][row] = x3[12];
    lookup.rc_9_9_g_2[1][row] = x3[13];
    // rc_9_9_h_2: [x3[14], x3[15]]
    lookup.rc_9_9_h_2[0][row] = x3[14];
    lookup.rc_9_9_h_2[1][row] = x3[15];
    // rc_9_9_5: [x3[16], x3[17]]
    lookup.rc_9_9_5[0][row] = x3[16];
    lookup.rc_9_9_5[1][row] = x3[17];
    // rc_9_9_b_5: [x3[18], x3[19]]
    lookup.rc_9_9_b_5[0][row] = x3[18];
    lookup.rc_9_9_b_5[1][row] = x3[19];
    // rc_9_9_c_5: [x3[20], x3[21]]
    lookup.rc_9_9_c_5[0][row] = x3[20];
    lookup.rc_9_9_c_5[1][row] = x3[21];
    // rc_9_9_d_5: [x3[22], x3[23]]
    lookup.rc_9_9_d_5[0][row] = x3[22];
    lookup.rc_9_9_d_5[1][row] = x3[23];
    // rc_9_9_e_5: [x3[24], x3[25]]
    lookup.rc_9_9_e_5[0][row] = x3[24];
    lookup.rc_9_9_e_5[1][row] = x3[25];
    // rc_9_9_f_5: [x3[26], x3[27]]
    lookup.rc_9_9_f_5[0][row] = x3[26];
    lookup.rc_9_9_f_5[1][row] = x3[27];

    // Compute second multiplication verification (x * x² = x³)
    // Use M31 arithmetic throughout to match SIMD version
    int64_t product2[55];
    double_karatsuba_n7_product(x, x2, product2);

    // Convert product to M31 and compute convolution: product - result
    m31 conv2[55];
    for (int i = 0; i < 28; i++) {
        conv2[i] = sub((m31)(product2[i] % P), x3[i]);
    }
    for (int i = 28; i < 55; i++) {
        conv2[i] = (m31)(product2[i] % P);
    }

    // Compute conv_mod for second multiplication using M31 arithmetic
    m31 conv_mod2[28];
    compute_conv_mod_m31(conv2, conv_mod2);

    // Compute k2 and carry2 using M31 arithmetic
    m31 k2;
    m31 carry2[27];
    compute_k_and_carries_m31(conv_mod2, k2, carry2);

    // Write k2 to column 112
    trace_columns[112][row] = k2;

    // Write carry2 to columns 113-139
    for (int i = 0; i < 27; i++) {
        trace_columns[113 + i][row] = carry2[i];
    }

    // ========================================================================
    // Populate rc_19 lookups from second mul (k2 and carry2)
    // Following SIMD pattern from cube_252.rs lines 2884-3019
    // ========================================================================
    // rc_19_h_4: k2 + 524288
    lookup.rc_19_h_4[0][row] = add(k2, M31_524288);
    // rc_19_4: carry2[0] + 131072
    lookup.rc_19_4[0][row] = add(carry2[0], M31_524288);
    // rc_19_b_4: carry2[1] + 131072
    lookup.rc_19_b_4[0][row] = add(carry2[1], M31_524288);
    // rc_19_c_4: carry2[2] + 131072
    lookup.rc_19_c_4[0][row] = add(carry2[2], M31_524288);
    // rc_19_d_3: carry2[3] + 131072
    lookup.rc_19_d_3[0][row] = add(carry2[3], M31_524288);
    // rc_19_e_3: carry2[4] + 131072
    lookup.rc_19_e_3[0][row] = add(carry2[4], M31_524288);
    // rc_19_f_3: carry2[5] + 131072
    lookup.rc_19_f_3[0][row] = add(carry2[5], M31_524288);
    // rc_19_g_3: carry2[6] + 131072
    lookup.rc_19_g_3[0][row] = add(carry2[6], M31_524288);
    // rc_19_h_5: carry2[7] + 131072
    lookup.rc_19_h_5[0][row] = add(carry2[7], M31_524288);
    // rc_19_5: carry2[8] + 131072
    lookup.rc_19_5[0][row] = add(carry2[8], M31_524288);
    // rc_19_b_5: carry2[9] + 131072
    lookup.rc_19_b_5[0][row] = add(carry2[9], M31_524288);
    // rc_19_c_5: carry2[10] + 131072
    lookup.rc_19_c_5[0][row] = add(carry2[10], M31_524288);
    // rc_19_d_4: carry2[11] + 131072
    lookup.rc_19_d_4[0][row] = add(carry2[11], M31_524288);
    // rc_19_e_4: carry2[12] + 131072
    lookup.rc_19_e_4[0][row] = add(carry2[12], M31_524288);
    // rc_19_f_4: carry2[13] + 131072
    lookup.rc_19_f_4[0][row] = add(carry2[13], M31_524288);
    // rc_19_g_4: carry2[14] + 131072
    lookup.rc_19_g_4[0][row] = add(carry2[14], M31_524288);
    // rc_19_h_6: carry2[15] + 131072
    lookup.rc_19_h_6[0][row] = add(carry2[15], M31_524288);
    // rc_19_6: carry2[16] + 131072
    lookup.rc_19_6[0][row] = add(carry2[16], M31_524288);
    // rc_19_b_6: carry2[17] + 131072
    lookup.rc_19_b_6[0][row] = add(carry2[17], M31_524288);
    // rc_19_c_6: carry2[18] + 131072
    lookup.rc_19_c_6[0][row] = add(carry2[18], M31_524288);
    // rc_19_d_5: carry2[19] + 131072
    lookup.rc_19_d_5[0][row] = add(carry2[19], M31_524288);
    // rc_19_e_5: carry2[20] + 131072
    lookup.rc_19_e_5[0][row] = add(carry2[20], M31_524288);
    // rc_19_f_5: carry2[21] + 131072
    lookup.rc_19_f_5[0][row] = add(carry2[21], M31_524288);
    // rc_19_g_5: carry2[22] + 131072
    lookup.rc_19_g_5[0][row] = add(carry2[22], M31_524288);
    // rc_19_h_7: carry2[23] + 131072
    lookup.rc_19_h_7[0][row] = add(carry2[23], M31_524288);
    // rc_19_7: carry2[24] + 131072
    lookup.rc_19_7[0][row] = add(carry2[24], M31_524288);
    // rc_19_b_7: carry2[25] + 131072
    lookup.rc_19_b_7[0][row] = add(carry2[25], M31_524288);
    // rc_19_c_7: carry2[26] + 131072
    lookup.rc_19_c_7[0][row] = add(carry2[26], M31_524288);

    // Write enabler to column 140 (1 for valid rows)
    trace_columns[140][row] = (m31){1};

    // ========================================================================
    // Populate cube_252_0 self-lookup (20 elements)
    // Input limbs (0-9) + packed output limbs (10-19)
    // Following SIMD pattern from cube_252.rs lines 3023-3053
    // ========================================================================
    // Input limbs 0-9
    for (int i = 0; i < 10; i++) {
        lookup.cube_252_0[i][row] = input27[i];
    }
    // Output limbs: pack x³ 9-bit limbs into 27-bit
    // out[i] = x3[i*3] + x3[i*3+1]*512 + x3[i*3+2]*262144
    for (int i = 0; i < 9; i++) {
        lookup.cube_252_0[10 + i][row] = add(add(x3[i*3], mul(x3[i*3+1], M31_512)), mul(x3[i*3+2], M31_262144));
    }
    // Last element is just x3[27] (the last 9-bit limb)
    lookup.cube_252_0[19][row] = x3[27];

    // ========================================================================
    // Populate sub_component_inputs (same values as lookup, flattened arrays)
    // sub_inputs.rc_9_9 layout: [rc_9_9_0[0], rc_9_9_0[1], rc_9_9_1[0], rc_9_9_1[1], ...]
    // ========================================================================

    // rc_9_9 (6 lookups × 2 elements = 12 pointers)
    sub_inputs.rc_9_9[0][row] = x[0];      // rc_9_9_0[0]
    sub_inputs.rc_9_9[1][row] = x[1];      // rc_9_9_0[1]
    sub_inputs.rc_9_9[2][row] = x[16];     // rc_9_9_1[0]
    sub_inputs.rc_9_9[3][row] = x[17];     // rc_9_9_1[1]
    sub_inputs.rc_9_9[4][row] = x2[0];     // rc_9_9_2[0]
    sub_inputs.rc_9_9[5][row] = x2[1];     // rc_9_9_2[1]
    sub_inputs.rc_9_9[6][row] = x2[16];    // rc_9_9_3[0]
    sub_inputs.rc_9_9[7][row] = x2[17];    // rc_9_9_3[1]
    sub_inputs.rc_9_9[8][row] = x3[0];     // rc_9_9_4[0]
    sub_inputs.rc_9_9[9][row] = x3[1];     // rc_9_9_4[1]
    sub_inputs.rc_9_9[10][row] = x3[16];   // rc_9_9_5[0]
    sub_inputs.rc_9_9[11][row] = x3[17];   // rc_9_9_5[1]

    // rc_9_9_b (6 lookups × 2 elements = 12 pointers)
    sub_inputs.rc_9_9_b[0][row] = x[2];    // rc_9_9_b_0[0]
    sub_inputs.rc_9_9_b[1][row] = x[3];    // rc_9_9_b_0[1]
    sub_inputs.rc_9_9_b[2][row] = x[18];   // rc_9_9_b_1[0]
    sub_inputs.rc_9_9_b[3][row] = x[19];   // rc_9_9_b_1[1]
    sub_inputs.rc_9_9_b[4][row] = x2[2];   // rc_9_9_b_2[0]
    sub_inputs.rc_9_9_b[5][row] = x2[3];   // rc_9_9_b_2[1]
    sub_inputs.rc_9_9_b[6][row] = x2[18];  // rc_9_9_b_3[0]
    sub_inputs.rc_9_9_b[7][row] = x2[19];  // rc_9_9_b_3[1]
    sub_inputs.rc_9_9_b[8][row] = x3[2];   // rc_9_9_b_4[0]
    sub_inputs.rc_9_9_b[9][row] = x3[3];   // rc_9_9_b_4[1]
    sub_inputs.rc_9_9_b[10][row] = x3[18]; // rc_9_9_b_5[0]
    sub_inputs.rc_9_9_b[11][row] = x3[19]; // rc_9_9_b_5[1]

    // rc_9_9_c (6 lookups × 2 elements = 12 pointers)
    sub_inputs.rc_9_9_c[0][row] = x[4];    // rc_9_9_c_0[0]
    sub_inputs.rc_9_9_c[1][row] = x[5];    // rc_9_9_c_0[1]
    sub_inputs.rc_9_9_c[2][row] = x[20];   // rc_9_9_c_1[0]
    sub_inputs.rc_9_9_c[3][row] = x[21];   // rc_9_9_c_1[1]
    sub_inputs.rc_9_9_c[4][row] = x2[4];   // rc_9_9_c_2[0]
    sub_inputs.rc_9_9_c[5][row] = x2[5];   // rc_9_9_c_2[1]
    sub_inputs.rc_9_9_c[6][row] = x2[20];  // rc_9_9_c_3[0]
    sub_inputs.rc_9_9_c[7][row] = x2[21];  // rc_9_9_c_3[1]
    sub_inputs.rc_9_9_c[8][row] = x3[4];   // rc_9_9_c_4[0]
    sub_inputs.rc_9_9_c[9][row] = x3[5];   // rc_9_9_c_4[1]
    sub_inputs.rc_9_9_c[10][row] = x3[20]; // rc_9_9_c_5[0]
    sub_inputs.rc_9_9_c[11][row] = x3[21]; // rc_9_9_c_5[1]

    // rc_9_9_d (6 lookups × 2 elements = 12 pointers)
    sub_inputs.rc_9_9_d[0][row] = x[6];    // rc_9_9_d_0[0]
    sub_inputs.rc_9_9_d[1][row] = x[7];    // rc_9_9_d_0[1]
    sub_inputs.rc_9_9_d[2][row] = x[22];   // rc_9_9_d_1[0]
    sub_inputs.rc_9_9_d[3][row] = x[23];   // rc_9_9_d_1[1]
    sub_inputs.rc_9_9_d[4][row] = x2[6];   // rc_9_9_d_2[0]
    sub_inputs.rc_9_9_d[5][row] = x2[7];   // rc_9_9_d_2[1]
    sub_inputs.rc_9_9_d[6][row] = x2[22];  // rc_9_9_d_3[0]
    sub_inputs.rc_9_9_d[7][row] = x2[23];  // rc_9_9_d_3[1]
    sub_inputs.rc_9_9_d[8][row] = x3[6];   // rc_9_9_d_4[0]
    sub_inputs.rc_9_9_d[9][row] = x3[7];   // rc_9_9_d_4[1]
    sub_inputs.rc_9_9_d[10][row] = x3[22]; // rc_9_9_d_5[0]
    sub_inputs.rc_9_9_d[11][row] = x3[23]; // rc_9_9_d_5[1]

    // rc_9_9_e (6 lookups × 2 elements = 12 pointers)
    sub_inputs.rc_9_9_e[0][row] = x[8];    // rc_9_9_e_0[0]
    sub_inputs.rc_9_9_e[1][row] = x[9];    // rc_9_9_e_0[1]
    sub_inputs.rc_9_9_e[2][row] = x[24];   // rc_9_9_e_1[0]
    sub_inputs.rc_9_9_e[3][row] = x[25];   // rc_9_9_e_1[1]
    sub_inputs.rc_9_9_e[4][row] = x2[8];   // rc_9_9_e_2[0]
    sub_inputs.rc_9_9_e[5][row] = x2[9];   // rc_9_9_e_2[1]
    sub_inputs.rc_9_9_e[6][row] = x2[24];  // rc_9_9_e_3[0]
    sub_inputs.rc_9_9_e[7][row] = x2[25];  // rc_9_9_e_3[1]
    sub_inputs.rc_9_9_e[8][row] = x3[8];   // rc_9_9_e_4[0]
    sub_inputs.rc_9_9_e[9][row] = x3[9];   // rc_9_9_e_4[1]
    sub_inputs.rc_9_9_e[10][row] = x3[24]; // rc_9_9_e_5[0]
    sub_inputs.rc_9_9_e[11][row] = x3[25]; // rc_9_9_e_5[1]

    // rc_9_9_f (6 lookups × 2 elements = 12 pointers)
    sub_inputs.rc_9_9_f[0][row] = x[10];   // rc_9_9_f_0[0]
    sub_inputs.rc_9_9_f[1][row] = x[11];   // rc_9_9_f_0[1]
    sub_inputs.rc_9_9_f[2][row] = x[26];   // rc_9_9_f_1[0]
    sub_inputs.rc_9_9_f[3][row] = input27[9]; // rc_9_9_f_1[1] (input_limb_9)
    sub_inputs.rc_9_9_f[4][row] = x2[10];  // rc_9_9_f_2[0]
    sub_inputs.rc_9_9_f[5][row] = x2[11];  // rc_9_9_f_2[1]
    sub_inputs.rc_9_9_f[6][row] = x2[26];  // rc_9_9_f_3[0]
    sub_inputs.rc_9_9_f[7][row] = x2[27];  // rc_9_9_f_3[1]
    sub_inputs.rc_9_9_f[8][row] = x3[10];  // rc_9_9_f_4[0]
    sub_inputs.rc_9_9_f[9][row] = x3[11];  // rc_9_9_f_4[1]
    sub_inputs.rc_9_9_f[10][row] = x3[26]; // rc_9_9_f_5[0]
    sub_inputs.rc_9_9_f[11][row] = x3[27]; // rc_9_9_f_5[1]

    // rc_9_9_g (3 lookups × 2 elements = 6 pointers)
    sub_inputs.rc_9_9_g[0][row] = x[12];   // rc_9_9_g_0[0]
    sub_inputs.rc_9_9_g[1][row] = x[13];   // rc_9_9_g_0[1]
    sub_inputs.rc_9_9_g[2][row] = x2[12];  // rc_9_9_g_1[0]
    sub_inputs.rc_9_9_g[3][row] = x2[13];  // rc_9_9_g_1[1]
    sub_inputs.rc_9_9_g[4][row] = x3[12];  // rc_9_9_g_2[0]
    sub_inputs.rc_9_9_g[5][row] = x3[13];  // rc_9_9_g_2[1]

    // rc_9_9_h (3 lookups × 2 elements = 6 pointers)
    sub_inputs.rc_9_9_h[0][row] = x[14];   // rc_9_9_h_0[0]
    sub_inputs.rc_9_9_h[1][row] = x[15];   // rc_9_9_h_0[1]
    sub_inputs.rc_9_9_h[2][row] = x2[14];  // rc_9_9_h_1[0]
    sub_inputs.rc_9_9_h[3][row] = x2[15];  // rc_9_9_h_1[1]
    sub_inputs.rc_9_9_h[4][row] = x3[14];  // rc_9_9_h_2[0]
    sub_inputs.rc_9_9_h[5][row] = x3[15];  // rc_9_9_h_2[1]

    // rc_19 (8 lookups × 1 element = 8 pointers)
    sub_inputs.rc_19[0][row] = add(carry1[0], M31_524288);  // rc_19_0
    sub_inputs.rc_19[1][row] = add(carry1[8], M31_524288);  // rc_19_1
    sub_inputs.rc_19[2][row] = add(carry1[16], M31_524288); // rc_19_2
    sub_inputs.rc_19[3][row] = add(carry1[24], M31_524288); // rc_19_3
    sub_inputs.rc_19[4][row] = add(carry2[0], M31_524288);  // rc_19_4
    sub_inputs.rc_19[5][row] = add(carry2[8], M31_524288);  // rc_19_5
    sub_inputs.rc_19[6][row] = add(carry2[16], M31_524288); // rc_19_6
    sub_inputs.rc_19[7][row] = add(carry2[24], M31_524288); // rc_19_7

    // rc_19_b (8 lookups × 1 element = 8 pointers)
    sub_inputs.rc_19_b[0][row] = add(carry1[1], M31_524288);  // rc_19_b_0
    sub_inputs.rc_19_b[1][row] = add(carry1[9], M31_524288);  // rc_19_b_1
    sub_inputs.rc_19_b[2][row] = add(carry1[17], M31_524288); // rc_19_b_2
    sub_inputs.rc_19_b[3][row] = add(carry1[25], M31_524288); // rc_19_b_3
    sub_inputs.rc_19_b[4][row] = add(carry2[1], M31_524288);  // rc_19_b_4
    sub_inputs.rc_19_b[5][row] = add(carry2[9], M31_524288);  // rc_19_b_5
    sub_inputs.rc_19_b[6][row] = add(carry2[17], M31_524288); // rc_19_b_6
    sub_inputs.rc_19_b[7][row] = add(carry2[25], M31_524288); // rc_19_b_7

    // rc_19_c (8 lookups × 1 element = 8 pointers)
    sub_inputs.rc_19_c[0][row] = add(carry1[2], M31_524288);  // rc_19_c_0
    sub_inputs.rc_19_c[1][row] = add(carry1[10], M31_524288); // rc_19_c_1
    sub_inputs.rc_19_c[2][row] = add(carry1[18], M31_524288); // rc_19_c_2
    sub_inputs.rc_19_c[3][row] = add(carry1[26], M31_524288); // rc_19_c_3
    sub_inputs.rc_19_c[4][row] = add(carry2[2], M31_524288);  // rc_19_c_4
    sub_inputs.rc_19_c[5][row] = add(carry2[10], M31_524288); // rc_19_c_5
    sub_inputs.rc_19_c[6][row] = add(carry2[18], M31_524288); // rc_19_c_6
    sub_inputs.rc_19_c[7][row] = add(carry2[26], M31_524288); // rc_19_c_7

    // rc_19_d (6 lookups × 1 element = 6 pointers)
    sub_inputs.rc_19_d[0][row] = add(carry1[3], M31_524288);  // rc_19_d_0
    sub_inputs.rc_19_d[1][row] = add(carry1[11], M31_524288); // rc_19_d_1
    sub_inputs.rc_19_d[2][row] = add(carry1[19], M31_524288); // rc_19_d_2
    sub_inputs.rc_19_d[3][row] = add(carry2[3], M31_524288);  // rc_19_d_3
    sub_inputs.rc_19_d[4][row] = add(carry2[11], M31_524288); // rc_19_d_4
    sub_inputs.rc_19_d[5][row] = add(carry2[19], M31_524288); // rc_19_d_5

    // rc_19_e (6 lookups × 1 element = 6 pointers)
    sub_inputs.rc_19_e[0][row] = add(carry1[4], M31_524288);  // rc_19_e_0
    sub_inputs.rc_19_e[1][row] = add(carry1[12], M31_524288); // rc_19_e_1
    sub_inputs.rc_19_e[2][row] = add(carry1[20], M31_524288); // rc_19_e_2
    sub_inputs.rc_19_e[3][row] = add(carry2[4], M31_524288);  // rc_19_e_3
    sub_inputs.rc_19_e[4][row] = add(carry2[12], M31_524288); // rc_19_e_4
    sub_inputs.rc_19_e[5][row] = add(carry2[20], M31_524288); // rc_19_e_5

    // rc_19_f (6 lookups × 1 element = 6 pointers)
    sub_inputs.rc_19_f[0][row] = add(carry1[5], M31_524288);  // rc_19_f_0
    sub_inputs.rc_19_f[1][row] = add(carry1[13], M31_524288); // rc_19_f_1
    sub_inputs.rc_19_f[2][row] = add(carry1[21], M31_524288); // rc_19_f_2
    sub_inputs.rc_19_f[3][row] = add(carry2[5], M31_524288);  // rc_19_f_3
    sub_inputs.rc_19_f[4][row] = add(carry2[13], M31_524288); // rc_19_f_4
    sub_inputs.rc_19_f[5][row] = add(carry2[21], M31_524288); // rc_19_f_5

    // rc_19_g (6 lookups × 1 element = 6 pointers)
    sub_inputs.rc_19_g[0][row] = add(carry1[6], M31_524288);  // rc_19_g_0
    sub_inputs.rc_19_g[1][row] = add(carry1[14], M31_524288); // rc_19_g_1
    sub_inputs.rc_19_g[2][row] = add(carry1[22], M31_524288); // rc_19_g_2
    sub_inputs.rc_19_g[3][row] = add(carry2[6], M31_524288);  // rc_19_g_3
    sub_inputs.rc_19_g[4][row] = add(carry2[14], M31_524288); // rc_19_g_4
    sub_inputs.rc_19_g[5][row] = add(carry2[22], M31_524288); // rc_19_g_5

    // rc_19_h (8 lookups × 1 element = 8 pointers)
    sub_inputs.rc_19_h[0][row] = add(k1, M31_524288);         // rc_19_h_0 (k1 + 524288)
    sub_inputs.rc_19_h[1][row] = add(carry1[7], M31_524288);  // rc_19_h_1
    sub_inputs.rc_19_h[2][row] = add(carry1[15], M31_524288); // rc_19_h_2
    sub_inputs.rc_19_h[3][row] = add(carry1[23], M31_524288); // rc_19_h_3
    sub_inputs.rc_19_h[4][row] = add(k2, M31_524288);         // rc_19_h_4 (k2 + 524288)
    sub_inputs.rc_19_h[5][row] = add(carry2[7], M31_524288);  // rc_19_h_5
    sub_inputs.rc_19_h[6][row] = add(carry2[15], M31_524288); // rc_19_h_6
    sub_inputs.rc_19_h[7][row] = add(carry2[23], M31_524288); // rc_19_h_7
}

// ============================================================================
// Host functions
// ============================================================================

extern "C" void generate_cube_252_trace(
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
) {
    unsigned int n_rows = 1u << trace_log_size;

    // Copy input pointers to device
    m31** d_inputs;
    d_inputs = cuda_mem_pool_allocate<m31*>(10);
    cudaMemcpyAsync(d_inputs, inputs, 10 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy trace column pointers to device
    m31** d_trace_columns;
    d_trace_columns = cuda_mem_pool_allocate<m31*>(CUBE_252_N_TRACE_COLUMNS);
    cudaMemcpyAsync(d_trace_columns, trace_columns, CUBE_252_N_TRACE_COLUMNS * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy lookup data pointers to device
    // cube_252_0 has 20 elements
    m31** d_cube_252_0;
    d_cube_252_0 = cuda_mem_pool_allocate<m31*>(20);
    cudaMemcpyAsync(d_cube_252_0, lookup_cube_252_0, 20 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // rc_9_9 variants (2 elements each)
    m31** d_rc_9_9_0; d_rc_9_9_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_0, lookup_rc_9_9_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_1; d_rc_9_9_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_1, lookup_rc_9_9_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_2; d_rc_9_9_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_2, lookup_rc_9_9_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_3; d_rc_9_9_3 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_3, lookup_rc_9_9_3, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_4; d_rc_9_9_4 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_4, lookup_rc_9_9_4, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_5; d_rc_9_9_5 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_5, lookup_rc_9_9_5, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_b_0; d_rc_9_9_b_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_b_0, lookup_rc_9_9_b_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_b_1; d_rc_9_9_b_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_b_1, lookup_rc_9_9_b_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_b_2; d_rc_9_9_b_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_b_2, lookup_rc_9_9_b_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_b_3; d_rc_9_9_b_3 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_b_3, lookup_rc_9_9_b_3, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_b_4; d_rc_9_9_b_4 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_b_4, lookup_rc_9_9_b_4, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_b_5; d_rc_9_9_b_5 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_b_5, lookup_rc_9_9_b_5, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_c_0; d_rc_9_9_c_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_c_0, lookup_rc_9_9_c_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_c_1; d_rc_9_9_c_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_c_1, lookup_rc_9_9_c_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_c_2; d_rc_9_9_c_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_c_2, lookup_rc_9_9_c_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_c_3; d_rc_9_9_c_3 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_c_3, lookup_rc_9_9_c_3, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_c_4; d_rc_9_9_c_4 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_c_4, lookup_rc_9_9_c_4, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_c_5; d_rc_9_9_c_5 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_c_5, lookup_rc_9_9_c_5, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_d_0; d_rc_9_9_d_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_d_0, lookup_rc_9_9_d_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_d_1; d_rc_9_9_d_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_d_1, lookup_rc_9_9_d_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_d_2; d_rc_9_9_d_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_d_2, lookup_rc_9_9_d_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_d_3; d_rc_9_9_d_3 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_d_3, lookup_rc_9_9_d_3, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_d_4; d_rc_9_9_d_4 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_d_4, lookup_rc_9_9_d_4, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_d_5; d_rc_9_9_d_5 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_d_5, lookup_rc_9_9_d_5, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_e_0; d_rc_9_9_e_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_e_0, lookup_rc_9_9_e_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_e_1; d_rc_9_9_e_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_e_1, lookup_rc_9_9_e_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_e_2; d_rc_9_9_e_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_e_2, lookup_rc_9_9_e_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_e_3; d_rc_9_9_e_3 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_e_3, lookup_rc_9_9_e_3, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_e_4; d_rc_9_9_e_4 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_e_4, lookup_rc_9_9_e_4, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_e_5; d_rc_9_9_e_5 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_e_5, lookup_rc_9_9_e_5, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_f_0; d_rc_9_9_f_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_f_0, lookup_rc_9_9_f_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_f_1; d_rc_9_9_f_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_f_1, lookup_rc_9_9_f_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_f_2; d_rc_9_9_f_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_f_2, lookup_rc_9_9_f_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_f_3; d_rc_9_9_f_3 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_f_3, lookup_rc_9_9_f_3, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_f_4; d_rc_9_9_f_4 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_f_4, lookup_rc_9_9_f_4, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_f_5; d_rc_9_9_f_5 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_f_5, lookup_rc_9_9_f_5, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_g_0; d_rc_9_9_g_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_g_0, lookup_rc_9_9_g_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_g_1; d_rc_9_9_g_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_g_1, lookup_rc_9_9_g_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_g_2; d_rc_9_9_g_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_g_2, lookup_rc_9_9_g_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_h_0; d_rc_9_9_h_0 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_h_0, lookup_rc_9_9_h_0, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_h_1; d_rc_9_9_h_1 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_h_1, lookup_rc_9_9_h_1, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_9_9_h_2; d_rc_9_9_h_2 = cuda_mem_pool_allocate<m31*>(2); cudaMemcpyAsync(d_rc_9_9_h_2, lookup_rc_9_9_h_2, 2 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // rc_19 variants (1 element each)
    m31** d_rc_19_0; d_rc_19_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_0, lookup_rc_19_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_1; d_rc_19_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_1, lookup_rc_19_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_2; d_rc_19_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_2, lookup_rc_19_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_3; d_rc_19_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_3, lookup_rc_19_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_4; d_rc_19_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_4, lookup_rc_19_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_5; d_rc_19_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_5, lookup_rc_19_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_6; d_rc_19_6 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_6, lookup_rc_19_6, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_7; d_rc_19_7 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_7, lookup_rc_19_7, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_0; d_rc_19_b_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_0, lookup_rc_19_b_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_1; d_rc_19_b_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_1, lookup_rc_19_b_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_2; d_rc_19_b_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_2, lookup_rc_19_b_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_3; d_rc_19_b_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_3, lookup_rc_19_b_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_4; d_rc_19_b_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_4, lookup_rc_19_b_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_5; d_rc_19_b_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_5, lookup_rc_19_b_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_6; d_rc_19_b_6 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_6, lookup_rc_19_b_6, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_b_7; d_rc_19_b_7 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_b_7, lookup_rc_19_b_7, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_0; d_rc_19_c_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_0, lookup_rc_19_c_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_1; d_rc_19_c_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_1, lookup_rc_19_c_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_2; d_rc_19_c_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_2, lookup_rc_19_c_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_3; d_rc_19_c_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_3, lookup_rc_19_c_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_4; d_rc_19_c_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_4, lookup_rc_19_c_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_5; d_rc_19_c_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_5, lookup_rc_19_c_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_6; d_rc_19_c_6 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_6, lookup_rc_19_c_6, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_c_7; d_rc_19_c_7 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_c_7, lookup_rc_19_c_7, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_d_0; d_rc_19_d_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_d_0, lookup_rc_19_d_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_d_1; d_rc_19_d_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_d_1, lookup_rc_19_d_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_d_2; d_rc_19_d_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_d_2, lookup_rc_19_d_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_d_3; d_rc_19_d_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_d_3, lookup_rc_19_d_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_d_4; d_rc_19_d_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_d_4, lookup_rc_19_d_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_d_5; d_rc_19_d_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_d_5, lookup_rc_19_d_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_e_0; d_rc_19_e_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_e_0, lookup_rc_19_e_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_e_1; d_rc_19_e_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_e_1, lookup_rc_19_e_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_e_2; d_rc_19_e_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_e_2, lookup_rc_19_e_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_e_3; d_rc_19_e_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_e_3, lookup_rc_19_e_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_e_4; d_rc_19_e_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_e_4, lookup_rc_19_e_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_e_5; d_rc_19_e_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_e_5, lookup_rc_19_e_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_f_0; d_rc_19_f_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_f_0, lookup_rc_19_f_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_f_1; d_rc_19_f_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_f_1, lookup_rc_19_f_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_f_2; d_rc_19_f_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_f_2, lookup_rc_19_f_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_f_3; d_rc_19_f_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_f_3, lookup_rc_19_f_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_f_4; d_rc_19_f_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_f_4, lookup_rc_19_f_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_f_5; d_rc_19_f_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_f_5, lookup_rc_19_f_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_g_0; d_rc_19_g_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_g_0, lookup_rc_19_g_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_g_1; d_rc_19_g_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_g_1, lookup_rc_19_g_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_g_2; d_rc_19_g_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_g_2, lookup_rc_19_g_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_g_3; d_rc_19_g_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_g_3, lookup_rc_19_g_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_g_4; d_rc_19_g_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_g_4, lookup_rc_19_g_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_g_5; d_rc_19_g_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_g_5, lookup_rc_19_g_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_0; d_rc_19_h_0 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_0, lookup_rc_19_h_0, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_1; d_rc_19_h_1 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_1, lookup_rc_19_h_1, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_2; d_rc_19_h_2 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_2, lookup_rc_19_h_2, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_3; d_rc_19_h_3 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_3, lookup_rc_19_h_3, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_4; d_rc_19_h_4 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_4, lookup_rc_19_h_4, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_5; d_rc_19_h_5 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_5, lookup_rc_19_h_5, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_6; d_rc_19_h_6 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_6, lookup_rc_19_h_6, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_rc_19_h_7; d_rc_19_h_7 = cuda_mem_pool_allocate<m31*>(1); cudaMemcpyAsync(d_rc_19_h_7, lookup_rc_19_h_7, 1 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy sub_component_inputs pointers to device
    // rc_9_9 variants
    m31** d_sub_rc_9_9; d_sub_rc_9_9 = cuda_mem_pool_allocate<m31*>(12); cudaMemcpyAsync(d_sub_rc_9_9, sub_rc_9_9, 12 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_9_9_b; d_sub_rc_9_9_b = cuda_mem_pool_allocate<m31*>(12); cudaMemcpyAsync(d_sub_rc_9_9_b, sub_rc_9_9_b, 12 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_9_9_c; d_sub_rc_9_9_c = cuda_mem_pool_allocate<m31*>(12); cudaMemcpyAsync(d_sub_rc_9_9_c, sub_rc_9_9_c, 12 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_9_9_d; d_sub_rc_9_9_d = cuda_mem_pool_allocate<m31*>(12); cudaMemcpyAsync(d_sub_rc_9_9_d, sub_rc_9_9_d, 12 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_9_9_e; d_sub_rc_9_9_e = cuda_mem_pool_allocate<m31*>(12); cudaMemcpyAsync(d_sub_rc_9_9_e, sub_rc_9_9_e, 12 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_9_9_f; d_sub_rc_9_9_f = cuda_mem_pool_allocate<m31*>(12); cudaMemcpyAsync(d_sub_rc_9_9_f, sub_rc_9_9_f, 12 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_9_9_g; d_sub_rc_9_9_g = cuda_mem_pool_allocate<m31*>(6); cudaMemcpyAsync(d_sub_rc_9_9_g, sub_rc_9_9_g, 6 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_9_9_h; d_sub_rc_9_9_h = cuda_mem_pool_allocate<m31*>(6); cudaMemcpyAsync(d_sub_rc_9_9_h, sub_rc_9_9_h, 6 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    // rc_19 variants
    m31** d_sub_rc_19; d_sub_rc_19 = cuda_mem_pool_allocate<m31*>(8); cudaMemcpyAsync(d_sub_rc_19, sub_rc_19, 8 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_19_b; d_sub_rc_19_b = cuda_mem_pool_allocate<m31*>(8); cudaMemcpyAsync(d_sub_rc_19_b, sub_rc_19_b, 8 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_19_c; d_sub_rc_19_c = cuda_mem_pool_allocate<m31*>(8); cudaMemcpyAsync(d_sub_rc_19_c, sub_rc_19_c, 8 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_19_d; d_sub_rc_19_d = cuda_mem_pool_allocate<m31*>(6); cudaMemcpyAsync(d_sub_rc_19_d, sub_rc_19_d, 6 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_19_e; d_sub_rc_19_e = cuda_mem_pool_allocate<m31*>(6); cudaMemcpyAsync(d_sub_rc_19_e, sub_rc_19_e, 6 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_19_f; d_sub_rc_19_f = cuda_mem_pool_allocate<m31*>(6); cudaMemcpyAsync(d_sub_rc_19_f, sub_rc_19_f, 6 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_19_g; d_sub_rc_19_g = cuda_mem_pool_allocate<m31*>(6); cudaMemcpyAsync(d_sub_rc_19_g, sub_rc_19_g, 6 * sizeof(m31*), cudaMemcpyHostToDevice, 0);
    m31** d_sub_rc_19_h; d_sub_rc_19_h = cuda_mem_pool_allocate<m31*>(8); cudaMemcpyAsync(d_sub_rc_19_h, sub_rc_19_h, 8 * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Build the lookup pointers structure
    Cube252LookupPtrs lookup = {
        d_cube_252_0,
        // rc_9_9 variants
        d_rc_9_9_0, d_rc_9_9_1, d_rc_9_9_2, d_rc_9_9_3, d_rc_9_9_4, d_rc_9_9_5,
        d_rc_9_9_b_0, d_rc_9_9_b_1, d_rc_9_9_b_2, d_rc_9_9_b_3, d_rc_9_9_b_4, d_rc_9_9_b_5,
        d_rc_9_9_c_0, d_rc_9_9_c_1, d_rc_9_9_c_2, d_rc_9_9_c_3, d_rc_9_9_c_4, d_rc_9_9_c_5,
        d_rc_9_9_d_0, d_rc_9_9_d_1, d_rc_9_9_d_2, d_rc_9_9_d_3, d_rc_9_9_d_4, d_rc_9_9_d_5,
        d_rc_9_9_e_0, d_rc_9_9_e_1, d_rc_9_9_e_2, d_rc_9_9_e_3, d_rc_9_9_e_4, d_rc_9_9_e_5,
        d_rc_9_9_f_0, d_rc_9_9_f_1, d_rc_9_9_f_2, d_rc_9_9_f_3, d_rc_9_9_f_4, d_rc_9_9_f_5,
        d_rc_9_9_g_0, d_rc_9_9_g_1, d_rc_9_9_g_2,
        d_rc_9_9_h_0, d_rc_9_9_h_1, d_rc_9_9_h_2,
        // rc_19 variants
        d_rc_19_0, d_rc_19_1, d_rc_19_2, d_rc_19_3, d_rc_19_4, d_rc_19_5, d_rc_19_6, d_rc_19_7,
        d_rc_19_b_0, d_rc_19_b_1, d_rc_19_b_2, d_rc_19_b_3, d_rc_19_b_4, d_rc_19_b_5, d_rc_19_b_6, d_rc_19_b_7,
        d_rc_19_c_0, d_rc_19_c_1, d_rc_19_c_2, d_rc_19_c_3, d_rc_19_c_4, d_rc_19_c_5, d_rc_19_c_6, d_rc_19_c_7,
        d_rc_19_d_0, d_rc_19_d_1, d_rc_19_d_2, d_rc_19_d_3, d_rc_19_d_4, d_rc_19_d_5,
        d_rc_19_e_0, d_rc_19_e_1, d_rc_19_e_2, d_rc_19_e_3, d_rc_19_e_4, d_rc_19_e_5,
        d_rc_19_f_0, d_rc_19_f_1, d_rc_19_f_2, d_rc_19_f_3, d_rc_19_f_4, d_rc_19_f_5,
        d_rc_19_g_0, d_rc_19_g_1, d_rc_19_g_2, d_rc_19_g_3, d_rc_19_g_4, d_rc_19_g_5,
        d_rc_19_h_0, d_rc_19_h_1, d_rc_19_h_2, d_rc_19_h_3, d_rc_19_h_4, d_rc_19_h_5, d_rc_19_h_6, d_rc_19_h_7
    };

    // Build the sub_component_inputs structure
    Cube252SubComponentInputs sub_inputs = {
        d_sub_rc_9_9, d_sub_rc_9_9_b, d_sub_rc_9_9_c, d_sub_rc_9_9_d,
        d_sub_rc_9_9_e, d_sub_rc_9_9_f, d_sub_rc_9_9_g, d_sub_rc_9_9_h,
        d_sub_rc_19, d_sub_rc_19_b, d_sub_rc_19_c, d_sub_rc_19_d,
        d_sub_rc_19_e, d_sub_rc_19_f, d_sub_rc_19_g, d_sub_rc_19_h
    };

    // Launch kernel
    int block_size = CUBE_252_BLOCK_SIZE;
    int num_blocks = (n_rows + block_size - 1) / block_size;

    cube_252_trace_kernel<<<num_blocks, block_size>>>(
        d_inputs,
        n_rows,
        d_trace_columns,
        lookup,
        sub_inputs
    );


    // Cleanup - free all device pointer arrays
    cuda_mem_pool_free(d_inputs);
    cuda_mem_pool_free(d_trace_columns);
    cuda_mem_pool_free(d_cube_252_0);
    // rc_9_9 variants
    cuda_mem_pool_free(d_rc_9_9_0); cuda_mem_pool_free(d_rc_9_9_1); cuda_mem_pool_free(d_rc_9_9_2);
    cuda_mem_pool_free(d_rc_9_9_3); cuda_mem_pool_free(d_rc_9_9_4); cuda_mem_pool_free(d_rc_9_9_5);
    cuda_mem_pool_free(d_rc_9_9_b_0); cuda_mem_pool_free(d_rc_9_9_b_1); cuda_mem_pool_free(d_rc_9_9_b_2);
    cuda_mem_pool_free(d_rc_9_9_b_3); cuda_mem_pool_free(d_rc_9_9_b_4); cuda_mem_pool_free(d_rc_9_9_b_5);
    cuda_mem_pool_free(d_rc_9_9_c_0); cuda_mem_pool_free(d_rc_9_9_c_1); cuda_mem_pool_free(d_rc_9_9_c_2);
    cuda_mem_pool_free(d_rc_9_9_c_3); cuda_mem_pool_free(d_rc_9_9_c_4); cuda_mem_pool_free(d_rc_9_9_c_5);
    cuda_mem_pool_free(d_rc_9_9_d_0); cuda_mem_pool_free(d_rc_9_9_d_1); cuda_mem_pool_free(d_rc_9_9_d_2);
    cuda_mem_pool_free(d_rc_9_9_d_3); cuda_mem_pool_free(d_rc_9_9_d_4); cuda_mem_pool_free(d_rc_9_9_d_5);
    cuda_mem_pool_free(d_rc_9_9_e_0); cuda_mem_pool_free(d_rc_9_9_e_1); cuda_mem_pool_free(d_rc_9_9_e_2);
    cuda_mem_pool_free(d_rc_9_9_e_3); cuda_mem_pool_free(d_rc_9_9_e_4); cuda_mem_pool_free(d_rc_9_9_e_5);
    cuda_mem_pool_free(d_rc_9_9_f_0); cuda_mem_pool_free(d_rc_9_9_f_1); cuda_mem_pool_free(d_rc_9_9_f_2);
    cuda_mem_pool_free(d_rc_9_9_f_3); cuda_mem_pool_free(d_rc_9_9_f_4); cuda_mem_pool_free(d_rc_9_9_f_5);
    cuda_mem_pool_free(d_rc_9_9_g_0); cuda_mem_pool_free(d_rc_9_9_g_1); cuda_mem_pool_free(d_rc_9_9_g_2);
    cuda_mem_pool_free(d_rc_9_9_h_0); cuda_mem_pool_free(d_rc_9_9_h_1); cuda_mem_pool_free(d_rc_9_9_h_2);
    // rc_19 variants
    cuda_mem_pool_free(d_rc_19_0); cuda_mem_pool_free(d_rc_19_1); cuda_mem_pool_free(d_rc_19_2); cuda_mem_pool_free(d_rc_19_3);
    cuda_mem_pool_free(d_rc_19_4); cuda_mem_pool_free(d_rc_19_5); cuda_mem_pool_free(d_rc_19_6); cuda_mem_pool_free(d_rc_19_7);
    cuda_mem_pool_free(d_rc_19_b_0); cuda_mem_pool_free(d_rc_19_b_1); cuda_mem_pool_free(d_rc_19_b_2); cuda_mem_pool_free(d_rc_19_b_3);
    cuda_mem_pool_free(d_rc_19_b_4); cuda_mem_pool_free(d_rc_19_b_5); cuda_mem_pool_free(d_rc_19_b_6); cuda_mem_pool_free(d_rc_19_b_7);
    cuda_mem_pool_free(d_rc_19_c_0); cuda_mem_pool_free(d_rc_19_c_1); cuda_mem_pool_free(d_rc_19_c_2); cuda_mem_pool_free(d_rc_19_c_3);
    cuda_mem_pool_free(d_rc_19_c_4); cuda_mem_pool_free(d_rc_19_c_5); cuda_mem_pool_free(d_rc_19_c_6); cuda_mem_pool_free(d_rc_19_c_7);
    cuda_mem_pool_free(d_rc_19_d_0); cuda_mem_pool_free(d_rc_19_d_1); cuda_mem_pool_free(d_rc_19_d_2);
    cuda_mem_pool_free(d_rc_19_d_3); cuda_mem_pool_free(d_rc_19_d_4); cuda_mem_pool_free(d_rc_19_d_5);
    cuda_mem_pool_free(d_rc_19_e_0); cuda_mem_pool_free(d_rc_19_e_1); cuda_mem_pool_free(d_rc_19_e_2);
    cuda_mem_pool_free(d_rc_19_e_3); cuda_mem_pool_free(d_rc_19_e_4); cuda_mem_pool_free(d_rc_19_e_5);
    cuda_mem_pool_free(d_rc_19_f_0); cuda_mem_pool_free(d_rc_19_f_1); cuda_mem_pool_free(d_rc_19_f_2);
    cuda_mem_pool_free(d_rc_19_f_3); cuda_mem_pool_free(d_rc_19_f_4); cuda_mem_pool_free(d_rc_19_f_5);
    cuda_mem_pool_free(d_rc_19_g_0); cuda_mem_pool_free(d_rc_19_g_1); cuda_mem_pool_free(d_rc_19_g_2);
    cuda_mem_pool_free(d_rc_19_g_3); cuda_mem_pool_free(d_rc_19_g_4); cuda_mem_pool_free(d_rc_19_g_5);
    cuda_mem_pool_free(d_rc_19_h_0); cuda_mem_pool_free(d_rc_19_h_1); cuda_mem_pool_free(d_rc_19_h_2); cuda_mem_pool_free(d_rc_19_h_3);
    cuda_mem_pool_free(d_rc_19_h_4); cuda_mem_pool_free(d_rc_19_h_5); cuda_mem_pool_free(d_rc_19_h_6); cuda_mem_pool_free(d_rc_19_h_7);
    // sub_component_inputs variants
    cuda_mem_pool_free(d_sub_rc_9_9); cuda_mem_pool_free(d_sub_rc_9_9_b); cuda_mem_pool_free(d_sub_rc_9_9_c); cuda_mem_pool_free(d_sub_rc_9_9_d);
    cuda_mem_pool_free(d_sub_rc_9_9_e); cuda_mem_pool_free(d_sub_rc_9_9_f); cuda_mem_pool_free(d_sub_rc_9_9_g); cuda_mem_pool_free(d_sub_rc_9_9_h);
    cuda_mem_pool_free(d_sub_rc_19); cuda_mem_pool_free(d_sub_rc_19_b); cuda_mem_pool_free(d_sub_rc_19_c); cuda_mem_pool_free(d_sub_rc_19_d);
    cuda_mem_pool_free(d_sub_rc_19_e); cuda_mem_pool_free(d_sub_rc_19_f); cuda_mem_pool_free(d_sub_rc_19_g); cuda_mem_pool_free(d_sub_rc_19_h);
}

// ============================================================================
// Range check multiplicities update kernel
// Updates all 16 range check tables (8 x rc_9_9 variants + 8 x rc_19 variants)
// ============================================================================

// Helper: compute 9_9 table index from two 9-bit limbs
// The index formula is: limb0 * 512 + limb1 = (limb0 << 9) | limb1
// This matches range_check_vector_add_inputs_kernel which computes:
//   index = (inputs[0] << 9) + inputs[1]
// So the first argument goes into the HIGH part (bits 9-17).
__device__ __forceinline__ unsigned int rc_9_9_index(m31 limb0, m31 limb1) {
    return (limb0 << 9) + limb1;
}

// ============================================================================
// Interaction trace generation kernels
// ============================================================================

// Number of logup columns for cube_252 (50 logup columns)
#define CUBE_252_N_LOGUP_COLS 50

// Helper: compute unpacked limb 2 from trace
// unpacked_2 = ((input_0 - unpacked_0) - (unpacked_1 * 512)) * 8192
__device__ __forceinline__ m31 compute_unpacked_2(m31 input_0, m31 unpacked_0, m31 unpacked_1) {
    return mul(sub(sub(input_0, unpacked_0), mul(unpacked_1, (m31)512)), (m31)8192);
}

// Compute all 50 logup fractions for cube_252
__global__ void cube_252_compute_fractions_kernel(
    m31** trace_columns,
    unsigned int trace_size,
    // Lookup elements (17 relations)
    LookupElementsBasic<20>* cube_252_le,
    LookupElementsBasic<2>* rc_9_9_le,
    LookupElementsBasic<2>* rc_9_9_b_le,
    LookupElementsBasic<2>* rc_9_9_c_le,
    LookupElementsBasic<2>* rc_9_9_d_le,
    LookupElementsBasic<2>* rc_9_9_e_le,
    LookupElementsBasic<2>* rc_9_9_f_le,
    LookupElementsBasic<2>* rc_9_9_g_le,
    LookupElementsBasic<2>* rc_9_9_h_le,
    LookupElementsBasic<1>* rc_19_le,
    LookupElementsBasic<1>* rc_19_b_le,
    LookupElementsBasic<1>* rc_19_c_le,
    LookupElementsBasic<1>* rc_19_d_le,
    LookupElementsBasic<1>* rc_19_e_le,
    LookupElementsBasic<1>* rc_19_f_le,
    LookupElementsBasic<1>* rc_19_g_le,
    LookupElementsBasic<1>* rc_19_h_le,
    // Output arrays for fractions
    qm31* denom_ptr,          // [50 * trace_size]
    m31* numerator0,          // [50 * trace_size]
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Read trace columns
    // Input limbs (0-9)
    m31 input_0 = trace_columns[0][row];
    m31 input_1 = trace_columns[1][row];
    m31 input_2 = trace_columns[2][row];
    m31 input_3 = trace_columns[3][row];
    m31 input_4 = trace_columns[4][row];
    m31 input_5 = trace_columns[5][row];
    m31 input_6 = trace_columns[6][row];
    m31 input_7 = trace_columns[7][row];
    m31 input_8 = trace_columns[8][row];
    m31 input_9 = trace_columns[9][row];

    // Unpacked limbs (10-27)
    m31 unpacked_0  = trace_columns[10][row];  // limb 0
    m31 unpacked_1  = trace_columns[11][row];  // limb 1
    m31 unpacked_3  = trace_columns[12][row];  // limb 3
    m31 unpacked_4  = trace_columns[13][row];  // limb 4
    m31 unpacked_6  = trace_columns[14][row];  // limb 6
    m31 unpacked_7  = trace_columns[15][row];  // limb 7
    m31 unpacked_9  = trace_columns[16][row];  // limb 9
    m31 unpacked_10 = trace_columns[17][row];  // limb 10
    m31 unpacked_12 = trace_columns[18][row];  // limb 12
    m31 unpacked_13 = trace_columns[19][row];  // limb 13
    m31 unpacked_15 = trace_columns[20][row];  // limb 15
    m31 unpacked_16 = trace_columns[21][row];  // limb 16
    m31 unpacked_18 = trace_columns[22][row];  // limb 18
    m31 unpacked_19 = trace_columns[23][row];  // limb 19
    m31 unpacked_21 = trace_columns[24][row];  // limb 21
    m31 unpacked_22 = trace_columns[25][row];  // limb 22
    m31 unpacked_24 = trace_columns[26][row];  // limb 24
    m31 unpacked_25 = trace_columns[27][row];  // limb 25

    // Compute derived unpacked limbs (limbs 2,5,8,11,14,17,20,23,26)
    m31 unpacked_2  = compute_unpacked_2(input_0, unpacked_0, unpacked_1);
    m31 unpacked_5  = compute_unpacked_2(input_1, unpacked_3, unpacked_4);
    m31 unpacked_8  = compute_unpacked_2(input_2, unpacked_6, unpacked_7);
    m31 unpacked_11 = compute_unpacked_2(input_3, unpacked_9, unpacked_10);
    m31 unpacked_14 = compute_unpacked_2(input_4, unpacked_12, unpacked_13);
    m31 unpacked_17 = compute_unpacked_2(input_5, unpacked_15, unpacked_16);
    m31 unpacked_20 = compute_unpacked_2(input_6, unpacked_18, unpacked_19);
    m31 unpacked_23 = compute_unpacked_2(input_7, unpacked_21, unpacked_22);
    m31 unpacked_26 = compute_unpacked_2(input_8, unpacked_24, unpacked_25);

    // First mul result (28-55)
    m31 mul1_0  = trace_columns[28][row];
    m31 mul1_1  = trace_columns[29][row];
    m31 mul1_2  = trace_columns[30][row];
    m31 mul1_3  = trace_columns[31][row];
    m31 mul1_4  = trace_columns[32][row];
    m31 mul1_5  = trace_columns[33][row];
    m31 mul1_6  = trace_columns[34][row];
    m31 mul1_7  = trace_columns[35][row];
    m31 mul1_8  = trace_columns[36][row];
    m31 mul1_9  = trace_columns[37][row];
    m31 mul1_10 = trace_columns[38][row];
    m31 mul1_11 = trace_columns[39][row];
    m31 mul1_12 = trace_columns[40][row];
    m31 mul1_13 = trace_columns[41][row];
    m31 mul1_14 = trace_columns[42][row];
    m31 mul1_15 = trace_columns[43][row];
    m31 mul1_16 = trace_columns[44][row];
    m31 mul1_17 = trace_columns[45][row];
    m31 mul1_18 = trace_columns[46][row];
    m31 mul1_19 = trace_columns[47][row];
    m31 mul1_20 = trace_columns[48][row];
    m31 mul1_21 = trace_columns[49][row];
    m31 mul1_22 = trace_columns[50][row];
    m31 mul1_23 = trace_columns[51][row];
    m31 mul1_24 = trace_columns[52][row];
    m31 mul1_25 = trace_columns[53][row];
    m31 mul1_26 = trace_columns[54][row];
    m31 mul1_27 = trace_columns[55][row];

    // k and carries for first mul (56-83)
    m31 k1 = trace_columns[56][row];
    m31 carry1[27];
    for (int i = 0; i < 27; i++) {
        carry1[i] = trace_columns[57 + i][row];
    }

    // Second mul result (84-111)
    m31 mul2_0  = trace_columns[84][row];
    m31 mul2_1  = trace_columns[85][row];
    m31 mul2_2  = trace_columns[86][row];
    m31 mul2_3  = trace_columns[87][row];
    m31 mul2_4  = trace_columns[88][row];
    m31 mul2_5  = trace_columns[89][row];
    m31 mul2_6  = trace_columns[90][row];
    m31 mul2_7  = trace_columns[91][row];
    m31 mul2_8  = trace_columns[92][row];
    m31 mul2_9  = trace_columns[93][row];
    m31 mul2_10 = trace_columns[94][row];
    m31 mul2_11 = trace_columns[95][row];
    m31 mul2_12 = trace_columns[96][row];
    m31 mul2_13 = trace_columns[97][row];
    m31 mul2_14 = trace_columns[98][row];
    m31 mul2_15 = trace_columns[99][row];
    m31 mul2_16 = trace_columns[100][row];
    m31 mul2_17 = trace_columns[101][row];
    m31 mul2_18 = trace_columns[102][row];
    m31 mul2_19 = trace_columns[103][row];
    m31 mul2_20 = trace_columns[104][row];
    m31 mul2_21 = trace_columns[105][row];
    m31 mul2_22 = trace_columns[106][row];
    m31 mul2_23 = trace_columns[107][row];
    m31 mul2_24 = trace_columns[108][row];
    m31 mul2_25 = trace_columns[109][row];
    m31 mul2_26 = trace_columns[110][row];
    m31 mul2_27 = trace_columns[111][row];

    // k and carries for second mul (112-139)
    m31 k2 = trace_columns[112][row];
    m31 carry2[27];
    for (int i = 0; i < 27; i++) {
        carry2[i] = trace_columns[113 + i][row];
    }

    // Enabler (140)
    m31 enabler = trace_columns[140][row];

    // M31 constants for rc_19/rc_20 lookups: M31_524288 (2^19) for both carry and k values

    // ========================================================================
    // LogUp columns 0-13: Range check 9_9 pairs for input unpacking
    // ========================================================================

    // Col 0: rc_9_9_0 [unpacked_0, unpacked_1] + rc_9_9_b_0 [unpacked_2, unpacked_3]
    {
        m31 vals0[2] = {unpacked_0, unpacked_1};
        m31 vals1[2] = {unpacked_2, unpacked_3};
        qm31 denom0 = rc_9_9_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_b_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(0 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 1: rc_9_9_c_0 [unpacked_4, unpacked_5] + rc_9_9_d_0 [unpacked_6, unpacked_7]
    {
        m31 vals0[2] = {unpacked_4, unpacked_5};
        m31 vals1[2] = {unpacked_6, unpacked_7};
        qm31 denom0 = rc_9_9_c_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_d_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(1 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 2: rc_9_9_e_0 [unpacked_8, unpacked_9] + rc_9_9_f_0 [unpacked_10, unpacked_11]
    {
        m31 vals0[2] = {unpacked_8, unpacked_9};
        m31 vals1[2] = {unpacked_10, unpacked_11};
        qm31 denom0 = rc_9_9_e_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_f_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(2 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 3: rc_9_9_g_0 [unpacked_12, unpacked_13] + rc_9_9_h_0 [unpacked_14, unpacked_15]
    {
        m31 vals0[2] = {unpacked_12, unpacked_13};
        m31 vals1[2] = {unpacked_14, unpacked_15};
        qm31 denom0 = rc_9_9_g_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_h_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(3 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 4: rc_9_9_1 [unpacked_16, unpacked_17] + rc_9_9_b_1 [unpacked_18, unpacked_19]
    {
        m31 vals0[2] = {unpacked_16, unpacked_17};
        m31 vals1[2] = {unpacked_18, unpacked_19};
        qm31 denom0 = rc_9_9_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_b_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(4 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 5: rc_9_9_c_1 [unpacked_20, unpacked_21] + rc_9_9_d_1 [unpacked_22, unpacked_23]
    {
        m31 vals0[2] = {unpacked_20, unpacked_21};
        m31 vals1[2] = {unpacked_22, unpacked_23};
        qm31 denom0 = rc_9_9_c_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_d_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(5 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 6: rc_9_9_e_1 [unpacked_24, unpacked_25] + rc_9_9_f_1 [unpacked_26, input_9]
    {
        m31 vals0[2] = {unpacked_24, unpacked_25};
        m31 vals1[2] = {unpacked_26, input_9};
        qm31 denom0 = rc_9_9_e_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_f_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(6 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // ========================================================================
    // LogUp columns 7-13: Range check 9_9 pairs for first mul result
    // ========================================================================

    // Col 7: rc_9_9_2 [mul1_0, mul1_1] + rc_9_9_b_2 [mul1_2, mul1_3]
    {
        m31 vals0[2] = {mul1_0, mul1_1};
        m31 vals1[2] = {mul1_2, mul1_3};
        qm31 denom0 = rc_9_9_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_b_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(7 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 8: rc_9_9_c_2 [mul1_4, mul1_5] + rc_9_9_d_2 [mul1_6, mul1_7]
    {
        m31 vals0[2] = {mul1_4, mul1_5};
        m31 vals1[2] = {mul1_6, mul1_7};
        qm31 denom0 = rc_9_9_c_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_d_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(8 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 9: rc_9_9_e_2 [mul1_8, mul1_9] + rc_9_9_f_2 [mul1_10, mul1_11]
    {
        m31 vals0[2] = {mul1_8, mul1_9};
        m31 vals1[2] = {mul1_10, mul1_11};
        qm31 denom0 = rc_9_9_e_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_f_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(9 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 10: rc_9_9_g_1 [mul1_12, mul1_13] + rc_9_9_h_1 [mul1_14, mul1_15]
    {
        m31 vals0[2] = {mul1_12, mul1_13};
        m31 vals1[2] = {mul1_14, mul1_15};
        qm31 denom0 = rc_9_9_g_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_h_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(10 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 11: rc_9_9_3 [mul1_16, mul1_17] + rc_9_9_b_3 [mul1_18, mul1_19]
    {
        m31 vals0[2] = {mul1_16, mul1_17};
        m31 vals1[2] = {mul1_18, mul1_19};
        qm31 denom0 = rc_9_9_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_b_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(11 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 12: rc_9_9_c_3 [mul1_20, mul1_21] + rc_9_9_d_3 [mul1_22, mul1_23]
    {
        m31 vals0[2] = {mul1_20, mul1_21};
        m31 vals1[2] = {mul1_22, mul1_23};
        qm31 denom0 = rc_9_9_c_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_d_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(12 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 13: rc_9_9_e_3 [mul1_24, mul1_25] + rc_9_9_f_3 [mul1_26, mul1_27]
    {
        m31 vals0[2] = {mul1_24, mul1_25};
        m31 vals1[2] = {mul1_26, mul1_27};
        qm31 denom0 = rc_9_9_e_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_f_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(13 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // ========================================================================
    // LogUp columns 14-27: Range check 19 pairs for first mul carries
    // ========================================================================

    // Col 14: rc_19_h_0 [k1 + 524288] + rc_19_0 [carry1[0] + 524288]
    {
        m31 vals0[1] = {add(k1, M31_524288)};
        m31 vals1[1] = {add(carry1[0], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(14 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 15: rc_19_b_0 [carry1[1] + 131072] + rc_19_c_0 [carry1[2] + 131072]
    {
        m31 vals0[1] = {add(carry1[1], M31_524288)};
        m31 vals1[1] = {add(carry1[2], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(15 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 16: rc_19_d_0 [carry1[3] + 131072] + rc_19_e_0 [carry1[4] + 131072]
    {
        m31 vals0[1] = {add(carry1[3], M31_524288)};
        m31 vals1[1] = {add(carry1[4], M31_524288)};
        qm31 denom0 = rc_19_d_le->combine(vals0, 1);
        qm31 denom1 = rc_19_e_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(16 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 17: rc_19_f_0 [carry1[5] + 131072] + rc_19_g_0 [carry1[6] + 131072]
    {
        m31 vals0[1] = {add(carry1[5], M31_524288)};
        m31 vals1[1] = {add(carry1[6], M31_524288)};
        qm31 denom0 = rc_19_f_le->combine(vals0, 1);
        qm31 denom1 = rc_19_g_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(17 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 18: rc_19_h_1 [carry1[7] + 131072] + rc_19_1 [carry1[8] + 131072]
    {
        m31 vals0[1] = {add(carry1[7], M31_524288)};
        m31 vals1[1] = {add(carry1[8], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(18 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 19: rc_19_b_1 [carry1[9] + 131072] + rc_19_c_1 [carry1[10] + 131072]
    {
        m31 vals0[1] = {add(carry1[9], M31_524288)};
        m31 vals1[1] = {add(carry1[10], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(19 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 20: rc_19_d_1 [carry1[11] + 131072] + rc_19_e_1 [carry1[12] + 131072]
    {
        m31 vals0[1] = {add(carry1[11], M31_524288)};
        m31 vals1[1] = {add(carry1[12], M31_524288)};
        qm31 denom0 = rc_19_d_le->combine(vals0, 1);
        qm31 denom1 = rc_19_e_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(20 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 21: rc_19_f_1 [carry1[13] + 131072] + rc_19_g_1 [carry1[14] + 131072]
    {
        m31 vals0[1] = {add(carry1[13], M31_524288)};
        m31 vals1[1] = {add(carry1[14], M31_524288)};
        qm31 denom0 = rc_19_f_le->combine(vals0, 1);
        qm31 denom1 = rc_19_g_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(21 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 22: rc_19_h_2 [carry1[15] + 131072] + rc_19_2 [carry1[16] + 131072]
    {
        m31 vals0[1] = {add(carry1[15], M31_524288)};
        m31 vals1[1] = {add(carry1[16], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(22 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 23: rc_19_b_2 [carry1[17] + 131072] + rc_19_c_2 [carry1[18] + 131072]
    {
        m31 vals0[1] = {add(carry1[17], M31_524288)};
        m31 vals1[1] = {add(carry1[18], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(23 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 24: rc_19_d_2 [carry1[19] + 131072] + rc_19_e_2 [carry1[20] + 131072]
    {
        m31 vals0[1] = {add(carry1[19], M31_524288)};
        m31 vals1[1] = {add(carry1[20], M31_524288)};
        qm31 denom0 = rc_19_d_le->combine(vals0, 1);
        qm31 denom1 = rc_19_e_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(24 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 25: rc_19_f_2 [carry1[21] + 131072] + rc_19_g_2 [carry1[22] + 131072]
    {
        m31 vals0[1] = {add(carry1[21], M31_524288)};
        m31 vals1[1] = {add(carry1[22], M31_524288)};
        qm31 denom0 = rc_19_f_le->combine(vals0, 1);
        qm31 denom1 = rc_19_g_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(25 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 26: rc_19_h_3 [carry1[23] + 131072] + rc_19_3 [carry1[24] + 131072]
    {
        m31 vals0[1] = {add(carry1[23], M31_524288)};
        m31 vals1[1] = {add(carry1[24], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(26 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 27: rc_19_b_3 [carry1[25] + 131072] + rc_19_c_3 [carry1[26] + 131072]
    {
        m31 vals0[1] = {add(carry1[25], M31_524288)};
        m31 vals1[1] = {add(carry1[26], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(27 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // ========================================================================
    // LogUp columns 28-34: Range check 9_9 pairs for second mul result
    // ========================================================================

    // Col 28: rc_9_9_4 [mul2_0, mul2_1] + rc_9_9_b_4 [mul2_2, mul2_3]
    {
        m31 vals0[2] = {mul2_0, mul2_1};
        m31 vals1[2] = {mul2_2, mul2_3};
        qm31 denom0 = rc_9_9_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_b_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(28 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 29: rc_9_9_c_4 [mul2_4, mul2_5] + rc_9_9_d_4 [mul2_6, mul2_7]
    {
        m31 vals0[2] = {mul2_4, mul2_5};
        m31 vals1[2] = {mul2_6, mul2_7};
        qm31 denom0 = rc_9_9_c_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_d_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(29 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 30: rc_9_9_e_4 [mul2_8, mul2_9] + rc_9_9_f_4 [mul2_10, mul2_11]
    {
        m31 vals0[2] = {mul2_8, mul2_9};
        m31 vals1[2] = {mul2_10, mul2_11};
        qm31 denom0 = rc_9_9_e_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_f_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(30 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 31: rc_9_9_g_2 [mul2_12, mul2_13] + rc_9_9_h_2 [mul2_14, mul2_15]
    {
        m31 vals0[2] = {mul2_12, mul2_13};
        m31 vals1[2] = {mul2_14, mul2_15};
        qm31 denom0 = rc_9_9_g_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_h_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(31 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 32: rc_9_9_5 [mul2_16, mul2_17] + rc_9_9_b_5 [mul2_18, mul2_19]
    {
        m31 vals0[2] = {mul2_16, mul2_17};
        m31 vals1[2] = {mul2_18, mul2_19};
        qm31 denom0 = rc_9_9_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_b_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(32 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 33: rc_9_9_c_5 [mul2_20, mul2_21] + rc_9_9_d_5 [mul2_22, mul2_23]
    {
        m31 vals0[2] = {mul2_20, mul2_21};
        m31 vals1[2] = {mul2_22, mul2_23};
        qm31 denom0 = rc_9_9_c_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_d_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(33 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 34: rc_9_9_e_5 [mul2_24, mul2_25] + rc_9_9_f_5 [mul2_26, mul2_27]
    {
        m31 vals0[2] = {mul2_24, mul2_25};
        m31 vals1[2] = {mul2_26, mul2_27};
        qm31 denom0 = rc_9_9_e_le->combine(vals0, 2);
        qm31 denom1 = rc_9_9_f_le->combine(vals1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(34 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // ========================================================================
    // LogUp columns 35-48: Range check 19 pairs for second mul carries
    // ========================================================================

    // Col 35: rc_19_h_4 [k2 + 524288] + rc_19_4 [carry2[0] + 524288]
    {
        m31 vals0[1] = {add(k2, M31_524288)};
        m31 vals1[1] = {add(carry2[0], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(35 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 36: rc_19_b_4 [carry2[1] + 131072] + rc_19_c_4 [carry2[2] + 131072]
    {
        m31 vals0[1] = {add(carry2[1], M31_524288)};
        m31 vals1[1] = {add(carry2[2], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(36 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 37: rc_19_d_3 [carry2[3] + 131072] + rc_19_e_3 [carry2[4] + 131072]
    {
        m31 vals0[1] = {add(carry2[3], M31_524288)};
        m31 vals1[1] = {add(carry2[4], M31_524288)};
        qm31 denom0 = rc_19_d_le->combine(vals0, 1);
        qm31 denom1 = rc_19_e_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(37 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 38: rc_19_f_3 [carry2[5] + 131072] + rc_19_g_3 [carry2[6] + 131072]
    {
        m31 vals0[1] = {add(carry2[5], M31_524288)};
        m31 vals1[1] = {add(carry2[6], M31_524288)};
        qm31 denom0 = rc_19_f_le->combine(vals0, 1);
        qm31 denom1 = rc_19_g_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(38 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 39: rc_19_h_5 [carry2[7] + 131072] + rc_19_5 [carry2[8] + 131072]
    {
        m31 vals0[1] = {add(carry2[7], M31_524288)};
        m31 vals1[1] = {add(carry2[8], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(39 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 40: rc_19_b_5 [carry2[9] + 131072] + rc_19_c_5 [carry2[10] + 131072]
    {
        m31 vals0[1] = {add(carry2[9], M31_524288)};
        m31 vals1[1] = {add(carry2[10], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(40 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 41: rc_19_d_4 [carry2[11] + 131072] + rc_19_e_4 [carry2[12] + 131072]
    {
        m31 vals0[1] = {add(carry2[11], M31_524288)};
        m31 vals1[1] = {add(carry2[12], M31_524288)};
        qm31 denom0 = rc_19_d_le->combine(vals0, 1);
        qm31 denom1 = rc_19_e_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(41 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 42: rc_19_f_4 [carry2[13] + 131072] + rc_19_g_4 [carry2[14] + 131072]
    {
        m31 vals0[1] = {add(carry2[13], M31_524288)};
        m31 vals1[1] = {add(carry2[14], M31_524288)};
        qm31 denom0 = rc_19_f_le->combine(vals0, 1);
        qm31 denom1 = rc_19_g_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(42 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 43: rc_19_h_6 [carry2[15] + 131072] + rc_19_6 [carry2[16] + 131072]
    {
        m31 vals0[1] = {add(carry2[15], M31_524288)};
        m31 vals1[1] = {add(carry2[16], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(43 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 44: rc_19_b_6 [carry2[17] + 131072] + rc_19_c_6 [carry2[18] + 131072]
    {
        m31 vals0[1] = {add(carry2[17], M31_524288)};
        m31 vals1[1] = {add(carry2[18], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(44 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 45: rc_19_d_5 [carry2[19] + 131072] + rc_19_e_5 [carry2[20] + 131072]
    {
        m31 vals0[1] = {add(carry2[19], M31_524288)};
        m31 vals1[1] = {add(carry2[20], M31_524288)};
        qm31 denom0 = rc_19_d_le->combine(vals0, 1);
        qm31 denom1 = rc_19_e_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(45 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 46: rc_19_f_5 [carry2[21] + 131072] + rc_19_g_5 [carry2[22] + 131072]
    {
        m31 vals0[1] = {add(carry2[21], M31_524288)};
        m31 vals1[1] = {add(carry2[22], M31_524288)};
        qm31 denom0 = rc_19_f_le->combine(vals0, 1);
        qm31 denom1 = rc_19_g_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(46 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 47: rc_19_h_7 [carry2[23] + 131072] + rc_19_7 [carry2[24] + 131072]
    {
        m31 vals0[1] = {add(carry2[23], M31_524288)};
        m31 vals1[1] = {add(carry2[24], M31_524288)};
        qm31 denom0 = rc_19_h_le->combine(vals0, 1);
        qm31 denom1 = rc_19_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(47 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // Col 48: rc_19_b_7 [carry2[25] + 131072] + rc_19_c_7 [carry2[26] + 131072]
    {
        m31 vals0[1] = {add(carry2[25], M31_524288)};
        m31 vals1[1] = {add(carry2[26], M31_524288)};
        qm31 denom0 = rc_19_b_le->combine(vals0, 1);
        qm31 denom1 = rc_19_c_le->combine(vals1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        logup_col_write_frac(48 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // ========================================================================
    // LogUp column 49: Self-lookup with cube_252
    // ========================================================================

    // Col 49: cube_252 self-lookup (20 values)
    // Values are: input_0..input_9, and x^3 packed back to Width27 (10 values)
    {
        // Pack second mul result (x^3) back to Width27 format
        m31 cube_w27_0 = add(add(mul2_0, mul(mul2_1, M31_512)), mul(mul2_2, M31_262144));
        m31 cube_w27_1 = add(add(mul2_3, mul(mul2_4, M31_512)), mul(mul2_5, M31_262144));
        m31 cube_w27_2 = add(add(mul2_6, mul(mul2_7, M31_512)), mul(mul2_8, M31_262144));
        m31 cube_w27_3 = add(add(mul2_9, mul(mul2_10, M31_512)), mul(mul2_11, M31_262144));
        m31 cube_w27_4 = add(add(mul2_12, mul(mul2_13, M31_512)), mul(mul2_14, M31_262144));
        m31 cube_w27_5 = add(add(mul2_15, mul(mul2_16, M31_512)), mul(mul2_17, M31_262144));
        m31 cube_w27_6 = add(add(mul2_18, mul(mul2_19, M31_512)), mul(mul2_20, M31_262144));
        m31 cube_w27_7 = add(add(mul2_21, mul(mul2_22, M31_512)), mul(mul2_23, M31_262144));
        m31 cube_w27_8 = add(add(mul2_24, mul(mul2_25, M31_512)), mul(mul2_26, M31_262144));
        m31 cube_w27_9 = mul2_27;

        m31 vals[20] = {
            input_0, input_1, input_2, input_3, input_4,
            input_5, input_6, input_7, input_8, input_9,
            cube_w27_0, cube_w27_1, cube_w27_2, cube_w27_3, cube_w27_4,
            cube_w27_5, cube_w27_6, cube_w27_7, cube_w27_8, cube_w27_9
        };
        qm31 denom = cube_252_le->combine(vals, 20);
        // Numerator is -enabler
        qm31 numer = qm31{cm31{neg(enabler), 0}, cm31{0, 0}};
        logup_col_write_frac(49 * trace_size + row, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize interaction columns with accumulation
__global__ void cube_252_finalize_interaction_kernel(
    unsigned int trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces   // [200][trace_size] - 50 logup cols × 4 BaseField each
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Running sum of all fractions
    qm31 running_sum = qm31{cm31{0, 0}, cm31{0, 0}};

    // Write each of the 50 logup fractions with ACCUMULATION
    for (int i = 0; i < CUBE_252_N_LOGUP_COLS; ++i) {
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

// Compute cumulative sum (only sum last column)
__global__ void cube_252_cumsum_kernel(
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

    // Only sum the LAST column (index 49 = columns 196-199)
    int last_base_col = (CUBE_252_N_LOGUP_COLS - 1) * 4;
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

// Apply cumsum shift to last column
__global__ void cube_252_apply_cumsum_shift_kernel(
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

    // Only apply shift to the LAST column (196-199)
    int last_base_col = (CUBE_252_N_LOGUP_COLS - 1) * 4;
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

// Host wrapper for interaction trace generation
extern "C" void generate_cube_252_interaction_trace(
    m31** trace_columns,
    unsigned int trace_size,
    void* cube_252_lookup_elements,
    // Order matches SIMD: rc_19 first, then rc_9_9
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
    m31** interaction_trace_columns,
    qm31* claimed_sum
) {
    // Copy trace column pointers to device
    m31** d_trace_columns;
    d_trace_columns = cuda_mem_pool_allocate<m31*>(CUBE_252_N_TRACE_COLUMNS);
    cudaMemcpyAsync(d_trace_columns, trace_columns, CUBE_252_N_TRACE_COLUMNS * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    // Copy interaction trace column pointers to device
    m31** d_interaction_traces;
    d_interaction_traces = cuda_mem_pool_allocate<m31*>(4 * CUBE_252_N_LOGUP_COLS);
    cudaMemcpyAsync(d_interaction_traces, interaction_trace_columns, 4 * CUBE_252_N_LOGUP_COLS * sizeof(m31*), cudaMemcpyHostToDevice, 0);

    unsigned int n_fractions = CUBE_252_N_LOGUP_COLS * trace_size;

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
    LookupElementsBasic<20>* d_cube_252_le;
    LookupElementsBasic<2>* d_rc_9_9_le;
    LookupElementsBasic<2>* d_rc_9_9_b_le;
    LookupElementsBasic<2>* d_rc_9_9_c_le;
    LookupElementsBasic<2>* d_rc_9_9_d_le;
    LookupElementsBasic<2>* d_rc_9_9_e_le;
    LookupElementsBasic<2>* d_rc_9_9_f_le;
    LookupElementsBasic<2>* d_rc_9_9_g_le;
    LookupElementsBasic<2>* d_rc_9_9_h_le;
    LookupElementsBasic<1>* d_rc_19_le;
    LookupElementsBasic<1>* d_rc_19_b_le;
    LookupElementsBasic<1>* d_rc_19_c_le;
    LookupElementsBasic<1>* d_rc_19_d_le;
    LookupElementsBasic<1>* d_rc_19_e_le;
    LookupElementsBasic<1>* d_rc_19_f_le;
    LookupElementsBasic<1>* d_rc_19_g_le;
    LookupElementsBasic<1>* d_rc_19_h_le;

    d_cube_252_le = cuda_mem_pool_allocate<LookupElementsBasic<20>>(1);
    d_rc_9_9_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_9_9_b_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_9_9_c_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_9_9_d_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_9_9_e_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_9_9_f_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_9_9_g_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_9_9_h_le = cuda_mem_pool_allocate<LookupElementsBasic<2>>(1);
    d_rc_19_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);
    d_rc_19_b_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);
    d_rc_19_c_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);
    d_rc_19_d_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);
    d_rc_19_e_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);
    d_rc_19_f_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);
    d_rc_19_g_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);
    d_rc_19_h_le = cuda_mem_pool_allocate<LookupElementsBasic<1>>(1);

    cudaMemcpyAsync(d_cube_252_le, cube_252_lookup_elements, sizeof(LookupElementsBasic<20>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_le, rc_9_9_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_b_le, rc_9_9_b_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_c_le, rc_9_9_c_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_d_le, rc_9_9_d_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_e_le, rc_9_9_e_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_f_le, rc_9_9_f_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_g_le, rc_9_9_g_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_9_9_h_le, rc_9_9_h_lookup_elements, sizeof(LookupElementsBasic<2>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_le, rc_19_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_b_le, rc_19_b_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_c_le, rc_19_c_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_d_le, rc_19_d_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_e_le, rc_19_e_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_f_le, rc_19_f_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_g_le, rc_19_g_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_rc_19_h_le, rc_19_h_lookup_elements, sizeof(LookupElementsBasic<1>), cudaMemcpyHostToDevice, 0);

    unsigned int block_size = CUBE_252_BLOCK_SIZE;
    unsigned int grid_size = (trace_size + block_size - 1) / block_size;

    timer global_timer;
    global_timer.start("generate cube_252 interaction trace");
    // Phase 1: Compute fractions
    cube_252_compute_fractions_kernel<<<grid_size, block_size>>>(
        d_trace_columns,
        trace_size,
        d_cube_252_le,
        d_rc_9_9_le, d_rc_9_9_b_le, d_rc_9_9_c_le, d_rc_9_9_d_le,
        d_rc_9_9_e_le, d_rc_9_9_f_le, d_rc_9_9_g_le, d_rc_9_9_h_le,
        d_rc_19_le, d_rc_19_b_le, d_rc_19_c_le, d_rc_19_d_le,
        d_rc_19_e_le, d_rc_19_f_le, d_rc_19_g_le, d_rc_19_h_le,
        denom_ptr, numerator0, numerator1, numerator2, numerator3
    );

    // Phase 2: Batch inverse on denominators
    batch_inverse_secure_field(denom_ptr, denom_inv, n_fractions);

    // Phase 3: Finalize with accumulation
    cube_252_finalize_interaction_kernel<<<grid_size, block_size>>>(
        trace_size,
        denom_inv,
        numerator0, numerator1, numerator2, numerator3,
        d_interaction_traces
    );

    // Free fraction temporaries — no longer needed after finalize
    cuda_mem_pool_free(denom_ptr);
    cuda_mem_pool_free(denom_inv);
    cuda_mem_pool_free(numerator0);
    cuda_mem_pool_free(numerator1);
    cuda_mem_pool_free(numerator2);
    cuda_mem_pool_free(numerator3);
    // Free lookup elements — no longer needed
    cuda_mem_pool_free(d_cube_252_le);
    cuda_mem_pool_free(d_rc_9_9_le);
    cuda_mem_pool_free(d_rc_9_9_b_le);
    cuda_mem_pool_free(d_rc_9_9_c_le);
    cuda_mem_pool_free(d_rc_9_9_d_le);
    cuda_mem_pool_free(d_rc_9_9_e_le);
    cuda_mem_pool_free(d_rc_9_9_f_le);
    cuda_mem_pool_free(d_rc_9_9_g_le);
    cuda_mem_pool_free(d_rc_9_9_h_le);
    cuda_mem_pool_free(d_rc_19_le);
    cuda_mem_pool_free(d_rc_19_b_le);
    cuda_mem_pool_free(d_rc_19_c_le);
    cuda_mem_pool_free(d_rc_19_d_le);
    cuda_mem_pool_free(d_rc_19_e_le);
    cuda_mem_pool_free(d_rc_19_f_le);
    cuda_mem_pool_free(d_rc_19_g_le);
    cuda_mem_pool_free(d_rc_19_h_le);

    // Phase 4: Compute cumulative sum (for claimed_sum)
    m31* d_coordinate_sums;
    d_coordinate_sums = cuda_mem_pool_allocate<m31>(4);
    cudaMemsetAsync(d_coordinate_sums, 0, 4 * sizeof(m31), 0);

    int cumsum_block_size = 256;
    int cumsum_grid_size = (trace_size + cumsum_block_size - 1) / cumsum_block_size;
    if (cumsum_grid_size > 256) cumsum_grid_size = 256;

    cube_252_cumsum_kernel<<<cumsum_grid_size, cumsum_block_size, 4 * cumsum_block_size * sizeof(m31)>>>(
        trace_size,
        d_interaction_traces,
        d_coordinate_sums
    );

    // Read claimed_sum from device and write to device output pointer
    m31 h_sums[4];
    cudaMemcpyAsync(h_sums, d_coordinate_sums, 4 * sizeof(m31), cudaMemcpyDeviceToHost, 0);
    qm31 h_claimed_sum = qm31{cm31{h_sums[0], h_sums[1]}, cm31{h_sums[2], h_sums[3]}};
    cudaMemcpyAsync(claimed_sum, &h_claimed_sum, sizeof(qm31), cudaMemcpyHostToDevice, 0);

    // Phase 5: Apply cumsum shift to last column
    cube_252_apply_cumsum_shift_kernel<<<grid_size, block_size>>>(
        d_coordinate_sums,
        trace_size,
        d_interaction_traces
    );

    // Phase 6: Inclusive prefix sum on last column
    int last_base_col = (CUBE_252_N_LOGUP_COLS - 1) * 4;
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 0], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 1], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 2], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 3], trace_size);

    global_timer.end("generate cube_252 interaction trace");
    // Cleanup (fraction temps and lookup elements already freed after Phase 3)
    cuda_mem_pool_free(d_trace_columns);
    cuda_mem_pool_free(d_interaction_traces);
    cuda_mem_pool_free(d_coordinate_sums);
}
