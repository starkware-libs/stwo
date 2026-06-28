// CUDA trace generation for mul_mod_builtin component
// mul_mod_builtin handles modular multiplication operations
// 426 trace columns, 94 interaction trace columns

#include "relations.cuh"

#include <cstdint>
#include <cstdio>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include <stdint.h>

#include "gen_mul_mod_builtin_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

// ============================================================================
// Base trace generation kernel for mul_mod_builtin
// 426 trace columns, 29 memory_address_to_id, 24 memory_id_to_big,
// 32 range_check_12, 62 range_check_18, 40 range_check_3_6_6_3 lookups
// ============================================================================

// Helper: Convert 11 limbs (99-bit Felt252) to 12-bit array for multiplication
__device__ void felt252_to_12bit_array(const m31* limbs, int64_t* arr) {
    // Each limb is 9 bits, convert to 12-bit representation
    // 99 bits / 12 bits = 8.25 => 9 12-bit limbs
    // Limbs: limb0(9), limb1(9), ..., limb10(9)
    // 12-bit: arr[0..7]

    // First accumulate the full value in 64-bit chunks then extract 12-bit
    uint64_t accum = 0;
    int accum_bits = 0;
    int arr_idx = 0;

    for (int i = 0; i < 11 && arr_idx < 9; i++) {
        accum |= ((uint64_t)(limbs[i]) << accum_bits);
        accum_bits += 9;
        while (accum_bits >= 12 && arr_idx < 9) {
            arr[arr_idx++] = accum & 0xFFF;
            accum >>= 12;
            accum_bits -= 12;
        }
    }
    if (arr_idx < 9 && accum_bits > 0) {
        arr[arr_idx++] = accum & 0xFFF;
    }
    while (arr_idx < 9) {
        arr[arr_idx++] = 0;
    }
}

// Helper: 8x8 schoolbook multiplication for single Karatsuba
// a, b each have 8 12-bit limbs
// Result has 15 terms (indices 0-14)
__device__ void mul_8x8_schoolbook(const int64_t* a, const int64_t* b, int64_t* result) {
    for (int i = 0; i < 15; i++) result[i] = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            result[i + j] += a[i] * b[j];
        }
    }
}

// Helper: Single Karatsuba N=8 for 16-limb inputs
// a, b each have 16 12-bit limbs (indices 0-15)
// Result has 31 terms (indices 0-30)
__device__ void single_karatsuba_n8(const int64_t* a, const int64_t* b, int64_t* result) {
    // z0 = a_low * b_low (indices 0-7)
    int64_t z0[15];
    mul_8x8_schoolbook(a, b, z0);

    // z2 = a_high * b_high (indices 8-15)
    int64_t z2[15];
    mul_8x8_schoolbook(a + 8, b + 8, z2);

    // z1 = (a_low + a_high) * (b_low + b_high)
    int64_t a_sum[8], b_sum[8];
    for (int i = 0; i < 8; i++) {
        a_sum[i] = a[i] + a[i + 8];
        b_sum[i] = b[i] + b[i + 8];
    }
    int64_t z1[15];
    mul_8x8_schoolbook(a_sum, b_sum, z1);

    // Combine using Karatsuba formula:
    // result[0..7] = z0[0..7]
    // result[8..14] = z0[8..14] + z1[0..6] - z0[0..6] - z2[0..6]
    // result[15] = z1[7] - z0[7] - z2[7]
    // result[16..22] = z2[0..6] + z1[8..14] - z0[8..14] - z2[8..14]
    // result[23..30] = z2[7..14]
    for (int i = 0; i < 8; i++) {
        result[i] = z0[i];
    }
    for (int i = 0; i < 7; i++) {
        result[8 + i] = z0[8 + i] + z1[i] - z0[i] - z2[i];
    }
    result[15] = z1[7] - z0[7] - z2[7];
    for (int i = 0; i < 7; i++) {
        result[16 + i] = z2[i] + z1[8 + i] - z0[8 + i] - z2[8 + i];
    }
    for (int i = 0; i < 8; i++) {
        result[23 + i] = z2[7 + i];
    }
}

// Helper: Double Karatsuba N=8 for 32-limb inputs
// a, b each have 32 12-bit limbs (indices 0-31)
// Result has 63 terms (indices 0-62)
__device__ void double_karatsuba_n8(const int64_t* a, const int64_t* b, int64_t* result) {
    // z0 = a_low * b_low (indices 0-15)
    int64_t z0[31];
    single_karatsuba_n8(a, b, z0);

    // z2 = a_high * b_high (indices 16-31)
    int64_t z2[31];
    single_karatsuba_n8(a + 16, b + 16, z2);

    // z1 = (a_low + a_high) * (b_low + b_high)
    int64_t a_sum[16], b_sum[16];
    for (int i = 0; i < 16; i++) {
        a_sum[i] = a[i] + a[i + 16];
        b_sum[i] = b[i] + b[i + 16];
    }
    int64_t z1[31];
    single_karatsuba_n8(a_sum, b_sum, z1);

    // Combine using double Karatsuba formula:
    // result[0..15] = z0[0..15]
    // result[16..30] = z0[16..30] + z1[0..14] - z0[0..14] - z2[0..14]
    // result[31] = z1[15] - z0[15] - z2[15]
    // result[32..46] = z2[0..14] + z1[16..30] - z0[16..30] - z2[16..30]
    // result[47..62] = z2[15..30]
    for (int i = 0; i < 16; i++) {
        result[i] = z0[i];
    }
    for (int i = 0; i < 15; i++) {
        result[16 + i] = z0[16 + i] + z1[i] - z0[i] - z2[i];
    }
    result[31] = z1[15] - z0[15] - z2[15];
    for (int i = 0; i < 15; i++) {
        result[32 + i] = z2[i] + z1[16 + i] - z0[16 + i] - z2[16 + i];
    }
    for (int i = 0; i < 16; i++) {
        result[47 + i] = z2[15 + i];
    }
}

// Helper: 384-bit multiplication (A * B) - results in 768-bit
// A, B each have 32 12-bit limbs (384 bits)
// Result has 64 12-bit limbs (768 bits)
// Uses schoolbook for normalized result, Karatsuba for constraint-compatible result
__device__ void mul_384x384(const int64_t* a, const int64_t* b, int64_t* result) {
    // Clear result
    for (int i = 0; i < 64; i++) result[i] = 0;

    // Schoolbook multiplication
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            result[i + j] += a[i] * b[j];
        }
    }
}

// Helper: 768-bit subtraction (a - b), returns sign (0 = positive, 1 = negative)
__device__ int sub_768(int64_t* a, const int64_t* b) {
    int64_t borrow = 0;
    for (int i = 0; i < 64; i++) {
        int64_t diff = a[i] - b[i] - borrow;
        if (diff < 0) {
            diff += (1LL << 12);
            borrow = 1;
        } else {
            borrow = 0;
        }
        a[i] = diff;
    }
    return (int)borrow;
}

// Helper: Find the highest non-zero limb position (returns -1 if all zeros)
__device__ int find_highest_nonzero(const int64_t* arr, int len) {
    for (int i = len - 1; i >= 0; i--) {
        if (arr[i] != 0) return i;
    }
    return -1;
}

// Helper: Compare two big integers (a >= b shifted by offset limbs)
// Returns true if a >= b * 2^(12*offset)
__device__ bool compare_ge_shifted(const int64_t* a, int a_len, const int64_t* b, int b_len, int offset) {
    // Check if a >= b shifted by offset positions
    int b_top = find_highest_nonzero(b, b_len);
    if (b_top < 0) return true; // b is zero, a >= 0 always true

    int a_top = find_highest_nonzero(a, a_len);
    if (a_top < b_top + offset) return false;
    if (a_top > b_top + offset) return true;

    // Same top position, compare limb by limb
    for (int i = a_top; i >= offset; i--) {
        int64_t a_val = a[i];
        int64_t b_val = (i - offset >= 0 && i - offset < b_len) ? b[i - offset] : 0;
        if (a_val > b_val) return true;
        if (a_val < b_val) return false;
    }
    // Check remaining a limbs below offset should all be >= 0
    return true;
}

// Helper: Subtract b * q shifted by offset from a, in place
// a = a - b * q * 2^(12*offset)
__device__ void subtract_scaled_shifted(int64_t* a, int a_len, const int64_t* b, int b_len, int64_t q, int offset) {
    int64_t borrow = 0;
    for (int i = 0; i < b_len; i++) {
        int idx = i + offset;
        if (idx >= a_len) break;

        int64_t sub = b[i] * q + borrow;
        int64_t lo = sub & 0xFFF;
        borrow = sub >> 12;

        if (a[idx] >= lo) {
            a[idx] -= lo;
        } else {
            a[idx] = a[idx] + (1LL << 12) - lo;
            borrow++;
        }
    }
    // Propagate remaining borrow
    for (int idx = b_len + offset; borrow > 0 && idx < a_len; idx++) {
        if (a[idx] >= borrow) {
            a[idx] -= borrow;
            borrow = 0;
        } else {
            int64_t old = a[idx];
            a[idx] = a[idx] + (1LL << 12) - borrow;
            borrow = (borrow - old + 0xFFF) >> 12;
        }
    }
}

// Helper: 768-bit division by 384-bit (a / b), result is 384-bit quotient
// Assumes b divides a evenly
__device__ void div_768_by_384(int64_t* a, const int64_t* b, int64_t* quotient) {
    // Clear quotient
    for (int i = 0; i < 32; i++) quotient[i] = 0;

    // Find highest non-zero limb of divisor b
    int b_top = find_highest_nonzero(b, 32);
    if (b_top < 0) return; // Division by zero guard

    // Copy a to working remainder
    int64_t remainder[64];
    for (int i = 0; i < 64; i++) remainder[i] = a[i];

    // Division: compute quotient digits from high to low
    // Quotient has at most 32 limbs (384 bits)
    for (int q_pos = 31; q_pos >= 0; q_pos--) {
        // We need: remainder / (b * 2^(12*q_pos))
        // This quotient digit should fit in 12 bits (0 to 4095)

        // Get the relevant portion of remainder for estimation
        // Look at remainder[q_pos + b_top] and remainder[q_pos + b_top + 1]
        int r_idx = q_pos + b_top;
        if (r_idx >= 64) continue;

        // Form a 24-bit value from top two limbs of relevant portion
        int64_t r_hi = (r_idx + 1 < 64) ? remainder[r_idx + 1] : 0;
        int64_t r_lo = remainder[r_idx];
        int64_t r_combined = (r_hi << 12) | r_lo;

        // Estimate: q_est = r_combined / b[b_top]
        // But we need to be careful about underestimation
        int64_t q_est = 0;
        if (b[b_top] > 0) {
            q_est = r_combined / b[b_top];
            // Clamp to 12 bits
            if (q_est > 0xFFF) q_est = 0xFFF;
        }

        // Refine: try q_est, if remainder becomes negative, decrease
        // Use binary search style refinement
        while (q_est > 0) {
            // Check if q_est * b * 2^(12*q_pos) <= remainder
            // We'll do this by subtracting and checking sign

            // Temporarily compute the subtraction
            int64_t temp_rem[64];
            for (int i = 0; i < 64; i++) temp_rem[i] = remainder[i];

            int64_t borrow = 0;
            for (int i = 0; i < 32; i++) {
                int idx = i + q_pos;
                if (idx >= 64) break;

                int64_t sub = b[i] * q_est + borrow;
                int64_t lo = sub & 0xFFF;
                borrow = sub >> 12;

                if (temp_rem[idx] >= lo) {
                    temp_rem[idx] -= lo;
                } else {
                    temp_rem[idx] = temp_rem[idx] + (1LL << 12) - lo;
                    borrow++;
                }
            }
            // Propagate borrow
            for (int idx = 32 + q_pos; borrow > 0 && idx < 64; idx++) {
                if (temp_rem[idx] >= borrow) {
                    temp_rem[idx] -= borrow;
                    borrow = 0;
                } else {
                    int64_t old = temp_rem[idx];
                    temp_rem[idx] = temp_rem[idx] + (1LL << 12) - borrow;
                    borrow = 1;
                }
            }

            if (borrow == 0) {
                // q_est is valid, use it
                for (int i = 0; i < 64; i++) remainder[i] = temp_rem[i];
                quotient[q_pos] = q_est;
                break;
            } else {
                // q_est too large, decrease
                q_est--;
            }
        }
    }
}

// Helper: Normalize carries in a limb array (12-bit limbs)
__device__ void normalize_12bit(int64_t* arr, int len) {
    for (int i = 0; i < len - 1; i++) {
        if (arr[i] >= (1LL << 12)) {
            arr[i + 1] += arr[i] >> 12;
            arr[i] &= 0xFFF;
        } else if (arr[i] < 0) {
            int64_t borrow = (-arr[i] + 0xFFF) >> 12;
            arr[i] += borrow << 12;
            arr[i + 1] -= borrow;
        }
    }
}

__launch_bounds__(MUL_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 1)
__global__ void generate_mul_mod_builtin_trace_kernel(
    m31 **traces,

    // Lookup data arrays - 29 MemoryAddressToId
    m31 **lookup_memory_address_to_id_0, m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2, m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4, m31 **lookup_memory_address_to_id_5,
    m31 **lookup_memory_address_to_id_6, m31 **lookup_memory_address_to_id_7,
    m31 **lookup_memory_address_to_id_8, m31 **lookup_memory_address_to_id_9,
    m31 **lookup_memory_address_to_id_10, m31 **lookup_memory_address_to_id_11,
    m31 **lookup_memory_address_to_id_12, m31 **lookup_memory_address_to_id_13,
    m31 **lookup_memory_address_to_id_14, m31 **lookup_memory_address_to_id_15,
    m31 **lookup_memory_address_to_id_16, m31 **lookup_memory_address_to_id_17,
    m31 **lookup_memory_address_to_id_18, m31 **lookup_memory_address_to_id_19,
    m31 **lookup_memory_address_to_id_20, m31 **lookup_memory_address_to_id_21,
    m31 **lookup_memory_address_to_id_22, m31 **lookup_memory_address_to_id_23,
    m31 **lookup_memory_address_to_id_24, m31 **lookup_memory_address_to_id_25,
    m31 **lookup_memory_address_to_id_26, m31 **lookup_memory_address_to_id_27,
    m31 **lookup_memory_address_to_id_28,

    // Lookup data arrays - 24 MemoryIdToBig
    m31 **lookup_memory_id_to_big_0, m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2, m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4, m31 **lookup_memory_id_to_big_5,
    m31 **lookup_memory_id_to_big_6, m31 **lookup_memory_id_to_big_7,
    m31 **lookup_memory_id_to_big_8, m31 **lookup_memory_id_to_big_9,
    m31 **lookup_memory_id_to_big_10, m31 **lookup_memory_id_to_big_11,
    m31 **lookup_memory_id_to_big_12, m31 **lookup_memory_id_to_big_13,
    m31 **lookup_memory_id_to_big_14, m31 **lookup_memory_id_to_big_15,
    m31 **lookup_memory_id_to_big_16, m31 **lookup_memory_id_to_big_17,
    m31 **lookup_memory_id_to_big_18, m31 **lookup_memory_id_to_big_19,
    m31 **lookup_memory_id_to_big_20, m31 **lookup_memory_id_to_big_21,
    m31 **lookup_memory_id_to_big_22, m31 **lookup_memory_id_to_big_23,

    // Lookup data arrays - 32 RangeCheck_12
    m31 **lookup_range_check_12_0, m31 **lookup_range_check_12_1,
    m31 **lookup_range_check_12_2, m31 **lookup_range_check_12_3,
    m31 **lookup_range_check_12_4, m31 **lookup_range_check_12_5,
    m31 **lookup_range_check_12_6, m31 **lookup_range_check_12_7,
    m31 **lookup_range_check_12_8, m31 **lookup_range_check_12_9,
    m31 **lookup_range_check_12_10, m31 **lookup_range_check_12_11,
    m31 **lookup_range_check_12_12, m31 **lookup_range_check_12_13,
    m31 **lookup_range_check_12_14, m31 **lookup_range_check_12_15,
    m31 **lookup_range_check_12_16, m31 **lookup_range_check_12_17,
    m31 **lookup_range_check_12_18, m31 **lookup_range_check_12_19,
    m31 **lookup_range_check_12_20, m31 **lookup_range_check_12_21,
    m31 **lookup_range_check_12_22, m31 **lookup_range_check_12_23,
    m31 **lookup_range_check_12_24, m31 **lookup_range_check_12_25,
    m31 **lookup_range_check_12_26, m31 **lookup_range_check_12_27,
    m31 **lookup_range_check_12_28, m31 **lookup_range_check_12_29,
    m31 **lookup_range_check_12_30, m31 **lookup_range_check_12_31,

    // Lookup data arrays - 62 RangeCheck_18
    m31 **lookup_range_check_18_0, m31 **lookup_range_check_18_1,
    m31 **lookup_range_check_18_2, m31 **lookup_range_check_18_3,
    m31 **lookup_range_check_18_4, m31 **lookup_range_check_18_5,
    m31 **lookup_range_check_18_6, m31 **lookup_range_check_18_7,
    m31 **lookup_range_check_18_8, m31 **lookup_range_check_18_9,
    m31 **lookup_range_check_18_10, m31 **lookup_range_check_18_11,
    m31 **lookup_range_check_18_12, m31 **lookup_range_check_18_13,
    m31 **lookup_range_check_18_14, m31 **lookup_range_check_18_15,
    m31 **lookup_range_check_18_16, m31 **lookup_range_check_18_17,
    m31 **lookup_range_check_18_18, m31 **lookup_range_check_18_19,
    m31 **lookup_range_check_18_20, m31 **lookup_range_check_18_21,
    m31 **lookup_range_check_18_22, m31 **lookup_range_check_18_23,
    m31 **lookup_range_check_18_24, m31 **lookup_range_check_18_25,
    m31 **lookup_range_check_18_26, m31 **lookup_range_check_18_27,
    m31 **lookup_range_check_18_28, m31 **lookup_range_check_18_29,
    m31 **lookup_range_check_18_30, m31 **lookup_range_check_18_31,
    m31 **lookup_range_check_18_32, m31 **lookup_range_check_18_33,
    m31 **lookup_range_check_18_34, m31 **lookup_range_check_18_35,
    m31 **lookup_range_check_18_36, m31 **lookup_range_check_18_37,
    m31 **lookup_range_check_18_38, m31 **lookup_range_check_18_39,
    m31 **lookup_range_check_18_40, m31 **lookup_range_check_18_41,
    m31 **lookup_range_check_18_42, m31 **lookup_range_check_18_43,
    m31 **lookup_range_check_18_44, m31 **lookup_range_check_18_45,
    m31 **lookup_range_check_18_46, m31 **lookup_range_check_18_47,
    m31 **lookup_range_check_18_48, m31 **lookup_range_check_18_49,
    m31 **lookup_range_check_18_50, m31 **lookup_range_check_18_51,
    m31 **lookup_range_check_18_52, m31 **lookup_range_check_18_53,
    m31 **lookup_range_check_18_54, m31 **lookup_range_check_18_55,
    m31 **lookup_range_check_18_56, m31 **lookup_range_check_18_57,
    m31 **lookup_range_check_18_58, m31 **lookup_range_check_18_59,
    m31 **lookup_range_check_18_60, m31 **lookup_range_check_18_61,

    // Lookup data arrays - 40 RangeCheck_3_6_6_3
    m31 **lookup_range_check_3_6_6_3_0, m31 **lookup_range_check_3_6_6_3_1,
    m31 **lookup_range_check_3_6_6_3_2, m31 **lookup_range_check_3_6_6_3_3,
    m31 **lookup_range_check_3_6_6_3_4, m31 **lookup_range_check_3_6_6_3_5,
    m31 **lookup_range_check_3_6_6_3_6, m31 **lookup_range_check_3_6_6_3_7,
    m31 **lookup_range_check_3_6_6_3_8, m31 **lookup_range_check_3_6_6_3_9,
    m31 **lookup_range_check_3_6_6_3_10, m31 **lookup_range_check_3_6_6_3_11,
    m31 **lookup_range_check_3_6_6_3_12, m31 **lookup_range_check_3_6_6_3_13,
    m31 **lookup_range_check_3_6_6_3_14, m31 **lookup_range_check_3_6_6_3_15,
    m31 **lookup_range_check_3_6_6_3_16, m31 **lookup_range_check_3_6_6_3_17,
    m31 **lookup_range_check_3_6_6_3_18, m31 **lookup_range_check_3_6_6_3_19,
    m31 **lookup_range_check_3_6_6_3_20, m31 **lookup_range_check_3_6_6_3_21,
    m31 **lookup_range_check_3_6_6_3_22, m31 **lookup_range_check_3_6_6_3_23,
    m31 **lookup_range_check_3_6_6_3_24, m31 **lookup_range_check_3_6_6_3_25,
    m31 **lookup_range_check_3_6_6_3_26, m31 **lookup_range_check_3_6_6_3_27,
    m31 **lookup_range_check_3_6_6_3_28, m31 **lookup_range_check_3_6_6_3_29,
    m31 **lookup_range_check_3_6_6_3_30, m31 **lookup_range_check_3_6_6_3_31,
    m31 **lookup_range_check_3_6_6_3_32, m31 **lookup_range_check_3_6_6_3_33,
    m31 **lookup_range_check_3_6_6_3_34, m31 **lookup_range_check_3_6_6_3_35,
    m31 **lookup_range_check_3_6_6_3_36, m31 **lookup_range_check_3_6_6_3_37,
    m31 **lookup_range_check_3_6_6_3_38, m31 **lookup_range_check_3_6_6_3_39,

    // Sub-component inputs
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_range_check_12,
    m31 **sub_component_inputs_range_check_18,
    m31 **sub_component_inputs_range_check_3_6_6_3,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    // Constants
    const m31 M31_0         = {0};
    const m31 M31_1         = {1};
    const m31 M31_2         = {2};
    const m31 M31_3         = {3};
    const m31 M31_4         = {4};
    const m31 M31_5         = {5};
    const m31 M31_6         = {6};
    const m31 M31_7         = {7};
    const m31 M31_8         = {8};
    const m31 M31_64        = {64};
    const m31 M31_128       = {128};
    const m31 M31_131072    = {131072};
    const m31 M31_134217728 = {134217728};
    const m31 M31_136       = {136};
    const m31 M31_256       = {256};
    const m31 M31_262144    = {262144};
    const m31 M31_508       = {508};
    const m31 M31_511       = {511};
    const m31 M31_512       = {512};
    const m31 M31_524288    = {524288};
    const m31 M31_536870912 = {536870912};

    const uint16_t UInt16_1 = 1;
    const uint16_t UInt16_2 = 2;
    const uint16_t UInt16_3 = 3;
    const uint16_t UInt16_6 = 6;

    if (row < trace_size) {
        // seq = row index
        m31 seq = {row};

        // is_instance_0 = (seq == 0)
        m31 is_instance_0_col0 = (row == 0) ? M31_1 : M31_0;
        traces[0][row] = is_instance_0_col0;

        // Instance addressing
        m31 segment_start_m31 = {segment_start};
        m31 prev_seq = (row == 0) ? M31_0 : sub(seq, M31_1);
        m31 prev_instance_addr = add(segment_start_m31, mul(M31_7, prev_seq));
        m31 instance_addr = add(segment_start_m31, mul(M31_7, seq));

        // ============ Read P0 (instance_addr + 0) ============
        m31 p0_id_col1 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, instance_addr, &p0_id_col1);
        traces[1][row] = p0_id_col1;
        sub_component_inputs_memory_address_to_id[0][row] = instance_addr;
        lookup_memory_address_to_id_0[0][row] = instance_addr;
        lookup_memory_address_to_id_0[1][row] = p0_id_col1;

        m31 p0_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, p0_id_col1, p0_value);

        // P0 limbs (columns 2-12)
        m31 p0_limbs[11];
        for (int i = 0; i < 11; i++) {
            p0_limbs[i] = p0_value[i];
            traces[2 + i][row] = p0_limbs[i];
        }

        sub_component_inputs_memory_id_to_big[0][row] = p0_id_col1;
        lookup_memory_id_to_big_0[0][row] = p0_id_col1;
        for (int i = 0; i < 11; i++) lookup_memory_id_to_big_0[1 + i][row] = p0_limbs[i];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_0[i][row] = M31_0;

        // ============ Read P1 (instance_addr + 1) ============
        m31 addr_p1 = add(instance_addr, M31_1);
        m31 p1_id_col13 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_p1, &p1_id_col13);
        traces[13][row] = p1_id_col13;
        sub_component_inputs_memory_address_to_id[1][row] = addr_p1;
        lookup_memory_address_to_id_1[0][row] = addr_p1;
        lookup_memory_address_to_id_1[1][row] = p1_id_col13;

        m31 p1_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, p1_id_col13, p1_value);

        m31 p1_limbs[11];
        for (int i = 0; i < 11; i++) {
            p1_limbs[i] = p1_value[i];
            traces[14 + i][row] = p1_limbs[i];
        }

        sub_component_inputs_memory_id_to_big[1][row] = p1_id_col13;
        lookup_memory_id_to_big_1[0][row] = p1_id_col13;
        for (int i = 0; i < 11; i++) lookup_memory_id_to_big_1[1 + i][row] = p1_limbs[i];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_1[i][row] = M31_0;

        // ============ Read P2 (instance_addr + 2) ============
        m31 addr_p2 = add(instance_addr, M31_2);
        m31 p2_id_col25 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_p2, &p2_id_col25);
        traces[25][row] = p2_id_col25;
        sub_component_inputs_memory_address_to_id[2][row] = addr_p2;
        lookup_memory_address_to_id_2[0][row] = addr_p2;
        lookup_memory_address_to_id_2[1][row] = p2_id_col25;

        m31 p2_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, p2_id_col25, p2_value);

        m31 p2_limbs[11];
        for (int i = 0; i < 11; i++) {
            p2_limbs[i] = p2_value[i];
            traces[26 + i][row] = p2_limbs[i];
        }

        sub_component_inputs_memory_id_to_big[2][row] = p2_id_col25;
        lookup_memory_id_to_big_2[0][row] = p2_id_col25;
        for (int i = 0; i < 11; i++) lookup_memory_id_to_big_2[1 + i][row] = p2_limbs[i];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_2[i][row] = M31_0;

        // ============ Read P3 (instance_addr + 3) ============
        m31 addr_p3 = add(instance_addr, M31_3);
        m31 p3_id_col37 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_p3, &p3_id_col37);
        traces[37][row] = p3_id_col37;
        sub_component_inputs_memory_address_to_id[3][row] = addr_p3;
        lookup_memory_address_to_id_3[0][row] = addr_p3;
        lookup_memory_address_to_id_3[1][row] = p3_id_col37;

        m31 p3_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, p3_id_col37, p3_value);

        m31 p3_limbs[11];
        for (int i = 0; i < 11; i++) {
            p3_limbs[i] = p3_value[i];
            traces[38 + i][row] = p3_limbs[i];
        }

        sub_component_inputs_memory_id_to_big[3][row] = p3_id_col37;
        lookup_memory_id_to_big_3[0][row] = p3_id_col37;
        for (int i = 0; i < 11; i++) lookup_memory_id_to_big_3[1 + i][row] = p3_limbs[i];
        for (int i = 12; i < 29; i++) lookup_memory_id_to_big_3[i][row] = M31_0;

        // ============ Read values_ptr (instance_addr + 4) ============
        m31 addr_values_ptr = add(instance_addr, M31_4);
        m31 values_ptr_id_col49 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_values_ptr, &values_ptr_id_col49);
        traces[49][row] = values_ptr_id_col49;
        sub_component_inputs_memory_address_to_id[4][row] = addr_values_ptr;
        lookup_memory_address_to_id_4[0][row] = addr_values_ptr;
        lookup_memory_address_to_id_4[1][row] = values_ptr_id_col49;

        m31 values_ptr_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, values_ptr_id_col49, values_ptr_value);

        m31 values_ptr_limb_0_col50 = values_ptr_value[0]; traces[50][row] = values_ptr_limb_0_col50;
        m31 values_ptr_limb_1_col51 = values_ptr_value[1]; traces[51][row] = values_ptr_limb_1_col51;
        m31 values_ptr_limb_2_col52 = values_ptr_value[2]; traces[52][row] = values_ptr_limb_2_col52;
        m31 values_ptr_limb_3_col53 = values_ptr_value[3]; traces[53][row] = values_ptr_limb_3_col53;

        uint16_t partial_limb_msb_tmp_0 = (((uint16_t)(values_ptr_limb_3_col53)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col54 = (m31)partial_limb_msb_tmp_0;
        traces[54][row] = partial_limb_msb_col54;

        sub_component_inputs_memory_id_to_big[4][row] = values_ptr_id_col49;
        lookup_memory_id_to_big_4[0][row] = values_ptr_id_col49;
        lookup_memory_id_to_big_4[1][row] = values_ptr_limb_0_col50;
        lookup_memory_id_to_big_4[2][row] = values_ptr_limb_1_col51;
        lookup_memory_id_to_big_4[3][row] = values_ptr_limb_2_col52;
        lookup_memory_id_to_big_4[4][row] = values_ptr_limb_3_col53;
        for (int i = 5; i < 29; i++) lookup_memory_id_to_big_4[i][row] = M31_0;

        // Compute values_ptr combined
        m31 values_ptr = add(add(add(values_ptr_limb_0_col50, mul(values_ptr_limb_1_col51, M31_512)),
                                mul(values_ptr_limb_2_col52, M31_262144)),
                            mul(values_ptr_limb_3_col53, M31_134217728));

        // ============ Read offsets_ptr (instance_addr + 5) ============
        m31 addr_offsets_ptr = add(instance_addr, M31_5);
        m31 offsets_ptr_id_col55 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_offsets_ptr, &offsets_ptr_id_col55);
        traces[55][row] = offsets_ptr_id_col55;
        sub_component_inputs_memory_address_to_id[5][row] = addr_offsets_ptr;
        lookup_memory_address_to_id_5[0][row] = addr_offsets_ptr;
        lookup_memory_address_to_id_5[1][row] = offsets_ptr_id_col55;

        m31 offsets_ptr_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, offsets_ptr_id_col55, offsets_ptr_value);

        m31 offsets_ptr_limb_0_col56 = offsets_ptr_value[0]; traces[56][row] = offsets_ptr_limb_0_col56;
        m31 offsets_ptr_limb_1_col57 = offsets_ptr_value[1]; traces[57][row] = offsets_ptr_limb_1_col57;
        m31 offsets_ptr_limb_2_col58 = offsets_ptr_value[2]; traces[58][row] = offsets_ptr_limb_2_col58;
        m31 offsets_ptr_limb_3_col59 = offsets_ptr_value[3]; traces[59][row] = offsets_ptr_limb_3_col59;

        uint16_t partial_limb_msb_tmp_1 = (((uint16_t)(offsets_ptr_limb_3_col59)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col60 = (m31)partial_limb_msb_tmp_1;
        traces[60][row] = partial_limb_msb_col60;

        sub_component_inputs_memory_id_to_big[5][row] = offsets_ptr_id_col55;
        lookup_memory_id_to_big_5[0][row] = offsets_ptr_id_col55;
        lookup_memory_id_to_big_5[1][row] = offsets_ptr_limb_0_col56;
        lookup_memory_id_to_big_5[2][row] = offsets_ptr_limb_1_col57;
        lookup_memory_id_to_big_5[3][row] = offsets_ptr_limb_2_col58;
        lookup_memory_id_to_big_5[4][row] = offsets_ptr_limb_3_col59;
        for (int i = 5; i < 29; i++) lookup_memory_id_to_big_5[i][row] = M31_0;

        m31 offsets_ptr = add(add(add(offsets_ptr_limb_0_col56, mul(offsets_ptr_limb_1_col57, M31_512)),
                                 mul(offsets_ptr_limb_2_col58, M31_262144)),
                             mul(offsets_ptr_limb_3_col59, M31_134217728));

        // ============ Read offsets_ptr_prev (prev_instance_addr + 5) ============
        m31 addr_offsets_ptr_prev = add(prev_instance_addr, M31_5);
        m31 offsets_ptr_prev_id_col61 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_offsets_ptr_prev, &offsets_ptr_prev_id_col61);
        traces[61][row] = offsets_ptr_prev_id_col61;
        sub_component_inputs_memory_address_to_id[6][row] = addr_offsets_ptr_prev;
        lookup_memory_address_to_id_6[0][row] = addr_offsets_ptr_prev;
        lookup_memory_address_to_id_6[1][row] = offsets_ptr_prev_id_col61;

        m31 offsets_ptr_prev_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, offsets_ptr_prev_id_col61, offsets_ptr_prev_value);

        m31 offsets_ptr_prev_limb_0_col62 = offsets_ptr_prev_value[0]; traces[62][row] = offsets_ptr_prev_limb_0_col62;
        m31 offsets_ptr_prev_limb_1_col63 = offsets_ptr_prev_value[1]; traces[63][row] = offsets_ptr_prev_limb_1_col63;
        m31 offsets_ptr_prev_limb_2_col64 = offsets_ptr_prev_value[2]; traces[64][row] = offsets_ptr_prev_limb_2_col64;
        m31 offsets_ptr_prev_limb_3_col65 = offsets_ptr_prev_value[3]; traces[65][row] = offsets_ptr_prev_limb_3_col65;

        uint16_t partial_limb_msb_tmp_2 = (((uint16_t)(offsets_ptr_prev_limb_3_col65)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col66 = (m31)partial_limb_msb_tmp_2;
        traces[66][row] = partial_limb_msb_col66;

        sub_component_inputs_memory_id_to_big[6][row] = offsets_ptr_prev_id_col61;
        lookup_memory_id_to_big_6[0][row] = offsets_ptr_prev_id_col61;
        lookup_memory_id_to_big_6[1][row] = offsets_ptr_prev_limb_0_col62;
        lookup_memory_id_to_big_6[2][row] = offsets_ptr_prev_limb_1_col63;
        lookup_memory_id_to_big_6[3][row] = offsets_ptr_prev_limb_2_col64;
        lookup_memory_id_to_big_6[4][row] = offsets_ptr_prev_limb_3_col65;
        for (int i = 5; i < 29; i++) lookup_memory_id_to_big_6[i][row] = M31_0;

        // ============ Read n (instance_addr + 6) ============
        m31 addr_n = add(instance_addr, M31_6);
        m31 n_id_col67 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_n, &n_id_col67);
        traces[67][row] = n_id_col67;
        sub_component_inputs_memory_address_to_id[7][row] = addr_n;
        lookup_memory_address_to_id_7[0][row] = addr_n;
        lookup_memory_address_to_id_7[1][row] = n_id_col67;

        m31 n_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, n_id_col67, n_value);

        m31 n_limb_0_col68 = n_value[0]; traces[68][row] = n_limb_0_col68;
        m31 n_limb_1_col69 = n_value[1]; traces[69][row] = n_limb_1_col69;
        m31 n_limb_2_col70 = n_value[2]; traces[70][row] = n_limb_2_col70;
        m31 n_limb_3_col71 = n_value[3]; traces[71][row] = n_limb_3_col71;

        uint16_t partial_limb_msb_tmp_3 = (((uint16_t)(n_limb_3_col71)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col72 = (m31)partial_limb_msb_tmp_3;
        traces[72][row] = partial_limb_msb_col72;

        sub_component_inputs_memory_id_to_big[7][row] = n_id_col67;
        lookup_memory_id_to_big_7[0][row] = n_id_col67;
        lookup_memory_id_to_big_7[1][row] = n_limb_0_col68;
        lookup_memory_id_to_big_7[2][row] = n_limb_1_col69;
        lookup_memory_id_to_big_7[3][row] = n_limb_2_col70;
        lookup_memory_id_to_big_7[4][row] = n_limb_3_col71;
        for (int i = 5; i < 29; i++) lookup_memory_id_to_big_7[i][row] = M31_0;

        // ============ Read n_prev (prev_instance_addr + 6) ============
        m31 addr_n_prev = add(prev_instance_addr, M31_6);
        m31 n_prev_id_col73 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_n_prev, &n_prev_id_col73);
        traces[73][row] = n_prev_id_col73;
        sub_component_inputs_memory_address_to_id[8][row] = addr_n_prev;
        lookup_memory_address_to_id_8[0][row] = addr_n_prev;
        lookup_memory_address_to_id_8[1][row] = n_prev_id_col73;

        m31 n_prev_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, n_prev_id_col73, n_prev_value);

        m31 n_prev_limb_0_col74 = n_prev_value[0]; traces[74][row] = n_prev_limb_0_col74;
        m31 n_prev_limb_1_col75 = n_prev_value[1]; traces[75][row] = n_prev_limb_1_col75;
        m31 n_prev_limb_2_col76 = n_prev_value[2]; traces[76][row] = n_prev_limb_2_col76;
        m31 n_prev_limb_3_col77 = n_prev_value[3]; traces[77][row] = n_prev_limb_3_col77;

        uint16_t partial_limb_msb_tmp_4 = (((uint16_t)(n_prev_limb_3_col77)) & UInt16_2) >> UInt16_1;
        m31 partial_limb_msb_col78 = (m31)partial_limb_msb_tmp_4;
        traces[78][row] = partial_limb_msb_col78;

        sub_component_inputs_memory_id_to_big[8][row] = n_prev_id_col73;
        lookup_memory_id_to_big_8[0][row] = n_prev_id_col73;
        lookup_memory_id_to_big_8[1][row] = n_prev_limb_0_col74;
        lookup_memory_id_to_big_8[2][row] = n_prev_limb_1_col75;
        lookup_memory_id_to_big_8[3][row] = n_prev_limb_2_col76;
        lookup_memory_id_to_big_8[4][row] = n_prev_limb_3_col77;
        for (int i = 5; i < 29; i++) lookup_memory_id_to_big_8[i][row] = M31_0;

        // ============ Read prev IDs (columns 79-83) ============
        m31 values_ptr_prev_id_col79 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, add(prev_instance_addr, M31_4), &values_ptr_prev_id_col79);
        traces[79][row] = values_ptr_prev_id_col79;
        sub_component_inputs_memory_address_to_id[9][row] = add(prev_instance_addr, M31_4);
        lookup_memory_address_to_id_9[0][row] = add(prev_instance_addr, M31_4);
        lookup_memory_address_to_id_9[1][row] = values_ptr_prev_id_col79;

        m31 p_prev0_id_col80 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, prev_instance_addr, &p_prev0_id_col80);
        traces[80][row] = p_prev0_id_col80;
        sub_component_inputs_memory_address_to_id[10][row] = prev_instance_addr;
        lookup_memory_address_to_id_10[0][row] = prev_instance_addr;
        lookup_memory_address_to_id_10[1][row] = p_prev0_id_col80;

        m31 p_prev1_id_col81 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, add(prev_instance_addr, M31_1), &p_prev1_id_col81);
        traces[81][row] = p_prev1_id_col81;
        sub_component_inputs_memory_address_to_id[11][row] = add(prev_instance_addr, M31_1);
        lookup_memory_address_to_id_11[0][row] = add(prev_instance_addr, M31_1);
        lookup_memory_address_to_id_11[1][row] = p_prev1_id_col81;

        m31 p_prev2_id_col82 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, add(prev_instance_addr, M31_2), &p_prev2_id_col82);
        traces[82][row] = p_prev2_id_col82;
        sub_component_inputs_memory_address_to_id[12][row] = add(prev_instance_addr, M31_2);
        lookup_memory_address_to_id_12[0][row] = add(prev_instance_addr, M31_2);
        lookup_memory_address_to_id_12[1][row] = p_prev2_id_col82;

        m31 p_prev3_id_col83 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, add(prev_instance_addr, M31_3), &p_prev3_id_col83);
        traces[83][row] = p_prev3_id_col83;
        sub_component_inputs_memory_address_to_id[13][row] = add(prev_instance_addr, M31_3);
        lookup_memory_address_to_id_13[0][row] = add(prev_instance_addr, M31_3);
        lookup_memory_address_to_id_13[1][row] = p_prev3_id_col83;

        // ============ Decode offsets_a, offsets_b, offsets_c (columns 84-107) ============
        // Read offsets_a
        m31 offsets_a_id_col84 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, offsets_ptr, &offsets_a_id_col84);
        traces[84][row] = offsets_a_id_col84;
        sub_component_inputs_memory_address_to_id[14][row] = offsets_ptr;
        lookup_memory_address_to_id_14[0][row] = offsets_ptr;
        lookup_memory_address_to_id_14[1][row] = offsets_a_id_col84;

        m31 offsets_a_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, offsets_a_id_col84, offsets_a_value);

        m31 msb_col85 = (offsets_a_value[27] == M31_256) ? M31_1 : M31_0;
        traces[85][row] = msb_col85;
        m31 mid_limbs_set_col86 = (offsets_a_value[20] == M31_511) ? M31_1 : M31_0;
        traces[86][row] = mid_limbs_set_col86;

        m31 offsets_a_limb_0_col87 = offsets_a_value[0]; traces[87][row] = offsets_a_limb_0_col87;
        m31 offsets_a_limb_1_col88 = offsets_a_value[1]; traces[88][row] = offsets_a_limb_1_col88;
        m31 offsets_a_limb_2_col89 = offsets_a_value[2]; traces[89][row] = offsets_a_limb_2_col89;
        m31 remainder_bits_col90 = (m31)(((uint16_t)offsets_a_value[3]) & UInt16_3);
        traces[90][row] = remainder_bits_col90;
        m31 partial_limb_msb_col91 = (m31)((((uint16_t)remainder_bits_col90) & UInt16_2) >> UInt16_1);
        traces[91][row] = partial_limb_msb_col91;

        sub_component_inputs_memory_id_to_big[9][row] = offsets_a_id_col84;
        lookup_memory_id_to_big_9[0][row] = offsets_a_id_col84;
        lookup_memory_id_to_big_9[1][row] = offsets_a_limb_0_col87;
        lookup_memory_id_to_big_9[2][row] = offsets_a_limb_1_col88;
        lookup_memory_id_to_big_9[3][row] = offsets_a_limb_2_col89;
        lookup_memory_id_to_big_9[4][row] = add(remainder_bits_col90, mul(mid_limbs_set_col86, M31_508));
        for (int i = 5; i < 21; i++) lookup_memory_id_to_big_9[i][row] = mul(mid_limbs_set_col86, M31_511);
        lookup_memory_id_to_big_9[21][row] = sub(mul(M31_136, msb_col85), mid_limbs_set_col86);
        for (int i = 22; i < 28; i++) lookup_memory_id_to_big_9[i][row] = M31_0;
        lookup_memory_id_to_big_9[28][row] = mul(msb_col85, M31_256);

        m31 offset_a = sub(sub(add(add(add(offsets_a_limb_0_col87, mul(offsets_a_limb_1_col88, M31_512)),
                                       mul(offsets_a_limb_2_col89, M31_262144)),
                                   mul(remainder_bits_col90, M31_134217728)),
                               msb_col85),
                           mul(M31_536870912, mid_limbs_set_col86));

        // Read offsets_b
        m31 addr_offsets_b = add(offsets_ptr, M31_1);
        m31 offsets_b_id_col92 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_offsets_b, &offsets_b_id_col92);
        traces[92][row] = offsets_b_id_col92;
        sub_component_inputs_memory_address_to_id[15][row] = addr_offsets_b;
        lookup_memory_address_to_id_15[0][row] = addr_offsets_b;
        lookup_memory_address_to_id_15[1][row] = offsets_b_id_col92;

        m31 offsets_b_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, offsets_b_id_col92, offsets_b_value);

        m31 msb_col93 = (offsets_b_value[27] == M31_256) ? M31_1 : M31_0;
        traces[93][row] = msb_col93;
        m31 mid_limbs_set_col94 = (offsets_b_value[20] == M31_511) ? M31_1 : M31_0;
        traces[94][row] = mid_limbs_set_col94;

        m31 offsets_b_limb_0_col95 = offsets_b_value[0]; traces[95][row] = offsets_b_limb_0_col95;
        m31 offsets_b_limb_1_col96 = offsets_b_value[1]; traces[96][row] = offsets_b_limb_1_col96;
        m31 offsets_b_limb_2_col97 = offsets_b_value[2]; traces[97][row] = offsets_b_limb_2_col97;
        m31 remainder_bits_col98 = (m31)(((uint16_t)offsets_b_value[3]) & UInt16_3);
        traces[98][row] = remainder_bits_col98;
        m31 partial_limb_msb_col99 = (m31)((((uint16_t)remainder_bits_col98) & UInt16_2) >> UInt16_1);
        traces[99][row] = partial_limb_msb_col99;

        sub_component_inputs_memory_id_to_big[10][row] = offsets_b_id_col92;
        lookup_memory_id_to_big_10[0][row] = offsets_b_id_col92;
        lookup_memory_id_to_big_10[1][row] = offsets_b_limb_0_col95;
        lookup_memory_id_to_big_10[2][row] = offsets_b_limb_1_col96;
        lookup_memory_id_to_big_10[3][row] = offsets_b_limb_2_col97;
        lookup_memory_id_to_big_10[4][row] = add(remainder_bits_col98, mul(mid_limbs_set_col94, M31_508));
        for (int i = 5; i < 21; i++) lookup_memory_id_to_big_10[i][row] = mul(mid_limbs_set_col94, M31_511);
        lookup_memory_id_to_big_10[21][row] = sub(mul(M31_136, msb_col93), mid_limbs_set_col94);
        for (int i = 22; i < 28; i++) lookup_memory_id_to_big_10[i][row] = M31_0;
        lookup_memory_id_to_big_10[28][row] = mul(msb_col93, M31_256);

        m31 offset_b = sub(sub(add(add(add(offsets_b_limb_0_col95, mul(offsets_b_limb_1_col96, M31_512)),
                                       mul(offsets_b_limb_2_col97, M31_262144)),
                                   mul(remainder_bits_col98, M31_134217728)),
                               msb_col93),
                           mul(M31_536870912, mid_limbs_set_col94));

        // Read offsets_c
        m31 addr_offsets_c = add(offsets_ptr, M31_2);
        m31 offsets_c_id_col100 = {0};
        memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, addr_offsets_c, &offsets_c_id_col100);
        traces[100][row] = offsets_c_id_col100;
        sub_component_inputs_memory_address_to_id[16][row] = addr_offsets_c;
        lookup_memory_address_to_id_16[0][row] = addr_offsets_c;
        lookup_memory_address_to_id_16[1][row] = offsets_c_id_col100;

        m31 offsets_c_value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, offsets_c_id_col100, offsets_c_value);

        m31 msb_col101 = (offsets_c_value[27] == M31_256) ? M31_1 : M31_0;
        traces[101][row] = msb_col101;
        m31 mid_limbs_set_col102 = (offsets_c_value[20] == M31_511) ? M31_1 : M31_0;
        traces[102][row] = mid_limbs_set_col102;

        m31 offsets_c_limb_0_col103 = offsets_c_value[0]; traces[103][row] = offsets_c_limb_0_col103;
        m31 offsets_c_limb_1_col104 = offsets_c_value[1]; traces[104][row] = offsets_c_limb_1_col104;
        m31 offsets_c_limb_2_col105 = offsets_c_value[2]; traces[105][row] = offsets_c_limb_2_col105;
        m31 remainder_bits_col106 = (m31)(((uint16_t)offsets_c_value[3]) & UInt16_3);
        traces[106][row] = remainder_bits_col106;
        m31 partial_limb_msb_col107 = (m31)((((uint16_t)remainder_bits_col106) & UInt16_2) >> UInt16_1);
        traces[107][row] = partial_limb_msb_col107;

        sub_component_inputs_memory_id_to_big[11][row] = offsets_c_id_col100;
        lookup_memory_id_to_big_11[0][row] = offsets_c_id_col100;
        lookup_memory_id_to_big_11[1][row] = offsets_c_limb_0_col103;
        lookup_memory_id_to_big_11[2][row] = offsets_c_limb_1_col104;
        lookup_memory_id_to_big_11[3][row] = offsets_c_limb_2_col105;
        lookup_memory_id_to_big_11[4][row] = add(remainder_bits_col106, mul(mid_limbs_set_col102, M31_508));
        for (int i = 5; i < 21; i++) lookup_memory_id_to_big_11[i][row] = mul(mid_limbs_set_col102, M31_511);
        lookup_memory_id_to_big_11[21][row] = sub(mul(M31_136, msb_col101), mid_limbs_set_col102);
        for (int i = 22; i < 28; i++) lookup_memory_id_to_big_11[i][row] = M31_0;
        lookup_memory_id_to_big_11[28][row] = mul(msb_col101, M31_256);

        m31 offset_c = sub(sub(add(add(add(offsets_c_limb_0_col103, mul(offsets_c_limb_1_col104, M31_512)),
                                       mul(offsets_c_limb_2_col105, M31_262144)),
                                   mul(remainder_bits_col106, M31_134217728)),
                               msb_col101),
                           mul(M31_536870912, mid_limbs_set_col102));

        // ============ Read A0-A3, B0-B3, C0-C3 operands (columns 108-251) ============
        // Each operand is 99 bits = 11 limbs + 1 id = 12 columns
        // Total: 12 operands * 12 columns = 144 columns (108-251)

        // Helper macro for reading operands
        #define READ_OPERAND(NAME, ADDR_EXPR, COL_START, MEM_ADDR_IDX, MEM_ID_IDX, LOOKUP_ADDR, LOOKUP_ID) \
            m31 NAME##_id = {0}; \
            memory_address_to_id_deduce_output(memory_address_to_id_address_to_raw_id, ADDR_EXPR, &NAME##_id); \
            traces[COL_START][row] = NAME##_id; \
            sub_component_inputs_memory_address_to_id[MEM_ADDR_IDX][row] = ADDR_EXPR; \
            LOOKUP_ADDR[0][row] = ADDR_EXPR; \
            LOOKUP_ADDR[1][row] = NAME##_id; \
            m31 NAME##_value[N_M31_IN_FELT252] = {0}; \
            memory_id_to_big_state_deduce_output(memory_id_to_big_transposed_big_values, memory_id_to_big_small_values, NAME##_id, NAME##_value); \
            m31 NAME##_limbs[11]; \
            for (int i = 0; i < 11; i++) { \
                NAME##_limbs[i] = NAME##_value[i]; \
                traces[COL_START + 1 + i][row] = NAME##_limbs[i]; \
            } \
            sub_component_inputs_memory_id_to_big[MEM_ID_IDX][row] = NAME##_id; \
            LOOKUP_ID[0][row] = NAME##_id; \
            for (int i = 0; i < 11; i++) LOOKUP_ID[1 + i][row] = NAME##_limbs[i]; \
            for (int i = 12; i < 29; i++) LOOKUP_ID[i][row] = M31_0;

        // Compute operand addresses: values_ptr + offset (NOT multiplied by 4)
        // Each operand (a0, a1, a2, a3) is at consecutive addresses starting from values_ptr + offset
        m31 a0_addr = add(values_ptr, offset_a);
        m31 a1_addr = add(a0_addr, M31_1);
        m31 a2_addr = add(a0_addr, M31_2);
        m31 a3_addr = add(a0_addr, M31_3);

        m31 b0_addr = add(values_ptr, offset_b);
        m31 b1_addr = add(b0_addr, M31_1);
        m31 b2_addr = add(b0_addr, M31_2);
        m31 b3_addr = add(b0_addr, M31_3);

        m31 c0_addr = add(values_ptr, offset_c);
        m31 c1_addr = add(c0_addr, M31_1);
        m31 c2_addr = add(c0_addr, M31_2);
        m31 c3_addr = add(c0_addr, M31_3);

        // Read A0-A3 (columns 108-155)
        READ_OPERAND(a0, a0_addr, 108, 17, 12, lookup_memory_address_to_id_17, lookup_memory_id_to_big_12)
        READ_OPERAND(a1, a1_addr, 120, 18, 13, lookup_memory_address_to_id_18, lookup_memory_id_to_big_13)
        READ_OPERAND(a2, a2_addr, 132, 19, 14, lookup_memory_address_to_id_19, lookup_memory_id_to_big_14)
        READ_OPERAND(a3, a3_addr, 144, 20, 15, lookup_memory_address_to_id_20, lookup_memory_id_to_big_15)

        // Read B0-B3 (columns 156-203)
        READ_OPERAND(b0, b0_addr, 156, 21, 16, lookup_memory_address_to_id_21, lookup_memory_id_to_big_16)
        READ_OPERAND(b1, b1_addr, 168, 22, 17, lookup_memory_address_to_id_22, lookup_memory_id_to_big_17)
        READ_OPERAND(b2, b2_addr, 180, 23, 18, lookup_memory_address_to_id_23, lookup_memory_id_to_big_18)
        READ_OPERAND(b3, b3_addr, 192, 24, 19, lookup_memory_address_to_id_24, lookup_memory_id_to_big_19)

        // Read C0-C3 (columns 204-251)
        READ_OPERAND(c0, c0_addr, 204, 25, 20, lookup_memory_address_to_id_25, lookup_memory_id_to_big_20)
        READ_OPERAND(c1, c1_addr, 216, 26, 21, lookup_memory_address_to_id_26, lookup_memory_id_to_big_21)
        READ_OPERAND(c2, c2_addr, 228, 27, 22, lookup_memory_address_to_id_27, lookup_memory_id_to_big_22)
        READ_OPERAND(c3, c3_addr, 240, 28, 23, lookup_memory_address_to_id_28, lookup_memory_id_to_big_23)

        // ============ Compute ab_minus_c_div_p (columns 252-283) ============
        // This is (A * B - C) / P where A, B, C, P are 384-bit numbers (4x99-bit words)
        // For simplicity, we'll compute this step by step

        // Build 384-bit P from P0-P3 (each 99-bit = 11 9-bit limbs)
        int64_t p_12bit[32] = {0};
        int64_t a_384_12bit[32] = {0};
        int64_t b_384_12bit[32] = {0};
        int64_t c_384_12bit[32] = {0};

        // Convert P0-P3 to 12-bit representation
        // Each Px has 11 9-bit limbs = 99 bits
        // 4 x 99 = 396 bits, but we use 384 bits (32 x 12-bit limbs)
        {
            int64_t p_temp[9];
            felt252_to_12bit_array(p0_limbs, p_temp);
            for (int i = 0; i < 8; i++) p_12bit[i] = p_temp[i];

            felt252_to_12bit_array(p1_limbs, p_temp);
            for (int i = 0; i < 8; i++) p_12bit[8 + i] = p_temp[i];

            felt252_to_12bit_array(p2_limbs, p_temp);
            for (int i = 0; i < 8; i++) p_12bit[16 + i] = p_temp[i];

            felt252_to_12bit_array(p3_limbs, p_temp);
            for (int i = 0; i < 8; i++) p_12bit[24 + i] = p_temp[i];
        }

        // Convert A (A0-A3), B (B0-B3), C (C0-C3) to 12-bit
        {
            int64_t temp[9];

            felt252_to_12bit_array(a0_limbs, temp);
            for (int i = 0; i < 8; i++) a_384_12bit[i] = temp[i];
            felt252_to_12bit_array(a1_limbs, temp);
            for (int i = 0; i < 8; i++) a_384_12bit[8 + i] = temp[i];
            felt252_to_12bit_array(a2_limbs, temp);
            for (int i = 0; i < 8; i++) a_384_12bit[16 + i] = temp[i];
            felt252_to_12bit_array(a3_limbs, temp);
            for (int i = 0; i < 8; i++) a_384_12bit[24 + i] = temp[i];

            felt252_to_12bit_array(b0_limbs, temp);
            for (int i = 0; i < 8; i++) b_384_12bit[i] = temp[i];
            felt252_to_12bit_array(b1_limbs, temp);
            for (int i = 0; i < 8; i++) b_384_12bit[8 + i] = temp[i];
            felt252_to_12bit_array(b2_limbs, temp);
            for (int i = 0; i < 8; i++) b_384_12bit[16 + i] = temp[i];
            felt252_to_12bit_array(b3_limbs, temp);
            for (int i = 0; i < 8; i++) b_384_12bit[24 + i] = temp[i];

            felt252_to_12bit_array(c0_limbs, temp);
            for (int i = 0; i < 8; i++) c_384_12bit[i] = temp[i];
            felt252_to_12bit_array(c1_limbs, temp);
            for (int i = 0; i < 8; i++) c_384_12bit[8 + i] = temp[i];
            felt252_to_12bit_array(c2_limbs, temp);
            for (int i = 0; i < 8; i++) c_384_12bit[16 + i] = temp[i];
            felt252_to_12bit_array(c3_limbs, temp);
            for (int i = 0; i < 8; i++) c_384_12bit[24 + i] = temp[i];
        }

        // Compute A * B using Double Karatsuba for constraint-compatible values
        // ab_768_karatsuba has 63 elements (indices 0-62)
        int64_t ab_768_karatsuba[63] = {0};
        double_karatsuba_n8(a_384_12bit, b_384_12bit, ab_768_karatsuba);

        // Also compute schoolbook version for quotient computation
        int64_t ab_768[64] = {0};
        mul_384x384(a_384_12bit, b_384_12bit, ab_768);
        normalize_12bit(ab_768, 64);

        // Convert C to 768-bit (zero-extend)
        int64_t c_768[64] = {0};
        for (int i = 0; i < 32; i++) c_768[i] = c_384_12bit[i];

        // Compute A*B - C (for quotient computation)
        sub_768(ab_768, c_768);
        normalize_12bit(ab_768, 64);

        // Convert P to 768-bit for division
        int64_t p_768[64] = {0};
        for (int i = 0; i < 32; i++) p_768[i] = p_12bit[i];

        // Divide (A*B - C) by P to get quotient
        int64_t quotient[32] = {0};
        div_768_by_384(ab_768, p_12bit, quotient);

        // Write quotient to columns 252-283 (32 12-bit limbs)
        m31 ab_minus_c_div_p_limbs[32];
        for (int i = 0; i < 32; i++) {
            ab_minus_c_div_p_limbs[i] = m31{(uint32_t)(quotient[i] & 0xFFF)};
            traces[252 + i][row] = ab_minus_c_div_p_limbs[i];
        }

        // Populate range_check_12 lookups for quotient limbs
        // Note: sub_component_inputs_range_check_12 is not used (passed as empty vec from Rust)
        lookup_range_check_12_0[0][row] = ab_minus_c_div_p_limbs[0];
        lookup_range_check_12_1[0][row] = ab_minus_c_div_p_limbs[1];
        lookup_range_check_12_2[0][row] = ab_minus_c_div_p_limbs[2];
        lookup_range_check_12_3[0][row] = ab_minus_c_div_p_limbs[3];
        lookup_range_check_12_4[0][row] = ab_minus_c_div_p_limbs[4];
        lookup_range_check_12_5[0][row] = ab_minus_c_div_p_limbs[5];
        lookup_range_check_12_6[0][row] = ab_minus_c_div_p_limbs[6];
        lookup_range_check_12_7[0][row] = ab_minus_c_div_p_limbs[7];
        lookup_range_check_12_8[0][row] = ab_minus_c_div_p_limbs[8];
        lookup_range_check_12_9[0][row] = ab_minus_c_div_p_limbs[9];
        lookup_range_check_12_10[0][row] = ab_minus_c_div_p_limbs[10];
        lookup_range_check_12_11[0][row] = ab_minus_c_div_p_limbs[11];
        lookup_range_check_12_12[0][row] = ab_minus_c_div_p_limbs[12];
        lookup_range_check_12_13[0][row] = ab_minus_c_div_p_limbs[13];
        lookup_range_check_12_14[0][row] = ab_minus_c_div_p_limbs[14];
        lookup_range_check_12_15[0][row] = ab_minus_c_div_p_limbs[15];
        lookup_range_check_12_16[0][row] = ab_minus_c_div_p_limbs[16];
        lookup_range_check_12_17[0][row] = ab_minus_c_div_p_limbs[17];
        lookup_range_check_12_18[0][row] = ab_minus_c_div_p_limbs[18];
        lookup_range_check_12_19[0][row] = ab_minus_c_div_p_limbs[19];
        lookup_range_check_12_20[0][row] = ab_minus_c_div_p_limbs[20];
        lookup_range_check_12_21[0][row] = ab_minus_c_div_p_limbs[21];
        lookup_range_check_12_22[0][row] = ab_minus_c_div_p_limbs[22];
        lookup_range_check_12_23[0][row] = ab_minus_c_div_p_limbs[23];
        lookup_range_check_12_24[0][row] = ab_minus_c_div_p_limbs[24];
        lookup_range_check_12_25[0][row] = ab_minus_c_div_p_limbs[25];
        lookup_range_check_12_26[0][row] = ab_minus_c_div_p_limbs[26];
        lookup_range_check_12_27[0][row] = ab_minus_c_div_p_limbs[27];
        lookup_range_check_12_28[0][row] = ab_minus_c_div_p_limbs[28];
        lookup_range_check_12_29[0][row] = ab_minus_c_div_p_limbs[29];
        lookup_range_check_12_30[0][row] = ab_minus_c_div_p_limbs[30];
        lookup_range_check_12_31[0][row] = ab_minus_c_div_p_limbs[31];

        // ============ Limb decomposition for range checks (columns 284-363) ============
        // For each operand (P0-P3, A0-A3, B0-B3, C0-C3), we decompose limbs 1,2,5,6,9
        // Each operand gets 5 columns: limb1b, limb2b, limb5b, limb6b, limb9b
        // Total: 16 operands × 5 columns = 80 columns (284-363)
        // Decomposition: limb1b = limb1 >> 3, limb2b = limb2 >> 6, etc.

        // Helper macro for limb decomposition
        #define DECOMPOSE_LIMBS(NAME, LIMBS, COL_BASE, LOOKUP_BASE) \
            m31 limb1b_##NAME = (m31)(((uint16_t)LIMBS[1]) >> 3); \
            m31 limb1a_##NAME = sub(LIMBS[1], mul(limb1b_##NAME, M31_8)); \
            m31 limb2b_##NAME = (m31)(((uint16_t)LIMBS[2]) >> 6); \
            m31 limb2a_##NAME = sub(LIMBS[2], mul(limb2b_##NAME, M31_64)); \
            m31 limb5b_##NAME = (m31)(((uint16_t)LIMBS[5]) >> 3); \
            m31 limb5a_##NAME = sub(LIMBS[5], mul(limb5b_##NAME, M31_8)); \
            m31 limb6b_##NAME = (m31)(((uint16_t)LIMBS[6]) >> 6); \
            m31 limb6a_##NAME = sub(LIMBS[6], mul(limb6b_##NAME, M31_64)); \
            m31 limb9b_##NAME = (m31)(((uint16_t)LIMBS[9]) >> 3); \
            m31 limb9a_##NAME = sub(LIMBS[9], mul(limb9b_##NAME, M31_8)); \
            traces[COL_BASE + 0][row] = limb1b_##NAME; \
            traces[COL_BASE + 1][row] = limb2b_##NAME; \
            traces[COL_BASE + 2][row] = limb5b_##NAME; \
            traces[COL_BASE + 3][row] = limb6b_##NAME; \
            traces[COL_BASE + 4][row] = limb9b_##NAME;

        // Decompose P0-P3 (columns 284-303)
        DECOMPOSE_LIMBS(p0, p0_limbs, 284, 0)
        DECOMPOSE_LIMBS(p1, p1_limbs, 289, 2)
        DECOMPOSE_LIMBS(p2, p2_limbs, 294, 5)
        DECOMPOSE_LIMBS(p3, p3_limbs, 299, 7)

        // Decompose A0-A3 (columns 304-323)
        DECOMPOSE_LIMBS(a0, a0_limbs, 304, 10)
        DECOMPOSE_LIMBS(a1, a1_limbs, 309, 12)
        DECOMPOSE_LIMBS(a2, a2_limbs, 314, 15)
        DECOMPOSE_LIMBS(a3, a3_limbs, 319, 17)

        // Decompose B0-B3 (columns 324-343)
        DECOMPOSE_LIMBS(b0, b0_limbs, 324, 20)
        DECOMPOSE_LIMBS(b1, b1_limbs, 329, 22)
        DECOMPOSE_LIMBS(b2, b2_limbs, 334, 25)
        DECOMPOSE_LIMBS(b3, b3_limbs, 339, 27)

        // Decompose C0-C3 (columns 344-363)
        DECOMPOSE_LIMBS(c0, c0_limbs, 344, 30)
        DECOMPOSE_LIMBS(c1, c1_limbs, 349, 32)
        DECOMPOSE_LIMBS(c2, c2_limbs, 354, 35)
        DECOMPOSE_LIMBS(c3, c3_limbs, 359, 37)

        #undef DECOMPOSE_LIMBS

        // Fill range_check_3_6_6_3 lookups (40 lookups organized as pairs)
        // P0+P1: lookups 0-4
        lookup_range_check_3_6_6_3_0[0][row] = limb1a_p0; lookup_range_check_3_6_6_3_0[1][row] = limb1b_p0;
        lookup_range_check_3_6_6_3_0[2][row] = limb2a_p0; lookup_range_check_3_6_6_3_0[3][row] = limb2b_p0;
        lookup_range_check_3_6_6_3_1[0][row] = limb5a_p0; lookup_range_check_3_6_6_3_1[1][row] = limb5b_p0;
        lookup_range_check_3_6_6_3_1[2][row] = limb6a_p0; lookup_range_check_3_6_6_3_1[3][row] = limb6b_p0;
        lookup_range_check_3_6_6_3_2[0][row] = limb1a_p1; lookup_range_check_3_6_6_3_2[1][row] = limb1b_p1;
        lookup_range_check_3_6_6_3_2[2][row] = limb2a_p1; lookup_range_check_3_6_6_3_2[3][row] = limb2b_p1;
        lookup_range_check_3_6_6_3_3[0][row] = limb5a_p1; lookup_range_check_3_6_6_3_3[1][row] = limb5b_p1;
        lookup_range_check_3_6_6_3_3[2][row] = limb6a_p1; lookup_range_check_3_6_6_3_3[3][row] = limb6b_p1;
        lookup_range_check_3_6_6_3_4[0][row] = limb9a_p0; lookup_range_check_3_6_6_3_4[1][row] = limb9b_p0;
        lookup_range_check_3_6_6_3_4[2][row] = limb9b_p1; lookup_range_check_3_6_6_3_4[3][row] = limb9a_p1;

        // P2+P3: lookups 5-9
        lookup_range_check_3_6_6_3_5[0][row] = limb1a_p2; lookup_range_check_3_6_6_3_5[1][row] = limb1b_p2;
        lookup_range_check_3_6_6_3_5[2][row] = limb2a_p2; lookup_range_check_3_6_6_3_5[3][row] = limb2b_p2;
        lookup_range_check_3_6_6_3_6[0][row] = limb5a_p2; lookup_range_check_3_6_6_3_6[1][row] = limb5b_p2;
        lookup_range_check_3_6_6_3_6[2][row] = limb6a_p2; lookup_range_check_3_6_6_3_6[3][row] = limb6b_p2;
        lookup_range_check_3_6_6_3_7[0][row] = limb1a_p3; lookup_range_check_3_6_6_3_7[1][row] = limb1b_p3;
        lookup_range_check_3_6_6_3_7[2][row] = limb2a_p3; lookup_range_check_3_6_6_3_7[3][row] = limb2b_p3;
        lookup_range_check_3_6_6_3_8[0][row] = limb5a_p3; lookup_range_check_3_6_6_3_8[1][row] = limb5b_p3;
        lookup_range_check_3_6_6_3_8[2][row] = limb6a_p3; lookup_range_check_3_6_6_3_8[3][row] = limb6b_p3;
        lookup_range_check_3_6_6_3_9[0][row] = limb9a_p2; lookup_range_check_3_6_6_3_9[1][row] = limb9b_p2;
        lookup_range_check_3_6_6_3_9[2][row] = limb9b_p3; lookup_range_check_3_6_6_3_9[3][row] = limb9a_p3;

        // A0+A1: lookups 10-14
        lookup_range_check_3_6_6_3_10[0][row] = limb1a_a0; lookup_range_check_3_6_6_3_10[1][row] = limb1b_a0;
        lookup_range_check_3_6_6_3_10[2][row] = limb2a_a0; lookup_range_check_3_6_6_3_10[3][row] = limb2b_a0;
        lookup_range_check_3_6_6_3_11[0][row] = limb5a_a0; lookup_range_check_3_6_6_3_11[1][row] = limb5b_a0;
        lookup_range_check_3_6_6_3_11[2][row] = limb6a_a0; lookup_range_check_3_6_6_3_11[3][row] = limb6b_a0;
        lookup_range_check_3_6_6_3_12[0][row] = limb1a_a1; lookup_range_check_3_6_6_3_12[1][row] = limb1b_a1;
        lookup_range_check_3_6_6_3_12[2][row] = limb2a_a1; lookup_range_check_3_6_6_3_12[3][row] = limb2b_a1;
        lookup_range_check_3_6_6_3_13[0][row] = limb5a_a1; lookup_range_check_3_6_6_3_13[1][row] = limb5b_a1;
        lookup_range_check_3_6_6_3_13[2][row] = limb6a_a1; lookup_range_check_3_6_6_3_13[3][row] = limb6b_a1;
        lookup_range_check_3_6_6_3_14[0][row] = limb9a_a0; lookup_range_check_3_6_6_3_14[1][row] = limb9b_a0;
        lookup_range_check_3_6_6_3_14[2][row] = limb9b_a1; lookup_range_check_3_6_6_3_14[3][row] = limb9a_a1;

        // A2+A3: lookups 15-19
        lookup_range_check_3_6_6_3_15[0][row] = limb1a_a2; lookup_range_check_3_6_6_3_15[1][row] = limb1b_a2;
        lookup_range_check_3_6_6_3_15[2][row] = limb2a_a2; lookup_range_check_3_6_6_3_15[3][row] = limb2b_a2;
        lookup_range_check_3_6_6_3_16[0][row] = limb5a_a2; lookup_range_check_3_6_6_3_16[1][row] = limb5b_a2;
        lookup_range_check_3_6_6_3_16[2][row] = limb6a_a2; lookup_range_check_3_6_6_3_16[3][row] = limb6b_a2;
        lookup_range_check_3_6_6_3_17[0][row] = limb1a_a3; lookup_range_check_3_6_6_3_17[1][row] = limb1b_a3;
        lookup_range_check_3_6_6_3_17[2][row] = limb2a_a3; lookup_range_check_3_6_6_3_17[3][row] = limb2b_a3;
        lookup_range_check_3_6_6_3_18[0][row] = limb5a_a3; lookup_range_check_3_6_6_3_18[1][row] = limb5b_a3;
        lookup_range_check_3_6_6_3_18[2][row] = limb6a_a3; lookup_range_check_3_6_6_3_18[3][row] = limb6b_a3;
        lookup_range_check_3_6_6_3_19[0][row] = limb9a_a2; lookup_range_check_3_6_6_3_19[1][row] = limb9b_a2;
        lookup_range_check_3_6_6_3_19[2][row] = limb9b_a3; lookup_range_check_3_6_6_3_19[3][row] = limb9a_a3;

        // B0+B1: lookups 20-24
        lookup_range_check_3_6_6_3_20[0][row] = limb1a_b0; lookup_range_check_3_6_6_3_20[1][row] = limb1b_b0;
        lookup_range_check_3_6_6_3_20[2][row] = limb2a_b0; lookup_range_check_3_6_6_3_20[3][row] = limb2b_b0;
        lookup_range_check_3_6_6_3_21[0][row] = limb5a_b0; lookup_range_check_3_6_6_3_21[1][row] = limb5b_b0;
        lookup_range_check_3_6_6_3_21[2][row] = limb6a_b0; lookup_range_check_3_6_6_3_21[3][row] = limb6b_b0;
        lookup_range_check_3_6_6_3_22[0][row] = limb1a_b1; lookup_range_check_3_6_6_3_22[1][row] = limb1b_b1;
        lookup_range_check_3_6_6_3_22[2][row] = limb2a_b1; lookup_range_check_3_6_6_3_22[3][row] = limb2b_b1;
        lookup_range_check_3_6_6_3_23[0][row] = limb5a_b1; lookup_range_check_3_6_6_3_23[1][row] = limb5b_b1;
        lookup_range_check_3_6_6_3_23[2][row] = limb6a_b1; lookup_range_check_3_6_6_3_23[3][row] = limb6b_b1;
        lookup_range_check_3_6_6_3_24[0][row] = limb9a_b0; lookup_range_check_3_6_6_3_24[1][row] = limb9b_b0;
        lookup_range_check_3_6_6_3_24[2][row] = limb9b_b1; lookup_range_check_3_6_6_3_24[3][row] = limb9a_b1;

        // B2+B3: lookups 25-29
        lookup_range_check_3_6_6_3_25[0][row] = limb1a_b2; lookup_range_check_3_6_6_3_25[1][row] = limb1b_b2;
        lookup_range_check_3_6_6_3_25[2][row] = limb2a_b2; lookup_range_check_3_6_6_3_25[3][row] = limb2b_b2;
        lookup_range_check_3_6_6_3_26[0][row] = limb5a_b2; lookup_range_check_3_6_6_3_26[1][row] = limb5b_b2;
        lookup_range_check_3_6_6_3_26[2][row] = limb6a_b2; lookup_range_check_3_6_6_3_26[3][row] = limb6b_b2;
        lookup_range_check_3_6_6_3_27[0][row] = limb1a_b3; lookup_range_check_3_6_6_3_27[1][row] = limb1b_b3;
        lookup_range_check_3_6_6_3_27[2][row] = limb2a_b3; lookup_range_check_3_6_6_3_27[3][row] = limb2b_b3;
        lookup_range_check_3_6_6_3_28[0][row] = limb5a_b3; lookup_range_check_3_6_6_3_28[1][row] = limb5b_b3;
        lookup_range_check_3_6_6_3_28[2][row] = limb6a_b3; lookup_range_check_3_6_6_3_28[3][row] = limb6b_b3;
        lookup_range_check_3_6_6_3_29[0][row] = limb9a_b2; lookup_range_check_3_6_6_3_29[1][row] = limb9b_b2;
        lookup_range_check_3_6_6_3_29[2][row] = limb9b_b3; lookup_range_check_3_6_6_3_29[3][row] = limb9a_b3;

        // C0+C1: lookups 30-34
        lookup_range_check_3_6_6_3_30[0][row] = limb1a_c0; lookup_range_check_3_6_6_3_30[1][row] = limb1b_c0;
        lookup_range_check_3_6_6_3_30[2][row] = limb2a_c0; lookup_range_check_3_6_6_3_30[3][row] = limb2b_c0;
        lookup_range_check_3_6_6_3_31[0][row] = limb5a_c0; lookup_range_check_3_6_6_3_31[1][row] = limb5b_c0;
        lookup_range_check_3_6_6_3_31[2][row] = limb6a_c0; lookup_range_check_3_6_6_3_31[3][row] = limb6b_c0;
        lookup_range_check_3_6_6_3_32[0][row] = limb1a_c1; lookup_range_check_3_6_6_3_32[1][row] = limb1b_c1;
        lookup_range_check_3_6_6_3_32[2][row] = limb2a_c1; lookup_range_check_3_6_6_3_32[3][row] = limb2b_c1;
        lookup_range_check_3_6_6_3_33[0][row] = limb5a_c1; lookup_range_check_3_6_6_3_33[1][row] = limb5b_c1;
        lookup_range_check_3_6_6_3_33[2][row] = limb6a_c1; lookup_range_check_3_6_6_3_33[3][row] = limb6b_c1;
        lookup_range_check_3_6_6_3_34[0][row] = limb9a_c0; lookup_range_check_3_6_6_3_34[1][row] = limb9b_c0;
        lookup_range_check_3_6_6_3_34[2][row] = limb9b_c1; lookup_range_check_3_6_6_3_34[3][row] = limb9a_c1;

        // C2+C3: lookups 35-39
        lookup_range_check_3_6_6_3_35[0][row] = limb1a_c2; lookup_range_check_3_6_6_3_35[1][row] = limb1b_c2;
        lookup_range_check_3_6_6_3_35[2][row] = limb2a_c2; lookup_range_check_3_6_6_3_35[3][row] = limb2b_c2;
        lookup_range_check_3_6_6_3_36[0][row] = limb5a_c2; lookup_range_check_3_6_6_3_36[1][row] = limb5b_c2;
        lookup_range_check_3_6_6_3_36[2][row] = limb6a_c2; lookup_range_check_3_6_6_3_36[3][row] = limb6b_c2;
        lookup_range_check_3_6_6_3_37[0][row] = limb1a_c3; lookup_range_check_3_6_6_3_37[1][row] = limb1b_c3;
        lookup_range_check_3_6_6_3_37[2][row] = limb2a_c3; lookup_range_check_3_6_6_3_37[3][row] = limb2b_c3;
        lookup_range_check_3_6_6_3_38[0][row] = limb5a_c3; lookup_range_check_3_6_6_3_38[1][row] = limb5b_c3;
        lookup_range_check_3_6_6_3_38[2][row] = limb6a_c3; lookup_range_check_3_6_6_3_38[3][row] = limb6b_c3;
        lookup_range_check_3_6_6_3_39[0][row] = limb9a_c2; lookup_range_check_3_6_6_3_39[1][row] = limb9b_c2;
        lookup_range_check_3_6_6_3_39[2][row] = limb9b_c3; lookup_range_check_3_6_6_3_39[3][row] = limb9a_c3;

        // ============ Carry columns (columns 364-425) ============
        // Compute Q*P using Double Karatsuba for constraint-compatible values
        // qp_768_karatsuba has 63 elements (indices 0-62)
        int64_t qp_768_karatsuba[63] = {0};
        double_karatsuba_n8(quotient, p_12bit, qp_768_karatsuba);

        // Carry computation in M31 field arithmetic:
        // carry[i] = (carry[i-1] - c[i] + ab[i] - qp[i]) * M31_524288 (mod P)
        // M31_524288 = 524288 = 2^19
        // ab and qp are from Karatsuba (un-normalized, can be large or negative)
        const m31 M31_524288 = 524288;

        m31 carries[62];
        m31 carry = M31_0;

        // First 32 carries - need to subtract c
        for (int i = 0; i < 32; i++) {
            // Convert Karatsuba values to M31 (handle potential negatives)
            m31 ab_m31 = (m31)(((ab_768_karatsuba[i] % (int64_t)P) + P) % P);
            m31 qp_m31 = (m31)(((qp_768_karatsuba[i] % (int64_t)P) + P) % P);
            m31 c_m31 = (m31)c_384_12bit[i];

            // diff = carry - c + ab - qp
            m31 diff = sub(add(carry, sub(ab_m31, qp_m31)), c_m31);
            // carry = diff * M31_524288
            carry = mul(diff, M31_524288);
            carries[i] = carry;
        }

        // Remaining 30 carries - no c to subtract
        for (int i = 32; i < 62; i++) {
            m31 ab_m31 = (m31)(((ab_768_karatsuba[i] % (int64_t)P) + P) % P);
            m31 qp_m31 = (m31)(((qp_768_karatsuba[i] % (int64_t)P) + P) % P);

            // diff = carry + ab - qp
            m31 diff = add(carry, sub(ab_m31, qp_m31));
            // carry = diff * M31_524288
            carry = mul(diff, M31_524288);
            carries[i] = carry;
        }

        // Write carry columns 364-425 (carries are already m31 values)
        for (int i = 0; i < 62; i++) {
            traces[364 + i][row] = carries[i];
        }

        // Fill range_check_18 lookups (carries + bias M31_131072)
        // The range check verifies carry + 131072 is in range [0, 2^18)
        const m31 M31_131072 = 131072;
        #define SET_CARRY_LOOKUP(IDX) \
            { \
                lookup_range_check_18_##IDX[0][row] = add(carries[IDX], M31_131072); \
            }

        SET_CARRY_LOOKUP(0)  SET_CARRY_LOOKUP(1)  SET_CARRY_LOOKUP(2)  SET_CARRY_LOOKUP(3)
        SET_CARRY_LOOKUP(4)  SET_CARRY_LOOKUP(5)  SET_CARRY_LOOKUP(6)  SET_CARRY_LOOKUP(7)
        SET_CARRY_LOOKUP(8)  SET_CARRY_LOOKUP(9)  SET_CARRY_LOOKUP(10) SET_CARRY_LOOKUP(11)
        SET_CARRY_LOOKUP(12) SET_CARRY_LOOKUP(13) SET_CARRY_LOOKUP(14) SET_CARRY_LOOKUP(15)
        SET_CARRY_LOOKUP(16) SET_CARRY_LOOKUP(17) SET_CARRY_LOOKUP(18) SET_CARRY_LOOKUP(19)
        SET_CARRY_LOOKUP(20) SET_CARRY_LOOKUP(21) SET_CARRY_LOOKUP(22) SET_CARRY_LOOKUP(23)
        SET_CARRY_LOOKUP(24) SET_CARRY_LOOKUP(25) SET_CARRY_LOOKUP(26) SET_CARRY_LOOKUP(27)
        SET_CARRY_LOOKUP(28) SET_CARRY_LOOKUP(29) SET_CARRY_LOOKUP(30) SET_CARRY_LOOKUP(31)
        SET_CARRY_LOOKUP(32) SET_CARRY_LOOKUP(33) SET_CARRY_LOOKUP(34) SET_CARRY_LOOKUP(35)
        SET_CARRY_LOOKUP(36) SET_CARRY_LOOKUP(37) SET_CARRY_LOOKUP(38) SET_CARRY_LOOKUP(39)
        SET_CARRY_LOOKUP(40) SET_CARRY_LOOKUP(41) SET_CARRY_LOOKUP(42) SET_CARRY_LOOKUP(43)
        SET_CARRY_LOOKUP(44) SET_CARRY_LOOKUP(45) SET_CARRY_LOOKUP(46) SET_CARRY_LOOKUP(47)
        SET_CARRY_LOOKUP(48) SET_CARRY_LOOKUP(49) SET_CARRY_LOOKUP(50) SET_CARRY_LOOKUP(51)
        SET_CARRY_LOOKUP(52) SET_CARRY_LOOKUP(53) SET_CARRY_LOOKUP(54) SET_CARRY_LOOKUP(55)
        SET_CARRY_LOOKUP(56) SET_CARRY_LOOKUP(57) SET_CARRY_LOOKUP(58) SET_CARRY_LOOKUP(59)
        SET_CARRY_LOOKUP(60) SET_CARRY_LOOKUP(61)

        #undef SET_CARRY_LOOKUP

        #undef READ_OPERAND
    }
}

// Template kernel for generating interaction trace column with two lookup types
template <int N, int M>
__global__ void generate_mul_mod_builtin_interaction_col_gen_kernel(
    LookupElementsBasic<N> *lookup_elements_n,
    LookupElementsBasic<M> *lookup_elements_m,
    m31 **lookup_state_0,
    m31 **lookup_state_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 init_combine_reg[N] = {};
    m31 final_combine_reg[M] = {};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }
    for (int i = 0; i < M; i++) {
        final_combine_reg[i] = lookup_state_1[i][vec_index];
    }
    if (vec_index < trace_size) {
        qm31 denom0 = lookup_elements_n->combine(init_combine_reg, N);
        qm31 denom1 = lookup_elements_m->combine(final_combine_reg, M);
        logup_col_write_frac(vec_index, add(denom1, denom0), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Last column kernel - single lookup
template <int N>
__global__ void generate_mul_mod_builtin_interaction_last_col_kernel(
    LookupElementsBasic<N> *lookup_elements,
    m31 **lookup_state,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 combine_reg[N] = {};

    for (int i = 0; i < N; i++) {
        combine_reg[i] = lookup_state[i][vec_index];
    }
    if (vec_index < trace_size) {
        qm31 denom = lookup_elements->combine(combine_reg, N);
        qm31 one = {1, 0, 0, 0};
        logup_col_write_frac(vec_index, one, denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize column kernel - accumulates interaction trace values
__global__ void generate_mul_mod_builtin_interaction_finalize_col_kernel(
    unsigned col_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = (col_index == 0) ? -1 : static_cast<int>(col_index - 1);

    if (vec_index < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[vec_index], numerator1[vec_index]},
                cm31{numerator2[vec_index], numerator3[vec_index]}
            },
            denom_inv_ptr[vec_index]
        );

        if (pre_index == -1) {
            interaction_traces[0][vec_index] = {0};
            interaction_traces[1][vec_index] = {0};
            interaction_traces[2][vec_index] = {0};
            interaction_traces[3][vec_index] = {0};
            qm31 tmp = value;
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        } else {
            qm31 prev_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][vec_index],
                     interaction_traces[pre_index * 4 + 1][vec_index]},
                cm31{interaction_traces[pre_index * 4 + 2][vec_index],
                     interaction_traces[pre_index * 4 + 3][vec_index]}
            };
            qm31 tmp = add(value, prev_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        }

        interaction_traces[col_index * 4 + 0][vec_index] = numerator0[vec_index];
        interaction_traces[col_index * 4 + 1][vec_index] = numerator1[vec_index];
        interaction_traces[col_index * 4 + 2][vec_index] = numerator2[vec_index];
        interaction_traces[col_index * 4 + 3][vec_index] = numerator3[vec_index];
    }
}

// Cumsum shift kernel - computes the sum for shifting
__global__ void generate_mul_mod_builtin_interaction_cumsum_shift_kernel(
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_traces,
    m31 *coordinate_sums
) {
    int idx0 = 4 * last_index - 4;
    int idx1 = 4 * last_index - 3;
    int idx2 = 4 * last_index - 2;
    int idx3 = 4 * last_index - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = 0;
    m31 sum1 = 0;
    m31 sum2 = 0;
    m31 sum3 = 0;

    for (int i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_traces[idx0][i]);
        sum1 = add(sum1, interaction_traces[idx1][i]);
        sum2 = add(sum2, interaction_traces[idx2][i]);
        sum3 = add(sum3, interaction_traces[idx3][i]);
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

// Coord prefix sum kernel - applies the shift
__global__ void generate_mul_mod_builtin_interaction_coord_prefix_sum_kernel(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interaction_traces[4 * last_index - 4][vec_index] = sub(interaction_traces[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interaction_traces[4 * last_index - 3][vec_index] = sub(interaction_traces[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interaction_traces[4 * last_index - 2][vec_index] = sub(interaction_traces[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interaction_traces[4 * last_index - 1][vec_index] = sub(interaction_traces[4 * last_index - 1][vec_index], cumsum_shift.b.b);
    }
}

extern "C"
void generate_mul_mod_builtin_traces(
    unsigned **traces,

    // All lookup data arrays
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_address_to_id_3,
    unsigned **lookup_memory_address_to_id_4,
    unsigned **lookup_memory_address_to_id_5,
    unsigned **lookup_memory_address_to_id_6,
    unsigned **lookup_memory_address_to_id_7,
    unsigned **lookup_memory_address_to_id_8,
    unsigned **lookup_memory_address_to_id_9,
    unsigned **lookup_memory_address_to_id_10,
    unsigned **lookup_memory_address_to_id_11,
    unsigned **lookup_memory_address_to_id_12,
    unsigned **lookup_memory_address_to_id_13,
    unsigned **lookup_memory_address_to_id_14,
    unsigned **lookup_memory_address_to_id_15,
    unsigned **lookup_memory_address_to_id_16,
    unsigned **lookup_memory_address_to_id_17,
    unsigned **lookup_memory_address_to_id_18,
    unsigned **lookup_memory_address_to_id_19,
    unsigned **lookup_memory_address_to_id_20,
    unsigned **lookup_memory_address_to_id_21,
    unsigned **lookup_memory_address_to_id_22,
    unsigned **lookup_memory_address_to_id_23,
    unsigned **lookup_memory_address_to_id_24,
    unsigned **lookup_memory_address_to_id_25,
    unsigned **lookup_memory_address_to_id_26,
    unsigned **lookup_memory_address_to_id_27,
    unsigned **lookup_memory_address_to_id_28,

    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_memory_id_to_big_3,
    unsigned **lookup_memory_id_to_big_4,
    unsigned **lookup_memory_id_to_big_5,
    unsigned **lookup_memory_id_to_big_6,
    unsigned **lookup_memory_id_to_big_7,
    unsigned **lookup_memory_id_to_big_8,
    unsigned **lookup_memory_id_to_big_9,
    unsigned **lookup_memory_id_to_big_10,
    unsigned **lookup_memory_id_to_big_11,
    unsigned **lookup_memory_id_to_big_12,
    unsigned **lookup_memory_id_to_big_13,
    unsigned **lookup_memory_id_to_big_14,
    unsigned **lookup_memory_id_to_big_15,
    unsigned **lookup_memory_id_to_big_16,
    unsigned **lookup_memory_id_to_big_17,
    unsigned **lookup_memory_id_to_big_18,
    unsigned **lookup_memory_id_to_big_19,
    unsigned **lookup_memory_id_to_big_20,
    unsigned **lookup_memory_id_to_big_21,
    unsigned **lookup_memory_id_to_big_22,
    unsigned **lookup_memory_id_to_big_23,

    unsigned **lookup_range_check_12_0,
    unsigned **lookup_range_check_12_1,
    unsigned **lookup_range_check_12_2,
    unsigned **lookup_range_check_12_3,
    unsigned **lookup_range_check_12_4,
    unsigned **lookup_range_check_12_5,
    unsigned **lookup_range_check_12_6,
    unsigned **lookup_range_check_12_7,
    unsigned **lookup_range_check_12_8,
    unsigned **lookup_range_check_12_9,
    unsigned **lookup_range_check_12_10,
    unsigned **lookup_range_check_12_11,
    unsigned **lookup_range_check_12_12,
    unsigned **lookup_range_check_12_13,
    unsigned **lookup_range_check_12_14,
    unsigned **lookup_range_check_12_15,
    unsigned **lookup_range_check_12_16,
    unsigned **lookup_range_check_12_17,
    unsigned **lookup_range_check_12_18,
    unsigned **lookup_range_check_12_19,
    unsigned **lookup_range_check_12_20,
    unsigned **lookup_range_check_12_21,
    unsigned **lookup_range_check_12_22,
    unsigned **lookup_range_check_12_23,
    unsigned **lookup_range_check_12_24,
    unsigned **lookup_range_check_12_25,
    unsigned **lookup_range_check_12_26,
    unsigned **lookup_range_check_12_27,
    unsigned **lookup_range_check_12_28,
    unsigned **lookup_range_check_12_29,
    unsigned **lookup_range_check_12_30,
    unsigned **lookup_range_check_12_31,

    unsigned **lookup_range_check_18_0,
    unsigned **lookup_range_check_18_1,
    unsigned **lookup_range_check_18_2,
    unsigned **lookup_range_check_18_3,
    unsigned **lookup_range_check_18_4,
    unsigned **lookup_range_check_18_5,
    unsigned **lookup_range_check_18_6,
    unsigned **lookup_range_check_18_7,
    unsigned **lookup_range_check_18_8,
    unsigned **lookup_range_check_18_9,
    unsigned **lookup_range_check_18_10,
    unsigned **lookup_range_check_18_11,
    unsigned **lookup_range_check_18_12,
    unsigned **lookup_range_check_18_13,
    unsigned **lookup_range_check_18_14,
    unsigned **lookup_range_check_18_15,
    unsigned **lookup_range_check_18_16,
    unsigned **lookup_range_check_18_17,
    unsigned **lookup_range_check_18_18,
    unsigned **lookup_range_check_18_19,
    unsigned **lookup_range_check_18_20,
    unsigned **lookup_range_check_18_21,
    unsigned **lookup_range_check_18_22,
    unsigned **lookup_range_check_18_23,
    unsigned **lookup_range_check_18_24,
    unsigned **lookup_range_check_18_25,
    unsigned **lookup_range_check_18_26,
    unsigned **lookup_range_check_18_27,
    unsigned **lookup_range_check_18_28,
    unsigned **lookup_range_check_18_29,
    unsigned **lookup_range_check_18_30,
    unsigned **lookup_range_check_18_31,
    unsigned **lookup_range_check_18_32,
    unsigned **lookup_range_check_18_33,
    unsigned **lookup_range_check_18_34,
    unsigned **lookup_range_check_18_35,
    unsigned **lookup_range_check_18_36,
    unsigned **lookup_range_check_18_37,
    unsigned **lookup_range_check_18_38,
    unsigned **lookup_range_check_18_39,
    unsigned **lookup_range_check_18_40,
    unsigned **lookup_range_check_18_41,
    unsigned **lookup_range_check_18_42,
    unsigned **lookup_range_check_18_43,
    unsigned **lookup_range_check_18_44,
    unsigned **lookup_range_check_18_45,
    unsigned **lookup_range_check_18_46,
    unsigned **lookup_range_check_18_47,
    unsigned **lookup_range_check_18_48,
    unsigned **lookup_range_check_18_49,
    unsigned **lookup_range_check_18_50,
    unsigned **lookup_range_check_18_51,
    unsigned **lookup_range_check_18_52,
    unsigned **lookup_range_check_18_53,
    unsigned **lookup_range_check_18_54,
    unsigned **lookup_range_check_18_55,
    unsigned **lookup_range_check_18_56,
    unsigned **lookup_range_check_18_57,
    unsigned **lookup_range_check_18_58,
    unsigned **lookup_range_check_18_59,
    unsigned **lookup_range_check_18_60,
    unsigned **lookup_range_check_18_61,

    unsigned **lookup_range_check_3_6_6_3_0,
    unsigned **lookup_range_check_3_6_6_3_1,
    unsigned **lookup_range_check_3_6_6_3_2,
    unsigned **lookup_range_check_3_6_6_3_3,
    unsigned **lookup_range_check_3_6_6_3_4,
    unsigned **lookup_range_check_3_6_6_3_5,
    unsigned **lookup_range_check_3_6_6_3_6,
    unsigned **lookup_range_check_3_6_6_3_7,
    unsigned **lookup_range_check_3_6_6_3_8,
    unsigned **lookup_range_check_3_6_6_3_9,
    unsigned **lookup_range_check_3_6_6_3_10,
    unsigned **lookup_range_check_3_6_6_3_11,
    unsigned **lookup_range_check_3_6_6_3_12,
    unsigned **lookup_range_check_3_6_6_3_13,
    unsigned **lookup_range_check_3_6_6_3_14,
    unsigned **lookup_range_check_3_6_6_3_15,
    unsigned **lookup_range_check_3_6_6_3_16,
    unsigned **lookup_range_check_3_6_6_3_17,
    unsigned **lookup_range_check_3_6_6_3_18,
    unsigned **lookup_range_check_3_6_6_3_19,
    unsigned **lookup_range_check_3_6_6_3_20,
    unsigned **lookup_range_check_3_6_6_3_21,
    unsigned **lookup_range_check_3_6_6_3_22,
    unsigned **lookup_range_check_3_6_6_3_23,
    unsigned **lookup_range_check_3_6_6_3_24,
    unsigned **lookup_range_check_3_6_6_3_25,
    unsigned **lookup_range_check_3_6_6_3_26,
    unsigned **lookup_range_check_3_6_6_3_27,
    unsigned **lookup_range_check_3_6_6_3_28,
    unsigned **lookup_range_check_3_6_6_3_29,
    unsigned **lookup_range_check_3_6_6_3_30,
    unsigned **lookup_range_check_3_6_6_3_31,
    unsigned **lookup_range_check_3_6_6_3_32,
    unsigned **lookup_range_check_3_6_6_3_33,
    unsigned **lookup_range_check_3_6_6_3_34,
    unsigned **lookup_range_check_3_6_6_3_35,
    unsigned **lookup_range_check_3_6_6_3_36,
    unsigned **lookup_range_check_3_6_6_3_37,
    unsigned **lookup_range_check_3_6_6_3_38,
    unsigned **lookup_range_check_3_6_6_3_39,

    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,
    unsigned **sub_component_inputs_range_check_12,
    unsigned **sub_component_inputs_range_check_18,
    unsigned **sub_component_inputs_range_check_3_6_6_3,

    unsigned segment_start,

    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    timer global_timer;
    global_timer.start("generate mul_mod_builtin traces");

    unsigned trace_size = 1 << log_size;
    unsigned num_blocks = (n_rows + MUL_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX - 1) / MUL_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX;

    // Copy pointer arrays to device memory
    m31 **device_traces = clone_to_device<m31*>((m31**)traces, MUL_MOD_BUILTIN_N_TRACE_COLUMNS);

    // Memory address to id lookups (2 elements each)
    m31 **device_lookup_addr2id_0 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_addr2id_1 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_addr2id_2 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_addr2id_3 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_addr2id_4 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_4, 2);
    m31 **device_lookup_addr2id_5 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_5, 2);
    m31 **device_lookup_addr2id_6 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_6, 2);
    m31 **device_lookup_addr2id_7 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_7, 2);
    m31 **device_lookup_addr2id_8 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_8, 2);
    m31 **device_lookup_addr2id_9 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_9, 2);
    m31 **device_lookup_addr2id_10 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_10, 2);
    m31 **device_lookup_addr2id_11 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_11, 2);
    m31 **device_lookup_addr2id_12 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_12, 2);
    m31 **device_lookup_addr2id_13 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_13, 2);
    m31 **device_lookup_addr2id_14 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_14, 2);
    m31 **device_lookup_addr2id_15 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_15, 2);
    m31 **device_lookup_addr2id_16 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_16, 2);
    m31 **device_lookup_addr2id_17 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_17, 2);
    m31 **device_lookup_addr2id_18 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_18, 2);
    m31 **device_lookup_addr2id_19 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_19, 2);
    m31 **device_lookup_addr2id_20 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_20, 2);
    m31 **device_lookup_addr2id_21 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_21, 2);
    m31 **device_lookup_addr2id_22 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_22, 2);
    m31 **device_lookup_addr2id_23 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_23, 2);
    m31 **device_lookup_addr2id_24 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_24, 2);
    m31 **device_lookup_addr2id_25 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_25, 2);
    m31 **device_lookup_addr2id_26 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_26, 2);
    m31 **device_lookup_addr2id_27 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_27, 2);
    m31 **device_lookup_addr2id_28 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_28, 2);

    // Memory id to big lookups (29 elements each)
    m31 **device_lookup_id2big_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_id2big_1 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_id2big_2 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_id2big_3 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_id2big_4 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_4, 29);
    m31 **device_lookup_id2big_5 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_5, 29);
    m31 **device_lookup_id2big_6 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_6, 29);
    m31 **device_lookup_id2big_7 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_7, 29);
    m31 **device_lookup_id2big_8 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_8, 29);
    m31 **device_lookup_id2big_9 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_9, 29);
    m31 **device_lookup_id2big_10 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_10, 29);
    m31 **device_lookup_id2big_11 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_11, 29);
    m31 **device_lookup_id2big_12 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_12, 29);
    m31 **device_lookup_id2big_13 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_13, 29);
    m31 **device_lookup_id2big_14 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_14, 29);
    m31 **device_lookup_id2big_15 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_15, 29);
    m31 **device_lookup_id2big_16 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_16, 29);
    m31 **device_lookup_id2big_17 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_17, 29);
    m31 **device_lookup_id2big_18 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_18, 29);
    m31 **device_lookup_id2big_19 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_19, 29);
    m31 **device_lookup_id2big_20 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_20, 29);
    m31 **device_lookup_id2big_21 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_21, 29);
    m31 **device_lookup_id2big_22 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_22, 29);
    m31 **device_lookup_id2big_23 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_23, 29);

    // RangeCheck_12 lookups (1 element each)
    m31 **device_lookup_rc12_0 = clone_to_device<m31*>((m31**)lookup_range_check_12_0, 1);
    m31 **device_lookup_rc12_1 = clone_to_device<m31*>((m31**)lookup_range_check_12_1, 1);
    m31 **device_lookup_rc12_2 = clone_to_device<m31*>((m31**)lookup_range_check_12_2, 1);
    m31 **device_lookup_rc12_3 = clone_to_device<m31*>((m31**)lookup_range_check_12_3, 1);
    m31 **device_lookup_rc12_4 = clone_to_device<m31*>((m31**)lookup_range_check_12_4, 1);
    m31 **device_lookup_rc12_5 = clone_to_device<m31*>((m31**)lookup_range_check_12_5, 1);
    m31 **device_lookup_rc12_6 = clone_to_device<m31*>((m31**)lookup_range_check_12_6, 1);
    m31 **device_lookup_rc12_7 = clone_to_device<m31*>((m31**)lookup_range_check_12_7, 1);
    m31 **device_lookup_rc12_8 = clone_to_device<m31*>((m31**)lookup_range_check_12_8, 1);
    m31 **device_lookup_rc12_9 = clone_to_device<m31*>((m31**)lookup_range_check_12_9, 1);
    m31 **device_lookup_rc12_10 = clone_to_device<m31*>((m31**)lookup_range_check_12_10, 1);
    m31 **device_lookup_rc12_11 = clone_to_device<m31*>((m31**)lookup_range_check_12_11, 1);
    m31 **device_lookup_rc12_12 = clone_to_device<m31*>((m31**)lookup_range_check_12_12, 1);
    m31 **device_lookup_rc12_13 = clone_to_device<m31*>((m31**)lookup_range_check_12_13, 1);
    m31 **device_lookup_rc12_14 = clone_to_device<m31*>((m31**)lookup_range_check_12_14, 1);
    m31 **device_lookup_rc12_15 = clone_to_device<m31*>((m31**)lookup_range_check_12_15, 1);
    m31 **device_lookup_rc12_16 = clone_to_device<m31*>((m31**)lookup_range_check_12_16, 1);
    m31 **device_lookup_rc12_17 = clone_to_device<m31*>((m31**)lookup_range_check_12_17, 1);
    m31 **device_lookup_rc12_18 = clone_to_device<m31*>((m31**)lookup_range_check_12_18, 1);
    m31 **device_lookup_rc12_19 = clone_to_device<m31*>((m31**)lookup_range_check_12_19, 1);
    m31 **device_lookup_rc12_20 = clone_to_device<m31*>((m31**)lookup_range_check_12_20, 1);
    m31 **device_lookup_rc12_21 = clone_to_device<m31*>((m31**)lookup_range_check_12_21, 1);
    m31 **device_lookup_rc12_22 = clone_to_device<m31*>((m31**)lookup_range_check_12_22, 1);
    m31 **device_lookup_rc12_23 = clone_to_device<m31*>((m31**)lookup_range_check_12_23, 1);
    m31 **device_lookup_rc12_24 = clone_to_device<m31*>((m31**)lookup_range_check_12_24, 1);
    m31 **device_lookup_rc12_25 = clone_to_device<m31*>((m31**)lookup_range_check_12_25, 1);
    m31 **device_lookup_rc12_26 = clone_to_device<m31*>((m31**)lookup_range_check_12_26, 1);
    m31 **device_lookup_rc12_27 = clone_to_device<m31*>((m31**)lookup_range_check_12_27, 1);
    m31 **device_lookup_rc12_28 = clone_to_device<m31*>((m31**)lookup_range_check_12_28, 1);
    m31 **device_lookup_rc12_29 = clone_to_device<m31*>((m31**)lookup_range_check_12_29, 1);
    m31 **device_lookup_rc12_30 = clone_to_device<m31*>((m31**)lookup_range_check_12_30, 1);
    m31 **device_lookup_rc12_31 = clone_to_device<m31*>((m31**)lookup_range_check_12_31, 1);

    // RangeCheck_18 lookups (1 element each)
    m31 **device_lookup_rc18_0 = clone_to_device<m31*>((m31**)lookup_range_check_18_0, 1);
    m31 **device_lookup_rc18_1 = clone_to_device<m31*>((m31**)lookup_range_check_18_1, 1);
    m31 **device_lookup_rc18_2 = clone_to_device<m31*>((m31**)lookup_range_check_18_2, 1);
    m31 **device_lookup_rc18_3 = clone_to_device<m31*>((m31**)lookup_range_check_18_3, 1);
    m31 **device_lookup_rc18_4 = clone_to_device<m31*>((m31**)lookup_range_check_18_4, 1);
    m31 **device_lookup_rc18_5 = clone_to_device<m31*>((m31**)lookup_range_check_18_5, 1);
    m31 **device_lookup_rc18_6 = clone_to_device<m31*>((m31**)lookup_range_check_18_6, 1);
    m31 **device_lookup_rc18_7 = clone_to_device<m31*>((m31**)lookup_range_check_18_7, 1);
    m31 **device_lookup_rc18_8 = clone_to_device<m31*>((m31**)lookup_range_check_18_8, 1);
    m31 **device_lookup_rc18_9 = clone_to_device<m31*>((m31**)lookup_range_check_18_9, 1);
    m31 **device_lookup_rc18_10 = clone_to_device<m31*>((m31**)lookup_range_check_18_10, 1);
    m31 **device_lookup_rc18_11 = clone_to_device<m31*>((m31**)lookup_range_check_18_11, 1);
    m31 **device_lookup_rc18_12 = clone_to_device<m31*>((m31**)lookup_range_check_18_12, 1);
    m31 **device_lookup_rc18_13 = clone_to_device<m31*>((m31**)lookup_range_check_18_13, 1);
    m31 **device_lookup_rc18_14 = clone_to_device<m31*>((m31**)lookup_range_check_18_14, 1);
    m31 **device_lookup_rc18_15 = clone_to_device<m31*>((m31**)lookup_range_check_18_15, 1);
    m31 **device_lookup_rc18_16 = clone_to_device<m31*>((m31**)lookup_range_check_18_16, 1);
    m31 **device_lookup_rc18_17 = clone_to_device<m31*>((m31**)lookup_range_check_18_17, 1);
    m31 **device_lookup_rc18_18 = clone_to_device<m31*>((m31**)lookup_range_check_18_18, 1);
    m31 **device_lookup_rc18_19 = clone_to_device<m31*>((m31**)lookup_range_check_18_19, 1);
    m31 **device_lookup_rc18_20 = clone_to_device<m31*>((m31**)lookup_range_check_18_20, 1);
    m31 **device_lookup_rc18_21 = clone_to_device<m31*>((m31**)lookup_range_check_18_21, 1);
    m31 **device_lookup_rc18_22 = clone_to_device<m31*>((m31**)lookup_range_check_18_22, 1);
    m31 **device_lookup_rc18_23 = clone_to_device<m31*>((m31**)lookup_range_check_18_23, 1);
    m31 **device_lookup_rc18_24 = clone_to_device<m31*>((m31**)lookup_range_check_18_24, 1);
    m31 **device_lookup_rc18_25 = clone_to_device<m31*>((m31**)lookup_range_check_18_25, 1);
    m31 **device_lookup_rc18_26 = clone_to_device<m31*>((m31**)lookup_range_check_18_26, 1);
    m31 **device_lookup_rc18_27 = clone_to_device<m31*>((m31**)lookup_range_check_18_27, 1);
    m31 **device_lookup_rc18_28 = clone_to_device<m31*>((m31**)lookup_range_check_18_28, 1);
    m31 **device_lookup_rc18_29 = clone_to_device<m31*>((m31**)lookup_range_check_18_29, 1);
    m31 **device_lookup_rc18_30 = clone_to_device<m31*>((m31**)lookup_range_check_18_30, 1);
    m31 **device_lookup_rc18_31 = clone_to_device<m31*>((m31**)lookup_range_check_18_31, 1);
    m31 **device_lookup_rc18_32 = clone_to_device<m31*>((m31**)lookup_range_check_18_32, 1);
    m31 **device_lookup_rc18_33 = clone_to_device<m31*>((m31**)lookup_range_check_18_33, 1);
    m31 **device_lookup_rc18_34 = clone_to_device<m31*>((m31**)lookup_range_check_18_34, 1);
    m31 **device_lookup_rc18_35 = clone_to_device<m31*>((m31**)lookup_range_check_18_35, 1);
    m31 **device_lookup_rc18_36 = clone_to_device<m31*>((m31**)lookup_range_check_18_36, 1);
    m31 **device_lookup_rc18_37 = clone_to_device<m31*>((m31**)lookup_range_check_18_37, 1);
    m31 **device_lookup_rc18_38 = clone_to_device<m31*>((m31**)lookup_range_check_18_38, 1);
    m31 **device_lookup_rc18_39 = clone_to_device<m31*>((m31**)lookup_range_check_18_39, 1);
    m31 **device_lookup_rc18_40 = clone_to_device<m31*>((m31**)lookup_range_check_18_40, 1);
    m31 **device_lookup_rc18_41 = clone_to_device<m31*>((m31**)lookup_range_check_18_41, 1);
    m31 **device_lookup_rc18_42 = clone_to_device<m31*>((m31**)lookup_range_check_18_42, 1);
    m31 **device_lookup_rc18_43 = clone_to_device<m31*>((m31**)lookup_range_check_18_43, 1);
    m31 **device_lookup_rc18_44 = clone_to_device<m31*>((m31**)lookup_range_check_18_44, 1);
    m31 **device_lookup_rc18_45 = clone_to_device<m31*>((m31**)lookup_range_check_18_45, 1);
    m31 **device_lookup_rc18_46 = clone_to_device<m31*>((m31**)lookup_range_check_18_46, 1);
    m31 **device_lookup_rc18_47 = clone_to_device<m31*>((m31**)lookup_range_check_18_47, 1);
    m31 **device_lookup_rc18_48 = clone_to_device<m31*>((m31**)lookup_range_check_18_48, 1);
    m31 **device_lookup_rc18_49 = clone_to_device<m31*>((m31**)lookup_range_check_18_49, 1);
    m31 **device_lookup_rc18_50 = clone_to_device<m31*>((m31**)lookup_range_check_18_50, 1);
    m31 **device_lookup_rc18_51 = clone_to_device<m31*>((m31**)lookup_range_check_18_51, 1);
    m31 **device_lookup_rc18_52 = clone_to_device<m31*>((m31**)lookup_range_check_18_52, 1);
    m31 **device_lookup_rc18_53 = clone_to_device<m31*>((m31**)lookup_range_check_18_53, 1);
    m31 **device_lookup_rc18_54 = clone_to_device<m31*>((m31**)lookup_range_check_18_54, 1);
    m31 **device_lookup_rc18_55 = clone_to_device<m31*>((m31**)lookup_range_check_18_55, 1);
    m31 **device_lookup_rc18_56 = clone_to_device<m31*>((m31**)lookup_range_check_18_56, 1);
    m31 **device_lookup_rc18_57 = clone_to_device<m31*>((m31**)lookup_range_check_18_57, 1);
    m31 **device_lookup_rc18_58 = clone_to_device<m31*>((m31**)lookup_range_check_18_58, 1);
    m31 **device_lookup_rc18_59 = clone_to_device<m31*>((m31**)lookup_range_check_18_59, 1);
    m31 **device_lookup_rc18_60 = clone_to_device<m31*>((m31**)lookup_range_check_18_60, 1);
    m31 **device_lookup_rc18_61 = clone_to_device<m31*>((m31**)lookup_range_check_18_61, 1);

    // RangeCheck_3_6_6_3 lookups (4 elements each)
    m31 **device_lookup_rc3663_0 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_0, 4);
    m31 **device_lookup_rc3663_1 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_1, 4);
    m31 **device_lookup_rc3663_2 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_2, 4);
    m31 **device_lookup_rc3663_3 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_3, 4);
    m31 **device_lookup_rc3663_4 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_4, 4);
    m31 **device_lookup_rc3663_5 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_5, 4);
    m31 **device_lookup_rc3663_6 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_6, 4);
    m31 **device_lookup_rc3663_7 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_7, 4);
    m31 **device_lookup_rc3663_8 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_8, 4);
    m31 **device_lookup_rc3663_9 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_9, 4);
    m31 **device_lookup_rc3663_10 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_10, 4);
    m31 **device_lookup_rc3663_11 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_11, 4);
    m31 **device_lookup_rc3663_12 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_12, 4);
    m31 **device_lookup_rc3663_13 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_13, 4);
    m31 **device_lookup_rc3663_14 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_14, 4);
    m31 **device_lookup_rc3663_15 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_15, 4);
    m31 **device_lookup_rc3663_16 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_16, 4);
    m31 **device_lookup_rc3663_17 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_17, 4);
    m31 **device_lookup_rc3663_18 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_18, 4);
    m31 **device_lookup_rc3663_19 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_19, 4);
    m31 **device_lookup_rc3663_20 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_20, 4);
    m31 **device_lookup_rc3663_21 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_21, 4);
    m31 **device_lookup_rc3663_22 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_22, 4);
    m31 **device_lookup_rc3663_23 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_23, 4);
    m31 **device_lookup_rc3663_24 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_24, 4);
    m31 **device_lookup_rc3663_25 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_25, 4);
    m31 **device_lookup_rc3663_26 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_26, 4);
    m31 **device_lookup_rc3663_27 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_27, 4);
    m31 **device_lookup_rc3663_28 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_28, 4);
    m31 **device_lookup_rc3663_29 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_29, 4);
    m31 **device_lookup_rc3663_30 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_30, 4);
    m31 **device_lookup_rc3663_31 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_31, 4);
    m31 **device_lookup_rc3663_32 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_32, 4);
    m31 **device_lookup_rc3663_33 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_33, 4);
    m31 **device_lookup_rc3663_34 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_34, 4);
    m31 **device_lookup_rc3663_35 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_35, 4);
    m31 **device_lookup_rc3663_36 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_36, 4);
    m31 **device_lookup_rc3663_37 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_37, 4);
    m31 **device_lookup_rc3663_38 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_38, 4);
    m31 **device_lookup_rc3663_39 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_39, 4);

    // Sub-component inputs
    m31 **device_sub_addr2id = clone_to_device<m31*>((m31**)sub_component_inputs_memory_address_to_id, 29);
    m31 **device_sub_id2big = clone_to_device<m31*>((m31**)sub_component_inputs_memory_id_to_big, 24);

    // Memory id to big transposed values
    m31 **device_id2big_transposed = clone_to_device<m31*>((m31**)memory_id_to_big_transposed_big_values, N_M31_IN_FELT252);

    generate_mul_mod_builtin_trace_kernel<<<num_blocks, MUL_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX>>>(
        device_traces,
        // 29 MemoryAddressToId lookups
        device_lookup_addr2id_0, device_lookup_addr2id_1,
        device_lookup_addr2id_2, device_lookup_addr2id_3,
        device_lookup_addr2id_4, device_lookup_addr2id_5,
        device_lookup_addr2id_6, device_lookup_addr2id_7,
        device_lookup_addr2id_8, device_lookup_addr2id_9,
        device_lookup_addr2id_10, device_lookup_addr2id_11,
        device_lookup_addr2id_12, device_lookup_addr2id_13,
        device_lookup_addr2id_14, device_lookup_addr2id_15,
        device_lookup_addr2id_16, device_lookup_addr2id_17,
        device_lookup_addr2id_18, device_lookup_addr2id_19,
        device_lookup_addr2id_20, device_lookup_addr2id_21,
        device_lookup_addr2id_22, device_lookup_addr2id_23,
        device_lookup_addr2id_24, device_lookup_addr2id_25,
        device_lookup_addr2id_26, device_lookup_addr2id_27,
        device_lookup_addr2id_28,
        // 24 MemoryIdToBig lookups
        device_lookup_id2big_0, device_lookup_id2big_1,
        device_lookup_id2big_2, device_lookup_id2big_3,
        device_lookup_id2big_4, device_lookup_id2big_5,
        device_lookup_id2big_6, device_lookup_id2big_7,
        device_lookup_id2big_8, device_lookup_id2big_9,
        device_lookup_id2big_10, device_lookup_id2big_11,
        device_lookup_id2big_12, device_lookup_id2big_13,
        device_lookup_id2big_14, device_lookup_id2big_15,
        device_lookup_id2big_16, device_lookup_id2big_17,
        device_lookup_id2big_18, device_lookup_id2big_19,
        device_lookup_id2big_20, device_lookup_id2big_21,
        device_lookup_id2big_22, device_lookup_id2big_23,
        // 32 RangeCheck_12 lookups
        device_lookup_rc12_0, device_lookup_rc12_1,
        device_lookup_rc12_2, device_lookup_rc12_3,
        device_lookup_rc12_4, device_lookup_rc12_5,
        device_lookup_rc12_6, device_lookup_rc12_7,
        device_lookup_rc12_8, device_lookup_rc12_9,
        device_lookup_rc12_10, device_lookup_rc12_11,
        device_lookup_rc12_12, device_lookup_rc12_13,
        device_lookup_rc12_14, device_lookup_rc12_15,
        device_lookup_rc12_16, device_lookup_rc12_17,
        device_lookup_rc12_18, device_lookup_rc12_19,
        device_lookup_rc12_20, device_lookup_rc12_21,
        device_lookup_rc12_22, device_lookup_rc12_23,
        device_lookup_rc12_24, device_lookup_rc12_25,
        device_lookup_rc12_26, device_lookup_rc12_27,
        device_lookup_rc12_28, device_lookup_rc12_29,
        device_lookup_rc12_30, device_lookup_rc12_31,
        // 62 RangeCheck_18 lookups
        device_lookup_rc18_0, device_lookup_rc18_1,
        device_lookup_rc18_2, device_lookup_rc18_3,
        device_lookup_rc18_4, device_lookup_rc18_5,
        device_lookup_rc18_6, device_lookup_rc18_7,
        device_lookup_rc18_8, device_lookup_rc18_9,
        device_lookup_rc18_10, device_lookup_rc18_11,
        device_lookup_rc18_12, device_lookup_rc18_13,
        device_lookup_rc18_14, device_lookup_rc18_15,
        device_lookup_rc18_16, device_lookup_rc18_17,
        device_lookup_rc18_18, device_lookup_rc18_19,
        device_lookup_rc18_20, device_lookup_rc18_21,
        device_lookup_rc18_22, device_lookup_rc18_23,
        device_lookup_rc18_24, device_lookup_rc18_25,
        device_lookup_rc18_26, device_lookup_rc18_27,
        device_lookup_rc18_28, device_lookup_rc18_29,
        device_lookup_rc18_30, device_lookup_rc18_31,
        device_lookup_rc18_32, device_lookup_rc18_33,
        device_lookup_rc18_34, device_lookup_rc18_35,
        device_lookup_rc18_36, device_lookup_rc18_37,
        device_lookup_rc18_38, device_lookup_rc18_39,
        device_lookup_rc18_40, device_lookup_rc18_41,
        device_lookup_rc18_42, device_lookup_rc18_43,
        device_lookup_rc18_44, device_lookup_rc18_45,
        device_lookup_rc18_46, device_lookup_rc18_47,
        device_lookup_rc18_48, device_lookup_rc18_49,
        device_lookup_rc18_50, device_lookup_rc18_51,
        device_lookup_rc18_52, device_lookup_rc18_53,
        device_lookup_rc18_54, device_lookup_rc18_55,
        device_lookup_rc18_56, device_lookup_rc18_57,
        device_lookup_rc18_58, device_lookup_rc18_59,
        device_lookup_rc18_60, device_lookup_rc18_61,
        // 40 RangeCheck_3_6_6_3 lookups
        device_lookup_rc3663_0, device_lookup_rc3663_1,
        device_lookup_rc3663_2, device_lookup_rc3663_3,
        device_lookup_rc3663_4, device_lookup_rc3663_5,
        device_lookup_rc3663_6, device_lookup_rc3663_7,
        device_lookup_rc3663_8, device_lookup_rc3663_9,
        device_lookup_rc3663_10, device_lookup_rc3663_11,
        device_lookup_rc3663_12, device_lookup_rc3663_13,
        device_lookup_rc3663_14, device_lookup_rc3663_15,
        device_lookup_rc3663_16, device_lookup_rc3663_17,
        device_lookup_rc3663_18, device_lookup_rc3663_19,
        device_lookup_rc3663_20, device_lookup_rc3663_21,
        device_lookup_rc3663_22, device_lookup_rc3663_23,
        device_lookup_rc3663_24, device_lookup_rc3663_25,
        device_lookup_rc3663_26, device_lookup_rc3663_27,
        device_lookup_rc3663_28, device_lookup_rc3663_29,
        device_lookup_rc3663_30, device_lookup_rc3663_31,
        device_lookup_rc3663_32, device_lookup_rc3663_33,
        device_lookup_rc3663_34, device_lookup_rc3663_35,
        device_lookup_rc3663_36, device_lookup_rc3663_37,
        device_lookup_rc3663_38, device_lookup_rc3663_39,
        // Sub-component inputs
        device_sub_addr2id,
        device_sub_id2big,
        (m31**)sub_component_inputs_range_check_12,
        (m31**)sub_component_inputs_range_check_18,
        (m31**)sub_component_inputs_range_check_3_6_6_3,
        // Segment info
        segment_start,
        // Memory data
        memory_address_to_id_address_to_raw_id,
        device_id2big_transposed,
        memory_id_to_big_small_values,
        // Sizes
        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("generate mul_mod_builtin traces");
}

extern "C"
void generate_mul_mod_builtin_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,
    void *range_check_12,
    void *range_check_18,
    void *range_check_3_6_6_3,

    // Lookup data arrays - 29 MemoryAddressToId lookups
    unsigned **lookup_memory_address_to_id_0,
    unsigned **lookup_memory_address_to_id_1,
    unsigned **lookup_memory_address_to_id_2,
    unsigned **lookup_memory_address_to_id_3,
    unsigned **lookup_memory_address_to_id_4,
    unsigned **lookup_memory_address_to_id_5,
    unsigned **lookup_memory_address_to_id_6,
    unsigned **lookup_memory_address_to_id_7,
    unsigned **lookup_memory_address_to_id_8,
    unsigned **lookup_memory_address_to_id_9,
    unsigned **lookup_memory_address_to_id_10,
    unsigned **lookup_memory_address_to_id_11,
    unsigned **lookup_memory_address_to_id_12,
    unsigned **lookup_memory_address_to_id_13,
    unsigned **lookup_memory_address_to_id_14,
    unsigned **lookup_memory_address_to_id_15,
    unsigned **lookup_memory_address_to_id_16,
    unsigned **lookup_memory_address_to_id_17,
    unsigned **lookup_memory_address_to_id_18,
    unsigned **lookup_memory_address_to_id_19,
    unsigned **lookup_memory_address_to_id_20,
    unsigned **lookup_memory_address_to_id_21,
    unsigned **lookup_memory_address_to_id_22,
    unsigned **lookup_memory_address_to_id_23,
    unsigned **lookup_memory_address_to_id_24,
    unsigned **lookup_memory_address_to_id_25,
    unsigned **lookup_memory_address_to_id_26,
    unsigned **lookup_memory_address_to_id_27,
    unsigned **lookup_memory_address_to_id_28,

    // Lookup data arrays - 24 MemoryIdToBig lookups
    unsigned **lookup_memory_id_to_big_0,
    unsigned **lookup_memory_id_to_big_1,
    unsigned **lookup_memory_id_to_big_2,
    unsigned **lookup_memory_id_to_big_3,
    unsigned **lookup_memory_id_to_big_4,
    unsigned **lookup_memory_id_to_big_5,
    unsigned **lookup_memory_id_to_big_6,
    unsigned **lookup_memory_id_to_big_7,
    unsigned **lookup_memory_id_to_big_8,
    unsigned **lookup_memory_id_to_big_9,
    unsigned **lookup_memory_id_to_big_10,
    unsigned **lookup_memory_id_to_big_11,
    unsigned **lookup_memory_id_to_big_12,
    unsigned **lookup_memory_id_to_big_13,
    unsigned **lookup_memory_id_to_big_14,
    unsigned **lookup_memory_id_to_big_15,
    unsigned **lookup_memory_id_to_big_16,
    unsigned **lookup_memory_id_to_big_17,
    unsigned **lookup_memory_id_to_big_18,
    unsigned **lookup_memory_id_to_big_19,
    unsigned **lookup_memory_id_to_big_20,
    unsigned **lookup_memory_id_to_big_21,
    unsigned **lookup_memory_id_to_big_22,
    unsigned **lookup_memory_id_to_big_23,

    // Lookup data arrays - 32 RangeCheck_12 lookups
    unsigned **lookup_range_check_12_0,
    unsigned **lookup_range_check_12_1,
    unsigned **lookup_range_check_12_2,
    unsigned **lookup_range_check_12_3,
    unsigned **lookup_range_check_12_4,
    unsigned **lookup_range_check_12_5,
    unsigned **lookup_range_check_12_6,
    unsigned **lookup_range_check_12_7,
    unsigned **lookup_range_check_12_8,
    unsigned **lookup_range_check_12_9,
    unsigned **lookup_range_check_12_10,
    unsigned **lookup_range_check_12_11,
    unsigned **lookup_range_check_12_12,
    unsigned **lookup_range_check_12_13,
    unsigned **lookup_range_check_12_14,
    unsigned **lookup_range_check_12_15,
    unsigned **lookup_range_check_12_16,
    unsigned **lookup_range_check_12_17,
    unsigned **lookup_range_check_12_18,
    unsigned **lookup_range_check_12_19,
    unsigned **lookup_range_check_12_20,
    unsigned **lookup_range_check_12_21,
    unsigned **lookup_range_check_12_22,
    unsigned **lookup_range_check_12_23,
    unsigned **lookup_range_check_12_24,
    unsigned **lookup_range_check_12_25,
    unsigned **lookup_range_check_12_26,
    unsigned **lookup_range_check_12_27,
    unsigned **lookup_range_check_12_28,
    unsigned **lookup_range_check_12_29,
    unsigned **lookup_range_check_12_30,
    unsigned **lookup_range_check_12_31,

    // Lookup data arrays - 62 RangeCheck_18 lookups
    unsigned **lookup_range_check_18_0,
    unsigned **lookup_range_check_18_1,
    unsigned **lookup_range_check_18_2,
    unsigned **lookup_range_check_18_3,
    unsigned **lookup_range_check_18_4,
    unsigned **lookup_range_check_18_5,
    unsigned **lookup_range_check_18_6,
    unsigned **lookup_range_check_18_7,
    unsigned **lookup_range_check_18_8,
    unsigned **lookup_range_check_18_9,
    unsigned **lookup_range_check_18_10,
    unsigned **lookup_range_check_18_11,
    unsigned **lookup_range_check_18_12,
    unsigned **lookup_range_check_18_13,
    unsigned **lookup_range_check_18_14,
    unsigned **lookup_range_check_18_15,
    unsigned **lookup_range_check_18_16,
    unsigned **lookup_range_check_18_17,
    unsigned **lookup_range_check_18_18,
    unsigned **lookup_range_check_18_19,
    unsigned **lookup_range_check_18_20,
    unsigned **lookup_range_check_18_21,
    unsigned **lookup_range_check_18_22,
    unsigned **lookup_range_check_18_23,
    unsigned **lookup_range_check_18_24,
    unsigned **lookup_range_check_18_25,
    unsigned **lookup_range_check_18_26,
    unsigned **lookup_range_check_18_27,
    unsigned **lookup_range_check_18_28,
    unsigned **lookup_range_check_18_29,
    unsigned **lookup_range_check_18_30,
    unsigned **lookup_range_check_18_31,
    unsigned **lookup_range_check_18_32,
    unsigned **lookup_range_check_18_33,
    unsigned **lookup_range_check_18_34,
    unsigned **lookup_range_check_18_35,
    unsigned **lookup_range_check_18_36,
    unsigned **lookup_range_check_18_37,
    unsigned **lookup_range_check_18_38,
    unsigned **lookup_range_check_18_39,
    unsigned **lookup_range_check_18_40,
    unsigned **lookup_range_check_18_41,
    unsigned **lookup_range_check_18_42,
    unsigned **lookup_range_check_18_43,
    unsigned **lookup_range_check_18_44,
    unsigned **lookup_range_check_18_45,
    unsigned **lookup_range_check_18_46,
    unsigned **lookup_range_check_18_47,
    unsigned **lookup_range_check_18_48,
    unsigned **lookup_range_check_18_49,
    unsigned **lookup_range_check_18_50,
    unsigned **lookup_range_check_18_51,
    unsigned **lookup_range_check_18_52,
    unsigned **lookup_range_check_18_53,
    unsigned **lookup_range_check_18_54,
    unsigned **lookup_range_check_18_55,
    unsigned **lookup_range_check_18_56,
    unsigned **lookup_range_check_18_57,
    unsigned **lookup_range_check_18_58,
    unsigned **lookup_range_check_18_59,
    unsigned **lookup_range_check_18_60,
    unsigned **lookup_range_check_18_61,

    // Lookup data arrays - 40 RangeCheck_3_6_6_3 lookups
    unsigned **lookup_range_check_3_6_6_3_0,
    unsigned **lookup_range_check_3_6_6_3_1,
    unsigned **lookup_range_check_3_6_6_3_2,
    unsigned **lookup_range_check_3_6_6_3_3,
    unsigned **lookup_range_check_3_6_6_3_4,
    unsigned **lookup_range_check_3_6_6_3_5,
    unsigned **lookup_range_check_3_6_6_3_6,
    unsigned **lookup_range_check_3_6_6_3_7,
    unsigned **lookup_range_check_3_6_6_3_8,
    unsigned **lookup_range_check_3_6_6_3_9,
    unsigned **lookup_range_check_3_6_6_3_10,
    unsigned **lookup_range_check_3_6_6_3_11,
    unsigned **lookup_range_check_3_6_6_3_12,
    unsigned **lookup_range_check_3_6_6_3_13,
    unsigned **lookup_range_check_3_6_6_3_14,
    unsigned **lookup_range_check_3_6_6_3_15,
    unsigned **lookup_range_check_3_6_6_3_16,
    unsigned **lookup_range_check_3_6_6_3_17,
    unsigned **lookup_range_check_3_6_6_3_18,
    unsigned **lookup_range_check_3_6_6_3_19,
    unsigned **lookup_range_check_3_6_6_3_20,
    unsigned **lookup_range_check_3_6_6_3_21,
    unsigned **lookup_range_check_3_6_6_3_22,
    unsigned **lookup_range_check_3_6_6_3_23,
    unsigned **lookup_range_check_3_6_6_3_24,
    unsigned **lookup_range_check_3_6_6_3_25,
    unsigned **lookup_range_check_3_6_6_3_26,
    unsigned **lookup_range_check_3_6_6_3_27,
    unsigned **lookup_range_check_3_6_6_3_28,
    unsigned **lookup_range_check_3_6_6_3_29,
    unsigned **lookup_range_check_3_6_6_3_30,
    unsigned **lookup_range_check_3_6_6_3_31,
    unsigned **lookup_range_check_3_6_6_3_32,
    unsigned **lookup_range_check_3_6_6_3_33,
    unsigned **lookup_range_check_3_6_6_3_34,
    unsigned **lookup_range_check_3_6_6_3_35,
    unsigned **lookup_range_check_3_6_6_3_36,
    unsigned **lookup_range_check_3_6_6_3_37,
    unsigned **lookup_range_check_3_6_6_3_38,
    unsigned **lookup_range_check_3_6_6_3_39,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate mul_mod_builtin interaction trace");
    unsigned trace_size = 1 << log_size;

    int block_dim_val = trace_size < MUL_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : MUL_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX;
    int num_blocks = block_dim_val < MUL_MOD_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim_val - 1) / block_dim_val;

    // Allocate temporary storage
    qm31 *denom_ptr = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    // Clone lookup elements to device
    LookupElementsBasic<2> *mem_addr_to_id_host = (LookupElementsBasic<2>*)memory_address_to_id;
    LookupElementsBasic<29> *mem_id_to_big_host = (LookupElementsBasic<29>*)memory_id_to_big;
    LookupElementsBasic<1> *range_check_12_host = (LookupElementsBasic<1>*)range_check_12;
    LookupElementsBasic<1> *range_check_18_host = (LookupElementsBasic<1>*)range_check_18;
    LookupElementsBasic<4> *range_check_3_6_6_3_host = (LookupElementsBasic<4>*)range_check_3_6_6_3;

    LookupElementsBasic<2> *device_mem_addr_to_id = cuda_malloc<LookupElementsBasic<2>>(1);
    LookupElementsBasic<29> *device_mem_id_to_big = cuda_malloc<LookupElementsBasic<29>>(1);
    LookupElementsBasic<1> *device_range_check_12 = cuda_malloc<LookupElementsBasic<1>>(1);
    LookupElementsBasic<1> *device_range_check_18 = cuda_malloc<LookupElementsBasic<1>>(1);
    LookupElementsBasic<4> *device_range_check_3_6_6_3 = cuda_malloc<LookupElementsBasic<4>>(1);

    cuda_mem_copy_host_to_device<LookupElementsBasic<2>>(mem_addr_to_id_host, device_mem_addr_to_id, 1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<29>>(mem_id_to_big_host, device_mem_id_to_big, 1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<1>>(range_check_12_host, device_range_check_12, 1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<1>>(range_check_18_host, device_range_check_18, 1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>(range_check_3_6_6_3_host, device_range_check_3_6_6_3, 1);

    m31 **device_interaction_traces = clone_to_device<m31*>((m31**)interaction_trace, 4 * MUL_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS);

    // Clone all lookup arrays to device (29 memory_address_to_id)
    m31 **device_lookup_memory_address_to_id_0 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_memory_address_to_id_1 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_1, 2);
    m31 **device_lookup_memory_address_to_id_2 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_2, 2);
    m31 **device_lookup_memory_address_to_id_3 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_3, 2);
    m31 **device_lookup_memory_address_to_id_4 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_4, 2);
    m31 **device_lookup_memory_address_to_id_5 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_5, 2);
    m31 **device_lookup_memory_address_to_id_6 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_6, 2);
    m31 **device_lookup_memory_address_to_id_7 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_7, 2);
    m31 **device_lookup_memory_address_to_id_8 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_8, 2);
    m31 **device_lookup_memory_address_to_id_9 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_9, 2);
    m31 **device_lookup_memory_address_to_id_10 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_10, 2);
    m31 **device_lookup_memory_address_to_id_11 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_11, 2);
    m31 **device_lookup_memory_address_to_id_12 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_12, 2);
    m31 **device_lookup_memory_address_to_id_13 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_13, 2);
    m31 **device_lookup_memory_address_to_id_14 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_14, 2);
    m31 **device_lookup_memory_address_to_id_15 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_15, 2);
    m31 **device_lookup_memory_address_to_id_16 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_16, 2);
    m31 **device_lookup_memory_address_to_id_17 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_17, 2);
    m31 **device_lookup_memory_address_to_id_18 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_18, 2);
    m31 **device_lookup_memory_address_to_id_19 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_19, 2);
    m31 **device_lookup_memory_address_to_id_20 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_20, 2);
    m31 **device_lookup_memory_address_to_id_21 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_21, 2);
    m31 **device_lookup_memory_address_to_id_22 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_22, 2);
    m31 **device_lookup_memory_address_to_id_23 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_23, 2);
    m31 **device_lookup_memory_address_to_id_24 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_24, 2);
    m31 **device_lookup_memory_address_to_id_25 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_25, 2);
    m31 **device_lookup_memory_address_to_id_26 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_26, 2);
    m31 **device_lookup_memory_address_to_id_27 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_27, 2);
    m31 **device_lookup_memory_address_to_id_28 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_28, 2);

    // Clone 24 memory_id_to_big lookups
    m31 **device_lookup_memory_id_to_big_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_memory_id_to_big_1 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_1, 29);
    m31 **device_lookup_memory_id_to_big_2 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_2, 29);
    m31 **device_lookup_memory_id_to_big_3 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_3, 29);
    m31 **device_lookup_memory_id_to_big_4 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_4, 29);
    m31 **device_lookup_memory_id_to_big_5 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_5, 29);
    m31 **device_lookup_memory_id_to_big_6 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_6, 29);
    m31 **device_lookup_memory_id_to_big_7 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_7, 29);
    m31 **device_lookup_memory_id_to_big_8 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_8, 29);
    m31 **device_lookup_memory_id_to_big_9 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_9, 29);
    m31 **device_lookup_memory_id_to_big_10 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_10, 29);
    m31 **device_lookup_memory_id_to_big_11 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_11, 29);
    m31 **device_lookup_memory_id_to_big_12 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_12, 29);
    m31 **device_lookup_memory_id_to_big_13 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_13, 29);
    m31 **device_lookup_memory_id_to_big_14 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_14, 29);
    m31 **device_lookup_memory_id_to_big_15 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_15, 29);
    m31 **device_lookup_memory_id_to_big_16 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_16, 29);
    m31 **device_lookup_memory_id_to_big_17 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_17, 29);
    m31 **device_lookup_memory_id_to_big_18 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_18, 29);
    m31 **device_lookup_memory_id_to_big_19 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_19, 29);
    m31 **device_lookup_memory_id_to_big_20 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_20, 29);
    m31 **device_lookup_memory_id_to_big_21 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_21, 29);
    m31 **device_lookup_memory_id_to_big_22 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_22, 29);
    m31 **device_lookup_memory_id_to_big_23 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_23, 29);

    // Clone 32 range_check_12 lookups
    m31 **device_lookup_range_check_12_0 = clone_to_device<m31*>((m31**)lookup_range_check_12_0, 1);
    m31 **device_lookup_range_check_12_1 = clone_to_device<m31*>((m31**)lookup_range_check_12_1, 1);
    m31 **device_lookup_range_check_12_2 = clone_to_device<m31*>((m31**)lookup_range_check_12_2, 1);
    m31 **device_lookup_range_check_12_3 = clone_to_device<m31*>((m31**)lookup_range_check_12_3, 1);
    m31 **device_lookup_range_check_12_4 = clone_to_device<m31*>((m31**)lookup_range_check_12_4, 1);
    m31 **device_lookup_range_check_12_5 = clone_to_device<m31*>((m31**)lookup_range_check_12_5, 1);
    m31 **device_lookup_range_check_12_6 = clone_to_device<m31*>((m31**)lookup_range_check_12_6, 1);
    m31 **device_lookup_range_check_12_7 = clone_to_device<m31*>((m31**)lookup_range_check_12_7, 1);
    m31 **device_lookup_range_check_12_8 = clone_to_device<m31*>((m31**)lookup_range_check_12_8, 1);
    m31 **device_lookup_range_check_12_9 = clone_to_device<m31*>((m31**)lookup_range_check_12_9, 1);
    m31 **device_lookup_range_check_12_10 = clone_to_device<m31*>((m31**)lookup_range_check_12_10, 1);
    m31 **device_lookup_range_check_12_11 = clone_to_device<m31*>((m31**)lookup_range_check_12_11, 1);
    m31 **device_lookup_range_check_12_12 = clone_to_device<m31*>((m31**)lookup_range_check_12_12, 1);
    m31 **device_lookup_range_check_12_13 = clone_to_device<m31*>((m31**)lookup_range_check_12_13, 1);
    m31 **device_lookup_range_check_12_14 = clone_to_device<m31*>((m31**)lookup_range_check_12_14, 1);
    m31 **device_lookup_range_check_12_15 = clone_to_device<m31*>((m31**)lookup_range_check_12_15, 1);
    m31 **device_lookup_range_check_12_16 = clone_to_device<m31*>((m31**)lookup_range_check_12_16, 1);
    m31 **device_lookup_range_check_12_17 = clone_to_device<m31*>((m31**)lookup_range_check_12_17, 1);
    m31 **device_lookup_range_check_12_18 = clone_to_device<m31*>((m31**)lookup_range_check_12_18, 1);
    m31 **device_lookup_range_check_12_19 = clone_to_device<m31*>((m31**)lookup_range_check_12_19, 1);
    m31 **device_lookup_range_check_12_20 = clone_to_device<m31*>((m31**)lookup_range_check_12_20, 1);
    m31 **device_lookup_range_check_12_21 = clone_to_device<m31*>((m31**)lookup_range_check_12_21, 1);
    m31 **device_lookup_range_check_12_22 = clone_to_device<m31*>((m31**)lookup_range_check_12_22, 1);
    m31 **device_lookup_range_check_12_23 = clone_to_device<m31*>((m31**)lookup_range_check_12_23, 1);
    m31 **device_lookup_range_check_12_24 = clone_to_device<m31*>((m31**)lookup_range_check_12_24, 1);
    m31 **device_lookup_range_check_12_25 = clone_to_device<m31*>((m31**)lookup_range_check_12_25, 1);
    m31 **device_lookup_range_check_12_26 = clone_to_device<m31*>((m31**)lookup_range_check_12_26, 1);
    m31 **device_lookup_range_check_12_27 = clone_to_device<m31*>((m31**)lookup_range_check_12_27, 1);
    m31 **device_lookup_range_check_12_28 = clone_to_device<m31*>((m31**)lookup_range_check_12_28, 1);
    m31 **device_lookup_range_check_12_29 = clone_to_device<m31*>((m31**)lookup_range_check_12_29, 1);
    m31 **device_lookup_range_check_12_30 = clone_to_device<m31*>((m31**)lookup_range_check_12_30, 1);
    m31 **device_lookup_range_check_12_31 = clone_to_device<m31*>((m31**)lookup_range_check_12_31, 1);

    // Clone 62 range_check_18 lookups
    m31 **device_lookup_range_check_18_0 = clone_to_device<m31*>((m31**)lookup_range_check_18_0, 1);
    m31 **device_lookup_range_check_18_1 = clone_to_device<m31*>((m31**)lookup_range_check_18_1, 1);
    m31 **device_lookup_range_check_18_2 = clone_to_device<m31*>((m31**)lookup_range_check_18_2, 1);
    m31 **device_lookup_range_check_18_3 = clone_to_device<m31*>((m31**)lookup_range_check_18_3, 1);
    m31 **device_lookup_range_check_18_4 = clone_to_device<m31*>((m31**)lookup_range_check_18_4, 1);
    m31 **device_lookup_range_check_18_5 = clone_to_device<m31*>((m31**)lookup_range_check_18_5, 1);
    m31 **device_lookup_range_check_18_6 = clone_to_device<m31*>((m31**)lookup_range_check_18_6, 1);
    m31 **device_lookup_range_check_18_7 = clone_to_device<m31*>((m31**)lookup_range_check_18_7, 1);
    m31 **device_lookup_range_check_18_8 = clone_to_device<m31*>((m31**)lookup_range_check_18_8, 1);
    m31 **device_lookup_range_check_18_9 = clone_to_device<m31*>((m31**)lookup_range_check_18_9, 1);
    m31 **device_lookup_range_check_18_10 = clone_to_device<m31*>((m31**)lookup_range_check_18_10, 1);
    m31 **device_lookup_range_check_18_11 = clone_to_device<m31*>((m31**)lookup_range_check_18_11, 1);
    m31 **device_lookup_range_check_18_12 = clone_to_device<m31*>((m31**)lookup_range_check_18_12, 1);
    m31 **device_lookup_range_check_18_13 = clone_to_device<m31*>((m31**)lookup_range_check_18_13, 1);
    m31 **device_lookup_range_check_18_14 = clone_to_device<m31*>((m31**)lookup_range_check_18_14, 1);
    m31 **device_lookup_range_check_18_15 = clone_to_device<m31*>((m31**)lookup_range_check_18_15, 1);
    m31 **device_lookup_range_check_18_16 = clone_to_device<m31*>((m31**)lookup_range_check_18_16, 1);
    m31 **device_lookup_range_check_18_17 = clone_to_device<m31*>((m31**)lookup_range_check_18_17, 1);
    m31 **device_lookup_range_check_18_18 = clone_to_device<m31*>((m31**)lookup_range_check_18_18, 1);
    m31 **device_lookup_range_check_18_19 = clone_to_device<m31*>((m31**)lookup_range_check_18_19, 1);
    m31 **device_lookup_range_check_18_20 = clone_to_device<m31*>((m31**)lookup_range_check_18_20, 1);
    m31 **device_lookup_range_check_18_21 = clone_to_device<m31*>((m31**)lookup_range_check_18_21, 1);
    m31 **device_lookup_range_check_18_22 = clone_to_device<m31*>((m31**)lookup_range_check_18_22, 1);
    m31 **device_lookup_range_check_18_23 = clone_to_device<m31*>((m31**)lookup_range_check_18_23, 1);
    m31 **device_lookup_range_check_18_24 = clone_to_device<m31*>((m31**)lookup_range_check_18_24, 1);
    m31 **device_lookup_range_check_18_25 = clone_to_device<m31*>((m31**)lookup_range_check_18_25, 1);
    m31 **device_lookup_range_check_18_26 = clone_to_device<m31*>((m31**)lookup_range_check_18_26, 1);
    m31 **device_lookup_range_check_18_27 = clone_to_device<m31*>((m31**)lookup_range_check_18_27, 1);
    m31 **device_lookup_range_check_18_28 = clone_to_device<m31*>((m31**)lookup_range_check_18_28, 1);
    m31 **device_lookup_range_check_18_29 = clone_to_device<m31*>((m31**)lookup_range_check_18_29, 1);
    m31 **device_lookup_range_check_18_30 = clone_to_device<m31*>((m31**)lookup_range_check_18_30, 1);
    m31 **device_lookup_range_check_18_31 = clone_to_device<m31*>((m31**)lookup_range_check_18_31, 1);
    m31 **device_lookup_range_check_18_32 = clone_to_device<m31*>((m31**)lookup_range_check_18_32, 1);
    m31 **device_lookup_range_check_18_33 = clone_to_device<m31*>((m31**)lookup_range_check_18_33, 1);
    m31 **device_lookup_range_check_18_34 = clone_to_device<m31*>((m31**)lookup_range_check_18_34, 1);
    m31 **device_lookup_range_check_18_35 = clone_to_device<m31*>((m31**)lookup_range_check_18_35, 1);
    m31 **device_lookup_range_check_18_36 = clone_to_device<m31*>((m31**)lookup_range_check_18_36, 1);
    m31 **device_lookup_range_check_18_37 = clone_to_device<m31*>((m31**)lookup_range_check_18_37, 1);
    m31 **device_lookup_range_check_18_38 = clone_to_device<m31*>((m31**)lookup_range_check_18_38, 1);
    m31 **device_lookup_range_check_18_39 = clone_to_device<m31*>((m31**)lookup_range_check_18_39, 1);
    m31 **device_lookup_range_check_18_40 = clone_to_device<m31*>((m31**)lookup_range_check_18_40, 1);
    m31 **device_lookup_range_check_18_41 = clone_to_device<m31*>((m31**)lookup_range_check_18_41, 1);
    m31 **device_lookup_range_check_18_42 = clone_to_device<m31*>((m31**)lookup_range_check_18_42, 1);
    m31 **device_lookup_range_check_18_43 = clone_to_device<m31*>((m31**)lookup_range_check_18_43, 1);
    m31 **device_lookup_range_check_18_44 = clone_to_device<m31*>((m31**)lookup_range_check_18_44, 1);
    m31 **device_lookup_range_check_18_45 = clone_to_device<m31*>((m31**)lookup_range_check_18_45, 1);
    m31 **device_lookup_range_check_18_46 = clone_to_device<m31*>((m31**)lookup_range_check_18_46, 1);
    m31 **device_lookup_range_check_18_47 = clone_to_device<m31*>((m31**)lookup_range_check_18_47, 1);
    m31 **device_lookup_range_check_18_48 = clone_to_device<m31*>((m31**)lookup_range_check_18_48, 1);
    m31 **device_lookup_range_check_18_49 = clone_to_device<m31*>((m31**)lookup_range_check_18_49, 1);
    m31 **device_lookup_range_check_18_50 = clone_to_device<m31*>((m31**)lookup_range_check_18_50, 1);
    m31 **device_lookup_range_check_18_51 = clone_to_device<m31*>((m31**)lookup_range_check_18_51, 1);
    m31 **device_lookup_range_check_18_52 = clone_to_device<m31*>((m31**)lookup_range_check_18_52, 1);
    m31 **device_lookup_range_check_18_53 = clone_to_device<m31*>((m31**)lookup_range_check_18_53, 1);
    m31 **device_lookup_range_check_18_54 = clone_to_device<m31*>((m31**)lookup_range_check_18_54, 1);
    m31 **device_lookup_range_check_18_55 = clone_to_device<m31*>((m31**)lookup_range_check_18_55, 1);
    m31 **device_lookup_range_check_18_56 = clone_to_device<m31*>((m31**)lookup_range_check_18_56, 1);
    m31 **device_lookup_range_check_18_57 = clone_to_device<m31*>((m31**)lookup_range_check_18_57, 1);
    m31 **device_lookup_range_check_18_58 = clone_to_device<m31*>((m31**)lookup_range_check_18_58, 1);
    m31 **device_lookup_range_check_18_59 = clone_to_device<m31*>((m31**)lookup_range_check_18_59, 1);
    m31 **device_lookup_range_check_18_60 = clone_to_device<m31*>((m31**)lookup_range_check_18_60, 1);
    m31 **device_lookup_range_check_18_61 = clone_to_device<m31*>((m31**)lookup_range_check_18_61, 1);

    // Clone 40 range_check_3_6_6_3 lookups
    m31 **device_lookup_range_check_3_6_6_3_0 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_0, 4);
    m31 **device_lookup_range_check_3_6_6_3_1 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_1, 4);
    m31 **device_lookup_range_check_3_6_6_3_2 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_2, 4);
    m31 **device_lookup_range_check_3_6_6_3_3 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_3, 4);
    m31 **device_lookup_range_check_3_6_6_3_4 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_4, 4);
    m31 **device_lookup_range_check_3_6_6_3_5 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_5, 4);
    m31 **device_lookup_range_check_3_6_6_3_6 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_6, 4);
    m31 **device_lookup_range_check_3_6_6_3_7 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_7, 4);
    m31 **device_lookup_range_check_3_6_6_3_8 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_8, 4);
    m31 **device_lookup_range_check_3_6_6_3_9 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_9, 4);
    m31 **device_lookup_range_check_3_6_6_3_10 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_10, 4);
    m31 **device_lookup_range_check_3_6_6_3_11 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_11, 4);
    m31 **device_lookup_range_check_3_6_6_3_12 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_12, 4);
    m31 **device_lookup_range_check_3_6_6_3_13 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_13, 4);
    m31 **device_lookup_range_check_3_6_6_3_14 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_14, 4);
    m31 **device_lookup_range_check_3_6_6_3_15 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_15, 4);
    m31 **device_lookup_range_check_3_6_6_3_16 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_16, 4);
    m31 **device_lookup_range_check_3_6_6_3_17 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_17, 4);
    m31 **device_lookup_range_check_3_6_6_3_18 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_18, 4);
    m31 **device_lookup_range_check_3_6_6_3_19 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_19, 4);
    m31 **device_lookup_range_check_3_6_6_3_20 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_20, 4);
    m31 **device_lookup_range_check_3_6_6_3_21 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_21, 4);
    m31 **device_lookup_range_check_3_6_6_3_22 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_22, 4);
    m31 **device_lookup_range_check_3_6_6_3_23 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_23, 4);
    m31 **device_lookup_range_check_3_6_6_3_24 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_24, 4);
    m31 **device_lookup_range_check_3_6_6_3_25 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_25, 4);
    m31 **device_lookup_range_check_3_6_6_3_26 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_26, 4);
    m31 **device_lookup_range_check_3_6_6_3_27 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_27, 4);
    m31 **device_lookup_range_check_3_6_6_3_28 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_28, 4);
    m31 **device_lookup_range_check_3_6_6_3_29 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_29, 4);
    m31 **device_lookup_range_check_3_6_6_3_30 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_30, 4);
    m31 **device_lookup_range_check_3_6_6_3_31 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_31, 4);
    m31 **device_lookup_range_check_3_6_6_3_32 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_32, 4);
    m31 **device_lookup_range_check_3_6_6_3_33 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_33, 4);
    m31 **device_lookup_range_check_3_6_6_3_34 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_34, 4);
    m31 **device_lookup_range_check_3_6_6_3_35 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_35, 4);
    m31 **device_lookup_range_check_3_6_6_3_36 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_36, 4);
    m31 **device_lookup_range_check_3_6_6_3_37 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_37, 4);
    m31 **device_lookup_range_check_3_6_6_3_38 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_38, 4);
    m31 **device_lookup_range_check_3_6_6_3_39 = clone_to_device<m31*>((m31**)lookup_range_check_3_6_6_3_39, 4);

    // Lambda for finalizing each column
    auto launch_finalize = [&](unsigned col_index) {
        batch_inverse_secure_field(denom_ptr, denom_inv, trace_size);
        generate_mul_mod_builtin_interaction_finalize_col_kernel<<<num_blocks, block_dim_val>>>(
            col_index, trace_size, denom_inv,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3,
            device_interaction_traces
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    };

    // 94 interaction trace columns based on mul_mod_builtin CPU implementation
    // Columns 0-8: pair(memory_address_to_id_n, memory_id_to_big_n) for n=0..8
    unsigned col_idx = 0;

    // Columns 0-8: memory_address_to_id paired with memory_id_to_big
    #define GEN_MEM_PAIR(N) \
        generate_mul_mod_builtin_interaction_col_gen_kernel<2, 29><<<num_blocks, block_dim_val>>>( \
            device_mem_addr_to_id, device_mem_id_to_big, \
            device_lookup_memory_address_to_id_##N, device_lookup_memory_id_to_big_##N, \
            trace_size, denom_ptr, \
            device_numerator0, device_numerator1, device_numerator2, device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        launch_finalize(col_idx++);

    GEN_MEM_PAIR(0)
    GEN_MEM_PAIR(1)
    GEN_MEM_PAIR(2)
    GEN_MEM_PAIR(3)
    GEN_MEM_PAIR(4)
    GEN_MEM_PAIR(5)
    GEN_MEM_PAIR(6)
    GEN_MEM_PAIR(7)
    GEN_MEM_PAIR(8)

    // Column 9: pair(memory_address_to_id_9, memory_address_to_id_10)
    generate_mul_mod_builtin_interaction_col_gen_kernel<2, 2><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_addr_to_id,
        device_lookup_memory_address_to_id_9, device_lookup_memory_address_to_id_10,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(col_idx++);

    // Column 10: pair(memory_address_to_id_11, memory_address_to_id_12)
    generate_mul_mod_builtin_interaction_col_gen_kernel<2, 2><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_addr_to_id,
        device_lookup_memory_address_to_id_11, device_lookup_memory_address_to_id_12,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(col_idx++);

    // Column 11: pair(memory_address_to_id_13, memory_address_to_id_14)
    generate_mul_mod_builtin_interaction_col_gen_kernel<2, 2><<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_addr_to_id,
        device_lookup_memory_address_to_id_13, device_lookup_memory_address_to_id_14,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(col_idx++);

    // Columns 12-25: pair(memory_id_to_big_n, memory_address_to_id_m)
    #define GEN_MEM_ID_ADDR_PAIR(N, M) \
        generate_mul_mod_builtin_interaction_col_gen_kernel<29, 2><<<num_blocks, block_dim_val>>>( \
            device_mem_id_to_big, device_mem_addr_to_id, \
            device_lookup_memory_id_to_big_##N, device_lookup_memory_address_to_id_##M, \
            trace_size, denom_ptr, \
            device_numerator0, device_numerator1, device_numerator2, device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        launch_finalize(col_idx++);

    GEN_MEM_ID_ADDR_PAIR(9, 15)
    GEN_MEM_ID_ADDR_PAIR(10, 16)
    GEN_MEM_ID_ADDR_PAIR(11, 17)
    GEN_MEM_ID_ADDR_PAIR(12, 18)
    GEN_MEM_ID_ADDR_PAIR(13, 19)
    GEN_MEM_ID_ADDR_PAIR(14, 20)
    GEN_MEM_ID_ADDR_PAIR(15, 21)
    GEN_MEM_ID_ADDR_PAIR(16, 22)
    GEN_MEM_ID_ADDR_PAIR(17, 23)
    GEN_MEM_ID_ADDR_PAIR(18, 24)
    GEN_MEM_ID_ADDR_PAIR(19, 25)
    GEN_MEM_ID_ADDR_PAIR(20, 26)
    GEN_MEM_ID_ADDR_PAIR(21, 27)
    GEN_MEM_ID_ADDR_PAIR(22, 28)

    // Column 26: pair(memory_id_to_big_23, range_check_12_0)
    generate_mul_mod_builtin_interaction_col_gen_kernel<29, 1><<<num_blocks, block_dim_val>>>(
        device_mem_id_to_big, device_range_check_12,
        device_lookup_memory_id_to_big_23, device_lookup_range_check_12_0,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(col_idx++);

    // Columns 27-41: range_check_12 pairs (1,2), (3,4), ... (29, 30)
    #define GEN_RC12_PAIR(N, M) \
        generate_mul_mod_builtin_interaction_col_gen_kernel<1, 1><<<num_blocks, block_dim_val>>>( \
            device_range_check_12, device_range_check_12, \
            device_lookup_range_check_12_##N, device_lookup_range_check_12_##M, \
            trace_size, denom_ptr, \
            device_numerator0, device_numerator1, device_numerator2, device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        launch_finalize(col_idx++);

    GEN_RC12_PAIR(1, 2)
    GEN_RC12_PAIR(3, 4)
    GEN_RC12_PAIR(5, 6)
    GEN_RC12_PAIR(7, 8)
    GEN_RC12_PAIR(9, 10)
    GEN_RC12_PAIR(11, 12)
    GEN_RC12_PAIR(13, 14)
    GEN_RC12_PAIR(15, 16)
    GEN_RC12_PAIR(17, 18)
    GEN_RC12_PAIR(19, 20)
    GEN_RC12_PAIR(21, 22)
    GEN_RC12_PAIR(23, 24)
    GEN_RC12_PAIR(25, 26)
    GEN_RC12_PAIR(27, 28)
    GEN_RC12_PAIR(29, 30)

    // Column 42: pair(range_check_12_31, range_check_3_6_6_3_0)
    generate_mul_mod_builtin_interaction_col_gen_kernel<1, 4><<<num_blocks, block_dim_val>>>(
        device_range_check_12, device_range_check_3_6_6_3,
        device_lookup_range_check_12_31, device_lookup_range_check_3_6_6_3_0,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(col_idx++);

    // Columns 43-61: range_check_3_6_6_3 pairs (1,2), (3,4), ... (37, 38)
    #define GEN_RC3663_PAIR(N, M) \
        generate_mul_mod_builtin_interaction_col_gen_kernel<4, 4><<<num_blocks, block_dim_val>>>( \
            device_range_check_3_6_6_3, device_range_check_3_6_6_3, \
            device_lookup_range_check_3_6_6_3_##N, device_lookup_range_check_3_6_6_3_##M, \
            trace_size, denom_ptr, \
            device_numerator0, device_numerator1, device_numerator2, device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        launch_finalize(col_idx++);

    GEN_RC3663_PAIR(1, 2)
    GEN_RC3663_PAIR(3, 4)
    GEN_RC3663_PAIR(5, 6)
    GEN_RC3663_PAIR(7, 8)
    GEN_RC3663_PAIR(9, 10)
    GEN_RC3663_PAIR(11, 12)
    GEN_RC3663_PAIR(13, 14)
    GEN_RC3663_PAIR(15, 16)
    GEN_RC3663_PAIR(17, 18)
    GEN_RC3663_PAIR(19, 20)
    GEN_RC3663_PAIR(21, 22)
    GEN_RC3663_PAIR(23, 24)
    GEN_RC3663_PAIR(25, 26)
    GEN_RC3663_PAIR(27, 28)
    GEN_RC3663_PAIR(29, 30)
    GEN_RC3663_PAIR(31, 32)
    GEN_RC3663_PAIR(33, 34)
    GEN_RC3663_PAIR(35, 36)
    GEN_RC3663_PAIR(37, 38)

    // Column 62: pair(range_check_3_6_6_3_39, range_check_18_0)
    generate_mul_mod_builtin_interaction_col_gen_kernel<4, 1><<<num_blocks, block_dim_val>>>(
        device_range_check_3_6_6_3, device_range_check_18,
        device_lookup_range_check_3_6_6_3_39, device_lookup_range_check_18_0,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(col_idx++);

    // Columns 63-92: range_check_18 pairs (1,2), (3,4), ... (59, 60)
    #define GEN_RC18_PAIR(N, M) \
        generate_mul_mod_builtin_interaction_col_gen_kernel<1, 1><<<num_blocks, block_dim_val>>>( \
            device_range_check_18, device_range_check_18, \
            device_lookup_range_check_18_##N, device_lookup_range_check_18_##M, \
            trace_size, denom_ptr, \
            device_numerator0, device_numerator1, device_numerator2, device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        launch_finalize(col_idx++);

    GEN_RC18_PAIR(1, 2)
    GEN_RC18_PAIR(3, 4)
    GEN_RC18_PAIR(5, 6)
    GEN_RC18_PAIR(7, 8)
    GEN_RC18_PAIR(9, 10)
    GEN_RC18_PAIR(11, 12)
    GEN_RC18_PAIR(13, 14)
    GEN_RC18_PAIR(15, 16)
    GEN_RC18_PAIR(17, 18)
    GEN_RC18_PAIR(19, 20)
    GEN_RC18_PAIR(21, 22)
    GEN_RC18_PAIR(23, 24)
    GEN_RC18_PAIR(25, 26)
    GEN_RC18_PAIR(27, 28)
    GEN_RC18_PAIR(29, 30)
    GEN_RC18_PAIR(31, 32)
    GEN_RC18_PAIR(33, 34)
    GEN_RC18_PAIR(35, 36)
    GEN_RC18_PAIR(37, 38)
    GEN_RC18_PAIR(39, 40)
    GEN_RC18_PAIR(41, 42)
    GEN_RC18_PAIR(43, 44)
    GEN_RC18_PAIR(45, 46)
    GEN_RC18_PAIR(47, 48)
    GEN_RC18_PAIR(49, 50)
    GEN_RC18_PAIR(51, 52)
    GEN_RC18_PAIR(53, 54)
    GEN_RC18_PAIR(55, 56)
    GEN_RC18_PAIR(57, 58)
    GEN_RC18_PAIR(59, 60)

    // Column 93 (last): single range_check_18_61
    generate_mul_mod_builtin_interaction_last_col_kernel<1><<<num_blocks, block_dim_val>>>(
        device_range_check_18,
        device_lookup_range_check_18_61,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(col_idx++);

    // Compute cumsum_shift and apply coord_prefix_sum
    {
        size_t shared_size = 4 * block_dim_val * sizeof(m31);
        generate_mul_mod_builtin_interaction_cumsum_shift_kernel<<<num_blocks, block_dim_val, shared_size>>>(
            MUL_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS,
            trace_size,
            device_interaction_traces,
            (m31*)claimed_sum
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        generate_mul_mod_builtin_interaction_coord_prefix_sum_kernel<<<num_blocks, block_dim_val>>>(
            (m31*)claimed_sum,
            MUL_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS,
            trace_size,
            device_interaction_traces
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // Apply inclusive_prefix_sum only to the last 4 columns
        inclusive_prefix_sum(interaction_trace[4 * MUL_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * MUL_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * MUL_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
        inclusive_prefix_sum(interaction_trace[4 * MUL_MOD_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);
    }

    // Free all device memory
    cuda_free_memory(denom_ptr);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_mem_addr_to_id);
    cuda_free_memory(device_mem_id_to_big);
    cuda_free_memory(device_range_check_12);
    cuda_free_memory(device_range_check_18);
    cuda_free_memory(device_range_check_3_6_6_3);
    cuda_free_memory(device_interaction_traces);

    // Free lookup arrays - memory_address_to_id
    cuda_free_memory(device_lookup_memory_address_to_id_0);
    cuda_free_memory(device_lookup_memory_address_to_id_1);
    cuda_free_memory(device_lookup_memory_address_to_id_2);
    cuda_free_memory(device_lookup_memory_address_to_id_3);
    cuda_free_memory(device_lookup_memory_address_to_id_4);
    cuda_free_memory(device_lookup_memory_address_to_id_5);
    cuda_free_memory(device_lookup_memory_address_to_id_6);
    cuda_free_memory(device_lookup_memory_address_to_id_7);
    cuda_free_memory(device_lookup_memory_address_to_id_8);
    cuda_free_memory(device_lookup_memory_address_to_id_9);
    cuda_free_memory(device_lookup_memory_address_to_id_10);
    cuda_free_memory(device_lookup_memory_address_to_id_11);
    cuda_free_memory(device_lookup_memory_address_to_id_12);
    cuda_free_memory(device_lookup_memory_address_to_id_13);
    cuda_free_memory(device_lookup_memory_address_to_id_14);
    cuda_free_memory(device_lookup_memory_address_to_id_15);
    cuda_free_memory(device_lookup_memory_address_to_id_16);
    cuda_free_memory(device_lookup_memory_address_to_id_17);
    cuda_free_memory(device_lookup_memory_address_to_id_18);
    cuda_free_memory(device_lookup_memory_address_to_id_19);
    cuda_free_memory(device_lookup_memory_address_to_id_20);
    cuda_free_memory(device_lookup_memory_address_to_id_21);
    cuda_free_memory(device_lookup_memory_address_to_id_22);
    cuda_free_memory(device_lookup_memory_address_to_id_23);
    cuda_free_memory(device_lookup_memory_address_to_id_24);
    cuda_free_memory(device_lookup_memory_address_to_id_25);
    cuda_free_memory(device_lookup_memory_address_to_id_26);
    cuda_free_memory(device_lookup_memory_address_to_id_27);
    cuda_free_memory(device_lookup_memory_address_to_id_28);

    // Free lookup arrays - memory_id_to_big
    cuda_free_memory(device_lookup_memory_id_to_big_0);
    cuda_free_memory(device_lookup_memory_id_to_big_1);
    cuda_free_memory(device_lookup_memory_id_to_big_2);
    cuda_free_memory(device_lookup_memory_id_to_big_3);
    cuda_free_memory(device_lookup_memory_id_to_big_4);
    cuda_free_memory(device_lookup_memory_id_to_big_5);
    cuda_free_memory(device_lookup_memory_id_to_big_6);
    cuda_free_memory(device_lookup_memory_id_to_big_7);
    cuda_free_memory(device_lookup_memory_id_to_big_8);
    cuda_free_memory(device_lookup_memory_id_to_big_9);
    cuda_free_memory(device_lookup_memory_id_to_big_10);
    cuda_free_memory(device_lookup_memory_id_to_big_11);
    cuda_free_memory(device_lookup_memory_id_to_big_12);
    cuda_free_memory(device_lookup_memory_id_to_big_13);
    cuda_free_memory(device_lookup_memory_id_to_big_14);
    cuda_free_memory(device_lookup_memory_id_to_big_15);
    cuda_free_memory(device_lookup_memory_id_to_big_16);
    cuda_free_memory(device_lookup_memory_id_to_big_17);
    cuda_free_memory(device_lookup_memory_id_to_big_18);
    cuda_free_memory(device_lookup_memory_id_to_big_19);
    cuda_free_memory(device_lookup_memory_id_to_big_20);
    cuda_free_memory(device_lookup_memory_id_to_big_21);
    cuda_free_memory(device_lookup_memory_id_to_big_22);
    cuda_free_memory(device_lookup_memory_id_to_big_23);

    // Free lookup arrays - range_check_12
    cuda_free_memory(device_lookup_range_check_12_0);
    cuda_free_memory(device_lookup_range_check_12_1);
    cuda_free_memory(device_lookup_range_check_12_2);
    cuda_free_memory(device_lookup_range_check_12_3);
    cuda_free_memory(device_lookup_range_check_12_4);
    cuda_free_memory(device_lookup_range_check_12_5);
    cuda_free_memory(device_lookup_range_check_12_6);
    cuda_free_memory(device_lookup_range_check_12_7);
    cuda_free_memory(device_lookup_range_check_12_8);
    cuda_free_memory(device_lookup_range_check_12_9);
    cuda_free_memory(device_lookup_range_check_12_10);
    cuda_free_memory(device_lookup_range_check_12_11);
    cuda_free_memory(device_lookup_range_check_12_12);
    cuda_free_memory(device_lookup_range_check_12_13);
    cuda_free_memory(device_lookup_range_check_12_14);
    cuda_free_memory(device_lookup_range_check_12_15);
    cuda_free_memory(device_lookup_range_check_12_16);
    cuda_free_memory(device_lookup_range_check_12_17);
    cuda_free_memory(device_lookup_range_check_12_18);
    cuda_free_memory(device_lookup_range_check_12_19);
    cuda_free_memory(device_lookup_range_check_12_20);
    cuda_free_memory(device_lookup_range_check_12_21);
    cuda_free_memory(device_lookup_range_check_12_22);
    cuda_free_memory(device_lookup_range_check_12_23);
    cuda_free_memory(device_lookup_range_check_12_24);
    cuda_free_memory(device_lookup_range_check_12_25);
    cuda_free_memory(device_lookup_range_check_12_26);
    cuda_free_memory(device_lookup_range_check_12_27);
    cuda_free_memory(device_lookup_range_check_12_28);
    cuda_free_memory(device_lookup_range_check_12_29);
    cuda_free_memory(device_lookup_range_check_12_30);
    cuda_free_memory(device_lookup_range_check_12_31);

    // Free lookup arrays - range_check_18
    cuda_free_memory(device_lookup_range_check_18_0);
    cuda_free_memory(device_lookup_range_check_18_1);
    cuda_free_memory(device_lookup_range_check_18_2);
    cuda_free_memory(device_lookup_range_check_18_3);
    cuda_free_memory(device_lookup_range_check_18_4);
    cuda_free_memory(device_lookup_range_check_18_5);
    cuda_free_memory(device_lookup_range_check_18_6);
    cuda_free_memory(device_lookup_range_check_18_7);
    cuda_free_memory(device_lookup_range_check_18_8);
    cuda_free_memory(device_lookup_range_check_18_9);
    cuda_free_memory(device_lookup_range_check_18_10);
    cuda_free_memory(device_lookup_range_check_18_11);
    cuda_free_memory(device_lookup_range_check_18_12);
    cuda_free_memory(device_lookup_range_check_18_13);
    cuda_free_memory(device_lookup_range_check_18_14);
    cuda_free_memory(device_lookup_range_check_18_15);
    cuda_free_memory(device_lookup_range_check_18_16);
    cuda_free_memory(device_lookup_range_check_18_17);
    cuda_free_memory(device_lookup_range_check_18_18);
    cuda_free_memory(device_lookup_range_check_18_19);
    cuda_free_memory(device_lookup_range_check_18_20);
    cuda_free_memory(device_lookup_range_check_18_21);
    cuda_free_memory(device_lookup_range_check_18_22);
    cuda_free_memory(device_lookup_range_check_18_23);
    cuda_free_memory(device_lookup_range_check_18_24);
    cuda_free_memory(device_lookup_range_check_18_25);
    cuda_free_memory(device_lookup_range_check_18_26);
    cuda_free_memory(device_lookup_range_check_18_27);
    cuda_free_memory(device_lookup_range_check_18_28);
    cuda_free_memory(device_lookup_range_check_18_29);
    cuda_free_memory(device_lookup_range_check_18_30);
    cuda_free_memory(device_lookup_range_check_18_31);
    cuda_free_memory(device_lookup_range_check_18_32);
    cuda_free_memory(device_lookup_range_check_18_33);
    cuda_free_memory(device_lookup_range_check_18_34);
    cuda_free_memory(device_lookup_range_check_18_35);
    cuda_free_memory(device_lookup_range_check_18_36);
    cuda_free_memory(device_lookup_range_check_18_37);
    cuda_free_memory(device_lookup_range_check_18_38);
    cuda_free_memory(device_lookup_range_check_18_39);
    cuda_free_memory(device_lookup_range_check_18_40);
    cuda_free_memory(device_lookup_range_check_18_41);
    cuda_free_memory(device_lookup_range_check_18_42);
    cuda_free_memory(device_lookup_range_check_18_43);
    cuda_free_memory(device_lookup_range_check_18_44);
    cuda_free_memory(device_lookup_range_check_18_45);
    cuda_free_memory(device_lookup_range_check_18_46);
    cuda_free_memory(device_lookup_range_check_18_47);
    cuda_free_memory(device_lookup_range_check_18_48);
    cuda_free_memory(device_lookup_range_check_18_49);
    cuda_free_memory(device_lookup_range_check_18_50);
    cuda_free_memory(device_lookup_range_check_18_51);
    cuda_free_memory(device_lookup_range_check_18_52);
    cuda_free_memory(device_lookup_range_check_18_53);
    cuda_free_memory(device_lookup_range_check_18_54);
    cuda_free_memory(device_lookup_range_check_18_55);
    cuda_free_memory(device_lookup_range_check_18_56);
    cuda_free_memory(device_lookup_range_check_18_57);
    cuda_free_memory(device_lookup_range_check_18_58);
    cuda_free_memory(device_lookup_range_check_18_59);
    cuda_free_memory(device_lookup_range_check_18_60);
    cuda_free_memory(device_lookup_range_check_18_61);

    // Free lookup arrays - range_check_3_6_6_3
    cuda_free_memory(device_lookup_range_check_3_6_6_3_0);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_1);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_2);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_3);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_4);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_5);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_6);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_7);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_8);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_9);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_10);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_11);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_12);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_13);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_14);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_15);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_16);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_17);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_18);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_19);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_20);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_21);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_22);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_23);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_24);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_25);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_26);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_27);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_28);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_29);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_30);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_31);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_32);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_33);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_34);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_35);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_36);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_37);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_38);
    cuda_free_memory(device_lookup_range_check_3_6_6_3_39);

    global_timer.end("generate mul_mod_builtin interaction trace");
}
