/**
 * CUDA trace generation for partial_ec_mul_window_bits_9 (311-col "now" architecture).
 *
 * Implements AIR-compatible trace generation with:
 *   - 311 base trace columns
 *   - 65 logup interaction columns
 *   - Sub-component feeds to pedersen_points_table, rc_9_9, and rc_20
 *
 * Column layout:
 *   0-85:    Input (86 cols)
 *   86-113:  Table point x (28 limbs)
 *   114-141: Table point y (28 limbs)
 *   142-169: Slope (28 limbs)
 *   170:     VerifyMul #1 k
 *   171-197: VerifyMul #1 carries (27)
 *   198-225: Result x (28 limbs)
 *   226:     VerifyMul #2 k
 *   227-253: VerifyMul #2 carries (27)
 *   254-281: Result y (28 limbs)
 *   282:     VerifyMul #3 k
 *   283-309: VerifyMul #3 carries (27)
 *   310:     Enabler
 *
 * EC point addition:
 *   slope = (y2 - y1) / (x2 - x1)
 *   result_x = slope^2 - x1 - x2
 *   result_y = slope * (x1 - result_x) - y1
 * where (x1,y1) = accumulator (input cols 30-85), (x2,y2) = table point (cols 86-141)
 */

#include "gen_partial_ec_mul_wb9_trace.cuh"
#include "../fields.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
#include "../utils.cuh"
#include "logup.cuh"
#include "batch_inverse.cuh"
#include "cuda_mem_pool.cuh"
#include "../prefix_sum.cuh"

// Pedersen small table — defined in pedersen_table_init.cu
#define PEDERSEN_TABLE_SMALL_N_COLUMNS 56
extern __device__ m31* g_pedersen_table_small_columns[PEDERSEN_TABLE_SMALL_N_COLUMNS];

// Block size for kernel launch
#define WB9_BLOCK_SIZE 256

// ============================================================================
// Felt252 type and field operations
// ============================================================================

typedef ff_storage<8> Felt252Field;

static __device__ __forceinline__ Felt252Field wb9_felt_add(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

static __device__ __forceinline__ Felt252Field wb9_felt_sub(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

static __device__ __forceinline__ Felt252Field wb9_felt_to_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::to_montgomery(a);
}

static __device__ __forceinline__ Felt252Field wb9_felt_from_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::from_montgomery(a);
}

static __device__ __forceinline__ Felt252Field wb9_felt_mul(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

static __device__ __forceinline__ Felt252Field wb9_felt_inverse(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::inverse(a);
}

// ============================================================================
// Limb conversion utilities
// ============================================================================

// Convert 28 x 9-bit M31 limbs to Felt252Field (standard form)
static __device__ Felt252Field wb9_limbs28_to_felt252(const m31* limbs) {
    uint64_t accum = 0;
    int bit_pos = 0;
    Felt252Field result = {0};
    int out_idx = 0;

    for (int i = 0; i < 28 && out_idx < 8; i++) {
        accum |= ((uint64_t)limbs[i]) << bit_pos;
        bit_pos += 9;

        while (bit_pos >= 32 && out_idx < 8) {
            result.limbs[out_idx++] = (uint32_t)(accum & 0xFFFFFFFF);
            accum >>= 32;
            bit_pos -= 32;
        }
    }
    if (out_idx < 8) {
        result.limbs[out_idx] = (uint32_t)(accum & 0xFFFFFFFF);
    }

    return result;
}

// Convert Felt252Field (standard form) to 28 x 9-bit limbs
static __device__ void wb9_felt252_to_limbs28(const Felt252Field& felt, m31* limbs) {
    uint64_t val0 = ((uint64_t)felt.limbs[1] << 32) | felt.limbs[0];
    uint64_t val1 = ((uint64_t)felt.limbs[3] << 32) | felt.limbs[2];
    uint64_t val2 = ((uint64_t)felt.limbs[5] << 32) | felt.limbs[4];
    uint64_t val3 = ((uint64_t)felt.limbs[7] << 32) | felt.limbs[6];

    for (int i = 0; i < 7; i++) {
        limbs[i] = (uint32_t)((val0 >> (i * 9)) & 0x1FF);
    }
    uint64_t cross01 = (val0 >> 63) | (val1 << 1);
    limbs[7] = (uint32_t)(cross01 & 0x1FF);

    for (int i = 0; i < 6; i++) {
        limbs[8 + i] = (uint32_t)((val1 >> (8 + i * 9)) & 0x1FF);
    }
    uint64_t cross12 = (val1 >> 62) | (val2 << 2);
    limbs[14] = (uint32_t)(cross12 & 0x1FF);

    for (int i = 0; i < 6; i++) {
        limbs[15 + i] = (uint32_t)((val2 >> (7 + i * 9)) & 0x1FF);
    }
    uint64_t cross23 = (val2 >> 61) | (val3 << 3);
    limbs[21] = (uint32_t)(cross23 & 0x1FF);

    for (int i = 0; i < 6; i++) {
        limbs[22 + i] = (uint32_t)((val3 >> (6 + i * 9)) & 0x1FF);
    }
}

// ============================================================================
// Schoolbook multiplication: 28 x 28 limbs -> 55 coefficients
// Both factors are int64 arrays (can be signed for limb differences)
// ============================================================================

static __device__ void wb9_schoolbook_mul_28x28(
    const int64_t* a_limbs,
    const int64_t* b_limbs,
    int64_t* product  // 55 coefficients output
) {
    for (int i = 0; i < 55; i++) product[i] = 0;

    for (int i = 0; i < 28; i++) {
        int64_t ai = a_limbs[i];
        for (int j = 0; j < 28; j++) {
            product[i + j] += ai * b_limbs[j];
        }
    }
}

// ============================================================================
// Modular reduction: conv (55 coefficients) -> conv_mod (28 values)
// For Starknet prime p = 2^252 + 17*2^192 + 1
// ============================================================================

static __device__ void wb9_compute_conv_mod(
    const int64_t* conv,
    int64_t* conv_mod
) {
    conv_mod[0]  = 32 * conv[0] - 4 * conv[21] + 8 * conv[49];
    conv_mod[1]  = conv[0] + 32 * conv[1] - 4 * conv[22] + 8 * conv[50];
    conv_mod[2]  = conv[1] + 32 * conv[2] - 4 * conv[23] + 8 * conv[51];
    conv_mod[3]  = conv[2] + 32 * conv[3] - 4 * conv[24] + 8 * conv[52];
    conv_mod[4]  = conv[3] + 32 * conv[4] - 4 * conv[25] + 8 * conv[53];
    conv_mod[5]  = conv[4] + 32 * conv[5] - 4 * conv[26] + 8 * conv[54];
    conv_mod[6]  = conv[5] + 32 * conv[6] - 4 * conv[27];
    conv_mod[7]  = 2 * conv[0] + conv[6] + 32 * conv[7] - 4 * conv[28];
    conv_mod[8]  = 2 * conv[1] + conv[7] + 32 * conv[8] - 4 * conv[29];
    conv_mod[9]  = 2 * conv[2] + conv[8] + 32 * conv[9] - 4 * conv[30];
    conv_mod[10] = 2 * conv[3] + conv[9] + 32 * conv[10] - 4 * conv[31];
    conv_mod[11] = 2 * conv[4] + conv[10] + 32 * conv[11] - 4 * conv[32];
    conv_mod[12] = 2 * conv[5] + conv[11] + 32 * conv[12] - 4 * conv[33];
    conv_mod[13] = 2 * conv[6] + conv[12] + 32 * conv[13] - 4 * conv[34];
    conv_mod[14] = 2 * conv[7] + conv[13] + 32 * conv[14] - 4 * conv[35];
    conv_mod[15] = 2 * conv[8] + conv[14] + 32 * conv[15] - 4 * conv[36];
    conv_mod[16] = 2 * conv[9] + conv[15] + 32 * conv[16] - 4 * conv[37];
    conv_mod[17] = 2 * conv[10] + conv[16] + 32 * conv[17] - 4 * conv[38];
    conv_mod[18] = 2 * conv[11] + conv[17] + 32 * conv[18] - 4 * conv[39];
    conv_mod[19] = 2 * conv[12] + conv[18] + 32 * conv[19] - 4 * conv[40];
    conv_mod[20] = 2 * conv[13] + conv[19] + 32 * conv[20] - 4 * conv[41];
    conv_mod[21] = 2 * conv[14] + conv[20] - 4 * conv[42] + 64 * conv[49];
    conv_mod[22] = 2 * conv[15] - 4 * conv[43] + 2 * conv[49] + 64 * conv[50];
    conv_mod[23] = 2 * conv[16] - 4 * conv[44] + 2 * conv[50] + 64 * conv[51];
    conv_mod[24] = 2 * conv[17] - 4 * conv[45] + 2 * conv[51] + 64 * conv[52];
    conv_mod[25] = 2 * conv[18] - 4 * conv[46] + 2 * conv[52] + 64 * conv[53];
    conv_mod[26] = 2 * conv[19] - 4 * conv[47] + 2 * conv[53] + 64 * conv[54];
    conv_mod[27] = 2 * conv[20] - 4 * conv[48] + 2 * conv[54];
}

// ============================================================================
// Extract k value from conv_mod using biased arithmetic
// ============================================================================

static __device__ __forceinline__ int64_t wb9_compute_k(const int64_t* conv_mod) {
    uint32_t k_mod_tmp = (
        (uint32_t)(conv_mod[0] + 134217728) +
        (((uint32_t)(conv_mod[1] + 134217728) & 511) << 9) +
        131072
    ) & 262143;

    int64_t k_val = (int64_t)(k_mod_tmp & 0xFFFF) +
                    (int64_t)((int32_t)((k_mod_tmp >> 16) & 0x3) - 2) * 65536;
    return k_val;
}

// ============================================================================
// Compute carry chain from conv_mod and k
// ============================================================================

static __device__ void wb9_compute_carries(
    const int64_t* conv_mod,
    int64_t k_val,
    int64_t* carry  // 27 carry values
) {
    carry[0] = (conv_mod[0] - k_val) / 512;
    for (int i = 1; i < 21; i++) {
        carry[i] = (conv_mod[i] + carry[i-1]) / 512;
    }
    // Special case at carry[21]: includes -136*k term
    carry[21] = (conv_mod[21] - 136 * k_val + carry[20]) / 512;
    for (int i = 22; i < 27; i++) {
        carry[i] = (conv_mod[i] + carry[i-1]) / 512;
    }
}

// ============================================================================
// Convert int64 to M31 (proper modular reduction)
// ============================================================================

static __device__ __forceinline__ m31 wb9_int64_to_m31(int64_t val) {
    const int64_t P = 2147483647LL;  // 2^31 - 1
    int64_t result = val % P;
    if (result < 0) result += P;
    return (m31)(uint32_t)result;
}

// ============================================================================
// M31 addition (modular)
// ============================================================================

static __device__ __forceinline__ m31 wb9_m31_add(m31 a, m31 b) {
    uint32_t sum = a + b;
    if (sum >= 2147483647u) sum -= 2147483647u;
    return sum;
}

// ============================================================================
// VerifyMul252: compute k and carries for proving a*b = c (mod P)
//
// All inputs are int64 arrays (can hold signed limb differences/sums).
// Computes: schoolbook(a_limbs, b_limbs) - c_limbs = k*P (in limb space)
// Outputs k as M31 and 27 carries as M31.
// ============================================================================

static __device__ void wb9_verify_mul_252(
    const int64_t* a_limbs,  // 28 limbs (first factor)
    const int64_t* b_limbs,  // 28 limbs (second factor)
    const int64_t* c_limbs,  // 28 limbs (expected: a*b = c mod P)
    m31* k_out,              // output k value
    m31* carry_out           // 27 output carry values
) {
    // Schoolbook product
    int64_t product[55];
    wb9_schoolbook_mul_28x28(a_limbs, b_limbs, product);

    // conv = product - expected
    int64_t conv[55];
    for (int i = 0; i < 28; i++) {
        conv[i] = product[i] - c_limbs[i];
    }
    for (int i = 28; i < 55; i++) {
        conv[i] = product[i];
    }

    // Modular reduction
    int64_t conv_mod[28];
    wb9_compute_conv_mod(conv, conv_mod);

    // Extract k
    int64_t k_val = wb9_compute_k(conv_mod);
    *k_out = wb9_int64_to_m31(k_val);

    // Compute carries
    int64_t carry[27];
    wb9_compute_carries(conv_mod, k_val, carry);
    for (int i = 0; i < 27; i++) {
        carry_out[i] = wb9_int64_to_m31(carry[i]);
    }
}

// ============================================================================
// Kernel arguments (passed via constant memory)
// ============================================================================

struct PemWb9Args {
    m31** traces;              // [311] device ptrs
    m31** inputs;              // [86] device ptrs
    m31** sub_ppt;             // [1] device ptr
    m31** sub_rc_9_9[8];       // 8 variant arrays
    m31** sub_rc_20[8];        // 8 variant arrays
    m31** lk_pem_0;            // [87] device ptrs
    m31** lk_pem_1;            // [87] device ptrs
    m31** lk_ppt_0;            // [58] device ptrs
    m31** lk_rc_20[8];         // 8 variant arrays
    m31** lk_rc_9_9[8];        // 8 variant arrays
    uint32_t n_rows;
    uint32_t trace_size;
};

static __constant__ PemWb9Args d_wb9_args;

// ============================================================================
// Sub-component input helpers
// ============================================================================

// Write RC_9_9 sub-component inputs for one 28-limb field element.
// field_idx: 0=slope, 1=result_x, 2=result_y
// Distribution: 14 pairs round-robin across 8 variants [a..h, a..f]
// Variant counts per field element: [2,2,2,2,2,2,1,1]
static __device__ void wb9_write_rc_9_9_sub_inputs(
    const m31* limbs,
    int field_idx,
    m31** sub_rc_9_9[8],
    uint32_t row
) {
    for (int p = 0; p < 14; p++) {
        int variant, local_idx;
        if (p < 8) {
            variant = p;
            local_idx = 0;
        } else {
            variant = p - 8;
            local_idx = 1;
        }

        int entries_per_field = (variant < 6) ? 2 : 1;
        int entry = field_idx * entries_per_field + local_idx;

        sub_rc_9_9[variant][2 * entry][row] = limbs[2 * p];
        sub_rc_9_9[variant][2 * entry + 1][row] = limbs[2 * p + 1];
    }
}

// Write RC_20 sub-component inputs for one VerifyMul (k + 27 carries = 28 values).
// vm_idx: 0, 1, or 2 (which VerifyMul)
// Distribution: 28 values round-robin across 8 variants [a..h]
// Variant counts per VerifyMul: [4,4,4,4,3,3,3,3]
static __device__ void wb9_write_rc_20_sub_inputs(
    m31 k_val,
    const m31* carries,
    int vm_idx,
    m31** sub_rc_20[8],
    uint32_t row
) {
    const uint32_t BIAS = 524288u;  // 2^19

    for (int v = 0; v < 28; v++) {
        int variant = v % 8;
        int local_idx = v / 8;

        int entries_per_vm = (variant < 4) ? 4 : 3;
        int entry = vm_idx * entries_per_vm + local_idx;

        m31 val = (v == 0) ? k_val : carries[v - 1];
        sub_rc_20[variant][entry][row] = wb9_m31_add(val, BIAS);
    }
}

// ============================================================================
// Lookup data helpers
// ============================================================================

// Relation ID constants (from SIMD reference)
#define WB9_PEM_RELATION_ID     2038149019u
#define WB9_PPT_RELATION_ID     1791500038u

// Per-variant relation IDs for RC_20 [a..h] — same as wb18 (global shared)
static __device__ const uint32_t WB9_RC_20_RELATION_IDS[8] = {
    1410849886u,  // a
    514232941u,   // b
    531010560u,   // c
    480677703u,   // d
    497455322u,   // e
    447122465u,   // f
    463900084u,   // g
    682009131u    // h
};

// Per-variant relation IDs for RC_9_9 [a..h] — same as wb18 (global shared)
static __device__ const uint32_t WB9_RC_9_9_RELATION_IDS[8] = {
    517791011u, 1897792095u, 1881014476u, 1864236857u,
    1847459238u, 1830681619u, 1813904000u, 2065568285u
};

// Write RC_9_9 lookup data for one 28-limb field element.
static __device__ void wb9_write_rc_9_9_lookup(
    const m31* limbs,
    int field_idx,
    m31** lk_rc_9_9[8],
    uint32_t row
) {
    for (int p = 0; p < 14; p++) {
        int variant, local_idx;
        if (p < 8) {
            variant = p;
            local_idx = 0;
        } else {
            variant = p - 8;
            local_idx = 1;
        }

        int entries_per_field = (variant < 6) ? 2 : 1;
        int entry = field_idx * entries_per_field + local_idx;

        lk_rc_9_9[variant][3 * entry][row] = WB9_RC_9_9_RELATION_IDS[variant];
        lk_rc_9_9[variant][3 * entry + 1][row] = limbs[2 * p];
        lk_rc_9_9[variant][3 * entry + 2][row] = limbs[2 * p + 1];
    }
}

// Write RC_20 lookup data for one VerifyMul.
static __device__ void wb9_write_rc_20_lookup(
    m31 k_val,
    const m31* carries,
    int vm_idx,
    m31** lk_rc_20[8],
    uint32_t row
) {
    const uint32_t BIAS = 524288u;

    for (int v = 0; v < 28; v++) {
        int variant = v % 8;
        int local_idx = v / 8;

        int entries_per_vm = (variant < 4) ? 4 : 3;
        int entry = vm_idx * entries_per_vm + local_idx;

        m31 val = (v == 0) ? k_val : carries[v - 1];

        lk_rc_20[variant][2 * entry][row] = WB9_RC_20_RELATION_IDS[variant];
        lk_rc_20[variant][2 * entry + 1][row] = wb9_m31_add(val, BIAS);
    }
}

// ============================================================================
// Main trace generation kernel
// ============================================================================

__global__ void __launch_bounds__(WB9_BLOCK_SIZE, 2)
wb9_trace_kernel() {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_wb9_args.trace_size) return;

    m31** traces = d_wb9_args.traces;
    m31** inputs = d_wb9_args.inputs;
    uint32_t n_rows = d_wb9_args.n_rows;

    // Padding rows use the input data directly (already padded by Rust with
    // first-packed-row cycling, matching the SIMD path).
    uint32_t src_row = row;

    // ====================================================================
    // 1. Read and write 86 input columns (cols 0-85)
    // ====================================================================
    m31 input_vals[86];
    for (int i = 0; i < 86; i++) {
        input_vals[i] = inputs[i][src_row];
        traces[i][row] = input_vals[i];
    }

    // ====================================================================
    // 2. Compute table index and lookup point from pedersen small table
    // ====================================================================
    uint32_t table_idx = 512u * (uint32_t)input_vals[1] + (uint32_t)input_vals[2];

    m31 table_x_limbs[28], table_y_limbs[28];
    for (int i = 0; i < 28; i++) {
        table_x_limbs[i] = g_pedersen_table_small_columns[i][table_idx];
    }
    for (int i = 0; i < 28; i++) {
        table_y_limbs[i] = g_pedersen_table_small_columns[28 + i][table_idx];
    }

    // Write table point to trace (cols 86-141)
    for (int i = 0; i < 28; i++) traces[86 + i][row] = table_x_limbs[i];
    for (int i = 0; i < 28; i++) traces[114 + i][row] = table_y_limbs[i];

    // ====================================================================
    // 3. Store raw limb arrays and convert to Felt252 for field ops
    // ====================================================================
    // acc_x limbs = input cols 30-57, acc_y limbs = input cols 58-85
    m31* acc_x_limbs = &input_vals[30];   // 28 limbs
    m31* acc_y_limbs = &input_vals[58];   // 28 limbs

    Felt252Field acc_x = wb9_limbs28_to_felt252(acc_x_limbs);
    Felt252Field acc_y = wb9_limbs28_to_felt252(acc_y_limbs);
    Felt252Field table_x = wb9_limbs28_to_felt252(table_x_limbs);
    Felt252Field table_y = wb9_limbs28_to_felt252(table_y_limbs);

    // ====================================================================
    // 4. Compute slope = (y2 - y1) / (x2 - x1)
    // ====================================================================
    Felt252Field dy = wb9_felt_sub(table_y, acc_y);
    Felt252Field dx = wb9_felt_sub(table_x, acc_x);

    // Compute slope via Montgomery field arithmetic
    Felt252Field num_mont = wb9_felt_to_mont(dy);
    Felt252Field denom_mont = wb9_felt_to_mont(dx);
    Felt252Field inv_denom_mont = wb9_felt_inverse(denom_mont);
    Felt252Field slope_mont = wb9_felt_mul(num_mont, inv_denom_mont);
    Felt252Field slope = wb9_felt_from_mont(slope_mont);

    // Decompose slope to 9-bit canonical limbs
    m31 slope_limbs[28];
    wb9_felt252_to_limbs28(slope, slope_limbs);

    // Write slope to trace (cols 142-169)
    for (int i = 0; i < 28; i++) traces[142 + i][row] = slope_limbs[i];

    // ====================================================================
    // 5. VerifyMul #1: slope * (table_x - acc_x) = (table_y - acc_y) (mod P)
    //
    // The AIR evaluates: schoolbook(slope, table_x_limbs - acc_x_limbs)
    //                   - (table_y_limbs - acc_y_limbs)
    // Using raw limb-by-limb differences (not canonical Felt252 decomposition).
    // ====================================================================
    int64_t slope_i64[28], dx_limbs[28], dy_limbs[28];
    for (int i = 0; i < 28; i++) {
        slope_i64[i] = (int64_t)slope_limbs[i];
        dx_limbs[i] = (int64_t)table_x_limbs[i] - (int64_t)acc_x_limbs[i];
        dy_limbs[i] = (int64_t)table_y_limbs[i] - (int64_t)acc_y_limbs[i];
    }

    m31 k1;
    m31 carries1[27];
    wb9_verify_mul_252(slope_i64, dx_limbs, dy_limbs, &k1, carries1);

    // Write VerifyMul #1: k (col 170), carries (cols 171-197)
    traces[170][row] = k1;
    for (int i = 0; i < 27; i++) traces[171 + i][row] = carries1[i];

    // ====================================================================
    // 6. Compute result_x = slope^2 - acc_x - table_x
    // ====================================================================
    Felt252Field slope_sq = wb9_felt_from_mont(
        wb9_felt_mul(wb9_felt_to_mont(slope), wb9_felt_to_mont(slope)));
    Felt252Field result_x_felt = wb9_felt_sub(wb9_felt_sub(slope_sq, acc_x), table_x);
    m31 result_x_limbs[28];
    wb9_felt252_to_limbs28(result_x_felt, result_x_limbs);

    // Write result_x to trace (cols 198-225)
    for (int i = 0; i < 28; i++) traces[198 + i][row] = result_x_limbs[i];

    // ====================================================================
    // 7. VerifyMul #2: slope * slope = acc_x + table_x + result_x (mod P)
    //
    // The AIR evaluates: schoolbook(slope, slope)
    //                   - (acc_x_limbs + table_x_limbs + result_x_limbs)
    // Using raw limb-by-limb sum (not canonical Felt252 decomposition).
    // ====================================================================
    int64_t vm2_expected[28];
    for (int i = 0; i < 28; i++) {
        vm2_expected[i] = (int64_t)acc_x_limbs[i] + (int64_t)table_x_limbs[i]
                        + (int64_t)result_x_limbs[i];
    }

    m31 k2;
    m31 carries2[27];
    wb9_verify_mul_252(slope_i64, slope_i64, vm2_expected, &k2, carries2);

    // Write VerifyMul #2: k (col 226), carries (cols 227-253)
    traces[226][row] = k2;
    for (int i = 0; i < 27; i++) traces[227 + i][row] = carries2[i];

    // ====================================================================
    // 8. Compute result_y = slope * (acc_x - result_x) - acc_y
    // ====================================================================
    Felt252Field slope_times_diff = wb9_felt_from_mont(
        wb9_felt_mul(wb9_felt_to_mont(slope),
                      wb9_felt_to_mont(wb9_felt_sub(acc_x, result_x_felt))));
    Felt252Field result_y_felt = wb9_felt_sub(slope_times_diff, acc_y);
    m31 result_y_limbs[28];
    wb9_felt252_to_limbs28(result_y_felt, result_y_limbs);

    // Write result_y to trace (cols 254-281)
    for (int i = 0; i < 28; i++) traces[254 + i][row] = result_y_limbs[i];

    // ====================================================================
    // 9. VerifyMul #3: slope * (acc_x - result_x) = acc_y + result_y (mod P)
    //
    // The AIR evaluates: schoolbook(slope, acc_x_limbs - result_x_limbs)
    //                   - (acc_y_limbs + result_y_limbs)
    // Using raw limb-by-limb operations.
    // ====================================================================
    int64_t vm3_b[28], vm3_expected[28];
    for (int i = 0; i < 28; i++) {
        vm3_b[i] = (int64_t)acc_x_limbs[i] - (int64_t)result_x_limbs[i];
        vm3_expected[i] = (int64_t)acc_y_limbs[i] + (int64_t)result_y_limbs[i];
    }

    m31 k3;
    m31 carries3[27];
    wb9_verify_mul_252(slope_i64, vm3_b, vm3_expected, &k3, carries3);

    // Write VerifyMul #3: k (col 282), carries (cols 283-309)
    traces[282][row] = k3;
    for (int i = 0; i < 27; i++) traces[283 + i][row] = carries3[i];

    // ====================================================================
    // 9. Enabler (col 310)
    // ====================================================================
    traces[310][row] = (row < n_rows) ? 1u : 0u;

    // ====================================================================
    // 10. Sub-component inputs
    // ====================================================================

    // PPT: table index
    d_wb9_args.sub_ppt[0][row] = table_idx;

    // RC_9_9: distribute slope/result_x/result_y limb pairs across 8 variants
    wb9_write_rc_9_9_sub_inputs(slope_limbs, 0, d_wb9_args.sub_rc_9_9, row);
    wb9_write_rc_9_9_sub_inputs(result_x_limbs, 1, d_wb9_args.sub_rc_9_9, row);
    wb9_write_rc_9_9_sub_inputs(result_y_limbs, 2, d_wb9_args.sub_rc_9_9, row);

    // RC_20: distribute k+carries from 3 VerifyMuls across 8 variants
    wb9_write_rc_20_sub_inputs(k1, carries1, 0, d_wb9_args.sub_rc_20, row);
    wb9_write_rc_20_sub_inputs(k2, carries2, 1, d_wb9_args.sub_rc_20, row);
    wb9_write_rc_20_sub_inputs(k3, carries3, 2, d_wb9_args.sub_rc_20, row);

    // ====================================================================
    // 11. Lookup data (for interaction trace)
    // ====================================================================

    // partial_ec_mul_0: [relation_id, input_0..85]
    d_wb9_args.lk_pem_0[0][row] = WB9_PEM_RELATION_ID;
    for (int i = 0; i < 86; i++) {
        d_wb9_args.lk_pem_0[1 + i][row] = input_vals[i];
    }

    // partial_ec_mul_1: [relation_id, input_0, input_1+1, input_3..29, 0, rx..., ry...]
    d_wb9_args.lk_pem_1[0][row] = WB9_PEM_RELATION_ID;
    d_wb9_args.lk_pem_1[1][row] = input_vals[0];
    d_wb9_args.lk_pem_1[2][row] = wb9_m31_add(input_vals[1], 1u);
    for (int i = 3; i < 30; i++) {
        d_wb9_args.lk_pem_1[i][row] = input_vals[i];
    }
    d_wb9_args.lk_pem_1[30][row] = 0u;  // zero
    for (int i = 0; i < 28; i++) {
        d_wb9_args.lk_pem_1[31 + i][row] = result_x_limbs[i];
    }
    for (int i = 0; i < 28; i++) {
        d_wb9_args.lk_pem_1[59 + i][row] = result_y_limbs[i];
    }

    // pedersen_points_table_0: [relation_id, table_idx, output_0..55]
    d_wb9_args.lk_ppt_0[0][row] = WB9_PPT_RELATION_ID;
    d_wb9_args.lk_ppt_0[1][row] = table_idx;
    for (int i = 0; i < 28; i++) {
        d_wb9_args.lk_ppt_0[2 + i][row] = table_x_limbs[i];
    }
    for (int i = 0; i < 28; i++) {
        d_wb9_args.lk_ppt_0[30 + i][row] = table_y_limbs[i];
    }

    // RC_20 lookup data
    wb9_write_rc_20_lookup(k1, carries1, 0, d_wb9_args.lk_rc_20, row);
    wb9_write_rc_20_lookup(k2, carries2, 1, d_wb9_args.lk_rc_20, row);
    wb9_write_rc_20_lookup(k3, carries3, 2, d_wb9_args.lk_rc_20, row);

    // RC_9_9 lookup data
    wb9_write_rc_9_9_lookup(slope_limbs, 0, d_wb9_args.lk_rc_9_9, row);
    wb9_write_rc_9_9_lookup(result_x_limbs, 1, d_wb9_args.lk_rc_9_9, row);
    wb9_write_rc_9_9_lookup(result_y_limbs, 2, d_wb9_args.lk_rc_9_9, row);
}

// ============================================================================
// Exported C functions
// ============================================================================

extern "C" void gen_partial_ec_mul_wb9_trace(
    m31** traces,
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_ppt_0,
    m31** lookup_rc_20,
    m31** lookup_rc_20_b,
    m31** lookup_rc_20_c,
    m31** lookup_rc_20_d,
    m31** lookup_rc_20_e,
    m31** lookup_rc_20_f,
    m31** lookup_rc_20_g,
    m31** lookup_rc_20_h,
    m31** lookup_rc_9_9,
    m31** lookup_rc_9_9_b,
    m31** lookup_rc_9_9_c,
    m31** lookup_rc_9_9_d,
    m31** lookup_rc_9_9_e,
    m31** lookup_rc_9_9_f,
    m31** lookup_rc_9_9_g,
    m31** lookup_rc_9_9_h,
    m31** sub_inputs_ppt,
    m31** sub_inputs_rc_9_9,
    m31** sub_inputs_rc_9_9_b,
    m31** sub_inputs_rc_9_9_c,
    m31** sub_inputs_rc_9_9_d,
    m31** sub_inputs_rc_9_9_e,
    m31** sub_inputs_rc_9_9_f,
    m31** sub_inputs_rc_9_9_g,
    m31** sub_inputs_rc_9_9_h,
    m31** sub_inputs_rc_20,
    m31** sub_inputs_rc_20_b,
    m31** sub_inputs_rc_20_c,
    m31** sub_inputs_rc_20_d,
    m31** sub_inputs_rc_20_e,
    m31** sub_inputs_rc_20_f,
    m31** sub_inputs_rc_20_g,
    m31** sub_inputs_rc_20_h,
    m31** inputs,
    uint32_t n_rows,
    uint32_t log_size
) {
    uint32_t trace_size = 1u << log_size;

    // Increase stack size for the kernel.
    // The kernel uses large local arrays (schoolbook ~3KB) plus deep call stacks
    // for Felt252 field operations (inverse requires many multiplications).
    // Stack size: set once, skip on subsequent calls (avoids device sync).
    static bool stack_set = false;
    if (!stack_set) {
        cudaDeviceSetLimit(cudaLimitStackSize, 32768);
        stack_set = true;
    }

    // Copy all host pointer arrays to device
    m31** d_traces = clone_to_device<m31*>(traces, 311);
    m31** d_inputs = clone_to_device<m31*>(inputs, 86);

    m31** d_sub_ppt = clone_to_device<m31*>(sub_inputs_ppt, 1);

    m31** d_sub_rc_9_9[8] = {
        clone_to_device<m31*>(sub_inputs_rc_9_9,   12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_b,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_c,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_d,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_e,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_f,  12),
        clone_to_device<m31*>(sub_inputs_rc_9_9_g,  6),
        clone_to_device<m31*>(sub_inputs_rc_9_9_h,  6),
    };

    m31** d_sub_rc_20[8] = {
        clone_to_device<m31*>(sub_inputs_rc_20,   12),
        clone_to_device<m31*>(sub_inputs_rc_20_b,  12),
        clone_to_device<m31*>(sub_inputs_rc_20_c,  12),
        clone_to_device<m31*>(sub_inputs_rc_20_d,  12),
        clone_to_device<m31*>(sub_inputs_rc_20_e,  9),
        clone_to_device<m31*>(sub_inputs_rc_20_f,  9),
        clone_to_device<m31*>(sub_inputs_rc_20_g,  9),
        clone_to_device<m31*>(sub_inputs_rc_20_h,  9),
    };

    m31** d_lk_pem_0 = clone_to_device<m31*>(lookup_partial_ec_mul_0, 87);
    m31** d_lk_pem_1 = clone_to_device<m31*>(lookup_partial_ec_mul_1, 87);
    m31** d_lk_ppt_0 = clone_to_device<m31*>(lookup_ppt_0, 58);

    m31** d_lk_rc_20[8] = {
        clone_to_device<m31*>(lookup_rc_20,   24),
        clone_to_device<m31*>(lookup_rc_20_b,  24),
        clone_to_device<m31*>(lookup_rc_20_c,  24),
        clone_to_device<m31*>(lookup_rc_20_d,  24),
        clone_to_device<m31*>(lookup_rc_20_e,  18),
        clone_to_device<m31*>(lookup_rc_20_f,  18),
        clone_to_device<m31*>(lookup_rc_20_g,  18),
        clone_to_device<m31*>(lookup_rc_20_h,  18),
    };

    m31** d_lk_rc_9_9[8] = {
        clone_to_device<m31*>(lookup_rc_9_9,   18),
        clone_to_device<m31*>(lookup_rc_9_9_b,  18),
        clone_to_device<m31*>(lookup_rc_9_9_c,  18),
        clone_to_device<m31*>(lookup_rc_9_9_d,  18),
        clone_to_device<m31*>(lookup_rc_9_9_e,  18),
        clone_to_device<m31*>(lookup_rc_9_9_f,  18),
        clone_to_device<m31*>(lookup_rc_9_9_g,  9),
        clone_to_device<m31*>(lookup_rc_9_9_h,  9),
    };

    // Fill kernel args struct and copy to constant memory
    PemWb9Args args;
    args.traces = d_traces;
    args.inputs = d_inputs;
    args.sub_ppt = d_sub_ppt;
    for (int i = 0; i < 8; i++) {
        args.sub_rc_9_9[i] = d_sub_rc_9_9[i];
        args.sub_rc_20[i] = d_sub_rc_20[i];
    }
    args.lk_pem_0 = d_lk_pem_0;
    args.lk_pem_1 = d_lk_pem_1;
    args.lk_ppt_0 = d_lk_ppt_0;
    for (int i = 0; i < 8; i++) {
        args.lk_rc_20[i] = d_lk_rc_20[i];
        args.lk_rc_9_9[i] = d_lk_rc_9_9[i];
    }
    args.n_rows = n_rows;
    args.trace_size = trace_size;

    cudaMemcpyToSymbol(d_wb9_args, &args, sizeof(PemWb9Args));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Launch kernel
    int grid_size = (trace_size + WB9_BLOCK_SIZE - 1) / WB9_BLOCK_SIZE;
    wb9_trace_kernel<<<grid_size, WB9_BLOCK_SIZE>>>();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Cleanup device pointer arrays
    cuda_free_memory(d_traces);
    cuda_free_memory(d_inputs);
    cuda_free_memory(d_sub_ppt);
    for (int i = 0; i < 8; i++) {
        cuda_free_memory(d_sub_rc_9_9[i]);
        cuda_free_memory(d_sub_rc_20[i]);
    }
    cuda_free_memory(d_lk_pem_0);
    cuda_free_memory(d_lk_pem_1);
    cuda_free_memory(d_lk_ppt_0);
    for (int i = 0; i < 8; i++) {
        cuda_free_memory(d_lk_rc_20[i]);
        cuda_free_memory(d_lk_rc_9_9[i]);
    }

    // Stack restore removed — keep 32KB permanently.
}

// ============================================================================
// Interaction trace kernel templates for wb9 (65 logup columns)
// ============================================================================

#define WB9_IT_BLOCK_SIZE 256

// Standard pair kernel: computes frac = (d0 + d1) / (d0 * d1)
template <int N, int M>
__launch_bounds__(WB9_IT_BLOCK_SIZE, 2)
__global__ void wb9_it_col_gen_kernel(
    LookupElementsBasic<N>* lookup_elements_n,
    LookupElementsBasic<M>* lookup_elements_m,
    m31** lookup_state_0,
    m31** lookup_state_1,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    m31 reg0[N] = {};
    m31 reg1[M] = {};

    for (int i = 0; i < N; i++) reg0[i] = lookup_state_0[i][idx];
    for (int i = 0; i < M; i++) reg1[i] = lookup_state_1[i][idx];

    if (idx < trace_size) {
        qm31 d0 = lookup_elements_n->combine(reg0, N);
        qm31 d1 = lookup_elements_m->combine(reg1, M);
        logup_col_write_frac(idx, add(d0, d1), mul(d0, d1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Enabler pair kernel: frac = (d0 * en + d1) / (d0 * d1)
template <int N, int M>
__launch_bounds__(WB9_IT_BLOCK_SIZE, 2)
__global__ void wb9_it_enabler_col_gen_kernel(
    LookupElementsBasic<N>* lookup_elements_n,
    LookupElementsBasic<M>* lookup_elements_m,
    m31** lookup_state_0,
    m31** lookup_state_1,
    unsigned n_rows,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    qm31 en = {0};
    if (idx < n_rows) en = {1};

    m31 reg0[N] = {};
    m31 reg1[M] = {};

    for (int i = 0; i < N; i++) reg0[i] = lookup_state_0[i][idx];
    for (int i = 0; i < M; i++) reg1[i] = lookup_state_1[i][idx];

    if (idx < trace_size) {
        qm31 d0 = lookup_elements_n->combine(reg0, N);
        qm31 d1 = lookup_elements_m->combine(reg1, M);
        logup_col_write_frac(idx, add(mul(d0, en), d1), mul(d0, d1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Negative enabler kernel: frac = -en / d
template <int N>
__launch_bounds__(WB9_IT_BLOCK_SIZE, 2)
__global__ void wb9_it_neg_enabler_col_gen_kernel(
    LookupElementsBasic<N>* lookup_elements_n,
    m31** lookup_state_0,
    unsigned n_rows,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    qm31 en = {0};
    if (idx < n_rows) en = {1};

    m31 reg0[N] = {};
    for (int i = 0; i < N; i++) reg0[i] = lookup_state_0[i][idx];

    if (idx < trace_size) {
        qm31 d = lookup_elements_n->combine(reg0, N);
        logup_col_write_frac(idx, mul(qm31{P-1, 0, 0, 0}, en), d,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize kernel: multiply numerator by inverse denominator and accumulate
__global__ void wb9_it_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = rep_index - 1;

    if (idx < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[idx], numerator1[idx]},
                cm31{numerator2[idx], numerator3[idx]}
            },
            denom_inv_ptr[idx]
        );

        if (pre_index == -1) {
            interaction_traces[0][idx] = 0;
            interaction_traces[1][idx] = 0;
            interaction_traces[2][idx] = 0;
            interaction_traces[3][idx] = 0;
            qm31 tmp = value;
            numerator0[idx] = tmp.a.a;
            numerator1[idx] = tmp.a.b;
            numerator2[idx] = tmp.b.a;
            numerator3[idx] = tmp.b.b;
        } else {
            qm31 pre_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][idx], interaction_traces[pre_index * 4 + 1][idx]},
                cm31{interaction_traces[pre_index * 4 + 2][idx], interaction_traces[pre_index * 4 + 3][idx]}
            };
            qm31 tmp = add(value, pre_value);
            numerator0[idx] = tmp.a.a;
            numerator1[idx] = tmp.a.b;
            numerator2[idx] = tmp.b.a;
            numerator3[idx] = tmp.b.b;
        }

        interaction_traces[rep_index * 4 + 0][idx] = numerator0[idx];
        interaction_traces[rep_index * 4 + 1][idx] = numerator1[idx];
        interaction_traces[rep_index * 4 + 2][idx] = numerator2[idx];
        interaction_traces[rep_index * 4 + 3][idx] = numerator3[idx];
    }
}

// Cumsum shift kernel (parameterized N_COLS)
__global__ void wb9_it_cumsum_shift(
    unsigned n_cols,
    unsigned trace_size,
    m31** interactive_traces,
    m31* coordinate_sums
) {
    int idx0 = 4 * n_cols - 4;
    int idx1 = 4 * n_cols - 3;
    int idx2 = 4 * n_cols - 2;
    int idx3 = 4 * n_cols - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = tid; i < trace_size; i += gridSize) {
        s0 = add(s0, interactive_traces[idx0][i]);
        s1 = add(s1, interactive_traces[idx1][i]);
        s2 = add(s2, interactive_traces[idx2][i]);
        s3 = add(s3, interactive_traces[idx3][i]);
    }

    extern __shared__ m31 shared[];
    m31* sd0 = &shared[0];
    m31* sd1 = &shared[blockDim.x];
    m31* sd2 = &shared[2 * blockDim.x];
    m31* sd3 = &shared[3 * blockDim.x];

    sd0[threadIdx.x] = s0;
    sd1[threadIdx.x] = s1;
    sd2[threadIdx.x] = s2;
    sd3[threadIdx.x] = s3;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sd0[threadIdx.x] = add(sd0[threadIdx.x], sd0[threadIdx.x + s]);
            sd1[threadIdx.x] = add(sd1[threadIdx.x], sd1[threadIdx.x + s]);
            sd2[threadIdx.x] = add(sd2[threadIdx.x], sd2[threadIdx.x + s]);
            sd3[threadIdx.x] = add(sd3[threadIdx.x], sd3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sd0[0]);
        atomic_add(&coordinate_sums[1], sd1[0]);
        atomic_add(&coordinate_sums[2], sd2[0]);
        atomic_add(&coordinate_sums[3], sd3[0]);
    }
}

// Coordinate prefix sum kernel (parameterized N_COLS)
__global__ void wb9_it_coord_prefix_sum(
    m31* coordinate_sums,
    unsigned n_cols,
    unsigned trace_size,
    m31** interactive_traces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        qm31 cs = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 shift = div(cs, m31(trace_size));

        interactive_traces[4 * n_cols - 4][idx] = sub(interactive_traces[4 * n_cols - 4][idx], shift.a.a);
        interactive_traces[4 * n_cols - 3][idx] = sub(interactive_traces[4 * n_cols - 3][idx], shift.a.b);
        interactive_traces[4 * n_cols - 2][idx] = sub(interactive_traces[4 * n_cols - 2][idx], shift.b.a);
        interactive_traces[4 * n_cols - 1][idx] = sub(interactive_traces[4 * n_cols - 1][idx], shift.b.b);
    }
}

// Helper macro for standard pair columns
#define WB9_PROCESS_COL(col_idx, elem1, elem2, lookup1, lookup2, N1, N2) \
    wb9_it_col_gen_kernel<N1, N2><<<num_blocks, block_dim>>>( \
        elem1, elem2, lookup1, lookup2, trace_size, \
        device_logup_denom, numerator0, numerator1, numerator2, numerator3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col_idx, trace_size, denom_inv, \
        numerator0, numerator1, numerator2, numerator3, device_it); \

// ============================================================================
// Interaction trace type aliases for wb9
// ============================================================================

typedef LookupElementsBasic<86> PemLookup;
typedef LookupElementsBasic<57> PptLookup;
typedef LookupElementsBasic<2> Rc99Lookup;
typedef LookupElementsBasic<1> Rc20Lookup;

// ============================================================================
// Main interaction trace generation function
// ============================================================================

extern "C" void gen_partial_ec_mul_wb9_interaction_trace(
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
    m31** lookup_partial_ec_mul_0,
    m31** lookup_partial_ec_mul_1,
    m31** lookup_ppt_0,
    m31** lookup_rc_20,
    m31** lookup_rc_20_b,
    m31** lookup_rc_20_c,
    m31** lookup_rc_20_d,
    m31** lookup_rc_20_e,
    m31** lookup_rc_20_f,
    m31** lookup_rc_20_g,
    m31** lookup_rc_20_h,
    m31** lookup_rc_9_9,
    m31** lookup_rc_9_9_b,
    m31** lookup_rc_9_9_c,
    m31** lookup_rc_9_9_d,
    m31** lookup_rc_9_9_e,
    m31** lookup_rc_9_9_f,
    m31** lookup_rc_9_9_g,
    m31** lookup_rc_9_9_h,
    uint32_t n_rows,
    uint32_t log_size,
    m31** interaction_trace_columns,
    m31* claimed_sum
) {
    uint32_t trace_size = 1u << log_size;

    // ========================================================================
    // Copy lookup elements to device (18 total: pem, ppt, 8xrc20, 8xrc99)
    // ========================================================================

    PemLookup* d_pem = cuda_malloc<PemLookup>(1);
    PptLookup* d_ppt = cuda_malloc<PptLookup>(1);
    Rc20Lookup* d_rc20[8];
    Rc99Lookup* d_rc99[8];
    for (int i = 0; i < 8; i++) {
        d_rc20[i] = cuda_malloc<Rc20Lookup>(1);
        d_rc99[i] = cuda_malloc<Rc99Lookup>(1);
    }

    cuda_mem_copy_host_to_device<PemLookup>((PemLookup*)partial_ec_mul_lookup_elements, d_pem, 1);
    cuda_mem_copy_host_to_device<PptLookup>((PptLookup*)pedersen_points_table_lookup_elements, d_ppt, 1);

    void* rc20_elems[8] = {
        rc_20_lookup_elements, rc_20_b_lookup_elements,
        rc_20_c_lookup_elements, rc_20_d_lookup_elements,
        rc_20_e_lookup_elements, rc_20_f_lookup_elements,
        rc_20_g_lookup_elements, rc_20_h_lookup_elements
    };
    void* rc99_elems[8] = {
        rc_9_9_lookup_elements, rc_9_9_b_lookup_elements,
        rc_9_9_c_lookup_elements, rc_9_9_d_lookup_elements,
        rc_9_9_e_lookup_elements, rc_9_9_f_lookup_elements,
        rc_9_9_g_lookup_elements, rc_9_9_h_lookup_elements
    };
    for (int i = 0; i < 8; i++) {
        cuda_mem_copy_host_to_device<Rc20Lookup>((Rc20Lookup*)rc20_elems[i], d_rc20[i], 1);
        cuda_mem_copy_host_to_device<Rc99Lookup>((Rc99Lookup*)rc99_elems[i], d_rc99[i], 1);
    }

    // ========================================================================
    // Clone lookup data arrays to device
    // ========================================================================

    // PEM self-lookups (87 arrays each, but kernel uses shifted elements -> index 1..86)
    m31** d_lk_pem_0 = clone_to_device<m31*>(lookup_partial_ec_mul_0, 87);
    m31** d_lk_pem_1 = clone_to_device<m31*>(lookup_partial_ec_mul_1, 87);

    // PPT lookup (58 arrays, kernel uses index 1..57)
    m31** d_lk_ppt = clone_to_device<m31*>(lookup_ppt_0, 58);

    // RC_20 variants (2 elements per entry, includes relation constant)
    m31** lk_rc20_host[8] = {
        lookup_rc_20, lookup_rc_20_b, lookup_rc_20_c, lookup_rc_20_d,
        lookup_rc_20_e, lookup_rc_20_f, lookup_rc_20_g, lookup_rc_20_h
    };
    int rc20_flat[8] = {24, 24, 24, 24, 18, 18, 18, 18};
    m31** d_lk_rc20[8];
    for (int i = 0; i < 8; i++) {
        d_lk_rc20[i] = clone_to_device<m31*>(lk_rc20_host[i], rc20_flat[i]);
    }

    // RC_9_9 variants (3 elements per entry, includes relation constant)
    m31** lk_rc99_host[8] = {
        lookup_rc_9_9, lookup_rc_9_9_b, lookup_rc_9_9_c, lookup_rc_9_9_d,
        lookup_rc_9_9_e, lookup_rc_9_9_f, lookup_rc_9_9_g, lookup_rc_9_9_h
    };
    int rc99_flat[8] = {18, 18, 18, 18, 18, 18, 9, 9};
    m31** d_lk_rc99[8];
    for (int i = 0; i < 8; i++) {
        d_lk_rc99[i] = clone_to_device<m31*>(lk_rc99_host[i], rc99_flat[i]);
    }

    // ========================================================================
    // Allocate working memory
    // ========================================================================

    qm31* device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31* denom_inv = cuda_malloc<qm31>(trace_size);
    m31* numerator0 = cuda_malloc<m31>(trace_size);
    m31* numerator1 = cuda_malloc<m31>(trace_size);
    m31* numerator2 = cuda_malloc<m31>(trace_size);
    m31* numerator3 = cuda_malloc<m31>(trace_size);

    m31** device_it = clone_to_device<m31*>(interaction_trace_columns, 4 * PEM_WB9_N_LOGUP_COLUMNS);

    int block_dim = trace_size < WB9_IT_BLOCK_SIZE ? trace_size : WB9_IT_BLOCK_SIZE;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;
    int block_dim_fin = block_dim;
    int num_blocks_fin = num_blocks;

    // ========================================================================
    // Process 65 interaction columns
    // Column pairing matches Rust standard_pairs exactly
    //
    // Lookup data layout:
    //   rc_20[variant]:  [rel_const, val] per entry -> 2*count arrays
    //                    Entry e at offsets [2*e, 2*e+1]
    //                    Kernel uses shifted lookup elements -> starts at index 1
    //   rc_9_9[variant]: [rel_const, val0, val1] per entry -> 3*count arrays
    //                    Entry e at offsets [3*e, 3*e+1, 3*e+2]
    //                    Kernel uses shifted lookup elements -> starts at index 1
    //   ppt:             [rel_const, table_idx, x0..x27, y0..y27] -> 58 arrays
    //                    Kernel uses shifted -> starts at index 1
    //   pem:             [rel_const, 86 values] -> 87 arrays
    //                    Kernel uses shifted -> starts at index 1
    //
    // Since create_modified_lookup_for_cuda shifts alpha_powers by 1 and
    // absorbs rel_const into z, the kernel's combine(vals, N) will read
    // N values starting from the pointer given. We must point to index 1
    // (skipping the rel_const at index 0) for all lookups.
    // ========================================================================

    int col = 0;

    // Col 0: (Ppt0, Rc99(0,0))   -- ppt uses 57 vals, rc99 uses 2 vals
    wb9_it_col_gen_kernel<57, 2><<<num_blocks, block_dim>>>(
        d_ppt, d_rc99[0], &d_lk_ppt[1], &d_lk_rc99[0][1], trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // Col 1: (Rc99(1,0), Rc99(2,0))
    WB9_PROCESS_COL(col++, d_rc99[1], d_rc99[2], &d_lk_rc99[1][1], &d_lk_rc99[2][1], 2, 2);
    // Col 2: (Rc99(3,0), Rc99(4,0))
    WB9_PROCESS_COL(col++, d_rc99[3], d_rc99[4], &d_lk_rc99[3][1], &d_lk_rc99[4][1], 2, 2);
    // Col 3: (Rc99(5,0), Rc99(6,0))
    WB9_PROCESS_COL(col++, d_rc99[5], d_rc99[6], &d_lk_rc99[5][1], &d_lk_rc99[6][1], 2, 2);
    // Col 4: (Rc99(7,0), Rc99(0,1))
    WB9_PROCESS_COL(col++, d_rc99[7], d_rc99[0], &d_lk_rc99[7][1], &d_lk_rc99[0][4], 2, 2);
    // Col 5: (Rc99(1,1), Rc99(2,1))
    WB9_PROCESS_COL(col++, d_rc99[1], d_rc99[2], &d_lk_rc99[1][4], &d_lk_rc99[2][4], 2, 2);
    // Col 6: (Rc99(3,1), Rc99(4,1))
    WB9_PROCESS_COL(col++, d_rc99[3], d_rc99[4], &d_lk_rc99[3][4], &d_lk_rc99[4][4], 2, 2);
    // Col 7: (Rc99(5,1), Rc20(0,0))
    wb9_it_col_gen_kernel<2, 1><<<num_blocks, block_dim>>>(
        d_rc99[5], d_rc20[0], &d_lk_rc99[5][4], &d_lk_rc20[0][1], trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // Col 8: (Rc20(1,0), Rc20(2,0))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][1], &d_lk_rc20[2][1], 1, 1);
    // Col 9: (Rc20(3,0), Rc20(4,0))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][1], &d_lk_rc20[4][1], 1, 1);
    // Col 10: (Rc20(5,0), Rc20(6,0))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][1], &d_lk_rc20[6][1], 1, 1);
    // Col 11: (Rc20(7,0), Rc20(0,1))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][1], &d_lk_rc20[0][3], 1, 1);
    // Col 12: (Rc20(1,1), Rc20(2,1))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][3], &d_lk_rc20[2][3], 1, 1);
    // Col 13: (Rc20(3,1), Rc20(4,1))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][3], &d_lk_rc20[4][3], 1, 1);
    // Col 14: (Rc20(5,1), Rc20(6,1))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][3], &d_lk_rc20[6][3], 1, 1);
    // Col 15: (Rc20(7,1), Rc20(0,2))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][3], &d_lk_rc20[0][5], 1, 1);
    // Col 16: (Rc20(1,2), Rc20(2,2))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][5], &d_lk_rc20[2][5], 1, 1);
    // Col 17: (Rc20(3,2), Rc20(4,2))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][5], &d_lk_rc20[4][5], 1, 1);
    // Col 18: (Rc20(5,2), Rc20(6,2))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][5], &d_lk_rc20[6][5], 1, 1);
    // Col 19: (Rc20(7,2), Rc20(0,3))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][5], &d_lk_rc20[0][7], 1, 1);
    // Col 20: (Rc20(1,3), Rc20(2,3))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][7], &d_lk_rc20[2][7], 1, 1);

    // Col 21: (Rc20(3,3), Rc99(0,2))
    wb9_it_col_gen_kernel<1, 2><<<num_blocks, block_dim>>>(
        d_rc20[3], d_rc99[0], &d_lk_rc20[3][7], &d_lk_rc99[0][7], trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // Col 22: (Rc99(1,2), Rc99(2,2))
    WB9_PROCESS_COL(col++, d_rc99[1], d_rc99[2], &d_lk_rc99[1][7], &d_lk_rc99[2][7], 2, 2);
    // Col 23: (Rc99(3,2), Rc99(4,2))
    WB9_PROCESS_COL(col++, d_rc99[3], d_rc99[4], &d_lk_rc99[3][7], &d_lk_rc99[4][7], 2, 2);
    // Col 24: (Rc99(5,2), Rc99(6,1))
    WB9_PROCESS_COL(col++, d_rc99[5], d_rc99[6], &d_lk_rc99[5][7], &d_lk_rc99[6][4], 2, 2);
    // Col 25: (Rc99(7,1), Rc99(0,3))
    WB9_PROCESS_COL(col++, d_rc99[7], d_rc99[0], &d_lk_rc99[7][4], &d_lk_rc99[0][10], 2, 2);
    // Col 26: (Rc99(1,3), Rc99(2,3))
    WB9_PROCESS_COL(col++, d_rc99[1], d_rc99[2], &d_lk_rc99[1][10], &d_lk_rc99[2][10], 2, 2);
    // Col 27: (Rc99(3,3), Rc99(4,3))
    WB9_PROCESS_COL(col++, d_rc99[3], d_rc99[4], &d_lk_rc99[3][10], &d_lk_rc99[4][10], 2, 2);

    // Col 28: (Rc99(5,3), Rc20(0,4))
    wb9_it_col_gen_kernel<2, 1><<<num_blocks, block_dim>>>(
        d_rc99[5], d_rc20[0], &d_lk_rc99[5][10], &d_lk_rc20[0][9], trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // Col 29: (Rc20(1,4), Rc20(2,4))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][9], &d_lk_rc20[2][9], 1, 1);
    // Col 30: (Rc20(3,4), Rc20(4,3))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][9], &d_lk_rc20[4][7], 1, 1);
    // Col 31: (Rc20(5,3), Rc20(6,3))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][7], &d_lk_rc20[6][7], 1, 1);
    // Col 32: (Rc20(7,3), Rc20(0,5))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][7], &d_lk_rc20[0][11], 1, 1);
    // Col 33: (Rc20(1,5), Rc20(2,5))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][11], &d_lk_rc20[2][11], 1, 1);
    // Col 34: (Rc20(3,5), Rc20(4,4))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][11], &d_lk_rc20[4][9], 1, 1);
    // Col 35: (Rc20(5,4), Rc20(6,4))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][9], &d_lk_rc20[6][9], 1, 1);
    // Col 36: (Rc20(7,4), Rc20(0,6))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][9], &d_lk_rc20[0][13], 1, 1);
    // Col 37: (Rc20(1,6), Rc20(2,6))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][13], &d_lk_rc20[2][13], 1, 1);
    // Col 38: (Rc20(3,6), Rc20(4,5))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][13], &d_lk_rc20[4][11], 1, 1);
    // Col 39: (Rc20(5,5), Rc20(6,5))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][11], &d_lk_rc20[6][11], 1, 1);
    // Col 40: (Rc20(7,5), Rc20(0,7))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][11], &d_lk_rc20[0][15], 1, 1);
    // Col 41: (Rc20(1,7), Rc20(2,7))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][15], &d_lk_rc20[2][15], 1, 1);

    // Col 42: (Rc20(3,7), Rc99(0,4))
    wb9_it_col_gen_kernel<1, 2><<<num_blocks, block_dim>>>(
        d_rc20[3], d_rc99[0], &d_lk_rc20[3][15], &d_lk_rc99[0][13], trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // Col 43: (Rc99(1,4), Rc99(2,4))
    WB9_PROCESS_COL(col++, d_rc99[1], d_rc99[2], &d_lk_rc99[1][13], &d_lk_rc99[2][13], 2, 2);
    // Col 44: (Rc99(3,4), Rc99(4,4))
    WB9_PROCESS_COL(col++, d_rc99[3], d_rc99[4], &d_lk_rc99[3][13], &d_lk_rc99[4][13], 2, 2);
    // Col 45: (Rc99(5,4), Rc99(6,2))
    WB9_PROCESS_COL(col++, d_rc99[5], d_rc99[6], &d_lk_rc99[5][13], &d_lk_rc99[6][7], 2, 2);
    // Col 46: (Rc99(7,2), Rc99(0,5))
    WB9_PROCESS_COL(col++, d_rc99[7], d_rc99[0], &d_lk_rc99[7][7], &d_lk_rc99[0][16], 2, 2);
    // Col 47: (Rc99(1,5), Rc99(2,5))
    WB9_PROCESS_COL(col++, d_rc99[1], d_rc99[2], &d_lk_rc99[1][16], &d_lk_rc99[2][16], 2, 2);
    // Col 48: (Rc99(3,5), Rc99(4,5))
    WB9_PROCESS_COL(col++, d_rc99[3], d_rc99[4], &d_lk_rc99[3][16], &d_lk_rc99[4][16], 2, 2);

    // Col 49: (Rc99(5,5), Rc20(0,8))
    wb9_it_col_gen_kernel<2, 1><<<num_blocks, block_dim>>>(
        d_rc99[5], d_rc20[0], &d_lk_rc99[5][16], &d_lk_rc20[0][17], trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // Col 50: (Rc20(1,8), Rc20(2,8))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][17], &d_lk_rc20[2][17], 1, 1);
    // Col 51: (Rc20(3,8), Rc20(4,6))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][17], &d_lk_rc20[4][13], 1, 1);
    // Col 52: (Rc20(5,6), Rc20(6,6))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][13], &d_lk_rc20[6][13], 1, 1);
    // Col 53: (Rc20(7,6), Rc20(0,9))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][13], &d_lk_rc20[0][19], 1, 1);
    // Col 54: (Rc20(1,9), Rc20(2,9))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][19], &d_lk_rc20[2][19], 1, 1);
    // Col 55: (Rc20(3,9), Rc20(4,7))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][19], &d_lk_rc20[4][15], 1, 1);
    // Col 56: (Rc20(5,7), Rc20(6,7))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][15], &d_lk_rc20[6][15], 1, 1);
    // Col 57: (Rc20(7,7), Rc20(0,10))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][15], &d_lk_rc20[0][21], 1, 1);
    // Col 58: (Rc20(1,10), Rc20(2,10))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][21], &d_lk_rc20[2][21], 1, 1);
    // Col 59: (Rc20(3,10), Rc20(4,8))
    WB9_PROCESS_COL(col++, d_rc20[3], d_rc20[4], &d_lk_rc20[3][21], &d_lk_rc20[4][17], 1, 1);
    // Col 60: (Rc20(5,8), Rc20(6,8))
    WB9_PROCESS_COL(col++, d_rc20[5], d_rc20[6], &d_lk_rc20[5][17], &d_lk_rc20[6][17], 1, 1);
    // Col 61: (Rc20(7,8), Rc20(0,11))
    WB9_PROCESS_COL(col++, d_rc20[7], d_rc20[0], &d_lk_rc20[7][17], &d_lk_rc20[0][23], 1, 1);
    // Col 62: (Rc20(1,11), Rc20(2,11))
    WB9_PROCESS_COL(col++, d_rc20[1], d_rc20[2], &d_lk_rc20[1][23], &d_lk_rc20[2][23], 1, 1);

    // Col 63: ENABLER -- (Rc20(3,11), Pem0)  frac = (d0*en + d1) / (d0*d1)
    wb9_it_enabler_col_gen_kernel<1, 86><<<num_blocks, block_dim>>>(
        d_rc20[3], d_pem, &d_lk_rc20[3][23], &d_lk_pem_0[1], n_rows, trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // Col 64: NEG_ENABLER -- (Pem1)  frac = -en / d
    wb9_it_neg_enabler_col_gen_kernel<86><<<num_blocks, block_dim>>>(
        d_pem, &d_lk_pem_1[1], n_rows, trace_size,
        device_logup_denom, numerator0, numerator1, numerator2, numerator3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    wb9_it_finalize_col_kernel<<<num_blocks_fin, block_dim_fin>>>(col++, trace_size, denom_inv,
        numerator0, numerator1, numerator2, numerator3, device_it);

    // ========================================================================
    // Finalize: cumsum shift + prefix sum
    // ========================================================================

    cudaMemsetAsync(claimed_sum, 0, 4 * sizeof(m31), 0);

    size_t shared_size = 4 * block_dim * sizeof(m31);
    wb9_it_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        PEM_WB9_N_LOGUP_COLUMNS, trace_size, device_it, claimed_sum);

    wb9_it_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum, PEM_WB9_N_LOGUP_COLUMNS, trace_size, device_it);

    // Inclusive prefix sum on last 4 columns
    inclusive_prefix_sum(interaction_trace_columns[4 * PEM_WB9_N_LOGUP_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * PEM_WB9_N_LOGUP_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * PEM_WB9_N_LOGUP_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * PEM_WB9_N_LOGUP_COLUMNS - 1], trace_size);

    // ========================================================================
    // Cleanup
    // ========================================================================

    cuda_free_memory(d_pem);
    cuda_free_memory(d_ppt);
    for (int i = 0; i < 8; i++) {
        cuda_free_memory(d_rc20[i]);
        cuda_free_memory(d_rc99[i]);
    }

    cuda_free_memory(d_lk_pem_0);
    cuda_free_memory(d_lk_pem_1);
    cuda_free_memory(d_lk_ppt);
    for (int i = 0; i < 8; i++) {
        cuda_free_memory(d_lk_rc20[i]);
        cuda_free_memory(d_lk_rc99[i]);
    }

    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(numerator0);
    cuda_free_memory(numerator1);
    cuda_free_memory(numerator2);
    cuda_free_memory(numerator3);
    cuda_free_memory(device_it);
}
