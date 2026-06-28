/**
 * CUDA trace generation for pedersen_aggregator_window_bits_18 (206-col trace).
 *
 * Generates the pedersen aggregator base trace directly on GPU, using the
 * GPU-resident pedersen table (g_pedersen_table_columns). No CPU table needed.
 *
 * Architecture:
 *   - 1 thread per row
 *   - Uses memory_id_to_big_state_deduce_output() from gen_memory_id_to_big_trace.cuh
 *   - Uses fp256 field arithmetic from fp256_config.cuh / fp256_dispatch_st.cuh
 *   - Uses g_pedersen_table_columns from pedersen_table_init.cu
 *
 * Column layout (206 columns):
 *   0-2:     Input limbs (3 cols)
 *   3-30:    memory_id_to_big(input_0) -> value_a (28 M31 limbs)
 *   31-58:   memory_id_to_big(input_1) -> value_b (28 M31 limbs)
 *   59-61:   Verify Reduced 252 for value A (ms_is_max, ms_and_mid_max, rc_input)
 *   62-64:   Verify Reduced 252 for value B (ms_is_max, ms_and_mid_max, rc_input)
 *   65-134:  PEM chain 0 output (14 zeros + 28 result_x + 28 result_y)
 *   135-204: PEM chain 1 output (14 zeros + 28 result_x + 28 result_y)
 *   205:     Multiplicity
 *
 * EC point addition (affine):
 *   slope = (y2 - y1) / (x2 - x1)
 *   result_x = slope^2 - x1 - x2
 *   result_y = slope * (x1 - result_x) - y1
 */

#include "gen_pedersen_aggregator_wb18_trace.cuh"
#include "../fields.cuh"
#include "../fp256_config.cuh"
#include "../fp256_dispatch_st.cuh"
#include "../utils.cuh"
#include "../logup.cuh"
#include "../batch_inverse.cuh"
#include "../prefix_sum.cuh"
#include "gen_memory_id_to_big_trace.cuh"

// ============================================================================
// Pedersen table -- defined in pedersen_table_init.cu
// ============================================================================

#define PEDERSEN_TABLE_N_COLUMNS 56
extern __device__ m31* g_pedersen_table_columns[PEDERSEN_TABLE_N_COLUMNS];

// ============================================================================
// Block size for kernel launch
// ============================================================================

#define AGG_BLOCK_SIZE 256

// ============================================================================
// Relation ID constants (from SIMD reference)
// ============================================================================

#define AGG_MEM_ID_TO_BIG_RELATION_ID  1662111297u
#define AGG_RC_8_RELATION_ID           1420243005u
#define AGG_PEM_RELATION_ID            1621226978u
#define AGG_SELF_RELATION_ID           520578465u

// ============================================================================
// Initial accumulator constants -- the Starknet shift point as 28 M31 limbs
// ============================================================================

__constant__ uint32_t AGG_SHIFT_POINT_X_LIMBS[28] = {
    510, 315, 208, 480, 418, 115, 155, 54,
    162, 449, 428, 466, 484, 169, 497, 373,
    98, 64, 464, 498, 124, 68, 379, 140,
    26, 22, 135, 202
};

__constant__ uint32_t AGG_SHIFT_POINT_Y_LIMBS[28] = {
    156, 120, 213, 389, 377, 20, 325, 303,
    473, 334, 223, 160, 225, 297, 101, 420,
    377, 72, 191, 49, 314, 27, 199, 222,
    79, 97, 108, 141
};

// ============================================================================
// Felt252 type and field operations (prefixed with agg_ to avoid linker
// conflicts with wb18 kernel which has identical static device functions)
// ============================================================================

typedef ff_storage<8> Felt252Field;

static __device__ __forceinline__ Felt252Field agg_felt_add(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::add(a, b);
}

static __device__ __forceinline__ Felt252Field agg_felt_sub(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::sub(a, b);
}

static __device__ __forceinline__ Felt252Field agg_felt_to_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::to_montgomery(a);
}

static __device__ __forceinline__ Felt252Field agg_felt_from_mont(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::from_montgomery(a);
}

static __device__ __forceinline__ Felt252Field agg_felt_mul(
    const Felt252Field& a, const Felt252Field& b) {
    return ff_dispatch_st<ff_config_starknet>::mul(a, b);
}

static __device__ __forceinline__ Felt252Field agg_felt_inverse(const Felt252Field& a) {
    return ff_dispatch_st<ff_config_starknet>::inverse(a);
}

// ============================================================================
// Limb conversion utilities
// ============================================================================

// Convert 28 x 9-bit M31 limbs to Felt252Field (standard form)
static __device__ Felt252Field agg_limbs28_to_felt252(const m31* limbs) {
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
static __device__ void agg_felt252_to_limbs28(const Felt252Field& felt, m31* limbs) {
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
// EC point addition (affine coordinates, full Felt252 arithmetic)
// ============================================================================

static __device__ void agg_ec_point_add(
    const m31* acc_x_limbs,
    const m31* acc_y_limbs,
    const m31* table_x_limbs,
    const m31* table_y_limbs,
    m31* result_x_limbs,
    m31* result_y_limbs
) {
    Felt252Field acc_x = agg_limbs28_to_felt252(acc_x_limbs);
    Felt252Field acc_y = agg_limbs28_to_felt252(acc_y_limbs);
    Felt252Field table_x = agg_limbs28_to_felt252(table_x_limbs);
    Felt252Field table_y = agg_limbs28_to_felt252(table_y_limbs);

    // slope = (table_y - acc_y) / (table_x - acc_x)
    Felt252Field dy = agg_felt_sub(table_y, acc_y);
    Felt252Field dx = agg_felt_sub(table_x, acc_x);
    Felt252Field num_mont = agg_felt_to_mont(dy);
    Felt252Field denom_mont = agg_felt_to_mont(dx);
    Felt252Field slope_mont = agg_felt_mul(num_mont, agg_felt_inverse(denom_mont));
    Felt252Field slope = agg_felt_from_mont(slope_mont);

    // result_x = slope^2 - acc_x - table_x
    Felt252Field slope_sq = agg_felt_from_mont(
        agg_felt_mul(agg_felt_to_mont(slope), agg_felt_to_mont(slope)));
    Felt252Field rx = agg_felt_sub(agg_felt_sub(slope_sq, acc_x), table_x);

    // result_y = slope * (acc_x - result_x) - acc_y
    Felt252Field ry = agg_felt_sub(
        agg_felt_from_mont(
            agg_felt_mul(agg_felt_to_mont(slope),
                         agg_felt_to_mont(agg_felt_sub(acc_x, rx)))),
        acc_y);

    agg_felt252_to_limbs28(rx, result_x_limbs);
    agg_felt252_to_limbs28(ry, result_y_limbs);
}

// ============================================================================
// Kernel arguments (passed via constant memory)
// ============================================================================

struct agg_kernel_args {
    m31** traces;                   // 206 trace output columns
    m31** inputs;                   // 3 input columns
    unsigned** transpose_big_value_ptr;
    unsigned* small_value_ptr;
    uint32_t n_rows;
    uint32_t trace_size;
    // Lookup data arrays
    m31** lk_mem_0;                 // 30 arrays
    m31** lk_mem_1;                 // 30 arrays
    m31** lk_mem_2;                 // 30 arrays
    m31** lk_rc8_0;                 // 2 arrays
    m31** lk_rc8_1;                 // 2 arrays
    m31** lk_rc8_2;                 // 2 arrays
    m31** lk_rc8_3;                 // 2 arrays
    m31** lk_pem_0;                 // 73 arrays
    m31** lk_pem_1;                 // 73 arrays
    m31** lk_pem_2;                 // 73 arrays
    m31** lk_pem_3;                 // 73 arrays
    m31** lk_agg_0;                 // 4 arrays
    m31* mults;                     // multiplicity column (already populated by Rust)
    // Sub-component inputs
    m31** sub_mem;                  // 3 arrays (one per memory_id_to_big feed)
    m31** sub_rc8;                  // 4 arrays (one per range_check_8 feed)
    m31** sub_pem;                  // 72 arrays, each 28*trace_size (sub_pem[col][round*trace_size + row])
};

__constant__ agg_kernel_args d_agg_args;

// ============================================================================
// Main trace generation kernel
// ============================================================================

__global__ void __launch_bounds__(AGG_BLOCK_SIZE, 2)
agg_trace_kernel() {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_agg_args.trace_size) return;

    m31** traces = d_agg_args.traces;
    m31** inputs = d_agg_args.inputs;

    // ==================================================================
    // 1. Read 3 inputs -> trace cols 0-2
    // ==================================================================
    m31 input_limb_0 = inputs[0][row];
    m31 input_limb_1 = inputs[1][row];
    m31 input_limb_2 = inputs[2][row];

    traces[0][row] = input_limb_0;
    traces[1][row] = input_limb_1;
    traces[2][row] = input_limb_2;

    // ==================================================================
    // 2. memory_id_to_big for input_0 -> value_a (28 limbs)
    //    trace cols 3-30
    // ==================================================================
    m31 value_a_limbs[28];
    memory_id_to_big_state_deduce_output(
        d_agg_args.transpose_big_value_ptr,
        d_agg_args.small_value_ptr,
        (unsigned)input_limb_0,
        value_a_limbs);

    for (int i = 0; i < 28; i++) {
        traces[3 + i][row] = value_a_limbs[i];
    }

    // ==================================================================
    // 3. memory_id_to_big for input_1 -> value_b (28 limbs)
    //    trace cols 31-58
    // ==================================================================
    m31 value_b_limbs[28];
    memory_id_to_big_state_deduce_output(
        d_agg_args.transpose_big_value_ptr,
        d_agg_args.small_value_ptr,
        (unsigned)input_limb_1,
        value_b_limbs);

    for (int i = 0; i < 28; i++) {
        traces[31 + i][row] = value_b_limbs[i];
    }

    // ==================================================================
    // 4. Verify Reduced 252 for value A -> trace cols 59-61
    // ==================================================================
    uint32_t ms_is_max_a = ((uint32_t)value_a_limbs[27] == 256u) ? 1u : 0u;
    uint32_t ms_and_mid_max_a = (((uint32_t)value_a_limbs[27] == 256u) &&
                                  ((uint32_t)value_a_limbs[21] == 136u)) ? 1u : 0u;
    uint32_t rc_input_a = ms_is_max_a * (120u + (uint32_t)value_a_limbs[21] - ms_and_mid_max_a);

    traces[59][row] = (m31)ms_is_max_a;
    traces[60][row] = (m31)ms_and_mid_max_a;
    traces[61][row] = (m31)rc_input_a;

    // ==================================================================
    // 5. Verify Reduced 252 for value B -> trace cols 62-64
    // ==================================================================
    uint32_t ms_is_max_b = ((uint32_t)value_b_limbs[27] == 256u) ? 1u : 0u;
    uint32_t ms_and_mid_max_b = (((uint32_t)value_b_limbs[27] == 256u) &&
                                  ((uint32_t)value_b_limbs[21] == 136u)) ? 1u : 0u;
    uint32_t rc_input_b = ms_is_max_b * (120u + (uint32_t)value_b_limbs[21] - ms_and_mid_max_b);

    traces[62][row] = (m31)ms_is_max_b;
    traces[63][row] = (m31)ms_and_mid_max_b;
    traces[64][row] = (m31)rc_input_b;

    // ==================================================================
    // 6. Compute m_shifted values for each chain
    //    m_shifted_x[i] = value_x_limbs[2*i] + (value_x_limbs[2*i+1] << 9)
    // ==================================================================
    uint32_t m_shifted_a[14];
    for (int i = 0; i < 14; i++) {
        m_shifted_a[i] = (uint32_t)value_a_limbs[2 * i] +
                          ((uint32_t)value_a_limbs[2 * i + 1] << 9);
    }

    uint32_t m_shifted_b[14];
    for (int i = 0; i < 14; i++) {
        m_shifted_b[i] = (uint32_t)value_b_limbs[2 * i] +
                          ((uint32_t)value_b_limbs[2 * i + 1] << 9);
    }

    // ==================================================================
    // 7. Chain 0: 14 rounds of EC point addition using value A
    //    Initialize accumulator from shift point
    // ==================================================================
    m31 acc_x[28], acc_y[28];
    for (int i = 0; i < 28; i++) {
        acc_x[i] = (m31)AGG_SHIFT_POINT_X_LIMBS[i];
        acc_y[i] = (m31)AGG_SHIFT_POINT_Y_LIMBS[i];
    }

    uint32_t chain_id_0 = row * 2u;

    m31 chain0_result_x[28];
    m31 chain0_result_y[28];

    for (int r = 0; r < 14; r++) {
        // Table lookup
        uint32_t table_idx = 262144u * (uint32_t)r + m_shifted_a[r];
        m31 table_x[28], table_y[28];
        for (int i = 0; i < 28; i++) {
            table_x[i] = g_pedersen_table_columns[i][table_idx];
        }
        for (int i = 0; i < 28; i++) {
            table_y[i] = g_pedersen_table_columns[28 + i][table_idx];
        }

        // Write PEM sub-component inputs for this round
        // sub_pem[col][round * trace_size + row] — 72 cols, 28 rounds interleaved
        uint32_t pem_off = r * d_agg_args.trace_size + row;  // chain 0, round r
        d_agg_args.sub_pem[0][pem_off] = (m31)chain_id_0;
        d_agg_args.sub_pem[1][pem_off] = (m31)((uint32_t)r);
        // Shifted m_shifted values: m_shifted[r..13], then zeros
        for (int i = 0; i < 14; i++) {
            if (r + i < 14) {
                d_agg_args.sub_pem[2 + i][pem_off] = (m31)m_shifted_a[r + i];
            } else {
                d_agg_args.sub_pem[2 + i][pem_off] = 0u;
            }
        }
        // acc_x BEFORE addition
        for (int i = 0; i < 28; i++) {
            d_agg_args.sub_pem[16 + i][pem_off] = acc_x[i];
        }
        // acc_y BEFORE addition
        for (int i = 0; i < 28; i++) {
            d_agg_args.sub_pem[44 + i][pem_off] = acc_y[i];
        }

        // EC point addition: acc = acc + table_point
        m31 new_x[28], new_y[28];
        agg_ec_point_add(acc_x, acc_y, table_x, table_y, new_x, new_y);

        for (int i = 0; i < 28; i++) {
            acc_x[i] = new_x[i];
            acc_y[i] = new_y[i];
        }
    }

    // Store chain 0 result
    for (int i = 0; i < 28; i++) {
        chain0_result_x[i] = acc_x[i];
        chain0_result_y[i] = acc_y[i];
    }

    // Write chain 0 trace cols 65-134: 14 zeros + 28 result_x + 28 result_y
    for (int i = 0; i < 14; i++) {
        traces[65 + i][row] = 0u;
    }
    for (int i = 0; i < 28; i++) {
        traces[79 + i][row] = chain0_result_x[i];
    }
    for (int i = 0; i < 28; i++) {
        traces[107 + i][row] = chain0_result_y[i];
    }

    // ==================================================================
    // 8. Chain 1: 14 rounds of EC point addition using value B
    //    Initialize accumulator from chain 0 result
    // ==================================================================
    for (int i = 0; i < 28; i++) {
        acc_x[i] = chain0_result_x[i];
        acc_y[i] = chain0_result_y[i];
    }

    uint32_t chain_id_1 = row * 2u + 1u;

    m31 chain1_result_x[28];
    m31 chain1_result_y[28];

    for (int r = 0; r < 14; r++) {
        // Table lookup
        uint32_t table_idx = 262144u * (uint32_t)(14 + r) + m_shifted_b[r];
        m31 table_x[28], table_y[28];
        for (int i = 0; i < 28; i++) {
            table_x[i] = g_pedersen_table_columns[i][table_idx];
        }
        for (int i = 0; i < 28; i++) {
            table_y[i] = g_pedersen_table_columns[28 + i][table_idx];
        }

        // Write PEM sub-component inputs for this round
        // sub_pem[col][(14 + r) * trace_size + row] — chain 1, round r
        uint32_t pem_off = (14 + r) * d_agg_args.trace_size + row;
        d_agg_args.sub_pem[0][pem_off] = (m31)chain_id_1;
        d_agg_args.sub_pem[1][pem_off] = (m31)((uint32_t)(14 + r));
        // Shifted m_shifted values: m_shifted_b[r..13], then zeros
        for (int i = 0; i < 14; i++) {
            if (r + i < 14) {
                d_agg_args.sub_pem[2 + i][pem_off] = (m31)m_shifted_b[r + i];
            } else {
                d_agg_args.sub_pem[2 + i][pem_off] = 0u;
            }
        }
        // acc_x BEFORE addition
        for (int i = 0; i < 28; i++) {
            d_agg_args.sub_pem[16 + i][pem_off] = acc_x[i];
        }
        // acc_y BEFORE addition
        for (int i = 0; i < 28; i++) {
            d_agg_args.sub_pem[44 + i][pem_off] = acc_y[i];
        }

        // EC point addition: acc = acc + table_point
        m31 new_x[28], new_y[28];
        agg_ec_point_add(acc_x, acc_y, table_x, table_y, new_x, new_y);

        for (int i = 0; i < 28; i++) {
            acc_x[i] = new_x[i];
            acc_y[i] = new_y[i];
        }
    }

    // Store chain 1 result
    for (int i = 0; i < 28; i++) {
        chain1_result_x[i] = acc_x[i];
        chain1_result_y[i] = acc_y[i];
    }

    // Write chain 1 trace cols 135-204: 14 zeros + 28 result_x + 28 result_y
    for (int i = 0; i < 14; i++) {
        traces[135 + i][row] = 0u;
    }
    for (int i = 0; i < 28; i++) {
        traces[149 + i][row] = chain1_result_x[i];
    }
    for (int i = 0; i < 28; i++) {
        traces[177 + i][row] = chain1_result_y[i];
    }

    // ==================================================================
    // 9. Multiplicity (col 205)
    // ==================================================================
    traces[205][row] = d_agg_args.mults[row];

    // ==================================================================
    // 10. Lookup data arrays
    // ==================================================================

    // --- lk_mem_0: [relation_id, input_limb_0, value_a_limbs[0..27]] ---
    d_agg_args.lk_mem_0[0][row] = (m31)AGG_MEM_ID_TO_BIG_RELATION_ID;
    d_agg_args.lk_mem_0[1][row] = input_limb_0;
    for (int i = 0; i < 28; i++) {
        d_agg_args.lk_mem_0[2 + i][row] = value_a_limbs[i];
    }

    // --- lk_mem_1: [relation_id, input_limb_1, value_b_limbs[0..27]] ---
    d_agg_args.lk_mem_1[0][row] = (m31)AGG_MEM_ID_TO_BIG_RELATION_ID;
    d_agg_args.lk_mem_1[1][row] = input_limb_1;
    for (int i = 0; i < 28; i++) {
        d_agg_args.lk_mem_1[2 + i][row] = value_b_limbs[i];
    }

    // --- lk_mem_2: [relation_id, input_limb_2, chain1_result_x[0..27]] ---
    d_agg_args.lk_mem_2[0][row] = (m31)AGG_MEM_ID_TO_BIG_RELATION_ID;
    d_agg_args.lk_mem_2[1][row] = input_limb_2;
    for (int i = 0; i < 28; i++) {
        d_agg_args.lk_mem_2[2 + i][row] = chain1_result_x[i];
    }

    // --- lk_rc8_0: [relation_id, value_a_limbs[27] - ms_is_max_a] ---
    d_agg_args.lk_rc8_0[0][row] = (m31)AGG_RC_8_RELATION_ID;
    d_agg_args.lk_rc8_0[1][row] = (m31)((uint32_t)value_a_limbs[27] - ms_is_max_a);

    // --- lk_rc8_1: [relation_id, rc_input_a] ---
    d_agg_args.lk_rc8_1[0][row] = (m31)AGG_RC_8_RELATION_ID;
    d_agg_args.lk_rc8_1[1][row] = (m31)rc_input_a;

    // --- lk_rc8_2: [relation_id, value_b_limbs[27] - ms_is_max_b] ---
    d_agg_args.lk_rc8_2[0][row] = (m31)AGG_RC_8_RELATION_ID;
    d_agg_args.lk_rc8_2[1][row] = (m31)((uint32_t)value_b_limbs[27] - ms_is_max_b);

    // --- lk_rc8_3: [relation_id, rc_input_b] ---
    d_agg_args.lk_rc8_3[0][row] = (m31)AGG_RC_8_RELATION_ID;
    d_agg_args.lk_rc8_3[1][row] = (m31)rc_input_b;

    // --- lk_pem_0: chain 0 input ---
    // [relation_id, chain_id_0, 0, m_shifted_a[0..13],
    //  shift_x[0..27], shift_y[0..27]]
    d_agg_args.lk_pem_0[0][row] = (m31)AGG_PEM_RELATION_ID;
    d_agg_args.lk_pem_0[1][row] = (m31)chain_id_0;
    d_agg_args.lk_pem_0[2][row] = 0u;
    for (int i = 0; i < 14; i++) {
        d_agg_args.lk_pem_0[3 + i][row] = (m31)m_shifted_a[i];
    }
    for (int i = 0; i < 28; i++) {
        d_agg_args.lk_pem_0[17 + i][row] = (m31)AGG_SHIFT_POINT_X_LIMBS[i];
    }
    for (int i = 0; i < 28; i++) {
        d_agg_args.lk_pem_0[45 + i][row] = (m31)AGG_SHIFT_POINT_Y_LIMBS[i];
    }

    // --- lk_pem_1: chain 0 output ---
    // [relation_id, chain_id_0, 14, trace_cols[65..134]]
    d_agg_args.lk_pem_1[0][row] = (m31)AGG_PEM_RELATION_ID;
    d_agg_args.lk_pem_1[1][row] = (m31)chain_id_0;
    d_agg_args.lk_pem_1[2][row] = 14u;
    for (int i = 0; i < 70; i++) {
        d_agg_args.lk_pem_1[3 + i][row] = traces[65 + i][row];
    }

    // --- lk_pem_2: chain 1 input ---
    // [relation_id, chain_id_1, 14, m_shifted_b[0..13],
    //  chain0_result_x[0..27], chain0_result_y[0..27]]
    d_agg_args.lk_pem_2[0][row] = (m31)AGG_PEM_RELATION_ID;
    d_agg_args.lk_pem_2[1][row] = (m31)chain_id_1;
    d_agg_args.lk_pem_2[2][row] = 14u;
    for (int i = 0; i < 14; i++) {
        d_agg_args.lk_pem_2[3 + i][row] = (m31)m_shifted_b[i];
    }
    for (int i = 0; i < 28; i++) {
        d_agg_args.lk_pem_2[17 + i][row] = chain0_result_x[i];
    }
    for (int i = 0; i < 28; i++) {
        d_agg_args.lk_pem_2[45 + i][row] = chain0_result_y[i];
    }

    // --- lk_pem_3: chain 1 output ---
    // [relation_id, chain_id_1, 28, trace_cols[135..204]]
    d_agg_args.lk_pem_3[0][row] = (m31)AGG_PEM_RELATION_ID;
    d_agg_args.lk_pem_3[1][row] = (m31)chain_id_1;
    d_agg_args.lk_pem_3[2][row] = 28u;
    for (int i = 0; i < 70; i++) {
        d_agg_args.lk_pem_3[3 + i][row] = traces[135 + i][row];
    }

    // --- lk_agg_0: self-lookup [relation_id, input_0, input_1, input_2] ---
    d_agg_args.lk_agg_0[0][row] = (m31)AGG_SELF_RELATION_ID;
    d_agg_args.lk_agg_0[1][row] = input_limb_0;
    d_agg_args.lk_agg_0[2][row] = input_limb_1;
    d_agg_args.lk_agg_0[3][row] = input_limb_2;

    // ==================================================================
    // 11. Sub-component inputs
    // ==================================================================

    // memory_id_to_big feeds
    d_agg_args.sub_mem[0][row] = input_limb_0;
    d_agg_args.sub_mem[1][row] = input_limb_1;
    d_agg_args.sub_mem[2][row] = input_limb_2;

    // range_check_8 feeds
    d_agg_args.sub_rc8[0][row] = (m31)((uint32_t)value_a_limbs[27] - ms_is_max_a);
    d_agg_args.sub_rc8[1][row] = (m31)rc_input_a;
    d_agg_args.sub_rc8[2][row] = (m31)((uint32_t)value_b_limbs[27] - ms_is_max_b);
    d_agg_args.sub_rc8[3][row] = (m31)rc_input_b;

    // PEM sub-inputs are already written in steps 7 and 8
}

// ============================================================================
// Exported C wrapper function
// ============================================================================

extern "C" void gen_pedersen_aggregator_wb18_trace(
    m31** traces,
    m31** lk_mem_0,
    m31** lk_mem_1,
    m31** lk_mem_2,
    m31** lk_rc8_0,
    m31** lk_rc8_1,
    m31** lk_rc8_2,
    m31** lk_rc8_3,
    m31** lk_pem_0,
    m31** lk_pem_1,
    m31** lk_pem_2,
    m31** lk_pem_3,
    m31** lk_agg_0,
    m31* mults,
    m31** sub_mem,
    m31** sub_rc8,
    m31** sub_pem,
    m31** inputs,
    unsigned** transpose_big_value_ptr,
    unsigned* small_value_ptr,
    uint32_t n_rows,
    uint32_t log_size
) {
    uint32_t trace_size = 1u << log_size;

    // Increase stack size for the kernel.
    // Each thread performs 28 EC point additions (14 per chain), each requiring
    // a Felt252 field inversion. The inverse function uses deep recursion with
    // large local state, so we need a generous stack.
    // Stack size: set once, skip on subsequent calls (avoids device sync).
    static bool stack_set = false;
    if (!stack_set) {
        cudaDeviceSetLimit(cudaLimitStackSize, 32768);
        stack_set = true;
    }

    // Clone all host pointer arrays to device
    m31** d_traces = clone_to_device<m31*>(traces, 206);
    m31** d_inputs = clone_to_device<m31*>(inputs, 3);
    unsigned** d_transpose = clone_to_device<unsigned*>(transpose_big_value_ptr, 8);
    m31** d_lk_mem_0 = clone_to_device<m31*>(lk_mem_0, 30);
    m31** d_lk_mem_1 = clone_to_device<m31*>(lk_mem_1, 30);
    m31** d_lk_mem_2 = clone_to_device<m31*>(lk_mem_2, 30);
    m31** d_lk_rc8_0 = clone_to_device<m31*>(lk_rc8_0, 2);
    m31** d_lk_rc8_1 = clone_to_device<m31*>(lk_rc8_1, 2);
    m31** d_lk_rc8_2 = clone_to_device<m31*>(lk_rc8_2, 2);
    m31** d_lk_rc8_3 = clone_to_device<m31*>(lk_rc8_3, 2);
    m31** d_lk_pem_0 = clone_to_device<m31*>(lk_pem_0, 73);
    m31** d_lk_pem_1 = clone_to_device<m31*>(lk_pem_1, 73);
    m31** d_lk_pem_2 = clone_to_device<m31*>(lk_pem_2, 73);
    m31** d_lk_pem_3 = clone_to_device<m31*>(lk_pem_3, 73);
    m31** d_lk_agg_0 = clone_to_device<m31*>(lk_agg_0, 4);
    m31** d_sub_mem = clone_to_device<m31*>(sub_mem, 3);
    m31** d_sub_rc8 = clone_to_device<m31*>(sub_rc8, 4);
    m31** d_sub_pem = clone_to_device<m31*>(sub_pem, 72);

    // Fill kernel args struct and copy to constant memory
    agg_kernel_args args;
    args.traces = d_traces;
    args.inputs = d_inputs;
    args.transpose_big_value_ptr = d_transpose;
    args.small_value_ptr = small_value_ptr;
    args.n_rows = n_rows;
    args.trace_size = trace_size;
    args.lk_mem_0 = d_lk_mem_0;
    args.lk_mem_1 = d_lk_mem_1;
    args.lk_mem_2 = d_lk_mem_2;
    args.lk_rc8_0 = d_lk_rc8_0;
    args.lk_rc8_1 = d_lk_rc8_1;
    args.lk_rc8_2 = d_lk_rc8_2;
    args.lk_rc8_3 = d_lk_rc8_3;
    args.lk_pem_0 = d_lk_pem_0;
    args.lk_pem_1 = d_lk_pem_1;
    args.lk_pem_2 = d_lk_pem_2;
    args.lk_pem_3 = d_lk_pem_3;
    args.lk_agg_0 = d_lk_agg_0;
    args.mults = mults;
    args.sub_mem = d_sub_mem;
    args.sub_rc8 = d_sub_rc8;
    args.sub_pem = d_sub_pem;

    cudaMemcpyToSymbol(d_agg_args, &args, sizeof(agg_kernel_args));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Launch kernel
    uint32_t num_blocks = (trace_size + AGG_BLOCK_SIZE - 1) / AGG_BLOCK_SIZE;
    agg_trace_kernel<<<num_blocks, AGG_BLOCK_SIZE>>>();
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Cleanup device pointer arrays
    cuda_free_memory(d_traces);
    cuda_free_memory(d_inputs);
    cuda_free_memory(d_transpose);
    cuda_free_memory(d_lk_mem_0);
    cuda_free_memory(d_lk_mem_1);
    cuda_free_memory(d_lk_mem_2);
    cuda_free_memory(d_lk_rc8_0);
    cuda_free_memory(d_lk_rc8_1);
    cuda_free_memory(d_lk_rc8_2);
    cuda_free_memory(d_lk_rc8_3);
    cuda_free_memory(d_lk_pem_0);
    cuda_free_memory(d_lk_pem_1);
    cuda_free_memory(d_lk_pem_2);
    cuda_free_memory(d_lk_pem_3);
    cuda_free_memory(d_lk_agg_0);
    cuda_free_memory(d_sub_mem);
    cuda_free_memory(d_sub_rc8);
    cuda_free_memory(d_sub_pem);

    // Restore stack size
    // Stack restore removed — keep 32KB permanently.
}

// ============================================================================
// Interaction trace generation for pedersen_aggregator_wb18 (6 logup columns)
// ============================================================================
//
// Column layout (matching the SIMD LogupTraceGenerator in pedersen_aggregator_cuda.rs):
//   Col 0: mem_id_to_big pair (mem_0 + mem_1)  — ADD: 30-elem combine each
//   Col 1: range_check_8 pair (rc8_0 + rc8_1)  — ADD: 2-elem combine each
//   Col 2: range_check_8 pair (rc8_2 + rc8_3)  — ADD: 2-elem combine each
//   Col 3: partial_ec_mul pair (pem_0 - pem_1)  — SUB: 73-elem combine each
//   Col 4: partial_ec_mul pair (pem_2 - pem_3)  — SUB: 73-elem combine each
//   Col 5: mem_id_to_big_2 + self-lookup with mults — SPECIAL
//
// All lookup data arrays reside on GPU (produced by gen_pedersen_aggregator_wb18_trace).
// The kernel uses CommonLookupElements (= LookupElementsBasic<128>) directly,
// since the lookup data already includes the relation constant at index 0.
// ============================================================================

#define AGG_IT_BLOCK_SIZE 256

// ADD pair kernel: frac = (d0 + d1) / (d0 * d1)
template <int N, int M>
__launch_bounds__(AGG_IT_BLOCK_SIZE, 2)
__global__ void agg_it_add_pair_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31** data_0,
    m31** data_1,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        m31 vals0[N], vals1[M];
        for (int i = 0; i < N; i++) vals0[i] = data_0[i][idx];
        for (int i = 0; i < M; i++) vals1[i] = data_1[i][idx];
        qm31 d0 = lookup_elements->combine(vals0, N);
        qm31 d1 = lookup_elements->combine(vals1, M);
        logup_col_write_frac(idx, add(d0, d1), mul(d0, d1),
                            denom_ptr, numer0, numer1, numer2, numer3);
    }
}

// SUB pair kernel: frac = (d0 - d1) / (d0 * d1)
template <int N, int M>
__launch_bounds__(AGG_IT_BLOCK_SIZE, 2)
__global__ void agg_it_sub_pair_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31** data_0,
    m31** data_1,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        m31 vals0[N], vals1[M];
        for (int i = 0; i < N; i++) vals0[i] = data_0[i][idx];
        for (int i = 0; i < M; i++) vals1[i] = data_1[i][idx];
        qm31 d0 = lookup_elements->combine(vals0, N);
        qm31 d1 = lookup_elements->combine(vals1, M);
        logup_col_write_frac(idx, sub(d0, d1), mul(d0, d1),
                            denom_ptr, numer0, numer1, numer2, numer3);
    }
}

// Special mult-weighted kernel: frac = (d1 - d0 * mult) / (d0 * d1)
template <int N, int M>
__launch_bounds__(AGG_IT_BLOCK_SIZE, 2)
__global__ void agg_it_special_mults_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31** data_0,
    m31** data_1,
    m31* mults,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        m31 vals0[N], vals1[M];
        for (int i = 0; i < N; i++) vals0[i] = data_0[i][idx];
        for (int i = 0; i < M; i++) vals1[i] = data_1[i][idx];
        qm31 d0 = lookup_elements->combine(vals0, N);
        qm31 d1 = lookup_elements->combine(vals1, M);
        qm31 m_val = qm31{cm31{mults[idx], 0}, cm31{0, 0}};
        logup_col_write_frac(idx, sub(d1, mul(d0, m_val)), mul(d0, d1),
                            denom_ptr, numer0, numer1, numer2, numer3);
    }
}

// Finalize kernel: multiply numerator by inverse denominator and accumulate
__global__ void agg_it_finalize_col_kernel(
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

// Cumsum shift kernel — computes claimed_sum from last column
__global__ void agg_it_cumsum_shift(
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

// Coordinate prefix sum kernel — subtracts shift from last column
__global__ void agg_it_coord_prefix_sum(
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

// Helper macro for processing a column
#define AGG_IT_PROCESS_ADD(col_idx, N1, N2, d0_ptrs, d1_ptrs) \
    agg_it_add_pair_kernel<N1, N2><<<num_blocks, block_dim>>>( \
        d_lookup, d0_ptrs, d1_ptrs, trace_size, \
        device_logup_denom, numer0, numer1, numer2, numer3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    agg_it_finalize_col_kernel<<<num_blocks, block_dim>>>(col_idx, trace_size, denom_inv, \
        numer0, numer1, numer2, numer3, device_it); \

#define AGG_IT_PROCESS_SUB(col_idx, N1, N2, d0_ptrs, d1_ptrs) \
    agg_it_sub_pair_kernel<N1, N2><<<num_blocks, block_dim>>>( \
        d_lookup, d0_ptrs, d1_ptrs, trace_size, \
        device_logup_denom, numer0, numer1, numer2, numer3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    agg_it_finalize_col_kernel<<<num_blocks, block_dim>>>(col_idx, trace_size, denom_inv, \
        numer0, numer1, numer2, numer3, device_it); \

extern "C" void gen_pedersen_aggregator_wb18_interaction_trace(
    // CommonLookupElements (= LookupElements<128>)
    void* lookup_elements,
    // Lookup data (all device pointers, from base trace generation)
    m31** lk_mem_0,         // 30 arrays
    m31** lk_mem_1,         // 30 arrays
    m31** lk_mem_2,         // 30 arrays
    m31** lk_rc8_0,         // 2 arrays
    m31** lk_rc8_1,         // 2 arrays
    m31** lk_rc8_2,         // 2 arrays
    m31** lk_rc8_3,         // 2 arrays
    m31** lk_pem_0,         // 73 arrays
    m31** lk_pem_1,         // 73 arrays
    m31** lk_pem_2,         // 73 arrays
    m31** lk_pem_3,         // 73 arrays
    m31** lk_agg_0,         // 4 arrays
    m31* mults,             // multiplicities (device pointer)
    // Sizes
    uint32_t log_size,
    // Output
    m31** interaction_trace_columns,   // 4 * AGG_N_LOGUP_COLUMNS = 24 columns
    m31* claimed_sum                   // 4 m31s for qm31
) {
    uint32_t trace_size = 1u << log_size;

    // Copy lookup elements to device
    LookupElementsBasic<128>* d_lookup = cuda_malloc<LookupElementsBasic<128>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<128>>(
        (LookupElementsBasic<128>*)lookup_elements, d_lookup, 1);

    // Clone lookup data pointer arrays to device
    m31** d_mem_0 = clone_to_device<m31*>(lk_mem_0, 30);
    m31** d_mem_1 = clone_to_device<m31*>(lk_mem_1, 30);
    m31** d_mem_2 = clone_to_device<m31*>(lk_mem_2, 30);
    m31** d_rc8_0 = clone_to_device<m31*>(lk_rc8_0, 2);
    m31** d_rc8_1 = clone_to_device<m31*>(lk_rc8_1, 2);
    m31** d_rc8_2 = clone_to_device<m31*>(lk_rc8_2, 2);
    m31** d_rc8_3 = clone_to_device<m31*>(lk_rc8_3, 2);
    m31** d_pem_0 = clone_to_device<m31*>(lk_pem_0, 73);
    m31** d_pem_1 = clone_to_device<m31*>(lk_pem_1, 73);
    m31** d_pem_2 = clone_to_device<m31*>(lk_pem_2, 73);
    m31** d_pem_3 = clone_to_device<m31*>(lk_pem_3, 73);
    m31** d_agg_0 = clone_to_device<m31*>(lk_agg_0, 4);

    // Allocate working memory
    qm31* device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31* denom_inv = cuda_malloc<qm31>(trace_size);
    m31* numer0 = cuda_malloc<m31>(trace_size);
    m31* numer1 = cuda_malloc<m31>(trace_size);
    m31* numer2 = cuda_malloc<m31>(trace_size);
    m31* numer3 = cuda_malloc<m31>(trace_size);

    m31** device_it = clone_to_device<m31*>(interaction_trace_columns, 4 * AGG_N_LOGUP_COLUMNS);

    int block_dim = trace_size < AGG_IT_BLOCK_SIZE ? trace_size : AGG_IT_BLOCK_SIZE;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Col 0: mem_id_to_big pair (mem_0 + mem_1) — ADD, 30 elements each
    AGG_IT_PROCESS_ADD(0, 30, 30, d_mem_0, d_mem_1);

    // Col 1: range_check_8 pair (rc8_0 + rc8_1) — ADD, 2 elements each
    AGG_IT_PROCESS_ADD(1, 2, 2, d_rc8_0, d_rc8_1);

    // Col 2: range_check_8 pair (rc8_2 + rc8_3) — ADD, 2 elements each
    AGG_IT_PROCESS_ADD(2, 2, 2, d_rc8_2, d_rc8_3);

    // Col 3: partial_ec_mul pair (pem_0 - pem_1) — SUB, 73 elements each
    AGG_IT_PROCESS_SUB(3, 73, 73, d_pem_0, d_pem_1);

    // Col 4: partial_ec_mul pair (pem_2 - pem_3) — SUB, 73 elements each
    AGG_IT_PROCESS_SUB(4, 73, 73, d_pem_2, d_pem_3);

    // Col 5: special — frac = (d1 - d0 * mults) / (d0 * d1)
    // where d0 = mem_2 (30 elems), d1 = agg_0 (4 elems), m = multiplicities
    agg_it_special_mults_kernel<30, 4><<<num_blocks, block_dim>>>(
        d_lookup, d_mem_2, d_agg_0, mults, trace_size,
        device_logup_denom, numer0, numer1, numer2, numer3);
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);
    agg_it_finalize_col_kernel<<<num_blocks, block_dim>>>(5, trace_size, denom_inv,
        numer0, numer1, numer2, numer3, device_it);

    // Finalize: cumsum_shift + prefix sum on last 4 columns
    cudaMemsetAsync(claimed_sum, 0, 4 * sizeof(m31), 0);

    size_t shared_size = 4 * block_dim * sizeof(m31);
    agg_it_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        AGG_N_LOGUP_COLUMNS, trace_size, device_it, claimed_sum);

    agg_it_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum, AGG_N_LOGUP_COLUMNS, trace_size, device_it);

    // Inclusive prefix sum on last 4 columns
    inclusive_prefix_sum(interaction_trace_columns[4 * AGG_N_LOGUP_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * AGG_N_LOGUP_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * AGG_N_LOGUP_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * AGG_N_LOGUP_COLUMNS - 1], trace_size);

    // Cleanup
    cuda_free_memory(d_lookup);
    cuda_free_memory(d_mem_0);
    cuda_free_memory(d_mem_1);
    cuda_free_memory(d_mem_2);
    cuda_free_memory(d_rc8_0);
    cuda_free_memory(d_rc8_1);
    cuda_free_memory(d_rc8_2);
    cuda_free_memory(d_rc8_3);
    cuda_free_memory(d_pem_0);
    cuda_free_memory(d_pem_1);
    cuda_free_memory(d_pem_2);
    cuda_free_memory(d_pem_3);
    cuda_free_memory(d_agg_0);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(numer0);
    cuda_free_memory(numer1);
    cuda_free_memory(numer2);
    cuda_free_memory(numer3);
    cuda_free_memory(device_it);
}
