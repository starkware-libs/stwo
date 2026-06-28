// CUDA Pedersen Small Table GPU-Native Initialization (window_bits_9)
// Adapted from pedersen_table_init.cu for the CanonicalSmall table variant.
//
// This file generates the PEDERSEN_SMALL_TABLE directly on GPU and stores pointers
// in the global device memory symbols (g_pedersen_table_small_columns).
//
// Key differences from the w18 table (pedersen_table_init.cu):
// - 9-bit windows instead of 18-bit windows
// - 27 low windows instead of 13
// - 512 rows per low window instead of 262144
// - 5-bit high rows instead of 14-bit
// - 32 rows per high sub-block instead of 16384
// - Total: 28672 rows (padded to 32768) vs 7340032 rows
// - GPU memory: ~7 MB vs ~1.6 GB

#include "fields.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "batch_inverse.cuh"
#include "fp256_config.cuh"
#include "fp256_dispatch_st.cuh"
#include "cuda_mem_pool.cuh"
#include <cstdint>
#include <cstdio>

// Include ec_ops.cuh for felt252 operations
#include "ec_ops.cuh"

// ============================================================================
// Table Parameters (must match pedersen_table_small.cuh)
// ============================================================================
#define INIT_PED_SMALL_BITS_PER_WINDOW 9
#define INIT_PED_SMALL_NUM_LOW_WINDOWS 27   // 252 / 9 - 1 = 27
#define INIT_PED_SMALL_ROWS_PER_WINDOW (1 << INIT_PED_SMALL_BITS_PER_WINDOW)  // 512

// High section: last 9 bits split as 5 bits (row index) + 4 bits (sub-block index)
#define INIT_PED_SMALL_HIGH_BITS (INIT_PED_SMALL_BITS_PER_WINDOW - 4)  // 5
#define INIT_PED_SMALL_HIGH_ROWS_PER_SUBBLOCK (1 << INIT_PED_SMALL_HIGH_BITS)  // 32
#define INIT_PED_SMALL_HIGH_NUM_SUBBLOCKS 16  // 2^4

// Section layout:
// P0 low:      27 windows x 512 = 13,824 rows
// P0/P1 high:  16 sub-blocks x 32 = 512 rows
// P2 low:      27 windows x 512 = 13,824 rows
// P2/P3 high:  16 sub-blocks x 32 = 512 rows
// Total:       28,672 rows
#define INIT_PED_SMALL_P0_LOW_START 0
#define INIT_PED_SMALL_P0P1_HIGH_START (INIT_PED_SMALL_NUM_LOW_WINDOWS * INIT_PED_SMALL_ROWS_PER_WINDOW)  // 13824
#define INIT_PED_SMALL_P2_LOW_START (INIT_PED_SMALL_P0P1_HIGH_START + INIT_PED_SMALL_HIGH_NUM_SUBBLOCKS * INIT_PED_SMALL_HIGH_ROWS_PER_SUBBLOCK)  // 14336
#define INIT_PED_SMALL_P2P3_HIGH_START (INIT_PED_SMALL_P2_LOW_START + INIT_PED_SMALL_NUM_LOW_WINDOWS * INIT_PED_SMALL_ROWS_PER_WINDOW)  // 28160
#define INIT_PED_SMALL_TABLE_N_ROWS_UNPADDED (INIT_PED_SMALL_P2P3_HIGH_START + INIT_PED_SMALL_HIGH_NUM_SUBBLOCKS * INIT_PED_SMALL_HIGH_ROWS_PER_SUBBLOCK)  // 28672
#define INIT_PED_SMALL_TABLE_N_COLUMNS 56

// ============================================================================
// Global Device Memory Storage for Pedersen Small Table
// These are the actual definitions (declared extern in pedersen_table_small.cuh)
// ============================================================================
__device__ m31* g_pedersen_table_small_columns[56];
__device__ uint32_t g_pedersen_table_small_n_rows = 0;

// Host-side tracking for GPU-generated table
static bool s_pedersen_table_small_initialized = false;
static m31* s_pedersen_table_small_ptrs[INIT_PED_SMALL_TABLE_N_COLUMNS] = {nullptr};
static uint32_t s_pedersen_table_small_n_rows = 0;

// ============================================================================
// Starknet Pedersen Curve Constants (stored in __constant__ for fast access)
// Same base points as w18 table, with SMALL_ prefix to avoid symbol conflicts
// ============================================================================

// PEDERSEN_P0 (generator for low 252 bits of value A)
// P0.x = 0x0234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b
// P0.y = 0x03b056f100f96fb21e889527d41f4e39940135dd7a6c94cc6ed0268ee89e5615
__constant__ uint32_t SMALL_CONST_PEDERSEN_P0_X[8] = {
    0x57ebe47b, 0x1080d179, 0x6d56eb0c, 0x8fa8120b,
    0x55fca9e5, 0x969c7486, 0xcbaffe7f, 0x0234287d
};
__constant__ uint32_t SMALL_CONST_PEDERSEN_P0_Y[8] = {
    0xe89e5615, 0x6ed0268e, 0x7a6c94cc, 0x940135dd,
    0xd41f4e39, 0x1e889527, 0x00f96fb2, 0x03b056f1
};

// PEDERSEN_P1 (generator for high 4 bits of value A)
// P1.x = 0x04fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7a6932dba8aa378
// P1.y = 0x03fa0984c931c9e38113e0c0e47e4401562761f92a7a23b45168f4e80ff5b54d
__constant__ uint32_t SMALL_CONST_PEDERSEN_P1_X[8] = {
    0xba8aa378, 0xb7a6932d, 0xde5e3018, 0x99099ec1,
    0x56558f33, 0x3f9dab26, 0x76c83db3, 0x04fa56f3
};
__constant__ uint32_t SMALL_CONST_PEDERSEN_P1_Y[8] = {
    0x0ff5b54d, 0x5168f4e8, 0x2a7a23b4, 0x562761f9,
    0xe47e4401, 0x8113e0c0, 0xc931c9e3, 0x03fa0984
};

// PEDERSEN_P2 (generator for low 252 bits of value B)
// P2.x = 0x04ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997
// P2.y = 0x0040301cf5c1751f4b971e46c4ede85fcac5c59a5ce5ae7c48151f27b24b219c
__constant__ uint32_t SMALL_CONST_PEDERSEN_P2_X[8] = {
    0xbd2d6997, 0x3aa372f0, 0x4709e90f, 0x40c690c7,
    0x5b45f74b, 0x764910f7, 0x66be8dec, 0x04ba4cc1
};
__constant__ uint32_t SMALL_CONST_PEDERSEN_P2_Y[8] = {
    0xb24b219c, 0x48151f27, 0x5ce5ae7c, 0xcac5c59a,
    0xc4ede85f, 0x4b971e46, 0xf5c1751f, 0x0040301c
};

// PEDERSEN_P3 (generator for high 4 bits of value B)
// P3.x = 0x054302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202
// P3.y = 0x01b77b3e37d13504b348046268d8ae25ce98ad783c25561a879dcc77e99c2426
__constant__ uint32_t SMALL_CONST_PEDERSEN_P3_X[8] = {
    0x49a58202, 0xd36ff12c, 0xd53fb325, 0x2ca65048,
    0xf61a63bb, 0x6e44cca8, 0xb0e6cc1c, 0x054302dc
};
__constant__ uint32_t SMALL_CONST_PEDERSEN_P3_Y[8] = {
    0xe99c2426, 0x879dcc77, 0x3c25561a, 0xce98ad78,
    0x68d8ae25, 0xb3480462, 0x37d13504, 0x01b77b3e
};

// SHIFT_POINT (negated and added to all table entries)
// shift_point.x = 0x049ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804
// shift_point.y = 0x03ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a
__constant__ uint32_t SMALL_CONST_SHIFT_POINT_X[8] = {
    0x50ca6804, 0x551fde40, 0x22947733, 0x716b0b10,
    0xeb599f16, 0x00ee1b87, 0xa8c16007, 0x049ee3eb
};
__constant__ uint32_t SMALL_CONST_SHIFT_POINT_Y[8] = {
    0x6e10268a, 0xd0405d26, 0xc0e056c1, 0x4e621062,
    0x06ea0ed3, 0xf346d49d, 0x4b3bc6dd, 0x03ca0cfe
};

// ============================================================================
// Device Helper Functions
// ============================================================================

// Load base points directly from constant memory (avoids pointer passing issues)
__device__ void small_load_P0(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = SMALL_CONST_PEDERSEN_P0_X[i];
        y.limbs[i] = SMALL_CONST_PEDERSEN_P0_Y[i];
    }
}

__device__ void small_load_P1(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = SMALL_CONST_PEDERSEN_P1_X[i];
        y.limbs[i] = SMALL_CONST_PEDERSEN_P1_Y[i];
    }
}

__device__ void small_load_P2(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = SMALL_CONST_PEDERSEN_P2_X[i];
        y.limbs[i] = SMALL_CONST_PEDERSEN_P2_Y[i];
    }
}

__device__ void small_load_P3(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = SMALL_CONST_PEDERSEN_P3_X[i];
        y.limbs[i] = SMALL_CONST_PEDERSEN_P3_Y[i];
    }
}

__device__ void small_load_shift_point(felt252& x, felt252& y) {
    for (int i = 0; i < 8; i++) {
        x.limbs[i] = SMALL_CONST_SHIFT_POINT_X[i];
        y.limbs[i] = SMALL_CONST_SHIFT_POINT_Y[i];
    }
}

// Negate Y coordinate of a projective point (in Montgomery form)
__device__ void small_negate_projective_y(ProjectivePointCuda& P) {
    felt252 zero = {};
    for (int i = 0; i < 8; i++) zero.limbs[i] = 0;
    P.Y = felt_sub(zero, P.Y);
}

// Convert felt252 to 28 M31 limbs (9 bits each)
__device__ void small_felt252_to_m31_28_limbs(const felt252& value, m31* limbs) {
    uint64_t accumulator = 0;
    int bits_in_acc = 0;
    int limb_idx = 0;

    for (int i = 0; i < 28; i++) {
        while (bits_in_acc < 9 && limb_idx < 8) {
            accumulator |= ((uint64_t)value.limbs[limb_idx]) << bits_in_acc;
            bits_in_acc += 32;
            limb_idx++;
        }
        limbs[i] = (m31){(uint32_t)(accumulator & 0x1FF)};
        accumulator >>= 9;
        bits_in_acc -= 9;
    }
}

// ============================================================================
// EC Point Operations for Table Generation
// ============================================================================

__device__ void small_ec_double_projective(ProjectivePointCuda& P) {
    felt252 X = P.X;
    felt252 Y = P.Y;
    felt252 Z = P.Z;

    // Formula dbl-2007-bl from https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html
    // For curve y^2 = x^3 + ax + b with a = 1 (Starknet curve)
    // w = 3 * X^2 + a * Z^2
    felt252 XX = felt_mul(X, X);
    felt252 ZZ = felt_mul(Z, Z);
    felt252 w = felt_add(felt_add(XX, XX), XX);  // 3*X^2
    w = felt_add(w, ZZ);  // + Z^2

    // s = 2*Y*Z
    felt252 YZ = felt_mul(Y, Z);
    felt252 s = felt_add(YZ, YZ);  // s = 2*Y*Z
    felt252 ss = felt_mul(s, s);
    felt252 sss = felt_mul(s, ss);
    felt252 R = felt_mul(Y, s);    // R = Y*s = 2*Y^2*Z
    felt252 RR = felt_mul(R, R);   // RR = R^2 = 4*Y^4*Z^2

    felt252 X_plus_R = felt_add(X, R);
    felt252 B = felt_mul(X_plus_R, X_plus_R);
    B = felt_sub(B, XX);
    B = felt_sub(B, RR);  // B = (X+R)^2 - X^2 - R^2 = 2*X*R = 4*X*Y^2*Z

    felt252 ww = felt_mul(w, w);
    felt252 two_B = felt_add(B, B);
    felt252 h = felt_sub(ww, two_B);  // h = w^2 - 2*B

    P.X = felt_mul(h, s);  // X3 = h*s

    felt252 B_minus_h = felt_sub(B, h);
    felt252 w_Bh = felt_mul(w, B_minus_h);
    felt252 two_RR = felt_add(RR, RR);
    P.Y = felt_sub(w_Bh, two_RR);  // Y3 = w*(B-h) - 2*RR

    P.Z = sss;  // Z3 = s^3
}

// ============================================================================
// Table Generation Kernels
// ============================================================================

// Point type enum to select which base point to use
enum SmallPedersenPointType {
    SMALL_POINT_P0 = 0,
    SMALL_POINT_P1 = 1,
    SMALL_POINT_P2 = 2,
    SMALL_POINT_P3 = 3
};

__device__ void small_load_base_point(SmallPedersenPointType point_type, felt252& x, felt252& y) {
    switch (point_type) {
        case SMALL_POINT_P0: small_load_P0(x, y); break;
        case SMALL_POINT_P1: small_load_P1(x, y); break;
        case SMALL_POINT_P2: small_load_P2(x, y); break;
        case SMALL_POINT_P3: small_load_P3(x, y); break;
    }
}

// ============================================================================
// Low Section Kernel -- Binary Decomposition (9-bit windows)
// ============================================================================
//
// Generates one window of a low section.
// Entry[k] = -SHIFT + k * (base_point << (9 * window))
// Binary decomposition: k has 9 bits, so at most 9 EC additions per thread.
//
__global__ void gen_pedersen_small_block_optimized_kernel(
    m31** columns,
    uint32_t block_start_row,
    uint32_t n_rows_in_block,
    SmallPedersenPointType point_type,
    uint32_t window
) {
    // Shared memory for 9 precomputed powers: powers[i] = 2^i * scaled_base
    __shared__ AffinePointCuda s_small_powers[INIT_PED_SMALL_BITS_PER_WINDOW];  // 9 powers

    // Phase 0: Thread 0 computes the 9 powers
    if (threadIdx.x == 0) {
        // Load base point from constant memory
        AffinePointCuda base;
        small_load_base_point(point_type, base.x, base.y);

        // Convert to projective (Montgomery form)
        ProjectivePointCuda P = affine_to_projective(base);

        // Scale by 2^(9*window) to get the starting point for this window
        for (uint32_t i = 0; i < INIT_PED_SMALL_BITS_PER_WINDOW * window; i++) {
            small_ec_double_projective(P);
        }

        // Compute 9 powers: powers[i] = 2^i * scaled_base
        for (int i = 0; i < INIT_PED_SMALL_BITS_PER_WINDOW; i++) {
            projective_to_affine(P, s_small_powers[i]);
            small_ec_double_projective(P);
        }
    }
    __syncthreads();

    // Each thread computes its entry using binary decomposition
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_rows_in_block) return;

    // Phase 1: Start with -SHIFT_POINT
    AffinePointCuda shift;
    small_load_shift_point(shift.x, shift.y);
    ProjectivePointCuda acc = affine_to_projective(shift);
    small_negate_projective_y(acc);

    // Phase 2: Binary decomposition - add only powers for set bits
    #pragma unroll
    for (int bit = 0; bit < INIT_PED_SMALL_BITS_PER_WINDOW; bit++) {
        if (k & (1u << bit)) {
            ec_add_mixed(acc, s_small_powers[bit]);
        }
    }

    // Phase 3: Convert to affine (includes field inversion)
    AffinePointCuda result;
    projective_to_affine(acc, result);

    // Phase 4: Store to columns
    m31 x_limbs[28], y_limbs[28];
    small_felt252_to_m31_28_limbs(result.x, x_limbs);
    small_felt252_to_m31_28_limbs(result.y, y_limbs);

    uint32_t output_row = block_start_row + k;
    for (int i = 0; i < 28; i++) {
        columns[i][output_row] = x_limbs[i];
        columns[28 + i][output_row] = y_limbs[i];
    }
}

// ============================================================================
// High Section Kernel -- Binary Decomposition (5-bit rows, 16 sub-blocks)
// ============================================================================
//
// Generates one sub-block of a high section.
// Entry[k] = -SHIFT + subblock_idx * high_point + k * raised_low
// where raised_low = low_point << (num_low_windows * bits_per_window) = low_point << 243
//
// Binary decomposition: k has 5 bits, so at most 5 EC additions per thread.
// subblock_idx additions (at most 15) are done per-thread since it's negligible.
//
__global__ void gen_pedersen_small_high_section_kernel(
    m31** columns,
    uint32_t section_start_row,
    SmallPedersenPointType low_point_type,
    SmallPedersenPointType high_point_type,
    uint32_t subblock_idx
) {
    // Shared memory for precomputed values
    __shared__ AffinePointCuda s_small_powers[INIT_PED_SMALL_HIGH_BITS];  // 5 binary decomposition powers
    __shared__ AffinePointCuda s_small_high_point;  // high point (P1 or P3) in affine form

    // Phase 0: Thread 0 precomputes raised_low powers and high_point
    if (threadIdx.x == 0) {
        // Load low point (P0 or P2) and compute raised_low = low_point << 243
        // (27 low windows * 9 bits per window = 243)
        AffinePointCuda low_base;
        small_load_base_point(low_point_type, low_base.x, low_base.y);
        ProjectivePointCuda raised_low = affine_to_projective(low_base);
        for (int i = 0; i < INIT_PED_SMALL_NUM_LOW_WINDOWS * INIT_PED_SMALL_BITS_PER_WINDOW; i++) {
            small_ec_double_projective(raised_low);
        }

        // Compute 5 binary decomposition powers of raised_low
        for (int i = 0; i < INIT_PED_SMALL_HIGH_BITS; i++) {
            projective_to_affine(raised_low, s_small_powers[i]);
            small_ec_double_projective(raised_low);
        }

        // Load high point (P1 or P3) for sub-block offset
        small_load_base_point(high_point_type, s_small_high_point.x, s_small_high_point.y);
    }
    __syncthreads();

    // Each thread computes its entry
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= INIT_PED_SMALL_HIGH_ROWS_PER_SUBBLOCK) return;

    // Phase 1: Start with -SHIFT_POINT
    AffinePointCuda shift;
    small_load_shift_point(shift.x, shift.y);
    ProjectivePointCuda acc = affine_to_projective(shift);
    small_negate_projective_y(acc);

    // Phase 2: Add subblock_idx copies of high_point (at most 15 additions)
    for (uint32_t j = 0; j < subblock_idx; j++) {
        ec_add_mixed(acc, s_small_high_point);
    }

    // Phase 3: Binary decomposition of k using 5 precomputed powers of raised_low
    #pragma unroll
    for (int bit = 0; bit < INIT_PED_SMALL_HIGH_BITS; bit++) {
        if (k & (1u << bit)) {
            ec_add_mixed(acc, s_small_powers[bit]);
        }
    }

    // Phase 4: Convert to affine (includes field inversion)
    AffinePointCuda result;
    projective_to_affine(acc, result);

    // Phase 5: Store to columns
    m31 x_limbs[28], y_limbs[28];
    small_felt252_to_m31_28_limbs(result.x, x_limbs);
    small_felt252_to_m31_28_limbs(result.y, y_limbs);

    uint32_t output_row = section_start_row + subblock_idx * INIT_PED_SMALL_HIGH_ROWS_PER_SUBBLOCK + k;
    for (int i = 0; i < 28; i++) {
        columns[i][output_row] = x_limbs[i];
        columns[28 + i][output_row] = y_limbs[i];
    }
}

// ============================================================================
// Padding Kernel -- copies row 0 values to padding rows
// ============================================================================
//
// CPU pads with copies of rows[0]. This kernel does the same on GPU.
//
__global__ void pad_pedersen_small_table_kernel(
    m31** columns,
    uint32_t src_row,
    uint32_t pad_start,
    uint32_t pad_end
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dst_row = pad_start + idx;
    if (dst_row >= pad_end) return;

    for (int col = 0; col < INIT_PED_SMALL_TABLE_N_COLUMNS; col++) {
        columns[col][dst_row] = columns[col][src_row];
    }
}

__global__ void set_global_pedersen_small_table_pointers_kernel(m31** ptrs, uint32_t n_rows) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < INIT_PED_SMALL_TABLE_N_COLUMNS; i++) {
            g_pedersen_table_small_columns[i] = ptrs[i];
        }
        g_pedersen_table_small_n_rows = n_rows;
    }
}

// ============================================================================
// External C API
// ============================================================================

extern "C" void initialize_pedersen_table_small() {
    if (s_pedersen_table_small_initialized) {
        return;
    }

    // Compute padded size (next power of 2)
    uint32_t n_rows_unpadded = INIT_PED_SMALL_TABLE_N_ROWS_UNPADDED;  // 28672
    uint32_t n_rows = 1;
    while (n_rows < n_rows_unpadded) n_rows <<= 1;
    // n_rows = 32768 (2^15)

    s_pedersen_table_small_n_rows = n_rows;

    // Allocate GPU memory for each column
    for (int i = 0; i < INIT_PED_SMALL_TABLE_N_COLUMNS; i++) {
        s_pedersen_table_small_ptrs[i] = cuda_malloc<m31>(n_rows);
        cudaMemsetAsync(s_pedersen_table_small_ptrs[i], 0, n_rows * sizeof(m31), 0);
    }

    // Clone pointers to device
    m31** d_columns = clone_to_device<m31*>(s_pedersen_table_small_ptrs, INIT_PED_SMALL_TABLE_N_COLUMNS);

    const uint32_t BLOCK_SIZE = 256;

    // ---- P0 low section: 27 windows x 512 rows ----
    for (uint32_t window = 0; window < INIT_PED_SMALL_NUM_LOW_WINDOWS; window++) {
        uint32_t block_start = INIT_PED_SMALL_P0_LOW_START + window * INIT_PED_SMALL_ROWS_PER_WINDOW;
        uint32_t num_blocks = (INIT_PED_SMALL_ROWS_PER_WINDOW + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gen_pedersen_small_block_optimized_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, block_start, INIT_PED_SMALL_ROWS_PER_WINDOW,
            SMALL_POINT_P0, window
        );
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // ---- P0/P1 high section: 16 sub-blocks x 32 rows ----
    for (uint32_t sb = 0; sb < INIT_PED_SMALL_HIGH_NUM_SUBBLOCKS; sb++) {
        uint32_t num_blocks = (INIT_PED_SMALL_HIGH_ROWS_PER_SUBBLOCK + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gen_pedersen_small_high_section_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, INIT_PED_SMALL_P0P1_HIGH_START,
            SMALL_POINT_P0, SMALL_POINT_P1, sb
        );
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // ---- P2 low section: 27 windows x 512 rows ----
    for (uint32_t window = 0; window < INIT_PED_SMALL_NUM_LOW_WINDOWS; window++) {
        uint32_t block_start = INIT_PED_SMALL_P2_LOW_START + window * INIT_PED_SMALL_ROWS_PER_WINDOW;
        uint32_t num_blocks = (INIT_PED_SMALL_ROWS_PER_WINDOW + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gen_pedersen_small_block_optimized_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, block_start, INIT_PED_SMALL_ROWS_PER_WINDOW,
            SMALL_POINT_P2, window
        );
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // ---- P2/P3 high section: 16 sub-blocks x 32 rows ----
    for (uint32_t sb = 0; sb < INIT_PED_SMALL_HIGH_NUM_SUBBLOCKS; sb++) {
        uint32_t num_blocks = (INIT_PED_SMALL_HIGH_ROWS_PER_SUBBLOCK + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gen_pedersen_small_high_section_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, INIT_PED_SMALL_P2P3_HIGH_START,
            SMALL_POINT_P2, SMALL_POINT_P3, sb
        );
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    // ---- Padding: copy row 0 to fill up to next power of 2 ----
    if (n_rows > n_rows_unpadded) {
        uint32_t pad_count = n_rows - n_rows_unpadded;
        uint32_t num_blocks = (pad_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_pedersen_small_table_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, 0, n_rows_unpadded, n_rows
        );
        ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    }

    // Set global device symbol pointers
    set_global_pedersen_small_table_pointers_kernel<<<1, 1>>>(d_columns, n_rows);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    cuda_free_memory(d_columns);

    s_pedersen_table_small_initialized = true;
}

extern "C" bool is_pedersen_table_small_initialized() {
    return s_pedersen_table_small_initialized;
}

// Get the device pointers and row count for the pedersen small table columns.
// Table must be initialized first (via initialize_pedersen_table_small).
extern "C" void get_pedersen_table_small_column_ptrs(
    m31** output_ptrs,     // Output: 56 device pointers (host-side array)
    uint32_t* out_n_rows   // Output: padded row count
) {
    for (int i = 0; i < INIT_PED_SMALL_TABLE_N_COLUMNS; i++) {
        output_ptrs[i] = s_pedersen_table_small_ptrs[i];
    }
    *out_n_rows = s_pedersen_table_small_n_rows;
}

extern "C" void free_pedersen_table_small() {
    if (!s_pedersen_table_small_initialized) {
        return;
    }

    for (int i = 0; i < INIT_PED_SMALL_TABLE_N_COLUMNS; i++) {
        if (s_pedersen_table_small_ptrs[i] != nullptr) {
            cuda_free_memory(s_pedersen_table_small_ptrs[i]);
            s_pedersen_table_small_ptrs[i] = nullptr;
        }
    }

    s_pedersen_table_small_initialized = false;
    s_pedersen_table_small_n_rows = 0;
}
