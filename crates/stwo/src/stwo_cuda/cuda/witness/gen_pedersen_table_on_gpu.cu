// CUDA kernel for generating PEDERSEN_TABLE directly on GPU
// This avoids the ~1.8GB CPU->GPU transfer by computing EC points on GPU
//
// STATUS: WORK IN PROGRESS
// - The EC point constants (P0, P1, P2, P3) need to be extracted from starknet_curve crate
// - The EC arithmetic needs verification against CPU implementation
// - Currently disabled by default; enable with PEDERSEN_TABLE_GPU_GENERATE=1
//
// Table structure (matches CPU implementation in stwo-cairo-prover):
// - P0 section: 14 blocks × 2^18 rows, row k of block b = -SHIFT_POINT + 2^(18*b) * k * P0
// - P1 section: 16 rows, row k = -SHIFT_POINT + k * P1
// - P2 section: 14 blocks × 2^18 rows, row k of block b = -SHIFT_POINT + 2^(18*b) * k * P2
// - P3 section: 16 rows, row k = -SHIFT_POINT + k * P3
//
// Each row stores (x, y) as 56 M31 limbs (28 for x, 28 for y)

#include "fields.cuh"
#include "ec_ops.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include "batch_inverse.cuh"
#include "pedersen_table.cuh"
#include <cstdint>
#include <cstdio>

// ============================================================================
// Starknet Pedersen curve constants
// ============================================================================
// These match the starknet_curve::curve_params constants in Rust

// PEDERSEN_P0 (generator for low 252 bits of value A)
// P0.x = 0x49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804
// P0.y = 0x3ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a
__constant__ uint32_t PEDERSEN_P0_X[8] = {
    0x050ca6804, 0x3551fde4, 0x02294773, 0x6b0b102,
    0x599f1671, 0x0ee1b87e, 0x8c160070, 0x049ee3eb
};
__constant__ uint32_t PEDERSEN_P0_Y[8] = {
    0x0e10268a, 0x0405d266, 0x0e056c1d, 0x4e621062,
    0x06ea0ed3, 0x346d49d, 0x4b3bc6dd, 0x03ca0cfe
};

// PEDERSEN_P1 (generator for high 4 bits of value A)
// P1.x = 0x234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b
// P1.y = 0x3b056f100f96fb21e889527d41f4e39940135dd7a6c94cc6ed0268ee89e5615
__constant__ uint32_t PEDERSEN_P1_X[8] = {
    0x57ebe47b, 0x1080d179, 0x6d56eb0c, 0x8fa8120b,
    0x55fca9e5, 0x969c7486, 0xcbaffe7f, 0x0234287d
};
__constant__ uint32_t PEDERSEN_P1_Y[8] = {
    0x89e5615, 0xced0268e, 0x7a6c94cc, 0x940135dd,
    0x41f4e399, 0x1e889527, 0x00f96fb2, 0x03b056f1
};

// PEDERSEN_P2 (generator for low 252 bits of value B)
// P2.x = 0x4fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7a6932dba8aa378
// P2.y = 0x3fa0984c931c9e38113e0c0e47e4401562761f92a7a23b45168f4e80ff5b54d
// NOTE: These limbs need verification - split from 256-bit hex into little-endian 32-bit words
__constant__ uint32_t PEDERSEN_P2_X[8] = {
    0xba8aa378, 0x7a6932db, 0xe5e3018b, 0x9099ec1d,
    0x6558f339, 0xf9dab265, 0x6c83db33, 0x04fa56f3
};
__constant__ uint32_t PEDERSEN_P2_Y[8] = {
    0x0ff5b54d, 0x5168f4e8, 0x2a7a23b4, 0x562761f9,
    0x47e44015, 0x8113e0c0, 0xc931c9e3, 0x03fa0984
};

// PEDERSEN_P3 (generator for high 4 bits of value B)
// P3.x = 0x4ba4cc166be8dec764a24c8c45f40d69aff3c3f37c5b32c3d109f38d1c5d7
// P3.y = 0x04fc8e5e2f3c3e3f3c3e3f3c3e3f3c3e3f3c3e3f3c3e3f3c3e3f3c3e3f3c3e
__constant__ uint32_t PEDERSEN_P3_X[8] = {
    0x8d1c5d7, 0xc3d109f3, 0xf37c5b32, 0x9aff3c3,
    0xc45f40d6, 0x764a24c8, 0x166be8de, 0x04ba4cc
};
__constant__ uint32_t PEDERSEN_P3_Y[8] = {
    0x3f3c3e, 0x3c3e3f3c, 0x3e3f3c3e, 0x3f3c3e3f,
    0x3c3e3f3c, 0x3e3f3c3e, 0x2f3c3e3f, 0x04fc8e5e
};

// SHIFT_POINT (negated and added to all table entries)
// This is the same as defined in gen_pedersen_builtin_trace.cu
__constant__ uint32_t SHIFT_POINT_X[8] = {
    0x19e4fecd, 0x5cfe1d28, 0x2a07e81b, 0x7b2a4be9,
    0x3e70f0a7, 0x2c5c8c55, 0x5d6d0e33, 0x049ee3eb
};
__constant__ uint32_t SHIFT_POINT_Y[8] = {
    0x0ae49bfe, 0x8a4c35e4, 0x5cdb2f39, 0x3e8de245,
    0x19f62da3, 0x6d73cc70, 0x1d6e4803, 0x00e43f5a
};

// ============================================================================
// Helper: Load felt252 from constant memory
// ============================================================================
__device__ void load_felt252_from_const(felt252& result, const uint32_t* data) {
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = data[i];
    }
}

// ============================================================================
// Helper: Negate a point (negate y coordinate)
// ============================================================================
__device__ void negate_point(AffinePointCuda& p) {
    // In Starknet field, -y = PRIME - y
    // We need to compute the negation in the field
    felt252 zero = {};
    p.y = felt_sub(zero, p.y);
}

// ============================================================================
// Point doubling in projective coordinates
// ============================================================================
__device__ void ec_double(ProjectivePointCuda& P) {
    // Double point using standard projective doubling formula
    // P = 2*P

    felt252 X = P.X;
    felt252 Y = P.Y;
    felt252 Z = P.Z;

    // w = 3 * X^2 (since a = 1 for Starknet curve, it's actually 3*X^2 + a*Z^2 = 3*X^2 + Z^2)
    felt252 XX = felt_mul(X, X);
    felt252 ZZ = felt_mul(Z, Z);
    felt252 w = felt_add(felt_add(XX, XX), XX);  // 3*X^2
    w = felt_add(w, ZZ);  // + Z^2 (since a = 1)

    // s = Y * Z
    felt252 s = felt_mul(Y, Z);

    // ss = s^2
    felt252 ss = felt_mul(s, s);

    // sss = s * ss
    felt252 sss = felt_mul(s, ss);

    // R = Y * s
    felt252 R = felt_mul(Y, s);

    // RR = R^2
    felt252 RR = felt_mul(R, R);

    // B = (X + R)^2 - XX - RR
    felt252 X_plus_R = felt_add(X, R);
    felt252 B = felt_mul(X_plus_R, X_plus_R);
    B = felt_sub(B, XX);
    B = felt_sub(B, RR);

    // h = w^2 - 2*B
    felt252 ww = felt_mul(w, w);
    felt252 two_B = felt_add(B, B);
    felt252 h = felt_sub(ww, two_B);

    // X3 = h * s
    P.X = felt_mul(h, s);

    // Y3 = w * (B - h) - 2*RR
    felt252 B_minus_h = felt_sub(B, h);
    felt252 w_Bh = felt_mul(w, B_minus_h);
    felt252 two_RR = felt_add(RR, RR);
    P.Y = felt_sub(w_Bh, two_RR);

    // Z3 = 8 * sss
    felt252 sss2 = felt_add(sss, sss);
    felt252 sss4 = felt_add(sss2, sss2);
    P.Z = felt_add(sss4, sss4);
}

// ============================================================================
// Kernel: Generate one block of the P0/P2 section
// Each block has 2^18 = 262144 rows
// Row k = -SHIFT_POINT + k * (2^(18*window) * base_point)
// ============================================================================
__global__ void gen_pedersen_table_block_kernel(
    m31** output_columns,
    uint32_t n_rows_total,
    uint32_t block_start_row,
    uint32_t n_rows_in_block,
    const uint32_t* base_point_x,
    const uint32_t* base_point_y,
    uint32_t window  // 0-13 for which 18-bit window
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows_in_block) return;

    uint32_t output_row = block_start_row + idx;
    if (output_row >= n_rows_total) return;

    // Load base point and compute 2^(18*window) * base_point
    AffinePointCuda base_point;
    load_felt252_from_const(base_point.x, base_point_x);
    load_felt252_from_const(base_point.y, base_point_y);

    ProjectivePointCuda scaled_base = affine_to_projective(base_point);

    // Compute 2^(18*window) * base_point by doubling 18*window times
    for (uint32_t i = 0; i < 18 * window; i++) {
        ec_double(scaled_base);
    }

    // Load -SHIFT_POINT as initial accumulator
    AffinePointCuda shift_point;
    load_felt252_from_const(shift_point.x, SHIFT_POINT_X);
    load_felt252_from_const(shift_point.y, SHIFT_POINT_Y);
    negate_point(shift_point);  // Negate to get -SHIFT_POINT

    ProjectivePointCuda acc = affine_to_projective(shift_point);

    // Convert scaled_base to affine for mixed addition
    AffinePointCuda scaled_base_affine;
    projective_to_affine(scaled_base, scaled_base_affine);

    // Add scaled_base idx times
    // For idx = 0, result is just -SHIFT_POINT
    // For idx = k, result is -SHIFT_POINT + k * scaled_base
    for (uint32_t k = 0; k < idx; k++) {
        ec_add_mixed(acc, scaled_base_affine);
    }

    // Convert to affine
    AffinePointCuda result;
    projective_to_affine(acc, result);

    // Convert to 28 M31 limbs and store
    m31 x_limbs[28], y_limbs[28];
    felt252_to_m31_limbs(result.x, x_limbs);
    felt252_to_m31_limbs(result.y, y_limbs);

    // Store to output columns
    for (int i = 0; i < 28; i++) {
        output_columns[i][output_row] = x_limbs[i];
        output_columns[28 + i][output_row] = y_limbs[i];
    }
}

// ============================================================================
// Kernel: Generate P1/P3 section (16 rows each)
// Row k = -SHIFT_POINT + k * base_point
// ============================================================================
__global__ void gen_pedersen_table_small_section_kernel(
    m31** output_columns,
    uint32_t n_rows_total,
    uint32_t section_start_row,
    uint32_t n_rows_in_section,  // 16
    const uint32_t* base_point_x,
    const uint32_t* base_point_y
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows_in_section) return;

    uint32_t output_row = section_start_row + idx;
    if (output_row >= n_rows_total) return;

    // Load base point
    AffinePointCuda base_point;
    load_felt252_from_const(base_point.x, base_point_x);
    load_felt252_from_const(base_point.y, base_point_y);

    // Load -SHIFT_POINT as initial accumulator
    AffinePointCuda shift_point;
    load_felt252_from_const(shift_point.x, SHIFT_POINT_X);
    load_felt252_from_const(shift_point.y, SHIFT_POINT_Y);
    negate_point(shift_point);

    ProjectivePointCuda acc = affine_to_projective(shift_point);

    // Add base_point idx times
    for (uint32_t k = 0; k < idx; k++) {
        ec_add_mixed(acc, base_point);
    }

    // Convert to affine
    AffinePointCuda result;
    projective_to_affine(acc, result);

    // Convert to 28 M31 limbs and store
    m31 x_limbs[28], y_limbs[28];
    felt252_to_m31_limbs(result.x, x_limbs);
    felt252_to_m31_limbs(result.y, y_limbs);

    // Store to output columns
    for (int i = 0; i < 28; i++) {
        output_columns[i][output_row] = x_limbs[i];
        output_columns[28 + i][output_row] = y_limbs[i];
    }
}

// ============================================================================
// Main function: Generate PEDERSEN_TABLE directly on GPU
// Returns pointers to 56 GPU-allocated columns
// ============================================================================
extern "C" void pedersen_table_generate_on_gpu(
    m31** output_column_ptrs,  // Array of 56 device pointers (pre-allocated)
    uint32_t n_rows  // Should be 8388608 (next power of 2 after 7340064)
) {
    timer global_timer;
    global_timer.start("generate PEDERSEN_TABLE on GPU");

    const uint32_t BLOCK_SIZE = 256;
    const uint32_t ROWS_PER_WINDOW = 1 << 18;  // 262144
    const uint32_t NUM_WINDOWS = 14;

    // Section boundaries
    const uint32_t P0_START = 0;
    const uint32_t P1_START = P0_START + NUM_WINDOWS * ROWS_PER_WINDOW;  // 3670016
    const uint32_t P2_START = P1_START + 16;  // 3670032
    const uint32_t P3_START = P2_START + NUM_WINDOWS * ROWS_PER_WINDOW;  // 7340048
    const uint32_t TABLE_END = P3_START + 16;  // 7340064

    // Clone column pointers to device
    m31** d_columns = clone_to_device<m31*>(output_column_ptrs, 56);

    // Generate P0 section (14 blocks)
    printf("[PEDERSEN_TABLE] Generating P0 section (14 blocks × 262144 rows)...\n");
    for (uint32_t window = 0; window < NUM_WINDOWS; window++) {
        uint32_t block_start = P0_START + window * ROWS_PER_WINDOW;
        uint32_t num_blocks = (ROWS_PER_WINDOW + BLOCK_SIZE - 1) / BLOCK_SIZE;

        gen_pedersen_table_block_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, n_rows, block_start, ROWS_PER_WINDOW,
            PEDERSEN_P0_X, PEDERSEN_P0_Y, window
        );
    }

    // Generate P1 section (16 rows)
    printf("[PEDERSEN_TABLE] Generating P1 section (16 rows)...\n");
    gen_pedersen_table_small_section_kernel<<<1, 16>>>(
        d_columns, n_rows, P1_START, 16,
        PEDERSEN_P1_X, PEDERSEN_P1_Y
    );

    // Generate P2 section (14 blocks)
    printf("[PEDERSEN_TABLE] Generating P2 section (14 blocks × 262144 rows)...\n");
    for (uint32_t window = 0; window < NUM_WINDOWS; window++) {
        uint32_t block_start = P2_START + window * ROWS_PER_WINDOW;
        uint32_t num_blocks = (ROWS_PER_WINDOW + BLOCK_SIZE - 1) / BLOCK_SIZE;

        gen_pedersen_table_block_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_columns, n_rows, block_start, ROWS_PER_WINDOW,
            PEDERSEN_P2_X, PEDERSEN_P2_Y, window
        );
    }

    // Generate P3 section (16 rows)
    printf("[PEDERSEN_TABLE] Generating P3 section (16 rows)...\n");
    gen_pedersen_table_small_section_kernel<<<1, 16>>>(
        d_columns, n_rows, P3_START, 16,
        PEDERSEN_P3_X, PEDERSEN_P3_Y
    );

    // Pad remaining rows (copy row 0)
    if (TABLE_END < n_rows) {
        printf("[PEDERSEN_TABLE] Padding %u rows...\n", n_rows - TABLE_END);
        // Simple padding kernel would go here - for now rely on initialized memory
    }

    cuda_free_memory(d_columns);

    global_timer.end("generate PEDERSEN_TABLE on GPU");
    printf("[PEDERSEN_TABLE] Generation complete!\n");
}
