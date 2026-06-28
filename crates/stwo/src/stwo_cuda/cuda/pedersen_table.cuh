// CUDA implementation of Pedersen table storage and lookup
// The table stores pre-computed EC points for the Pedersen hash
//
// Table structure (matches CPU implementation):
// - P0 low section:  13 windows x 2^18 rows for value A low bits
// - P0P1 high section: 16 sub-blocks x 2^14 rows for value A high bits
// - P2 low section:  13 windows x 2^18 rows for value B low bits
// - P2P3 high section: 16 sub-blocks x 2^14 rows for value B high bits
//
// Each row contains an (x, y) point where x and y are Felt252 (252-bit field elements)
// stored as 28 M31 limbs each (56 total columns)

#ifndef PEDERSEN_TABLE_CUH
#define PEDERSEN_TABLE_CUH

#include "fields.cuh"
#include "ec_ops.cuh"

// Table parameters (must match Rust constants and pedersen_table_init.cu)
#define PEDERSEN_BITS_PER_WINDOW 18
#define PEDERSEN_NUM_LOW_WINDOWS 13  // 252 / 18 - 1 = 13 low windows per section
#define PEDERSEN_ROWS_PER_WINDOW (1 << PEDERSEN_BITS_PER_WINDOW)  // 262144

#define PEDERSEN_HIGH_BITS (PEDERSEN_BITS_PER_WINDOW - 4)  // 14
#define PEDERSEN_HIGH_ROWS_PER_SUBBLOCK (1 << PEDERSEN_HIGH_BITS)  // 16384
#define PEDERSEN_HIGH_NUM_SUBBLOCKS 16

#define PEDERSEN_P0_LOW_START 0
#define PEDERSEN_P0P1_HIGH_START (PEDERSEN_NUM_LOW_WINDOWS * PEDERSEN_ROWS_PER_WINDOW)  // 3407872
#define PEDERSEN_P2_LOW_START (PEDERSEN_P0P1_HIGH_START + PEDERSEN_HIGH_NUM_SUBBLOCKS * PEDERSEN_HIGH_ROWS_PER_SUBBLOCK)  // 3670016
#define PEDERSEN_P2P3_HIGH_START (PEDERSEN_P2_LOW_START + PEDERSEN_NUM_LOW_WINDOWS * PEDERSEN_ROWS_PER_WINDOW)  // 7077888
#define PEDERSEN_TABLE_N_ROWS_UNPADDED (PEDERSEN_P2P3_HIGH_START + PEDERSEN_HIGH_NUM_SUBBLOCKS * PEDERSEN_HIGH_ROWS_PER_SUBBLOCK)  // 7340032

// Table column count: 28 M31 limbs for x + 28 M31 limbs for y = 56 columns
#define PEDERSEN_TABLE_N_COLUMNS 56

// Global GPU storage for the Pedersen table
// This is allocated once and reused across all kernel calls
// Note: Definitions are in pedersen_table_init.cu
extern __device__ m31* g_pedersen_table_columns[PEDERSEN_TABLE_N_COLUMNS];
extern __device__ uint32_t g_pedersen_table_n_rows;

// Device function to look up a point from the table
// table_row: row index in the table
// x_limbs, y_limbs: output arrays (28 M31 each)
__device__ void pedersen_table_lookup(
    uint32_t table_row,
    m31* x_limbs,
    m31* y_limbs
);

// Device function to reconstruct Felt252 from 28 M31 limbs
// Input: 28 M31 values where each is a 9-bit limb
// Output: 256-bit value in ff_storage<8> format
__device__ void felt252_from_28_limbs(felt252& result, const m31* limbs);

// Device function to split Felt252 into 28 M31 limbs
// Input: 256-bit value in ff_storage<8> format
// Output: 28 M31 values (9 bits each)
__device__ void felt252_to_28_limbs(const felt252& value, m31* limbs);

// ============================================================================
// Implementation
// ============================================================================

__device__ __forceinline__ void pedersen_table_lookup(
    uint32_t table_row,
    m31* x_limbs,
    m31* y_limbs
) {
    // Read x coordinate (first 28 columns)
    for (int i = 0; i < 28; i++) {
        x_limbs[i] = g_pedersen_table_columns[i][table_row];
    }
    // Read y coordinate (next 28 columns)
    for (int i = 0; i < 28; i++) {
        y_limbs[i] = g_pedersen_table_columns[28 + i][table_row];
    }
}

__device__ __forceinline__ void felt252_from_28_limbs(felt252& result, const m31* limbs) {
    // Clear result
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = 0;
    }

    // Each M31 limb is 9 bits (252 bits total = 28 x 9)
    // We need to combine them into 8 x 32 = 256 bits
    uint64_t accumulator = 0;
    int bits_in_acc = 0;
    int result_idx = 0;

    for (int i = 0; i < 28; i++) {
        uint64_t value = limbs[i] & 0x1FF;  // 9 bits
        accumulator |= (value << bits_in_acc);
        bits_in_acc += 9;

        // Extract 32-bit chunks when we have enough
        while (bits_in_acc >= 32 && result_idx < 8) {
            result.limbs[result_idx] = (uint32_t)(accumulator & 0xFFFFFFFF);
            accumulator >>= 32;
            bits_in_acc -= 32;
            result_idx++;
        }
    }

    // Handle any remaining bits
    if (result_idx < 8 && accumulator != 0) {
        result.limbs[result_idx] = (uint32_t)accumulator;
    }
}

__device__ __forceinline__ void felt252_to_28_limbs(const felt252& value, m31* limbs) {
    // Extract 9 bits at a time from the 256-bit value
    uint64_t accumulator = 0;
    int bits_in_acc = 0;
    int limb_idx = 0;

    for (int i = 0; i < 28; i++) {
        // Load more bits if needed
        while (bits_in_acc < 9 && limb_idx < 8) {
            accumulator |= ((uint64_t)value.limbs[limb_idx]) << bits_in_acc;
            bits_in_acc += 32;
            limb_idx++;
        }

        // Extract 9 bits
        limbs[i] = (m31){(uint32_t)(accumulator & 0x1FF)};
        accumulator >>= 9;
        bits_in_acc -= 9;
    }
}

#endif // PEDERSEN_TABLE_CUH
