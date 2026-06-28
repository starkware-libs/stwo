// CUDA implementation of Pedersen small table storage and lookup (window_bits_9)
// The table stores pre-computed EC points for the Pedersen hash (CanonicalSmall variant)
//
// Table structure (matches CPU implementation with BITS_PER_WINDOW=9):
// - P0 low section:    27 windows x 2^9 rows for value A low bits
// - P0P1 high section: 16 sub-blocks x 2^5 rows for value A high bits
// - P2 low section:    27 windows x 2^9 rows for value B low bits
// - P2P3 high section: 16 sub-blocks x 2^5 rows for value B high bits
//
// Each row contains an (x, y) point where x and y are Felt252 (252-bit field elements)
// stored as 28 M31 limbs each (56 total columns)

#ifndef PEDERSEN_TABLE_SMALL_CUH
#define PEDERSEN_TABLE_SMALL_CUH

#include "fields.cuh"
#include "ec_ops.cuh"

// Table parameters for window_bits_9 (CanonicalSmall)
#define PEDERSEN_SMALL_BITS_PER_WINDOW 9
#define PEDERSEN_SMALL_NUM_LOW_WINDOWS 27   // 252 / 9 - 1 = 27
#define PEDERSEN_SMALL_ROWS_PER_WINDOW (1 << PEDERSEN_SMALL_BITS_PER_WINDOW)  // 512

#define PEDERSEN_SMALL_HIGH_BITS (PEDERSEN_SMALL_BITS_PER_WINDOW - 4)  // 5
#define PEDERSEN_SMALL_HIGH_ROWS_PER_SUBBLOCK (1 << PEDERSEN_SMALL_HIGH_BITS)  // 32
#define PEDERSEN_SMALL_HIGH_NUM_SUBBLOCKS 16

// Section start offsets
#define PEDERSEN_SMALL_P0_LOW_START 0
#define PEDERSEN_SMALL_P0P1_HIGH_START (PEDERSEN_SMALL_NUM_LOW_WINDOWS * PEDERSEN_SMALL_ROWS_PER_WINDOW)  // 13824
#define PEDERSEN_SMALL_P2_LOW_START (PEDERSEN_SMALL_P0P1_HIGH_START + PEDERSEN_SMALL_HIGH_NUM_SUBBLOCKS * PEDERSEN_SMALL_HIGH_ROWS_PER_SUBBLOCK)  // 14336
#define PEDERSEN_SMALL_P2P3_HIGH_START (PEDERSEN_SMALL_P2_LOW_START + PEDERSEN_SMALL_NUM_LOW_WINDOWS * PEDERSEN_SMALL_ROWS_PER_WINDOW)  // 28160
#define PEDERSEN_SMALL_TABLE_N_ROWS_UNPADDED (PEDERSEN_SMALL_P2P3_HIGH_START + PEDERSEN_SMALL_HIGH_NUM_SUBBLOCKS * PEDERSEN_SMALL_HIGH_ROWS_PER_SUBBLOCK)  // 28672

// Table column count: 28 M31 limbs for x + 28 M31 limbs for y = 56 columns
#define PEDERSEN_SMALL_TABLE_N_COLUMNS 56

// Global GPU storage for the Pedersen small table
// This is allocated once and reused across all kernel calls
// Note: Definitions are in pedersen_table_small_init.cu
extern __device__ m31* g_pedersen_table_small_columns[PEDERSEN_SMALL_TABLE_N_COLUMNS];
extern __device__ uint32_t g_pedersen_table_small_n_rows;

// Device function to look up a point from the small table
// table_row: row index in the table
// x_limbs, y_limbs: output arrays (28 M31 each)
__device__ __forceinline__ void pedersen_table_small_lookup(
    uint32_t table_row,
    m31* x_limbs,
    m31* y_limbs
) {
    // Read x coordinate (first 28 columns)
    for (int i = 0; i < 28; i++) {
        x_limbs[i] = g_pedersen_table_small_columns[i][table_row];
    }
    // Read y coordinate (next 28 columns)
    for (int i = 0; i < 28; i++) {
        y_limbs[i] = g_pedersen_table_small_columns[28 + i][table_row];
    }
}

// Reuse felt252_from_28_limbs and felt252_to_28_limbs from pedersen_table.cuh
// (included via that header or already available in compilation units that include both)

#endif // PEDERSEN_TABLE_SMALL_CUH
