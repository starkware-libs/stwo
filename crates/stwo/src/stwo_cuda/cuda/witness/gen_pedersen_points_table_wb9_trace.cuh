// CUDA interaction trace generation for pedersen_points_table_window_bits_9 component.
//
// The small pedersen_points_table is a preprocessed lookup table (~32K rows, LOG_SIZE=15)
// with 56 columns of EC point coordinates. The interaction trace has a single logup
// column: numerator = -mults[row], denominator = lookup_elements.combine(58 values)
// where values = [relation_id, seq (=row_index), table_col_0..table_col_55].

#ifndef GEN_PEDERSEN_POINTS_TABLE_WB9_TRACE_CUH
#define GEN_PEDERSEN_POINTS_TABLE_WB9_TRACE_CUH

#include "fields.cuh"

// Number of lookup values per row: 1 (relation_id) + 1 (seq) + 56 (table columns)
#define PEDERSEN_POINTS_TABLE_WB9_N_LOOKUP_VALUES 58

// Number of table columns (same as w18: 28 x-limbs + 28 y-limbs)
#define PEDERSEN_TABLE_SMALL_N_COLUMNS 56

// Relation ID for PedersenPointsTableWindowBits9
#define PEDERSEN_POINTS_TABLE_WB9_RELATION_ID 1791500038u

// Add inputs to small pedersen_points_table multiplicity tracking on GPU.
extern "C"
void pedersen_points_table_small_add_inputs(
    m31* indices,           // Table indices (one per row)
    unsigned int n_rows,    // Number of rows
    m31* mults,             // Output multiplicities (atomically updated)
    unsigned int mults_log_size  // Log2 of multiplicities array size
);

// Generate interaction trace for pedersen_points_table_wb9 on GPU.
extern "C"
void pedersen_points_table_wb9_interaction_trace(
    void *lookup_elements,          // LookupElementsBasic<58> from Rust
    m31 *multiplicities,            // GPU multiplicities (u32 per row)
    unsigned log_size,              // Log2 of table size (15)
    m31 **interaction_traces,       // 4 output columns (qm31 components)
    m31 *claimed_sum                // Output claimed sum (4 x m31)
);

#endif // GEN_PEDERSEN_POINTS_TABLE_WB9_TRACE_CUH
