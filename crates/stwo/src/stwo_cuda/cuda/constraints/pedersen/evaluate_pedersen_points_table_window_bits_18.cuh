/*
============================================
PedersenPointsTableWindowBits18 CUDA Evaluator Header
============================================

Component: PedersenPointsTableWindowBits18
Translated from: cairo-air/src/components/pedersen_points_table_window_bits_18.rs

Trace columns: 1 (multiplicity)
Preprocessed columns: 57 (seq_23 + pedersen_points_0..55)

This component is a lookup table provider for Pedersen hash precomputed
elliptic curve points (window_bits_18 variant). It has no algebraic
constraints -- only a single PROVIDE-side relation lookup using
PEDERSEN_POINTS_TABLE_RELATION_ID with 58 values:
  (relation_id, seq_23, pedersen_points_0..55)

Relation uses per row: none (RELATION_USES_PER_ROW = [])
LOG_SIZE = 23
============================================
*/

#ifndef EVALUATE_PEDERSEN_POINTS_TABLE_WINDOW_BITS_18_CUH
#define EVALUATE_PEDERSEN_POINTS_TABLE_WINDOW_BITS_18_CUH

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// Eval struct -- matches the Rust #[repr(C)] layout:
//   eval_id (unsigned, injected by FrameworkEval)
//   claim   (empty struct Claim {})
//   common_lookup_elements (CommonLookupElements = LookupElementsBasic<128>)
struct PedersenPointsTableWindowBits18_Eval {
    unsigned eval_id;
    // Claim is empty -- no fields
    CommonLookupElements common_lookup_elements;
};

extern "C" void evaluate_pedersen_points_table_window_bits_18(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations, unsigned trace0_evaluations_len,
    m31 **trace1_evaluations, unsigned trace1_evaluations_len,
    m31 **trace2_evaluations, unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size, unsigned int eval_domain_log_size,
    unsigned int number_of_columns, unsigned int logup_counts,
    void *eval, qm31 cumsum_shift,
    bool should_accumulate, bool use_assert_evaluator,
    cudaStream_t stream
);

#endif // EVALUATE_PEDERSEN_POINTS_TABLE_WINDOW_BITS_18_CUH
