/*
============================================
PartialEcMulWindowBits18 CUDA Evaluator Header
============================================

Component: PartialEcMulWindowBits18
Translated from: cairo-air/src/components/partial_ec_mul_window_bits_18.rs

Trace columns: 297 (no preprocessed columns)
  - input_limb[72]: cols 0-71
  - pedersen_points_table_window_bits_18_output[56]: cols 72-127
  - slope[28]: cols 128-155
  - k (VerifyMul252 #1): col 156
  - carry[27] (VerifyMul252 #1): cols 157-183
  - result_x[28]: cols 184-211
  - k (VerifyMul252 #2): col 212
  - carry[27] (VerifyMul252 #2): cols 213-239
  - result_y[28]: cols 240-267
  - k (VerifyMul252 #3): col 268
  - carry[27] (VerifyMul252 #3): cols 269-295
  - enabler: col 296

Interaction columns: SECURE_EXTENSION_DEGREE * 65 = 260

Constraint Logic:
  - 1 enabler constraint: enabler^2 = enabler
  - EcAdd subroutine (inlined):
    - 3x RangeCheckMemValueN28 (14 lookups each = 42 lookups total)
    - 3x VerifyMul252 (28 lookups + 27 constraints each = 84 lookups, 81 constraints)
  - 1x PedersenPointsTableWindowBits18 lookup (USE side)
  - 1x PartialEcMulWindowBits18 relation (+enabler multiplicity)
  - 1x PartialEcMulWindowBits18 relation (-enabler multiplicity)

Total: 129 add_to_relation calls -> 65 logup interaction columns (finalized in pairs)

Relation Lookups:
  - PartialEcMulWindowBits18 (M31_1621226978): 2 uses (+enabler, -enabler)
  - PedersenPointsTableWindowBits18 (M31_1444721856): 1 use
  - RangeCheck_20 (A-H variants): 84 uses (from 3x VerifyMul252)
  - RangeCheck_9_9 (A-H variants): 42 uses (from 3x RangeCheckMemValueN28)
============================================
*/

#ifndef EVALUATE_PARTIAL_EC_MUL_WINDOW_BITS_18_CUH
#define EVALUATE_PARTIAL_EC_MUL_WINDOW_BITS_18_CUH

#include "fields.cuh"
#include "logup.cuh"
#include "relations.cuh"

// PartialEcMulWindowBits18 Evaluation Structure
// Must match the Rust Eval structure layout: eval_id, claim (log_size), common_lookup_elements
struct PartialEcMulWindowBits18_Eval {
    unsigned eval_id;
    struct {
        unsigned log_size;
    } claim;
    CommonLookupElements common_lookup_elements;
};

// Host function declaration
extern "C" void evaluate_partial_ec_mul_window_bits_18(
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

#endif // EVALUATE_PARTIAL_EC_MUL_WINDOW_BITS_18_CUH
