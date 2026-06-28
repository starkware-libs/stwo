/*
============================================
PedersenAggregatorWindowBits18 CUDA Evaluator Header
============================================

Component: PedersenAggregatorWindowBits18 (206 trace columns, 1 preprocessed column)
Translated from: cairo-air/src/components/pedersen_aggregator_window_bits_18.rs

Trace columns: 206
Preprocessed columns: 1 (seq)

This component performs the Pedersen aggregation with window bits 18:
1. 2x ReadPositiveKnownIdNumBits252 (reads 2 input values from memory by known ID)
2. 2x VerifyReduced252 (verifies field element bounds)
3. 4x PartialEcMulWindowBits18 lookups (2 input, 2 output for 2 chains)
4. 1x MemoryIdToBig lookup for output
5. 1x PedersenAggregatorWindowBits18 PROVIDE relation

Relation uses per row:
  MemoryIdToBig: 3, PartialEcMulWindowBits18: 2, RangeCheck_8: 4

interaction_log_sizes = SECURE_EXTENSION_DEGREE * 6 = 24
logup_counts = 12 (6 pairs)
============================================
*/

#ifndef EVALUATE_PEDERSEN_AGGREGATOR_CUH
#define EVALUATE_PEDERSEN_AGGREGATOR_CUH

#include "fields.cuh"
#include "logup.cuh"
#include "relations.cuh"

// PedersenAggregatorWindowBits18 Evaluation Structure
// Must match the Rust Eval structure layout: eval_id, claim (log_size), common_lookup_elements
struct PedersenAggregator_WB18_Eval {
    unsigned eval_id;
    struct {
        unsigned log_size;
    } claim;
    CommonLookupElements common_lookup_elements;
};

// Host function declaration
extern "C" void evaluate_pedersen_aggregator_window_bits_18(
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

#endif // EVALUATE_PEDERSEN_AGGREGATOR_CUH
