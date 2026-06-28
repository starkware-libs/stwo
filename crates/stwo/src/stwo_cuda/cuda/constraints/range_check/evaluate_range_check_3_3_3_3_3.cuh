#ifndef EVALUATE_RANGE_CHECK_3_3_3_3_3_H
#define EVALUATE_RANGE_CHECK_3_3_3_3_3_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// Claim for range_check_3_3_3_3_3 component

// Eval parameters passed from Rust (mirrors cairo-air::components::range_check_3_3_3_3_3::Eval)
struct RangeCheck_3_3_3_3_3_Eval {
    unsigned eval_id;
    CommonLookupElements common_lookup_elements;
};

// CUDA AIR evaluator entry point (same signature pattern as other components)
extern "C"
void evaluate_range_check_3_3_3_3_3(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator,
    cudaStream_t stream
);

#endif // EVALUATE_RANGE_CHECK_3_3_3_3_3_H

