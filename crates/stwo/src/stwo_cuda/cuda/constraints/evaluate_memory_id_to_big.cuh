#ifndef EVALUATE_MEMORY_ID_TO_BIG_H
#define EVALUATE_MEMORY_ID_TO_BIG_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// Memory ID to Big component evaluator
// Corresponds to cairo-air::components::memory_id_to_big::BigEval

#define N_M31_IN_FELT252 28
#define N_M31_IN_SMALL_FELT252 8
#define LARGE_MEMORY_VALUE_ID_BASE 0x40000000U

struct MemoryIdToBig_BigEval {
    unsigned eval_id;
    unsigned log_n_rows;
    unsigned offset;
    CommonLookupElements common_lookup_elements;
};

struct MemoryIdToBig_SmallEval {
    unsigned eval_id;
    unsigned log_n_rows;
    CommonLookupElements common_lookup_elements;
};

extern "C"
void evaluate_memory_id_to_big_big(
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

extern "C"
void evaluate_memory_id_to_big_small(
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

#endif // EVALUATE_MEMORY_ID_TO_BIG_H
