#ifndef EVALUATE_PEDERSEN_BUILTIN_NARROW_WINDOWS_H
#define EVALUATE_PEDERSEN_BUILTIN_NARROW_WINDOWS_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// Reuse the same struct layout as PedersenBuiltin since the Claim fields are identical
// (log_size + pedersen_builtin_segment_start).
struct PedersenBuiltinNarrowWindows_Eval {
    unsigned eval_id;
    struct {
        unsigned log_size;
        unsigned pedersen_builtin_segment_start;
    } claim;
    CommonLookupElements common_lookup_elements;
};

extern "C"
void evaluate_pedersen_builtin_narrow_windows(
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

#endif // EVALUATE_PEDERSEN_BUILTIN_NARROW_WINDOWS_H
