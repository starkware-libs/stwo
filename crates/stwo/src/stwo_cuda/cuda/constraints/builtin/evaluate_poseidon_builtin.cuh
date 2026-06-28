// ============================================================================
// Poseidon Builtin CUDA Evaluator Header
// ============================================================================

#ifndef EVALUATE_POSEIDON_BUILTIN_CUH
#define EVALUATE_POSEIDON_BUILTIN_CUH

#include "relations.cuh"

// ============================================================================
// PoseidonBuiltin Evaluation Structure
// ============================================================================
// Contains claim data and relation lookup elements for Poseidon builtin
// Must match the Rust Eval structure in poseidon_builtin.rs
struct PoseidonBuiltin_Eval {
    uint32_t eval_id;

    // Claim data
    struct {
        uint32_t log_size;
        uint32_t poseidon_builtin_segment_start;
    } claim;

    // Relation lookup elements
    CommonLookupElements common_lookup_elements;
};

// Host function declaration
extern "C" void evaluate_poseidon_builtin(
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

#endif // EVALUATE_POSEIDON_BUILTIN_CUH
