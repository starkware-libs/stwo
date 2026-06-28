#ifndef EVALUATE_POSEIDON_ROUND_KEYS_CUH
#define EVALUATE_POSEIDON_ROUND_KEYS_CUH

#include "fields.cuh"
#include "logup.cuh"
#include "relations.cuh"

// Poseidon Round Keys evaluator structure
// Must match Rust layout: eval_id, claim (empty), common_lookup_elements
struct PoseidonRoundKeys_Eval {
    unsigned eval_id;
    // Claim struct is empty in Rust
    CommonLookupElements common_lookup_elements;
};

// Pre-kernel function declaration
template <typename EvaluatorT>
__global__ void evaluate_poseidon_round_keys_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PoseidonRoundKeys_Eval *poseidon_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
);

// Standard extern "C" entry point declaration
extern "C"
void evaluate_poseidon_round_keys(
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

#endif // EVALUATE_POSEIDON_ROUND_KEYS_CUH
