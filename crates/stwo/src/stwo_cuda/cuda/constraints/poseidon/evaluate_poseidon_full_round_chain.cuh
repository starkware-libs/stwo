#ifndef EVALUATE_POSEIDON_FULL_ROUND_CHAIN_CUH
#define EVALUATE_POSEIDON_FULL_ROUND_CHAIN_CUH

#include "fields.cuh"
#include "logup.cuh"
#include "relations.cuh"

// Poseidon Full Round Chain evaluator structure
// Must match Rust layout: eval_id, claim (log_size), common_lookup_elements
struct PoseidonFullRoundChain_Eval {
    unsigned eval_id;
    unsigned log_size;  // Claim struct
    CommonLookupElements common_lookup_elements;
};

// Standard extern "C" entry point declaration
extern "C"
void evaluate_poseidon_full_round_chain(
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

#endif // EVALUATE_POSEIDON_FULL_ROUND_CHAIN_CUH
