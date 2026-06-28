#ifndef EVALUATE_JNZ_OPCODE_CUH
#define EVALUATE_JNZ_OPCODE_CUH

#include "constraints/relations.cuh"

// JNZ opcode component has 37 trace columns
// This opcode checks if destination value is zero:
// - If dst == 0: PC advances by 2 (not taken)
// - If dst != 0: would be handled by jnz_opcode_taken (different component)
// This component specifically handles the "not taken" case where dst == 0

// Component Eval struct matching Rust definition
// Must match memory layout of jnz_opcode::Eval in jnz_opcode.rs
struct JnzOpcode_Eval {
    uint32_t eval_id;
    uint32_t log_size;  // from Claim
    CommonLookupElements common_lookup_elements;
};

extern "C"
void evaluate_jnz_opcode(
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

#endif // EVALUATE_JNZ_OPCODE_CUH
