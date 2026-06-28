#ifndef EVALUATE_JUMP_OPCODE_REL_H
#define EVALUATE_JUMP_OPCODE_REL_H

#include "constraints/relations.cuh"

// Claim structure matching the Rust Claim struct
struct JumpOpcodeRel_Claim {
    unsigned log_size;
};

// Structure to hold lookup elements for jump_opcode_rel component
struct JumpOpcodeRel_Eval {
    unsigned eval_id;
    JumpOpcodeRel_Claim claim;  // Must match Rust struct layout
    CommonLookupElements common_lookup_elements;
};

// External function declaration for CUDA kernel
extern "C" void evaluate_jump_opcode_rel(
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

#endif // EVALUATE_JUMP_OPCODE_REL_H
