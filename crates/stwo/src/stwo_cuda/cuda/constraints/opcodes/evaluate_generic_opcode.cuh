#ifndef EVALUATE_GENERIC_OPCODE_CUH
#define EVALUATE_GENERIC_OPCODE_CUH

#include "constraints/relations.cuh"

// Generic opcode component has 243 trace columns
// This is a complex opcode that handles generic Cairo VM instructions
// with four subroutines: DecodeGenericInstruction, EvalOperands, HandleOpcodes, UpdateRegisters

// Component Eval struct matching Rust definition
// Must match memory layout of generic_opcode::Eval in generic_opcode.rs
struct GenericOpcode_Eval {
    uint32_t eval_id;
    uint32_t log_size;  // from Claim
    CommonLookupElements common_lookup_elements;
};

extern "C"
void evaluate_generic_opcode(
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

#endif // EVALUATE_GENERIC_OPCODE_CUH
