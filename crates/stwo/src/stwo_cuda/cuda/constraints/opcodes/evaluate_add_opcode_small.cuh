#ifndef EVALUATE_ADD_SMALL_OPCODE_H
#define EVALUATE_ADD_SMALL_OPCODE_H

#include "constraints/relations.cuh"

struct AddCodeSmall_Claim {
    unsigned log_size;
};

struct AddCodeSmall_Eval {
    unsigned eval_id;
    AddCodeSmall_Claim Claim;
    CommonLookupElements common_lookup_elements;
};

extern "C"
void evaluate_add_opcode_small(
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




#endif // EVALUATE_ADD_SMALL_OPCODE_H