#ifndef EVALUATE_BLAKE_G_CONSTRAINT_H
#define EVALUATE_BLAKE_G_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define N_BLAKE_G_LOG_INSTANCES_PER_ROW 0
#define N_BLAKE_G_INSTANCES_PER_ROW (1 << N_BLAKE_G_LOG_INSTANCES_PER_ROW)

struct BlakeG_Claim {
    unsigned log_size;
};

struct BlakeG_Eval {
    unsigned eval_id;
    BlakeG_Claim Claim;
    CommonLookupElements common_lookup_elements;
};

template<typename EvaluatorT>
DEVICE_FORCEINLINE void triple_sum32_evaluate(
    m31 triple_sum_32_input_limb_0,
    m31 triple_sum_32_input_limb_1,
    m31 triple_sum_32_input_limb_2,
    m31 triple_sum_32_input_limb_3,
    m31 triple_sum_32_input_limb_4,
    m31 triple_sum_32_input_limb_5,
    m31 triple_sum32_res_limb_0_col0,
    m31 triple_sum32_res_limb_1_col1,

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_1 = {1};
    const m31 M31_2 = {2};
    const m31 M31_32768 = {32768};

    m31 carry_low_tmp_541fa_1 = add(triple_sum_32_input_limb_0, triple_sum_32_input_limb_2);
    carry_low_tmp_541fa_1 = add(carry_low_tmp_541fa_1, triple_sum_32_input_limb_4);
    carry_low_tmp_541fa_1 = sub(carry_low_tmp_541fa_1, triple_sum32_res_limb_0_col0);
    carry_low_tmp_541fa_1 = mul(carry_low_tmp_541fa_1, M31_32768);
    cuda_evaluator->add_constraint(mul(mul(sub(carry_low_tmp_541fa_1, M31_1), carry_low_tmp_541fa_1), sub(carry_low_tmp_541fa_1, M31_2)));

    m31 carry_high_tmp_541fa_2 = add(triple_sum_32_input_limb_1, triple_sum_32_input_limb_3);
    carry_high_tmp_541fa_2 = add(carry_high_tmp_541fa_2, triple_sum_32_input_limb_5);
    carry_high_tmp_541fa_2 = add(carry_high_tmp_541fa_2, carry_low_tmp_541fa_1);
    carry_high_tmp_541fa_2 = sub(carry_high_tmp_541fa_2, triple_sum32_res_limb_1_col1);
    carry_high_tmp_541fa_2 = mul(carry_high_tmp_541fa_2, M31_32768);
    cuda_evaluator->add_constraint(mul(mul(sub(carry_high_tmp_541fa_2, M31_1), carry_high_tmp_541fa_2), sub(carry_high_tmp_541fa_2, M31_2)));
}

extern "C" void evaluate_blake_g(
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




#endif // EVALUATE_BLAKE_G_CONSTRAINT_H