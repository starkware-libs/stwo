#ifndef EVALUATE_POSEIDON_CONSTRAINT_H
#define EVALUATE_POSEIDON_CONSTRAINT_H

#include "evaluate_common.cuh"

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"

#define N_LOG_INSTANCES_PER_ROW 0
#define N_INSTANCES_PER_ROW (1 << N_LOG_INSTANCES_PER_ROW)
#define N_STATE 16
#define N_PARTIAL_ROUNDS 14
#define N_HALF_FULL_ROUNDS 4
#define FULL_ROUNDS (2 * N_HALF_FULL_ROUNDS)
#define N_COLUMNS_PER_REP (N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS)
#define N_COLUMNS (N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP)
#define LOG_EXPAND 2
#define LOG_N_LANES 4

HOST_DEVICE_FORCEINLINE void apply_m4(const m31 x[4], m31 y[4]) {
    m31 t0 = add(x[0], x[1]);
    m31 t02 = add(t0, t0);
    m31 t1 = add(x[2], x[3]);
    m31 t12 = add(t1, t1);
    m31 t2 = add(add(x[1], x[1]), t1);
    m31 t3 = add(add(x[3], x[3]), t0);
    m31 t4 = add(add(t12, t12), t3);
    m31 t5 = add(add(t02, t02), t2);
    m31 t6 = add(t3, t5);
    m31 t7 = add(t2, t4);
    y[0] = t6;
    y[1] = t5;
    y[2] = t7;
    y[3] = t4;
}

HOST_DEVICE_FORCEINLINE void apply_external_round_matrix(m31 state[16]) {
    for (int i = 0; i < 4; ++i) {
        m31 block_input[4] = {
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3]
        };
        m31 block_output[4];
        apply_m4(block_input, block_output);
        state[4 * i]     = block_output[0];
        state[4 * i + 1] = block_output[1];
        state[4 * i + 2] = block_output[2];
        state[4 * i + 3] = block_output[3];
    }

    for (int j = 0; j < 4; ++j) {
        m31 s = add(add(state[j], state[j + 4]), add(state[j + 8], state[j + 12]));
        for (int i = 0; i < 4; ++i) {
            state[4 * i + j] = add(state[4 * i + j], s);
        }
    }
}

HOST_DEVICE_FORCEINLINE void apply_internal_round_matrix(m31 state[16]) {
    m31 sum = m31(0);
    for (int i = 0; i < 16; ++i) {
        sum = add(sum, state[i]);
    }

    m31 original_state[16];
    for (int i = 0; i < 16; ++i) {
        original_state[i] = state[i];
    }

    for (int i = 0; i < 16; ++i) {
        unsigned int scalar = 1u << (i + 1);
        m31 scalar_field = m31(scalar);
        m31 multiplied = mul(original_state[i], scalar_field);
        m31 new_val = add(multiplied, sum);
        state[i] = new_val;
    }
}

HOST_DEVICE_FORCEINLINE m31 pow5(m31 x) {
    m31 x2 = mul(x, x);
    m31 x4 = mul(x2, x2);
    return mul(x4, x);
}



struct PoseidonEval {
    unsigned eval_id;
    unsigned log_n_rows;
    LookupElementsBasic<N_STATE> lookup_elements;
    qm31 claim_sum;
};

extern "C"
void evaluate_poseidon_constraint_quotients_on_domain(
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
void generate_poseidon_traces(
    m31 **traces,
    m31 **lookup_init,
    m31 **lookup_final,
    unsigned trace_log_size
);

extern "C"
void generate_poseidon_interaction_traces(
    void *lookup_elements_ptr,
    m31 **lookup_init,
    m31 **lookup_final,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

#endif // EVALUATE_POSEIDON_CONSTRAINT_H
