#ifndef EVALUATE_WIDE_FIBONACCI_H
#define EVALUATE_WIDE_FIBONACCI_H

#include "fields.cuh"
#include "utils.cuh"

struct WideFibEval {
    unsigned eval_id;
    unsigned log_n_rows;
};

extern "C"
void evaluate_wide_fibonacci_constraint_quotients_on_domain(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts
);

extern "C"
void generate_wide_fibonacci_trace(
    m31 *input_a,
    m31 *input_b,
    unsigned input_len,
    m31 **traces,
    unsigned trace_len,
    unsigned n_columns
);

#endif // WIDE_FIBONACCI_H