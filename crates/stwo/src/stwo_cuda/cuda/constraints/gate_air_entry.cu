// Standalone extern "C" entry for the gate_air constraint kernel.
//
// The generic libstwo_cuda (stwo-cuda-backend) is circuit-agnostic: its
// `evaluate_constraints.cu` dispatch no longer contains a gate_air arm. This downstream crate
// therefore exposes its own entry point that calls `evaluate_gate_air` directly. It mirrors the
// generic outer FFI (`evaluate_constraint_quotients_on_domain`), which simply delegates to the
// per-component kernel on the default stream (0); `evaluate_gate_air` performs the algebraic eval,
// the LogUp post-kernel, and the finalize accumulation internally.
//
// !!! BOX-UNVALIDATED CUDA !!! Cannot be compiled without nvcc; build + validate on the GPU box.
#include "evaluate_gate_air.cuh"

extern "C" bool evaluate_gate_air_entry(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations, unsigned trace0_evaluations_len,
    m31 **trace1_evaluations, unsigned trace1_evaluations_len,
    m31 **trace2_evaluations, unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers, m31 *denominator_inverses,
    unsigned int domain_log_size, unsigned int eval_domain_log_size,
    unsigned int number_of_columns, unsigned int logup_counts,
    void *eval, qm31 cumsum_shift,
    bool should_accumulate, bool use_assert_evaluator
) {
    evaluate_gate_air(
        quotients_0, quotients_1, quotients_2, quotients_3,
        trace0_evaluations, trace0_evaluations_len,
        trace1_evaluations, trace1_evaluations_len,
        trace2_evaluations, trace2_evaluations_len,
        random_coeff_powers, denominator_inverses,
        domain_log_size, eval_domain_log_size,
        number_of_columns, logup_counts,
        eval, cumsum_shift,
        should_accumulate, use_assert_evaluator,
        0 /* default stream */);
    return true;
}
