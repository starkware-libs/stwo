#include "evaluate_common.cuh"
#include "eval_at_row.cuh"
#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"

// Note: generic_constraint_post_kernel template implementation is now in evaluate_common.cuh
// This is required for CUDA templates to be visible across compilation units

// Generic finalize kernel - used by all evaluation functions
__launch_bounds__(256, 2)
__global__ void generic_constraint_quotients_finalize_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    qm31 *numerators,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    bool should_accumulate  // NEW: support accumulation mode
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    const m31 denom_inv = denominator_inverses[row >> domain_log_size];
    const qm31 row_numer = numerators[row];
    const qm31 constraint_quotient = mul(denom_inv, row_numer);

    if (should_accumulate) {
        // Accumulate: add to existing values
        quotients_0[row] = add(quotients_0[row], constraint_quotient.a.a);
        quotients_1[row] = add(quotients_1[row], constraint_quotient.a.b);
        quotients_2[row] = add(quotients_2[row], constraint_quotient.b.a);
        quotients_3[row] = add(quotients_3[row], constraint_quotient.b.b);
    } else {
        // First write: direct assignment
        quotients_0[row] = constraint_quotient.a.a;
        quotients_1[row] = constraint_quotient.a.b;
        quotients_2[row] = constraint_quotient.b.a;
        quotients_3[row] = constraint_quotient.b.b;
    }
}
