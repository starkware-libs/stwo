#ifndef GENERIC_KERNELS_CUH
#define GENERIC_KERNELS_CUH

#include "fields.cuh"
#include "eval_at_row.cuh"
#include "utils.cuh"

// Global variable for accumulation strategy (defined in evaluate_constraints.cu)
extern bool g_should_accumulate_host;

struct Claim {
    unsigned int log_size;
};

struct CarioMRelation {
    LookupElementsBasic<2> registers_lookup_elements;
    LookupElementsBasic<6> memory_lookup_elements;
    LookupElementsBasic<4> merkle_lookup_elements;
    LookupElementsBasic<16> poseidon2_lookup_elements;
    LookupElementsBasic<1> range_check_8_lookup_elements;
    LookupElementsBasic<1> range_check_16_lookup_elements;
    LookupElementsBasic<1> range_check_20_lookup_elements;
    LookupElementsBasic<4> bitwise_lookup_elements;
};

/**
 * Generic finalize kernel for constraint quotients
 * This is used by almost all evaluation functions to compute final quotients
 */
__launch_bounds__(256, 2)
__global__ void generic_constraint_quotients_finalize_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    qm31 *numerators,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    bool should_accumulate  // NEW: support accumulation mode
);

/**
 * Generic post kernel - handles both range check and opcode style constraints
 * Handles Fraction batching and logup constraint processing
 * Templated to support both CudaEvaluator and CudaAssertEvaluator
 */
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void generic_constraint_post_kernel(
    qm31 *numerators,
    Fraction *intermediate_fractions,
    unsigned *constraint_index_array,
    m31 **trace2_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int logup_counts,
    unsigned int last_batch,
    qm31 cumsum_shift
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    EvaluatorT evaluator(
        trace2_evaluations,
        random_coeff_powers,
        constraint_index_array[row],
        row,
        numerators[row],
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    const unsigned logup_interaction = 2;
    qm31 prev_col_cumsum = { {0, 0}, {0, 0} };

    // Process complete batches
    for (unsigned i = 0; i < last_batch; ++i) {
        const Fraction cur_frac = Fraction::sum(&intermediate_fractions[2 * i + row * logup_counts], 2);

        qm31 cur_cumsum_arr[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
        int offsets[2] = { 0, 0 };
        evaluator.next_extension_interaction_mask(logup_interaction, offsets, 1, cur_cumsum_arr);

        const qm31 cur_cumsum = cur_cumsum_arr[0];
        const qm31 diff = sub(cur_cumsum, prev_col_cumsum);
        prev_col_cumsum = cur_cumsum;

        const qm31 constraint_val = sub(mul(diff, cur_frac.denominator), cur_frac.numerator);
        evaluator.add_constraint_ext(constraint_val);
    }

    // Process remaining fractions
    {
        unsigned remaining_fractions = logup_counts - last_batch * 2;
        const Fraction frac_sum = Fraction::sum(&intermediate_fractions[last_batch * 2 + row * logup_counts], remaining_fractions);

        int offsets2[2] = { 0, -1 };
        qm31 cumsum2[2] = { { {0, 0}, {0, 0} }, { {0, 0}, {0, 0} } };
        evaluator.next_extension_interaction_mask(logup_interaction, offsets2, 2, cumsum2);

        const qm31 prev_row_cumsum = cumsum2[1];
        const qm31 cur_cumsum = cumsum2[0];
        const qm31 diff = sub(sub(cur_cumsum, prev_row_cumsum), prev_col_cumsum);
        const qm31 fixed_diff = add(diff, cumsum_shift);

        const qm31 constraint_val = sub(mul(fixed_diff, frac_sum.denominator), frac_sum.numerator);

        evaluator.add_constraint_ext(constraint_val);
    }

    numerators[row] = evaluator.row_res;

}

#endif // GENERIC_KERNELS_CUH