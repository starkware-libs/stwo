/*
============================================
Partial EC Mul CUDA AIR Evaluator
============================================

Component: PartialEcMul
translated from: cairo-air/src/comptogethernts/partial_ec_mul.rs
AIR version: 54d95c0d

Functionality:
- Partial elliptic curve scalar multiplication
- Partial elliptic curve scalar multiplication for Pedersen hash

Data Structure:
- 472 trace columns:
  * input_limb[73]: input data
  * pedersen_points_table_output[56]: lookup table result
  * EC addition intermediate results (sub_res, add_res, div_res, mul_res, etc.)
  * enabler: enable bit

Constraint Logic:
- 1 enabler constraint: enabler^2 = enabler
- 1 PedersenPointsTable lookup (USE side)
- 1 PartialEcMul relation
- EcAdd subroutine call
- 18 types of range check lookups (202 total uses)

Relation Lookups:
- PartialEcMul: 1 use
- PedersenPointsTable (USE): 1 use
- RangeCheck_19 (A-H variants): 75 uses
- RangeCheck_9_9 (A-H variants): 126 uses
============================================
*/

#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_partial_ec_mul.cuh"
#include "evaluate_common.cuh"
#include "ec_add.cuh"

#define PARTIAL_EC_MUL_THREAD_COUNT_MAX 256

// ============================================================================
// PartialEcMul Pre-Kernel: main constraint evaluation
// ============================================================================
template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_partial_ec_mul_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PartialEcMul_Eval *partial_ec_mul_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) {
        return;
    }

    // Evaluator for preprocessed trace (trace0)
    EvaluatorT cuda_evaluator0(
        trace0_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // Evaluator for base trace (trace1)
    EvaluatorT cuda_evaluator1(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        qm31{{0, 0}, {0, 0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // ===================== read Trace column (472 columns) =====================
    // Columns 0-72: input_limb[73]
    m31 input_limb[73];
    for (int i = 0; i < 73; i++) {
        input_limb[i] = cuda_evaluator1.next_trace_mask();
    }

    // Columns 73-128: pedersen_points_table_output[56]
    m31 pedersen_output[56];
    for (int i = 0; i < 56; i++) {
        pedersen_output[i] = cuda_evaluator1.next_trace_mask();
    }

    // Columns 129-156: sub_res_0[28]
    m31 sub_res_0[28];
    for (int i = 0; i < 28; i++) {
        sub_res_0[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 157: sub_p_bit_0
    m31 sub_p_bit_0 = cuda_evaluator1.next_trace_mask();

    // Columns 158-185: add_res_0[28]
    m31 add_res_0[28];
    for (int i = 0; i < 28; i++) {
        add_res_0[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 186: sub_p_bit_1
    m31 sub_p_bit_1 = cuda_evaluator1.next_trace_mask();

    // Columns 187-214: sub_res_1[28]
    m31 sub_res_1[28];
    for (int i = 0; i < 28; i++) {
        sub_res_1[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 215: sub_p_bit_2
    m31 sub_p_bit_2 = cuda_evaluator1.next_trace_mask();

    // Columns 216-243: div_res[28]
    m31 div_res[28];
    for (int i = 0; i < 28; i++) {
        div_res[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 244: k_div
    m31 k_div = cuda_evaluator1.next_trace_mask();

    // Columns 245-271: carry_div[27]
    m31 carry_div[27];
    for (int i = 0; i < 27; i++) {
        carry_div[i] = cuda_evaluator1.next_trace_mask();
    }

    // Columns 272-299: mul_res_0[28]
    m31 mul_res_0[28];
    for (int i = 0; i < 28; i++) {
        mul_res_0[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 300: k_mul_0
    m31 k_mul_0 = cuda_evaluator1.next_trace_mask();

    // Columns 301-327: carry_mul_0[27]
    m31 carry_mul_0[27];
    for (int i = 0; i < 27; i++) {
        carry_mul_0[i] = cuda_evaluator1.next_trace_mask();
    }

    // Columns 328-355: sub_res_2[28]
    m31 sub_res_2[28];
    for (int i = 0; i < 28; i++) {
        sub_res_2[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 356: sub_p_bit_3
    m31 sub_p_bit_3 = cuda_evaluator1.next_trace_mask();

    // Columns 357-384: sub_res_3[28]
    m31 sub_res_3[28];
    for (int i = 0; i < 28; i++) {
        sub_res_3[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 385: sub_p_bit_4
    m31 sub_p_bit_4 = cuda_evaluator1.next_trace_mask();

    // Columns 386-413: mul_res_1[28]
    m31 mul_res_1[28];
    for (int i = 0; i < 28; i++) {
        mul_res_1[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 414: k_mul_1
    m31 k_mul_1 = cuda_evaluator1.next_trace_mask();

    // Columns 415-441: carry_mul_1[27]
    m31 carry_mul_1[27];
    for (int i = 0; i < 27; i++) {
        carry_mul_1[i] = cuda_evaluator1.next_trace_mask();
    }

    // Columns 442-469: sub_res_4[28]
    m31 sub_res_4[28];
    for (int i = 0; i < 28; i++) {
        sub_res_4[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 470: sub_p_bit_5
    m31 sub_p_bit_5 = cuda_evaluator1.next_trace_mask();

    // Column 471: enabler
    m31 enabler = cuda_evaluator1.next_trace_mask();

    // ===================== constraint 1: Enabler Boolean =====================
    // enabler^2 = enabler (ensures enabler is 0 or 1)
    cuda_evaluator1.add_constraint(
        sub(mul(enabler, enabler), enabler)
    );

    // ===================== Relation Lookup: PedersenPointsTable (USE) =====================
    // Construct table index from input limbs
    m31 M31_262144 = m31(262144);
    m31 table_index = add(add(input_limb[2], mul(M31_262144, input_limb[1])), input_limb[3]);

    m31 pedersen_lookup_values[57];
    pedersen_lookup_values[0] = table_index;
    for (int i = 0; i < 56; i++) {
        pedersen_lookup_values[i + 1] = pedersen_output[i];
    }

    // USE side: multiplicity is positive
    {
        m31 pedersen_lookup_values_ext[58];
        pedersen_lookup_values_ext[0] = PEDERSEN_POINTS_TABLE_RELATION_ID;
        for (int i = 0; i < 57; i++) pedersen_lookup_values_ext[i + 1] = pedersen_lookup_values[i];
        qm31 multiplicity = qm31{{1, 0}, {0, 0}};
        cuda_evaluator1.add_to_relation<58>(partial_ec_mul_eval->common_lookup_elements, multiplicity, pedersen_lookup_values_ext);
    }

    // ===================== Call EC Add Subroutine (inlined) =====================
    // Extract EC points from input and pedersen output
    // Point 1: (x1, y1) from input_limb[17..44], input_limb[45..72]
    // Point 2: (x2, y2) from pedersen_output[0..27], pedersen_output[28..55]

    // Call ec_add_evaluate which will verify all intermediate results
    // and add all necessary range check relations
    {
        m31 x1[28], y1[28], x2[28], y2[28];

        // Extract x1, y1 from input_limb
        for (int i = 0; i < 28; i++) {
            x1[i] = input_limb[17 + i];
            y1[i] = input_limb[45 + i];
        }

        // Extract x2, y2 from pedersen_output
        for (int i = 0; i < 28; i++) {
            x2[i] = pedersen_output[i];
            y2[i] = pedersen_output[28 + i];
        }

        // Call EC point addition - this will verify all trace columns
        // and add range check relations
        ec_add_evaluate(
            x1, y1, x2, y2,
            sub_res_0, &sub_p_bit_0,
            add_res_0, &sub_p_bit_1,
            sub_res_1, &sub_p_bit_2,
            div_res, &k_div, carry_div,
            mul_res_0, &k_mul_0, carry_mul_0,
            sub_res_2, &sub_p_bit_3,
            sub_res_3, &sub_p_bit_4,
            mul_res_1, &k_mul_1, carry_mul_1,
            sub_res_4, &sub_p_bit_5,
            partial_ec_mul_eval->common_lookup_elements,
            &cuda_evaluator1
        );
    }

    // ===================== Relation Lookup: PartialEcMul (first entry: positive) =====================
    // Construct full relation entry: RELATION_ID + all 73 input values
    // Multiplicity is +enabler (from Rust line 1167)
    m31 partial_ec_mul_values[74];
    partial_ec_mul_values[0] = PARTIAL_EC_MUL_RELATION_ID;
    for (int i = 0; i < 73; i++) {
        partial_ec_mul_values[i + 1] = input_limb[i];
    }

    {
        qm31 multiplicity = qm31{{enabler, 0}, {0, 0}};
        cuda_evaluator1.add_to_relation<74>(partial_ec_mul_eval->common_lookup_elements, multiplicity, partial_ec_mul_values);
    }

    // ===================== Relation Lookup: PartialEcMul (second entry: negative) =====================
    // From Rust lines 1245-1323: second entry with -enabler multiplicity and output values
    // Values mapping (from Rust code):
    // [0]: RELATION_ID
    // [1]: input_limb_0
    // [2]: input_limb_1 + 1
    // [3]: input_limb_2
    // [4..17]: input_limb_4..input_limb_16 (shifted, skipping input_limb_3)
    // [17]: 0
    // [18..45]: sub_res_2 (x3 result)
    // [46..73]: sub_res_4 (y3 result)
    m31 partial_ec_mul_values_neg[74];
    partial_ec_mul_values_neg[0] = PARTIAL_EC_MUL_RELATION_ID;
    partial_ec_mul_values_neg[1] = input_limb[0];
    partial_ec_mul_values_neg[2] = add(input_limb[1], m31(1));
    partial_ec_mul_values_neg[3] = input_limb[2];
    // Shift: [4..17] gets input_limb[4..17]
    for (int i = 4; i < 18; i++) {
        partial_ec_mul_values_neg[i] = input_limb[i];
    }
    partial_ec_mul_values_neg[17] = m31(0);  // M31_0 at position 17
    // [18..45]: x3 result (sub_res_2)
    for (int i = 0; i < 28; i++) {
        partial_ec_mul_values_neg[18 + i] = sub_res_2[i];
    }
    // [46..73]: y3 result (sub_res_4)
    for (int i = 0; i < 28; i++) {
        partial_ec_mul_values_neg[46 + i] = sub_res_4[i];
    }

    {
        // Negative multiplicity: -enabler
        qm31 neg_multiplicity = qm31{{neg(enabler), 0}, {0, 0}};
        cuda_evaluator1.add_to_relation<74>(partial_ec_mul_eval->common_lookup_elements, neg_multiplicity, partial_ec_mul_values_neg);
    }

    // ===================== Complete constraint evaluation =====================
    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

// ============================================================================
// PartialEcMul main function: orchestrates the entire evaluation process
// ============================================================================

extern "C"
void evaluate_partial_ec_mul(
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
) {
    (void)number_of_columns;

    PartialEcMul_Eval *partial_ec_mul_eval =
        (PartialEcMul_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    PartialEcMul_Eval *device_partial_ec_mul_eval =
        cuda_malloc<PartialEcMul_Eval>(1);
    cuda_mem_copy_host_to_device<PartialEcMul_Eval>(
        partial_ec_mul_eval, device_partial_ec_mul_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_partial_ec_mul");

    int block_dim = eval_domain_size < PARTIAL_EC_MUL_THREAD_COUNT_MAX
        ? eval_domain_size
        : PARTIAL_EC_MUL_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_partial_ec_mul_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_partial_ec_mul_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_partial_ec_mul_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_partial_ec_mul_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (unsigned i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = logup_counts ? batching[logup_counts - 1] : 0;

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    } else {
        generic_constraint_post_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            d_intermediate_fractions,
            constraint_index_array,
            device_trace2_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            logup_counts,
            last_batch,
            cumsum_shift
        );
    }

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generic_constraint_quotients_finalize_kernel<<<num_blocks, block_dim, 0, stream>>>(
        quotients_0,
        quotients_1,
        quotients_2,
        quotients_3,
        numerators,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        should_accumulate
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_partial_ec_mul");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_partial_ec_mul_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
