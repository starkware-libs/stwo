/*
============================================
PartialEcMulWindowBits18 CUDA Evaluator
============================================

Component: PartialEcMulWindowBits18 (297 trace columns, 0 preprocessed columns)
Translated from: cairo-air/src/components/partial_ec_mul_window_bits_18.rs

Flow:
1. Read 297 trace columns (input_limb[72], ped_pts_table_output[56],
   slope[28], k1, carry1[27], result_x[28], k2, carry2[27],
   result_y[28], k3, carry3[27], enabler)
2. enabler constraint: enabler^2 - enabler = 0
3. PedersenPointsTableWindowBits18 lookup (58 values)
4. EcAdd subroutine (inlined as new-style ec_add):
   a. RangeCheckMemValueN28(slope)       -- 14 lookups
   b. VerifyMul252(slope, x2-x1, y2-y1)  -- 28 lookups + 27 constraints
   c. RangeCheckMemValueN28(result_x)     -- 14 lookups
   d. VerifyMul252(slope, slope, x1+x2+result_x) -- 28 lookups + 27 constraints
   e. RangeCheckMemValueN28(result_y)     -- 14 lookups
   f. VerifyMul252(slope, x1-result_x, y1+result_y) -- 28 lookups + 27 constraints
5. PartialEcMulWindowBits18 positive relation (+enabler multiplicity)
6. PartialEcMulWindowBits18 negative relation (-enabler multiplicity)

Total: 129 add_to_relation calls -> 65 logup interaction columns (finalized in pairs)
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
#include "evaluate_partial_ec_mul_window_bits_18.cuh"
#include "evaluate_common.cuh"
#include "relations.cuh"
#include "verify_mul_252.cuh"

#define PARTIAL_EC_MUL_WB18_THREAD_COUNT_MAX 256

// =====================================================================
// Inline: RangeCheckMemValueN28
// 14 RangeCheck_9_9 lookups cycling through A-H variants for 28 limbs.
// Cycle: 9_9, 9_9_B, 9_9_C, 9_9_D, 9_9_E, 9_9_F, 9_9_G, 9_9_H, then repeats
// =====================================================================
template<typename EvaluatorT>
DEVICE_FORCEINLINE void range_check_mem_value_n_28_evaluate(
    const m31 limbs[28],
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    const qm31 ONE = qm31{{1, 0}, {0, 0}};
    // Lookup 0: RangeCheck_9_9 (limbs[0], limbs[1])
    { m31 v[3] = {RANGE_CHECK_9_9_RELATION_ID, limbs[0], limbs[1]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 1: RangeCheck_9_9_B (limbs[2], limbs[3])
    { m31 v[3] = {RANGE_CHECK_9_9_B_RELATION_ID, limbs[2], limbs[3]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 2: RangeCheck_9_9_C (limbs[4], limbs[5])
    { m31 v[3] = {RANGE_CHECK_9_9_C_RELATION_ID, limbs[4], limbs[5]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 3: RangeCheck_9_9_D (limbs[6], limbs[7])
    { m31 v[3] = {RANGE_CHECK_9_9_D_RELATION_ID, limbs[6], limbs[7]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 4: RangeCheck_9_9_E (limbs[8], limbs[9])
    { m31 v[3] = {RANGE_CHECK_9_9_E_RELATION_ID, limbs[8], limbs[9]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 5: RangeCheck_9_9_F (limbs[10], limbs[11])
    { m31 v[3] = {RANGE_CHECK_9_9_F_RELATION_ID, limbs[10], limbs[11]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 6: RangeCheck_9_9_G (limbs[12], limbs[13])
    { m31 v[3] = {RANGE_CHECK_9_9_G_RELATION_ID, limbs[12], limbs[13]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 7: RangeCheck_9_9_H (limbs[14], limbs[15])
    { m31 v[3] = {RANGE_CHECK_9_9_H_RELATION_ID, limbs[14], limbs[15]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 8: RangeCheck_9_9 (limbs[16], limbs[17])
    { m31 v[3] = {RANGE_CHECK_9_9_RELATION_ID, limbs[16], limbs[17]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 9: RangeCheck_9_9_B (limbs[18], limbs[19])
    { m31 v[3] = {RANGE_CHECK_9_9_B_RELATION_ID, limbs[18], limbs[19]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 10: RangeCheck_9_9_C (limbs[20], limbs[21])
    { m31 v[3] = {RANGE_CHECK_9_9_C_RELATION_ID, limbs[20], limbs[21]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 11: RangeCheck_9_9_D (limbs[22], limbs[23])
    { m31 v[3] = {RANGE_CHECK_9_9_D_RELATION_ID, limbs[22], limbs[23]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 12: RangeCheck_9_9_E (limbs[24], limbs[25])
    { m31 v[3] = {RANGE_CHECK_9_9_E_RELATION_ID, limbs[24], limbs[25]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
    // Lookup 13: RangeCheck_9_9_F (limbs[26], limbs[27])
    { m31 v[3] = {RANGE_CHECK_9_9_F_RELATION_ID, limbs[26], limbs[27]};
      cuda_evaluator->template add_to_relation<3>(common_lookup_elements, ONE, v); }
}

// =====================================================================
// Pre-Kernel: Read trace columns, evaluate constraints, build logup fractions
// =====================================================================
template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_partial_ec_mul_window_bits_18_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PartialEcMulWindowBits18_Eval *comp_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    // Constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_262144 = m31(262144);

    // Evaluator for base trace (trace1) -- reads 297 trace columns + builds logup
    EvaluatorT cuda_evaluator(
        trace1_evaluations,
        random_coeff_powers,
        0,
        row,
        {0},
        0,
        {0},
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );

    // ===================== Read all 297 trace columns =====================

    // input_limb[0..71] -- cols 0-71
    m31 input_limb[72];
    for (int i = 0; i < 72; i++) {
        input_limb[i] = cuda_evaluator.next_trace_mask();
    }

    // pedersen_points_table output[0..55] -- cols 72-127
    m31 ped_out[56];
    for (int i = 0; i < 56; i++) {
        ped_out[i] = cuda_evaluator.next_trace_mask();
    }

    // slope[0..27] -- cols 128-155
    m31 slope[28];
    for (int i = 0; i < 28; i++) {
        slope[i] = cuda_evaluator.next_trace_mask();
    }

    // VerifyMul252 #1: k_col156, carry[0..26] cols 157-183
    m31 k1 = cuda_evaluator.next_trace_mask();
    m31 carry1[27];
    for (int i = 0; i < 27; i++) {
        carry1[i] = cuda_evaluator.next_trace_mask();
    }

    // result_x[0..27] -- cols 184-211
    m31 result_x[28];
    for (int i = 0; i < 28; i++) {
        result_x[i] = cuda_evaluator.next_trace_mask();
    }

    // VerifyMul252 #2: k_col212, carry[0..26] cols 213-239
    m31 k2 = cuda_evaluator.next_trace_mask();
    m31 carry2[27];
    for (int i = 0; i < 27; i++) {
        carry2[i] = cuda_evaluator.next_trace_mask();
    }

    // result_y[0..27] -- cols 240-267
    m31 result_y[28];
    for (int i = 0; i < 28; i++) {
        result_y[i] = cuda_evaluator.next_trace_mask();
    }

    // VerifyMul252 #3: k_col268, carry[0..26] cols 269-295
    m31 k3 = cuda_evaluator.next_trace_mask();
    m31 carry3[27];
    for (int i = 0; i < 27; i++) {
        carry3[i] = cuda_evaluator.next_trace_mask();
    }

    // enabler -- col 296
    m31 enabler = cuda_evaluator.next_trace_mask();

    // ===================== 1. PedersenPointsTableWindowBits18 lookup =====================
    // 58 values: [RELATION_ID, (M31_262144 * input_limb_1 + input_limb_2), ped_out[0..55]]
    {
        m31 values[58];
        values[0] = PEDERSEN_POINTS_TABLE_WINDOW_BITS_18_RELATION_ID;
        values[1] = add(mul(M31_262144, input_limb[1]), input_limb[2]);
        for (int i = 0; i < 56; i++) {
            values[2 + i] = ped_out[i];
        }
        cuda_evaluator.template add_to_relation<58>(
            comp_eval->common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // ===================== 2. EcAdd subroutine (new-style, inlined) =====================
    // Inputs from the component:
    //   x1 = input_limb[16..43]  (28 limbs)
    //   y1 = input_limb[44..71]  (28 limbs)
    //   x2 = ped_out[0..27]      (28 limbs)
    //   y2 = ped_out[28..55]     (28 limbs)

    // Build the ec_add input arrays (convenience aliases)
    m31 x1[28], y1[28], x2[28], y2[28];
    for (int i = 0; i < 28; i++) {
        x1[i] = input_limb[16 + i];
        y1[i] = input_limb[44 + i];
        x2[i] = ped_out[i];
        y2[i] = ped_out[28 + i];
    }

    // --- 2a. RangeCheckMemValueN28(slope) : 14 lookups ---
    range_check_mem_value_n_28_evaluate(slope, comp_eval->common_lookup_elements, &cuda_evaluator);

    // --- 2b. VerifyMul252 #1: slope * (x2-x1) = (y2-y1) ---
    // input_a = slope, input_b = (x2 - x1), input_c = (y2 - y1)
    {
        m31 diff_x[28], diff_y[28];
        for (int i = 0; i < 28; i++) {
            diff_x[i] = sub(x2[i], x1[i]);
            diff_y[i] = sub(y2[i], y1[i]);
        }
        verify_mul_252_evaluate(slope, diff_x, diff_y, k1, carry1,
            comp_eval->common_lookup_elements, &cuda_evaluator);
    }

    // --- 2c. RangeCheckMemValueN28(result_x) : 14 lookups ---
    range_check_mem_value_n_28_evaluate(result_x, comp_eval->common_lookup_elements, &cuda_evaluator);

    // --- 2d. VerifyMul252 #2: slope * slope = (x1 + x2 + result_x) ---
    {
        m31 sum_x_plus_rx[28];
        for (int i = 0; i < 28; i++) {
            sum_x_plus_rx[i] = add(add(x1[i], x2[i]), result_x[i]);
        }
        verify_mul_252_evaluate(slope, slope, sum_x_plus_rx, k2, carry2,
            comp_eval->common_lookup_elements, &cuda_evaluator);
    }

    // --- 2e. RangeCheckMemValueN28(result_y) : 14 lookups ---
    range_check_mem_value_n_28_evaluate(result_y, comp_eval->common_lookup_elements, &cuda_evaluator);

    // --- 2f. VerifyMul252 #3: slope * (x1 - result_x) = (y1 + result_y) ---
    {
        m31 x1_minus_rx[28], y1_plus_ry[28];
        for (int i = 0; i < 28; i++) {
            x1_minus_rx[i] = sub(x1[i], result_x[i]);
            y1_plus_ry[i] = add(y1[i], result_y[i]);
        }
        verify_mul_252_evaluate(slope, x1_minus_rx, y1_plus_ry, k3, carry3,
            comp_eval->common_lookup_elements, &cuda_evaluator);
    }

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // ===================== 3. PartialEcMulWindowBits18 positive (+enabler) =====================
    // 73 values: [RELATION_ID, input_limb_0..input_limb_71]
    {
        m31 values[73];
        values[0] = PARTIAL_EC_MUL_WINDOW_BITS_18_RELATION_ID;
        for (int i = 0; i < 72; i++) {
            values[1 + i] = input_limb[i];
        }
        qm31 multiplicity_ext = {{enabler, 0}, {0, 0}};
        cuda_evaluator.template add_to_relation<73>(
            comp_eval->common_lookup_elements, multiplicity_ext, values);
    }

    // ===================== 4. PartialEcMulWindowBits18 negative (-enabler) =====================
    // 73 values: [RELATION_ID, input_limb_0, (input_limb_1 + 1),
    //             input_limb_3..input_limb_15, M31_0,
    //             result_x[0..27], result_y[0..27]]
    {
        m31 values[73];
        values[0] = PARTIAL_EC_MUL_WINDOW_BITS_18_RELATION_ID;
        values[1] = input_limb[0];
        values[2] = add(input_limb[1], M31_1);
        // input_limb_3 through input_limb_15 (13 values)
        for (int i = 3; i <= 15; i++) {
            values[i] = input_limb[i];
        }
        values[16] = M31_0;
        for (int i = 0; i < 28; i++) {
            values[17 + i] = result_x[i];
        }
        for (int i = 0; i < 28; i++) {
            values[45 + i] = result_y[i];
        }

        m31 neg_enabler = neg(enabler);
        qm31 multiplicity_ext = {{neg_enabler, 0}, {0, 0}};
        cuda_evaluator.template add_to_relation<73>(
            comp_eval->common_lookup_elements, multiplicity_ext, values);
    }

    // Store results
    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// =====================================================================
// Host Wrapper Function
// =====================================================================
extern "C"
void evaluate_partial_ec_mul_window_bits_18(
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
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    PartialEcMulWindowBits18_Eval *device_comp_eval = cuda_malloc<PartialEcMulWindowBits18_Eval>(1);
    cuda_mem_copy_host_to_device<PartialEcMulWindowBits18_Eval>(
        (PartialEcMulWindowBits18_Eval*)eval, device_comp_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);

    int block_dim = eval_domain_size < PARTIAL_EC_MUL_WB18_THREAD_COUNT_MAX
                    ? eval_domain_size : PARTIAL_EC_MUL_WB18_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_partial_ec_mul_window_bits_18_pre_kernel<CudaAssertEvaluator>
            <<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_comp_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_partial_ec_mul_window_bits_18_pre_kernel<CudaEvaluator>
            <<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_comp_eval,
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
    unsigned last_batch = batching[logup_counts - 1];

    if (use_assert_evaluator) {
        generic_constraint_post_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        generic_constraint_post_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
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
        g_should_accumulate_host
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_comp_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
