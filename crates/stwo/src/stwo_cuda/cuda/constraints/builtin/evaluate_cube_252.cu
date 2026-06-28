#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_cube_252.cuh"
#include "evaluate_common.cuh"
#include "builtin/felt_252_unpack_from_27_range_check_output.cuh"
#include "builtin/mul_252.cuh"

#define CUBE_252_THREAD_COUNT_MAX 256

template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_cube_252_pre_kernel(
    qm31 *numerators,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    Cube252_Eval *cube_eval,
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

    // Constants
    const m31 M31_512 = m31(512);
    const m31 M31_262144 = m31(262144);

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

    // Read 141 trace columns
    // Columns 0-9: input limbs (10 27-bit limbs)
    m31 input[10];
    input[0] = cuda_evaluator1.next_trace_mask();
    input[1] = cuda_evaluator1.next_trace_mask();
    input[2] = cuda_evaluator1.next_trace_mask();
    input[3] = cuda_evaluator1.next_trace_mask();
    input[4] = cuda_evaluator1.next_trace_mask();
    input[5] = cuda_evaluator1.next_trace_mask();
    input[6] = cuda_evaluator1.next_trace_mask();
    input[7] = cuda_evaluator1.next_trace_mask();
    input[8] = cuda_evaluator1.next_trace_mask();
    input[9] = cuda_evaluator1.next_trace_mask();

    // Columns 10-27: unpacked limbs (18 columns for positions 0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25)
    m31 unpacked_limb_0 = cuda_evaluator1.next_trace_mask();   // col 10
    m31 unpacked_limb_1 = cuda_evaluator1.next_trace_mask();   // col 11
    m31 unpacked_limb_3 = cuda_evaluator1.next_trace_mask();   // col 12
    m31 unpacked_limb_4 = cuda_evaluator1.next_trace_mask();   // col 13
    m31 unpacked_limb_6 = cuda_evaluator1.next_trace_mask();   // col 14
    m31 unpacked_limb_7 = cuda_evaluator1.next_trace_mask();   // col 15
    m31 unpacked_limb_9 = cuda_evaluator1.next_trace_mask();   // col 16
    m31 unpacked_limb_10 = cuda_evaluator1.next_trace_mask();  // col 17
    m31 unpacked_limb_12 = cuda_evaluator1.next_trace_mask();  // col 18
    m31 unpacked_limb_13 = cuda_evaluator1.next_trace_mask();  // col 19
    m31 unpacked_limb_15 = cuda_evaluator1.next_trace_mask();  // col 20
    m31 unpacked_limb_16 = cuda_evaluator1.next_trace_mask();  // col 21
    m31 unpacked_limb_18 = cuda_evaluator1.next_trace_mask();  // col 22
    m31 unpacked_limb_19 = cuda_evaluator1.next_trace_mask();  // col 23
    m31 unpacked_limb_21 = cuda_evaluator1.next_trace_mask();  // col 24
    m31 unpacked_limb_22 = cuda_evaluator1.next_trace_mask();  // col 25
    m31 unpacked_limb_24 = cuda_evaluator1.next_trace_mask();  // col 26
    m31 unpacked_limb_25 = cuda_evaluator1.next_trace_mask();  // col 27

    // Columns 28-55: first mul result (28 limbs)
    m31 mul_res1[28];
    for (int i = 0; i < 28; i++) {
        mul_res1[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 56: k for first mul
    m31 k1 = cuda_evaluator1.next_trace_mask();

    // Columns 57-83: carry for first mul (27 values)
    m31 carry1[27];
    for (int i = 0; i < 27; i++) {
        carry1[i] = cuda_evaluator1.next_trace_mask();
    }

    // Columns 84-111: second mul result (28 limbs)
    m31 mul_res2[28];
    for (int i = 0; i < 28; i++) {
        mul_res2[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 112: k for second mul
    m31 k2 = cuda_evaluator1.next_trace_mask();

    // Columns 113-139: carry for second mul (27 values)
    m31 carry2[27];
    for (int i = 0; i < 27; i++) {
        carry2[i] = cuda_evaluator1.next_trace_mask();
    }

    // Column 140: enabler
    m31 enabler = cuda_evaluator1.next_trace_mask();

    // Step 1: Felt252UnpackFrom27RangeCheckOutput
    // This computes the derived limbs and range checks all 28 limbs
    m31 computed_limbs[10];  // Output: limbs 2,5,8,11,14,17,20,23,26,27
    felt_252_unpack_from_27_range_check_output_evaluate(
        input,
        unpacked_limb_0, unpacked_limb_1,
        unpacked_limb_3, unpacked_limb_4,
        unpacked_limb_6, unpacked_limb_7,
        unpacked_limb_9, unpacked_limb_10,
        unpacked_limb_12, unpacked_limb_13,
        unpacked_limb_15, unpacked_limb_16,
        unpacked_limb_18, unpacked_limb_19,
        unpacked_limb_21, unpacked_limb_22,
        unpacked_limb_24, unpacked_limb_25,
        cube_eval->common_lookup_elements,
        computed_limbs,
        &cuda_evaluator1
    );

    // Build the full 28-limb unpacked value for multiplication
    m31 x[28];
    x[0] = unpacked_limb_0;
    x[1] = unpacked_limb_1;
    x[2] = computed_limbs[0];   // limb 2
    x[3] = unpacked_limb_3;
    x[4] = unpacked_limb_4;
    x[5] = computed_limbs[1];   // limb 5
    x[6] = unpacked_limb_6;
    x[7] = unpacked_limb_7;
    x[8] = computed_limbs[2];   // limb 8
    x[9] = unpacked_limb_9;
    x[10] = unpacked_limb_10;
    x[11] = computed_limbs[3];  // limb 11
    x[12] = unpacked_limb_12;
    x[13] = unpacked_limb_13;
    x[14] = computed_limbs[4];  // limb 14
    x[15] = unpacked_limb_15;
    x[16] = unpacked_limb_16;
    x[17] = computed_limbs[5];  // limb 17
    x[18] = unpacked_limb_18;
    x[19] = unpacked_limb_19;
    x[20] = computed_limbs[6];  // limb 20
    x[21] = unpacked_limb_21;
    x[22] = unpacked_limb_22;
    x[23] = computed_limbs[7];  // limb 23
    x[24] = unpacked_limb_24;
    x[25] = unpacked_limb_25;
    x[26] = computed_limbs[8];  // limb 26
    x[27] = computed_limbs[9];  // limb 27 (same as input[9])

    // Step 2: First Mul252 - compute x * x = x^2
    // Note: The Rust code passes [x..., x...] as 56-limb input (a and b are the same)
    mul_252_evaluate(
        x,          // input_a = x
        x,          // input_b = x
        mul_res1,   // result = x^2
        k1,
        carry1,
        cube_eval->common_lookup_elements,
        &cuda_evaluator1
    );

    // Step 3: Second Mul252 - compute x * x^2 = x^3
    mul_252_evaluate(
        x,          // input_a = x
        mul_res1,   // input_b = x^2
        mul_res2,   // result = x^3
        k2,
        carry2,
        cube_eval->common_lookup_elements,
        &cuda_evaluator1
    );

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator1.add_constraint(sub(mul(enabler, enabler), enabler));

    // Step 4: Add Cube252 relation entry with -enabler multiplicity
    // The relation has 20 values:
    // - 10 input limbs
    // - 10 packed output limbs (each packed from 3 consecutive result limbs)
    {
        m31 values[21];

        // RELATION_ID first
        values[0] = CUBE_252_RELATION_ID;

        // Next 10: input limbs
        values[1] = input[0];
        values[2] = input[1];
        values[3] = input[2];
        values[4] = input[3];
        values[5] = input[4];
        values[6] = input[5];
        values[7] = input[6];
        values[8] = input[7];
        values[9] = input[8];
        values[10] = input[9];

        // Next 10: packed output limbs from mul_res2
        // Each is: limb[3i] + limb[3i+1]*512 + limb[3i+2]*262144
        // For last one (index 9), it's just limb[27]
        values[11] = add(add(mul_res2[0], mul(mul_res2[1], M31_512)), mul(mul_res2[2], M31_262144));
        values[12] = add(add(mul_res2[3], mul(mul_res2[4], M31_512)), mul(mul_res2[5], M31_262144));
        values[13] = add(add(mul_res2[6], mul(mul_res2[7], M31_512)), mul(mul_res2[8], M31_262144));
        values[14] = add(add(mul_res2[9], mul(mul_res2[10], M31_512)), mul(mul_res2[11], M31_262144));
        values[15] = add(add(mul_res2[12], mul(mul_res2[13], M31_512)), mul(mul_res2[14], M31_262144));
        values[16] = add(add(mul_res2[15], mul(mul_res2[16], M31_512)), mul(mul_res2[17], M31_262144));
        values[17] = add(add(mul_res2[18], mul(mul_res2[19], M31_512)), mul(mul_res2[20], M31_262144));
        values[18] = add(add(mul_res2[21], mul(mul_res2[22], M31_512)), mul(mul_res2[23], M31_262144));
        values[19] = add(add(mul_res2[24], mul(mul_res2[25], M31_512)), mul(mul_res2[26], M31_262144));
        values[20] = mul_res2[27];

        // Multiplicity is -enabler
        qm31 neg_enabler = qm31{{neg(enabler), 0}, {0, 0}};

        cuda_evaluator1.template add_to_relation<21>(cube_eval->common_lookup_elements, neg_enabler, values);
    }

    constraint_index_array[row] = cuda_evaluator1.constraint_index;
    numerators[row] = cuda_evaluator1.row_res;
}

extern "C"
void evaluate_cube_252(
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

    Cube252_Eval *cube_eval =
        (Cube252_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace0_evaluations =
        clone_to_device<m31 *>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    Cube252_Eval *device_cube_eval =
        cuda_malloc<Cube252_Eval>(1);
    cuda_mem_copy_host_to_device<Cube252_Eval>(
        cube_eval, device_cube_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_cube_252");

    int block_dim = eval_domain_size < CUBE_252_THREAD_COUNT_MAX
        ? eval_domain_size
        : CUBE_252_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_cube_252_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_cube_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_cube_252_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace0_evaluations,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            trace0_evaluations_len + trace1_evaluations_len,
            device_cube_eval,
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
    global_timer.end("evaluate_cube_252");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_cube_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
