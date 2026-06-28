#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_jnz_opcode_taken.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_read_small.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_common.cuh"

#define JNZ_OPCODE_TAKEN_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_jnz_opcode_taken_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    JnzOpcodeTaken_Eval *jnz_opcode_taken_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

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

    // Load all 47 trace columns
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset0_col3 = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col4 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col5 = cuda_evaluator.next_trace_mask();
    m31 mem_dst_base_col6 = cuda_evaluator.next_trace_mask();
    m31 dst_id_col7 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_0_col8 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_1_col9 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_2_col10 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_3_col11 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_4_col12 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_5_col13 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_6_col14 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_7_col15 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_8_col16 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_9_col17 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_10_col18 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_11_col19 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_12_col20 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_13_col21 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_14_col22 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_15_col23 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_16_col24 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_17_col25 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_18_col26 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_19_col27 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_20_col28 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_21_col29 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_22_col30 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_23_col31 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_24_col32 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_25_col33 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_26_col34 = cuda_evaluator.next_trace_mask();
    m31 dst_limb_27_col35 = cuda_evaluator.next_trace_mask();
    m31 res_col36 = cuda_evaluator.next_trace_mask();
    m31 res_squares_col37 = cuda_evaluator.next_trace_mask();
    m31 next_pc_id_col38 = cuda_evaluator.next_trace_mask();
    m31 msb_col39 = cuda_evaluator.next_trace_mask();
    m31 mid_limbs_set_col40 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_0_col41 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_1_col42 = cuda_evaluator.next_trace_mask();
    m31 next_pc_limb_2_col43 = cuda_evaluator.next_trace_mask();
    m31 remainder_bits_col44 = cuda_evaluator.next_trace_mask();
    m31 partial_limb_msb_col45 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // Define constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_136 = m31(136);
    const m31 M31_256 = m31(256);

    // DecodeInstructionDe75A
    m31 decode_output[1];
    evaluate_decode_instruction_de75a(
        input_pc_col0,
        offset0_col3,
        dst_base_fp_col4,
        ap_update_add_1_col5,
        decode_output,
        jnz_opcode_taken_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Constraint: mem_dst_base = dst_base_fp * input_fp + (1 - dst_base_fp) * input_ap
    m31 mem_dst_base_expected = add(
        mul(dst_base_fp_col4, input_fp_col2),
        mul(sub(M31_1, dst_base_fp_col4), input_ap_col1)
    );
    cuda_evaluator.add_constraint(sub(mem_dst_base_col6, mem_dst_base_expected));

    // Read dst value from [mem_dst_base + offset0] (ReadPositiveNumBits252)
    m31 read_dst_output[29] = {0};
    evaluate_read_positive_num_bits_252<EvaluatorT>(
        add(mem_dst_base_col6, decode_output[0]),  // Address: mem_dst_base + offset0
        dst_id_col7,
        dst_limb_0_col8,
        dst_limb_1_col9,
        dst_limb_2_col10,
        dst_limb_3_col11,
        dst_limb_4_col12,
        dst_limb_5_col13,
        dst_limb_6_col14,
        dst_limb_7_col15,
        dst_limb_8_col16,
        dst_limb_9_col17,
        dst_limb_10_col18,
        dst_limb_11_col19,
        dst_limb_12_col20,
        dst_limb_13_col21,
        dst_limb_14_col22,
        dst_limb_15_col23,
        dst_limb_16_col24,
        dst_limb_17_col25,
        dst_limb_18_col26,
        dst_limb_19_col27,
        dst_limb_20_col28,
        dst_limb_21_col29,
        dst_limb_22_col30,
        dst_limb_23_col31,
        dst_limb_24_col32,
        dst_limb_25_col33,
        dst_limb_26_col34,
        dst_limb_27_col35,
        read_dst_output,
        jnz_opcode_taken_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // Constraint: dst doesn't equal 0
    // (sum of all 28 limbs) * res_col36 = 1
    m31 dst_sum = dst_limb_0_col8;
    dst_sum = add(dst_sum, dst_limb_1_col9);
    dst_sum = add(dst_sum, dst_limb_2_col10);
    dst_sum = add(dst_sum, dst_limb_3_col11);
    dst_sum = add(dst_sum, dst_limb_4_col12);
    dst_sum = add(dst_sum, dst_limb_5_col13);
    dst_sum = add(dst_sum, dst_limb_6_col14);
    dst_sum = add(dst_sum, dst_limb_7_col15);
    dst_sum = add(dst_sum, dst_limb_8_col16);
    dst_sum = add(dst_sum, dst_limb_9_col17);
    dst_sum = add(dst_sum, dst_limb_10_col18);
    dst_sum = add(dst_sum, dst_limb_11_col19);
    dst_sum = add(dst_sum, dst_limb_12_col20);
    dst_sum = add(dst_sum, dst_limb_13_col21);
    dst_sum = add(dst_sum, dst_limb_14_col22);
    dst_sum = add(dst_sum, dst_limb_15_col23);
    dst_sum = add(dst_sum, dst_limb_16_col24);
    dst_sum = add(dst_sum, dst_limb_17_col25);
    dst_sum = add(dst_sum, dst_limb_18_col26);
    dst_sum = add(dst_sum, dst_limb_19_col27);
    dst_sum = add(dst_sum, dst_limb_20_col28);
    dst_sum = add(dst_sum, dst_limb_21_col29);
    dst_sum = add(dst_sum, dst_limb_22_col30);
    dst_sum = add(dst_sum, dst_limb_23_col31);
    dst_sum = add(dst_sum, dst_limb_24_col32);
    dst_sum = add(dst_sum, dst_limb_25_col33);
    dst_sum = add(dst_sum, dst_limb_26_col34);
    dst_sum = add(dst_sum, dst_limb_27_col35);
    cuda_evaluator.add_constraint(sub(mul(dst_sum, res_col36), M31_1));

    // Constraint: dst doesn't equal P
    // diff_from_p = (dst_limb_0 - 1, dst_limb_1, ..., dst_limb_21 - 136, ..., dst_limb_27 - 256)
    // (diff_from_p[0]^2 + ... + diff_from_p[21]^2 + ... + diff_from_p[27]^2) * res_squares_col37 = 1
    m31 diff_from_p_tmp_0 = sub(dst_limb_0_col8, M31_1);
    m31 diff_from_p_tmp_21 = sub(dst_limb_21_col29, M31_136);
    m31 diff_from_p_tmp_27 = sub(dst_limb_27_col35, M31_256);

    // Sum of squares of differences from P
    m31 diff_sum = mul(diff_from_p_tmp_0, diff_from_p_tmp_0);
    diff_sum = add(diff_sum, dst_limb_1_col9);
    diff_sum = add(diff_sum, dst_limb_2_col10);
    diff_sum = add(diff_sum, dst_limb_3_col11);
    diff_sum = add(diff_sum, dst_limb_4_col12);
    diff_sum = add(diff_sum, dst_limb_5_col13);
    diff_sum = add(diff_sum, dst_limb_6_col14);
    diff_sum = add(diff_sum, dst_limb_7_col15);
    diff_sum = add(diff_sum, dst_limb_8_col16);
    diff_sum = add(diff_sum, dst_limb_9_col17);
    diff_sum = add(diff_sum, dst_limb_10_col18);
    diff_sum = add(diff_sum, dst_limb_11_col19);
    diff_sum = add(diff_sum, dst_limb_12_col20);
    diff_sum = add(diff_sum, dst_limb_13_col21);
    diff_sum = add(diff_sum, dst_limb_14_col22);
    diff_sum = add(diff_sum, dst_limb_15_col23);
    diff_sum = add(diff_sum, dst_limb_16_col24);
    diff_sum = add(diff_sum, dst_limb_17_col25);
    diff_sum = add(diff_sum, dst_limb_18_col26);
    diff_sum = add(diff_sum, dst_limb_19_col27);
    diff_sum = add(diff_sum, dst_limb_20_col28);
    diff_sum = add(diff_sum, mul(diff_from_p_tmp_21, diff_from_p_tmp_21));
    diff_sum = add(diff_sum, dst_limb_22_col30);
    diff_sum = add(diff_sum, dst_limb_23_col31);
    diff_sum = add(diff_sum, dst_limb_24_col32);
    diff_sum = add(diff_sum, dst_limb_25_col33);
    diff_sum = add(diff_sum, dst_limb_26_col34);
    diff_sum = add(diff_sum, mul(diff_from_p_tmp_27, diff_from_p_tmp_27));
    cuda_evaluator.add_constraint(sub(mul(diff_sum, res_squares_col37), M31_1));

    // Read relative offset from [input_pc + 1] (ReadSmall)
    m31 read_small_output[2] = {0};
    evaluate_read_small(
        add(input_pc_col0, M31_1),  // Address: input_pc + 1
        next_pc_id_col38,
        msb_col39,
        mid_limbs_set_col40,
        next_pc_limb_0_col41,
        next_pc_limb_1_col42,
        next_pc_limb_2_col43,
        remainder_bits_col44,
        partial_limb_msb_col45,
        read_small_output,
        jnz_opcode_taken_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // read_small_output[0] contains the jump offset
    // next_pc = input_pc + offset

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Add first opcodes relation entry (positive)
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            input_pc_col0,
            input_ap_col1,
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            jnz_opcode_taken_eval->common_lookup_elements,
            qm31{enabler, 0, 0, 0},  // positive multiplicity
            values
        );
    }

    // Add second opcodes relation entry (negative)
    // JNZ taken: dst != 0, so PC jumps to offset
    // next_pc = input_pc + offset
    // next_ap = input_ap + ap_update_add_1
    // next_fp = input_fp (unchanged)
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            add(input_pc_col0, read_small_output[0]),  // next_pc = input_pc + offset
            add(input_ap_col1, ap_update_add_1_col5),
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            jnz_opcode_taken_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},  // negative multiplicity
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

// Host wrapper function
extern "C"
void evaluate_jnz_opcode_taken(
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

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    JnzOpcodeTaken_Eval *device_jnz_taken_eval = cuda_malloc<JnzOpcodeTaken_Eval>(1);
    cuda_mem_copy_host_to_device<JnzOpcodeTaken_Eval>((JnzOpcodeTaken_Eval*)eval, device_jnz_taken_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_jnz_opcode_taken");

    int block_dim = eval_domain_size < 256 ? eval_domain_size : 256;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_jnz_opcode_taken_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jnz_taken_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_jnz_opcode_taken_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jnz_taken_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    }
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
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
        g_should_accumulate_host  // Read from global variable set by dispatcher
    );

    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.end("evaluate_jnz_opcode_taken");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_jnz_taken_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
