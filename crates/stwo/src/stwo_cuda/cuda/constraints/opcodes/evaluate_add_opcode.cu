#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_add_opcode.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_verify_add_252.cuh"
#include "evaluate_common.cuh"

#define ADD_CODE_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_add_opcode_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    AddCode_Eval *add_opcode_eval,
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

    // 2. next_trace_mask
    m31 input_pc_col0        = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1        = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2        = cuda_evaluator.next_trace_mask();
    m31 offset0_col3         = cuda_evaluator.next_trace_mask();
    m31 offset1_col4         = cuda_evaluator.next_trace_mask();
    m31 offset2_col5         = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col6     = cuda_evaluator.next_trace_mask();
    m31 op0_base_fp_col7     = cuda_evaluator.next_trace_mask();
    m31 op1_imm_col8         = cuda_evaluator.next_trace_mask();
    m31 op1_base_fp_col9     = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col10= cuda_evaluator.next_trace_mask();
    m31 mem_dst_base_col11   = cuda_evaluator.next_trace_mask();
    m31 mem0_base_col12      = cuda_evaluator.next_trace_mask();
    m31 mem1_base_col13      = cuda_evaluator.next_trace_mask();
    m31 dst_id_col14         = cuda_evaluator.next_trace_mask();
    m31 dst_limb_0_col15     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_1_col16     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_2_col17     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_3_col18     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_4_col19     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_5_col20     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_6_col21     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_7_col22     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_8_col23     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_9_col24     = cuda_evaluator.next_trace_mask();
    m31 dst_limb_10_col25    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_11_col26    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_12_col27    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_13_col28    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_14_col29    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_15_col30    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_16_col31    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_17_col32    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_18_col33    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_19_col34    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_20_col35    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_21_col36    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_22_col37    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_23_col38    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_24_col39    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_25_col40    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_26_col41    = cuda_evaluator.next_trace_mask();
    m31 dst_limb_27_col42    = cuda_evaluator.next_trace_mask();
    m31 op0_id_col43         = cuda_evaluator.next_trace_mask();
    m31 op0_limb_0_col44     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_1_col45     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_2_col46     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_3_col47     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_4_col48     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_5_col49     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_6_col50     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_7_col51     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_8_col52     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_9_col53     = cuda_evaluator.next_trace_mask();
    m31 op0_limb_10_col54    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_11_col55    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_12_col56    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_13_col57    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_14_col58    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_15_col59    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_16_col60    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_17_col61    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_18_col62    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_19_col63    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_20_col64    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_21_col65    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_22_col66    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_23_col67    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_24_col68    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_25_col69    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_26_col70    = cuda_evaluator.next_trace_mask();
    m31 op0_limb_27_col71    = cuda_evaluator.next_trace_mask();
    m31 op1_id_col72         = cuda_evaluator.next_trace_mask();
    m31 op1_limb_0_col73     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_1_col74     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_2_col75     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_3_col76     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_4_col77     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_5_col78     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_6_col79     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_7_col80     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_8_col81     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_9_col82     = cuda_evaluator.next_trace_mask();
    m31 op1_limb_10_col83    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_11_col84    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_12_col85    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_13_col86    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_14_col87    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_15_col88    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_16_col89    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_17_col90    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_18_col91    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_19_col92    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_20_col93    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_21_col94    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_22_col95    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_23_col96    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_24_col97    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_25_col98    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_26_col99    = cuda_evaluator.next_trace_mask();
    m31 op1_limb_27_col100   = cuda_evaluator.next_trace_mask();
    m31 sub_p_bit_col101     = cuda_evaluator.next_trace_mask();
    m31 enabler              = cuda_evaluator.next_trace_mask();

    const m31 M31_1 = m31(1);

    m31 decode_instruction_bc3cd_output_tmp_3fa46_10[19] = {0};
    evaluate_decode_instruction_bc3cd(
        input_pc_col0,
        offset0_col3,
        offset1_col4,
        offset2_col5,
        dst_base_fp_col6,
        op0_base_fp_col7,
        op1_imm_col8,
        op1_base_fp_col9,
        ap_update_add_1_col10,
        decode_instruction_bc3cd_output_tmp_3fa46_10,
        add_opcode_eval->common_lookup_elements,
        &cuda_evaluator
    );

    cuda_evaluator.add_constraint(
        mul(op1_imm_col8, sub(M31_1, decode_instruction_bc3cd_output_tmp_3fa46_10[2]))
    );
    cuda_evaluator.add_constraint(
        sub(
            mem_dst_base_col11,
            add(
                mul(dst_base_fp_col6, input_fp_col2),
                mul(sub(M31_1, dst_base_fp_col6), input_ap_col1)
            )
        )
    );
    cuda_evaluator.add_constraint(
        sub(
            mem0_base_col12,
            add(
                mul(op0_base_fp_col7, input_fp_col2),
                mul(sub(M31_1, op0_base_fp_col7), input_ap_col1)
            )
        )
    );
    // if (row == 0)
    cuda_evaluator.add_constraint(
        sub(
            mem1_base_col13,
            add(
                add(
                    mul(op1_imm_col8, input_pc_col0),
                    mul(op1_base_fp_col9, input_fp_col2)
                ),
                mul(decode_instruction_bc3cd_output_tmp_3fa46_10[7], input_ap_col1)
            )
        )
    );
    // if (row == 0)

    m31 read_positive_num_bits_252_output_tmp_3fa46_13[29] = {0};
    evaluate_read_positive_num_bits_252(
        add(mem_dst_base_col11, decode_instruction_bc3cd_output_tmp_3fa46_10[0]),
        dst_id_col14,
        dst_limb_0_col15, dst_limb_1_col16, dst_limb_2_col17, dst_limb_3_col18, dst_limb_4_col19,
        dst_limb_5_col20, dst_limb_6_col21, dst_limb_7_col22, dst_limb_8_col23, dst_limb_9_col24,
        dst_limb_10_col25, dst_limb_11_col26, dst_limb_12_col27, dst_limb_13_col28, dst_limb_14_col29,
        dst_limb_15_col30, dst_limb_16_col31, dst_limb_17_col32, dst_limb_18_col33, dst_limb_19_col34,
        dst_limb_20_col35, dst_limb_21_col36, dst_limb_22_col37, dst_limb_23_col38, dst_limb_24_col39,
        dst_limb_25_col40, dst_limb_26_col41, dst_limb_27_col42,
        read_positive_num_bits_252_output_tmp_3fa46_13,
        add_opcode_eval->common_lookup_elements,
        &cuda_evaluator
    );
    // if (row == 0)
    m31 read_positive_num_bits_252_output_tmp_3fa46_16[29] = {0};
    evaluate_read_positive_num_bits_252(
        add(mem0_base_col12, decode_instruction_bc3cd_output_tmp_3fa46_10[1]),
        op0_id_col43,
        op0_limb_0_col44, op0_limb_1_col45, op0_limb_2_col46, op0_limb_3_col47, op0_limb_4_col48,
        op0_limb_5_col49, op0_limb_6_col50, op0_limb_7_col51, op0_limb_8_col52, op0_limb_9_col53,
        op0_limb_10_col54, op0_limb_11_col55, op0_limb_12_col56, op0_limb_13_col57, op0_limb_14_col58,
        op0_limb_15_col59, op0_limb_16_col60, op0_limb_17_col61, op0_limb_18_col62, op0_limb_19_col63,
        op0_limb_20_col64, op0_limb_21_col65, op0_limb_22_col66, op0_limb_23_col67, op0_limb_24_col68,
        op0_limb_25_col69, op0_limb_26_col70, op0_limb_27_col71,
        read_positive_num_bits_252_output_tmp_3fa46_16,
        add_opcode_eval->common_lookup_elements,
        &cuda_evaluator
    );
    // if (row == 0)
    m31 read_positive_num_bits_252_output_tmp_3fa46_19[29] = {0};
    evaluate_read_positive_num_bits_252(
        add(mem1_base_col13, decode_instruction_bc3cd_output_tmp_3fa46_10[2]),
        op1_id_col72,
        op1_limb_0_col73, op1_limb_1_col74, op1_limb_2_col75, op1_limb_3_col76, op1_limb_4_col77,
        op1_limb_5_col78, op1_limb_6_col79, op1_limb_7_col80, op1_limb_8_col81, op1_limb_9_col82,
        op1_limb_10_col83, op1_limb_11_col84, op1_limb_12_col85, op1_limb_13_col86, op1_limb_14_col87,
        op1_limb_15_col88, op1_limb_16_col89, op1_limb_17_col90, op1_limb_18_col91, op1_limb_19_col92,
        op1_limb_20_col93, op1_limb_21_col94, op1_limb_22_col95, op1_limb_23_col96, op1_limb_24_col97,
        op1_limb_25_col98, op1_limb_26_col99, op1_limb_27_col100,
        read_positive_num_bits_252_output_tmp_3fa46_19,
        add_opcode_eval->common_lookup_elements,
        &cuda_evaluator
    );
    // if (row == 0)

    evaluate_verify_add_252(
        op0_limb_0_col44,
        op0_limb_1_col45,
        op0_limb_2_col46,
        op0_limb_3_col47,
        op0_limb_4_col48,
        op0_limb_5_col49,
        op0_limb_6_col50,
        op0_limb_7_col51,
        op0_limb_8_col52,
        op0_limb_9_col53,
        op0_limb_10_col54,
        op0_limb_11_col55,
        op0_limb_12_col56,
        op0_limb_13_col57,
        op0_limb_14_col58,
        op0_limb_15_col59,
        op0_limb_16_col60,
        op0_limb_17_col61,
        op0_limb_18_col62,
        op0_limb_19_col63,
        op0_limb_20_col64,
        op0_limb_21_col65,
        op0_limb_22_col66,
        op0_limb_23_col67,
        op0_limb_24_col68,
        op0_limb_25_col69,
        op0_limb_26_col70,
        op0_limb_27_col71,
        op1_limb_0_col73,
        op1_limb_1_col74,
        op1_limb_2_col75,
        op1_limb_3_col76,
        op1_limb_4_col77,
        op1_limb_5_col78,
        op1_limb_6_col79,
        op1_limb_7_col80,
        op1_limb_8_col81,
        op1_limb_9_col82,
        op1_limb_10_col83,
        op1_limb_11_col84,
        op1_limb_12_col85,
        op1_limb_13_col86,
        op1_limb_14_col87,
        op1_limb_15_col88,
        op1_limb_16_col89,
        op1_limb_17_col90,
        op1_limb_18_col91,
        op1_limb_19_col92,
        op1_limb_20_col93,
        op1_limb_21_col94,
        op1_limb_22_col95,
        op1_limb_23_col96,
        op1_limb_24_col97,
        op1_limb_25_col98,
        op1_limb_26_col99,
        op1_limb_27_col100,
        dst_limb_0_col15,
        dst_limb_1_col16,
        dst_limb_2_col17,
        dst_limb_3_col18,
        dst_limb_4_col19,
        dst_limb_5_col20,
        dst_limb_6_col21,
        dst_limb_7_col22,
        dst_limb_8_col23,
        dst_limb_9_col24,
        dst_limb_10_col25,
        dst_limb_11_col26,
        dst_limb_12_col27,
        dst_limb_13_col28,
        dst_limb_14_col29,
        dst_limb_15_col30,
        dst_limb_16_col31,
        dst_limb_17_col32,
        dst_limb_18_col33,
        dst_limb_19_col34,
        dst_limb_20_col35,
        dst_limb_21_col36,
        dst_limb_22_col37,
        dst_limb_23_col38,
        dst_limb_24_col39,
        dst_limb_25_col40,
        dst_limb_26_col41,
        dst_limb_27_col42,
        sub_p_bit_col101,
        &cuda_evaluator
    );
    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            input_pc_col0, input_ap_col1, input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            add_opcode_eval->common_lookup_elements,
            qm31{enabler, 0, 0, 0},
            values
        );
    }
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            add(add(input_pc_col0, M31_1), op1_imm_col8),
            add(input_ap_col1, ap_update_add_1_col10),
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            add_opcode_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},
            values
        );
    }
    numerators[row] = cuda_evaluator.row_res;
    constraint_index_array[row] = cuda_evaluator.constraint_index;
}

void evaluate_add_opcode(
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
    // Note: should_accumulate and use_assert_evaluator parameters are not used in this implementation
    (void)should_accumulate;
    (void)use_assert_evaluator;
    AddCode_Eval *add_opcode_eval = (AddCode_Eval *) eval;
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    AddCode_Eval *device_add_opcode_eval = cuda_malloc<AddCode_Eval>(1);
    cuda_mem_copy_host_to_device<AddCode_Eval>(add_opcode_eval, device_add_opcode_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_add_opcode");

    int block_dim = eval_domain_size < ADD_CODE_THREAD_COUNT_MAX ? eval_domain_size : ADD_CODE_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_add_opcode_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_add_opcode_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_add_opcode_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_add_opcode_eval,
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
    global_timer.end("evaluate_add_opcode");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_add_opcode_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}