#include <cstdio>
#include <vector>

#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_verify_instruction.cuh"
#include "verify/encode_offsets_common.cuh"
#include "evaluate_mem_verify.cuh"
#include "evaluate_common.cuh"

#define VERIFY_INSTRUCTION_THREAD_COUNT_MAX 256

template <typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_verify_instruction_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    VerifyInstruction_Eval *verify_eval,
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

    EvaluatorT cuda_evaluator(
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

    const m31 M31_0 = 0;

    // Read trace columns in the same order as CPU Eval.
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_offset0_col1 = cuda_evaluator.next_trace_mask();
    m31 input_offset1_col2 = cuda_evaluator.next_trace_mask();
    m31 input_offset2_col3 = cuda_evaluator.next_trace_mask();
    m31 input_inst_felt5_high_col4 = cuda_evaluator.next_trace_mask();
    m31 input_inst_felt6_col5 = cuda_evaluator.next_trace_mask();
    m31 input_opcode_extension_col6 = cuda_evaluator.next_trace_mask();
    m31 offset0_low_col7 = cuda_evaluator.next_trace_mask();
    m31 offset0_mid_col8 = cuda_evaluator.next_trace_mask();
    m31 offset1_low_col9 = cuda_evaluator.next_trace_mask();
    m31 offset1_mid_col10 = cuda_evaluator.next_trace_mask();
    m31 offset1_high_col11 = cuda_evaluator.next_trace_mask();
    m31 offset2_low_col12 = cuda_evaluator.next_trace_mask();
    m31 offset2_mid_col13 = cuda_evaluator.next_trace_mask();
    m31 offset2_high_col14 = cuda_evaluator.next_trace_mask();
    m31 instruction_id_col15 = cuda_evaluator.next_trace_mask();
    m31 multiplicity = cuda_evaluator.next_trace_mask();

    // EncodeOffsets child program.
    m31 encode_offsets_limb_1 = 0;
    m31 encode_offsets_limb_3 = 0;
    encode_offsets_evaluate(
        input_offset0_col1,
        input_offset1_col2,
        input_offset2_col3,
        offset0_low_col7,
        offset0_mid_col8,
        offset1_low_col9,
        offset1_mid_col10,
        offset1_high_col11,
        offset2_low_col12,
        offset2_mid_col13,
        offset2_high_col14,
        verify_eval->common_lookup_elements,
        &encode_offsets_limb_1,
        &encode_offsets_limb_3,
        &cuda_evaluator
    );

    // MemVerify child program.
    mem_verify_evaluate(
        input_pc_col0,
        offset0_low_col7,
        encode_offsets_limb_1,
        offset1_mid_col10,
        encode_offsets_limb_3,
        offset2_mid_col13,
        add(offset2_high_col14, input_inst_felt5_high_col4),
        input_inst_felt6_col5,
        input_opcode_extension_col6,
        // Remaining 20 limbs filled with 0, consistent with CPU-side MemVerify::evaluate call.
        M31_0, M31_0, M31_0, M31_0, M31_0,
        M31_0, M31_0, M31_0, M31_0, M31_0,
        M31_0, M31_0, M31_0, M31_0, M31_0,
        M31_0, M31_0, M31_0, M31_0, M31_0,
        instruction_id_col15,
        verify_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // VerifyInstruction logup relation.
    {
        m31 values[8] = {
            VERIFY_INSTRUCTION_RELATION_ID,
            input_pc_col0,
            input_offset0_col1,
            input_offset1_col2,
            input_offset2_col3,
            input_inst_felt5_high_col4,
            input_inst_felt6_col5,
            input_opcode_extension_col6,
        };

        m31 neg_mult = neg(multiplicity);
        qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};

        cuda_evaluator.add_to_relation<8>(verify_eval->common_lookup_elements, multiplicity_ext, values);
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

extern "C"
void evaluate_verify_instruction(
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
    (void)trace0_evaluations;
    (void)trace0_evaluations_len;
    (void)number_of_columns;

    VerifyInstruction_Eval *verify_eval =
        (VerifyInstruction_Eval *)eval;
    unsigned int eval_domain_size = 1u << eval_domain_log_size;

    m31 **device_trace1_evaluations =
        clone_to_device<m31 *>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations =
        clone_to_device<m31 *>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators =
        (qm31 *)cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    VerifyInstruction_Eval *device_verify_eval =
        cuda_malloc<VerifyInstruction_Eval>(1);
    cuda_mem_copy_host_to_device<VerifyInstruction_Eval>(
        verify_eval, device_verify_eval, 1);

    Fraction *d_intermediate_fractions =
        cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array =
        cuda_alloc_zeroes_uint32_t(eval_domain_size);

    timer global_timer;
    global_timer.start("evaluate_verify_instruction");

    int block_dim = eval_domain_size < VERIFY_INSTRUCTION_THREAD_COUNT_MAX
        ? eval_domain_size
        : VERIFY_INSTRUCTION_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_verify_instruction_pre_kernel<CudaAssertEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_verify_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constraint_index_array
        );
    } else {
        evaluate_verify_instruction_pre_kernel<CudaEvaluator><<<
            num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_verify_eval,
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
    global_timer.end("evaluate_verify_instruction");

    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_verify_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}
