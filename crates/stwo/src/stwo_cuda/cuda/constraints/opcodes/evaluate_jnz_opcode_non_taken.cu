#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_jnz_opcode_non_taken.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_mem_verify.cuh"
#include "evaluate_common.cuh"

#define JNZ_OPCODE_NON_TAKEN_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_jnz_opcode_non_taken_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    JnzOpcodeNonTaken_Eval *jnz_opcode_non_taken_eval,
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
        {{0,0},{0,0}},
        0,
        cumsum_shift,
        domain_log_size,
        eval_domain_log_size,
        intermediate_fractions,
        logup_counts
    );
    const m31 M31_0 = { 0 };
    const m31 M31_1 = { 1 };
    const m31 M31_2 = { 2 };

    // Load all 9 trace columns
    m31 input_pc_col0 = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1 = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2 = cuda_evaluator.next_trace_mask();
    m31 offset0_col3 = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col4 = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col5 = cuda_evaluator.next_trace_mask();
    m31 mem_dst_base_col6 = cuda_evaluator.next_trace_mask();
    m31 dst_id_col7 = cuda_evaluator.next_trace_mask();
    m31 enabler = cuda_evaluator.next_trace_mask();

    // DecodeInstructionDe75A
    // Returns output_vec[0] = offset0 - 32768
    m31 decode_instruction_de75a_output[19] = {0};
    evaluate_decode_instruction_de75a(
        input_pc_col0,
        offset0_col3,
        dst_base_fp_col4,
        ap_update_add_1_col5,
        decode_instruction_de75a_output,
        jnz_opcode_non_taken_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // mem_dst_base constraint:
    // mem_dst_base = dst_base_fp * input_fp + (1 - dst_base_fp) * input_ap
    cuda_evaluator.add_constraint(
        sub(
            mem_dst_base_col6,
            add(
                mul(dst_base_fp_col4, input_fp_col2),
                mul(sub(M31_1, dst_base_fp_col4), input_ap_col1)
            )
        )
    );

    // MemVerify: verify memory at address (mem_dst_base + offset0_decoded)
    // with value = 0 (all 28 limbs are zero).
    // MemVerify calls ReadId (MemoryAddressToId) then MemoryIdToBig.
    mem_verify_evaluate(
        add(mem_dst_base_col6, decode_instruction_de75a_output[0]),  // address
        M31_0,  // value_limb_0
        M31_0,  // value_limb_1
        M31_0,  // value_limb_2
        M31_0,  // value_limb_3
        M31_0,  // value_limb_4
        M31_0,  // value_limb_5
        M31_0,  // value_limb_6
        M31_0,  // value_limb_7
        M31_0,  // value_limb_8
        M31_0,  // value_limb_9
        M31_0,  // value_limb_10
        M31_0,  // value_limb_11
        M31_0,  // value_limb_12
        M31_0,  // value_limb_13
        M31_0,  // value_limb_14
        M31_0,  // value_limb_15
        M31_0,  // value_limb_16
        M31_0,  // value_limb_17
        M31_0,  // value_limb_18
        M31_0,  // value_limb_19
        M31_0,  // value_limb_20
        M31_0,  // value_limb_21
        M31_0,  // value_limb_22
        M31_0,  // value_limb_23
        M31_0,  // value_limb_24
        M31_0,  // value_limb_25
        M31_0,  // value_limb_26
        M31_0,  // value_limb_27
        dst_id_col7,
        jnz_opcode_non_taken_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // enabler is boolean (v1.1.0: moved after subroutines)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // Opcodes relation: positive multiplicity (enabler)
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            input_pc_col0,
            input_ap_col1,
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            jnz_opcode_non_taken_eval->common_lookup_elements,
            qm31{{enabler, 0}, {0, 0}},
            values
        );
    }

    // Opcodes relation: negative multiplicity (-enabler)
    // JNZ not taken: PC advances by 2, AP += ap_update_add_1, FP unchanged
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            add(input_pc_col0, M31_2),
            add(input_ap_col1, ap_update_add_1_col5),
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            jnz_opcode_non_taken_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

void evaluate_jnz_opcode_non_taken(
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
    JnzOpcodeNonTaken_Eval *jnz_opcode_non_taken_eval = (JnzOpcodeNonTaken_Eval *) eval;
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    JnzOpcodeNonTaken_Eval *device_jnz_opcode_non_taken_eval = cuda_malloc<JnzOpcodeNonTaken_Eval>(1);
    cuda_mem_copy_host_to_device<JnzOpcodeNonTaken_Eval>(jnz_opcode_non_taken_eval, device_jnz_opcode_non_taken_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constrain_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_jnz_opcode_non_taken");

    int block_dim = eval_domain_size < JNZ_OPCODE_NON_TAKEN_THREAD_COUNT_MAX ? eval_domain_size : JNZ_OPCODE_NON_TAKEN_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_jnz_opcode_non_taken_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jnz_opcode_non_taken_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constrain_index_array
        );
    } else {
        evaluate_jnz_opcode_non_taken_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_jnz_opcode_non_taken_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constrain_index_array
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
            constrain_index_array,
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
            constrain_index_array,
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
    global_timer.end("evaluate_jnz_opcode_non_taken");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_jnz_opcode_non_taken_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constrain_index_array);
}
