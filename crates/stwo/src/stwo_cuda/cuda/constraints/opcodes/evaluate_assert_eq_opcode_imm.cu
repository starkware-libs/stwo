#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include "evaluate_assert_eq_opcode_imm.cuh"
#include "evaluate_decode_instruction.cuh"
#include "evaluate_mem_verify_equal.cuh"
#include "evaluate_common.cuh"

#define ADD_CODE_SMALL_THREAD_COUNT_MAX 256

template<typename EvaluatorT>
__launch_bounds__(256, 2)
__global__ void evaluate_assert_eq_imm_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    AssertEqImmEval *assert_eq_imm_eval,
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

    m31 input_pc_col0         = cuda_evaluator.next_trace_mask();
    m31 input_ap_col1         = cuda_evaluator.next_trace_mask();
    m31 input_fp_col2         = cuda_evaluator.next_trace_mask();
    m31 offset0_col3          = cuda_evaluator.next_trace_mask();
    m31 dst_base_fp_col4      = cuda_evaluator.next_trace_mask();
    m31 ap_update_add_1_col5  = cuda_evaluator.next_trace_mask();
    m31 mem_dst_base_col6     = cuda_evaluator.next_trace_mask();
    m31 dst_id_col7           = cuda_evaluator.next_trace_mask();
    m31 enabler               = cuda_evaluator.next_trace_mask();

    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);

    // --- DecodeInstruction161C9 ---
    m31 decode_instruction_161c9_output_tmp_bb09e_5[19] = {0};
    evaluate_decode_instruction_161c9(
        input_pc_col0,
        offset0_col3,
        dst_base_fp_col4,
        ap_update_add_1_col5,
        decode_instruction_161c9_output_tmp_bb09e_5,
        assert_eq_imm_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // mem_dst_base
    cuda_evaluator.add_constraint(
        sub(
            mem_dst_base_col6,
            add(
                mul(dst_base_fp_col4, input_fp_col2),
                mul(sub(M31_1, dst_base_fp_col4), input_ap_col1)
            )
        )
    );

    // --- MemVerifyEqual ---
    m31 mem_verify_equal_inputs[2] = {
        add(mem_dst_base_col6, decode_instruction_161c9_output_tmp_bb09e_5[0]),
        add(input_pc_col0, M31_1)
    };
    evaluate_mem_verify_equal(
        mem_verify_equal_inputs[0],
        mem_verify_equal_inputs[1],
        dst_id_col7,
        assert_eq_imm_eval->common_lookup_elements,
        &cuda_evaluator
    );

    // enabler is boolean (v1.1.0: moved after MemVerifyEqual)
    cuda_evaluator.add_constraint(sub(mul(enabler, enabler), enabler));

    // --- Lookup: opcodes_0 ---
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            input_pc_col0, input_ap_col1, input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            assert_eq_imm_eval->common_lookup_elements,
            qm31{enabler},
            values
        );
    }
    // --- Lookup: opcodes_1 ---
    {
        m31 values[4] = {
            OPCODES_RELATION_ID,
            add(input_pc_col0, M31_2),
            add(input_ap_col1, ap_update_add_1_col5),
            input_fp_col2
        };
        cuda_evaluator.add_to_relation<4>(
            assert_eq_imm_eval->common_lookup_elements,
            qm31{{neg(enabler), 0}, {0, 0}},
            values
        );
    }

    constraint_index_array[row] = cuda_evaluator.constraint_index;
    numerators[row] = cuda_evaluator.row_res;
}

void evaluate_assert_eq_opcode_imm(
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
    AssertEqImmEval *assert_eq_eval = (AssertEqImmEval *) eval;
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    AssertEqImmEval *device_assert_eq_eval = cuda_malloc<AssertEqImmEval>(1);
    cuda_mem_copy_host_to_device<AssertEqImmEval>(assert_eq_eval, device_assert_eq_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constrain_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_assert_eq_opcode_imm");

    int block_dim = eval_domain_size < ADD_CODE_SMALL_THREAD_COUNT_MAX ? eval_domain_size : ADD_CODE_SMALL_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    if (use_assert_evaluator) {
        evaluate_assert_eq_imm_pre_kernel<CudaAssertEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_assert_eq_eval,
            cumsum_shift,
            d_intermediate_fractions,
            logup_counts,
            constrain_index_array
        );
    } else {
        evaluate_assert_eq_imm_pre_kernel<CudaEvaluator><<<num_blocks, block_dim, 0, stream>>>(
            numerators,
            device_trace1_evaluations,
            random_coeff_powers,
            domain_log_size,
            eval_domain_log_size,
            number_of_columns,
            device_assert_eq_eval,
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
    global_timer.end("evaluate_assert_eq_opcode_imm");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_assert_eq_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constrain_index_array);
}