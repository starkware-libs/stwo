#include "eval_at_row.cuh"
#include "evaluate_poseidon_constraint.cuh"
#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "evaluate_common.cuh"

__constant__ m31 EXTERNAL_ROUND_CONSTS[2 * N_HALF_FULL_ROUNDS][N_STATE];
__constant__ m31 INTERNAL_ROUND_CONSTS[N_PARTIAL_ROUNDS];

static bool constant_inited = false;

#define POSEISON_THREAD_COUNT_MAX 256

void initialize_poseidon_constants() {
    m31 external_round_consts_init[2 * N_HALF_FULL_ROUNDS][N_STATE];
    m31 internal_round_consts_init[N_PARTIAL_ROUNDS];

    for(int r = 0; r < 2 * N_HALF_FULL_ROUNDS; r++) {
        for(int i = 0; i < N_STATE; i++) {
            external_round_consts_init[r][i] = 1234;
        }
    }

    for(int i = 0; i < N_PARTIAL_ROUNDS; i++) {
        internal_round_consts_init[i] = 1234;
    }

    cudaError_t err;
    err = cudaMemcpyToSymbol(EXTERNAL_ROUND_CONSTS, external_round_consts_init,
                             sizeof(m31) * 2 * N_HALF_FULL_ROUNDS * N_STATE);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy EXTERNAL_ROUND_CONSTS to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpyToSymbol(INTERNAL_ROUND_CONSTS, internal_round_consts_init,
                             sizeof(m31) * N_PARTIAL_ROUNDS);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy INTERNAL_ROUND_CONSTS to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Constants initialized and copied to device memory successfully.\n");
}

__launch_bounds__(256, 2)
__global__ void evaluate_poseidon_constraint_pre_kernel(
    qm31 *numerators,
    m31 **trace1_evaluations,
    qm31 *random_coeff_powers,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    PoseidonEval *poseidon_eval,
    qm31 cumsum_shift,
    Fraction *intermediate_fractions,
    unsigned logup_counts,
    unsigned *constraint_index_array
) {
    const unsigned eval_domain_size = 1u << eval_domain_log_size;
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= eval_domain_size) return;

    CudaEvaluator cuda_evaluator(
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

    for (unsigned instance_index = 0; instance_index < N_INSTANCES_PER_ROW; instance_index++) {
        // TODO: Tony (Move into L1/L2)
        m31 state[N_STATE] = { 0 };
        m31 init_state[N_STATE] = { 0 };
        for (unsigned i = 0; i < N_STATE; i++) {
            state[i] = cuda_evaluator.next_trace_mask();
            init_state[i] = state[i];
        }

        for (unsigned round = 0; round < N_HALF_FULL_ROUNDS; round++) {
            for (unsigned i = 0; i < N_STATE; i++) {
                state[i] = add(state[i], EXTERNAL_ROUND_CONSTS[round][i]);
            }
            apply_external_round_matrix(state);
            for (unsigned i = 0; i < N_STATE; i++) {
                state[i] = pow5(state[i]);
            }
            for (unsigned i = 0; i < N_STATE; i++) {
                const m31 m = cuda_evaluator.next_trace_mask();
                cuda_evaluator.add_constraint( sub(state[i], m));
                state[i] = m;
            }
        }

        for (unsigned round = 0; round < N_PARTIAL_ROUNDS; round++) {
            state[0] = add(state[0], INTERNAL_ROUND_CONSTS[round]);
            apply_internal_round_matrix(state);
            state[0] = pow5(state[0]);
            {
                const m31 m = cuda_evaluator.next_trace_mask();
                cuda_evaluator.add_constraint(sub(state[0], m));
                state[0] = m;
            }
        }

        for (unsigned round = 0; round < N_HALF_FULL_ROUNDS; round++) {
            for (unsigned i = 0; i < N_STATE; i++) {
                state[i] = add(state[i], EXTERNAL_ROUND_CONSTS[N_HALF_FULL_ROUNDS + round][i]);
            }
            apply_external_round_matrix(state);
            for (unsigned i = 0; i < N_STATE; i++) {
                state[i] = pow5(state[i]);
            }
            for (unsigned i = 0; i < N_STATE; i++) {
                const m31 m = cuda_evaluator.next_trace_mask();
                cuda_evaluator.add_constraint(sub(state[i], m));
                state[i] = m;
            }
        }

        RelationEntry entry = RelationEntry<N_STATE>(poseidon_eval->lookup_elements, qm31{{1,0}, {0,0}}, init_state);
        cuda_evaluator.add_to_relation<N_STATE>(entry);

        RelationEntry entry2 = RelationEntry<N_STATE>(poseidon_eval->lookup_elements, qm31{{2147483646,0}, {0,0}}, state);
        cuda_evaluator.add_to_relation<N_STATE>(entry2);
    }

    numerators[row] = cuda_evaluator.row_res;
    constraint_index_array[row] = cuda_evaluator.constraint_index;
}

void evaluate_poseidon_constraint_quotients_on_domain(
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
    if (!constant_inited) {
        initialize_poseidon_constants();
        constant_inited = true;
    }

    PoseidonEval *poseidon_eval = (PoseidonEval *) eval;
    unsigned int eval_domain_size = 1 << eval_domain_log_size;

    m31 **device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    m31 **device_trace2_evaluations = clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len);

    qm31 *numerators = (qm31 *) cuda_alloc_zeroes_uint32_t(sizeof(qm31) * eval_domain_size);

    PoseidonEval *device_poseidon_eval = cuda_malloc<PoseidonEval>(1);
    cuda_mem_copy_host_to_device<PoseidonEval>(poseidon_eval, device_poseidon_eval, 1);

    Fraction *d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    unsigned *constraint_index_array = cuda_alloc_zeroes_uint32_t(eval_domain_size);
    timer global_timer;
    global_timer.start("evaluate_poseidon_constraint_quotients_on_domain");

    int block_dim = eval_domain_size < POSEISON_THREAD_COUNT_MAX ? eval_domain_size : POSEISON_THREAD_COUNT_MAX;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    evaluate_poseidon_constraint_pre_kernel<<<num_blocks, block_dim, 0, stream>>>(
        numerators,
        device_trace1_evaluations,
        random_coeff_powers,
        domain_log_size,
        eval_domain_log_size,
        number_of_columns,
        device_poseidon_eval,
        cumsum_shift,
        d_intermediate_fractions,
        logup_counts,
        constraint_index_array
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Calculate batching
    std::vector<unsigned> batching(logup_counts);
    for (int i = 0; i < logup_counts; ++i) {
        batching[i] = i / 2;
    }
    unsigned last_batch = batching[logup_counts - 1];

    // Launch generic post-kernel
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

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Launch generic finalize kernel
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

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("evaluate_poseidon_constraint_quotients_on_domain");

    cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(device_trace2_evaluations);
    cuda_free_memory(numerators);
    cuda_free_memory(device_poseidon_eval);
    cuda_free_memory(d_intermediate_fractions);
    cuda_free_memory(constraint_index_array);
}


__global__ void generate_poseidon_trace_kernel(
    m31 **device_traces,
    m31 **device_lookup_init,
    m31 **device_lookup_final,
    unsigned trace_size
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_index < trace_size) {
        unsigned col_index = 0;
        for (unsigned rep_i = 0; rep_i < N_INSTANCES_PER_ROW; rep_i++) {
            // TODO: Tony (Move into L1/L2 Cache)
            m31 state[N_STATE] = { 0 };
            // Initial state.
            for (unsigned state_i = 0; state_i < N_STATE; state_i++) {
                state[state_i] = vec_index + state_i + rep_i;
                device_traces[col_index][vec_index] = state[state_i];
                device_lookup_init[state_i + N_STATE * rep_i][vec_index] = state[state_i];
                col_index++;
            }

            // 4 full rounds.
            for (unsigned round = 0; round < N_HALF_FULL_ROUNDS; round++) {
                for (unsigned i = 0; i < N_STATE; i++) {
                    state[i] = add(state[i], EXTERNAL_ROUND_CONSTS[round][i]);
                }
                apply_external_round_matrix(state);
                for (unsigned i = 0; i < N_STATE; i++) {
                    state[i] = pow5(state[i]);
                    device_traces[col_index][vec_index] = state[i];
                    col_index++;
                }
            }

            // Partial rounds.
            for (unsigned round = 0; round < N_PARTIAL_ROUNDS; round++) {
                state[0] = add(state[0], INTERNAL_ROUND_CONSTS[round]);
                apply_internal_round_matrix(state);
                state[0] = pow5(state[0]);
                device_traces[col_index][vec_index] = state[0];
                col_index++;
            }

            // 4 full rounds.
            for (unsigned round = 0; round < N_HALF_FULL_ROUNDS; round++) {
                for (unsigned i = 0; i < N_STATE; i++) {
                    state[i] = add(state[i], EXTERNAL_ROUND_CONSTS[N_HALF_FULL_ROUNDS + round][i]);
                }
                apply_external_round_matrix(state);
                for (unsigned state_i = 0; state_i < N_STATE; state_i++) {
                    state[state_i] = pow5(state[state_i]);
                    device_traces[col_index][vec_index] = state[state_i];
                    device_lookup_final[state_i + N_STATE * rep_i][vec_index] = state[state_i];
                    col_index++;
                }
            }
         }

    }
}

void generate_poseidon_traces(
    m31 **traces,
    m31 **lookup_init,
    m31 **lookup_final,
    unsigned trace_log_size
) {

    m31 **device_traces = clone_to_device<m31*>(traces, N_COLUMNS);
    m31 **device_lookup_init = clone_to_device<m31*>(lookup_init, N_STATE * N_INSTANCES_PER_ROW);
    m31 **device_lookup_final = clone_to_device<m31*>(lookup_final, N_STATE * N_INSTANCES_PER_ROW);

    if (!constant_inited) {
        initialize_poseidon_constants();
        constant_inited = true;
    }

    unsigned trace_size = 1 << trace_log_size;
    int block_dim = trace_size < POSEISON_THREAD_COUNT_MAX ? trace_size : POSEISON_THREAD_COUNT_MAX;
    int num_blocks = block_dim < POSEISON_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_poseidon_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        device_lookup_init,
        device_lookup_final,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_init);
    cuda_free_memory(device_lookup_final);
}

__launch_bounds__(256, 2)
__global__ void generate_poseidon_interaction_trace_col_gen_kernel(
    LookupElementsBasic<N_STATE> *lookup_elements,
    m31 **lookup_init,
    m31 **lookup_final,
    unsigned rep_index,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 init_combine_reg[N_STATE] = {};
    m31 final_combine_reg[N_STATE] = {};
    for (int i = 0; i < N_STATE; i++) {
        init_combine_reg[i] = lookup_init[i + rep_index * N_STATE][vec_index];
        final_combine_reg[i] = lookup_final[i + rep_index * N_STATE][vec_index];
    }

    if (vec_index < trace_size) {
        qm31 denom0 = lookup_elements->combine(init_combine_reg, N_STATE);
        qm31 denom1 = lookup_elements->combine(final_combine_reg, N_STATE);
        logup_col_write_frac(vec_index, sub(denom1, denom0), mul(denom0, denom1), denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

__global__ void generate_poseidon_interaction_trace_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = rep_index - 1;

    if (vec_index < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[vec_index], numerator1[vec_index]},
                cm31{numerator2[vec_index], numerator3[vec_index]}
            },
            denom_inv_ptr[vec_index]
        );

        if (pre_index == -1) {
            interaction_traces[0][vec_index] = 0;
            interaction_traces[1][vec_index] = 0;
            interaction_traces[2][vec_index] = 0;
            interaction_traces[3][vec_index] = 0;
            qm31 pre_value = qm31 {0};
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        } else {
            qm31 pre_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][vec_index], interaction_traces[pre_index * 4 + 1][vec_index]},
                cm31{interaction_traces[pre_index * 4 + 2][vec_index], interaction_traces[pre_index * 4 + 3][vec_index]}
            };
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        }

        interaction_traces[rep_index * 4 + 0][vec_index] = numerator0[vec_index];
        interaction_traces[rep_index * 4 + 1][vec_index] = numerator1[vec_index];
        interaction_traces[rep_index * 4 + 2][vec_index] = numerator2[vec_index];
        interaction_traces[rep_index * 4 + 3][vec_index] = numerator3[vec_index];
    }
}

__global__ void generate_poseidon_interaction_trace_cumsum_shift_ref(
    unsigned last_index,
    unsigned trace_size,
    m31 **interactive_traces,
    m31 *coordinate_sums
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_index == 0) {
        for (int i = 0; i < trace_size; i++) {
            coordinate_sums[0] = add(coordinate_sums[0], interactive_traces[4 * last_index - 4][i]);
        }
        for (int i = 0; i < trace_size; i++) {
            coordinate_sums[1] = add(coordinate_sums[1], interactive_traces[4 * last_index - 3][i]);
        }
        for (int i = 0; i < trace_size; i++) {
            coordinate_sums[2] = add(coordinate_sums[2], interactive_traces[4 * last_index - 2][i]);
        }
        for (int i = 0; i < trace_size; i++) {
            coordinate_sums[3] = add(coordinate_sums[3], interactive_traces[4 * last_index - 1][i]);
        }
    }
}

__global__ void generate_poseidon_interaction_trace_cumsum_shift(
    unsigned last_index,
    unsigned trace_size,
    m31 **interactive_traces,
    m31 *coordinate_sums
) {
    int idx0 = 4 * last_index - 4;
    int idx1 = 4 * last_index - 3;
    int idx2 = 4 * last_index - 2;
    int idx3 = 4 * last_index - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = 0;
    m31 sum1 = 0;
    m31 sum2 = 0;
    m31 sum3 = 0;

    for (int i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interactive_traces[idx0][i]);
        sum1 = add(sum1, interactive_traces[idx1][i]);
        sum2 = add(sum2, interactive_traces[idx2][i]);
        sum3 = add(sum3, interactive_traces[idx3][i]);
    }

    extern __shared__ m31 shared[];
    m31* sdata0 = &shared[0];
    m31* sdata1 = &shared[blockDim.x];
    m31* sdata2 = &shared[2 * blockDim.x];
    m31* sdata3 = &shared[3 * blockDim.x];

    sdata0[threadIdx.x] = sum0;
    sdata1[threadIdx.x] = sum1;
    sdata2[threadIdx.x] = sum2;
    sdata3[threadIdx.x] = sum3;

    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata0[threadIdx.x] = add(sdata0[threadIdx.x], sdata0[threadIdx.x + s]);
            sdata1[threadIdx.x] = add(sdata1[threadIdx.x], sdata1[threadIdx.x + s]);
            sdata2[threadIdx.x] = add(sdata2[threadIdx.x], sdata2[threadIdx.x + s]);
            sdata3[threadIdx.x] = add(sdata3[threadIdx.x], sdata3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sdata0[0]);
        atomic_add(&coordinate_sums[1], sdata1[0]);
        atomic_add(&coordinate_sums[2], sdata2[0]);
        atomic_add(&coordinate_sums[3], sdata3[0]);
    }
}

__global__ void generate_poseidon_interaction_trace_coord_prefix_sum(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    m31 **interactive_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interactive_traces[4 * last_index - 4][vec_index] = sub(interactive_traces[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interactive_traces[4 * last_index - 3][vec_index] = sub(interactive_traces[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interactive_traces[4 * last_index - 2][vec_index] = sub(interactive_traces[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interactive_traces[4 * last_index - 1][vec_index] = sub(interactive_traces[4 * last_index - 1][vec_index], cumsum_shift.b.b);

    }
}


void generate_poseidon_interaction_traces(
    void *lookup_elements_ptr,
    m31 **lookup_init,
    m31 **lookup_final,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    LookupElementsBasic<N_STATE> *lookup_elements = (LookupElementsBasic<N_STATE> *) lookup_elements_ptr;
    LookupElementsBasic<N_STATE> *device_lookup_elements = cuda_malloc<LookupElementsBasic<N_STATE>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<N_STATE>>(lookup_elements, device_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(1 << log_size);

    m31 **device_lookup_init = clone_to_device<m31*>(lookup_init, N_STATE * N_INSTANCES_PER_ROW);
    m31 **device_lookup_final = clone_to_device<m31*>(lookup_final, N_STATE * N_INSTANCES_PER_ROW);
    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4 * N_INSTANCES_PER_ROW);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    int block_dim = 0;
    int num_blocks = 0;
    for (unsigned rep_i = 0; rep_i < N_INSTANCES_PER_ROW; rep_i++) {
        block_dim = trace_size < POSEISON_THREAD_COUNT_MAX ? trace_size : POSEISON_THREAD_COUNT_MAX;
        num_blocks = block_dim < POSEISON_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        generate_poseidon_interaction_trace_col_gen_kernel<<<num_blocks, block_dim>>>(
            device_lookup_elements,
            device_lookup_init,
            device_lookup_final,
            rep_i,
            trace_size,
            device_logup_denom,
            device_numerator0,
            device_numerator1,
            device_numerator2,
            device_numerator3
        );

        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

        block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
        num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
        generate_poseidon_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
            rep_i,
            trace_size,
            denom_inv,
            device_numerator0,
            device_numerator1,
            device_numerator2,
            device_numerator3,
            device_interaction_traces
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

    }

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_poseidon_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        N_INSTANCES_PER_ROW,
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_poseidon_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        N_INSTANCES_PER_ROW,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[4 * N_INSTANCES_PER_ROW - 4], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * N_INSTANCES_PER_ROW - 3], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * N_INSTANCES_PER_ROW - 2], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * N_INSTANCES_PER_ROW - 1], trace_size);

    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_lookup_init);
    cuda_free_memory(device_lookup_final);
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}