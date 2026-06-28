#include "fields.cuh"
#include "evaluate_wide_fibonacci.cuh"
#include "evaluate_poseidon_constraint.cuh"
#include "timer.cuh"

__launch_bounds__(256, 2)
__global__ void evaluate_wide_fibonacci_constraint_quotients_kernel(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    m31 **trace1_evaluations,
    qm31 *numerators,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    Fraction *intermediate_fractions,
    unsigned logup_counts
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

    if (row < eval_domain_size) {
        m31 a = cuda_evaluator.next_trace_mask();
        m31 b = cuda_evaluator.next_trace_mask();

        for (unsigned instance_index = 2; instance_index < number_of_columns + 2; instance_index++) {
            m31 c = cuda_evaluator.next_trace_mask();

            cuda_evaluator.add_constraint(sub(c, add(square(a), square(b))));
            a = b;
            b = c;
        }
        numerators[row] = cuda_evaluator.row_res;

        m31 denom_inv = denominator_inverses[row >> domain_log_size];
        qm31 constraint_quotient = mul(
            denom_inv,
            numerators[row]
        );

        quotients_0[row] = constraint_quotient.a.a;
        quotients_1[row] = constraint_quotient.a.b;
        quotients_2[row] = constraint_quotient.b.a;
        quotients_3[row] = constraint_quotient.b.b;

    }
}

void evaluate_wide_fibonacci_constraint_quotients_on_domain(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts
) {
    // Clear any stale CUDA errors from previous API calls.
    cudaGetLastError();

    unsigned eval_domain_size = 1 << (eval_domain_log_size);
    m31 **device_trace0_evaluations = nullptr;
    if (trace0_evaluations_len > 0) {
        device_trace0_evaluations = clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len);
    }
    m31 **device_trace1_evaluations = clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len);
    qm31 *numerators = (qm31*) cuda_alloc_zeroes_uint32_t(4 * eval_domain_size);

    // Cap block_dim at 256 to match kernel's __launch_bounds__(256, 2).
    const unsigned MAX_BLOCK_DIM = 256;
    int block_dim = eval_domain_size < MAX_BLOCK_DIM ? eval_domain_size : MAX_BLOCK_DIM;
    int num_blocks = (eval_domain_size + block_dim - 1) / block_dim;

    Fraction *d_intermediate_fractions = nullptr;
    if (logup_counts > 0) {
        d_intermediate_fractions = cuda_malloc<Fraction>(eval_domain_size * logup_counts);
    }

    // Clear any errors from allocations above before kernel launch.
    cudaGetLastError();

    timer global_timer;
    // global_timer.start("evaluate_wide_fibonacci_constraint_quotients_on_domain");
    evaluate_wide_fibonacci_constraint_quotients_kernel<<<num_blocks, block_dim>>>(
        quotients_0, quotients_1, quotients_2, quotients_3,
        device_trace0_evaluations,
        device_trace1_evaluations,
        numerators,
        random_coeff_powers,
        denominator_inverses,
        domain_log_size,
        eval_domain_log_size,
        number_of_columns,
        d_intermediate_fractions,
        logup_counts
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // global_timer.end("evaluate_wide_fibonacci_constraint_quotients_on_domain");

    if (device_trace0_evaluations) cuda_free_memory(device_trace0_evaluations);
    cuda_free_memory(device_trace1_evaluations);
    cuda_free_memory(numerators);
    if (d_intermediate_fractions) cuda_free_memory(d_intermediate_fractions);
}

__launch_bounds__(256, 2)
__global__ void generate_wide_fibonacci_trace_kernel(
    m31 *input_a,
    m31 *input_b,
    unsigned input_len,
    m31 **device_trace,
    unsigned columns
) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < input_len) {
        device_trace[0][row_index] = input_a[row_index];
        device_trace[1][row_index] = input_b[row_index];
        for (int i = 2; i < columns; i++) {
            device_trace[i][row_index] = add(square(device_trace[i - 2][row_index]), square(device_trace[i - 1][row_index]));
        }
    }
}

void generate_wide_fibonacci_trace(
    m31 *input_a,
    m31 *input_b,
    unsigned input_len,
    m31 **traces,
    unsigned traces_len,
    unsigned n_columns
) {
    // Clear any stale CUDA errors from previous API calls.
    cudaGetLastError();

    m31 **device_trace = clone_to_device<m31*>(traces, traces_len);

    // Cap block_dim at 256 to match kernel's __launch_bounds__(256, 2).
    const unsigned MAX_BLOCK_DIM = 256;
    int block_dim = input_len < MAX_BLOCK_DIM ? input_len : MAX_BLOCK_DIM;
    int num_blocks = (input_len + block_dim - 1) / block_dim;

    // Clear any errors from allocations above before kernel launch.
    cudaGetLastError();

    generate_wide_fibonacci_trace_kernel<<<num_blocks, block_dim>>>(
        input_a,
        input_b,
        input_len,
        device_trace,
        n_columns
    );

    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_trace);
}