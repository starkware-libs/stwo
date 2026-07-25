#include "quotients.cuh"
#include <cstdio>


typedef struct {
    secure_field_point point;
    uint32_t *columns;
    qm31 *values;
    uint32_t size;
    size_t offset;
} column_sample_batch;

HOST_DEVICE_FORCEINLINE point index_to_point(uint32_t index) {
    return point_pow(m31_circle_gen, (int)index);
}

DEVICE_FORCEINLINE point domain_at_index(uint32_t half_coset_initial_index, uint32_t half_coset_step_size, uint32_t index, uint32_t domain_size) {
    uint32_t half_coset_size = domain_size >> 1;

    if (index < half_coset_size) {
        int modulo_u31_mask = 0x7fffffff;
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) index;
        return index_to_point(global_index & modulo_u31_mask);
    } else {
        int modulo_u31_mask = 0x7fffffff;
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) (index - half_coset_size);
        return index_to_point((2147483648 - global_index) & modulo_u31_mask);
    }
}

void column_sample_batches_for(
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        qm31 *sample_column_values,
        const uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        column_sample_batch *result
) {
    unsigned int offset = 0;
    for (unsigned int index = 0; index < sample_size; index++) {
        result[index].point = sample_points[index];
        result[index].columns = &sample_column_indexes[offset];
        result[index].values = &sample_column_values[offset];
        result[index].size = sample_column_and_values_sizes[index];
        result[index].offset = offset;
        offset += sample_column_and_values_sizes[index];
    }
}

DEVICE_FORCEINLINE void complex_conjugate_line_coeffs(secure_field_point point, qm31 value, qm31 alpha, qm31* a_out, qm31* b_out, qm31* c_out) {
    qm31 a = sub(qm31{value.a, neg(value.b)}, value);
    qm31 c = sub(qm31{point.y.a, neg(point.y.b)}, point.y);
    qm31 b = sub(mul(value, c), mul(a, point.y));

    *a_out = mul(alpha, a);
    *b_out = mul(alpha, b);
    *c_out = mul(alpha, c);
}

// Compute column line coefficients with global alpha counter across all batches.
// Alpha starts at 1 and is multiplied by random_coefficient after each column
// (monotonically across all sample batches).
// This must run sequentially because alpha is a running product.
__global__ void column_line_coeffs_kernel(
    column_sample_batch *sample_batches,
    uint32_t sample_size,
    qm31 random_coefficient,
    qm31 *flattened_line_coeffs,
    uint32_t *line_coeffs_sizes
) {
    // Single thread computes all coefficients sequentially
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    qm31 alpha = qm31{cm31{m31{1}, m31{0}}, cm31{m31{0}, m31{0}}};

    for (uint32_t i = 0; i < sample_size; i++) {
        line_coeffs_sizes[i] = sample_batches[i].size;
        size_t sample_batches_offset = sample_batches[i].offset * 3;

        for (size_t j = 0; j < sample_batches[i].size; ++j) {
            qm31 sampled_value = sample_batches[i].values[j];
            secure_field_point point = sample_batches[i].point;

            size_t sampled_offset = sample_batches_offset + (j * 3);
            complex_conjugate_line_coeffs(point, sampled_value, alpha,
                &flattened_line_coeffs[sampled_offset],
                &flattened_line_coeffs[sampled_offset + 1],
                &flattened_line_coeffs[sampled_offset + 2]);

            // Multiply alpha AFTER use (new algorithm)
            alpha = mul(alpha, random_coefficient);
        }
    }
}


DEVICE_FORCEINLINE void denominator_inverse(
        column_sample_batch *sample_batches,
        uint32_t sample_size,
        const point domain_point,
        cm31 *flat_denominators) {

    for (unsigned int i = 0; i < sample_size; i++) {
        cm31 prx = sample_batches[i].point.x.a;
        cm31 pry = sample_batches[i].point.y.a;
        cm31 pix = sample_batches[i].point.x.b;
        cm31 piy = sample_batches[i].point.y.b;

        cm31 first_substraction = {sub(prx.a, domain_point.x), prx.b};
        cm31 second_substraction = {sub(pry.a, domain_point.y), pry.b};
        cm31 result = sub(mul(first_substraction, piy),
                          mul(second_substraction, pix));
        flat_denominators[i] = inv(result);
    }
}

__global__ void accumulate_quotients_in_gpu(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_size,
        int domain_log_size,
        m31 **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        column_sample_batch *sample_batches,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3,
        qm31 *flattened_line_coeffs,
        uint32_t *line_coeffs_sizes,
        cm31 *denominator_inverses
) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    denominator_inverses = &denominator_inverses[row * sample_size];

    if (row < domain_size) {
        uint32_t domain_index = bit_reverse(row, domain_log_size);
        point domain_point = domain_at_index(half_coset_initial_index, half_coset_step_size, domain_index, domain_size);

        denominator_inverse(
            sample_batches,
            sample_size,
            domain_point,
            denominator_inverses
        );

        int i = 0;

        qm31 row_accumulator = qm31{cm31{0, 0}, cm31{0, 0}};
        int line_coeffs_offset = 0;
        while (i < sample_size) {
            column_sample_batch sample_batch = sample_batches[i];
            qm31 *line_coeffs = &flattened_line_coeffs[line_coeffs_offset * 3];
            int line_coeffs_size = line_coeffs_sizes[i];

            qm31 numerator = qm31{cm31{0, 0}, cm31{0, 0}};
            for(int j = 0; j < line_coeffs_size; j++) {
                qm31 a = line_coeffs[3 * j + 0];
                qm31 b = line_coeffs[3 * j + 1];
                qm31 c = line_coeffs[3 * j + 2];

                int column_index = sample_batch.columns[j];
                qm31 linear_term = add(mul_by_scalar(a, domain_point.y), b);
                qm31 value = mul_by_scalar(c, columns[column_index][row]);

                numerator = add(numerator, sub(value, linear_term));
            }

            // Simplified: no batch_coeff multiplication, just accumulate
            row_accumulator = add(row_accumulator, mul(numerator, denominator_inverses[i]));
            line_coeffs_offset += line_coeffs_size;
            i++;
        }

        result_column_0[row] = row_accumulator.a.a;
        result_column_1[row] = row_accumulator.a.b;
        result_column_2[row] = row_accumulator.b.a;
        result_column_3[row] = row_accumulator.b.b;

    }
}
// Kernel for accumulate_numerators: for each row, compute
//   partial_numerator[row] = sum_j (c_j * columns[column_indices[j]][row] - b_j)
__global__ void accumulate_numerators_kernel(
    int size,
    m31 **columns,
    qm31 *line_coeffs_b,
    qm31 *line_coeffs_c,
    uint32_t *column_indices,
    int num_coeffs,
    m31 *result_0, m31 *result_1, m31 *result_2, m31 *result_3
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= size) return;

    qm31 acc = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};
    for (int j = 0; j < num_coeffs; j++) {
        qm31 value = mul_by_scalar(line_coeffs_c[j], columns[column_indices[j]][row]);
        acc = add(acc, sub(value, line_coeffs_b[j]));
    }
    result_0[row] = acc.a.a;
    result_1[row] = acc.a.b;
    result_2[row] = acc.b.a;
    result_3[row] = acc.b.b;
}

void accumulate_numerators_batch(
    int size,
    m31 **columns,
    qm31 *line_coeffs_b_host,
    qm31 *line_coeffs_c_host,
    uint32_t *column_indices_host,
    int num_coeffs,
    m31 *result_0, m31 *result_1, m31 *result_2, m31 *result_3
) {
    qm31 *line_coeffs_b_device = clone_to_device<qm31>(line_coeffs_b_host, num_coeffs);
    qm31 *line_coeffs_c_device = clone_to_device<qm31>(line_coeffs_c_host, num_coeffs);
    uint32_t *column_indices_device = clone_to_device<uint32_t>(column_indices_host, num_coeffs);

    int block_dim = 1024;
    int num_blocks = (size + block_dim - 1) / block_dim;
    accumulate_numerators_kernel<<<num_blocks, block_dim>>>(
        size, columns,
        line_coeffs_b_device, line_coeffs_c_device, column_indices_device,
        num_coeffs,
        result_0, result_1, result_2, result_3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: stream ordering ensures kernel completes before async frees take effect.
    cuda_free_memory(line_coeffs_b_device);
    cuda_free_memory(line_coeffs_c_device);
    cuda_free_memory(column_indices_device);
}

// Kernel for compute_quotients_and_combine: for each row, compute
//   quotient[row] = sum over accumulations of:
//     (partial_numerator[lifted_idx] - first_linear_term_acc * domain_point.y) * den_inv
__global__ void compute_quotients_and_combine_kernel(
    int max_size, int max_log_size,
    uint32_t half_coset_initial_index, uint32_t half_coset_step_size,
    int num_accumulations,
    m31 **acc_partial_columns,
    int *acc_log_sizes,
    qm31 *first_linear_term_accs,
    secure_field_point *sample_points,
    m31 *result_0, m31 *result_1, m31 *result_2, m31 *result_3
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= max_size) return;

    uint32_t domain_index = bit_reverse(row, max_log_size);
    point domain_point = domain_at_index(half_coset_initial_index, half_coset_step_size, domain_index, max_size);

    qm31 quotient = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};

    for (int a = 0; a < num_accumulations; a++) {
        // Compute denominator inverse
        cm31 prx = sample_points[a].x.a;
        cm31 pry = sample_points[a].y.a;
        cm31 pix = sample_points[a].x.b;
        cm31 piy = sample_points[a].y.b;

        cm31 first_sub = {sub(prx.a, domain_point.x), prx.b};
        cm31 second_sub = {sub(pry.a, domain_point.y), pry.b};
        cm31 den_inv = inv(sub(mul(first_sub, piy), mul(second_sub, pix)));

        // Lifting index
        int log_ratio = max_log_size - acc_log_sizes[a];
        int lifted_idx = (row >> (log_ratio + 1) << 1) + (row & 1);

        // Read partial numerator
        int base = a * 4;
        qm31 partial = {
            cm31{acc_partial_columns[base + 0][lifted_idx], acc_partial_columns[base + 1][lifted_idx]},
            cm31{acc_partial_columns[base + 2][lifted_idx], acc_partial_columns[base + 3][lifted_idx]}
        };

        // full_numerator = partial - first_linear_term_acc * domain_point.y
        qm31 full_num = sub(partial, mul_by_scalar(first_linear_term_accs[a], domain_point.y));

        // quotient += full_numerator * den_inv (QM31 * CM31)
        quotient = add(quotient, mul(full_num, den_inv));
    }

    result_0[row] = quotient.a.a;
    result_1[row] = quotient.a.b;
    result_2[row] = quotient.b.a;
    result_3[row] = quotient.b.b;
}

void compute_quotients_and_combine(
    int max_size, int max_log_size,
    uint32_t half_coset_initial_index, uint32_t half_coset_step_size,
    int num_accumulations,
    m31 **acc_partial_columns_host,
    int *acc_log_sizes_host,
    qm31 *first_linear_term_accs_host,
    secure_field_point *sample_points_host,
    m31 *result_0, m31 *result_1, m31 *result_2, m31 *result_3
) {
    m31 **acc_partial_columns_device = clone_to_device<m31*>(acc_partial_columns_host, num_accumulations * 4);
    int *acc_log_sizes_device = clone_to_device<int>(acc_log_sizes_host, num_accumulations);
    qm31 *first_linear_term_accs_device = clone_to_device<qm31>(first_linear_term_accs_host, num_accumulations);
    secure_field_point *sample_points_device = clone_to_device<secure_field_point>(sample_points_host, num_accumulations);

    int block_dim = 1024;
    int num_blocks = (max_size + block_dim - 1) / block_dim;
    compute_quotients_and_combine_kernel<<<num_blocks, block_dim>>>(
        max_size, max_log_size,
        half_coset_initial_index, half_coset_step_size,
        num_accumulations,
        acc_partial_columns_device,
        acc_log_sizes_device,
        first_linear_term_accs_device,
        sample_points_device,
        result_0, result_1, result_2, result_3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: stream ordering ensures kernel completes before async frees.
    cuda_free_memory(acc_partial_columns_device);
    cuda_free_memory(acc_log_sizes_device);
    cuda_free_memory(first_linear_term_accs_device);
    cuda_free_memory(sample_points_device);
}

__global__ void dump_qm31_array(qm31 *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("(%d + %di) + (%d + %di)u, ", array[i].a.a, array[i].a.b, array[i].b.a, array[i].b.b);
    }
    printf("\n");
}

void accumulate_quotients(
        uint32_t half_coset_initial_index,
        uint32_t half_coset_step_size,
        uint32_t domain_size,
        m31 **columns,
        uint32_t number_of_columns,
        qm31 random_coefficient,
        secure_field_point *sample_points,
        uint32_t *sample_column_indexes,
        uint32_t sample_column_indexes_size,
        qm31 *sample_column_values,
        uint32_t *sample_column_and_values_sizes,
        uint32_t sample_size,
        uint32_t *result_column_0,
        uint32_t *result_column_1,
        uint32_t *result_column_2,
        uint32_t *result_column_3,
        uint32_t flattened_line_coeffs_size
) {
    int domain_log_size = log_2((int)domain_size);

    auto sample_batches = (column_sample_batch *)malloc(sizeof(column_sample_batch) * sample_size);
    memset(sample_batches, 0, sizeof(column_sample_batch) * sample_size);

    column_sample_batch *sample_batches_device = cuda_malloc<column_sample_batch>(sample_size);
    cm31* denominator_inverses = cuda_malloc<cm31>(sample_size * domain_size);

    uint32_t *sample_column_indexes_device = clone_to_device<uint32_t>(sample_column_indexes, sample_column_indexes_size);
    qm31 *sample_column_values_device = clone_to_device<qm31>(sample_column_values, sample_column_indexes_size);

    column_sample_batches_for(
            sample_points,
            sample_column_indexes_device,
            sample_column_values_device,
            sample_column_and_values_sizes,
            sample_size,
            sample_batches
    );

    cuda_mem_copy_host_to_device(sample_batches, sample_batches_device, sample_size);
    uint32_t *line_coeffs_sizes_device = cuda_malloc<uint32_t>(sample_size);
    qm31 *flattened_line_coeffs_device = cuda_malloc<qm31>(flattened_line_coeffs_size);

    // Compute column line coefficients (sequential, single thread on GPU)
    column_line_coeffs_kernel<<<1, 1>>>(
            sample_batches_device,
            sample_size,
            random_coefficient,
            flattened_line_coeffs_device,
            line_coeffs_sizes_device
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: next kernel on same stream reads this kernel's output.

    // TODO: set to 1024
    int block_dim = 512;
    int num_blocks = (domain_size + block_dim - 1) / block_dim;
    accumulate_quotients_in_gpu<<<num_blocks, block_dim>>>(
            half_coset_initial_index,
            half_coset_step_size,
            domain_size,
            domain_log_size,
            columns,
            number_of_columns,
            random_coefficient,
            sample_batches_device,
            sample_size,
            result_column_0,
            result_column_1,
            result_column_2,
            result_column_3,
            flattened_line_coeffs_device,
            line_coeffs_sizes_device,
            denominator_inverses
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: async frees are stream-ordered after kernel completion.
    free(sample_batches);
    cuda_free_memory(sample_batches_device);
    cuda_free_memory(denominator_inverses);
    cuda_free_memory(sample_column_indexes_device);
    cuda_free_memory(sample_column_values_device);
    cuda_free_memory(line_coeffs_sizes_device);
    cuda_free_memory(flattened_line_coeffs_device);
}
