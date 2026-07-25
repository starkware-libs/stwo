#ifndef QUOTIENTS_H
#define QUOTIENTS_H

#include "fields.cuh"
#include "point.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 1024;

extern "C"
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
);

extern "C"
void accumulate_numerators_batch(
    int size,
    m31 **columns,
    qm31 *line_coeffs_b_host,
    qm31 *line_coeffs_c_host,
    uint32_t *column_indices_host,
    int num_coeffs,
    m31 *result_0, m31 *result_1, m31 *result_2, m31 *result_3
);

extern "C"
void compute_quotients_and_combine(
    int max_size, int max_log_size,
    uint32_t half_coset_initial_index, uint32_t half_coset_step_size,
    int num_accumulations,
    m31 **acc_partial_columns_host,
    int *acc_log_sizes_host,
    qm31 *first_linear_term_accs_host,
    secure_field_point *sample_points_host,
    m31 *result_0, m31 *result_1, m31 *result_2, m31 *result_3
);

#endif // QUOTIENTS_H