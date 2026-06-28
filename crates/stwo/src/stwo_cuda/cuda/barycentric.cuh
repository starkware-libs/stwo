#ifndef BARYCENTRIC_H
#define BARYCENTRIC_H

#include "fields.cuh"

extern "C"
void barycentric_weights_cuda(
    uint32_t half_coset_initial_index,
    uint32_t half_coset_step_size,
    int domain_size,
    int log_size,
    qm31 vn_p,
    qm31 p_x,
    qm31 p_y,
    m31 exp_val,
    qm31 *result
);

extern "C"
void barycentric_eval_at_point_cuda(
    m31 *evals,
    qm31 *weights,
    int size,
    qm31 *host_result
);

#endif // BARYCENTRIC_H
