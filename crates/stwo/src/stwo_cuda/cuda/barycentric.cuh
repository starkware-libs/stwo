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

// Option A (batched OODS): launch all n dot-product kernels with NO interior sync, one terminal
// device sync, one bulk D2H, then the same per-column CPU reduction. evals[e]/weights[e]/sizes[e]
// describe eval e; host_results[e] receives eval e's value. Byte-identical to n single calls.
extern "C"
void barycentric_eval_at_point_batched_cuda(
    m31 **evals,
    qm31 **weights,
    const int *sizes,
    int n,
    qm31 *host_results
);

#endif // BARYCENTRIC_H
