#ifndef POLY_EVAL_AT_POINT_H
#define POLY_EVAL_AT_POINT_H

#include "fields.cuh"

extern "C"
qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y);

extern "C"
void batch_eval_at_points(
    m31 **coeffs_ptrs,
    int coeffs_size,
    int num_polys,
    qm31 point_x,
    qm31 point_y,
    qm31 *results
);

#endif // POLY_EVAL_AT_POINT_H