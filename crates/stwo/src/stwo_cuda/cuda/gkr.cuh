#ifndef GKR_H
#define GKR_H

#include "fields.cuh"

extern "C"
void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size);

#endif // GKR_H
