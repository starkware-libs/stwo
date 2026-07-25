#ifndef FRI_FOLD_CIRCLE_INTO_LINE_H
#define FRI_FOLD_CIRCLE_INTO_LINE_H

#include "fields.cuh"

extern "C"
void fold_circle_into_line(uint32_t *gpu_domain,
                           uint32_t twiddle_offset,
                           uint32_t n,
                           uint32_t *eval_values[],
                           qm31 alpha,
                           uint32_t *folded_values[]);

#endif // FRI_FOLD_CIRCLE_INTO_LINE_H