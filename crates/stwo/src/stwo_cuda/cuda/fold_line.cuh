#ifndef FRI_FOLD_LINE_H
#define FRI_FOLD_LINE_H

#include "fields.cuh"

extern "C"
uint32_t **fold_line_alloc_coord_ptrs(uint32_t **coord_ptrs);

extern "C"
void fold_line_launch(uint32_t *gpu_domain, uint32_t twiddle_offset, uint32_t n, qm31 alpha, uint32_t **eval_values_device, uint32_t **folded_values_device);

// Improvement 1a: fused batched fold. Runs all k line-fold steps of one layer in ONE kernel,
// block-resident in shared memory (1 global read + 1 global write instead of k round-trips).
// Returns true if the fused path launched; false if the layer/k is outside the tiling coverage
// (caller must fall back to k single-step `fold_line_launch` calls). See fold_line.cu for the
// exact local->global-step-r twiddle-index derivation and the tiling/fallback conditions.
extern "C"
bool fold_line_batch(
    uint32_t *gpu_domain,
    const uint32_t *twiddle_offsets, // k offsets: offset_r = twiddles_size - (1 << (log_n0 - r))
    uint32_t n0,                     // full layer length at entry (== eval length)
    uint32_t k,                      // number of fold steps (== alphas.len())
    const qm31 *alphas,              // k alphas, alphas[0] applied first
    uint32_t **eval_values_device,
    uint32_t **folded_values_device
);

#endif // FRI_FOLD_LINE_H