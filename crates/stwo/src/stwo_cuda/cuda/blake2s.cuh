#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// LEVER L3, change 1: occupancy tuning knobs for the lifted-Merkle kernels.
//
// The lifted kernels (build_leaves_fused, lifted_build_next_layer,
// update_columns, finalize_all) are the blake2s commit that dominates
// `fri_commit` (~43% of t_base_fold). ncu shows them low-occupancy and
// memory-pipe (not DRAM) bound. Two occupancy levers, both compile-time:
//
//   BLAKE2S_LIFTED_BLK       — thread-block size for the lifted kernels.
//                              Sweep 128 / 256 / 512 on-box to find the
//                              occupancy/register-spill sweet spot.
//   BLAKE2S_LIFTED_MINBLOCKS — the second `__launch_bounds__` argument
//                              (min blocks resident per SM). Tells ptxas to
//                              cap per-thread registers so at least this many
//                              blocks co-reside, raising occupancy. Only a
//                              launch/codegen hint — it never changes the hash
//                              output.
//
// These knobs are OUTPUT-PRESERVING: they change grid/block shape and the
// register budget, never what bytes are hashed.
// ---------------------------------------------------------------------------
#ifndef BLAKE2S_LIFTED_BLK
#define BLAKE2S_LIFTED_BLK 256
#endif
#ifndef BLAKE2S_LIFTED_MINBLOCKS
#define BLAKE2S_LIFTED_MINBLOCKS 6
#endif

extern "C"
void commit_on_first_layer(uint32_t size, uint32_t number_of_columns, uint32_t **columns, Blake2sHash* result);

extern "C"
void commit_on_layer_with_previous(uint32_t size, uint32_t number_of_columns, uint32_t **columns, Blake2sHash* previous_layer, Blake2sHash* result);

extern "C"
void blake2s_lifted_build_next_layer(
    int size, Blake2sHash *prev_layer, Blake2sHash *result, bool is_m31_output
);

// Opaque pointer to GPU-resident Blake2sState array
extern "C"
void* blake2s_alloc_init_states(int count);

extern "C"
void blake2s_lift_states(
    void *prev_states, int prev_size,
    void **next_states_out, int next_size,
    int log_ratio
);

extern "C"
void blake2s_update_columns(
    void *states, int size,
    m31 **column_ptrs_host, int num_columns
);

extern "C"
void blake2s_finalize_all(
    void *states, Blake2sHash *output,
    int size, bool is_m31_output
);

// Fused build_leaves: single-pass kernel, state in registers, no domain separation.
// Use when all columns share the same log_size.
extern "C"
void blake2s_build_leaves_fused(
    uint32_t size, uint32_t number_of_columns,
    uint32_t **device_columns, Blake2sHash *result,
    bool is_m31_output
);

#endif // BLAKE2S_H