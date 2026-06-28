#ifndef BLAKE2S_H
#define BLAKE2S_H

#include "fields.cuh"
#include "utils.cuh"

const unsigned int BLOCK_SIZE = 256;

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