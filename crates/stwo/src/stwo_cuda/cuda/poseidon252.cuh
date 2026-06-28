#ifndef POSEIDON252_H
#define POSEIDON252_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "utils.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Poseidon252 hash represented as 32 bytes (256 bits)
// This matches the starknet_ff::FieldElement representation
#pragma pack(push, 1)
typedef struct {
    uint8_t bytes[32];
} Poseidon252Hash;
#pragma pack(pop)

// Memory management functions
Poseidon252Hash* cuda_malloc_poseidon252_hash(size_t size);
Poseidon252Hash* cuda_alloc_zeroes_poseidon252_hash(size_t size);

// Copy functions
Poseidon252Hash* copy_poseidon252_hash_vec_from_host_to_device(const Poseidon252Hash* from, size_t size);
void copy_poseidon252_hash_vec_from_device_to_host(const Poseidon252Hash* from, Poseidon252Hash* to, size_t size);
void copy_poseidon252_hash_vec_from_device_to_device(const Poseidon252Hash* from, Poseidon252Hash* dst, size_t size);
void cuda_get_poseidon252_hash(const Poseidon252Hash* device_ptr, Poseidon252Hash* host_ptr, size_t index);
// Match Rust FFI order: (device_ptr, index, value)
void cuda_set_poseidon252_hash(Poseidon252Hash* device_ptr, size_t index, const Poseidon252Hash* host_ptr);

// Merkle tree functions (GPU, device pointers)
void poseidon252_commit_on_first_layer(
    size_t size,
    size_t amount_of_columns,
    const uint32_t* const* columns,
    Poseidon252Hash* result
);

void poseidon252_commit_on_layer_with_previous(
    size_t size,
    size_t amount_of_columns,
    const uint32_t* const* columns,
    const Poseidon252Hash* previous_layer,
    Poseidon252Hash* result
);

// Lifted Merkle operations
void poseidon252_lifted_build_next_layer(
    int size, const Poseidon252Hash *prev_layer, Poseidon252Hash *result
);

void* poseidon252_alloc_init_states(int count);

void poseidon252_lift_states(
    void *prev_states, int prev_size,
    void **next_states_out, int next_size,
    int log_ratio
);

void poseidon252_update_columns(
    void *states, int size,
    uint32_t **column_ptrs_host, int num_columns
);

void poseidon252_finalize_all(
    void *states, Poseidon252Hash *output,
    int size
);

// Fused build_leaves: single-pass kernel, state in registers.
// Use when all columns share the same log_size.
void poseidon252_build_leaves_fused(
    size_t size, size_t number_of_columns,
    const uint32_t* const* device_columns, Poseidon252Hash *result
);

#ifdef __cplusplus
}
#endif

#endif // POSEIDON252_H
