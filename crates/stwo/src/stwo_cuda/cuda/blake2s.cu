#include "blake2s.cuh"
#include "utils.cuh"

__device__ __constant__ uint32_t blake2s_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

__device__ __constant__ uint8_t blake2s_sigma[10][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 }
};


#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define G(r,i,a,b,c,d) \
    do { \
        a = a + b + m[blake2s_sigma[r][2*i+0]]; \
        d = ROTR32(d ^ a, 16); \
        c = c + d; \
        b = ROTR32(b ^ c, 12); \
        a = a + b + m[blake2s_sigma[r][2*i+1]]; \
        d = ROTR32(d ^ a, 8); \
        c = c + d; \
        b = ROTR32(b ^ c, 7); \
    } while(0)

typedef struct {
    uint32_t h[8];      // hash state
    uint32_t t;         // total bytes so far
    uint8_t  buf[64];   // buffer
    size_t   buflen;    // buffer usage
} Blake2sState;

__device__ void blake2s_compress(
    Blake2sState* S,
    const uint8_t block[64],
    uint32_t t,         // total bytes so far
    uint32_t lastblock  // 0 for normal, 0xFFFFFFFF for last block
) {
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        m[i] =  ((uint32_t)block[4*i+0]      ) |
                ((uint32_t)block[4*i+1] << 8 ) |
                ((uint32_t)block[4*i+2] << 16) |
                ((uint32_t)block[4*i+3] << 24);
    }

    uint32_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = S->h[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i+8] = blake2s_IV[i];

    v[12] ^= t;         // low 32 bits of offset
    v[13] ^= 0;         // high 32 bits (always 0 for <2^32 bytes)
    v[14] ^= lastblock; // 0xFFFFFFFF for last block

    // 10 rounds
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        G(r,0,v[0],v[4],v[8],v[12]);
        G(r,1,v[1],v[5],v[9],v[13]);
        G(r,2,v[2],v[6],v[10],v[14]);
        G(r,3,v[3],v[7],v[11],v[15]);
        G(r,4,v[0],v[5],v[10],v[15]);
        G(r,5,v[1],v[6],v[11],v[12]);
        G(r,6,v[2],v[7],v[8],v[13]);
        G(r,7,v[3],v[4],v[9],v[14]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++)
        S->h[i] ^= v[i] ^ v[i+8];
}


__device__ void blake2s_init(Blake2sState* S) {
    const uint32_t IV[8] = {
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    };
    S->h[0] = IV[0] ^ 0x01010020; // digest len = 32
    S->h[1] = IV[1];
    S->h[2] = IV[2];
    S->h[3] = IV[3];
    S->h[4] = IV[4];
    S->h[5] = IV[5];
    S->h[6] = IV[6];
    S->h[7] = IV[7];
    S->t = 0;
    S->buflen = 0;
}
__device__ void blake2s_update(Blake2sState* S, const uint8_t* in, size_t inlen) {
    size_t left = S->buflen;
    size_t fill = 64 - left;

    if (inlen > fill) {
        memcpy(S->buf + left, in, fill);
        S->t += 64;
        blake2s_compress(S, S->buf, S->t, 0);
        in += fill;
        inlen -= fill;
        while (inlen > 64) {
            S->t += 64;
            blake2s_compress(S, in, S->t, 0);
            in += 64;
            inlen -= 64;
        }
        left = 0;
    }
    memcpy(S->buf + left, in, inlen);
    S->buflen = left + inlen;
}

__device__ void blake2s_finalize(Blake2sState* S, Blake2sHash* out) {
    S->t += S->buflen;
    memset(S->buf + S->buflen, 0, 64 - S->buflen); // pad
    blake2s_compress(S, S->buf, S->t, 0xFFFFFFFF); // lastblock = 0xFFFFFFFF
    for (int i = 0; i < 8; i++) {
        out->s[i] = S->h[i];
    }
}


__global__ void commit_on_first_layer_in_gpu(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **data,
    Blake2sHash *result
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    Blake2sState state;
    blake2s_init(&state);

    for (int col = 0; col < number_of_columns; ++col) {
        uint32_t val = data[col][index];
        uint8_t bytes[4];
        bytes[0] = (val >>  0) & 0xFF;
        bytes[1] = (val >>  8) & 0xFF;
        bytes[2] = (val >> 16) & 0xFF;
        bytes[3] = (val >> 24) & 0xFF;
        blake2s_update(&state, bytes, sizeof(bytes));
    }
    blake2s_finalize(&state, &result[index]);
}
__global__ void commit_on_layer_using_previous_in_gpu(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **data,
    Blake2sHash *prev_layer,
    Blake2sHash *result
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    Blake2sState state;
    blake2s_init(&state);

    // left hash
    Blake2sHash left = prev_layer[2*index];
    for (int i = 0; i < 8; ++i) {
        uint32_t word = left.s[i];
        uint8_t bytes[4];
        bytes[0] = (word >>  0) & 0xFF;
        bytes[1] = (word >>  8) & 0xFF;
        bytes[2] = (word >> 16) & 0xFF;
        bytes[3] = (word >> 24) & 0xFF;
        blake2s_update(&state, bytes, sizeof(bytes));
    }
    // right hash
    Blake2sHash right = prev_layer[2*index+1];
    for (int i = 0; i < 8; ++i) {
        uint32_t word = right.s[i];
        uint8_t bytes[4];
        bytes[0] = (word >>  0) & 0xFF;
        bytes[1] = (word >>  8) & 0xFF;
        bytes[2] = (word >> 16) & 0xFF;
        bytes[3] = (word >> 24) & 0xFF;
        blake2s_update(&state, bytes, sizeof(bytes));
    }
    // current hash
    for (int col = 0; col < number_of_columns; ++col) {
        uint32_t val = data[col][index];
        uint8_t bytes[4];
        bytes[0] = (val >>  0) & 0xFF;
        bytes[1] = (val >>  8) & 0xFF;
        bytes[2] = (val >> 16) & 0xFF;
        bytes[3] = (val >> 24) & 0xFF;
        blake2s_update(&state, bytes, sizeof(bytes));
    }
    blake2s_finalize(&state, &result[index]);
}

// ============================================================================
// Lifted Merkle operations (no domain separation prefixes)
// ============================================================================

// build_next_layer for lifted Blake2s: hash pairs of children without NODE_PREFIX
__global__ void blake2s_lifted_build_next_layer_kernel(
    int size,
    Blake2sHash *prev_layer,
    Blake2sHash *result,
    bool is_m31_output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    Blake2sState state;
    blake2s_init(&state);

    // Update with left child hash (32 bytes as little-endian)
    Blake2sHash left = prev_layer[2 * i];
    for (int w = 0; w < 8; w++) {
        uint32_t word = left.s[w];
        uint8_t bytes[4];
        bytes[0] = (word >>  0) & 0xFF;
        bytes[1] = (word >>  8) & 0xFF;
        bytes[2] = (word >> 16) & 0xFF;
        bytes[3] = (word >> 24) & 0xFF;
        blake2s_update(&state, bytes, 4);
    }

    // Update with right child hash (32 bytes as little-endian)
    Blake2sHash right = prev_layer[2 * i + 1];
    for (int w = 0; w < 8; w++) {
        uint32_t word = right.s[w];
        uint8_t bytes[4];
        bytes[0] = (word >>  0) & 0xFF;
        bytes[1] = (word >>  8) & 0xFF;
        bytes[2] = (word >> 16) & 0xFF;
        bytes[3] = (word >> 24) & 0xFF;
        blake2s_update(&state, bytes, 4);
    }

    blake2s_finalize(&state, &result[i]);

    if (is_m31_output) {
        // reduce_to_m31: each u32 word mod P (P = 2^31 - 1)
        const uint32_t P = 0x7FFFFFFF;
        for (int w = 0; w < 8; w++) {
            uint64_t val = (uint64_t)result[i].s[w];
            uint32_t reduced = (uint32_t)(((((val >> 31) + val + 1) >> 31) + val) & P);
            result[i].s[w] = reduced;
        }
    }
}

void blake2s_lifted_build_next_layer(
    int size,
    Blake2sHash *prev_layer,
    Blake2sHash *result,
    bool is_m31_output
) {
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;
    blake2s_lifted_build_next_layer_kernel<<<num_blocks, block_dim>>>(
        size, prev_layer, result, is_m31_output
    );
    // cudaDeviceSynchronize removed: stream-ordered ops handle dependencies
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// Lift Blake2sState array: expand from prev_size to next_size using lifting index pattern
__global__ void blake2s_lift_states_kernel(
    Blake2sState *prev_states,
    Blake2sState *next_states,
    int next_size, int log_ratio
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= next_size) return;
    int src = (idx >> (log_ratio + 1) << 1) + (idx & 1);
    next_states[idx] = prev_states[src];
}

// Feed column data (M31 values as 4 little-endian bytes each) to Blake2s hasher states
__global__ void blake2s_update_columns_kernel(
    Blake2sState *states, int size,
    m31 **column_ptrs, int num_columns
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    for (int col = 0; col < num_columns; col++) {
        uint32_t val = column_ptrs[col][idx];
        uint8_t bytes[4];
        bytes[0] = (val >>  0) & 0xFF;
        bytes[1] = (val >>  8) & 0xFF;
        bytes[2] = (val >> 16) & 0xFF;
        bytes[3] = (val >> 24) & 0xFF;
        blake2s_update(&states[idx], bytes, 4);
    }
}

// Finalize all Blake2s hasher states into output hashes
__global__ void blake2s_finalize_all_kernel(
    Blake2sState *states, Blake2sHash *output,
    int size, bool is_m31_output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    blake2s_finalize(&states[idx], &output[idx]);
    if (is_m31_output) {
        const uint32_t P = 0x7FFFFFFF;
        for (int w = 0; w < 8; w++) {
            uint64_t val = (uint64_t)output[idx].s[w];
            uint32_t reduced = (uint32_t)(((((val >> 31) + val + 1) >> 31) + val) & P);
            output[idx].s[w] = reduced;
        }
    }
}

// Host function: allocate and init Blake2s states on GPU
void* blake2s_alloc_init_states(int count) {
    Blake2sState *states = cuda_malloc<Blake2sState>(count);
    Blake2sState init_state;
    const uint32_t IV[8] = {
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    };
    init_state.h[0] = IV[0] ^ 0x01010020;
    for (int i = 1; i < 8; i++) init_state.h[i] = IV[i];
    init_state.t = 0;
    init_state.buflen = 0;
    memset(init_state.buf, 0, 64);

    Blake2sState *host_states = (Blake2sState*)malloc(sizeof(Blake2sState) * count);
    for (int i = 0; i < count; i++) {
        host_states[i] = init_state;
    }
    cuda_mem_copy_host_to_device(host_states, states, count);
    free(host_states);
    return (void*)states;
}

// Host function: lift states
void blake2s_lift_states(
    void *prev_states_ptr, int prev_size,
    void **next_states_out, int next_size,
    int log_ratio
) {
    Blake2sState *prev_states = (Blake2sState*)prev_states_ptr;
    Blake2sState *next_states = cuda_malloc<Blake2sState>(next_size);
    int block_dim = 256;
    int num_blocks = (next_size + block_dim - 1) / block_dim;
    blake2s_lift_states_kernel<<<num_blocks, block_dim>>>(
        prev_states, next_states, next_size, log_ratio
    );
    // cudaDeviceSynchronize removed: stream-ordered ops handle dependencies
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    *next_states_out = (void*)next_states;
}

// Host function: update columns
void blake2s_update_columns(
    void *states_ptr, int size,
    m31 **column_ptrs_host, int num_columns
) {
    Blake2sState *states = (Blake2sState*)states_ptr;
    m31 **column_ptrs_device = clone_to_device<m31*>(column_ptrs_host, num_columns);
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;
    blake2s_update_columns_kernel<<<num_blocks, block_dim>>>(
        states, size, column_ptrs_device, num_columns
    );
    // cudaDeviceSynchronize removed: stream-ordered ops handle dependencies
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(column_ptrs_device);
}

// Host function: finalize all states
void blake2s_finalize_all(
    void *states_ptr, Blake2sHash *output,
    int size, bool is_m31_output
) {
    Blake2sState *states = (Blake2sState*)states_ptr;
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;
    blake2s_finalize_all_kernel<<<num_blocks, block_dim>>>(
        states, output, size, is_m31_output
    );
    // cudaDeviceSynchronize removed: stream-ordered ops handle dependencies
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// ============================================================================
// Fused build_leaves for lifted Merkle (GPU-optimized: state in registers)
// ============================================================================

// Single-pass kernel: init → process ALL columns → finalize per thread.
// State stays in registers the whole time — no global memory R/W for state.
// This replaces the multi-step alloc_init/lift/update_columns/finalize pipeline
// when all columns share the same log_size (the common case).
__global__ void blake2s_build_leaves_fused_kernel(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **data,
    Blake2sHash *result,
    bool is_m31_output
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;

    Blake2sState state;
    blake2s_init(&state);

    // No LEAF_PREFIX — lifted hasher has no domain separation
    for (int col = 0; col < number_of_columns; ++col) {
        uint32_t val = data[col][index];
        uint8_t bytes[4];
        bytes[0] = (val >>  0) & 0xFF;
        bytes[1] = (val >>  8) & 0xFF;
        bytes[2] = (val >> 16) & 0xFF;
        bytes[3] = (val >> 24) & 0xFF;
        blake2s_update(&state, bytes, sizeof(bytes));
    }
    blake2s_finalize(&state, &result[index]);

    if (is_m31_output) {
        const uint32_t P = 0x7FFFFFFF;
        for (int w = 0; w < 8; w++) {
            uint64_t val = (uint64_t)result[index].s[w];
            uint32_t reduced = (uint32_t)(((((val >> 31) + val + 1) >> 31) + val) & P);
            result[index].s[w] = reduced;
        }
    }
}

void blake2s_build_leaves_fused(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **device_columns,
    Blake2sHash *result,
    bool is_m31_output
) {
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;
    blake2s_build_leaves_fused_kernel<<<num_blocks, block_dim>>>(
        size, number_of_columns, device_columns, result, is_m31_output
    );
    // cudaDeviceSynchronize removed: stream-ordered ops handle dependencies
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// ============================================================================
// Original (non-lifted) Merkle operations
// ============================================================================

uint32_t number_of_blocks_for(uint32_t size) {
    return (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
}
void commit_on_first_layer(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **device_columns,
    Blake2sHash* result
) {
    commit_on_first_layer_in_gpu<<<number_of_blocks_for(size), BLOCK_SIZE>>>(
        size, number_of_columns, device_columns, result);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // cudaDeviceSynchronize removed: stream-ordered ops handle dependencies
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}
void commit_on_layer_with_previous(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **device_columns,
    Blake2sHash* previous_layer,
    Blake2sHash* result
) {
    commit_on_layer_using_previous_in_gpu<<<number_of_blocks_for(size), BLOCK_SIZE>>>(
        size, number_of_columns, device_columns, previous_layer, result);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // cudaDeviceSynchronize removed: stream-ordered ops handle dependencies
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}