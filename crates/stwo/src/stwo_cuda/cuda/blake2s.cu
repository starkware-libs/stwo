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

// ---------------------------------------------------------------------------
// L3 change 2: word-path Blake2s.
//
// The original byte-path streamed message bytes into `buf[64]`, then `compress`
// reassembled each 4-byte little-endian group back into a word `m[i]`. But every
// byte fed came from an M31 word via `val.to_le_bytes()`
// (bytes = { val>>0, val>>8, val>>16, val>>24 }), so `compress`'s reassembly
// `m[i] = b0 | b1<<8 | b2<<16 | b3<<24` reproduced EXACTLY the original word.
// The round-trip word -> 4 LE bytes -> word is the identity on u32. Every message
// this file hashes is a whole number of 4-byte M31 words (leaf columns are u32;
// child hashes are 8 u32 words), so the byte buffer never holds a partial word.
//
// The word-path therefore skips the byte detour: it feeds message words straight
// into `m[]`. Because `m[]` is bit-identical to the byte-path's, and the IV,
// parameter block (0x01010020, digest len 32), `t` byte counter, high-offset
// word (v[13]^=0), final-block flag (v[14]^=0xFFFFFFFF) and the 10 G-rounds are
// all unchanged, the output hash is bit-identical. `t` is still counted in BYTES
// (16 words = 64 bytes per full block), preserving the exact final-block offset.
//
// The word buffer holds the same words, in the same order, that the byte buffer
// held as LE bytes; a block boundary fires at the same cumulative 64-byte mark.
// ---------------------------------------------------------------------------

// Streaming state, word-path. Buffer is 16 WORDS (== 64 bytes), matching the
// byte-path's buf[64] exactly, but indexed in words. `wlen` counts BUFFERED
// WORDS (0..15). `t` still counts TOTAL BYTES so the final-block offset is
// identical to the byte-path. Used by the persistent (global-memory-resident)
// streaming path: alloc_init / lift_states / update_columns / finalize_all.
typedef struct {
    uint32_t h[8];      // hash state
    uint32_t t;         // total bytes so far
    uint32_t wbuf[16];  // buffered message words (== byte-path buf[64])
    uint32_t wlen;      // buffered words in wbuf (0..15)
} Blake2sState;

// Word-path compress: `m[]` is supplied directly (no byte reassembly). This is
// the single point that guarantees byte-identity — see the equivalence note
// above. Semantics (IV mix, t/lastblock XORs, 10 rounds, feed-forward) are
// copied verbatim from the byte-path `blake2s_compress`.
__device__ __forceinline__ void blake2s_compress_words(
    uint32_t h[8],
    const uint32_t m[16],
    uint32_t t,         // total bytes so far
    uint32_t lastblock  // 0 for normal, 0xFFFFFFFF for last block
) {
    uint32_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = h[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i+8] = blake2s_IV[i];

    v[12] ^= t;         // low 32 bits of offset
    v[13] ^= 0;         // high 32 bits (always 0 for <2^32 bytes)
    v[14] ^= lastblock; // 0xFFFFFFFF for last block

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
        h[i] ^= v[i] ^ v[i+8];
}

// ---------------------------------------------------------------------------
// Register-resident word-path hasher for SELF-CONTAINED kernels (init -> absorb
// whole message -> finalize within one thread, state never leaves registers).
// Keeps ONLY h[8] + a byte counter + a small word-staging buffer in registers —
// no Blake2sState struct, no buf[64]/buflen. This is the register relief: the
// 88-byte Blake2sState is gone from build_leaves_fused / lifted_build_next_layer.
//
// Byte-identity: absorb-associativity + the compress_words equivalence above.
// Feeding words w0,w1,... in order and firing compress_words every 16 words with
// t += 64, then finalizing the tail with t += 4*wlen and lastblock=0xFFFFFFFF,
// reproduces the exact block sequence, `m[]` contents, `t` values and final flag
// of the byte-path.
// ---------------------------------------------------------------------------
struct Blake2sWordHasher {
    uint32_t h[8];
    uint32_t m[16];     // current partial block (staging)
    uint32_t t;         // total bytes absorbed so far
    uint32_t wlen;      // words currently staged in m[] (0..15)

    __device__ __forceinline__ void init() {
        h[0] = blake2s_IV[0] ^ 0x01010020; // digest len = 32, matches blake2s_init
        #pragma unroll
        for (int i = 1; i < 8; i++) h[i] = blake2s_IV[i];
        t = 0;
        wlen = 0;
    }

    // Absorb one message word (an M31 value or a hash limb, little-endian).
    //
    // LAZY boundary — MUST match the byte-path exactly: the original
    // `blake2s_update` compresses a buffered block only when the NEXT input
    // arrives (`if (inlen > fill)`, strictly greater), never eagerly on fill. So
    // a full 16-word block is held back and compressed as the FINAL block by
    // finalize (with lastblock set), NOT as a non-final block. Here we mirror
    // that: if the buffer is already full (16 words) when a new word arrives,
    // flush it as a NON-final block first, then stage the new word.
    __device__ __forceinline__ void absorb(uint32_t word) {
        if (wlen == 16) {
            t += 64;
            blake2s_compress_words(h, m, t, 0);
            wlen = 0;
        }
        m[wlen++] = word;
    }

    // Finalize: pad the partial block with zero words and compress with the
    // final-block flag. Mirrors blake2s_finalize (t += buflen; zero-pad; compress
    // with lastblock=0xFFFFFFFF). Here buflen == 4*wlen bytes.
    __device__ __forceinline__ void finalize(Blake2sHash* out) {
        t += 4 * wlen;
        #pragma unroll
        for (int i = 0; i < 16; i++) if ((uint32_t)i >= wlen) m[i] = 0;
        blake2s_compress_words(h, m, t, 0xFFFFFFFF);
        #pragma unroll
        for (int i = 0; i < 8; i++) out->s[i] = h[i];
    }
};

__device__ void blake2s_compress(
    Blake2sState* S,
    uint32_t t,         // total bytes so far
    uint32_t lastblock  // 0 for normal, 0xFFFFFFFF for last block
) {
    // Word-path: wbuf already holds the block's message words (see equivalence
    // note above compress_words). No byte reassembly needed.
    blake2s_compress_words(S->h, S->wbuf, t, lastblock);
}


__device__ void blake2s_init(Blake2sState* S) {
    S->h[0] = blake2s_IV[0] ^ 0x01010020; // digest len = 32
    #pragma unroll
    for (int i = 1; i < 8; i++) S->h[i] = blake2s_IV[i];
    S->t = 0;
    S->wlen = 0;
}

// Word-path streaming absorb. Every caller in this file feeds whole 4-byte,
// 4-byte-ALIGNED little-endian M31 words / hash limbs, so `inlen` is always a
// multiple of 4 and the buffer never holds a partial word. Bytes are folded back
// into their source word (b0|b1<<8|b2<<16|b3<<24), the exact inverse of the
// `to_le_bytes` split every caller applied.
//
// This is a WORD-for-BYTE port of the original byte-path `blake2s_update` (16
// words == 64 bytes; fill/left in words; `t` still in bytes). The LAZY boundary
// is preserved: a full block is compressed only when MORE input follows
// (`nwords > fill` and `nwords > 16`, both strictly greater) — never eagerly on
// fill — so the last full block is deferred to `finalize` and compressed as the
// FINAL block, exactly as before. Result: identical buffered words, block
// boundaries, `t` values and final-block flag ⇒ bit-identical hash.
__device__ void blake2s_update(Blake2sState* S, const uint8_t* in, size_t inlen) {
    size_t nwords = inlen >> 2;         // inlen is a multiple of 4 for all callers
    size_t off = 0;                     // word offset into `in`
    size_t left = S->wlen;             // buffered words
    size_t fill = 16 - left;           // words to top up the current block

    if (nwords > fill) {
        // Top up and flush the current (now full) block as NON-final.
        for (size_t j = 0; j < fill; j++) {
            const uint8_t* p = in + 4 * (off + j);
            S->wbuf[left + j] = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
                                ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }
        S->t += 64;
        blake2s_compress(S, S->t, 0);
        off += fill;
        nwords -= fill;
        // Flush every remaining FULL block except the last (deferred to finalize).
        while (nwords > 16) {
            for (int j = 0; j < 16; j++) {
                const uint8_t* p = in + 4 * (off + j);
                S->wbuf[j] = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
                             ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            }
            S->t += 64;
            blake2s_compress(S, S->t, 0);
            off += 16;
            nwords -= 16;
        }
        left = 0;
    }
    // Buffer the tail (0..16 words) into wbuf.
    for (size_t j = 0; j < nwords; j++) {
        const uint8_t* p = in + 4 * (off + j);
        S->wbuf[left + j] = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
                            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    S->wlen = left + nwords;
}

__device__ void blake2s_finalize(Blake2sState* S, Blake2sHash* out) {
    S->t += 4 * S->wlen;                 // == byte-path S->t += buflen
    #pragma unroll
    for (int i = 0; i < 16; i++)         // zero-pad the tail (== memset)
        if ((uint32_t)i >= S->wlen) S->wbuf[i] = 0;
    blake2s_compress(S, S->t, 0xFFFFFFFF); // lastblock = 0xFFFFFFFF
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
// __launch_bounds__ (L3 change 1): cap registers so >= BLAKE2S_LIFTED_MINBLOCKS
// blocks co-reside per SM. Output-preserving (codegen hint only).
__global__ void __launch_bounds__(BLAKE2S_LIFTED_BLK, BLAKE2S_LIFTED_MINBLOCKS)
blake2s_lifted_build_next_layer_kernel(
    int size,
    Blake2sHash *prev_layer,
    Blake2sHash *result,
    bool is_m31_output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // Word-path, register-resident (L3 change 2): node message = 16 words
    // (left.s[0..8] then right.s[0..8]) == 64 bytes == exactly one block. Feeding
    // each hash limb directly as a message word is bit-identical to splitting it
    // into LE bytes and reassembling (see compress_words equivalence note).
    Blake2sWordHasher hasher;
    hasher.init();

    // Left child hash (8 words, little-endian limbs)
    Blake2sHash left = prev_layer[2 * i];
    #pragma unroll
    for (int w = 0; w < 8; w++) hasher.absorb(left.s[w]);

    // Right child hash (8 words, little-endian limbs)
    Blake2sHash right = prev_layer[2 * i + 1];
    #pragma unroll
    for (int w = 0; w < 8; w++) hasher.absorb(right.s[w]);

    hasher.finalize(&result[i]);

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
    int block_dim = BLAKE2S_LIFTED_BLK;
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
// __launch_bounds__ (L3 change 1): output-preserving occupancy hint.
__global__ void __launch_bounds__(BLAKE2S_LIFTED_BLK, BLAKE2S_LIFTED_MINBLOCKS)
blake2s_update_columns_kernel(
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
// __launch_bounds__ (L3 change 1): output-preserving occupancy hint.
__global__ void __launch_bounds__(BLAKE2S_LIFTED_BLK, BLAKE2S_LIFTED_MINBLOCKS)
blake2s_finalize_all_kernel(
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
    init_state.wlen = 0;                     // word-path buffer empty
    memset(init_state.wbuf, 0, sizeof(init_state.wbuf));

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
    int block_dim = BLAKE2S_LIFTED_BLK;
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
    int block_dim = BLAKE2S_LIFTED_BLK;
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
    int block_dim = BLAKE2S_LIFTED_BLK;
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
// __launch_bounds__ (L3 change 1): the hottest lifted kernel (leaf layer of the
// commit). Output-preserving occupancy hint; sweep BLAKE2S_LIFTED_BLK on-box.
__global__ void __launch_bounds__(BLAKE2S_LIFTED_BLK, BLAKE2S_LIFTED_MINBLOCKS)
blake2s_build_leaves_fused_kernel(
    uint32_t size,
    uint32_t number_of_columns,
    uint32_t **data,
    Blake2sHash *result,
    bool is_m31_output
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;

    // Word-path, register-resident (L3 change 2): absorb each column value
    // directly as a message word. For gate_air, number_of_columns = 22 words = 88
    // bytes = one full block (16 words, t+=64) + a 6-word tail. Feeding words is
    // bit-identical to the LE-byte split + reassembly (compress_words note).
    Blake2sWordHasher hasher;
    hasher.init();

    // No LEAF_PREFIX — lifted hasher has no domain separation. Column order is
    // load-bearing: it is the caller's `columns` slice order (already sorted).
    for (int col = 0; col < number_of_columns; ++col) {
        hasher.absorb(data[col][index]);
    }
    hasher.finalize(&result[index]);

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
    int block_dim = BLAKE2S_LIFTED_BLK;
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