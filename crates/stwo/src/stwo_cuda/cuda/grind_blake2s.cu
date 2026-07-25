#include "grind_blake2s.cuh"
#include "utils.cuh"

// Blake2s IV constants (same as in blake2s.cu, but we need our own copy
// since __device__ __constant__ variables have translation-unit scope)
static __device__ __constant__ uint32_t grind_blake2s_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static __device__ __constant__ uint8_t grind_blake2s_sigma[10][16] = {
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

#define GRIND_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define GRIND_G(r,i,a,b,c,d) \
    do { \
        a = a + b + m[grind_blake2s_sigma[r][2*i+0]]; \
        d = GRIND_ROTR32(d ^ a, 16); \
        c = c + d; \
        b = GRIND_ROTR32(b ^ c, 12); \
        a = a + b + m[grind_blake2s_sigma[r][2*i+1]]; \
        d = GRIND_ROTR32(d ^ a, 8); \
        c = c + d; \
        b = GRIND_ROTR32(b ^ c, 7); \
    } while(0)

// Keep the CUDA nonce search identical to the SIMD implementation:
//   nonce = (hi << 32) | low
// where 0 <= low < 2^20 and hi grows chunk-by-chunk.
static constexpr uint32_t GRIND_LOW_BITS = 20;
static constexpr uint32_t GRIND_LOW_RANGE = 1u << GRIND_LOW_BITS;
static constexpr uint32_t M31_P = 0x7FFFFFFF;

// Each thread tests one "low" nonce candidate in the current chunk.
// The full nonce is ((uint64_t)hi << 32) | low.
__global__ void grind_blake2s_kernel(
    const uint32_t* __restrict__ prefixed_digest,
    uint32_t pow_bits,
    uint32_t hi,
    unsigned long long* result_low
) {
    uint32_t low = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (low >= GRIND_LOW_RANGE) return;

    uint64_t nonce = ((uint64_t)hi << 32) | low;

    // Early exit within the current hi-chunk: if a smaller low part was already found.
    if ((unsigned long long)low >= *result_low) return;

    // Build 16-word message block:
    // m[0..7]  = prefixed_digest (32 bytes)
    // m[8]     = nonce low 32 bits
    // m[9]     = nonce high 32 bits
    // m[10..15] = 0 (padding)
    // Total input length = 40 bytes
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) m[i] = prefixed_digest[i];
    m[8]  = (uint32_t)(nonce);
    m[9]  = (uint32_t)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 16; i++) m[i] = 0;

    // Initialize Blake2s state for a single-block hash (32-byte output):
    //   h[0] = IV[0] ^ 0x01010020 (fan-out=1, depth=1, digest_length=32)
    //   h[1..7] = IV[1..7]
    // Initialize v-vector for compression:
    //   v[0..7] = h[0..7]
    //   v[8..15] = IV[0..7]
    //   v[12] ^= 40     (total bytes counter, low word)
    //   v[14] ^= 0xFFFFFFFF  (last block flag)
    uint32_t v[16];
    v[0] = grind_blake2s_IV[0] ^ 0x01010020;
    #pragma unroll
    for (int i = 1; i < 8; i++) v[i] = grind_blake2s_IV[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i + 8] = grind_blake2s_IV[i];
    v[12] ^= 40;          // t0 = 40 bytes total
    v[14] ^= 0xFFFFFFFF;  // last block

    // 10 rounds of Blake2s mixing
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        GRIND_G(r, 0, v[0], v[4], v[8],  v[12]);
        GRIND_G(r, 1, v[1], v[5], v[9],  v[13]);
        GRIND_G(r, 2, v[2], v[6], v[10], v[14]);
        GRIND_G(r, 3, v[3], v[7], v[11], v[15]);
        GRIND_G(r, 4, v[0], v[5], v[10], v[15]);
        GRIND_G(r, 5, v[1], v[6], v[11], v[12]);
        GRIND_G(r, 6, v[2], v[7], v[8],  v[13]);
        GRIND_G(r, 7, v[3], v[4], v[9],  v[14]);
    }

    // Finalize: h[i] = h[i] ^ v[i] ^ v[i+8], but we only need h[0]
    uint32_t h0 = (grind_blake2s_IV[0] ^ 0x01010020) ^ v[0] ^ v[8];

    // Count trailing zeros of h0
    // __clz counts leading zeros, __brev reverses bits
    // For h0 == 0, __clz returns 32, which means 32 trailing zeros
    uint32_t tz = (h0 == 0) ? 32 : __clz(__brev(h0));

    if (tz >= pow_bits) {
        atomicMin(result_low, (unsigned long long)low);
    }
}

// Host function: launches the GPU grind kernel chunk-by-chunk until a valid nonce is found.
// The search order and returned nonce now match the SIMD implementation.
uint64_t grind_blake2s(const uint32_t* host_prefixed_digest, uint32_t pow_bits) {
    // Allocate device memory for prefixed_digest (8 x u32)
    uint32_t* d_prefixed_digest;
    d_prefixed_digest = cuda_mem_pool_allocate<uint32_t>(8);
    ASSERT_CUDA_SUCCESS(cudaMemcpy(d_prefixed_digest, host_prefixed_digest,
                                   8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate device memory for the smallest successful low-part in the current chunk.
    unsigned long long* d_result_low;
    d_result_low = cuda_mem_pool_allocate<unsigned long long>(1);

    // Kernel launch parameters
    const int block_size = 256;
    const int grid_size = 4096;  // 4096 * 256 = 2^20 low values per hi chunk
    static_assert((uint32_t)(block_size * grid_size) == GRIND_LOW_RANGE,
                  "CUDA grind launch shape must match the SIMD low-range search space");

    unsigned long long host_result_low = UINT64_MAX;
    uint64_t host_result = UINT64_MAX;

    for (uint32_t hi = 0; hi < M31_P; ++hi) {
        unsigned long long init_val = UINT64_MAX;
        ASSERT_CUDA_SUCCESS(cudaMemcpy(d_result_low, &init_val,
                                       sizeof(unsigned long long), cudaMemcpyHostToDevice));
        grind_blake2s_kernel<<<grid_size, block_size>>>(
            d_prefixed_digest, pow_bits, hi, d_result_low
        );
        ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // Check if a valid nonce was found in this hi chunk.
        ASSERT_CUDA_SUCCESS(cudaMemcpy(&host_result_low, d_result_low,
                                       sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (host_result_low != UINT64_MAX) {
            host_result = ((uint64_t)hi << 32) | (uint64_t)host_result_low;
            break;
        }
    }

    cuda_mem_pool_free(d_prefixed_digest);
    cuda_mem_pool_free(d_result_low);

    if (host_result == UINT64_MAX) {
        fprintf(stderr, "CUDA grind failed to find a valid nonce\n");
        exit(1);
    }

    return host_result;
}
