#ifndef CUDA_MEM_POOL_H
#define CUDA_MEM_POOL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Max CUDA devices supported for in-process multi-GPU base proving ("option A"). GPUs 0..7 on the
// target box; sized generously.
#ifndef MAX_CUDA_DEVICES
#define MAX_CUDA_DEVICES 16
#endif

// PER-DEVICE memory pool table (see cuda_mem_pool.cu). Keyed by the runtime current device ordinal.
extern cudaMemPool_t g_mem_pool_table[MAX_CUDA_DEVICES];
extern bool g_mem_pool_initialized_table[MAX_CUDA_DEVICES];

// Current-device ordinal (clamped to [0, MAX_CUDA_DEVICES); returns 0 on error).
int cuda_mem_pool_current_device();

// Alias the historical single-pool names to the CURRENT DEVICE's table slot so every existing
// reference (`g_mem_pool` / `g_mem_pool_initialized`, read or written, in this header's templates
// and in utils.cu) resolves to the calling thread's device with NO call-site changes. For one
// thread on device 0 this is `[0]` => byte-identical to the previous single-pool behavior.
#define g_mem_pool             (g_mem_pool_table[cuda_mem_pool_current_device()])
#define g_mem_pool_initialized (g_mem_pool_initialized_table[cuda_mem_pool_current_device()])

// Initialize the CUDA memory pool
extern "C" cudaError_t cuda_mem_pool_init();

// Destroy the CUDA memory pool
extern "C" cudaError_t cuda_mem_pool_destroy();

// Allocate memory from the pool (with safe fallback to cudaMalloc when pool is unavailable)
template<typename T>
T* cuda_mem_pool_allocate(size_t count) {
    T* ptr = nullptr;
    size_t size = sizeof(T) * count;
    
    // Try initialize memory pool once
    if (!g_mem_pool_initialized) {
        cudaError_t init_err = cuda_mem_pool_init();
        if (init_err != cudaSuccess) {
            // Fallback to standard cudaMalloc when pool is unsupported/unavailable
            // (e.g., old drivers, restricted environments)
            cudaError_t aerr = cudaMalloc((void**)&ptr, size);
            if (aerr != cudaSuccess) {
                fprintf(stderr, "FATAL: failed to initialize memory pool: %s\n", cudaGetErrorString(init_err));
                fprintf(stderr, "FATAL: OOM allocating %zu bytes (fallback cudaMalloc) at %s:%d: %s\n",
                        size, __FILE__, __LINE__, cudaGetErrorString(aerr));
                exit(1);
            }
            return ptr;
        }
    }

    // Allocate from pool; on failure, fallback to cudaMalloc
    cudaError_t err = cudaMallocFromPoolAsync((void**)&ptr, size, g_mem_pool, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate %zu bytes from pool: %s\n", size, cudaGetErrorString(err));
        // Fallback
        cudaError_t aerr = cudaMalloc((void**)&ptr, size);
        if (aerr != cudaSuccess) {
            fprintf(stderr, "FATAL: OOM allocating %zu bytes (pool + fallback cudaMalloc) at %s:%d: %s\n",
                    size, __FILE__, __LINE__, cudaGetErrorString(aerr));
            exit(1);
        }
        return ptr;
    }

    // No sync needed: cudaMallocFromPoolAsync on stream 0 is ordered with
    // subsequent kernel launches on the same stream. The pointer is valid for
    // any operation enqueued after this call on stream 0.

    return ptr;
}

// Allocate zeroed memory from the pool
template<typename T>
T* cuda_mem_pool_allocate_zeroes(size_t count) {
    T* ptr = cuda_mem_pool_allocate<T>(count);
    if (ptr != nullptr) {
        cudaMemsetAsync(ptr, 0, sizeof(T) * count, 0);
        // No sync needed: memset is ordered on stream 0 with subsequent operations.
    }
    return ptr;
}

// Free memory back to the pool (fallback aware)
template<typename T>
void cuda_mem_pool_free(T* ptr) {
    if (ptr != nullptr) {
        if (!g_mem_pool_initialized || g_mem_pool == nullptr) {
            cudaFree(ptr);
        } else {
            // Async free: stream ordering guarantees the memory won't be reused
            // until all prior operations on stream 0 have completed.
            cudaFreeAsync(ptr, 0);
        }
    }
}

// C-style wrappers for specific types
extern "C" uint32_t* cuda_mem_pool_allocate_uint32(size_t count);
extern "C" uint32_t* cuda_mem_pool_allocate_zeroes_uint32(size_t count);
extern "C" void cuda_mem_pool_free_uint32(uint32_t* ptr);

#endif // CUDA_MEM_POOL_H
