#ifndef CUDA_MEM_POOL_H
#define CUDA_MEM_POOL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Global memory pool handle
extern cudaMemPool_t g_mem_pool;
extern bool g_mem_pool_initialized;

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
                printf("Failed to initialize memory pool: %s\n", cudaGetErrorString(init_err));
                printf("Also failed to fallback cudaMalloc(%zu): %s\n", size, cudaGetErrorString(aerr));
                return nullptr;
            }
            return ptr;
        }
    }

    // Allocate from pool; on failure, fallback to cudaMalloc
    cudaError_t err = cudaMallocFromPoolAsync((void**)&ptr, size, g_mem_pool, 0);
    if (err != cudaSuccess) {
        printf("Failed to allocate %zu bytes from pool: %s\n", size, cudaGetErrorString(err));
        // Fallback
        cudaError_t aerr = cudaMalloc((void**)&ptr, size);
        if (aerr != cudaSuccess) {
            printf("Also failed to fallback cudaMalloc(%zu): %s\n", size, cudaGetErrorString(aerr));
            return nullptr;
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
