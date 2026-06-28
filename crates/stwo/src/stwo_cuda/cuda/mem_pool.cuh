#ifndef MEM_POOL_H
#define MEM_POOL_H

// DEPRECATED: This file is deprecated. Please use cuda_mem_pool.cuh instead.
// The new implementation uses CUDA's built-in memory pool (cudaMemPool_t) for better performance.

#warning "mem_pool.cuh is deprecated. Please use cuda_mem_pool.cuh instead."

#include <cstdint>

// External C functions that will call the Rust memory pool
extern "C" {
    // Allocate memory from the pool
    uint32_t* pool_allocate_cuda(size_t size);
    
    // Deallocate memory back to the pool
    void pool_deallocate_cuda(uint32_t* ptr, size_t size);
    
    // Allocate zeroed memory from the pool
    uint32_t* pool_allocate_zeroes_cuda(size_t size);
}

// Template wrapper for typed allocations
template<typename T>
T* pool_malloc(size_t count) {
    // Calculate size in terms of uint32_t elements
    size_t size_in_bytes = sizeof(T) * count;
    size_t size_in_uint32 = (size_in_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    
    return reinterpret_cast<T*>(pool_allocate_cuda(size_in_uint32));
}

// Template wrapper for typed deallocations
template<typename T>
void pool_free(T* ptr, size_t count) {
    if (ptr == nullptr) return;
    
    // Calculate size in terms of uint32_t elements
    size_t size_in_bytes = sizeof(T) * count;
    size_t size_in_uint32 = (size_in_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    
    pool_deallocate_cuda(reinterpret_cast<uint32_t*>(ptr), size_in_uint32);
}

// Template wrapper for zeroed allocations
template<typename T>
T* pool_malloc_zeroes(size_t count) {
    // Calculate size in terms of uint32_t elements
    size_t size_in_bytes = sizeof(T) * count;
    size_t size_in_uint32 = (size_in_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    
    return reinterpret_cast<T*>(pool_allocate_zeroes_cuda(size_in_uint32));
}

#endif // MEM_POOL_H