#include "cuda_mem_pool.cuh"
#include <cuda_runtime.h>

// Global memory pool instance
cudaMemPool_t g_mem_pool = nullptr;
bool g_mem_pool_initialized = false;

extern "C" cudaError_t cuda_mem_pool_init() {
    if (g_mem_pool_initialized) {
        return cudaSuccess;
    }

    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        return err;
    }

    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = device_id;

    err = cudaMemPoolCreate(&g_mem_pool, &pool_props);
    if (err != cudaSuccess) {
        printf("Failed to create memory pool: %s\n", cudaGetErrorString(err));
        return err;
    }

    uint64_t threshold = UINT64_MAX;
    err = cudaMemPoolSetAttribute(g_mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    if (err != cudaSuccess) {
        printf("Failed to set memory pool threshold: %s\n", cudaGetErrorString(err));
        cudaMemPoolDestroy(g_mem_pool);
        g_mem_pool = nullptr;
        return err;
    }

    g_mem_pool_initialized = true;
    return cudaSuccess;
}

extern "C" cudaError_t cuda_mem_pool_destroy() {
    if (!g_mem_pool_initialized || g_mem_pool == nullptr) {
        return cudaSuccess;
    }

    cudaError_t err = cudaMemPoolDestroy(g_mem_pool);
    if (err == cudaSuccess) {
        g_mem_pool = nullptr;
        g_mem_pool_initialized = false;
    }

    return err;
}

extern "C" uint32_t* cuda_mem_pool_allocate_uint32(size_t count) {
    return cuda_mem_pool_allocate<uint32_t>(count);
}

extern "C" uint32_t* cuda_mem_pool_allocate_zeroes_uint32(size_t count) {
    return cuda_mem_pool_allocate_zeroes<uint32_t>(count);
}

extern "C" void cuda_mem_pool_free_uint32(uint32_t* ptr) {
    cuda_mem_pool_free(ptr);
}