#include "cuda_mem_pool.cuh"
#include <cuda_runtime.h>
#include <mutex>

// PER-DEVICE memory pool table (in-process multi-GPU base proving, "option A").
//
// Previously a single process-global `g_mem_pool` / `g_mem_pool_initialized` pair bound the whole
// backend to ONE device (the first thread's current device = default device 0). A CUDA memory pool
// is bound to exactly one device (`pool_props.location.id`), so `cudaMallocFromPoolAsync(...)` from
// a thread whose current device differs from the pool's device is a cross-device error. To prove
// base shards concurrently on GPUs 0..7 in one process (one host thread per GPU, each having done
// `cudaSetDevice(n)` once), the pool must be PER DEVICE.
//
// Shape: a fixed-size table keyed by the runtime current device ordinal (`cudaGetDevice`). Every
// alloc/free indexes `g_mem_pool_table[cudaGetDevice()]` — the caller thread has already selected
// its device, so the current-device read returns that ordinal with no extra plumbing and no
// signature changes. A mutex guards ONLY the lazy per-device create (the check-then-create was
// previously unguarded; two threads racing the first alloc on the SAME device would double-create).
//
// BYTE-IDENTITY (N=1): with a single thread on device 0, `cudaGetDevice()` returns 0, every access
// indexes slot [0], and behavior is identical to the old single `g_mem_pool`. `USE_CUDA_MEM_POOL`
// semantics are unchanged. This is a pure execution-plumbing change (WHERE bytes live), never WHAT
// is committed.
cudaMemPool_t g_mem_pool_table[MAX_CUDA_DEVICES] = {nullptr};
bool g_mem_pool_initialized_table[MAX_CUDA_DEVICES] = {false};
static std::mutex g_mem_pool_mtx; // guards the lazy per-device init only

// Current-device ordinal, clamped to the table. On any error selects slot 0 (matches the old
// single-pool behavior on a one-device box).
int cuda_mem_pool_current_device() {
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess || device_id < 0 || device_id >= MAX_CUDA_DEVICES) {
        return 0;
    }
    return device_id;
}

extern "C" cudaError_t cuda_mem_pool_init() {
    int device_id = cuda_mem_pool_current_device();

    // Fast path: already initialized for this device (no lock).
    if (g_mem_pool_initialized_table[device_id]) {
        return cudaSuccess;
    }

    std::lock_guard<std::mutex> lock(g_mem_pool_mtx);
    // Re-check under the lock (another thread may have created it for this device meanwhile).
    if (g_mem_pool_initialized_table[device_id]) {
        return cudaSuccess;
    }

    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = device_id;

    cudaMemPool_t pool = nullptr;
    cudaError_t err = cudaMemPoolCreate(&pool, &pool_props);
    if (err != cudaSuccess) {
        printf("Failed to create memory pool for device %d: %s\n", device_id, cudaGetErrorString(err));
        return err;
    }

    uint64_t threshold = UINT64_MAX;
    err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    if (err != cudaSuccess) {
        printf("Failed to set memory pool threshold for device %d: %s\n", device_id, cudaGetErrorString(err));
        cudaMemPoolDestroy(pool);
        return err;
    }

    g_mem_pool_table[device_id] = pool;
    g_mem_pool_initialized_table[device_id] = true;
    return cudaSuccess;
}

extern "C" cudaError_t cuda_mem_pool_destroy() {
    // Destroy every initialized per-device pool. Under the lock so it does not race a concurrent
    // lazy init (in practice destroy runs at process teardown with no provers in flight).
    std::lock_guard<std::mutex> lock(g_mem_pool_mtx);
    cudaError_t first_err = cudaSuccess;
    for (int d = 0; d < MAX_CUDA_DEVICES; ++d) {
        if (g_mem_pool_initialized_table[d] && g_mem_pool_table[d] != nullptr) {
            cudaError_t err = cudaMemPoolDestroy(g_mem_pool_table[d]);
            if (err == cudaSuccess) {
                g_mem_pool_table[d] = nullptr;
                g_mem_pool_initialized_table[d] = false;
            } else if (first_err == cudaSuccess) {
                first_err = err;
            }
        }
    }
    return first_err;
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
