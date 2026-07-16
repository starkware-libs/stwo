#include "utils.cuh"
#include "cuda_mem_pool.cuh"

#include <cstdio>

// Must match the definition in utils.cuh
#define USE_CUDA_MEM_POOL 1


__host__ int log_2(int value) {
    return __builtin_ctz(value);
}

void copy_uint32_t_vec_from_device_to_host(uint32_t *device_ptr, uint32_t *host_ptr, int size) {
    cuda_mem_copy_device_to_host<uint32_t>(device_ptr, host_ptr, size);
}

uint32_t* copy_uint32_t_vec_from_host_to_device(uint32_t *host_ptr, int size) {
    uint32_t* device_ptr = cuda_malloc<uint32_t>(size);
    cudaMemsetAsync(device_ptr, 0x00, sizeof(uint32_t) * size, 0);
    cuda_mem_copy_host_to_device(host_ptr, device_ptr, size);
    return device_ptr;
}

void copy_uint32_t_vec_from_device_to_device(uint32_t *from, uint32_t *dst, int size) {
    cuda_mem_copy_device_to_device<uint32_t>(from, dst, size);
}

void copy_uint32_t_vec_from_device_to_device_offset(uint32_t *from, uint32_t *dst, int size, int offset) {
    cuda_mem_copy_device_to_device<uint32_t>(from, dst + offset, size);
}

uint32_t* cuda_malloc_uint32_t(size_t size) {
#if USE_CUDA_MEM_POOL
    uint32_t* device_ptr = cuda_mem_pool_allocate<uint32_t>(size);
    if (device_ptr != nullptr) {
        cudaMemsetAsync(device_ptr, 0x00, sizeof(uint32_t) * size, 0);
    }
    return device_ptr;
#else
    uint32_t* device_ptr = cuda_malloc<uint32_t>(size);
    cudaMemsetAsync(device_ptr, 0x00, sizeof(uint32_t) * size, 0);
    return device_ptr;
#endif
}

Blake2sHash* cuda_malloc_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_malloc<Blake2sHash>(size);
    // cudaMemset(device_ptr, 0x00, sizeof(Blake2sHash) * size);
    return device_ptr;
}

__global__ void print_array(uint32_t *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) {
        printf("%d, ", array[idx]);
    }
}

uint32_t* cuda_alloc_zeroes_uint32_t(size_t size) {
#if USE_CUDA_MEM_POOL
    return cuda_mem_pool_allocate_zeroes<uint32_t>(size);
#else
    uint32_t* device_ptr = cuda_malloc_uint32_t(size);
    cudaMemsetAsync(device_ptr, 0x00, sizeof(uint32_t) * size, 0);
    return device_ptr;
#endif
}

void cuda_set_uint32_t(uint32_t *device_ptr, size_t index, uint32_t value) {
    cuda_mem_copy_host_to_device<uint32_t>(&value, device_ptr + index, 1);
}

// Single-element atomicAdd kernel — replaces GPU→CPU→GPU roundtrip.
__global__ void increase_at_kernel(uint32_t *ptr, uint32_t address) {
    atomicAdd(ptr + address, 1);
}

void cuda_increase_at(uint32_t *device_ptr, uint32_t address) {
    increase_at_kernel<<<1, 1>>>(device_ptr, address);
    // No sync needed — sequential kernel launches are ordered on default stream.
}

uint32_t cuda_get_uint32_t(uint32_t *device_ptr, size_t index) {
    uint32_t value = 0x0;
    cuda_mem_copy_device_to_host<uint32_t>(device_ptr + index, &value, 1);
    return value;
}

qm31 cuda_get_secure_field(qm31 *device_ptr, size_t index) {
    qm31 value = {};
    cuda_mem_copy_device_to_host<qm31>(device_ptr + index, &value, 1);
    return value;
}

Blake2sHash* cuda_alloc_zeroes_blake_2s_hash(int size) {
    Blake2sHash* device_ptr = cuda_malloc_blake_2s_hash(size);
    cudaMemsetAsync(device_ptr, 0x00, sizeof(uint32_t) * size, 0);
    return device_ptr;
}

Blake2sHash* copy_blake_2s_hash_vec_from_host_to_device(Blake2sHash *host_ptr, uint32_t size) {
    Blake2sHash* device_ptr = clone_to_device<Blake2sHash>(host_ptr, size);
    return device_ptr;
}

void cuda_get_blake_2s_hash(Blake2sHash *device_ptr, Blake2sHash *host_ptr, size_t index) {
    cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr + index, host_ptr, 1);
}

// Option B (pinned minimal-latency single-root read).
//
// The FRI-layer and tree-commit chains are intrinsically serial (each root is mixed to draw the
// challenge the next step consumes), so this does NOT parallelize anything — it only makes the
// single 32-byte root D2H cheaper. `cuda_get_blake_2s_hash` above routes through the pageable
// `cuda_mem_copy_device_to_host` (a cudaMemcpyAsync on the DEFAULT stream that, for pageable host
// memory, blocks the host until the copy completes). This variant instead copies into a PINNED
// staging buffer on a dedicated non-blocking COPY STREAM and synchronizes ONLY that stream, then
// memcpys the 32 bytes to the caller's host output. Lower launch/serialization latency, and the
// copy stream doesn't drag the default stream.
//
// The per-thread copy stream + 32-byte pinned staging buffer are created lazily and kept for the
// thread's lifetime. thread_local (not global) so, under the multi-GPU producer model, each commit
// thread's stream/buffer live on the device that thread is bound to when it first reads a root — a
// device-N reader never syncs device-M's stream. For the default single-GPU / single-thread path
// there is exactly one stream + one pinned buffer, created on device 0.
//
// ORDERING: the layer-build kernels that produce the root are enqueued on the DEFAULT stream (as in
// the pageable path). We record an event on the default stream and make the copy stream wait on it,
// so the D2H cannot run before the root bytes are computed — then sync the copy stream so the bytes
// have fully arrived before we return them. This preserves exactly what value is read (byte
// identical) and when it is available to the caller.
static thread_local cudaStream_t g_root_copy_stream = nullptr;
static thread_local Blake2sHash *g_root_pinned = nullptr;
static thread_local cudaEvent_t g_root_ready_event = nullptr;

void cuda_get_blake_2s_hash_pinned(Blake2sHash *device_ptr, Blake2sHash *host_ptr, size_t index) {
    if (g_root_copy_stream == nullptr) {
        // Dedicated non-blocking copy stream (does not implicitly sync the default stream).
        cudaError_t serr = cudaStreamCreateWithFlags(&g_root_copy_stream, cudaStreamNonBlocking);
        if (serr != cudaSuccess) {
            printf("cuda_get_blake_2s_hash_pinned: stream create failed: %s\n", cudaGetErrorString(serr));
            // Fail-safe: fall back to the pageable blocking read so we never return garbage.
            cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr + index, host_ptr, 1);
            return;
        }
    }
    if (g_root_pinned == nullptr) {
        cudaError_t herr = cudaHostAlloc((void**)&g_root_pinned, sizeof(Blake2sHash), cudaHostAllocDefault);
        if (herr != cudaSuccess || g_root_pinned == nullptr) {
            printf("cuda_get_blake_2s_hash_pinned: pinned alloc failed: %s\n", cudaGetErrorString(herr));
            cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr + index, host_ptr, 1);
            return;
        }
    }
    if (g_root_ready_event == nullptr) {
        cudaEventCreateWithFlags(&g_root_ready_event, cudaEventDisableTiming);
    }

    // Order the copy stream AFTER all default-stream work that produced the root, then copy on the
    // copy stream and sync only it.
    if (g_root_ready_event != nullptr) {
        cudaEventRecord(g_root_ready_event, 0);
        cudaStreamWaitEvent(g_root_copy_stream, g_root_ready_event, 0);
    }
    cudaMemcpyAsync(g_root_pinned, device_ptr + index, sizeof(Blake2sHash),
                    cudaMemcpyDeviceToHost, g_root_copy_stream);
    cudaStreamSynchronize(g_root_copy_stream);

    *host_ptr = *g_root_pinned;
}

// Kernel: Batch get Blake2s hashes from device memory by indices
__global__ void batch_get_blake2s_kernel(
    const Blake2sHash* src,
    Blake2sHash* dst,
    const uint32_t* indices,
    uint32_t n_indices
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_indices) {
        uint32_t src_idx = indices[idx];
        // Copy 32 bytes (Blake2sHash is 32 bytes)
        // Using uint4 for efficient 16-byte aligned memory access
        const uint4* src_ptr = reinterpret_cast<const uint4*>(&src[src_idx]);
        uint4* dst_ptr = reinterpret_cast<uint4*>(&dst[idx]);

        // Copy in two uint4 chunks (2 * 16 bytes = 32 bytes)
        dst_ptr[0] = src_ptr[0];
        dst_ptr[1] = src_ptr[1];
    }
}

// Host function: Batch get Blake2s hashes
void cuda_batch_get_blake_2s_hash(
    Blake2sHash *device_ptr,
    Blake2sHash *host_ptr,
    uint32_t *indices,
    uint32_t n_indices
) {
    if (n_indices == 0) {
        return;
    }

    // Use memory pool for faster allocation/deallocation
    // 1. Allocate GPU memory for indices array from pool
    uint32_t* d_indices = cuda_mem_pool_allocate<uint32_t>(n_indices);
    if (!d_indices) {
        printf("Failed to allocate indices buffer in batch_get\n");
        return;
    }

    // 2. Allocate GPU memory for result array from pool
    Blake2sHash* d_result = cuda_mem_pool_allocate<Blake2sHash>(n_indices);
    if (!d_result) {
        printf("Failed to allocate result buffer in batch_get\n");
        cuda_mem_pool_free(d_indices);
        return;
    }

    // 3. Copy indices to GPU asynchronously
    cudaMemcpyAsync(d_indices, indices, n_indices * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);

    // 4. Launch kernel to gather hashes in parallel
    const int block_size = 256;
    const int num_blocks = (n_indices + block_size - 1) / block_size;
    batch_get_blake2s_kernel<<<num_blocks, block_size, 0, 0>>>(
        device_ptr, d_result, d_indices, n_indices
    );

    // 5. Copy result back to CPU asynchronously
    cudaMemcpyAsync(host_ptr, d_result, n_indices * sizeof(Blake2sHash), cudaMemcpyDeviceToHost, 0);

    // 6. Synchronize stream to ensure all operations complete
    cudaStreamSynchronize(0);

    // 7. Free temporary GPU memory back to pool (fast)
    cuda_mem_pool_free(d_indices);
    cuda_mem_pool_free(d_result);
}

// Kernel: Batch get uint32_t values from device memory by indices
__global__ void batch_get_uint32_kernel(
    const uint32_t* src,
    uint32_t* dst,
    const uint32_t* indices,
    uint32_t n_indices
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_indices) {
        dst[idx] = src[indices[idx]];
    }
}

// Host function: Batch get uint32_t values
void cuda_batch_get_uint32_t(
    uint32_t *device_ptr,
    uint32_t *host_ptr,
    uint32_t *indices,
    uint32_t n_indices
) {
    if (n_indices == 0) {
        return;
    }

    // 1. Allocate GPU memory for indices array from pool
    uint32_t* d_indices = cuda_mem_pool_allocate<uint32_t>(n_indices);
    if (!d_indices) {
        printf("Failed to allocate indices buffer in batch_get_uint32\n");
        return;
    }

    // 2. Allocate GPU memory for result array from pool
    uint32_t* d_result = cuda_mem_pool_allocate<uint32_t>(n_indices);
    if (!d_result) {
        printf("Failed to allocate result buffer in batch_get_uint32\n");
        cuda_mem_pool_free(d_indices);
        return;
    }

    // 3. Copy indices to GPU asynchronously
    cudaMemcpyAsync(d_indices, indices, n_indices * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);

    // 4. Launch kernel to gather values in parallel
    const int block_size = 256;
    const int num_blocks = (n_indices + block_size - 1) / block_size;
    batch_get_uint32_kernel<<<num_blocks, block_size, 0, 0>>>(
        device_ptr, d_result, d_indices, n_indices
    );

    // 5. Copy result back to CPU asynchronously
    cudaMemcpyAsync(host_ptr, d_result, n_indices * sizeof(uint32_t), cudaMemcpyDeviceToHost, 0);

    // 6. Synchronize stream to ensure all operations complete
    cudaStreamSynchronize(0);

    // 7. Free temporary GPU memory back to pool
    cuda_mem_pool_free(d_indices);
    cuda_mem_pool_free(d_result);
}

// Multi-column batch gather: fetch the same indices from multiple columns in one kernel.
// Output layout is row-major: dst[idx * n_columns + col] = columns[col][indices[idx]]
__global__ void batch_gather_multi_uint32_kernel(
    const uint32_t* const* src_ptrs,
    uint32_t* dst,
    const uint32_t* indices,
    uint32_t n_indices,
    uint32_t n_columns
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n_indices * n_columns;
    if (tid >= total) return;

    uint32_t idx = tid / n_columns;
    uint32_t col = tid % n_columns;
    dst[tid] = src_ptrs[col][indices[idx]];
}

void cuda_batch_gather_multi_uint32(
    const uint32_t** column_device_ptrs,
    uint32_t n_columns,
    const uint32_t* host_indices,
    uint32_t n_indices,
    uint32_t* host_output
) {
    if (n_indices == 0 || n_columns == 0) return;

    uint32_t total = n_indices * n_columns;

    // Upload column pointers array to device.
    const uint32_t** d_col_ptrs = cuda_mem_pool_allocate<const uint32_t*>(n_columns);
    cudaMemcpyAsync((void*)d_col_ptrs, column_device_ptrs,
                    n_columns * sizeof(uint32_t*), cudaMemcpyHostToDevice, 0);

    // Upload indices to device.
    uint32_t* d_indices = cuda_mem_pool_allocate<uint32_t>(n_indices);
    cudaMemcpyAsync(d_indices, host_indices,
                    n_indices * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);

    // Allocate output on device.
    uint32_t* d_result = cuda_mem_pool_allocate<uint32_t>(total);

    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    batch_gather_multi_uint32_kernel<<<num_blocks, block_size, 0, 0>>>(
        d_col_ptrs, d_result, d_indices, n_indices, n_columns
    );

    cudaMemcpyAsync(host_output, d_result,
                    total * sizeof(uint32_t), cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);

    cuda_mem_pool_free((void*)d_col_ptrs);
    cuda_mem_pool_free(d_indices);
    cuda_mem_pool_free(d_result);
}

// Multi-layer batch get kernel
__global__ void multi_layer_batch_get_kernel(
    const Blake2sHash* const* layer_ptrs,  // Array of layer device pointers
    Blake2sHash* dst,
    const LayerIndexPair* pairs,
    uint32_t n_pairs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_pairs) {
        LayerIndexPair pair = pairs[idx];
        const Blake2sHash* src_layer = layer_ptrs[pair.layer_idx];

        // Copy 32 bytes using uint4 for efficient aligned access
        const uint4* src_ptr = reinterpret_cast<const uint4*>(&src_layer[pair.hash_idx]);
        uint4* dst_ptr = reinterpret_cast<uint4*>(&dst[idx]);

        // 2 * uint4 = 2 * 16 bytes = 32 bytes = 1 Blake2sHash
        dst_ptr[0] = src_ptr[0];
        dst_ptr[1] = src_ptr[1];
    }
}

// Multi-layer batch get host function
void cuda_multi_layer_batch_get_blake_2s_hash(
    const Blake2sHash **layer_device_ptrs,
    Blake2sHash *host_ptr,
    const LayerIndexPair *pairs,
    uint32_t n_pairs
) {
    if (n_pairs == 0) {
        return;
    }

    // 1. Allocate GPU memory for layer pointers array
    const Blake2sHash** d_layer_ptrs = cuda_mem_pool_allocate<const Blake2sHash*>(24);  // Max 24 layers
    if (!d_layer_ptrs) {
        printf("Failed to allocate layer pointers in multi_layer_batch_get\n");
        return;
    }

    // 2. Allocate GPU memory for pairs array
    LayerIndexPair* d_pairs = cuda_mem_pool_allocate<LayerIndexPair>(n_pairs);
    if (!d_pairs) {
        printf("Failed to allocate pairs buffer in multi_layer_batch_get\n");
        cuda_mem_pool_free((void*)d_layer_ptrs);
        return;
    }

    // 3. Allocate GPU memory for result array
    Blake2sHash* d_result = cuda_mem_pool_allocate<Blake2sHash>(n_pairs);
    if (!d_result) {
        printf("Failed to allocate result buffer in multi_layer_batch_get\n");
        cuda_mem_pool_free((void*)d_layer_ptrs);
        cuda_mem_pool_free(d_pairs);
        return;
    }

    // 4. Copy layer pointers and pairs to GPU asynchronously
    cudaMemcpyAsync((void*)d_layer_ptrs, layer_device_ptrs, 24 * sizeof(Blake2sHash*), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_pairs, pairs, n_pairs * sizeof(LayerIndexPair), cudaMemcpyHostToDevice, 0);

    // 5. Launch kernel to gather hashes in parallel from multiple layers
    const int block_size = 256;
    const int num_blocks = (n_pairs + block_size - 1) / block_size;
    multi_layer_batch_get_kernel<<<num_blocks, block_size, 0, 0>>>(
        d_layer_ptrs, d_result, d_pairs, n_pairs
    );

    // 6. Copy result back to CPU asynchronously
    cudaMemcpyAsync(host_ptr, d_result, n_pairs * sizeof(Blake2sHash), cudaMemcpyDeviceToHost, 0);

    // 7. Synchronize stream
    cudaStreamSynchronize(0);

    // 8. Free temporary GPU memory
    cuda_mem_pool_free((void*)d_layer_ptrs);
    cuda_mem_pool_free(d_pairs);
    cuda_mem_pool_free(d_result);
}

void copy_blake_2s_hash_vec_from_device_to_host(Blake2sHash *device_ptr, Blake2sHash *host_ptr, uint32_t size) {
    cuda_mem_copy_device_to_host<Blake2sHash>(device_ptr, host_ptr, size);
}

void copy_blake_2s_hash_vec_from_device_to_device(Blake2sHash *from, Blake2sHash *dst, int size) {
    cuda_mem_copy_device_to_device<Blake2sHash>(from, dst, size);
}

uint32_t** copy_device_pointer_vec_from_host_to_device(uint32_t** host_ptr, uint32_t size) {
    uint32_t** device_ptr = clone_to_device<uint32_t*>(host_ptr, size);
    return device_ptr;
}

// void** copy_device_pointer_vec_from_host_to_device(const void** ptrs, size_t n) {
//     void** d_ptrs;
//     cudaMalloc(&d_ptrs, n * sizeof(void*));
//     cudaMemcpy(d_ptrs, ptrs, n * sizeof(void*), cudaMemcpyHostToDevice);
//     return d_ptrs;
// }

void cuda_free_memory(void *device_ptr) {
#if USE_CUDA_MEM_POOL
    cuda_mem_pool_free(device_ptr);
#else
    cudaError_t err = cudaFree(device_ptr);
    if (err != cudaSuccess) {
        printf("Error freeing memory: %s\n", cudaGetErrorString(err));
    }
#endif
}

// A ONE-SHOT pool defrag at a shard boundary (GATE_AIR_POOL_TRIM, resident multi-shard path):
// cudaStreamSynchronize(0) drains the default stream so any deferred cudaFreeAsync has completed and
// its block is trimmable, then cudaMemPoolTrimTo releases ALL cached (already-freed) segments back
// to the OS so the next shard starts from a clean pool.
//
// CORRECTNESS: cudaMemPoolTrimTo only returns segments that are already FREE (freed + drained) to
// the OS; it never touches a LIVE allocation (d_cols, tree0, twiddles are untouched). Byte-identical
// proof, no working-set change. Guarded on pool init; no-op on the cudaMalloc-fallback path.
extern "C" void cuda_pool_trim() {
    cudaStreamSynchronize(0);
    if (g_mem_pool_initialized && g_mem_pool != nullptr) {
        cudaMemPoolTrimTo(g_mem_pool, 0);
    }
}

// MULTI-GPU ("option A"): bind the CALLING host thread to CUDA device `ordinal`. The CUDA runtime
// current-device is per-host-thread; a producer thread proving base shards on GPU n calls this ONCE
// at startup so every subsequent runtime-API alloc/kernel/commit on that thread targets device n
// (the per-device mem pool then indexes slot n). The default/single-GPU path never calls this, so
// the current device stays 0 => byte-identical. Returns the number of devices seen so the caller can
// validate the requested count against the box.
extern "C" int cuda_set_device(int ordinal) {
    cudaError_t err = cudaSetDevice(ordinal);
    if (err != cudaSuccess) {
        printf("cuda_set_device(%d) failed: %s\n", ordinal, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// MULTI-GPU: number of visible CUDA devices (respecting CUDA_VISIBLE_DEVICES). The harness uses this
// to clamp/validate the GATE_AIR_BASE_GPUS knob. Returns 0 on error.
extern "C" int cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

// M31 modular add offset: data[i] = add(data[i], offset) for all i < n.
__global__ void m31_vector_add_offset_kernel(m31 *data, unsigned int n, m31 offset) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = add(data[idx], offset);
    }
}

extern "C" void m31_vector_add_offset(uint32_t *data, unsigned int n, uint32_t offset) {
    if (n == 0) return;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    m31 offset_m31 = {offset};
    m31_vector_add_offset_kernel<<<num_blocks, block_size>>>((m31*)data, n, offset_m31);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("m31_vector_add_offset error: %s\n", cudaGetErrorString(err));
    }
}

// Pad GPU array by cycling: data[idx] = data[idx % cycle_len] for idx in [actual_size, padded_size).
__global__ void pad_with_cycle_kernel(uint32_t *data, unsigned int actual_size, unsigned int padded_size, unsigned int cycle_len) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + actual_size;
    if (idx < padded_size) {
        data[idx] = data[idx % cycle_len];
    }
}

extern "C" void pad_with_cycle(uint32_t *data, unsigned int actual_size, unsigned int padded_size, unsigned int cycle_len) {
    unsigned int count = padded_size - actual_size;
    if (count == 0) return;
    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    pad_with_cycle_kernel<<<num_blocks, block_size>>>(data, actual_size, padded_size, cycle_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("pad_with_cycle error: %s\n", cudaGetErrorString(err));
    }
}

// Fill GPU array with zeros: data[idx] = 0 for idx in [start, end).
__global__ void fill_zero_from_kernel(uint32_t *data, unsigned int start, unsigned int end) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end) {
        data[idx] = 0;
    }
}

extern "C" void fill_zero_from(uint32_t *data, unsigned int start, unsigned int end) {
    unsigned int count = end - start;
    if (count == 0) return;
    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    fill_zero_from_kernel<<<num_blocks, block_size>>>(data, start, end);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fill_zero_from error: %s\n", cudaGetErrorString(err));
    }
}

// Vector add in-place: dst[i] += src[i] for i in [0, n).
__global__ void vector_add_u32_kernel(uint32_t *dst, const uint32_t *src, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

extern "C" void vector_add_u32(uint32_t *dst, const uint32_t *src, unsigned int n) {
    if (n == 0) return;
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    vector_add_u32_kernel<<<num_blocks, block_size>>>(dst, src, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("vector_add_u32 error: %s\n", cudaGetErrorString(err));
    }
}

// GPU histogram via binary search: for each input value, find its index in a
// sorted key array and atomicAdd the corresponding multiplicity counter.
__global__ void histogram_by_binary_search_kernel(
    const uint32_t* input_values,
    uint32_t n_inputs,
    const uint32_t* sorted_keys,
    uint32_t n_keys,
    uint32_t* mults
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_inputs) return;
    uint32_t val = input_values[idx];
    // Binary search in sorted_keys
    int lo = 0, hi = (int)n_keys - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t key = sorted_keys[mid];
        if (key == val) {
            atomicAdd(mults + mid, 1);
            return;
        }
        if (key < val) lo = mid + 1; else hi = mid - 1;
    }
    // Value not found in keys — no-op (matches CPU behavior of skipping unknown PCs)
}

extern "C" void histogram_by_binary_search(
    const uint32_t* input_values,
    uint32_t n_inputs,
    const uint32_t* sorted_keys,
    uint32_t n_keys,
    uint32_t* mults
) {
    if (n_inputs == 0 || n_keys == 0) return;
    const int block_size = 256;
    const int num_blocks = (n_inputs + block_size - 1) / block_size;
    histogram_by_binary_search_kernel<<<num_blocks, block_size>>>(
        input_values, n_inputs, sorted_keys, n_keys, mults
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("histogram_by_binary_search error: %s\n", cudaGetErrorString(err));
    }
}

// GPU scatter-add: mults[indices[i] - offset] += 1 for each i.
__global__ void scatter_add_kernel(
    uint32_t *mults,
    const uint32_t *indices,
    uint32_t n_indices,
    uint32_t offset
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_indices) return;
    atomicAdd(mults + indices[idx] - offset, 1);
}

extern "C" void scatter_add(
    uint32_t *mults,
    const uint32_t *device_indices,
    uint32_t n_indices,
    uint32_t offset
) {
    if (n_indices == 0) return;
    const int block_size = 256;
    const int num_blocks = (n_indices + block_size - 1) / block_size;
    scatter_add_kernel<<<num_blocks, block_size>>>(mults, device_indices, n_indices, offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("scatter_add error: %s\n", cudaGetErrorString(err));
    }
}

// Stub implementations for backward compatibility
// These will be removed once all code is migrated to use CUDA memory pool directly
extern "C" uint32_t* pool_allocate_cuda(size_t size) {
    return cuda_mem_pool_allocate_uint32(size);
}

extern "C" void pool_deallocate_cuda(uint32_t* ptr, size_t size) {
    (void)size; // Unused parameter
    cuda_mem_pool_free_uint32(ptr);
}

extern "C" uint32_t* pool_allocate_zeroes_cuda(size_t size) {
    return cuda_mem_pool_allocate_zeroes_uint32(size);
}

// Test function to compute offset_bit_reversed_circle_domain_index on GPU
// This is used to verify CUDA matches Rust implementation
__global__ void test_offset_indices_kernel(
    unsigned int* result,
    unsigned int domain_log_size,
    unsigned int eval_log_size,
    int offset,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = offset_bit_reversed_circle_domain_index(i, domain_log_size, eval_log_size, offset);
    }
}

extern "C" void test_offset_bit_reversed_indices(
    unsigned int* result_host,
    unsigned int domain_log_size,
    unsigned int eval_log_size,
    int offset,
    unsigned int n
) {
    unsigned int* result_device = cuda_malloc<unsigned int>(n);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    test_offset_indices_kernel<<<num_blocks, block_size>>>(
        result_device, domain_log_size, eval_log_size, offset, n
    );

    cudaDeviceSynchronize();
    cuda_mem_copy_device_to_host(result_device, result_host, n);
    cuda_free_memory(result_device);
}

// Get CUDA memory info (free and total memory in bytes)
extern "C" void cuda_get_memory_info(size_t* free_mem, size_t* total_mem) {
    cudaError_t err = cudaMemGetInfo(free_mem, total_mem);
    if (err != cudaSuccess) {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        *free_mem = 0;
        *total_mem = 0;
    }
}

// MEM PROBE (diagnostic; read-only). Prints, for a labeled boundary:
//   - driver free/total    (cudaMemGetInfo)
//   - pool reserved         (cudaMemPoolAttrReservedMemCurrent: bytes the pool holds FROM the driver)
//   - pool used             (cudaMemPoolAttrUsedMemCurrent: LIVE bytes within the pool)
// The difference reserved-used = pool-cached-freed (the HOARDING candidate a TrimTo would return).
// Read-only: does not allocate, free, or trim; does NOT change allocation behavior. Safe to call
// unconditionally at phase boundaries. Values are MiB. `tag` names the probe point.
extern "C" void cuda_mem_probe(const char* tag) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    unsigned long long reserved = 0, used = 0;
    if (g_mem_pool_initialized && g_mem_pool != nullptr) {
        cudaMemPoolGetAttribute(g_mem_pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
        cudaMemPoolGetAttribute(g_mem_pool, cudaMemPoolAttrUsedMemCurrent, &used);
    }
    const double MiB = 1024.0 * 1024.0;
    printf("[mem_probe] %-28s free=%.0f total=%.0f pool_reserved=%.0f pool_used=%.0f pool_cached_freed=%.0f (MiB) pool_init=%d\n",
           tag,
           free_mem / MiB, total_mem / MiB,
           reserved / MiB, used / MiB,
           (double)(reserved >= used ? reserved - used : 0) / MiB,
           (g_mem_pool_initialized && g_mem_pool != nullptr) ? 1 : 0);
    fflush(stdout);
}
