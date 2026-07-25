#include <cstdio>
#include <vector>
#include <array>
#include "fields.cuh"
#include "timer.cuh"
#include "utils.cuh"
#include "prefix_sum.cuh"
#include "bit_reverse.cuh"
#include <cub/cub.cuh>

__global__ void circle_domain_order_to_coset_order_kernel(const m31* in, m31* out, int n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/2) {
        out[2 * i] = in[i];
        out[2 * i + 1] = in[n - 1 - i];
    }
}

__global__ void coset_order_to_circle_domain_order_kernel(const m31* d_in, m31* d_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int half_len = n / 2;
        if (tid < half_len) {
            d_out[tid] = d_in[tid << 1];
        } else {
            int i = tid - half_len;
            d_out[tid] = d_in[n - 1 - (i << 1)];
        }
    }
}


void inclusive_prefix_sum(
    m31 *device_bit_rev_circle_domain_evals,
    unsigned len
) {
    bit_reverse_base_field(device_bit_rev_circle_domain_evals, len);
    m31 *eval_tmp = cuda_alloc_zeroes_uint32_t(len);

    int total_threads = len / 2;
    int block_dim = total_threads < THREAD_COUNT_MAX ? total_threads : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (total_threads + block_dim - 1) / block_dim;
    circle_domain_order_to_coset_order_kernel<<<num_blocks, block_dim>>>(
        device_bit_rev_circle_domain_evals,
        eval_tmp,
        len
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: cub::DeviceScan on same stream will see this kernel's output.

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    // CUDA 13.0/CUB removed cub::Sum functor; use InclusiveSum specialization instead
    ASSERT_CUDA_SUCCESS(cub::DeviceScan::InclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        (M31 *)eval_tmp,
        (M31 *)device_bit_rev_circle_domain_evals,
        len));
    // Use custom allocator to avoid raw cudaMalloc usage
    d_temp_storage = cuda_malloc<uint8_t>(static_cast<unsigned int>(temp_storage_bytes));
    ASSERT_CUDA_SUCCESS(cub::DeviceScan::InclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        (M31 *)eval_tmp,
        (M31 *)device_bit_rev_circle_domain_evals,
        len));

    total_threads = len;
    block_dim = total_threads < THREAD_COUNT_MAX ? total_threads : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (total_threads + block_dim - 1) / block_dim;
    coset_order_to_circle_domain_order_kernel<<<num_blocks, block_dim>>>(
        device_bit_rev_circle_domain_evals,
        eval_tmp,
        len
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: next operation on same stream reads this output.

    bit_reverse_base_field(eval_tmp, len);

    copy_uint32_t_vec_from_device_to_device(eval_tmp, device_bit_rev_circle_domain_evals, len);

    cuda_free_memory(eval_tmp);
    cuda_free_memory(d_temp_storage);
}