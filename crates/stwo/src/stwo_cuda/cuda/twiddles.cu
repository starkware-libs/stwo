#include "twiddles.cuh"
#include "bit_reverse.cuh"
#include "utils.cuh"

__global__ void sort_values_kernel(m31 *from, m31 *dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        if (idx < (size >> 1)) {
            dst[idx] = from[idx << 1];
        } else {
            int tmp = idx - (size >> 1);
            dst[idx] = from[size - (tmp << 1) - 1];
        }
    }
}

m31 *sort_values_and_permute_with_bit_reverse_order(m31 *from, int size) {
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;
    m31 *dst = cuda_malloc<m31>(size);

    sort_values_kernel<<<num_blocks, block_dim>>>(from, dst, size);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    bit_reverse_base_field(dst, size);
    return dst;
}

__global__ void precompute_twiddles_kernel(m31 *dst, point initial, point step, int offset, int size, int log_size) {
    // Computes one level of twiddles for a particular Coset.
    //      dst: twiddles array.
    //  initial: coset factor.
    //     step: generator of the group.
    //   offset: store values in dst[offset]
    //     size: coset size
    // log_size: log(size)

    // TODO: when size is larger than the max number of concurrent threads,
    //       consecutive numbers can be computed with a multiplication within the same thread,
    //       instead of using another pow.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    size = size >> 1;
    if (idx < size) {
        point pow = point_pow(step, bit_reverse(idx, log_size - 1)); // TODO: Be aware when log_size == 0 !
        dst[offset + idx] = point_mul(initial, pow).x;
    }
}

m31 *precompute_twiddles(point initial, point step, int size) {
    m31 *twiddles = cuda_malloc_uint32_t(size);
    m31 one = 1;
    cuda_mem_copy_host_to_device<m31>(&one, &twiddles[size - 1], 1);
    int block_dim = 256;
    int num_blocks = (size + block_dim - 1) / block_dim;

    int log_size = log_2(size);

    int i = 0;
    int current_level_offset = 0;
    while (i < log_size) {
        precompute_twiddles_kernel<<<num_blocks, block_dim>>>(
            twiddles, initial, step, current_level_offset, size, log_2(size));
        initial = point_square(initial);
        step = point_square(step);
        size >>= 1;
        current_level_offset += size;
        i += 1;
    }
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: twiddles consumed by subsequent stream-ordered NTT operations.
    return twiddles;
}
