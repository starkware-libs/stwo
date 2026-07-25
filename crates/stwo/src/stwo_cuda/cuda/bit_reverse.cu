#include "bit_reverse.cuh"
#include "utils.cuh"

#include <cstdio>

template<typename T>
__global__ void bit_reverse_generic(T *array, int size, int bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rev_idx = bit_reverse(idx, bits);

    if (rev_idx > idx && idx < size) {
        T temp = array[idx];
        array[idx] = array[rev_idx];
        array[rev_idx] = temp;
    }
}

void bit_reverse_base_field(m31 *array, int size) {
    int bits = log_2(size);
    int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    bit_reverse_generic<<<num_blocks, block_size>>>(array, size, bits);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: in-place transform, stream ordering handles dependencies.
}


void bit_reverse_secure_field(qm31 *array, int size) {
    int bits = log_2(size);
    int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    bit_reverse_generic<<<num_blocks, block_size>>>(array, size, bits);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: in-place transform, stream ordering handles dependencies.
}