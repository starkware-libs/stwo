#include "accumulate.cuh"
#include "utils.cuh"

__global__
void accumulate_kernel(int size, m31 **left_columns, m31 **right_columns) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        left_columns[0][i] = add(left_columns[0][i], right_columns[0][i]);
        left_columns[1][i] = add(left_columns[1][i], right_columns[1][i]);
        left_columns[2][i] = add(left_columns[2][i], right_columns[2][i]);
        left_columns[3][i] = add(left_columns[3][i], right_columns[3][i]);
    }
}

void accumulate(int size, m31 **left_columns, m31 **right_columns) {
    m31 **left_columns_device = clone_to_device<m31*>(left_columns, 4);
    m31 **right_columns_device = clone_to_device<m31*>(right_columns, 4);

    int block_dim = 1024;
    int num_blocks = (size + block_dim - 1) / block_dim;
    accumulate_kernel<<<num_blocks, block_dim>>>(size, left_columns_device, right_columns_device);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: stream ordering + async free.
    cuda_free_memory(left_columns_device);
    cuda_free_memory(right_columns_device);
}

// Kernel: for each index i in col (size col_size), compute:
//   lifted_idx = (i >> (log_ratio + 1) << 1) + (i & 1)
//   col[i] = col[i] + curr[lifted_idx]
__global__
void lift_and_accumulate_kernel(
    int col_size,
    m31 *col_0, m31 *col_1, m31 *col_2, m31 *col_3,
    m31 *curr_0, m31 *curr_1, m31 *curr_2, m31 *curr_3,
    int log_ratio
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= col_size) return;
    int lifted_idx = (i >> (log_ratio + 1) << 1) + (i & 1);
    col_0[i] = add(col_0[i], curr_0[lifted_idx]);
    col_1[i] = add(col_1[i], curr_1[lifted_idx]);
    col_2[i] = add(col_2[i], curr_2[lifted_idx]);
    col_3[i] = add(col_3[i], curr_3[lifted_idx]);
}

void lift_and_accumulate(
    int col_size,
    m31 *col_0, m31 *col_1, m31 *col_2, m31 *col_3,
    m31 *curr_0, m31 *curr_1, m31 *curr_2, m31 *curr_3,
    int log_ratio
) {
    int block_dim = 1024;
    int num_blocks = (col_size + block_dim - 1) / block_dim;
    lift_and_accumulate_kernel<<<num_blocks, block_dim>>>(
        col_size,
        col_0, col_1, col_2, col_3,
        curr_0, curr_1, curr_2, curr_3,
        log_ratio
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: stream ordering handles dependencies.
}
