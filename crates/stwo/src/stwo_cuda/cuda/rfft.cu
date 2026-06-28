#include "rfft.cuh"
#include "poly_utils.cuh"
#include "utils.cuh"

// log_stride = 4,3,2,1,0
template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void shfl_xor_bf(m31* vals, const unsigned log_stride,
                                    const unsigned lane_id) {
  const unsigned mask = 1 << log_stride;
  const unsigned num_pair_per_thread = 1 << (LOG_VALS_PER_THREAD - 1);
  __syncwarp();
#pragma unroll
  for (unsigned i = 0; i < num_pair_per_thread; i++) {
    m31* ptr = lane_id & mask ? vals + 2 * i : vals + 2 * i + 1;
    *ptr = __shfl_xor_sync(0xffffffff, *ptr, mask);
  }
}

__global__ void rfft_circle_part(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < (values_size >> 1)) {
        m31 val0 = values[2 * idx];
        m31 val1 = values[2 * idx + 1];
        m31 twiddle = get_circle_twiddle(inverse_twiddles_tree, idx);

        m31 temp = mul(val1, twiddle);

        values[2 * idx] = add(val0, temp);
        values[2 * idx + 1] = sub(val0, temp);
    }
}

__global__ void rfft_line_part(m31 *values, m31 *inverse_twiddles_tree, int values_size, int inverse_twiddles_size,
                               int layer_domain_offset, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];

        m31 temp = mul(val1, twiddle);

        values[idx0] = add(val0, temp);
        values[idx1] = sub(val0, temp);
    }
}

void evaluate(int eval_domain_size, m31 *values, m31 *twiddles_tree, int twiddles_size, int values_size) {
    twiddles_tree = &twiddles_tree[twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;

    int log_values_size = log_2(values_size);
    int layer_domain_size = 1;
    int layer_domain_offset = (values_size >> 1) - 2;
    int i = log_values_size - 1;
    while (i > 0) {
        rfft_line_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size, layer_domain_size,
                                                  layer_domain_offset, i);
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
        i -= 1;
    }

    rfft_circle_part<<<num_blocks, block_dim>>>(values, twiddles_tree, values_size);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


__global__ void batch_rfft_circle_part(m31 **values, m31 *inverse_twiddles_tree, int number_of_columns, int number_of_rows) {
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (idx < (number_of_rows >> 1) && column_index < number_of_columns) {
        m31 *column = values[column_index];

        m31 val0 = column[2 * idx];
        m31 val1 = column[2 * idx + 1];
        m31 twiddle = get_circle_twiddle(inverse_twiddles_tree, idx);

        m31 temp = mul(val1, twiddle);

        column[2 * idx] = add(val0, temp);
        column[2 * idx + 1] = sub(val0, temp);
    }
}

__global__ void batch_rfft_line_part(
        m31 **values, m31 *inverse_twiddles_tree, int number_of_columns, int number_of_rows, int layer_domain_offset, int layer
) {
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (idx < (number_of_rows >> 1) && column_index < number_of_columns) {
        m31 *column = values[column_index];

        int number_polynomials = 1 << layer;
        int h = idx / number_polynomials;
        int l = idx % number_polynomials;
        int idx0 = (h << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = column[idx0];
        m31 val1 = column[idx1];
        m31 twiddle = inverse_twiddles_tree[layer_domain_offset + h];

        m31 temp = mul(val1, twiddle);

        column[idx0] = add(val0, temp);
        column[idx1] = sub(val0, temp);
    }
}

void evaluate_columns(const int *eval_domain_sizes, m31 **values, m31 *twiddles_tree, int twiddles_size, int number_of_columns, const int *column_sizes) {
    // TODO: Handle case where columns are of different sizes.
    int number_of_rows = column_sizes[0];
    int eval_domain_size = eval_domain_sizes[0];

    m31 **device_values = clone_to_device<m31*>(values, number_of_columns);

    twiddles_tree = &twiddles_tree[twiddles_size - eval_domain_size];

    int block_size = 1024;
    int number_of_blocks = ((number_of_rows >> 1) + block_size - 1) / block_size;
    dim3 grid_dimensions(number_of_columns, number_of_blocks);

    int log_number_of_rows = log_2(number_of_rows);
    int layer_domain_size = 1;
    int layer_domain_offset = (number_of_rows >> 1) - 2;
    int i = log_number_of_rows - 1;

    while (i > 0) {
        batch_rfft_line_part<<<grid_dimensions, block_size>>>(
                device_values, twiddles_tree, number_of_columns, number_of_rows, layer_domain_offset, i
        );
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
        i -= 1;
    }

    batch_rfft_circle_part<<<grid_dimensions, block_size>>>(device_values, twiddles_tree, number_of_columns, number_of_rows);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(device_values);
}



template <unsigned LOG_VALS_PER_THREAD>
__global__ void n2b_nofinal_block_batch(m31** input, m31** output,
                                  const unsigned log_n, const unsigned num_poly,
                                  unsigned min_stage, unsigned max_stage, m31 *g_twiddles) {

    const unsigned min_stride = 1 << (log_n - max_stage);
    const unsigned middle_stage = min_stage + LOG_VALS_PER_THREAD;
    const unsigned num_stage = 2 * LOG_VALS_PER_THREAD;

    const unsigned ntt_idx = blockIdx.z;
    const unsigned block_index_x = blockIdx.x;
    const unsigned block_index_y = blockIdx.y;
    const unsigned warp_idx_in_block = threadIdx.y;
    const unsigned thread_idx_in_warp = threadIdx.x;

    const m31* input_ntt_start = input[ntt_idx];
    const unsigned block_start =
        (block_index_x << LOG_WARP) +
        (block_index_y << (log_n - max_stage + num_stage));

    __shared__ m31 smem[32 << (2 * LOG_VALS_PER_THREAD)]; // 32 * (2^(2 or 3))
    m31 vals[1 << LOG_VALS_PER_THREAD];

    unsigned offset = warp_idx_in_block * min_stride + thread_idx_in_warp;
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
            const unsigned address =
                block_start + i * (min_stride << LOG_VALS_PER_THREAD) + offset;
            vals[i] = input_ntt_start[address];
    }

    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    for (unsigned i = 1; i < min_stage; i++) {
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
#pragma unroll
  for (unsigned stage = min_stage; stage < middle_stage; stage++) {
    const unsigned log_inner_stride_size = LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
        const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
        const unsigned inner_pair_idx = gid >> log_inner_stride_size;
        const unsigned inner_left_idx = inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
        const unsigned inner_right_idx = inner_left_idx + (1 << log_inner_stride_size);
        const unsigned outer_pair_idx = (block_start + offset) >> (1 + log_n - stage);

        const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
        const m31 temp = vals[inner_left_idx];
        vals[inner_left_idx] = add(temp, twiddle);
        vals[inner_right_idx] = sub(temp, twiddle);

    }
    layer_domain_size <<= 1;
    layer_domain_offset -= layer_domain_size;
  }

#pragma unroll
  for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++)
    smem[thread_idx_in_warp + (i << (LOG_WARP + LOG_VALS_PER_THREAD)) +
         (warp_idx_in_block << LOG_WARP)] = vals[i];
  __syncthreads();
  offset = warp_idx_in_block * (min_stride << LOG_VALS_PER_THREAD) +
           thread_idx_in_warp;
#pragma unroll
  for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
    vals[i] = smem[thread_idx_in_warp + (i << LOG_WARP) +
                    (warp_idx_in_block << (LOG_WARP + LOG_VALS_PER_THREAD))];
  }

#pragma unroll
  for (unsigned stage = middle_stage; stage <= max_stage; stage++) {
    const unsigned log_inner_stride_size = LOG_VALS_PER_THREAD - 1 - (stage - middle_stage);
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
        const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
        const unsigned inner_pair_idx = gid >> log_inner_stride_size;
        const unsigned inner_left_idx = inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
        const unsigned inner_right_idx = inner_left_idx + (1 << log_inner_stride_size);
        const unsigned outer_pair_idx = (block_start + offset) >> (1 + log_n - stage);
        const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
        const m31 temp = vals[inner_left_idx];
        vals[inner_left_idx] = add(temp, twiddle);
        vals[inner_right_idx] = sub(temp, twiddle);
    }
    layer_domain_size <<= 1;
    layer_domain_offset -= layer_domain_size;
  }

  m31* output_ntt_start = output[ntt_idx];
#pragma unroll
  for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
    const unsigned address = block_start + i * min_stride + offset;
    output_ntt_start[address] = vals[i];
  }
}

EXTERN void ntt_n2b_nofinal_6_stage_batch(m31** input, m31** output,
                                unsigned log_n, unsigned num_poly, unsigned start_stage,
                                m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 3;
    dim3 block_dim{32, 1 << log_val_per_thread, 1};
    dim3 grid_dim{};
    constexpr unsigned num_stage = 2 * log_val_per_thread;
    unsigned end_stage = start_stage + num_stage - 1;
    unsigned min_stride = 1 << (log_n - end_stage);
    grid_dim.z = num_poly;
    grid_dim.x = min_stride / 32;
    grid_dim.y = (1 << log_n) / (min_stride << num_stage);

    n2b_nofinal_block_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output, log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

}

EXTERN void ntt_n2b_nofinal_8_stage_batch(m31** input, m31** output,
                                unsigned log_n, unsigned num_poly, unsigned start_stage,
                                m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 4;
    ASSERT_TRUE(
        log_n - start_stage + 1 >= 2 * log_val_per_thread + 5,
        "left stage < (2 * log_val_per_thread + 5) in ntt_nofinal_8_stage");
    dim3 block_dim{32, 1 << log_val_per_thread, 1};
    dim3 grid_dim{};
    constexpr unsigned num_stage = 2 * log_val_per_thread;
    unsigned end_stage = start_stage + num_stage - 1;
    unsigned min_stride = 1 << (log_n - end_stage);
    grid_dim.z = num_poly;
    grid_dim.x = min_stride / 32;
    grid_dim.y = (1 << log_n) / (min_stride << num_stage);

    ASSERT_TRUE(grid_dim.y * grid_dim.x * block_dim.x * block_dim.y
                        << log_val_per_thread ==
                    1 << log_n,
                "thread count mismatch in ntt_nofinal_8_stage");
    n2b_nofinal_block_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output, log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

template <unsigned LOG_VALS_PER_THREAD>
__global__ void n2b_final_warp_batch(m31** input, m31** output,
                               const unsigned log_n, const unsigned num_poly,
                               unsigned min_stage, m31 *g_twiddles) {
    const unsigned ntt_idx = blockIdx.y;
    const unsigned thread_idx_in_warp = threadIdx.x;
    const unsigned warps_idx_in_ntt = blockDim.y * blockIdx.x + threadIdx.y;
    const unsigned log_vals_per_warp = LOG_VALS_PER_THREAD + LOG_WARP;
    const m31* input_ntt_start = input[ntt_idx];
    unsigned warp_start =
        (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp;

    m31 vals[1 << LOG_VALS_PER_THREAD];
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
        vals[i] = input_ntt_start[warp_start + (i << LOG_WARP)];
    }

    // int log_values_size = log_n;
    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    for (unsigned i = 1; i < min_stage; i++) {
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    unsigned stage = min_stage;
    #pragma unroll
    for (; stage < min_stage + LOG_VALS_PER_THREAD; stage++) {
        const unsigned log_inner_stride_size =
            LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
    #pragma unroll
        for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
            const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
            const unsigned inner_pair_idx = gid >> log_inner_stride_size;
            const unsigned inner_left_idx =
                inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
            const unsigned inner_right_idx =
                inner_left_idx + (1 << log_inner_stride_size);
            const unsigned outer_pair_idx = warp_start >> (1 + log_n - stage);
            const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
            const m31 temp = vals[inner_left_idx];
            vals[inner_left_idx] = add(temp, twiddle);
            vals[inner_right_idx] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    #pragma unroll
    for (; stage <= log_n; stage++) {
        const unsigned log_stride = log_n - stage;
        shfl_xor_bf<LOG_VALS_PER_THREAD>(vals, log_stride, thread_idx_in_warp);
    #pragma unroll
        for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++) {
            const unsigned inner_pair_idx =
                (thread_idx_in_warp >> log_stride) + (i << (LOG_WARP - log_stride));
            const unsigned outer_pair_idx = warps_idx_in_ntt
                                            << (log_vals_per_warp - 1 - log_stride);
            m31 twiddle = m31(1);
            if (stage == log_n) {
                twiddle = mul(get_circle_twiddle(g_twiddles, inner_pair_idx + outer_pair_idx), vals[2 * i + 1]);
            } else {
                twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[2 * i + 1]);
            }

            const m31 temp = vals[2 * i];
            vals[2 * i] = add(temp, twiddle);
            vals[2 * i + 1] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    m31* output_ntt_start = output[ntt_idx];
    warp_start = (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp * 2;
    uint64_t* src = reinterpret_cast<uint64_t*>(vals);
    uint64_t* dst = reinterpret_cast<uint64_t*>(output_ntt_start + warp_start);


    #pragma unroll
    for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++) {
        dst[(i << LOG_WARP)] = src[i];
    }
}

EXTERN void ntt_n2b_final_7_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned num_poly,
                              unsigned start_stage,
                              m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 2;
    ASSERT_TRUE(log_n + 1 - (log_val_per_thread + LOG_WARP) == start_stage,
                "start_stage mismatch in ntt_final_7_stage");
    dim3 block_dim = dim3{32, 1, 1};
    dim3 grid_dim = dim3{1, 1, 1};
    const unsigned num_warp = 1 << (log_n - LOG_WARP - log_val_per_thread);
    block_dim.y = min(num_warp, 4);
    grid_dim.y = num_poly;
    grid_dim.x = num_warp / block_dim.y;

    n2b_final_warp_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output, log_n, num_poly, start_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

EXTERN void ntt_n2b_final_8_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned num_poly,
                              unsigned start_stage,
                              m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 3;
    ASSERT_TRUE(log_n + 1 - (log_val_per_thread + LOG_WARP) == start_stage,
                "start_stage mismatch in ntt_final_8_stage");
    dim3 block_dim = dim3{32, 1, 1};
    dim3 grid_dim = dim3{1, 1, 1};
    const unsigned num_warp = 1 << (log_n - LOG_WARP - log_val_per_thread);
    block_dim.y = min(num_warp, 4);
    grid_dim.y = num_poly;
    grid_dim.x = num_warp / block_dim.y;

    n2b_final_warp_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output, log_n, num_poly, start_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


template <unsigned LOG_WARP_PER_BLOCK>
__global__ void n2b_final_block_warp_batch(
    m31** input, m31** output, const unsigned log_n,
    const unsigned num_poly, unsigned min_stage, m31 *g_twiddles) {

    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    const unsigned ntt_idx = blockIdx.z;
    const unsigned warp_idx_in_block = threadIdx.y;
    const unsigned thread_idx_in_warp = threadIdx.x;

    const m31* input_ntt_start = input[ntt_idx];
    const unsigned block_index_y = blockIdx.x;
    const unsigned block_start = (block_index_y << (LOG_WARP + LOG_VALS_PER_THREAD + LOG_WARP_PER_BLOCK));

    __shared__ m31 smem[32 << (LOG_WARP_PER_BLOCK + LOG_VALS_PER_THREAD)];
    m31 vals[1 << LOG_VALS_PER_THREAD];

    unsigned offset = (warp_idx_in_block << LOG_WARP) + thread_idx_in_warp;
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
        const unsigned address = block_start + (i << (LOG_WARP + LOG_WARP_PER_BLOCK)) + offset;
        vals[i] = input_ntt_start[address];
    }

    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    for (unsigned stage = 1; stage < min_stage; stage++) {
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    unsigned stage = min_stage;
    #pragma unroll
    for (; stage < min_stage + LOG_WARP_PER_BLOCK; stage++) {
        const unsigned log_inner_stride_size = LOG_VALS_PER_THREAD - 1 - (stage - min_stage);
    #pragma unroll
        for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
            const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
            const unsigned inner_pair_idx = gid >> log_inner_stride_size;
            const unsigned inner_left_idx =
                inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
            const unsigned inner_right_idx =
                inner_left_idx + (1 << log_inner_stride_size);
            const unsigned outer_pair_idx =
                (block_start + offset) >> (1 + log_n - stage);
            const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
            const m31 temp = vals[inner_left_idx];
            vals[inner_left_idx] = add(temp, twiddle);
            vals[inner_right_idx] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    #pragma unroll
    for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
        smem[thread_idx_in_warp + (i << (LOG_WARP + LOG_WARP_PER_BLOCK)) +
            (warp_idx_in_block << LOG_WARP)] = vals[i];
    }
    __syncthreads();
    #pragma unroll
    for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
        vals[i] = smem[thread_idx_in_warp + (i << LOG_WARP) +
                    (warp_idx_in_block << (LOG_WARP + LOG_VALS_PER_THREAD))];
    }
    offset = (warp_idx_in_block << (LOG_WARP + LOG_VALS_PER_THREAD)) +
            thread_idx_in_warp;

    const unsigned warps_idx_in_ntt = blockDim.y * block_index_y + threadIdx.y;
    const unsigned log_vals_per_warp = LOG_VALS_PER_THREAD + LOG_WARP;
    unsigned warp_start =
        (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp;
    const unsigned new_min_stage = min_stage + LOG_WARP_PER_BLOCK;
    stage = new_min_stage;
    #pragma unroll
    for (; stage < new_min_stage + LOG_VALS_PER_THREAD; stage++) {
        const unsigned log_inner_stride_size =
            LOG_VALS_PER_THREAD - 1 - (stage - new_min_stage);
    #pragma unroll
        for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
            const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
            const unsigned inner_pair_idx = gid >> log_inner_stride_size;
            const unsigned inner_left_idx =
                inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
            const unsigned inner_right_idx =
                inner_left_idx + (1 << log_inner_stride_size);
            const unsigned outer_pair_idx = warp_start >> (1 + log_n - stage);
            const m31 twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[inner_right_idx]);
            const m31 temp = vals[inner_left_idx];
            vals[inner_left_idx] = add(temp, twiddle);
            vals[inner_right_idx] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    #pragma unroll
    for (; stage <= log_n; stage++) {
        const unsigned log_stride = log_n - stage;
        shfl_xor_bf<LOG_VALS_PER_THREAD>(vals, log_stride, thread_idx_in_warp);
    #pragma unroll
        for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++) {
            const unsigned inner_pair_idx =
                (thread_idx_in_warp >> log_stride) + (i << (LOG_WARP - log_stride));
            const unsigned outer_pair_idx = warps_idx_in_ntt
                                            << (log_vals_per_warp - 1 - log_stride);
            m31 twiddle = m31(1);
            if (stage == log_n) {
                twiddle = mul(get_circle_twiddle(g_twiddles, inner_pair_idx + outer_pair_idx), vals[2 * i + 1]);
            } else {
                twiddle = mul(g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx], vals[2 * i + 1]);
            }

            const m31 temp = vals[2 * i];
            vals[2 * i] = add(temp, twiddle);
            vals[2 * i + 1] = sub(temp, twiddle);
        }
        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }

    m31* output_ntt_start = output[ntt_idx];
    warp_start = (warps_idx_in_ntt << log_vals_per_warp) + thread_idx_in_warp * 2;
    uint64_t* src = reinterpret_cast<uint64_t*>(vals);
    uint64_t* dst = reinterpret_cast<uint64_t*>(output_ntt_start + warp_start);

    #pragma unroll
    for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++)
        dst[i << LOG_WARP] = src[i];
}

EXTERN void ntt_n2b_final_10_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly, unsigned start_stage,
                               m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 2;
    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    ASSERT_TRUE(log_n + 1 - start_stage ==
                    LOG_VALS_PER_THREAD + LOG_WARP + log_warp_per_block,
                "start_stage mismatch in ntt_final_10_stage");
    dim3 block_dim = {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - LOG_WARP - LOG_VALS_PER_THREAD - log_warp_per_block);

    n2b_final_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim>>>(
        input, output, log_n, num_poly, start_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

EXTERN void ntt_n2b_final_11_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly, unsigned start_stage,
                               m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 3;
    constexpr unsigned LOG_VALS_PER_THREAD = 3;
    ASSERT_TRUE(log_n + 1 - start_stage ==
                    LOG_VALS_PER_THREAD + LOG_WARP + log_warp_per_block,
                "start_stage mismatch in ntt_final_11_stage");
    dim3 block_dim = {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - LOG_WARP - LOG_VALS_PER_THREAD - log_warp_per_block);
    n2b_final_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim>>>(
        input, output,  log_n, num_poly, start_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


__global__ void ntt_n2b_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned stage, m31 *layer_twiddles) {
    const unsigned ntt_index = blockIdx.y;
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = 1 << (log_n - stage);
    const unsigned group_idx = gid & (stride - 1);
    const unsigned pair_idx = gid >> (log_n - stage);

    const m31* input_start = input[ntt_index];
    const unsigned left_index = group_idx + pair_idx * 2 * stride;
    const unsigned right_index = left_index + stride;

    m31 left = input_start[left_index];
    m31 right = input_start[right_index];

    m31 twiddle = m31(1);
    if (stage == log_n) {
        twiddle = get_circle_twiddle(layer_twiddles, pair_idx);
    } else {
        twiddle = layer_twiddles[pair_idx];
    }


    m31 twiddle_x = mul(twiddle, right);
    const m31 temp = left;
    m31 left_r = add(temp, twiddle_x);
    m31 right_r = sub(temp, twiddle_x);

    m31* output_start = output[ntt_index];

    output_start[left_index] = left_r;
    output_start[right_index] = right_r;
}

EXTERN void ntt_n2b_native_batch(m31** value,
                           unsigned log_n, unsigned num_poly,
                           unsigned start_stage, unsigned end_stage,
                           m31 *g_twiddles,
                           unsigned twiddles_size, unsigned eval_domain_size) {
    m31 **device_values = clone_to_device<m31*>(value, num_poly);
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    dim3 block_dim{};
    block_dim.x = log_n <= 9 ? 1 << (log_n - 1) : 256;
    dim3 grid_dim{};
    grid_dim.y = num_poly;
    grid_dim.x = log_n <= 9 ? 1 : 1 << (log_n - 9);

    unsigned layer_domain_size = 1;
    unsigned layer_domain_offset = ((1 << log_n) >> 1) - 2;

    ASSERT_TRUE(start_stage >= 1, "start_stage < 1 in ntt_n2b_native");
    ASSERT_TRUE(end_stage <= log_n, "end_stage <= log_n in ntt_n2b_native");
    for (unsigned stage = 1; stage < log_n; stage++) {

        if (stage >= start_stage) {
            ntt_n2b_stage_batch<<<grid_dim, block_dim, 0>>>(device_values, device_values, log_n, stage, &g_twiddles[layer_domain_offset]);
            ASSERT_CUDA_SUCCESS(cudaGetLastError());
        }

        layer_domain_size <<= 1;
        layer_domain_offset -= layer_domain_size;
    }
    if (end_stage == log_n) {
        ntt_n2b_stage_batch<<<grid_dim, block_dim, 0>>>(device_values, device_values, log_n, log_n, g_twiddles);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }
    cuda_free_memory(device_values);
}

EXTERN void ntt_n2b_columns(
    uint32_t** values_columns,
    unsigned log_n,
    unsigned num_poly,
    uint32_t* g_twiddles,
    unsigned twiddles_size,
    unsigned eval_domain_size
) {
    m31 **device_values = clone_to_device<m31*>(values_columns, num_poly);

    if (log_n < 13) {
        ntt_n2b_native_batch(device_values, log_n, num_poly, 1, log_n, g_twiddles, twiddles_size, eval_domain_size);
    } else if (log_n >= 13 && log_n <= 19) {
        const auto& config = LAUNCH_N2B_CONFIG_13_19[log_n - 13];
        auto kernel0 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[0]) {
                case 6: return ntt_n2b_nofinal_6_stage_batch;
                case 8: return ntt_n2b_nofinal_8_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel0 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        auto kernel1 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[1]) {
                case 7: return ntt_n2b_final_7_stage_batch;
                case 8: return ntt_n2b_final_8_stage_batch;
                case 10: return ntt_n2b_final_10_stage_batch;
                case 11: return ntt_n2b_final_11_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel1 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        const uint32_t start_stage1 = 1 + config[0];

        kernel0(device_values, device_values, log_n, num_poly, 1, g_twiddles, twiddles_size, eval_domain_size);
        kernel1(device_values, device_values, log_n, num_poly, start_stage1, g_twiddles, twiddles_size, eval_domain_size);

    } else if (log_n >= 20 && log_n <= 27) {
        const auto& config = LAUNCH_N2B_CONFIG_20_27[log_n - 20];
        auto kernel0 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[0]) {
                case 6: return ntt_n2b_nofinal_6_stage_batch;
                case 8: return ntt_n2b_nofinal_8_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel0 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        auto kernel1 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[1]) {
                case 6: return ntt_n2b_nofinal_6_stage_batch;
                case 8: return ntt_n2b_nofinal_8_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel1 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        auto kernel2 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[2]) {
                case 7: return ntt_n2b_final_7_stage_batch;
                case 8: return ntt_n2b_final_8_stage_batch;
                case 10: return ntt_n2b_final_10_stage_batch;
                case 11: return ntt_n2b_final_11_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel2 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();
        const uint32_t start_stage1 = 1 + config[0];
        const uint32_t start_stage2 = start_stage1 + config[1];

        kernel0(device_values, device_values, log_n, num_poly, 1, g_twiddles, twiddles_size, eval_domain_size);
        kernel1(device_values, device_values, log_n, num_poly, start_stage1, g_twiddles, twiddles_size, eval_domain_size);
        kernel2(device_values, device_values, log_n, num_poly, start_stage2, g_twiddles, twiddles_size, eval_domain_size);

    } else if (log_n >= 28 && log_n <= 30) {
        const auto& config = LAUNCH_N2B_CONFIG_28_30[log_n - 28];

        auto kernel0 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[0]) {
                case 6: return ntt_n2b_nofinal_6_stage_batch;
                case 8: return ntt_n2b_nofinal_8_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel0 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        auto kernel1 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[1]) {
                case 6: return ntt_n2b_nofinal_6_stage_batch;
                case 8: return ntt_n2b_nofinal_8_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel1 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        auto kernel2 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[2]) {
                case 6: return ntt_n2b_nofinal_6_stage_batch;
                case 8: return ntt_n2b_nofinal_8_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel2 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        auto kernel3 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[3]) {
                case 7: return ntt_n2b_final_7_stage_batch;
                case 8: return ntt_n2b_final_8_stage_batch;
                case 10: return ntt_n2b_final_10_stage_batch;
                case 11: return ntt_n2b_final_11_stage_batch;
                default: {
                    fprintf(stderr, "n2b kernel3 config error\n");
                    exit(EXIT_FAILURE);
                }
            }
        }();

        const uint32_t start_stage1 = 1 + config[0];
        const uint32_t start_stage2 = start_stage1 + config[1];
        const uint32_t start_stage3 = start_stage2 + config[2];

        kernel0(device_values, device_values, log_n, num_poly, 1, g_twiddles, twiddles_size, eval_domain_size);
        kernel1(device_values, device_values, log_n, num_poly, start_stage1, g_twiddles, twiddles_size, eval_domain_size);
        kernel2(device_values, device_values, log_n, num_poly, start_stage2, g_twiddles, twiddles_size, eval_domain_size);
        kernel3(device_values, device_values, log_n, num_poly, start_stage3, g_twiddles, twiddles_size, eval_domain_size);
    } else {
        fprintf(stderr, "n2b config too big\n");
        exit(EXIT_FAILURE);
    }
    cuda_free_memory(device_values);
}