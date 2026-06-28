#include "ifft.cuh"
#include "point.cuh"
#include "poly_utils.cuh"
#include "utils.cuh"

__global__ void ifft_circle_part(m31 *values, m31 *inverse_twiddles_tree, int values_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        m31 val0 = values[2 * idx];
        m31 val1 = values[2 * idx + 1];
        m31 twiddle = get_circle_twiddle(inverse_twiddles_tree, idx);

        values[2 * idx] = add(val0, val1);
        values[2 * idx + 1] = mul(sub(val0, val1), twiddle);
    }
}

__global__ void ifft_line_part(m31 *values, m31 *twiddles, int values_size, int layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (values_size >> 1)) {
        // `index` is in [0, values_size / 2).
        // It is interpreted as the n - 1 bit-string `twiddle_index || polynomial_index`,
        // where n = log_2(`values_size`), `polynomial_index` is the rightmost `layer` bits,
        // and `twiddle_index` is the rest `n - layer - 1` bits.
        // This thread performs a butterfly between the values at indexes `twiddle_index || 0 || polynomial_index`
        // and `twiddle_index || 1 || polynomial_index`.
        int number_polynomials = 1 << layer;
        int twiddle_index = idx >> layer;
        int l = idx & (number_polynomials - 1);
        int idx0 = (twiddle_index << (layer + 1)) + l;
        int idx1 = idx0 + number_polynomials;

        m31 val0 = values[idx0];
        m31 val1 = values[idx1];
        m31 twiddle = twiddles[twiddle_index];


        values[idx0] = add(val0, val1);
        values[idx1] = mul(sub(val0, val1), twiddle);

    }
}

void interpolate(int eval_domain_size, m31 *values, m31 *inverse_twiddles_tree, int inverse_twiddles_size, int values_size) {
    inverse_twiddles_tree = &inverse_twiddles_tree[inverse_twiddles_size - eval_domain_size];
    int block_dim = 256;
    int num_blocks = ((values_size >> 1) + block_dim - 1) / block_dim;
    ifft_circle_part<<<num_blocks, block_dim>>>(values, inverse_twiddles_tree, values_size);

    int log_values_size = log_2(values_size);
    int layer_domain_size = values_size >> 1;
    int layer_domain_offset = 0;
    int i = 1;
    while (i < log_values_size) {
        ifft_line_part<<<num_blocks, block_dim>>>(values, &inverse_twiddles_tree[layer_domain_offset], values_size, i);

        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
        i += 1;
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

    }

    block_dim = 1024;
    num_blocks = (values_size + block_dim - 1) / block_dim;
    m31 factor = inv(pow(m31{2}, log_values_size));
    rescale<<<num_blocks, block_dim>>>(values, values_size, factor);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}



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

__global__ void batch_ifft_circle_part(m31 **values, m31 *inverse_twiddles_tree, int values_size, int number_of_rows) {
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (index < (number_of_rows >> 1) && column_index < values_size) {
        m31 *column = values[column_index];

        m31 val0 = column[2 * index];
        m31 val1 = column[2 * index + 1];
        m31 twiddle = get_circle_twiddle(inverse_twiddles_tree, index);

        column[2 * index] = add(val0, val1);
        column[2 * index + 1] = mul(sub(val0, val1), twiddle);
    }
}

__global__
void batch_ifft_line_part(m31 **values, m31 *twiddles, int values_size, int number_of_rows, int layer) {
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (index < (number_of_rows >> 1) && column_index < values_size) {
        // `index` is in [0, number_of_rows / 2).
        // It is interpreted as the n - 1 bit-string `twiddle_index || polynomial_index`,
        // where n = log_2(`number_of_rows`), `polynomial_index` is the rightmost `layer` bits,
        // and `twiddle_index` is the rest `n - layer - 1` bits.
        // This thread performs a butterfly between the values at indexes `twiddle_index || 0 || polynomial_index`
        // and `twiddle_index || 1 || polynomial_index`.
        m31 *column = values[column_index];

        int number_polynomials = 1 << layer;
        int twiddle_index = index >> layer;
        int polynomial_index = index & (number_polynomials - 1);

        int idx0 = (twiddle_index << (layer + 1)) | polynomial_index;
        int idx1 = idx0 | number_polynomials;

        m31 val0 = column[idx0];
        m31 val1 = column[idx1];

        m31 twiddle = twiddles[twiddle_index];

        column[idx0] = add(val0, val1);
        column[idx1] = mul(sub(val0, val1), twiddle);
    }
}

__global__ void batch_rescale(m31 **values, int values_size, int number_of_rows, m31 factor) {
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int column_index = blockIdx.x;

    if (index < number_of_rows && column_index < values_size) {
        values[column_index][index] = mul(values[column_index][index], factor);
    }
}


void interpolate_columns(int eval_domain_size, m31 **values, m31 *inverse_twiddles_tree, int inverse_twiddles_size,
                         int values_size, int number_of_rows) {
    // TODO: Handle case where columns are of different sizes.
    int blockDimensions = 1024;

    m31 **device_values = clone_to_device<m31*>(values, values_size);

    m31 *inverseTwiddlesTree = inverse_twiddles_tree;
    inverseTwiddlesTree = &inverseTwiddlesTree[inverse_twiddles_size - eval_domain_size];
    int numBlocks = ((number_of_rows >> 1) + blockDimensions - 1) / blockDimensions;
    dim3 gridDimensions(values_size, numBlocks);

    batch_ifft_circle_part<<<gridDimensions, blockDimensions>>>(device_values, inverseTwiddlesTree, values_size, number_of_rows);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    int log_number_of_rows = log_2(number_of_rows);
    int layer_domain_size = number_of_rows >> 1;
    int layer_domain_offset = 0;
    int i = 1;
    while (i < log_number_of_rows) {
        batch_ifft_line_part<<<gridDimensions, blockDimensions>>>(
            device_values,
            &inverseTwiddlesTree[layer_domain_offset],
            values_size,
            number_of_rows,
            i
        );
        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
        i += 1;
    }
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    m31 factor = inv(pow(m31{2}, log_number_of_rows));
    numBlocks = (number_of_rows + blockDimensions - 1) / blockDimensions;
    dim3 rescaleGridDimensions(values_size, numBlocks);
    batch_rescale<<<rescaleGridDimensions, blockDimensions>>>(device_values, values_size, number_of_rows, factor);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_values);
}


template <unsigned LOG_VALS_PER_THREAD>
__global__ void b2n_init_warp_batch(m31** input, m31** output,
                                  const unsigned log_n, const unsigned num_poly,
                                  unsigned min_stage, unsigned max_stage,
                                   m31 *g_twiddles) {
  const unsigned ntt_idx = blockIdx.y;
  const unsigned thread_idx_in_warp = threadIdx.x;
  const unsigned warps_idx_in_ntt = blockDim.y * blockIdx.x + threadIdx.y;
  const unsigned log_vals_per_warp = LOG_VALS_PER_THREAD + LOG_THREADS_PER_WARP;
  m31* input_ntt_start = input[ntt_idx];
  unsigned warp_start = warps_idx_in_ntt << log_vals_per_warp;

  m31 vals[1 << LOG_VALS_PER_THREAD];
#pragma unroll
  for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
    vals[i] = input_ntt_start[warp_start + thread_idx_in_warp * (1 << LOG_VALS_PER_THREAD) + i];
  }

 unsigned layer_domain_size = (1 << log_n) >> 1;
 unsigned layer_domain_offset = 0;
 unsigned stage = min_stage;
#pragma unroll
  for (; stage < min_stage + LOG_VALS_PER_THREAD; stage++) {
    const unsigned log_inner_stride_size = stage - 1;
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
      const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
      const unsigned inner_pair_idx = gid >> log_inner_stride_size;
      const unsigned inner_left_idx =
          inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
      const unsigned inner_right_idx =
          inner_left_idx + (1 << log_inner_stride_size);
      const unsigned outer_pair_idx = (warp_start + (thread_idx_in_warp << LOG_VALS_PER_THREAD)) >> log_inner_stride_size >> 1;


      m31 twiddle = m31(1);

      if (stage == 1) {
        twiddle =  get_circle_twiddle(g_twiddles, inner_pair_idx + outer_pair_idx);
      } else {
        twiddle = g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx];
      }


      const m31 temp = vals[inner_left_idx];
      vals[inner_left_idx] = add(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = sub(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = mul(vals[inner_right_idx], twiddle);

     }
    if (stage >= 2) {
        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
    }
  }

#pragma unroll
  for (; stage <= max_stage; stage++) {
    const unsigned log_stride = stage - LOG_VALS_PER_THREAD - 1;
    shfl_xor_bf<LOG_VALS_PER_THREAD>(vals, log_stride, thread_idx_in_warp);
#pragma unroll
    for (unsigned i = 0; i < 1 << (LOG_VALS_PER_THREAD - 1); i++) {
      const unsigned log_inner_stride_size = stage - 1;
      const unsigned inner_pair_idx = i >> log_inner_stride_size;
      const unsigned outer_pair_idx = (warp_start + (thread_idx_in_warp << LOG_VALS_PER_THREAD)) >> stage;

      m31 twiddle = g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx];

      const m31 temp = vals[2 * i];
      vals[2 * i] = add(temp, vals[2 * i + 1]);
      vals[2 * i + 1] = sub(temp, vals[2 * i + 1]) ;
      vals[2 * i + 1] = mul(vals[2 * i + 1], twiddle);

    }
    layer_domain_size >>= 1;
    layer_domain_offset += layer_domain_size;
  }

  shfl_xor_bf<LOG_VALS_PER_THREAD>(vals, 0, thread_idx_in_warp);

  m31* output_ntt_start = output[ntt_idx];

  if ((thread_idx_in_warp & 1) == 1) {
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
      output_ntt_start[warp_start + i + (thread_idx_in_warp >> 1) * (1 << LOG_VALS_PER_THREAD) + (1 << (log_vals_per_warp - 1))] = vals[i];
    }
  } else {
    #pragma unroll
    for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
      output_ntt_start[warp_start + i + (thread_idx_in_warp >> 1) * (1 << LOG_VALS_PER_THREAD)] = vals[i];
    }
  }
}


EXTERN void ntt_b2n_init_7_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned num_poly,
                              unsigned start_stage, m31* g_twiddles, unsigned twiddles_size, unsigned eval_domain_size
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 2;
    dim3 block_dim = dim3{32, 1, 1};
    dim3 grid_dim = dim3{1, 1, 1};
    const unsigned num_warp = 1 << (log_n - LOG_THREADS_PER_WARP - log_val_per_thread);
    const unsigned num_stage = log_val_per_thread + LOG_THREADS_PER_WARP; // 2 + 5=7
    const unsigned end_stage = start_stage + num_stage - 1;
    block_dim.y = min(num_warp, 4);
    grid_dim.y = num_poly;
    grid_dim.x = num_warp / block_dim.y;
    b2n_init_warp_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}



EXTERN void ntt_b2n_init_8_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned num_poly,
                              unsigned start_stage, m31* g_twiddles, unsigned twiddles_size, unsigned eval_domain_size
) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 3;
    dim3 block_dim = dim3{32, 1, 1};
    dim3 grid_dim = dim3{1, 1, 1};
    const unsigned num_warp = 1 << (log_n - LOG_THREADS_PER_WARP - log_val_per_thread);
    const unsigned num_stage = log_val_per_thread + LOG_THREADS_PER_WARP; // 3 + 5 = 8
    const unsigned end_stage = start_stage + num_stage - 1;
    block_dim.y = min(num_warp, 4);
    grid_dim.y = num_poly;
    grid_dim.x = num_warp / block_dim.y;
    b2n_init_warp_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


template <unsigned LOG_WARP_PER_BLOCK>
__global__ void b2n_init_block_warp_batch(m31** input, m31** output,
                                  const unsigned log_n, const unsigned num_poly,
                                  unsigned min_stage, unsigned max_stage,
                                  m31 *g_twiddles) {
  constexpr unsigned log_vals_per_threads_default = 3;
  const unsigned log_threads_per_warp = 5; // LOG_THREADS_PER_WARP

  const unsigned ntt_idx = blockIdx.z;
  const unsigned warp_idx_in_block = threadIdx.y;
  const unsigned thread_idx_in_warp = threadIdx.x;

  const unsigned log_vals_per_warp = log_vals_per_threads_default + log_threads_per_warp;
  m31* input_ntt_start = input[ntt_idx];

  const unsigned block_index_y = blockIdx.x;
  const unsigned block_start = (block_index_y << (log_threads_per_warp + log_vals_per_threads_default + LOG_WARP_PER_BLOCK));

  m31 vals[1 << log_vals_per_threads_default];

  const unsigned warps_idx_in_ntt = blockDim.y * block_index_y + threadIdx.y;
  unsigned warp_start = warps_idx_in_ntt << log_vals_per_warp;

#pragma unroll
  for (unsigned i = 0; i < 1 << log_vals_per_threads_default; i++) {
    vals[i] = input_ntt_start[warp_start + thread_idx_in_warp * (1 << log_vals_per_threads_default) + i];
  }

 unsigned layer_domain_size = (1 << log_n) >> 1;
 unsigned layer_domain_offset = 0;
 unsigned stage = min_stage;
#pragma unroll
  for (; stage < min_stage + log_vals_per_threads_default; stage++) {
    const unsigned log_inner_stride_size = stage - 1;
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (log_vals_per_threads_default - 1); gid++) {
      const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
      const unsigned inner_pair_idx = gid >> log_inner_stride_size;
      const unsigned inner_left_idx =
          inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
      const unsigned inner_right_idx =
          inner_left_idx + (1 << log_inner_stride_size);
      const unsigned outer_pair_idx = (warp_start + (thread_idx_in_warp << log_vals_per_threads_default)) >> log_inner_stride_size >> 1;


      m31 twiddle = m31(1);

      if (stage == 1) {
        twiddle =  get_circle_twiddle(g_twiddles, inner_pair_idx + outer_pair_idx);
      } else {
        twiddle = g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx];
      }

      const m31 temp = vals[inner_left_idx];
      vals[inner_left_idx] = add(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = sub(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = mul(vals[inner_right_idx], twiddle);

    }

    if (stage >= 2) {
        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
    }
  }

  unsigned new_min_stage = min_stage + log_vals_per_threads_default;
  stage = new_min_stage;

#pragma unroll
  for (; stage < new_min_stage + log_threads_per_warp; stage++) {
    const unsigned log_stride = stage - log_vals_per_threads_default - 1;

    shfl_xor_bf<log_vals_per_threads_default>(vals, log_stride, thread_idx_in_warp);
#pragma unroll
    for (unsigned i = 0; i < 1 << (log_vals_per_threads_default - 1); i++) {
      const unsigned log_inner_stride_size = stage - 1;
      const unsigned inner_pair_idx = i >> log_inner_stride_size;
      const unsigned outer_pair_idx = (warp_start + (thread_idx_in_warp << log_vals_per_threads_default)) >> stage;

      m31 twiddle  = g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx];

      const m31 temp = vals[2 * i];
      vals[2 * i] = add(temp, vals[2 * i + 1]);
      vals[2 * i + 1] = sub(temp, vals[2 * i + 1]) ;
      vals[2 * i + 1] = mul(vals[2 * i + 1], twiddle);

    }
    layer_domain_size >>= 1;
    layer_domain_offset += layer_domain_size;
  }
  shfl_xor_bf<log_vals_per_threads_default>(vals, 0, thread_idx_in_warp);

  const unsigned log_vals_per_blocks = log_vals_per_threads_default + log_threads_per_warp + LOG_WARP_PER_BLOCK;

  __shared__ m31 smem[1 << log_vals_per_blocks];

  if ((thread_idx_in_warp & 1) == 1) {
    #pragma unroll
    for (unsigned i = 0; i < (1 << log_vals_per_threads_default); i++) {
      smem[warp_start + i + (thread_idx_in_warp >> 1) * (1 << log_vals_per_threads_default) + (1 << (log_vals_per_warp - 1)) - block_start] = vals[i];
    }
  } else {
    #pragma unroll
    for (unsigned i = 0; i < (1 << log_vals_per_threads_default); i++) {
      smem[warp_start + i + (thread_idx_in_warp >> 1) * (1 << log_vals_per_threads_default) - block_start] = vals[i];
    }
  }
  __syncthreads();

#pragma unroll
  for (unsigned i = 0; i < 1 << log_vals_per_threads_default; i++) {
    vals[i] = smem[thread_idx_in_warp + (i << (log_threads_per_warp + LOG_WARP_PER_BLOCK)) + (warp_idx_in_block << log_threads_per_warp)];
  }

  new_min_stage = min_stage + log_vals_per_threads_default + log_threads_per_warp;
  stage = new_min_stage;
#pragma unroll
  for (; stage < max_stage; stage++) {
    const unsigned log_inner_stride_size = stage - new_min_stage + (log_vals_per_threads_default - LOG_WARP_PER_BLOCK);
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (log_vals_per_threads_default - 1); gid++) {
      const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
      const unsigned inner_left_idx = inner_group_idx + ((gid >> log_inner_stride_size) << (log_inner_stride_size + 1));
      const unsigned inner_right_idx = inner_left_idx + (1 << log_inner_stride_size);

      const unsigned pair_offset_in_block = block_index_y * blockDim.y * (1 << (log_vals_per_threads_default - 1)) * (1 << log_threads_per_warp);
      const unsigned pair_idx = (gid << (log_threads_per_warp + LOG_WARP_PER_BLOCK)) + thread_idx_in_warp + (threadIdx.y << log_threads_per_warp) + pair_offset_in_block;
      const unsigned tw_idx = pair_idx >> (stage - 1);
      m31 twiddle = g_twiddles[layer_domain_offset + tw_idx];

      const m31 temp = vals[inner_left_idx];
      vals[inner_left_idx] = add(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = sub(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = mul(vals[inner_right_idx], twiddle);

    }
    layer_domain_size >>= 1;
    layer_domain_offset += layer_domain_size;
  }

  m31* output_ntt_start = output[ntt_idx];
  const unsigned offset_per_vals = 1 << (log_threads_per_warp + LOG_WARP_PER_BLOCK);
  #pragma unroll
  for (unsigned i = 0; i < (1 << log_vals_per_threads_default); i++) {
    output_ntt_start[thread_idx_in_warp + (threadIdx.y << log_threads_per_warp) + i * offset_per_vals + block_start] = vals[i];
  }
}


EXTERN void ntt_b2n_init_9_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly,
                               unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 1;
    constexpr unsigned log_vals_per_threads_default = 3;
    constexpr unsigned log_threads_per_warp = 5;  // LOG_THREADS_PER_WARP
    dim3 block_dim = {1 << log_threads_per_warp, 1 << log_warp_per_block, 1}; // {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - log_threads_per_warp - log_vals_per_threads_default - log_warp_per_block);
    unsigned end_stage = start_stage + log_threads_per_warp + log_vals_per_threads_default + log_warp_per_block;

    b2n_init_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

EXTERN void ntt_b2n_init_10_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly,
                               unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 2;
    constexpr unsigned log_vals_per_threads_default = 3;
    constexpr unsigned log_threads_per_warp = 5;  // LOG_THREADS_PER_WARP
    dim3 block_dim = {1 << log_threads_per_warp, 1 << log_warp_per_block, 1}; // {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - log_threads_per_warp - log_vals_per_threads_default - log_warp_per_block);
    unsigned end_stage = start_stage + log_threads_per_warp + log_vals_per_threads_default + log_warp_per_block;

    b2n_init_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

EXTERN void ntt_b2n_init_11_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly,
                               unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 3;
    constexpr unsigned log_vals_per_threads_default = 3;
    constexpr unsigned log_threads_per_warp = 5;  // LOG_THREADS_PER_WARP
    dim3 block_dim = {1 << log_threads_per_warp, 1 << log_warp_per_block, 1}; // {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - log_threads_per_warp - log_vals_per_threads_default - log_warp_per_block);
    unsigned end_stage = start_stage + log_threads_per_warp + log_vals_per_threads_default + log_warp_per_block;
    b2n_init_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

EXTERN void ntt_b2n_init_12_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly,
                               unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 4;
    constexpr unsigned log_vals_per_threads_default = 3;
    constexpr unsigned log_threads_per_warp = 5;  // LOG_THREADS_PER_WARP
    dim3 block_dim = {1 << log_threads_per_warp, 1 << log_warp_per_block, 1}; // {32, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - log_threads_per_warp - log_vals_per_threads_default - log_warp_per_block);
    unsigned end_stage = start_stage + log_threads_per_warp + log_vals_per_threads_default + log_warp_per_block;
    b2n_init_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}
EXTERN void ntt_b2n_init_6_3_stage_batch(m31** input, m31** output,
                               unsigned log_n, unsigned num_poly,
                               unsigned start_stage, m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_warp_per_block = 2;
    constexpr unsigned log_vals_per_threads_default = 3;
    constexpr unsigned log_threads_per_warp = 1;
    dim3 block_dim = {1 << log_threads_per_warp, 1 << log_warp_per_block, 1};
    dim3 grid_dim = {};
    grid_dim.z = num_poly;
    grid_dim.x = 1 << (log_n - log_threads_per_warp - log_vals_per_threads_default - log_warp_per_block);
    unsigned end_stage = start_stage + log_threads_per_warp + log_vals_per_threads_default + log_warp_per_block;

    b2n_init_block_warp_batch<log_warp_per_block><<<grid_dim, block_dim>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

template <unsigned LOG_VALS_PER_THREAD>
__global__ void  b2n_noinit_block_batch(m31** input, m31** output,
                                  const unsigned log_n, const unsigned num_poly,
                                  unsigned min_stage, unsigned max_stage,
                                  m31 *g_twiddles, m31 rescale_factor) {
  const unsigned min_stride = 1 << (min_stage - 1);
  const unsigned log_threads_per_warp = 5;
  const unsigned num_threads_per_warp = 1 << log_threads_per_warp;
 const unsigned num_stage = 2 * LOG_VALS_PER_THREAD;

  const unsigned ntt_idx = blockIdx.z;
  const unsigned block_index_x = blockIdx.x;
  const unsigned block_index_y = blockIdx.y;
  const unsigned warp_idx_in_block = threadIdx.y;
  const unsigned thread_idx_in_warp = threadIdx.x;

  const m31* input_ntt_start = input[ntt_idx];
  const unsigned block_start = (block_index_x << log_threads_per_warp) +
      (block_index_y << (min_stage + num_stage - 1));

  m31 vals[1 << LOG_VALS_PER_THREAD];

  unsigned offset = warp_idx_in_block * (min_stride << LOG_VALS_PER_THREAD) + thread_idx_in_warp;
#pragma unroll
  for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
    vals[i] = input_ntt_start[ block_start + i * min_stride + offset];
  }

  unsigned layer_domain_size = (1 << log_n) >> 1;
  unsigned layer_domain_offset = 0;

 for (unsigned i = 2; i <= min_stage - 1; i++) {
    layer_domain_size >>= 1;
    layer_domain_offset += layer_domain_size;
  }

 unsigned stage = min_stage;
#pragma unroll
  for (; stage < min_stage + LOG_VALS_PER_THREAD; stage++) {
    const unsigned log_inner_stride_size = stage - min_stage;
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
      const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
      const unsigned inner_pair_idx = gid >> log_inner_stride_size;
      const unsigned inner_left_idx = inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
      const unsigned inner_right_idx = inner_left_idx + (1 << log_inner_stride_size);
      const unsigned outer_pair_idx = (block_start + offset) >> stage;


      m31 twiddle = g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx];

      const m31 temp = vals[inner_left_idx];
      vals[inner_left_idx] = add(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = sub(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = mul(vals[inner_right_idx], twiddle);

     }
    layer_domain_size >>= 1;
    layer_domain_offset += layer_domain_size;
  }

  __shared__ m31 smem[(num_threads_per_warp << (2 * LOG_VALS_PER_THREAD)) + num_threads_per_warp]; // 32 * (2^(2 or 3))

  unsigned offset_store_between_warps = warp_idx_in_block * (min_stride << LOG_VALS_PER_THREAD);
#pragma unroll
  for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
    smem[(i * min_stride + offset_store_between_warps) / gridDim.x + thread_idx_in_warp] = vals[i];
  }
  __syncthreads();

  unsigned new_min_stage = min_stage + LOG_VALS_PER_THREAD;
  stage = new_min_stage;
  const unsigned new_min_stride = 1 << (new_min_stage - 1);
  unsigned new_offset = ((warp_idx_in_block * (min_stride << LOG_VALS_PER_THREAD)) >> LOG_VALS_PER_THREAD) + thread_idx_in_warp;
  unsigned offset_load_between_warps = ((warp_idx_in_block * (min_stride << LOG_VALS_PER_THREAD)) >> LOG_VALS_PER_THREAD);

#pragma unroll
  for (unsigned i = 0; i < 1 << LOG_VALS_PER_THREAD; i++) {
    vals[i] = smem[(i * new_min_stride + offset_load_between_warps) / gridDim.x + thread_idx_in_warp];
  }

#pragma unroll
  for (; stage < new_min_stage + LOG_VALS_PER_THREAD; stage++) {
    const unsigned log_inner_stride_size = stage - new_min_stage;
#pragma unroll
    for (unsigned gid = 0; gid < 1 << (LOG_VALS_PER_THREAD - 1); gid++) {
      const unsigned inner_group_idx = gid & ((1 << log_inner_stride_size) - 1);
      const unsigned inner_pair_idx = gid >> log_inner_stride_size;
      const unsigned inner_left_idx = inner_group_idx + (inner_pair_idx << (log_inner_stride_size + 1));
      const unsigned inner_right_idx = inner_left_idx + (1 << log_inner_stride_size);
      const unsigned outer_pair_idx = (block_start + new_offset) >> stage;

      m31 twiddle = g_twiddles[layer_domain_offset + inner_pair_idx + outer_pair_idx];

      // m31 a_debug = vals[inner_left_idx];
      // m31 b_debug = vals[inner_right_idx];

      const m31 temp = vals[inner_left_idx];
      vals[inner_left_idx] = add(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = sub(temp, vals[inner_right_idx]);
      vals[inner_right_idx] = mul(vals[inner_right_idx], twiddle);

      if (stage == log_n) {
        vals[inner_left_idx] = mul(vals[inner_left_idx], rescale_factor);
        vals[inner_right_idx] = mul(vals[inner_right_idx], rescale_factor);
      }
    }
    layer_domain_size >>= 1;
    layer_domain_offset += layer_domain_size;
  }

  m31* output_ntt_start = output[ntt_idx];
#pragma unroll
  for (unsigned i = 0; i < (1 << LOG_VALS_PER_THREAD); i++) {
    output_ntt_start[block_start + i * new_min_stride + new_offset] = vals[i];
  }

}

EXTERN void ntt_b2n_noinit_4_stage_batch(m31** input, m31** output,
                                unsigned log_n, unsigned num_poly,
                                unsigned start_stage,
                                m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 2;
    constexpr unsigned num_threads_per_warp = 32;
    dim3 block_dim{num_threads_per_warp, 1 << log_val_per_thread, 1};
    dim3 grid_dim{};
    constexpr unsigned num_stage = 2 * log_val_per_thread;
    unsigned end_stage = start_stage + num_stage - 1;
    unsigned min_stride = 1 << (start_stage - 1);
    grid_dim.z = num_poly;
    grid_dim.x = min_stride / num_threads_per_warp;
    grid_dim.y = (1 << log_n) / (min_stride << num_stage);
    m31 rescale_factor = inv(pow(m31{2}, log_n));

    b2n_noinit_block_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles, rescale_factor);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


EXTERN void ntt_b2n_noinit_6_stage_batch(m31** input, m31** output,
                                unsigned log_n, unsigned num_poly,
                                unsigned start_stage,
                                m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 3;
    constexpr unsigned num_threads_per_warp = 32;
     dim3 block_dim{num_threads_per_warp, 1 << log_val_per_thread, 1};
    dim3 grid_dim{};
    constexpr unsigned num_stage = 2 * log_val_per_thread;
    unsigned end_stage = start_stage + num_stage - 1;
    unsigned min_stride = 1 << (start_stage - 1);
    grid_dim.z = num_poly;
    grid_dim.x = min_stride / num_threads_per_warp;
    grid_dim.y = (1 << log_n) / (1 << end_stage);
    m31 rescale_factor = inv(pow(m31{2}, log_n));
    b2n_noinit_block_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles, rescale_factor);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


EXTERN void ntt_b2n_noinit_8_stage_batch(m31** input, m31** output,
                                unsigned log_n, unsigned num_poly,
                                unsigned start_stage,
                                m31 *g_twiddles, unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    constexpr unsigned log_val_per_thread = 4;
    constexpr unsigned num_threads_per_warp = 32;
    dim3 block_dim{num_threads_per_warp, 1 << log_val_per_thread, 1};
    dim3 grid_dim{};
    constexpr unsigned num_stage = 2 * log_val_per_thread;
    unsigned end_stage = start_stage + num_stage - 1;
    unsigned min_stride = 1 << (start_stage - 1);
    grid_dim.z = num_poly;
    grid_dim.x = min_stride / num_threads_per_warp;
    grid_dim.y = (1 << log_n) / (1 << end_stage);
    m31 rescale_factor = inv(pow(m31{2}, log_n));
    b2n_noinit_block_batch<log_val_per_thread><<<grid_dim, block_dim, 0>>>(
        input, output,
        log_n, num_poly, start_stage, end_stage, g_twiddles, rescale_factor);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

__global__ void ntt_b2n_stage_batch(m31** input, m31** output,
                              unsigned log_n, unsigned stage,
                              m31 *layer_twiddles, m31 rescale_factor) {
    const unsigned ntt_index = blockIdx.y;
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = 1 << (stage - 1);
    const unsigned group_idx = gid & (stride - 1);
    const unsigned pair_idx = gid >> (stage - 1);

    const m31* input_start = input[ntt_index];
    const unsigned left_index = group_idx + pair_idx * 2 * stride;
    const unsigned right_index = left_index + stride;

    m31 left = input_start[left_index];
    m31 right = input_start[right_index];

    m31 twiddle = m31(1);

    if (stage == 1) {
        twiddle = get_circle_twiddle(layer_twiddles, pair_idx);
    } else {
        twiddle = layer_twiddles[pair_idx];
    }

    const m31 temp = left;
    m31 left_r = add(temp, right);
    m31 right_r = mul(sub(temp, right), twiddle);

    if (stage == log_n) {
        left_r = mul(left_r, rescale_factor);
        right_r = mul(right_r, rescale_factor);
    }
    m31* output_start = output[ntt_index];

    output_start[left_index] = left_r;
    output_start[right_index] = right_r;

}


EXTERN void ntt_b2n_native_batch(m31** input, m31** output,
                           unsigned log_n, unsigned num_poly,
                           unsigned start_stage,
                           unsigned end_stage,
                           m31 *g_twiddles,
                           unsigned twiddles_size, unsigned eval_domain_size) {
    g_twiddles = &g_twiddles[twiddles_size - eval_domain_size];
    dim3 block_dim{};
    block_dim.x = log_n <= 8 ? 1 << (log_n - 1) : 128;
    dim3 grid_dim{};
    grid_dim.y = num_poly;
    grid_dim.x = log_n <= 8 ? 1 : 1 << (log_n - 8);

    m31 rescale_factor = inv(pow(m31{2}, log_n));
    unsigned layer_domain_size = (1 << log_n) >> 1;
    unsigned layer_domain_offset = 0;
    if (start_stage == 1) {
        ntt_b2n_stage_batch<<<grid_dim, block_dim, 0>>>(
            input, output, log_n, 1, g_twiddles, rescale_factor);
    }

    ASSERT_TRUE(start_stage >= 1, "start_stage < 1 in ntt_n2b_native");
    ASSERT_TRUE(end_stage <= log_n, "end_stage <= log_n in ntt_n2b_native");
    for (unsigned stage = 2; stage <= end_stage; stage++) {
        if (stage >= start_stage) {
            ntt_b2n_stage_batch<<<grid_dim, block_dim, 0>>>(
                output, output, log_n, stage, &g_twiddles[layer_domain_offset], rescale_factor);
            ASSERT_CUDA_SUCCESS(cudaGetLastError());
        }
        layer_domain_size >>= 1;
        layer_domain_offset += layer_domain_size;
    }
    // No sync: stream ordering handles dependencies with next kernel.
}

EXTERN void ntt_b2n_column(
    uint32_t** values_columns,
    uint32_t log_n,
    uint32_t num_poly,
    uint32_t* g_twiddles,
    uint32_t twiddles_size,
    uint32_t eval_domain_size
) {
    // No pre-sync: stream 0 ordering guarantees prior kernel output is ready.
    m31 **device_values = clone_to_device<m31*>(values_columns, num_poly);

    if (log_n < 13) {
        ntt_b2n_native_batch(device_values, device_values, log_n, num_poly, 1, log_n, g_twiddles, twiddles_size, eval_domain_size);
    } else if (log_n >= 13 && log_n <= 18) {
        const auto& config = LAUNCH_B2N_CONFIG_13_18[log_n - 13];
        auto kernel0 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[0]) {
                case 7: return ntt_b2n_init_7_stage_batch;
                case 8: return ntt_b2n_init_8_stage_batch;
                case 9: return ntt_b2n_init_9_stage_batch;
                case 10: return ntt_b2n_init_10_stage_batch;
                case 11: return ntt_b2n_init_11_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        auto kernel1 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[1]) {
                case 4: return ntt_b2n_noinit_4_stage_batch;
                case 6: return ntt_b2n_noinit_6_stage_batch;;
                case 8: return ntt_b2n_noinit_8_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        const uint32_t start_stage1 = 1 + config[0];

        kernel0(device_values, device_values, log_n, num_poly, 1, g_twiddles, twiddles_size, eval_domain_size);
        kernel1(device_values, device_values, log_n, num_poly, start_stage1, g_twiddles, twiddles_size, eval_domain_size);

    } else if (log_n >= 19 && log_n <= 24) {
        const auto& config = LAUNCH_B2N_CONFIG_19_24[log_n - 19];

        auto kernel0 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[0]) {
                case 7: return ntt_b2n_init_7_stage_batch;
                case 8: return ntt_b2n_init_8_stage_batch;
                case 9: return ntt_b2n_init_9_stage_batch;
                case 10: return ntt_b2n_init_10_stage_batch;
                case 11: return ntt_b2n_init_11_stage_batch;
                case 14: return ntt_b2n_init_12_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        auto kernel1 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[1]) {
                case 4: return ntt_b2n_noinit_4_stage_batch;;
                case 6: return ntt_b2n_noinit_6_stage_batch;
                case 8: return ntt_b2n_noinit_8_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        auto kernel2 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[2]) {
                case 4: return ntt_b2n_noinit_4_stage_batch;
                case 6: return ntt_b2n_noinit_6_stage_batch;
                case 8: return ntt_b2n_noinit_8_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        const uint32_t start_stage1 = 1 + config[0];
        const uint32_t start_stage2 = start_stage1 + config[1];

        kernel0(device_values, device_values, log_n, num_poly, 1, g_twiddles, twiddles_size, eval_domain_size);
        kernel1(device_values, device_values, log_n, num_poly, start_stage1, g_twiddles, twiddles_size, eval_domain_size);
        kernel2(device_values, device_values, log_n, num_poly, start_stage2, g_twiddles, twiddles_size, eval_domain_size);

    } else if (log_n >= 25 && log_n <= 29) {
        const auto& config = LAUNCH_B2N_CONFIG_25_29[log_n - 25];

        auto kernel0 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[0]) {
                case 7: return ntt_b2n_init_7_stage_batch;
                case 8: return ntt_b2n_init_8_stage_batch;
                case 9: return ntt_b2n_init_9_stage_batch;
                case 10: return ntt_b2n_init_10_stage_batch;
                case 11: return ntt_b2n_init_11_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        auto kernel1 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[1]) {
                case 4: return ntt_b2n_noinit_4_stage_batch;
                case 6: return ntt_b2n_noinit_6_stage_batch;
                case 8: return ntt_b2n_noinit_8_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        auto kernel2 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[2]) {
                case 4: return ntt_b2n_noinit_4_stage_batch;;
                case 6: return ntt_b2n_noinit_6_stage_batch;
                case 8: return ntt_b2n_noinit_8_stage_batch;
                default: throw std::runtime_error("invalid config");
            }
        }();

        auto kernel3 = [config]() -> void(*)(uint32_t**, uint32_t**, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t, uint32_t) {
            switch (config[3]) {
                case 4: return ntt_b2n_noinit_4_stage_batch;;
                case 6: return ntt_b2n_noinit_6_stage_batch;;
                case 8: return ntt_b2n_noinit_8_stage_batch;
                default: throw std::runtime_error("invalid config");
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
        throw std::runtime_error("b2n log_n too big");
    }

    // No sync: async free is stream-ordered after kernel completion.
    cuda_free_memory(device_values);
}
