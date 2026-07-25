#include "eval_at_point.cuh"
#include "utils.cuh"


__global__ void eval_at_point_first_pass(m31 *g_coeffs, qm31 *temp, qm31 *factors, int coeffs_size, int factors_size,
                                         int output_offset) {
    int idx = threadIdx.x;

    qm31 *output = &temp[output_offset];

    // Thread syncing happens within a block.
    // Split the problem to feed them to multiple blocks.
    if (coeffs_size >= 512) {
        coeffs_size = 512;
    }

    // Fix: Use single shared memory array with manual offset calculation
    // to avoid overlapping memory regions
    extern __shared__ char shared_mem[];
    m31* s_coeffs = (m31*)shared_mem;
    qm31* s_level = (qm31*)(shared_mem + 512 * sizeof(m31));

    s_coeffs[idx] = g_coeffs[2 * blockIdx.x * blockDim.x + idx];
    s_coeffs[idx + blockDim.x] = g_coeffs[2 * blockIdx.x * blockDim.x + idx + blockDim.x];
    __syncthreads();

    int level_size = coeffs_size >> 1;
    int factor_idx = factors_size - 1;

    if (idx < level_size) {
        m31 alpha = s_coeffs[2 * idx];
        m31 beta = s_coeffs[2 * idx + 1];
        qm31 factor = factors[factor_idx];

        qm31 result = {
                {add(mul(beta, factor.a.a), alpha), mul(factor.a.b, beta)},
                {mul(beta, factor.b.a),             mul(beta, factor.b.b)}
        };
        s_level[idx] = result;
    }
    factor_idx -= 1;
    level_size >>= 1;

    // Fix: Move __syncthreads() outside conditional block
    // All threads must execute __syncthreads() unconditionally
    while (level_size > 0) {
        __syncthreads();
        qm31 a, b;
        if (idx < level_size) {
            a = s_level[2 * idx];
            b = s_level[2 * idx + 1];
        }
        __syncthreads();
        if (idx < level_size) {
            s_level[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        level_size >>= 1;
    }

    if (idx == 0) {
        output[blockIdx.x] = s_level[0];
    }
}

__global__
void eval_at_point_second_pass(qm31 *temp, qm31 *factors, int level_size, int factor_offset, int level_offset,
                          int output_offset) {
    int idx = threadIdx.x;

    qm31 *level = &temp[level_offset];
    qm31 *output = &temp[output_offset];

    // Thread syncing happens within a block.
    // Split the problem to feed them to multiple blocks.
    if (level_size >= 512) {
        level_size = 512;
    }

    extern __shared__ qm31 s_level[];

    s_level[idx] = level[2 * blockIdx.x * blockDim.x + idx];
    s_level[idx + blockDim.x] = level[2 * blockIdx.x * blockDim.x + idx + blockDim.x];

    level_size >>= 1;

    int factor_idx = factor_offset;

    // Fix: Move __syncthreads() outside conditional block
    // All threads must execute __syncthreads() unconditionally
    while (level_size > 0) {
        __syncthreads();
        qm31 a, b;
        if (idx < level_size) {
            a = s_level[2 * idx];
            b = s_level[2 * idx + 1];
        }
        __syncthreads();
        if (idx < level_size) {
            s_level[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        level_size >>= 1;
    }

    if (idx == 0) {
        output[blockIdx.x] = s_level[0];
    }
}

qm31 eval_at_point(m31 *coeffs, int coeffs_size, qm31 point_x, qm31 point_y) {
    int log_coeffs_size = log_2(coeffs_size);

    qm31 *host_mappings = (qm31 *) malloc(sizeof(qm31) * log_coeffs_size);
    host_mappings[log_coeffs_size - 1] = point_y;
    host_mappings[log_coeffs_size - 2] = point_x;
    qm31 x = point_x;
    for (int i = 2; i < log_coeffs_size; i += 1) {
        x = sub(mul(qm31{cm31{2, 0}, cm31{0, 0}}, mul(x, x)), qm31{cm31{1, 0}, cm31{0, 0}});
        host_mappings[log_coeffs_size - 1 - i] = x;
    }

    int temp_memory_size = 0;
    int size = coeffs_size;
    while (size > 1) {
        size = (size + 511) / 512;
        temp_memory_size += size;
    }

    qm31 *temp = cuda_malloc<qm31>(temp_memory_size);
    qm31 *device_mappings = clone_to_device<qm31>(host_mappings, log_coeffs_size);

    free(host_mappings);

    // First pass
    int block_dim = 256;
    int num_blocks = ((coeffs_size >> 1) + block_dim - 1) / block_dim;
    int shared_memory_bytes = 512 * 4 + 512 * 8;
    int output_offset = temp_memory_size - num_blocks;

    eval_at_point_first_pass<<<num_blocks, block_dim, shared_memory_bytes>>>(coeffs, temp, device_mappings, coeffs_size,
                                                                             log_coeffs_size, output_offset);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: next pass on same stream reads this output.

    // Second pass
    int mappings_offset = log_coeffs_size - 1;
    int level_offset = output_offset;
    while (num_blocks > 1) {
        mappings_offset -= 9;
        int new_num_blocks = ((num_blocks >> 1) + block_dim - 1) / block_dim;
        shared_memory_bytes = 512 * 4 * 4;
        output_offset = level_offset - new_num_blocks;
        eval_at_point_second_pass<<<new_num_blocks, block_dim, shared_memory_bytes>>>(temp, device_mappings, num_blocks,
                                                                                      mappings_offset, level_offset,
                                                                                      output_offset);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
        // No sync: stream ordering handles inter-pass dependencies.
        num_blocks = new_num_blocks;
        level_offset = output_offset;
    }

    qm31 result = qm31{cm31{0, 0}, cm31{0, 1}};
    // cudaMemcpy D2H below acts as implicit sync.
    cuda_mem_copy_device_to_host<qm31>(temp, &result, 1);

    cuda_free_memory(temp);
    cuda_free_memory(device_mappings);
    return result;
}

// =============================================================================
// Batched evaluation: evaluate N same-size polynomials at the same point
// =============================================================================

// Batched first pass: evaluates N polynomials simultaneously.
// Grid: dim3(blocks_per_poly, num_polys), Block: 256
// Each block handles one chunk of one polynomial.
// blockIdx.x = block within the polynomial, blockIdx.y = polynomial index.
__global__ void batch_eval_at_point_first_pass(
    m31 **coeffs_ptrs,       // [num_polys] array of device pointers to polynomial coefficients
    qm31 *temp,              // Flat temp buffer
    qm31 *factors,           // Shared circle power mappings
    int coeffs_size,         // All polys have the same size
    int factors_size,
    int temp_stride           // Number of blocks per polynomial in this level
) {
    int poly_idx = blockIdx.y;
    int block_in_poly = blockIdx.x;
    int idx = threadIdx.x;

    m31 *g_coeffs = coeffs_ptrs[poly_idx];
    // Output for this polynomial goes at: poly_idx * temp_stride + block_in_poly
    qm31 *output = &temp[poly_idx * temp_stride + 0];

    int local_coeffs_size = coeffs_size;
    if (local_coeffs_size >= 512) {
        local_coeffs_size = 512;
    }

    extern __shared__ char shared_mem[];
    m31* s_coeffs = (m31*)shared_mem;
    qm31* s_level = (qm31*)(shared_mem + 512 * sizeof(m31));

    s_coeffs[idx] = g_coeffs[2 * block_in_poly * blockDim.x + idx];
    s_coeffs[idx + blockDim.x] = g_coeffs[2 * block_in_poly * blockDim.x + idx + blockDim.x];
    __syncthreads();

    int level_size = local_coeffs_size >> 1;
    int factor_idx = factors_size - 1;

    if (idx < level_size) {
        m31 alpha = s_coeffs[2 * idx];
        m31 beta = s_coeffs[2 * idx + 1];
        qm31 factor = factors[factor_idx];

        qm31 result = {
                {add(mul(beta, factor.a.a), alpha), mul(factor.a.b, beta)},
                {mul(beta, factor.b.a),             mul(beta, factor.b.b)}
        };
        s_level[idx] = result;
    }
    factor_idx -= 1;
    level_size >>= 1;

    while (level_size > 0) {
        __syncthreads();
        qm31 a, b;
        if (idx < level_size) {
            a = s_level[2 * idx];
            b = s_level[2 * idx + 1];
        }
        __syncthreads();
        if (idx < level_size) {
            s_level[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        level_size >>= 1;
    }

    if (idx == 0) {
        output[block_in_poly] = s_level[0];
    }
}

// Batched second pass: hierarchical reduction for N polynomials.
// Grid: dim3(new_blocks_per_poly, num_polys), Block: 256
__global__ void batch_eval_at_point_second_pass(
    qm31 *temp,
    qm31 *factors,
    int level_size,           // Number of elements in the input level per polynomial
    int factor_offset,
    int input_stride,         // Stride between polynomials in the input level
    int output_stride,        // Stride between polynomials in the output level
    int input_offset,         // Offset in temp for the start of input data
    int output_offset         // Offset in temp for the start of output data
) {
    int poly_idx = blockIdx.y;
    int block_in_poly = blockIdx.x;
    int idx = threadIdx.x;

    qm31 *level = &temp[input_offset + poly_idx * input_stride];
    qm31 *output = &temp[output_offset + poly_idx * output_stride];

    int local_level_size = level_size;
    if (local_level_size >= 512) {
        local_level_size = 512;
    }

    extern __shared__ qm31 s_level2[];

    s_level2[idx] = level[2 * block_in_poly * blockDim.x + idx];
    s_level2[idx + blockDim.x] = level[2 * block_in_poly * blockDim.x + idx + blockDim.x];

    local_level_size >>= 1;

    int factor_idx = factor_offset;

    while (local_level_size > 0) {
        __syncthreads();
        qm31 a, b;
        if (idx < local_level_size) {
            a = s_level2[2 * idx];
            b = s_level2[2 * idx + 1];
        }
        __syncthreads();
        if (idx < local_level_size) {
            s_level2[idx] = add(a, mul(b, factors[factor_idx]));
        }
        factor_idx -= 1;
        local_level_size >>= 1;
    }

    if (idx == 0) {
        output[block_in_poly] = s_level2[0];
    }
}

// Host function: evaluate multiple same-size polynomials at the same point.
// coeffs_ptrs: device array of num_polys device pointers (each pointing to m31 coefficients)
// coeffs_size: size of each polynomial (all must be the same)
// num_polys: number of polynomials
// point_x, point_y: the evaluation point (shared across all polys)
// results: host buffer for num_polys qm31 results
extern "C"
void batch_eval_at_points(
    m31 **coeffs_ptrs,
    int coeffs_size,
    int num_polys,
    qm31 point_x,
    qm31 point_y,
    qm31 *results
) {
    if (num_polys == 0) return;

    int log_coeffs_size = log_2(coeffs_size);

    // 1. Compute circle power mappings on CPU (same as single-poly version)
    qm31 *host_mappings = (qm31 *) malloc(sizeof(qm31) * log_coeffs_size);
    host_mappings[log_coeffs_size - 1] = point_y;
    host_mappings[log_coeffs_size - 2] = point_x;
    qm31 x = point_x;
    for (int i = 2; i < log_coeffs_size; i += 1) {
        x = sub(mul(qm31{cm31{2, 0}, cm31{0, 0}}, mul(x, x)), qm31{cm31{1, 0}, cm31{0, 0}});
        host_mappings[log_coeffs_size - 1 - i] = x;
    }

    // 2. Upload mappings to device (one copy for all polys)
    qm31 *device_mappings = clone_to_device<qm31>(host_mappings, log_coeffs_size);
    free(host_mappings);

    // 3. Compute temp memory layout per polynomial
    // Each reduction level produces ceil(prev_size / 512) elements per polynomial.
    // We store all levels for all polynomials in a single flat buffer.
    // Layout: [level0 for all polys | level1 for all polys | ...]
    // Within each level: [poly0 data | poly1 data | ... | polyN data]

    // Compute the number of blocks at each reduction level
    int max_levels = 32;
    int *level_sizes = (int *) malloc(sizeof(int) * max_levels);
    int num_levels = 0;
    int size = coeffs_size;

    // First level: from coeffs_size to num_blocks after first pass
    int first_num_blocks = ((size >> 1) + 255) / 256;
    level_sizes[0] = first_num_blocks;
    num_levels = 1;
    size = first_num_blocks;

    while (size > 1) {
        int new_size = ((size >> 1) + 255) / 256;
        level_sizes[num_levels] = new_size;
        num_levels++;
        size = new_size;
    }

    // Compute offsets for each level in the flat buffer
    // Level i data starts at offset level_offsets[i], with stride level_sizes[i] per polynomial
    int *level_offsets = (int *) malloc(sizeof(int) * num_levels);
    int total_temp = 0;
    for (int i = 0; i < num_levels; i++) {
        level_offsets[i] = total_temp;
        total_temp += level_sizes[i] * num_polys;
    }

    // 4. Allocate temp buffer
    qm31 *temp = cuda_malloc<qm31>(total_temp);

    // 5. Launch batched first pass
    int block_dim = 256;
    int bpp = first_num_blocks; // blocks per polynomial
    // Shared memory: 512 m31 coefficients (2048B) + 256 qm31 reduction entries (4096B) = 6144B
    int shared_memory_bytes = 512 * 4 + 512 * 8;
    dim3 grid(bpp, num_polys);

    batch_eval_at_point_first_pass<<<grid, block_dim, shared_memory_bytes>>>(
        coeffs_ptrs, temp, device_mappings, coeffs_size, log_coeffs_size, level_sizes[0]
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // 6. Reduction passes (stream ordering handles inter-pass dependencies)
    int mappings_offset = log_coeffs_size - 1;
    int current_num_blocks = first_num_blocks;

    for (int lvl = 1; lvl < num_levels; lvl++) {
        mappings_offset -= 9;
        int new_num_blocks = level_sizes[lvl];
        int shared_mem = 512 * 4 * 4;  // 512 qm31 entries

        dim3 grid2(new_num_blocks, num_polys);
        batch_eval_at_point_second_pass<<<grid2, block_dim, shared_mem>>>(
            temp, device_mappings, current_num_blocks, mappings_offset,
            level_sizes[lvl - 1],  // input stride per poly
            level_sizes[lvl],      // output stride per poly
            level_offsets[lvl - 1], // input offset
            level_offsets[lvl]      // output offset
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
        current_num_blocks = new_num_blocks;
    }

    // 7. Extract results: each poly's final result is at temp[last_level_offset + poly_idx * 1]
    // The last level has 1 element per polynomial, stride = 1
    int last_offset = level_offsets[num_levels - 1];

    // Copy all results from device to host at once
    // Results are contiguous: temp[last_offset], temp[last_offset+1], ..., temp[last_offset+num_polys-1]
    // Async D2H + stream sync (avoids implicit full-device sync from cudaMemcpy).
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(results, &temp[last_offset], sizeof(qm31) * num_polys, cudaMemcpyDeviceToHost, 0));
    cudaStreamSynchronize(0);

    // 8. Cleanup
    cuda_free_memory(temp);
    cuda_free_memory(device_mappings);
    free(level_sizes);
    free(level_offsets);
}

