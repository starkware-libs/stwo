#include "barycentric.cuh"
#include "batch_inverse.cuh"
#include "point.cuh"
#include "utils.cuh"

// Duplicated from quotients.cu to avoid cross-file dependencies.
// These are small inline helper functions.

static HOST_DEVICE_FORCEINLINE point bary_index_to_point(uint32_t index) {
    return point_pow(m31_circle_gen, (int)index);
}

static DEVICE_FORCEINLINE point bary_domain_at_index(
    uint32_t half_coset_initial_index,
    uint32_t half_coset_step_size,
    uint32_t index,
    uint32_t domain_size
) {
    uint32_t half_coset_size = domain_size >> 1;
    uint32_t modulo_u31_mask = 0x7fffffff;

    if (index < half_coset_size) {
        uint64_t global_index = (uint64_t)half_coset_initial_index
            + (uint64_t)half_coset_step_size * (uint64_t)index;
        return bary_index_to_point(global_index & modulo_u31_mask);
    } else {
        uint64_t global_index = (uint64_t)half_coset_initial_index
            + (uint64_t)half_coset_step_size * (uint64_t)(index - half_coset_size);
        return bary_index_to_point((2147483648ULL - global_index) & modulo_u31_mask);
    }
}

// double_x(x) = 2*x^2 - 1 (the circle doubling formula for x-coordinate)
static DEVICE_FORCEINLINE m31 double_x_m31(m31 x) {
    m31 sx = mul(x, x);
    return sub(add(sx, sx), 1);
}

// Kernel 1: Precompute numerators and denominators for barycentric weights.
//
// For each domain point i:
//   1. Compute bit-reversed index and get domain point (x_cp, y_cp) in BaseField.
//   2. Compute coset_vanishing_derivative at the domain point:
//      derivative = product of coset_vanishing(CanonicCoset::new(k).coset, cp) for k=1..log_size-1
//      For a CanonicCoset base field point, coset_vanishing reduces to repeated double_x.
//   3. si_i = -2 * y_cp * exp_val * derivative  (all in M31)
//   4. h = p - cp  (circle subtraction: h.x = p.x*cp.x + p.y*cp.y, h.y = p.y*cp.x - p.x*cp.y)
//      Since p is QM31 and cp is M31, h is QM31.
//   5. denom[i] = si_i * h.y  (M31 * QM31 = QM31)
//   6. result[i] = vn_p * (1 + h.x)  (QM31 * QM31 = QM31, storing numerator)
__global__ void barycentric_precompute_kernel(
    uint32_t half_coset_initial_index,
    uint32_t half_coset_step_size,
    int domain_size,
    int log_size,
    qm31 vn_p,
    qm31 p_x,
    qm31 p_y,
    m31 exp_val,
    qm31 *denom,
    qm31 *result
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= domain_size) return;

    // 1. Bit-reverse to get the actual domain index
    uint32_t br_i = bit_reverse(i, log_size);

    // 2. Get domain point (BaseField)
    point cp = bary_domain_at_index(
        half_coset_initial_index, half_coset_step_size, br_i, domain_size
    );

    // 3. Compute coset_vanishing_derivative for the domain point.
    //    For CanonicCoset, coset_vanishing(CanonicCoset::new(k).coset, cp)
    //    simplifies to just repeated double_x starting from cp.x.
    //
    //    coset_vanishing(CanonicCoset::new(k).coset, cp):
    //      The coset is Coset::odds(k), with initial = G_{2k+2}^1 and step = G_{2k}^1.
    //      The rotation p' = p - initial + step_size.half() cancels for CanonicCoset
    //      when the point is on the canonical domain. But since the domain point is
    //      BaseField and may not be on the exact coset, we need to do the full computation.
    //
    //    Actually, for CanonicCoset::new(k).coset = Coset::odds(k):
    //      initial_index = generator(k+1), step_size = generator(k)
    //      step_size.half() = generator(k+1) = initial_index
    //      So rotation = p - initial + initial = p (identity rotation!)
    //      Therefore coset_vanishing just does (k-1) double_x operations on p.x.
    //
    //    derivative = product of coset_vanishing(CanonicCoset::new(k).coset, cp) for k in 1..log_size
    //    For k=1: coset_vanishing does 0 double_x ops, returns cp.x
    //    For k=2: does 1 double_x op, returns double_x(cp.x)
    //    For k=j: does (j-1) double_x ops
    //
    //    So derivative = cp.x * double_x(cp.x) * double_x^2(cp.x) * ... * double_x^(log_size-2)(cp.x)
    m31 derivative = 1; // Start with 1 (identity for multiplication)
    m31 x_doubled = cp.x;
    for (int k = 1; k < log_size; k++) {
        // coset_vanishing(CanonicCoset::new(k).coset, cp) = double_x^(k-1)(cp.x)
        derivative = mul(derivative, x_doubled);
        x_doubled = double_x_m31(x_doubled);
    }

    // si_i = -2 * cp.y * exp_val * derivative
    m31 si_i = mul(neg(2), mul(cp.y, mul(exp_val, derivative)));

    // 4. Circle subtraction: h = p - cp
    //    For circle points, (p - cp) means: h = p * conj(cp)
    //    p is QM31, cp is M31, conj(cp) = (cp.x, -cp.y)
    //    h.x = p.x * cp.x + p.y * cp.y
    //    h.y = p.y * cp.x - p.x * cp.y
    qm31 h_x = add(mul_by_scalar(p_x, cp.x), mul_by_scalar(p_y, cp.y));
    qm31 h_y = sub(mul_by_scalar(p_y, cp.x), mul_by_scalar(p_x, cp.y));

    // 5. denom[i] = si_i * h.y
    denom[i] = mul(si_i, h_y);

    // 6. result[i] = vn_p * (1 + h.x)   [store numerator in result buffer]
    qm31 one_plus_hx = add((m31)1, h_x);
    result[i] = mul(vn_p, one_plus_hx);
}

// Kernel 2: Assemble final weights by multiplying numerators with inverse denominators.
//   result[i] = result[i] * inv_denom[i]
__global__ void barycentric_assemble_kernel(
    qm31 *result,
    qm31 *inv_denom,
    int domain_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= domain_size) return;

    result[i] = mul(result[i], inv_denom[i]);
}

// Host wrapper: barycentric_weights_cuda
//   1. Allocate temp: denom[domain_size], inv_denom[domain_size]
//   2. Launch precompute_kernel -> fills denom and result (numerators)
//   3. batch_inverse_secure_field(denom -> inv_denom)
//   4. Launch assemble_kernel -> result[i] = result[i] * inv_denom[i]
//   5. Free denom, inv_denom
void barycentric_weights_cuda(
    uint32_t half_coset_initial_index,
    uint32_t half_coset_step_size,
    int domain_size,
    int log_size,
    qm31 vn_p,
    qm31 p_x,
    qm31 p_y,
    m31 exp_val,
    qm31 *result
) {
    // Allocate temporary GPU buffers
    qm31 *denom = cuda_malloc<qm31>(domain_size);
    qm31 *inv_denom = cuda_malloc<qm31>(domain_size);

    // Step 1: Precompute numerators (into result) and denominators (into denom)
    int block_dim = 256;
    int num_blocks = (domain_size + block_dim - 1) / block_dim;

    barycentric_precompute_kernel<<<num_blocks, block_dim>>>(
        half_coset_initial_index, half_coset_step_size,
        domain_size, log_size,
        vn_p, p_x, p_y, exp_val,
        denom, result
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: batch_inverse on same stream reads denom after this kernel.

    // Step 2: Batch inverse of denominators
    batch_inverse_secure_field(denom, inv_denom, domain_size);

    // Step 3: Assemble final weights: result[i] *= inv_denom[i]
    barycentric_assemble_kernel<<<num_blocks, block_dim>>>(
        result, inv_denom, domain_size
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: async frees are stream-ordered.
    cuda_free_memory(denom);
    cuda_free_memory(inv_denom);
}

// Kernel 3: Dot product of M31 evals with QM31 weights using shared memory reduction.
//   Each thread computes partial sums via grid-stride loop.
//   Block-level tree reduction in shared memory.
//   Output: partial_sums[blockIdx.x] (one QM31 per block).
#define DOT_BLOCK_DIM 256
#define DOT_NUM_BLOCKS 256

__global__ void m31_qm31_dot_product_kernel(
    m31 *evals,
    qm31 *weights,
    int size,
    qm31 *partial_sums
) {
    __shared__ qm31 sdata[DOT_BLOCK_DIM];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // Grid-stride accumulation
    qm31 local_sum = {cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};
    for (int i = global_id; i < size; i += grid_stride) {
        local_sum = add(local_sum, mul(evals[i], weights[i]));
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Block-level tree reduction
    for (int s = DOT_BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = add(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Host wrapper: barycentric_eval_at_point_cuda
//   1. Allocate DOT_NUM_BLOCKS QM31 partial_sums on GPU
//   2. Launch dot_product_kernel
//   3. Download partial_sums to CPU (4KB)
//   4. CPU final sum (DOT_NUM_BLOCKS QM31 additions)
//   5. Write result to host_result
void barycentric_eval_at_point_cuda(
    m31 *evals,
    qm31 *weights,
    int size,
    qm31 *host_result
) {
    qm31 *partial_sums = cuda_malloc<qm31>(DOT_NUM_BLOCKS);

    m31_qm31_dot_product_kernel<<<DOT_NUM_BLOCKS, DOT_BLOCK_DIM>>>(
        evals, weights, size, partial_sums
    );
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Download partial sums to CPU
    qm31 host_partial[DOT_NUM_BLOCKS];
    cuda_mem_copy_device_to_host(partial_sums, host_partial, DOT_NUM_BLOCKS);

    // CPU final reduction
    qm31 final_sum = {cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};
    for (int i = 0; i < DOT_NUM_BLOCKS; i++) {
        // Manual QM31 addition on CPU (host-side)
        final_sum.a.a = (uint32_t)(((uint64_t)final_sum.a.a + (uint64_t)host_partial[i].a.a) % 2147483647ULL);
        final_sum.a.b = (uint32_t)(((uint64_t)final_sum.a.b + (uint64_t)host_partial[i].a.b) % 2147483647ULL);
        final_sum.b.a = (uint32_t)(((uint64_t)final_sum.b.a + (uint64_t)host_partial[i].b.a) % 2147483647ULL);
        final_sum.b.b = (uint32_t)(((uint64_t)final_sum.b.b + (uint64_t)host_partial[i].b.b) % 2147483647ULL);
    }

    *host_result = final_sum;

    cuda_free_memory(partial_sums);
}
