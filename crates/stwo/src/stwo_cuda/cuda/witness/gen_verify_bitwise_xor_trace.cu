#include <cstdio>
#include <vector>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "eval_at_row.cuh"
#include <stdint.h>

#include "gen_verify_bitwise_xor_trace.cuh"


__global__ void verify_bitwise_xor_4_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_4_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_4_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_4 base trace");

    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_4_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);

    global_timer.end("generate verify_bitwise_xor_4 base trace");
}

__global__ void verify_bitwise_xor_7_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_7_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_7_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_7 base trace");

    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_7_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);

    global_timer.end("generate verify_bitwise_xor_7 base trace");
}

__global__ void verify_bitwise_xor_8_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_8_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_8_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_8 base trace");

    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_8_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);

    global_timer.end("generate verify_bitwise_xor_8 base trace");
}

// verify_bitwise_xor_8_b uses the same 8-bit kernel as verify_bitwise_xor_8
void verify_bitwise_xor_8_b_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_8_b base trace");

    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    // Reuse the same kernel as verify_bitwise_xor_8 since they have the same bit width
    verify_bitwise_xor_8_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);

    global_timer.end("generate verify_bitwise_xor_8_b base trace");
}

__global__ void verify_bitwise_xor_9_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row] << VERIFY_BITWISE_9_N_BITS) + inputs[1][row];
        atomicAdd(&mults[addr], 1);
    }
}


void verify_bitwise_xor_9_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_9 base trace");

    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_9_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);

    global_timer.end("generate verify_bitwise_xor_9 base trace");
}


__global__ void verify_bitwise_xor_12_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults,
    unsigned mults_col_size,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        m31 a = m31 {inputs[0][row]};
        m31 b = m31 {inputs[1][row]};

        uint32_t al = (uint32_t)(a & ((1ULL << VERIFY_BITWISE_12_LIMB_BITS) - 1));
        uint32_t ah = (uint32_t)(a >> VERIFY_BITWISE_12_LIMB_BITS);
        uint32_t bl = (uint32_t)(b & ((1ULL << VERIFY_BITWISE_12_LIMB_BITS) - 1));
        uint32_t bh = (uint32_t)(b >> VERIFY_BITWISE_12_LIMB_BITS);

        uint32_t column_index = (ah << VERIFY_BITWISE_12_EXPAND_BITS) + bh;
        uint32_t row_index = (al << VERIFY_BITWISE_12_LIMB_BITS) + bl;
        atomicAdd(&mults[column_index][row_index], 1);
    }
}


void verify_bitwise_xor_12_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults,
    unsigned mults_col_size,
    unsigned mults_row_log_size
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_12 base trace");

    m31 **device_inputs = clone_to_device<m31*>(inputs, 3 * input_col_sizes);
    m31 **device_mults = clone_to_device<m31*>(mults, mults_col_size);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    verify_bitwise_xor_12_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        device_mults,
        mults_col_size,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
    cuda_free_memory(device_mults);

    global_timer.end("generate verify_bitwise_xor_12 base trace");
}

// =============================================================================
// Interaction trace generation for verify_bitwise_xor components
// =============================================================================

#define VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX 256

// Template kernel for single-column verify_bitwise_xor interaction trace (4, 7, 8, 8_b, 9)
// Computes a, b, c from row index using preprocessed BitwiseXor pattern
template <int N_BITS>
__global__ void verify_bitwise_xor_interaction_trace_col_gen_kernel(
    LookupElementsBasic<4> *lookup_elements,
    m31 relation_id,
    m31 *multiplicities,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Compute a, b, c from row index (BitwiseXor preprocessed pattern)
        uint32_t mask = (1 << N_BITS) - 1;
        m31 a = m31{row >> N_BITS};
        m31 b = m31{row & mask};
        m31 c = m31{(row >> N_BITS) ^ (row & mask)};

        // numerator = -multiplicities[row]
        m31 mult = multiplicities[row];
        qm31 numerator = qm31{cm31{P - mult, 0}, cm31{0, 0}};

        // denominator = lookup_elements.combine([relation_id, a, b, c])
        m31 values[4] = {relation_id, a, b, c};
        qm31 denom = lookup_elements->combine(values, 4);

        logup_col_write_frac(row, numerator, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize column kernel - compute value = numerator * denom_inv and accumulate
__global__ void verify_bitwise_xor_interaction_trace_finalize_col_kernel(
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        qm31 numerator = qm31{
            cm31{numerator0[row], numerator1[row]},
            cm31{numerator2[row], numerator3[row]}
        };
        qm31 value = mul(numerator, denom_inv_ptr[row]);

        // Store to interaction trace columns
        interaction_traces[0][row] = value.a.a;
        interaction_traces[1][row] = value.a.b;
        interaction_traces[2][row] = value.b.a;
        interaction_traces[3][row] = value.b.b;
    }
}

// Compute cumsum shift - sum all values in the last interaction trace column
__global__ void verify_bitwise_xor_interaction_trace_cumsum_shift(
    unsigned trace_size,
    m31 **interaction_traces,
    m31 *coordinate_sums
) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned gridSize = gridDim.x * blockDim.x;

    m31 sum0 = 0;
    m31 sum1 = 0;
    m31 sum2 = 0;
    m31 sum3 = 0;

    for (unsigned i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_traces[0][i]);
        sum1 = add(sum1, interaction_traces[1][i]);
        sum2 = add(sum2, interaction_traces[2][i]);
        sum3 = add(sum3, interaction_traces[3][i]);
    }

    extern __shared__ m31 shared[];
    m31* sdata0 = &shared[0];
    m31* sdata1 = &shared[blockDim.x];
    m31* sdata2 = &shared[2 * blockDim.x];
    m31* sdata3 = &shared[3 * blockDim.x];

    sdata0[threadIdx.x] = sum0;
    sdata1[threadIdx.x] = sum1;
    sdata2[threadIdx.x] = sum2;
    sdata3[threadIdx.x] = sum3;

    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata0[threadIdx.x] = add(sdata0[threadIdx.x], sdata0[threadIdx.x + s]);
            sdata1[threadIdx.x] = add(sdata1[threadIdx.x], sdata1[threadIdx.x + s]);
            sdata2[threadIdx.x] = add(sdata2[threadIdx.x], sdata2[threadIdx.x + s]);
            sdata3[threadIdx.x] = add(sdata3[threadIdx.x], sdata3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sdata0[0]);
        atomic_add(&coordinate_sums[1], sdata1[0]);
        atomic_add(&coordinate_sums[2], sdata2[0]);
        atomic_add(&coordinate_sums[3], sdata3[0]);
    }
}

// Apply cumsum shift to interaction traces
__global__ void verify_bitwise_xor_interaction_trace_apply_shift(
    m31 *coordinate_sums,
    unsigned trace_size,
    m31 **interaction_traces
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        qm31 claimed_sum = qm31{
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interaction_traces[0][row] = sub(interaction_traces[0][row], cumsum_shift.a.a);
        interaction_traces[1][row] = sub(interaction_traces[1][row], cumsum_shift.a.b);
        interaction_traces[2][row] = sub(interaction_traces[2][row], cumsum_shift.b.a);
        interaction_traces[3][row] = sub(interaction_traces[3][row], cumsum_shift.b.b);
    }
}

// Host function for verify_bitwise_xor_4 interaction trace
void verify_bitwise_xor_4_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_4 interaction trace");

    unsigned trace_size = 1 << log_size;

    // Copy lookup elements to device
    LookupElementsBasic<4> *device_lookup_elements = cuda_malloc<LookupElementsBasic<4>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)lookup_elements, device_lookup_elements, 1);

    // Allocate temporary buffers
    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4);

    int block_dim = trace_size < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? trace_size : VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // Step 1: Generate logup fractions
    verify_bitwise_xor_interaction_trace_col_gen_kernel<VERIFY_BITWISE_4_N_BITS><<<num_blocks, block_dim>>>(
        device_lookup_elements,
        VERIFY_BITWISE_XOR_4_RELATION_ID,
        multiplicities,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 2: Batch inverse
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    // Step 3: Finalize column
    verify_bitwise_xor_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 4: Compute cumsum shift
    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_bitwise_xor_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 5: Apply shift
    verify_bitwise_xor_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 6: Prefix sum
    inclusive_prefix_sum(interaction_traces[0], trace_size);
    inclusive_prefix_sum(interaction_traces[1], trace_size);
    inclusive_prefix_sum(interaction_traces[2], trace_size);
    inclusive_prefix_sum(interaction_traces[3], trace_size);

    // Cleanup
    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate verify_bitwise_xor_4 interaction trace");
}

// Host function for verify_bitwise_xor_7 interaction trace
void verify_bitwise_xor_7_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_7 interaction trace");

    unsigned trace_size = 1 << log_size;

    LookupElementsBasic<4> *device_lookup_elements = cuda_malloc<LookupElementsBasic<4>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)lookup_elements, device_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4);

    int block_dim = trace_size < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? trace_size : VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    verify_bitwise_xor_interaction_trace_col_gen_kernel<VERIFY_BITWISE_7_N_BITS><<<num_blocks, block_dim>>>(
        device_lookup_elements,
        VERIFY_BITWISE_XOR_7_RELATION_ID,
        multiplicities,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    verify_bitwise_xor_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_bitwise_xor_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    verify_bitwise_xor_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[0], trace_size);
    inclusive_prefix_sum(interaction_traces[1], trace_size);
    inclusive_prefix_sum(interaction_traces[2], trace_size);
    inclusive_prefix_sum(interaction_traces[3], trace_size);

    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate verify_bitwise_xor_7 interaction trace");
}

// Host function for verify_bitwise_xor_8 interaction trace
void verify_bitwise_xor_8_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_8 interaction trace");

    unsigned trace_size = 1 << log_size;

    LookupElementsBasic<4> *device_lookup_elements = cuda_malloc<LookupElementsBasic<4>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)lookup_elements, device_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4);

    int block_dim = trace_size < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? trace_size : VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    verify_bitwise_xor_interaction_trace_col_gen_kernel<VERIFY_BITWISE_8_N_BITS><<<num_blocks, block_dim>>>(
        device_lookup_elements,
        VERIFY_BITWISE_XOR_8_RELATION_ID,
        multiplicities,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    verify_bitwise_xor_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_bitwise_xor_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    verify_bitwise_xor_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[0], trace_size);
    inclusive_prefix_sum(interaction_traces[1], trace_size);
    inclusive_prefix_sum(interaction_traces[2], trace_size);
    inclusive_prefix_sum(interaction_traces[3], trace_size);

    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate verify_bitwise_xor_8 interaction trace");
}

// Host function for verify_bitwise_xor_8_b interaction trace
void verify_bitwise_xor_8_b_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_8_b interaction trace");

    unsigned trace_size = 1 << log_size;

    LookupElementsBasic<4> *device_lookup_elements = cuda_malloc<LookupElementsBasic<4>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)lookup_elements, device_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4);

    int block_dim = trace_size < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? trace_size : VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // Use VERIFY_BITWISE_8_N_BITS since 8_b has the same bit width as 8
    verify_bitwise_xor_interaction_trace_col_gen_kernel<VERIFY_BITWISE_8_N_BITS><<<num_blocks, block_dim>>>(
        device_lookup_elements,
        VERIFY_BITWISE_XOR_8_B_RELATION_ID,
        multiplicities,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    verify_bitwise_xor_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_bitwise_xor_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    verify_bitwise_xor_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[0], trace_size);
    inclusive_prefix_sum(interaction_traces[1], trace_size);
    inclusive_prefix_sum(interaction_traces[2], trace_size);
    inclusive_prefix_sum(interaction_traces[3], trace_size);

    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate verify_bitwise_xor_8_b interaction trace");
}

// Host function for verify_bitwise_xor_9 interaction trace
void verify_bitwise_xor_9_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_9 interaction trace");

    unsigned trace_size = 1 << log_size;

    LookupElementsBasic<4> *device_lookup_elements = cuda_malloc<LookupElementsBasic<4>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)lookup_elements, device_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4);

    int block_dim = trace_size < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? trace_size : VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    verify_bitwise_xor_interaction_trace_col_gen_kernel<VERIFY_BITWISE_9_N_BITS><<<num_blocks, block_dim>>>(
        device_lookup_elements,
        VERIFY_BITWISE_XOR_9_RELATION_ID,
        multiplicities,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    verify_bitwise_xor_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_bitwise_xor_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    verify_bitwise_xor_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[0], trace_size);
    inclusive_prefix_sum(interaction_traces[1], trace_size);
    inclusive_prefix_sum(interaction_traces[2], trace_size);
    inclusive_prefix_sum(interaction_traces[3], trace_size);

    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate verify_bitwise_xor_9 interaction trace");
}

// =============================================================================
// verify_bitwise_xor_8 paired interaction trace (dual-relation: VBX-8 + VBX-8_B)
// =============================================================================

// Col-gen kernel for vbx_8 paired: combines VBX-8 and VBX-8_B into one logup column.
// numerator = -(denom0 * mults_1 + denom1 * mults_0)
// denominator = denom0 * denom1
// where denom0 = combine([VERIFY_BITWISE_XOR_8_RELATION_ID, a, b, c])
//       denom1 = combine([VERIFY_BITWISE_XOR_8_B_RELATION_ID, a, b, c])
__global__ void verify_bitwise_xor_8_paired_interaction_trace_col_gen_kernel(
    LookupElementsBasic<4> *lookup_elements,
    m31 *mults0_ptr,
    m31 *mults1_ptr,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        uint32_t a_val = row >> VERIFY_BITWISE_8_N_BITS;
        uint32_t b_val = row & ((1 << VERIFY_BITWISE_8_N_BITS) - 1);
        uint32_t c_val = a_val ^ b_val;

        m31 a = m31{a_val};
        m31 b = m31{b_val};
        m31 c = m31{c_val};

        // combine([relation_id, a, b, c]) for both relations
        m31 values0[4] = {VERIFY_BITWISE_XOR_8_RELATION_ID, a, b, c};
        m31 values1[4] = {VERIFY_BITWISE_XOR_8_B_RELATION_ID, a, b, c};
        qm31 denom0 = lookup_elements->combine(values0, 4);
        qm31 denom1 = lookup_elements->combine(values1, 4);

        // numerator = -(denom0 * mults_1 + denom1 * mults_0)
        m31 m0 = mults0_ptr[row];
        m31 m1 = mults1_ptr[row];
        qm31 neg_m0 = qm31{cm31{P - m0, 0}, cm31{0, 0}};
        qm31 neg_m1 = qm31{cm31{P - m1, 0}, cm31{0, 0}};
        qm31 numerator = add(mul(denom0, neg_m1), mul(denom1, neg_m0));

        // denominator = denom0 * denom1
        qm31 denom = mul(denom0, denom1);

        logup_col_write_frac(row, numerator, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Host function for verify_bitwise_xor_8 paired interaction trace
void verify_bitwise_xor_8_paired_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities_0,
    m31 *multiplicities_1,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_8 paired interaction trace");

    unsigned trace_size = 1 << log_size;

    LookupElementsBasic<4> *device_lookup_elements = cuda_malloc<LookupElementsBasic<4>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)lookup_elements, device_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4);

    int block_dim = trace_size < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? trace_size : VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    verify_bitwise_xor_8_paired_interaction_trace_col_gen_kernel<<<num_blocks, block_dim>>>(
        device_lookup_elements,
        multiplicities_0,
        multiplicities_1,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    verify_bitwise_xor_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_bitwise_xor_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    verify_bitwise_xor_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[0], trace_size);
    inclusive_prefix_sum(interaction_traces[1], trace_size);
    inclusive_prefix_sum(interaction_traces[2], trace_size);
    inclusive_prefix_sum(interaction_traces[3], trace_size);

    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate verify_bitwise_xor_8 paired interaction trace");
}

// =============================================================================
// verify_bitwise_xor_12 interaction trace (expanded 16-column layout)
// =============================================================================

// Col-gen kernel for vbx_12: batches 2 lookups per column pair.
// For column pair (i0=2p, i1=2p+1), computes the logup fraction:
//   numerator = p0 * (-mults1[row]) + p1 * (-mults0[row])
//   denominator = p1 * p0
// where p0 = combine([relation_id, a0,b0,c0]), p1 = combine([relation_id, a1,b1,c1]).
__global__ void verify_bitwise_xor_12_interaction_trace_col_gen_kernel(
    LookupElementsBasic<4> *lookup_elements,
    m31 *mults0_ptr,
    m31 *mults1_ptr,
    unsigned ah0,
    unsigned bh0,
    unsigned ah1,
    unsigned bh1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Decode row index into low parts: al (high LIMB_BITS), bl (low LIMB_BITS)
        uint32_t al = row >> VERIFY_BITWISE_12_LIMB_BITS;
        uint32_t bl = row & ((1 << VERIFY_BITWISE_12_LIMB_BITS) - 1);

        // Reconstruct full a, b, c for both lookups
        m31 a0 = m31{(ah0 << VERIFY_BITWISE_12_LIMB_BITS) | al};
        m31 b0 = m31{(bh0 << VERIFY_BITWISE_12_LIMB_BITS) | bl};
        m31 c0 = m31{((ah0 << VERIFY_BITWISE_12_LIMB_BITS) | al) ^ ((bh0 << VERIFY_BITWISE_12_LIMB_BITS) | bl)};

        m31 a1 = m31{(ah1 << VERIFY_BITWISE_12_LIMB_BITS) | al};
        m31 b1 = m31{(bh1 << VERIFY_BITWISE_12_LIMB_BITS) | bl};
        m31 c1 = m31{((ah1 << VERIFY_BITWISE_12_LIMB_BITS) | al) ^ ((bh1 << VERIFY_BITWISE_12_LIMB_BITS) | bl)};

        // combine([relation_id, a, b, c]) for both lookups
        m31 values0[4] = {VERIFY_BITWISE_XOR_12_RELATION_ID, a0, b0, c0};
        m31 values1[4] = {VERIFY_BITWISE_XOR_12_RELATION_ID, a1, b1, c1};
        qm31 p0 = lookup_elements->combine(values0, 4);
        qm31 p1 = lookup_elements->combine(values1, 4);

        // numerator = p0 * (-mults1[row]) + p1 * (-mults0[row])
        m31 m0 = mults0_ptr[row];
        m31 m1 = mults1_ptr[row];
        qm31 neg_m0 = qm31{cm31{P - m0, 0}, cm31{0, 0}};
        qm31 neg_m1 = qm31{cm31{P - m1, 0}, cm31{0, 0}};
        qm31 numerator = add(mul(p0, neg_m1), mul(p1, neg_m0));

        // denominator = p1 * p0
        qm31 denom = mul(p1, p0);

        logup_col_write_frac(row, numerator, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Accumulate kernel: computes value = numerator * denom_inv, adds prev pair's
// accumulated value, and writes running sum to interaction_traces[pair_index*4..+4].
// This matches the SIMD LogupTraceGenerator::finalize_col() behavior where each
// column stores the cumulative sum of all fractions up to and including that pair.
__global__ void verify_bitwise_xor_12_interaction_trace_accumulate_kernel(
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces,
    unsigned pair_index
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        qm31 numerator = qm31{
            cm31{numerator0[row], numerator1[row]},
            cm31{numerator2[row], numerator3[row]}
        };
        qm31 value = mul(numerator, denom_inv_ptr[row]);

        // Read previous pair's accumulated value (or zero for first pair)
        qm31 prev;
        if (pair_index > 0) {
            unsigned prev_base = (pair_index - 1) * 4;
            prev = qm31{
                cm31{interaction_traces[prev_base][row], interaction_traces[prev_base + 1][row]},
                cm31{interaction_traces[prev_base + 2][row], interaction_traces[prev_base + 3][row]}
            };
        } else {
            prev = qm31{cm31{0, 0}, cm31{0, 0}};
        }

        qm31 result = add(value, prev);
        unsigned base = pair_index * 4;
        interaction_traces[base][row] = result.a.a;
        interaction_traces[base + 1][row] = result.a.b;
        interaction_traces[base + 2][row] = result.b.a;
        interaction_traces[base + 3][row] = result.b.b;
    }
}

// Host function for verify_bitwise_xor_12 interaction trace
void verify_bitwise_xor_12_interaction_trace(
    void *lookup_elements,
    m31 **multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate verify_bitwise_xor_12 interaction trace");

    unsigned trace_size = 1 << log_size;

    // Copy lookup elements to device
    LookupElementsBasic<4> *device_lookup_elements = cuda_malloc<LookupElementsBasic<4>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<4>>((LookupElementsBasic<4>*)lookup_elements, device_lookup_elements, 1);

    // Allocate temporary buffers (reused across all 8 pairs)
    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    // Clone multiplicity pointers (16) and interaction trace pointers (32) to device
    m31 **device_mults = clone_to_device<m31*>(multiplicities, 16);
    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 32);

    int block_dim = trace_size < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? trace_size : VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < VERIFY_BITWISE_XOR_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // Loop over 8 column pairs (i0=0,2,4,...,14; i1=1,3,5,...,15)
    for (int pair = 0; pair < 8; pair++) {
        unsigned i0 = pair * 2;
        unsigned i1 = pair * 2 + 1;

        unsigned ah0 = i0 >> VERIFY_BITWISE_12_EXPAND_BITS;
        unsigned bh0 = i0 & ((1 << VERIFY_BITWISE_12_EXPAND_BITS) - 1);
        unsigned ah1 = i1 >> VERIFY_BITWISE_12_EXPAND_BITS;
        unsigned bh1 = i1 & ((1 << VERIFY_BITWISE_12_EXPAND_BITS) - 1);

        // Step 1: Generate logup fractions for this pair
        verify_bitwise_xor_12_interaction_trace_col_gen_kernel<<<num_blocks, block_dim>>>(
            device_lookup_elements,
            multiplicities[i0],
            multiplicities[i1],
            ah0, bh0, ah1, bh1,
            trace_size,
            device_logup_denom,
            device_numerator0,
            device_numerator1,
            device_numerator2,
            device_numerator3
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        // Step 2: Batch inverse
        batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

        // Step 3: Accumulate (write running sum to pair's 4-column block)
        verify_bitwise_xor_12_interaction_trace_accumulate_kernel<<<num_blocks, block_dim>>>(
            trace_size,
            denom_inv,
            device_numerator0,
            device_numerator1,
            device_numerator2,
            device_numerator3,
            device_interaction_traces,
            pair
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }

    // Step 4: Compute cumsum shift on last 4 columns (pair 7's accumulated total)
    m31 **device_last_4_cols = device_interaction_traces + 28;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_bitwise_xor_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_last_4_cols,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 5: Apply shift to last 4 columns only
    verify_bitwise_xor_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_last_4_cols
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 6: Prefix sum on last 4 columns only
    inclusive_prefix_sum(interaction_traces[28], trace_size);
    inclusive_prefix_sum(interaction_traces[29], trace_size);
    inclusive_prefix_sum(interaction_traces[30], trace_size);
    inclusive_prefix_sum(interaction_traces[31], trace_size);

    // Cleanup
    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(device_mults);
    cuda_free_memory(device_interaction_traces);

    global_timer.end("generate verify_bitwise_xor_12 interaction trace");
}
