#include "relations.cuh"

#include <cstdint>
#include <cstdio>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include <stdint.h>

#include "gen_range_check_builtin_bits_128_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"
#include "gen_memory_id_to_big_trace.cuh"

// Base trace generation kernel for range_check_builtin_bits_128
// 17 trace columns, 1 memory_address_to_id lookup, 1 memory_id_to_big lookup
__launch_bounds__(RANGE_CHECK_128_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_range_check_builtin_bits_128_trace_kernel(
    m31 **traces,

    // Lookup data arrays - 1 MemoryAddressToId lookup (2 elements)
    m31 **lookup_memory_address_to_id_0,

    // Lookup data arrays - 1 MemoryIdToBig lookup (29 elements)
    m31 **lookup_memory_id_to_big_0,

    // Sub-component inputs
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    // Constants
    const m31 M31_0 = {0};

    if (row < trace_size) {
        // seq = row index
        m31 seq = {row};
        m31 segment_start_m31 = {segment_start};

        // Read Positive Num Bits 128
        // address = segment_start + seq
        m31 addr = add(segment_start_m31, seq);

        // Read Id: memory_address_to_id
        m31 value_id_col0 = {0};
        memory_address_to_id_deduce_output(
            memory_address_to_id_address_to_raw_id,
            addr,
            &value_id_col0
        );
        traces[0][row] = value_id_col0;

        // Fill sub-component inputs and lookup data for memory_address_to_id
        sub_component_inputs_memory_address_to_id[0][row] = addr;
        lookup_memory_address_to_id_0[0][row] = addr;
        lookup_memory_address_to_id_0[1][row] = value_id_col0;

        // Read Positive Known Id Num Bits 128: memory_id_to_big
        // 128 bits = 14 × 9-bit limbs + 1 × 2-bit limb = 15 limbs total
        m31 value[N_M31_IN_FELT252] = {0};
        memory_id_to_big_state_deduce_output(
            memory_id_to_big_transposed_big_values,
            memory_id_to_big_small_values,
            value_id_col0,
            value
        );

        // Write the 15 limbs to trace columns 1-15
        m31 value_limb[15];
        for (int i = 0; i < 15; i++) {
            value_limb[i] = value[i];
            traces[1 + i][row] = value_limb[i];
        }

        // Range Check Last Limb Bits In Ms Limb 2
        // Cond Range Check 2
        // partial_limb_msb = (value_limb_14 & 2) >> 1
        m31 value_limb_14_col15 = value_limb[14];
        m31 partial_limb_msb_col16 = (value_limb_14_col15 & 2u) >> 1;
        traces[16][row] = partial_limb_msb_col16;

        // Fill sub-component inputs and lookup data for memory_id_to_big
        sub_component_inputs_memory_id_to_big[0][row] = value_id_col0;
        lookup_memory_id_to_big_0[0][row] = value_id_col0;
        for (int i = 0; i < 15; i++) {
            lookup_memory_id_to_big_0[1 + i][row] = value_limb[i];
        }
        // Remaining 13 elements are zeros (128 bits uses only 15 limbs out of 28)
        for (int i = 15; i < 28; i++) {
            lookup_memory_id_to_big_0[1 + i][row] = M31_0;
        }
    }
}

// Host wrapper function for base trace
extern "C"
void generate_range_check_builtin_bits_128_traces(
    unsigned **traces,

    // Lookup data arrays - 1 MemoryAddressToId lookup (2 elements)
    unsigned **lookup_memory_address_to_id_0,

    // Lookup data arrays - 1 MemoryIdToBig lookup (29 elements)
    unsigned **lookup_memory_id_to_big_0,

    // Sub-component inputs
    unsigned **sub_component_inputs_memory_address_to_id,
    unsigned **sub_component_inputs_memory_id_to_big,

    // Builtin segment info
    unsigned segment_start,

    // Memory data
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,

    unsigned n_rows,
    unsigned log_size
) {
    timer global_timer;
    global_timer.start("generate range_check_builtin_bits_128 base trace");

    unsigned trace_size = 1 << log_size;

    dim3 blockDim(RANGE_CHECK_128_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX);
    dim3 gridDim((trace_size + blockDim.x - 1) / blockDim.x);

    // Copy pointer arrays to device memory
    m31 **device_traces = clone_to_device<m31*>((m31**)traces, RANGE_CHECK_128_BUILTIN_N_TRACE_COLUMNS);

    // Memory address to id lookup (2 elements)
    m31 **device_lookup_addr2id_0 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_0, 2);

    // Memory id to big lookup (29 elements)
    m31 **device_lookup_id2big_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);

    // Sub-component inputs
    m31 **device_sub_addr2id = clone_to_device<m31*>((m31**)sub_component_inputs_memory_address_to_id, 1);
    m31 **device_sub_id2big = clone_to_device<m31*>((m31**)sub_component_inputs_memory_id_to_big, 1);

    // Clone memory_id_to_big transposed_big_values array to device (8 pointers)
    m31 **device_id2big_transposed = clone_to_device<m31*>((m31**)memory_id_to_big_transposed_big_values, 8);

    generate_range_check_builtin_bits_128_trace_kernel<<<gridDim, blockDim>>>(
        device_traces,

        device_lookup_addr2id_0,
        device_lookup_id2big_0,

        device_sub_addr2id,
        device_sub_id2big,

        segment_start,

        (m31*)memory_address_to_id_address_to_raw_id,
        device_id2big_transposed,
        (m31*)memory_id_to_big_small_values,

        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Free device memory
    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_sub_addr2id);
    cuda_free_memory(device_sub_id2big);
    cuda_free_memory(device_id2big_transposed);

    global_timer.end("generate range_check_builtin_bits_128 base trace");
}

// Column generation kernel for the single pair (memory_address_to_id + memory_id_to_big)
__launch_bounds__(RANGE_CHECK_128_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX, 2)
__global__ void generate_range_check_128_interaction_col0_kernel(
    MemoryAddressToId *memory_address_to_id,
    MemoryIdToBig *memory_id_to_big,
    m31 **lookup_addr2id,
    m31 **lookup_id2big,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_index < trace_size) {
        m31 addr2id_values[2] = {
            lookup_addr2id[0][vec_index],
            lookup_addr2id[1][vec_index]
        };

        m31 id2big_values[29];
        for (int i = 0; i < 29; i++) {
            id2big_values[i] = lookup_id2big[i][vec_index];
        }

        qm31 denom0 = memory_address_to_id->combine(addr2id_values, 2);
        qm31 denom1 = memory_id_to_big->combine(id2big_values, 29);
        logup_col_write_frac(vec_index, add(denom0, denom1), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize column kernel - accumulates interaction trace values
__global__ void generate_range_check_128_interaction_finalize_col_kernel(
    unsigned col_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = (col_index == 0) ? -1 : static_cast<int>(col_index - 1);

    if (vec_index < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[vec_index], numerator1[vec_index]},
                cm31{numerator2[vec_index], numerator3[vec_index]}
            },
            denom_inv_ptr[vec_index]
        );

        if (pre_index == -1) {
            interaction_traces[0][vec_index] = {0};
            interaction_traces[1][vec_index] = {0};
            interaction_traces[2][vec_index] = {0};
            interaction_traces[3][vec_index] = {0};
            qm31 tmp = value;
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        } else {
            qm31 prev_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][vec_index],
                     interaction_traces[pre_index * 4 + 1][vec_index]},
                cm31{interaction_traces[pre_index * 4 + 2][vec_index],
                     interaction_traces[pre_index * 4 + 3][vec_index]}
            };
            qm31 tmp = add(value, prev_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        }

        interaction_traces[col_index * 4 + 0][vec_index] = numerator0[vec_index];
        interaction_traces[col_index * 4 + 1][vec_index] = numerator1[vec_index];
        interaction_traces[col_index * 4 + 2][vec_index] = numerator2[vec_index];
        interaction_traces[col_index * 4 + 3][vec_index] = numerator3[vec_index];
    }
}

// Cumsum shift kernel - computes the sum for shifting
__global__ void generate_range_check_128_interaction_cumsum_shift_kernel(
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_traces,
    m31 *coordinate_sums
) {
    int idx0 = 4 * last_index - 4;
    int idx1 = 4 * last_index - 3;
    int idx2 = 4 * last_index - 2;
    int idx3 = 4 * last_index - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = 0;
    m31 sum1 = 0;
    m31 sum2 = 0;
    m31 sum3 = 0;

    for (int i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_traces[idx0][i]);
        sum1 = add(sum1, interaction_traces[idx1][i]);
        sum2 = add(sum2, interaction_traces[idx2][i]);
        sum3 = add(sum3, interaction_traces[idx3][i]);
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

// Coord prefix sum kernel - applies the shift
__global__ void generate_range_check_128_interaction_coord_prefix_sum_kernel(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interaction_traces[4 * last_index - 4][vec_index] = sub(interaction_traces[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interaction_traces[4 * last_index - 3][vec_index] = sub(interaction_traces[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interaction_traces[4 * last_index - 2][vec_index] = sub(interaction_traces[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interaction_traces[4 * last_index - 1][vec_index] = sub(interaction_traces[4 * last_index - 1][vec_index], cumsum_shift.b.b);
    }
}

// Host wrapper function for interaction trace
extern "C"
void generate_range_check_builtin_bits_128_interaction_traces(
    void *memory_address_to_id,
    void *memory_id_to_big,

    // Lookup data arrays - 1 MemoryAddressToId lookup
    unsigned **lookup_memory_address_to_id_0,

    // Lookup data arrays - 1 MemoryIdToBig lookup
    unsigned **lookup_memory_id_to_big_0,

    unsigned n_rows,
    unsigned log_size,
    unsigned **interaction_trace,
    unsigned *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate range_check_builtin_bits_128 interaction trace");

    unsigned trace_size = 1 << log_size;
    unsigned block_dim_val = RANGE_CHECK_128_BUILTIN_TRACE_GEN_THREAD_COUNT_MAX;
    unsigned num_blocks = (trace_size + block_dim_val - 1) / block_dim_val;

    // Allocate intermediate buffers for logup accumulation
    m31 *device_numerator0, *device_numerator1, *device_numerator2, *device_numerator3;
    qm31 *denom_ptr, *denom_inv;
    device_numerator0 = cuda_malloc<m31>(trace_size);
    device_numerator1 = cuda_malloc<m31>(trace_size);
    device_numerator2 = cuda_malloc<m31>(trace_size);
    device_numerator3 = cuda_malloc<m31>(trace_size);
    denom_ptr = cuda_malloc<qm31>(trace_size);
    denom_inv = cuda_malloc<qm31>(trace_size);

    // Allocate coordinate sums buffer
    m31 *device_coordinate_sums = cuda_malloc<m31>(4);
    ASSERT_CUDA_SUCCESS(cudaMemsetAsync(device_coordinate_sums, 0, 4 * sizeof(m31), 0));

    // Copy lookup element structs to device memory
    MemoryAddressToId *device_mem_addr_to_id = cuda_malloc<MemoryAddressToId>(1);
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(device_mem_addr_to_id, memory_address_to_id, sizeof(MemoryAddressToId), cudaMemcpyHostToDevice, 0));

    MemoryIdToBig *device_mem_id_to_big = cuda_malloc<MemoryIdToBig>(1);
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(device_mem_id_to_big, memory_id_to_big, sizeof(MemoryIdToBig), cudaMemcpyHostToDevice, 0));

    // Copy pointer arrays to device memory
    m31 **device_lookup_addr2id_0 = clone_to_device<m31*>((m31**)lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_id2big_0 = clone_to_device<m31*>((m31**)lookup_memory_id_to_big_0, 29);

    // Interaction trace (1 pair × 4 m31 columns = 4 m31 columns total)
    m31 **device_interaction_trace = clone_to_device<m31*>((m31**)interaction_trace, RANGE_CHECK_128_BUILTIN_N_INTERACTION_TRACE_COLUMNS * 4);

    // Lambda for finalizing each column
    auto launch_finalize = [&](unsigned col_index) {
        batch_inverse_secure_field(denom_ptr, denom_inv, trace_size);
        generate_range_check_128_interaction_finalize_col_kernel<<<num_blocks, block_dim_val>>>(
            col_index, trace_size, denom_inv,
            device_numerator0, device_numerator1, device_numerator2, device_numerator3,
            device_interaction_trace
        );
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    };

    // Column 0: pair(memory_address_to_id_0, memory_id_to_big_0)
    generate_range_check_128_interaction_col0_kernel<<<num_blocks, block_dim_val>>>(
        device_mem_addr_to_id, device_mem_id_to_big,
        device_lookup_addr2id_0, device_lookup_id2big_0,
        trace_size, denom_ptr,
        device_numerator0, device_numerator1, device_numerator2, device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    launch_finalize(0);

    // Compute cumsum_shift and apply coord_prefix_sum
    unsigned last_index = RANGE_CHECK_128_BUILTIN_N_INTERACTION_TRACE_COLUMNS;
    size_t shared_mem_size = 4 * block_dim_val * sizeof(m31);
    generate_range_check_128_interaction_cumsum_shift_kernel<<<num_blocks, block_dim_val, shared_mem_size>>>(
        last_index, trace_size, device_interaction_trace, device_coordinate_sums
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generate_range_check_128_interaction_coord_prefix_sum_kernel<<<num_blocks, block_dim_val>>>(
        device_coordinate_sums, last_index, trace_size, device_interaction_trace
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Apply inclusive_prefix_sum only to the last 4 columns
    inclusive_prefix_sum(interaction_trace[4 * RANGE_CHECK_128_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * RANGE_CHECK_128_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * RANGE_CHECK_128_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace[4 * RANGE_CHECK_128_BUILTIN_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    // Copy coordinate sums to claimed_sum output
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(claimed_sum, device_coordinate_sums, 4 * sizeof(m31), cudaMemcpyDeviceToHost, 0));

    // Free intermediate buffers
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_ptr);
    cuda_free_memory(denom_inv);
    cuda_free_memory(device_coordinate_sums);

    // Free device memory - lookup element structs
    cuda_free_memory(device_mem_addr_to_id);
    cuda_free_memory(device_mem_id_to_big);

    // Free device memory - pointer arrays
    cuda_free_memory(device_lookup_addr2id_0);
    cuda_free_memory(device_lookup_id2big_0);
    cuda_free_memory(device_interaction_trace);

    global_timer.end("generate range_check_builtin_bits_128 interaction trace");
}
