// CUDA interaction trace generation for pedersen_points_table component.
//
// Follows the verify_bitwise_xor pattern:
//   1. Generate logup fractions (numerator/denominator per row)
//   2. Batch inverse denominators
//   3. Finalize column (value = numerator * denom_inv)
//   4. Compute cumsum shift
//   5. Apply cumsum shift
//   6. Prefix sum
//
// The pedersen_points_table lookup combines 57 values per row:
//   values[0] = row_index (seq)
//   values[1..56] = g_pedersen_table_columns[0..55][row]

#include <cstdio>
#include "fields.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "timer.cuh"
#include "pedersen_table.cuh"

#include "gen_pedersen_points_table_trace.cuh"

#define PPT_INTERACTION_THREAD_COUNT_MAX 256

// Kernel: generate logup fractions for pedersen_points_table
// For each row: numerator = -mults[row], denominator = lookup_elements.combine(58 values)
//
// All traces are stored in natural order (entry k at position k).
// The BitReversedOrder type marker is handled by the FFT during interpolation.
__global__ void pedersen_points_table_interaction_trace_col_gen_kernel(
    LookupElementsBasic<PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES> *lookup_elements,
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
        // Build the 58 lookup values: [relation_id, seq, p0, p1, ..., p55]
        m31 values[PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES];
        values[0] = PEDERSEN_POINTS_TABLE_RELATION_ID;  // relation ID constant
        values[1] = row;  // seq = table index (natural order)
        for (int i = 0; i < PEDERSEN_TABLE_N_COLUMNS; i++) {
            values[2 + i] = g_pedersen_table_columns[i][row];
        }

        // numerator = -multiplicities[row]
        m31 mult = multiplicities[row];
        qm31 numerator = qm31{cm31{P - mult, 0}, cm31{0, 0}};

        // denominator = lookup_elements.combine(values)
        qm31 denom = lookup_elements->combine(values, PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES);

        logup_col_write_frac(row, numerator, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize column kernel - compute value = numerator * denom_inv
__global__ void pedersen_points_table_interaction_trace_finalize_col_kernel(
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

        interaction_traces[0][row] = value.a.a;
        interaction_traces[1][row] = value.a.b;
        interaction_traces[2][row] = value.b.a;
        interaction_traces[3][row] = value.b.b;
    }
}

// Compute cumsum shift - sum all values across interaction trace columns
__global__ void pedersen_points_table_interaction_trace_cumsum_shift(
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
__global__ void pedersen_points_table_interaction_trace_apply_shift(
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

// Host function for pedersen_points_table interaction trace generation
void pedersen_points_table_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    timer global_timer;
    global_timer.start("generate pedersen_points_table interaction trace");

    unsigned trace_size = 1 << log_size;

    // Copy lookup elements to device
    LookupElementsBasic<PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES> *device_lookup_elements =
        cuda_malloc<LookupElementsBasic<PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES>>(
        (LookupElementsBasic<PEDERSEN_POINTS_TABLE_N_LOOKUP_VALUES>*)lookup_elements,
        device_lookup_elements, 1);

    // Allocate temporary buffers
    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31 *denom_inv = cuda_malloc<qm31>(trace_size);
    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4);

    int block_dim = trace_size < PPT_INTERACTION_THREAD_COUNT_MAX ? trace_size : PPT_INTERACTION_THREAD_COUNT_MAX;
    int num_blocks = block_dim < PPT_INTERACTION_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // Step 1: Generate logup fractions
    pedersen_points_table_interaction_trace_col_gen_kernel<<<num_blocks, block_dim>>>(
        device_lookup_elements,
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
    pedersen_points_table_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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
    pedersen_points_table_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 5: Apply shift
    pedersen_points_table_interaction_trace_apply_shift<<<num_blocks, block_dim>>>(
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

    global_timer.end("generate pedersen_points_table interaction trace");
}
