
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
#include "gen_range_check_vector.cuh"

#define RANGE_CHECK_INTERACTION_TRACE_COLUMNS 1
#define RANGE_CHECK_TRACE_GEN_THREAD_COUNT_MAX 256

HOST_DEVICE_FORCEINLINE void partition_into_bit_segments(
    // unsigned input_value,
    unsigned *n_bits_per_segments,
    unsigned N_RANGE,
    unsigned *output_value,
    unsigned total_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned input_tmp = row;
    // unsigned segments[N_RANGE] = {0};
    for (int seg = N_RANGE -1; seg >= 0; --seg) {
        uint32_t nbits = n_bits_per_segments[seg];
        uint32_t mask = (1U << nbits) - 1;
        output_value[seg] = input_tmp & mask;
        input_tmp >>= nbits;
    }
}

__global__ void partition_into_bit_segments_kernel(
    // unsigned  *input_value,
    unsigned *n_bits_per_segments,
    unsigned N_RANGE,
    unsigned **output_value,
    unsigned total_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < total_size) {
    unsigned input_tmp = row;
        for (int seg = N_RANGE -1; seg >= 0; --seg) {
            uint32_t nbits = n_bits_per_segments[seg];
            uint32_t mask = (1U << nbits) - 1;
            output_value[seg][row] = input_tmp & mask;
            input_tmp >>= nbits;
        }
    }
}

void partition_into_bit_segments_cuda(
    // unsigned *inputs,
    unsigned total_size,
    unsigned n_range,
    unsigned *n_bits_per_segments,
    unsigned **output_value
) {
    unsigned **device_outputs = clone_to_device<unsigned*>(output_value, n_range);
    unsigned *device_n_bits_per_segments = clone_to_device<unsigned>(n_bits_per_segments, n_range);

    int block_dim = total_size < THREAD_COUNT_MAX ? total_size : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (total_size + block_dim - 1) / block_dim;

    partition_into_bit_segments_kernel<<<num_blocks, block_dim>>>(
        // inputs,
        device_n_bits_per_segments,
        n_range,
        device_outputs,
        total_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_outputs);
    cuda_free_memory(device_n_bits_per_segments);
}


__global__ void range_check_vector_add_inputs_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    unsigned n_range,
    unsigned *ranges,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t value = 0;
        for (int i = 0; i < n_range; ++i) {
            value <<= ranges[i];
            value += inputs[i][row];
        }
        atomicAdd(&mults[value], 1);
    }
}

void range_check_vector_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    unsigned n_range,
    unsigned *ranges,
    m31 *mults,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, n_range * input_col_sizes);
    unsigned *device_ranges = clone_to_device<unsigned>(ranges, n_range);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    range_check_vector_add_inputs_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        n_range,
        device_ranges,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
    cuda_free_memory(device_ranges);
}

template <int N>
__launch_bounds__(256, 2)
__global__ void generate_range_check_interaction_trace_col_single_gen_kernel(
    LookupElementsBasic<N>  *lookup_elements_n,
    m31 **lookup_state_0,
    m31 *multiplicities,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 init_combine_reg[N] = {};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }

    if (vec_index < trace_size) {
        qm31 denom = lookup_elements_n->combine(init_combine_reg, N);
        // Numerator is -multiplicity[vec_index]
        qm31 numer = qm31{cm31{neg(multiplicities[vec_index]), 0}, cm31{0, 0}};
        logup_col_write_frac(vec_index, numer, denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

__global__ void generate_range_check_interaction_trace_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31 *denom_inv_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3,
    m31 **interaction_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = rep_index - 1;

    if (vec_index < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[vec_index], numerator1[vec_index]},
                cm31{numerator2[vec_index], numerator3[vec_index]}
            },
            denom_inv_ptr[vec_index]
        );

        if (pre_index == -1) {
            interaction_traces[0][vec_index] = 0;
            interaction_traces[1][vec_index] = 0;
            interaction_traces[2][vec_index] = 0;
            interaction_traces[3][vec_index] = 0;
            qm31 pre_value = qm31 {0};
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        } else {
            qm31 pre_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][vec_index], interaction_traces[pre_index * 4 + 1][vec_index]},
                cm31{interaction_traces[pre_index * 4 + 2][vec_index], interaction_traces[pre_index * 4 + 3][vec_index]}
            };
            qm31 tmp = add(value, pre_value);
            numerator0[vec_index] = tmp.a.a;
            numerator1[vec_index] = tmp.a.b;
            numerator2[vec_index] = tmp.b.a;
            numerator3[vec_index] = tmp.b.b;
        }

        interaction_traces[rep_index * 4 + 0][vec_index] = numerator0[vec_index];
        interaction_traces[rep_index * 4 + 1][vec_index] = numerator1[vec_index];
        interaction_traces[rep_index * 4 + 2][vec_index] = numerator2[vec_index];
        interaction_traces[rep_index * 4 + 3][vec_index] = numerator3[vec_index];
    }
}

__global__ void generate_range_check_interaction_trace_cumsum_shift(
    unsigned last_index,
    unsigned trace_size,
    m31 **interactive_traces,
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
        sum0 = add(sum0, interactive_traces[idx0][i]);
        sum1 = add(sum1, interactive_traces[idx1][i]);
        sum2 = add(sum2, interactive_traces[idx2][i]);
        sum3 = add(sum3, interactive_traces[idx3][i]);
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

__global__ void generate_range_check_interaction_trace_coord_prefix_sum(
    m31 *coordinate_sums,
    unsigned last_index,
    unsigned trace_size,
    m31 **interactive_traces
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_index < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        interactive_traces[4 * last_index - 4][vec_index] = sub(interactive_traces[4 * last_index - 4][vec_index], cumsum_shift.a.a);
        interactive_traces[4 * last_index - 3][vec_index] = sub(interactive_traces[4 * last_index - 3][vec_index], cumsum_shift.a.b);
        interactive_traces[4 * last_index - 2][vec_index] = sub(interactive_traces[4 * last_index - 2][vec_index], cumsum_shift.b.a);
        interactive_traces[4 * last_index - 1][vec_index] = sub(interactive_traces[4 * last_index - 1][vec_index], cumsum_shift.b.b);

    }
}


void range_check_vector_generate_interaction_trace(
    void *lookup_element_ptr,
    m31 *multiplicities,

    unsigned n_range,
    unsigned *ranges,

    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;
    unsigned *device_ranges = clone_to_device<unsigned>(ranges, n_range);

    // Allocate temporary columns for partitioned bit segments
    unsigned **cols_tmp_vec = new unsigned*[n_range];
    for (unsigned i = 0; i < n_range; i++) {
        cols_tmp_vec[i] = cuda_malloc<unsigned>(trace_size);
    }
    unsigned **device_cols_tmp_vec = clone_to_device<uint32_t *>(cols_tmp_vec, n_range);

    // multiplicities is already a device pointer
    m31 *device_multiplicities = multiplicities;

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4 * RANGE_CHECK_INTERACTION_TRACE_COLUMNS);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    timer global_timer;
    global_timer.start("generate range_check interaction trace");

    int block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    partition_into_bit_segments_kernel<<<num_blocks, block_dim>>>(
        device_ranges,
        n_range,
        device_cols_tmp_vec,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < RANGE_CHECK_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : RANGE_CHECK_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < RANGE_CHECK_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // Dispatch based on n_range - all range check types with same n_range use same kernel
    if (n_range == 1) {
        HANDLE_RANGE_CHECK_GENERIC(1)
    } else if (n_range == 2) {
        HANDLE_RANGE_CHECK_GENERIC(2)
    } else if (n_range == 3) {
        HANDLE_RANGE_CHECK_GENERIC(3)
    } else if (n_range == 4) {
        HANDLE_RANGE_CHECK_GENERIC(4)
    } else if (n_range == 5) {
        HANDLE_RANGE_CHECK_GENERIC(5)
    }

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_range_check_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        0,  // rep_index should be 0 for first (and only) interaction column
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_traces, 4, trace_size);

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_range_check_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        RANGE_CHECK_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_range_check_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        RANGE_CHECK_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[4 * RANGE_CHECK_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * RANGE_CHECK_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * RANGE_CHECK_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * RANGE_CHECK_INTERACTION_TRACE_COLUMNS - 1], trace_size);


    global_timer.end("generate range_check interaction trace");

    cuda_free_memory(device_logup_denom);

    // Free temporary partition columns
    for (unsigned i = 0; i < n_range; i++) {
        cuda_free_memory(cols_tmp_vec[i]);
    }
    delete[] cols_tmp_vec;
    cuda_free_memory(device_cols_tmp_vec);
    cuda_free_memory(device_ranges);

    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}

// ============================================================================
// Multi-relation range check interaction trace
// ============================================================================
//
// Handles N_RELATIONS relations (processed as N_PAIRS = N_RELATIONS/2 pairs)
// entirely on GPU. For each pair (a, b):
//   - Compute values from row index via bit-segment partitioning
//   - denom_a = lookup_a.combine(values), denom_b = lookup_b.combine(values)
//   - numerator = -mult_a * denom_b + -mult_b * denom_a
//   - denominator = denom_a * denom_b
//   - batch_inverse, finalize with accumulation across pairs
// Then: cumsum_shift + prefix_sum on last 4 columns.
//
// This eliminates the 2*N_PAIRS GPU↔CPU round-trips that the old approach
// required (download per-pair kernel output, CPU fraction recovery, re-upload).
// ============================================================================

#define RC_MULTI_REL_BLOCK_SIZE 256

// Paired logup column generation kernel for range checks.
// Computes values from row index (bit-segment partitioning), then does
// paired logup: frac = (-mult_a/denom_a) + (-mult_b/denom_b)
//             = (-mult_a * denom_b + -mult_b * denom_a) / (denom_a * denom_b)
template <int N>
__launch_bounds__(RC_MULTI_REL_BLOCK_SIZE, 2)
__global__ void rc_multi_relation_col_gen_kernel(
    LookupElementsBasic<N>* lookup_a,
    LookupElementsBasic<N>* lookup_b,
    m31* mults_a,
    m31* mults_b,
    unsigned* ranges,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        // Compute values from row index via bit-segment partitioning
        m31 values[N];
        unsigned row_tmp = idx;
        for (int seg = N - 1; seg >= 0; --seg) {
            unsigned nbits = ranges[seg];
            unsigned mask = (1u << nbits) - 1;
            values[seg] = row_tmp & mask;
            row_tmp >>= nbits;
        }

        qm31 d_a = lookup_a->combine(values, N);
        qm31 d_b = lookup_b->combine(values, N);

        // Paired logup numerator: -mult_a * d_b + -mult_b * d_a
        qm31 m_a = qm31{cm31{neg(mults_a[idx]), 0}, cm31{0, 0}};
        qm31 m_b = qm31{cm31{neg(mults_b[idx]), 0}, cm31{0, 0}};
        qm31 numer = add(mul(m_a, d_b), mul(m_b, d_a));
        qm31 denom = mul(d_a, d_b);

        logup_col_write_frac(idx, numer, denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Dispatch macro for multi-relation col gen kernel (template instantiation)
#define HANDLE_RC_MULTI_REL_COL_GEN(TEMPLATE_N) \
    { \
        LookupElementsBasic<TEMPLATE_N>* d_lookup_a = cuda_malloc<LookupElementsBasic<TEMPLATE_N>>(1); \
        LookupElementsBasic<TEMPLATE_N>* d_lookup_b = cuda_malloc<LookupElementsBasic<TEMPLATE_N>>(1); \
        cuda_mem_copy_host_to_device<LookupElementsBasic<TEMPLATE_N>>( \
            (LookupElementsBasic<TEMPLATE_N>*)lookup_elements[2 * pair], d_lookup_a, 1); \
        cuda_mem_copy_host_to_device<LookupElementsBasic<TEMPLATE_N>>( \
            (LookupElementsBasic<TEMPLATE_N>*)lookup_elements[2 * pair + 1], d_lookup_b, 1); \
        rc_multi_relation_col_gen_kernel<TEMPLATE_N><<<num_blocks, block_dim>>>( \
            d_lookup_a, d_lookup_b, \
            multiplicities[2 * pair], multiplicities[2 * pair + 1], \
            device_ranges, trace_size, \
            device_logup_denom, numerator0, numerator1, numerator2, numerator3); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        cuda_free_memory(d_lookup_a); \
        cuda_free_memory(d_lookup_b); \
    }

void range_check_multi_relation_interaction_trace(
    unsigned n_pairs,
    unsigned n_range,
    unsigned* ranges,
    void** lookup_elements,      // 2*n_pairs host pointers to LookupElements<N_RANGES>
    m31** multiplicities,        // 2*n_pairs device pointers
    unsigned log_size,
    m31** interaction_trace_columns,  // 4*n_pairs output column device pointers
    m31* claimed_sum                  // 4 device m31s for qm31
) {
    unsigned trace_size = 1u << log_size;
    unsigned* device_ranges = clone_to_device<unsigned>(ranges, n_range);

    // Allocate working memory
    qm31* device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31* denom_inv = cuda_malloc<qm31>(trace_size);
    m31* numerator0 = cuda_malloc<m31>(trace_size);
    m31* numerator1 = cuda_malloc<m31>(trace_size);
    m31* numerator2 = cuda_malloc<m31>(trace_size);
    m31* numerator3 = cuda_malloc<m31>(trace_size);

    m31** device_it = clone_to_device<m31*>(interaction_trace_columns, 4 * n_pairs);

    int block_dim = trace_size < RC_MULTI_REL_BLOCK_SIZE ? trace_size : RC_MULTI_REL_BLOCK_SIZE;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Process each pair
    for (unsigned pair = 0; pair < n_pairs; pair++) {
        // 1. Col gen kernel — dispatch based on n_range
        if (n_range == 1) {
            HANDLE_RC_MULTI_REL_COL_GEN(1)
        } else if (n_range == 2) {
            HANDLE_RC_MULTI_REL_COL_GEN(2)
        } else if (n_range == 3) {
            HANDLE_RC_MULTI_REL_COL_GEN(3)
        } else if (n_range == 4) {
            HANDLE_RC_MULTI_REL_COL_GEN(4)
        } else if (n_range == 5) {
            HANDLE_RC_MULTI_REL_COL_GEN(5)
        }

        // 2. Batch inverse
        batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

        // 3. Finalize col — accumulates with previous columns
        generate_range_check_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
            pair, trace_size, denom_inv,
            numerator0, numerator1, numerator2, numerator3,
            device_it);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }

    // Finalize: cumsum_shift + prefix sum on last 4 columns
    cudaMemsetAsync(claimed_sum, 0, 4 * sizeof(m31), 0);

    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_range_check_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        n_pairs, trace_size, device_it, claimed_sum);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    generate_range_check_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum, n_pairs, trace_size, device_it);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Inclusive prefix sum on last 4 columns
    inclusive_prefix_sum(interaction_trace_columns[4 * n_pairs - 4], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * n_pairs - 3], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * n_pairs - 2], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * n_pairs - 1], trace_size);

    // Cleanup
    cuda_free_memory(device_ranges);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(numerator0);
    cuda_free_memory(numerator1);
    cuda_free_memory(numerator2);
    cuda_free_memory(numerator3);
    cuda_free_memory(device_it);
}