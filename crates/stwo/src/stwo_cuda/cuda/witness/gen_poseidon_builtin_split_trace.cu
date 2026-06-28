/**
 * CUDA trace generation for poseidon_builtin (split, 6-col trace).
 *
 * This is the "split" poseidon_builtin kernel for v1.1.0's split AIR.
 * Only generates the 6 builtin-level columns (memory ID lookups).
 * Poseidon Hades permutation is delegated to poseidon_aggregator.
 *
 * Architecture:
 *   - 1 thread per row
 *   - Uses GPU-resident address_to_raw_id table for memory ID lookup
 *
 * Column layout (6 columns):
 *   0: input_state_0_id
 *   1: input_state_1_id
 *   2: input_state_2_id
 *   3: output_state_0_id
 *   4: output_state_1_id
 *   5: output_state_2_id
 *
 * Lookup data (7 lookups):
 *   memory_address_to_id_0..5: [rel_id, addr+i, id_i]  — 3 elements each
 *   poseidon_aggregator_0:     [rel_id, id0..id5]       — 7 elements
 *
 * Sub-component inputs:
 *   memory_address_to_id: 6 columns (addr+0 .. addr+5)
 *   poseidon_aggregator:  6 columns (id0..id5)
 *
 * Interaction trace (4 logup columns):
 *   Col 0: ADD pair (mem_0 + mem_1) — 3+3 elements
 *   Col 1: ADD pair (mem_2 + mem_3) — 3+3 elements
 *   Col 2: ADD pair (mem_4 + mem_5) — 3+3 elements
 *   Col 3: SINGLE poseidon_aggregator — numerator=1, 7 elements
 */

#include "gen_poseidon_builtin_split_trace.cuh"
#include "../fields.cuh"
#include "../utils.cuh"
#include "../logup.cuh"
#include "../batch_inverse.cuh"
#include "../prefix_sum.cuh"

#define POS_BLT_BLOCK_SIZE 256
#define POS_BLT_MEM_ADDR_TO_ID_RELATION_ID 1444891767u
#define POS_BLT_AGG_RELATION_ID            1551892206u
#define POS_BLT_N_LOGUP_COLUMNS            4

// ============================================================================
// Base trace kernel
// ============================================================================

__global__ void gen_poseidon_builtin_split_trace_kernel(
    m31* col0,    // output: input_state_0_id
    m31* col1,    // output: input_state_1_id
    m31* col2,    // output: input_state_2_id
    m31* col3,    // output: output_state_0_id
    m31* col4,    // output: output_state_1_id
    m31* col5,    // output: output_state_2_id
    // Lookup data outputs: 6 memory lookups (3 elements each)
    m31* lk_mem0_0, m31* lk_mem0_1, m31* lk_mem0_2,
    m31* lk_mem1_0, m31* lk_mem1_1, m31* lk_mem1_2,
    m31* lk_mem2_0, m31* lk_mem2_1, m31* lk_mem2_2,
    m31* lk_mem3_0, m31* lk_mem3_1, m31* lk_mem3_2,
    m31* lk_mem4_0, m31* lk_mem4_1, m31* lk_mem4_2,
    m31* lk_mem5_0, m31* lk_mem5_1, m31* lk_mem5_2,
    // Lookup data outputs: 1 aggregator lookup (7 elements)
    m31* lk_agg0_0, m31* lk_agg0_1, m31* lk_agg0_2, m31* lk_agg0_3,
    m31* lk_agg0_4, m31* lk_agg0_5, m31* lk_agg0_6,
    // Sub-component inputs: 6 memory addresses
    m31* sub_mem_0, m31* sub_mem_1, m31* sub_mem_2,
    m31* sub_mem_3, m31* sub_mem_4, m31* sub_mem_5,
    // Sub-component inputs: 6 IDs for aggregator
    m31* sub_agg_0, m31* sub_agg_1, m31* sub_agg_2,
    m31* sub_agg_3, m31* sub_agg_4, m31* sub_agg_5,
    // GPU memory table
    unsigned* address_to_raw_id,
    // Parameters
    unsigned segment_start,
    unsigned n_rows,
    unsigned trace_size
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trace_size) return;

    // Padding: rows beyond n_rows get zero
    if (idx >= n_rows) {
        col0[idx] = m31(0); col1[idx] = m31(0); col2[idx] = m31(0);
        col3[idx] = m31(0); col4[idx] = m31(0); col5[idx] = m31(0);
        // Memory lookups
        lk_mem0_0[idx] = m31(0); lk_mem0_1[idx] = m31(0); lk_mem0_2[idx] = m31(0);
        lk_mem1_0[idx] = m31(0); lk_mem1_1[idx] = m31(0); lk_mem1_2[idx] = m31(0);
        lk_mem2_0[idx] = m31(0); lk_mem2_1[idx] = m31(0); lk_mem2_2[idx] = m31(0);
        lk_mem3_0[idx] = m31(0); lk_mem3_1[idx] = m31(0); lk_mem3_2[idx] = m31(0);
        lk_mem4_0[idx] = m31(0); lk_mem4_1[idx] = m31(0); lk_mem4_2[idx] = m31(0);
        lk_mem5_0[idx] = m31(0); lk_mem5_1[idx] = m31(0); lk_mem5_2[idx] = m31(0);
        // Aggregator lookup
        lk_agg0_0[idx] = m31(0); lk_agg0_1[idx] = m31(0); lk_agg0_2[idx] = m31(0);
        lk_agg0_3[idx] = m31(0); lk_agg0_4[idx] = m31(0); lk_agg0_5[idx] = m31(0);
        lk_agg0_6[idx] = m31(0);
        // Sub-component inputs
        sub_mem_0[idx] = m31(0); sub_mem_1[idx] = m31(0); sub_mem_2[idx] = m31(0);
        sub_mem_3[idx] = m31(0); sub_mem_4[idx] = m31(0); sub_mem_5[idx] = m31(0);
        sub_agg_0[idx] = m31(0); sub_agg_1[idx] = m31(0); sub_agg_2[idx] = m31(0);
        sub_agg_3[idx] = m31(0); sub_agg_4[idx] = m31(0); sub_agg_5[idx] = m31(0);
        return;
    }

    // Poseidon uses 6 consecutive addresses per instance
    unsigned instance_addr = idx * 6 + segment_start;
    // address_to_raw_id is 1-indexed: for address N, access [N-1]
    m31 id0 = (m31) address_to_raw_id[instance_addr - 1];
    m31 id1 = (m31) address_to_raw_id[instance_addr];
    m31 id2 = (m31) address_to_raw_id[instance_addr + 1];
    m31 id3 = (m31) address_to_raw_id[instance_addr + 2];
    m31 id4 = (m31) address_to_raw_id[instance_addr + 3];
    m31 id5 = (m31) address_to_raw_id[instance_addr + 4];

    col0[idx] = id0;
    col1[idx] = id1;
    col2[idx] = id2;
    col3[idx] = id3;
    col4[idx] = id4;
    col5[idx] = id5;

    // Lookup data: memory_address_to_id (relation_id = 1444891767)
    m31 rel_id = m31(POS_BLT_MEM_ADDR_TO_ID_RELATION_ID);
    m31 addr = m31(instance_addr);
    lk_mem0_0[idx] = rel_id; lk_mem0_1[idx] = addr;              lk_mem0_2[idx] = id0;
    lk_mem1_0[idx] = rel_id; lk_mem1_1[idx] = add(addr, m31(1)); lk_mem1_2[idx] = id1;
    lk_mem2_0[idx] = rel_id; lk_mem2_1[idx] = add(addr, m31(2)); lk_mem2_2[idx] = id2;
    lk_mem3_0[idx] = rel_id; lk_mem3_1[idx] = add(addr, m31(3)); lk_mem3_2[idx] = id3;
    lk_mem4_0[idx] = rel_id; lk_mem4_1[idx] = add(addr, m31(4)); lk_mem4_2[idx] = id4;
    lk_mem5_0[idx] = rel_id; lk_mem5_1[idx] = add(addr, m31(5)); lk_mem5_2[idx] = id5;

    // Lookup data: poseidon_aggregator (relation_id = 1551892206)
    lk_agg0_0[idx] = m31(POS_BLT_AGG_RELATION_ID);
    lk_agg0_1[idx] = id0;
    lk_agg0_2[idx] = id1;
    lk_agg0_3[idx] = id2;
    lk_agg0_4[idx] = id3;
    lk_agg0_5[idx] = id4;
    lk_agg0_6[idx] = id5;

    // Sub-component inputs: memory_address_to_id
    sub_mem_0[idx] = addr;
    sub_mem_1[idx] = add(addr, m31(1));
    sub_mem_2[idx] = add(addr, m31(2));
    sub_mem_3[idx] = add(addr, m31(3));
    sub_mem_4[idx] = add(addr, m31(4));
    sub_mem_5[idx] = add(addr, m31(5));

    // Sub-component inputs: poseidon_aggregator (3 input IDs + 3 output IDs)
    sub_agg_0[idx] = id0;
    sub_agg_1[idx] = id1;
    sub_agg_2[idx] = id2;
    sub_agg_3[idx] = id3;
    sub_agg_4[idx] = id4;
    sub_agg_5[idx] = id5;
}

// ============================================================================
// Host wrapper for base trace
// ============================================================================

extern "C" void gen_poseidon_builtin_split_trace(
    m31** traces,            // 6 output columns
    m31** lk_mem_0,          // 3 arrays
    m31** lk_mem_1,          // 3 arrays
    m31** lk_mem_2,          // 3 arrays
    m31** lk_mem_3,          // 3 arrays
    m31** lk_mem_4,          // 3 arrays
    m31** lk_mem_5,          // 3 arrays
    m31** lk_agg_0,          // 7 arrays
    m31** sub_mem,           // 6 arrays
    m31** sub_agg,           // 6 arrays
    unsigned* address_to_raw_id,
    uint32_t segment_start,
    uint32_t n_rows,
    uint32_t log_size
) {
    uint32_t trace_size = 1u << log_size;
    int num_blocks = (trace_size + POS_BLT_BLOCK_SIZE - 1) / POS_BLT_BLOCK_SIZE;

    gen_poseidon_builtin_split_trace_kernel<<<num_blocks, POS_BLT_BLOCK_SIZE>>>(
        traces[0], traces[1], traces[2], traces[3], traces[4], traces[5],
        lk_mem_0[0], lk_mem_0[1], lk_mem_0[2],
        lk_mem_1[0], lk_mem_1[1], lk_mem_1[2],
        lk_mem_2[0], lk_mem_2[1], lk_mem_2[2],
        lk_mem_3[0], lk_mem_3[1], lk_mem_3[2],
        lk_mem_4[0], lk_mem_4[1], lk_mem_4[2],
        lk_mem_5[0], lk_mem_5[1], lk_mem_5[2],
        lk_agg_0[0], lk_agg_0[1], lk_agg_0[2], lk_agg_0[3],
        lk_agg_0[4], lk_agg_0[5], lk_agg_0[6],
        sub_mem[0], sub_mem[1], sub_mem[2], sub_mem[3], sub_mem[4], sub_mem[5],
        sub_agg[0], sub_agg[1], sub_agg[2], sub_agg[3], sub_agg[4], sub_agg[5],
        address_to_raw_id,
        segment_start,
        n_rows,
        trace_size
    );
}

// ============================================================================
// Interaction trace generation (4 logup columns)
// ============================================================================
//
// Column layout:
//   Col 0: ADD pair (mem_0 + mem_1) — 3 elements each
//   Col 1: ADD pair (mem_2 + mem_3) — 3 elements each
//   Col 2: ADD pair (mem_4 + mem_5) — 3 elements each
//   Col 3: SINGLE poseidon_aggregator — numerator=1, 7 elements
//
// Uses CommonLookupElements (= LookupElementsBasic<128>) directly since
// lookup data already includes relation constants at index 0.
// ============================================================================

#define POS_BLT_IT_BLOCK_SIZE 256

// ADD pair kernel: frac = (d0 + d1) / (d0 * d1)
template <int N, int M>
__launch_bounds__(POS_BLT_IT_BLOCK_SIZE, 2)
__global__ void pos_blt_it_add_pair_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31** data_0,
    m31** data_1,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        m31 vals0[N], vals1[M];
        for (int i = 0; i < N; i++) vals0[i] = data_0[i][idx];
        for (int i = 0; i < M; i++) vals1[i] = data_1[i][idx];
        qm31 d0 = lookup_elements->combine(vals0, N);
        qm31 d1 = lookup_elements->combine(vals1, M);
        logup_col_write_frac(idx, add(d0, d1), mul(d0, d1),
                            denom_ptr, numer0, numer1, numer2, numer3);
    }
}

// SINGLE kernel: frac = 1 / denom (numerator = 1)
template <int N>
__launch_bounds__(POS_BLT_IT_BLOCK_SIZE, 2)
__global__ void pos_blt_it_single_kernel(
    LookupElementsBasic<128>* lookup_elements,
    m31** data_0,
    unsigned trace_size,
    qm31* denom_ptr,
    m31* numer0, m31* numer1, m31* numer2, m31* numer3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        m31 vals[N];
        for (int i = 0; i < N; i++) vals[i] = data_0[i][idx];
        qm31 denom = lookup_elements->combine(vals, N);
        // numerator = 1 (as qm31: {cm31{m31(1), m31(0)}, cm31{m31(0), m31(0)}})
        qm31 one = qm31{cm31{m31(1), m31(0)}, cm31{m31(0), m31(0)}};
        logup_col_write_frac(idx, one, denom,
                            denom_ptr, numer0, numer1, numer2, numer3);
    }
}

// Finalize kernel: multiply numerator by inverse denominator and accumulate
__global__ void pos_blt_it_finalize_col_kernel(
    unsigned rep_index,
    unsigned trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_index = rep_index - 1;

    if (idx < trace_size) {
        qm31 value = mul(
            qm31 {
                cm31{numerator0[idx], numerator1[idx]},
                cm31{numerator2[idx], numerator3[idx]}
            },
            denom_inv_ptr[idx]
        );

        if (pre_index == -1) {
            qm31 tmp = value;
            numerator0[idx] = tmp.a.a;
            numerator1[idx] = tmp.a.b;
            numerator2[idx] = tmp.b.a;
            numerator3[idx] = tmp.b.b;
        } else {
            qm31 pre_value = qm31 {
                cm31{interaction_traces[pre_index * 4 + 0][idx], interaction_traces[pre_index * 4 + 1][idx]},
                cm31{interaction_traces[pre_index * 4 + 2][idx], interaction_traces[pre_index * 4 + 3][idx]}
            };
            qm31 tmp = add(value, pre_value);
            numerator0[idx] = tmp.a.a;
            numerator1[idx] = tmp.a.b;
            numerator2[idx] = tmp.b.a;
            numerator3[idx] = tmp.b.b;
        }

        interaction_traces[rep_index * 4 + 0][idx] = numerator0[idx];
        interaction_traces[rep_index * 4 + 1][idx] = numerator1[idx];
        interaction_traces[rep_index * 4 + 2][idx] = numerator2[idx];
        interaction_traces[rep_index * 4 + 3][idx] = numerator3[idx];
    }
}

// Cumsum shift kernel — computes claimed_sum from last column
__global__ void pos_blt_it_cumsum_shift(
    unsigned n_cols,
    unsigned trace_size,
    m31** interactive_traces,
    m31* coordinate_sums
) {
    int idx0 = 4 * n_cols - 4;
    int idx1 = 4 * n_cols - 3;
    int idx2 = 4 * n_cols - 2;
    int idx3 = 4 * n_cols - 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = tid; i < trace_size; i += gridSize) {
        s0 = add(s0, interactive_traces[idx0][i]);
        s1 = add(s1, interactive_traces[idx1][i]);
        s2 = add(s2, interactive_traces[idx2][i]);
        s3 = add(s3, interactive_traces[idx3][i]);
    }

    extern __shared__ m31 shared[];
    m31* sd0 = &shared[0];
    m31* sd1 = &shared[blockDim.x];
    m31* sd2 = &shared[2 * blockDim.x];
    m31* sd3 = &shared[3 * blockDim.x];

    sd0[threadIdx.x] = s0;
    sd1[threadIdx.x] = s1;
    sd2[threadIdx.x] = s2;
    sd3[threadIdx.x] = s3;
    __syncthreads();

    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sd0[threadIdx.x] = add(sd0[threadIdx.x], sd0[threadIdx.x + s]);
            sd1[threadIdx.x] = add(sd1[threadIdx.x], sd1[threadIdx.x + s]);
            sd2[threadIdx.x] = add(sd2[threadIdx.x], sd2[threadIdx.x + s]);
            sd3[threadIdx.x] = add(sd3[threadIdx.x], sd3[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_add(&coordinate_sums[0], sd0[0]);
        atomic_add(&coordinate_sums[1], sd1[0]);
        atomic_add(&coordinate_sums[2], sd2[0]);
        atomic_add(&coordinate_sums[3], sd3[0]);
    }
}

// Coordinate prefix sum kernel — subtracts shift from last column
__global__ void pos_blt_it_coord_prefix_sum(
    m31* coordinate_sums,
    unsigned n_cols,
    unsigned trace_size,
    m31** interactive_traces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < trace_size) {
        qm31 cs = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        qm31 shift = div(cs, m31(trace_size));

        interactive_traces[4 * n_cols - 4][idx] = sub(interactive_traces[4 * n_cols - 4][idx], shift.a.a);
        interactive_traces[4 * n_cols - 3][idx] = sub(interactive_traces[4 * n_cols - 3][idx], shift.a.b);
        interactive_traces[4 * n_cols - 2][idx] = sub(interactive_traces[4 * n_cols - 2][idx], shift.b.a);
        interactive_traces[4 * n_cols - 1][idx] = sub(interactive_traces[4 * n_cols - 1][idx], shift.b.b);
    }
}

// Helper macros for processing logup columns
#define POS_BLT_IT_PROCESS_ADD(col_idx, N1, N2, d0_ptrs, d1_ptrs) \
    pos_blt_it_add_pair_kernel<N1, N2><<<num_blocks, block_dim>>>( \
        d_lookup, d0_ptrs, d1_ptrs, trace_size, \
        device_logup_denom, numer0, numer1, numer2, numer3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    pos_blt_it_finalize_col_kernel<<<num_blocks, block_dim>>>(col_idx, trace_size, denom_inv, \
        numer0, numer1, numer2, numer3, device_it); \

#define POS_BLT_IT_PROCESS_SINGLE(col_idx, N, d_ptrs) \
    pos_blt_it_single_kernel<N><<<num_blocks, block_dim>>>( \
        d_lookup, d_ptrs, trace_size, \
        device_logup_denom, numer0, numer1, numer2, numer3); \
    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size); \
    pos_blt_it_finalize_col_kernel<<<num_blocks, block_dim>>>(col_idx, trace_size, denom_inv, \
        numer0, numer1, numer2, numer3, device_it); \

extern "C" void gen_poseidon_builtin_split_interaction_trace(
    void* lookup_elements,
    m31** lk_mem_0,          // 3 arrays
    m31** lk_mem_1,          // 3 arrays
    m31** lk_mem_2,          // 3 arrays
    m31** lk_mem_3,          // 3 arrays
    m31** lk_mem_4,          // 3 arrays
    m31** lk_mem_5,          // 3 arrays
    m31** lk_agg_0,          // 7 arrays
    uint32_t log_size,
    m31** interaction_trace_columns,   // 4 * POS_BLT_N_LOGUP_COLUMNS = 16 columns
    m31* claimed_sum                   // 4 m31s for qm31
) {
    uint32_t trace_size = 1u << log_size;

    // Copy lookup elements to device
    LookupElementsBasic<128>* d_lookup = cuda_malloc<LookupElementsBasic<128>>(1);
    cuda_mem_copy_host_to_device<LookupElementsBasic<128>>(
        (LookupElementsBasic<128>*)lookup_elements, d_lookup, 1);

    // Clone lookup data pointer arrays to device
    m31** d_mem_0 = clone_to_device<m31*>(lk_mem_0, 3);
    m31** d_mem_1 = clone_to_device<m31*>(lk_mem_1, 3);
    m31** d_mem_2 = clone_to_device<m31*>(lk_mem_2, 3);
    m31** d_mem_3 = clone_to_device<m31*>(lk_mem_3, 3);
    m31** d_mem_4 = clone_to_device<m31*>(lk_mem_4, 3);
    m31** d_mem_5 = clone_to_device<m31*>(lk_mem_5, 3);
    m31** d_agg_0 = clone_to_device<m31*>(lk_agg_0, 7);

    // Allocate working memory
    qm31* device_logup_denom = cuda_malloc<qm31>(trace_size);
    qm31* denom_inv = cuda_malloc<qm31>(trace_size);
    m31* numer0 = cuda_malloc<m31>(trace_size);
    m31* numer1 = cuda_malloc<m31>(trace_size);
    m31* numer2 = cuda_malloc<m31>(trace_size);
    m31* numer3 = cuda_malloc<m31>(trace_size);

    m31** device_it = clone_to_device<m31*>(interaction_trace_columns, 4 * POS_BLT_N_LOGUP_COLUMNS);

    int block_dim = trace_size < POS_BLT_IT_BLOCK_SIZE ? trace_size : POS_BLT_IT_BLOCK_SIZE;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Col 0: ADD pair (mem_0 + mem_1) — 3+3 elements
    POS_BLT_IT_PROCESS_ADD(0, 3, 3, d_mem_0, d_mem_1);

    // Col 1: ADD pair (mem_2 + mem_3) — 3+3 elements
    POS_BLT_IT_PROCESS_ADD(1, 3, 3, d_mem_2, d_mem_3);

    // Col 2: ADD pair (mem_4 + mem_5) — 3+3 elements
    POS_BLT_IT_PROCESS_ADD(2, 3, 3, d_mem_4, d_mem_5);

    // Col 3: SINGLE poseidon_aggregator — numerator=1, 7 elements
    POS_BLT_IT_PROCESS_SINGLE(3, 7, d_agg_0);

    // Finalize: cumsum_shift + prefix sum on last 4 columns
    cudaMemsetAsync(claimed_sum, 0, 4 * sizeof(m31), 0);

    size_t shared_size = 4 * block_dim * sizeof(m31);
    pos_blt_it_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        POS_BLT_N_LOGUP_COLUMNS, trace_size, device_it, claimed_sum);

    pos_blt_it_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum, POS_BLT_N_LOGUP_COLUMNS, trace_size, device_it);

    // Inclusive prefix sum on last 4 columns
    inclusive_prefix_sum(interaction_trace_columns[4 * POS_BLT_N_LOGUP_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * POS_BLT_N_LOGUP_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * POS_BLT_N_LOGUP_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[4 * POS_BLT_N_LOGUP_COLUMNS - 1], trace_size);

    // Cleanup
    cuda_free_memory(d_lookup);
    cuda_free_memory(d_mem_0);
    cuda_free_memory(d_mem_1);
    cuda_free_memory(d_mem_2);
    cuda_free_memory(d_mem_3);
    cuda_free_memory(d_mem_4);
    cuda_free_memory(d_mem_5);
    cuda_free_memory(d_agg_0);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(denom_inv);
    cuda_free_memory(numer0);
    cuda_free_memory(numer1);
    cuda_free_memory(numer2);
    cuda_free_memory(numer3);
    cuda_free_memory(device_it);
}
