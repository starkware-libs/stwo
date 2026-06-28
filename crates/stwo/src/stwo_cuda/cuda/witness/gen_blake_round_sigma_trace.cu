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

#include "relations.cuh"
#include "gen_blake_round_sigma_trace.cuh"

#define BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS 1
#define BLAKE_ROUND_SIGMA_TRACE_GEN_THREAD_COUNT_MAX 256

__constant__  m31 BLAKE_ROUND_SIGMA_CONSTS_DEV[17][16];
__constant__  m31 BLAKE_SIGMA_DEV[N_BLAKE_ROUNDS][N_BLAKE_SIGMA_COLS];


static bool constant_inited = false;
__global__ void blake_round_sigma_mults_init_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        uint32_t addr = (inputs[0][row]);
        atomicAdd(&mults[0][addr], 1);
    }
}


void blake_round_sigma_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults,
    unsigned mults_col_size,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 1 * input_col_sizes);
    m31 **device_mults = clone_to_device<m31*>(mults, 1 * mults_col_size);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    blake_round_sigma_mults_init_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        device_mults
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}

void init_packed_sigma(const m31 *round, m31 out[16][16]) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            m31 r = round[j];
            out[i][j] = (r >= N_BLAKE_ROUNDS) ? 0 : BLAKE_SIGMA[r][i];
        }
    }
}

template <int N>
__launch_bounds__(256, 2)
__global__ void generate_blake_round_sigma_interaction_trace_col_single_gen_kernel(
    LookupElementsBasic<N>  *lookup_elements_n,

    m31 **lookup_state_mults,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;

    m31 mults = lookup_state_mults[0][vec_index];

    // 17 => length of chain![[seq], sigmas]
    m31 init_combine_reg[17] = {};

    for (int i = 0; i < 17; i++) {
        init_combine_reg[i] = BLAKE_ROUND_SIGMA_CONSTS_DEV[i][vec_index];
    }

    if (vec_index < trace_size) {
        qm31 denom = lookup_elements_n->combine(init_combine_reg, 17);
        // if (vec_index == 0)
        //     print_qm31(denom, "denom");
       logup_col_write_frac(vec_index, mul(qm31{P-1, 0, 0, 0}, qm31{mults, 0, 0, 0}), denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

__global__ void generate_blake_round_sigma_interaction_trace_finalize_col_kernel(
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

__global__ void generate_blake_round_sigma_interaction_trace_cumsum_shift(
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

__global__ void generate_blake_round_sigma_interaction_trace_coord_prefix_sum(
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
void initialize_blake_round_sigma_constants() {
    m31 blake_round_sigma_consts[17][16];
    m31 sigmas[16][16];

    init_packed_sigma(SIGMA_DEDUCE_SOURCE, sigmas);

    memcpy(blake_round_sigma_consts[0], SIGMA_SEQ, 16 * sizeof(m31));
    memcpy(blake_round_sigma_consts[1], sigmas, 16 * 16 * sizeof(m31));

    // for (int i = 0; i < 17; ++i) {
    //     for (int j = 0; j < 16; ++j) {
    //     }
    // }

    cudaError_t err;

    err = cudaMemcpyToSymbol(BLAKE_ROUND_SIGMA_CONSTS_DEV, blake_round_sigma_consts,
                             sizeof(m31) * 17 * 16);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy BLAKE_ROUND_SIGMA_CONSTS_DEV to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMemcpyToSymbol(BLAKE_SIGMA_DEV, BLAKE_SIGMA,
                             sizeof(m31) * N_BLAKE_ROUNDS * N_BLAKE_SIGMA_COLS);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy BLAKE_SIGMA_DEV to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // printf("blake_round_sigma Constants initialized and copied to device memory successfully.\n");
}

void init_blake_round_sigma_constants_only_once() {
    if (!constant_inited) {
        initialize_blake_round_sigma_constants();
        constant_inited = true;
    }
}

void generate_blake_round_sigma_interaction_traces(
    void *blake_round_sigma,

    m31 **lookup_blake_round_sigma,

    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    init_blake_round_sigma_constants_only_once();

    unsigned trace_size = 1 << log_size;

    BlakeRoundSigma *blake_round_sigma_lookup_elements = (BlakeRoundSigma *)blake_round_sigma;

    BlakeRoundSigma *device_blake_round_sigma_lookup_elements = cuda_malloc<BlakeRoundSigma>(1);

    cuda_mem_copy_host_to_device<BlakeRoundSigma>(blake_round_sigma_lookup_elements, device_blake_round_sigma_lookup_elements, 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_blake_round_sigma = clone_to_device<m31*>(lookup_blake_round_sigma, 1);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4 * BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    // dump_lookup_data(lookup_blake_round_sigma, 1, trace_size);

    timer global_timer;
    global_timer.start("generate blake_round_sigma interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < BLAKE_ROUND_SIGMA_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : BLAKE_ROUND_SIGMA_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < BLAKE_ROUND_SIGMA_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For blake_round_sigma
    generate_blake_round_sigma_interaction_trace_col_single_gen_kernel<17><<<num_blocks, block_dim>>>(
        device_blake_round_sigma_lookup_elements,

        device_lookup_blake_round_sigma,

        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_numerator_data(device_numerator0, device_numerator1, device_numerator2, device_numerator3, trace_size);

    batch_inverse_secure_field(device_logup_denom, denom_inv, trace_size);

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_round_sigma_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        0,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // dump_interaction_traces(interaction_traces, 0, trace_size);

    // Compute cumsum_shift.
    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    size_t shared_size = 4 * block_dim * sizeof(m31);
    generate_blake_round_sigma_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_blake_round_sigma_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * BLAKE_ROUND_SIGMA_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate blake_round_sigma interaction trace");

    cuda_free_memory(device_blake_round_sigma_lookup_elements);
    cuda_free_memory(device_lookup_blake_round_sigma);

    // dump_interaction_traces(interaction_traces, 8, trace_size);

    cuda_free_memory(device_logup_denom);

    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}
