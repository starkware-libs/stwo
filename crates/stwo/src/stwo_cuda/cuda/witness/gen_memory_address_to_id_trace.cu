
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
#include "gen_memory_address_to_id_trace.cuh"
#include "../constraints/relations.cuh"

// Type alias for MemoryAddressToId
typedef LookupElementsBasic<2> MemoryAddressToId;

__global__ void memory_address_to_id_add_inputs_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        // Process all input columns (e.g., 29 for add_mod_builtin)
        for (unsigned col = 0; col < input_col_sizes; col++) {
            m31 value = inputs[col][row];
            // 0 indicates padding row, skip
            if (value != 0) {
                uint32_t addr = (value - 1);
                uint32_t mults_size = 1u << mults_row_log_size;
                if (addr < mults_size) {
                    atomicAdd(&mults[addr], 1);
                }
            }
        }
    }
}


void memory_address_to_id_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 1 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    memory_address_to_id_add_inputs_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        mults,
        mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}

#define MEMORY_ADDRESS_TO_ID_SPLIT 16
#define N_ID_AND_MULT_COLUMNS_PER_CHUNK 2
#define N_TRACE_COLUMNS (MEMORY_ADDRESS_TO_ID_SPLIT * N_ID_AND_MULT_COLUMNS_PER_CHUNK)

__launch_bounds__(256, 2)
__global__ void generate_memory_address_to_id_trace_kernel(
    m31 **traces,
    m31 *address_to_raw_id,
    m31 *multiplicities,
    unsigned total_size_log,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < trace_size) {
        // Calculate which chunk this row belongs to
        unsigned n_packed_rows = trace_size;

        // For each chunk, write ID and multiplicity
        for (unsigned chunk_idx = 0; chunk_idx < MEMORY_ADDRESS_TO_ID_SPLIT; chunk_idx++) {
            unsigned global_row = row + chunk_idx * trace_size;

            // Read ID and multiplicity from input arrays
            m31 id = {0};
            m31 mult = {0};

            if (global_row < (1u << total_size_log)) {
                id = address_to_raw_id[global_row];
                mult = multiplicities[global_row];
            }

            // Write to trace columns
            // Each chunk has 2 columns: [id, multiplicity]
            unsigned id_col = chunk_idx * N_ID_AND_MULT_COLUMNS_PER_CHUNK;
            unsigned mult_col = id_col + 1;

            traces[id_col][row] = id;
            traces[mult_col][row] = mult;
        }
    }
}

__launch_bounds__(256, 2)
__global__ void generate_memory_address_to_id_interaction_trace_kernel(
    m31 **interaction_traces,
    m31 **traces,
    unsigned log_size,
    unsigned trace_size,
    MemoryAddressToId *lookup_elements
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < trace_size) {
        // We process pairs of chunks (0,1), (2,3), ..., (14,15)
        // Each pair produces one interaction column
        unsigned num_pairs = MEMORY_ADDRESS_TO_ID_SPLIT / 2;

        for (unsigned pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
            unsigned chunk0 = pair_idx * 2;
            unsigned chunk1 = chunk0 + 1;

            // Read IDs and multiplicities for both chunks
            m31 id0 = traces[chunk0 * N_ID_AND_MULT_COLUMNS_PER_CHUNK][row];
            m31 mult0 = traces[chunk0 * N_ID_AND_MULT_COLUMNS_PER_CHUNK + 1][row];
            m31 id1 = traces[chunk1 * N_ID_AND_MULT_COLUMNS_PER_CHUNK][row];
            m31 mult1 = traces[chunk1 * N_ID_AND_MULT_COLUMNS_PER_CHUNK + 1][row];

            // Calculate addresses (row + 1 for offset, plus chunk offset)
            // Use proper M31 field arithmetic to avoid integer overflow
            m31 addr0 = add(add(m31{row}, m31{1}), mul(m31{chunk0}, m31{trace_size}));
            m31 addr1 = add(add(m31{row}, m31{1}), mul(m31{chunk1}, m31{trace_size}));

            // Compute logup values: combine(addr, id) for each
            m31 p0_inputs[2] = {addr0, id0};
            qm31 p0 = lookup_elements->combine(p0_inputs, 2);

            m31 p1_inputs[2] = {addr1, id1};
            qm31 p1 = lookup_elements->combine(p1_inputs, 2);

            // Compute numerator: p0 * (-mult1) + p1 * (-mult0)
            // Convert m31 to qm31: qm31{cm31{m, 0}, cm31{0, 0}}
            qm31 zero = qm31{cm31{0, 0}, cm31{0, 0}};
            qm31 qm31_mult0 = qm31{cm31{mult0, 0}, cm31{0, 0}};
            qm31 qm31_mult1 = qm31{cm31{mult1, 0}, cm31{0, 0}};
            qm31 neg_mult0 = sub(zero, qm31_mult0);
            qm31 neg_mult1 = sub(zero, qm31_mult1);

            qm31 term0 = mul(p0, neg_mult1);
            qm31 term1 = mul(p1, neg_mult0);
            qm31 numerator = add(term0, term1);

            // Compute denominator: p1 * p0
            qm31 denominator = mul(p1, p0);

            // Store numerator and denominator for this pair
            // Each interaction column is for one pair
            unsigned base_col = pair_idx * 8; // 8 M31 values per QM31 fraction (4 for num, 4 for denom)

            // Numerator (4 M31 values)
            interaction_traces[base_col + 0][row] = numerator.a.a;
            interaction_traces[base_col + 1][row] = numerator.a.b;
            interaction_traces[base_col + 2][row] = numerator.b.a;
            interaction_traces[base_col + 3][row] = numerator.b.b;

            // Denominator (4 M31 values)
            interaction_traces[base_col + 4][row] = denominator.a.a;
            interaction_traces[base_col + 5][row] = denominator.a.b;
            interaction_traces[base_col + 6][row] = denominator.b.a;
            interaction_traces[base_col + 7][row] = denominator.b.b;
        }
    }
}

void generate_memory_address_to_id_traces(
    m31 **traces,
    m31 **interaction_traces,
    m31 *address_to_raw_id,
    m31 *multiplicities,
    unsigned total_size_log,
    unsigned log_size,
    MemoryAddressToId *lookup_elements
) {
    unsigned trace_size = 1u << log_size;

    // Copy traces pointer to device
    m31 **device_traces = clone_to_device<m31*>(traces, N_TRACE_COLUMNS);

    // Generate main trace
    // Use 256 to match __launch_bounds__(256, 2) on the kernel
    int block_dim = trace_size < 256 ? trace_size : 256;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    generate_memory_address_to_id_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        address_to_raw_id,
        multiplicities,
        total_size_log,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Generate interaction trace if needed
    if (interaction_traces != nullptr && lookup_elements != nullptr) {
        unsigned num_interaction_cols = 8 * (MEMORY_ADDRESS_TO_ID_SPLIT / 2);
        m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, num_interaction_cols);

        // Copy lookup_elements to device memory
        MemoryAddressToId *device_lookup_elements = cuda_malloc<MemoryAddressToId>(1);
        ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(device_lookup_elements, lookup_elements, sizeof(MemoryAddressToId), cudaMemcpyHostToDevice, 0));

        generate_memory_address_to_id_interaction_trace_kernel<<<num_blocks, block_dim>>>(
            device_interaction_traces,
            device_traces,
            log_size,
            trace_size,
            device_lookup_elements
        );

        ASSERT_CUDA_SUCCESS(cudaGetLastError());

        cuda_free_memory(device_interaction_traces);
        cuda_free_memory(device_lookup_elements);
    }

    cuda_free_memory(device_traces);
}

// =============================================================================
// Full CUDA Interaction Trace Generation for memory_address_to_id
// =============================================================================

// Number of logup columns: 8 pairs of chunks
#define N_LOGUP_COLS (MEMORY_ADDRESS_TO_ID_SPLIT / 2)
// Number of BaseField columns: 8 logup columns × 4 BaseField = 32
#define N_INTERACTION_COLS (N_LOGUP_COLS * 4)

// Kernel to compute logup fractions for all 8 pairs.
// For each pair (chunk0, chunk1), computes:
//   p0 = lookup.combine([addr0, id0])
//   p1 = lookup.combine([addr1, id1])
//   numerator = p0 * (-mult1) + p1 * (-mult0)
//   denominator = p1 * p0
__global__ void memory_address_to_id_generate_interaction_frac_kernel(
    MemoryAddressToId *lookup_elements,
    m31 *padded_ids,                   // [padded_size] - all IDs
    m31 *padded_mults,                 // [padded_size] - all multiplicities
    unsigned total_size_log,           // log of padded total size
    unsigned trace_size,               // size of each trace chunk (1 << log_size)
    qm31 *denom_ptr,                   // [8 * trace_size] - 8 denominators per row
    m31 *numerator0,                   // [8 * trace_size]
    m31 *numerator1,                   // [8 * trace_size]
    m31 *numerator2,                   // [8 * trace_size]
    m31 *numerator3                    // [8 * trace_size]
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        unsigned padded_size = 1u << total_size_log;

        // Process all 8 pairs
        for (unsigned pair_idx = 0; pair_idx < N_LOGUP_COLS; ++pair_idx) {
            unsigned chunk0 = pair_idx * 2;
            unsigned chunk1 = chunk0 + 1;

            // Calculate global indices for both chunks
            unsigned global_idx0 = row + chunk0 * trace_size;
            unsigned global_idx1 = row + chunk1 * trace_size;

            // Read IDs and multiplicities (with bounds checking)
            m31 id0 = (global_idx0 < padded_size) ? padded_ids[global_idx0] : m31{0};
            m31 mult0 = (global_idx0 < padded_size) ? padded_mults[global_idx0] : m31{0};
            m31 id1 = (global_idx1 < padded_size) ? padded_ids[global_idx1] : m31{0};
            m31 mult1 = (global_idx1 < padded_size) ? padded_mults[global_idx1] : m31{0};

            // Calculate addresses (row + 1 for offset, plus chunk offset)
            // addr = row + 1 + chunk_idx * trace_size
            // Use proper M31 field arithmetic to avoid integer overflow
            m31 addr0 = add(add(m31{row}, m31{1}), mul(m31{chunk0}, m31{trace_size}));
            m31 addr1 = add(add(m31{row}, m31{1}), mul(m31{chunk1}, m31{trace_size}));

            // Compute p0 = lookup.combine([addr0, id0])
            m31 p0_inputs[2] = {addr0, id0};
            qm31 p0 = lookup_elements->combine(p0_inputs, 2);

            // Compute p1 = lookup.combine([addr1, id1])
            m31 p1_inputs[2] = {addr1, id1};
            qm31 p1 = lookup_elements->combine(p1_inputs, 2);

            // Compute numerator: p0 * (-mult1) + p1 * (-mult0)
            qm31 zero = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};
            qm31 qm31_mult0 = qm31{cm31{mult0, m31{0}}, cm31{m31{0}, m31{0}}};
            qm31 qm31_mult1 = qm31{cm31{mult1, m31{0}}, cm31{m31{0}, m31{0}}};
            qm31 neg_mult0 = sub(zero, qm31_mult0);
            qm31 neg_mult1 = sub(zero, qm31_mult1);

            qm31 term0 = mul(p0, neg_mult1);
            qm31 term1 = mul(p1, neg_mult0);
            qm31 numer = add(term0, term1);

            // Compute denominator: p1 * p0
            qm31 denom = mul(p1, p0);

            // Write frac using logup helper
            unsigned idx = pair_idx * trace_size + row;
            logup_col_write_frac(idx, numer, denom,
                                denom_ptr, numerator0, numerator1, numerator2, numerator3);
        }
    }
}

// Finalize interaction columns - write 8 logup columns with ACCUMULATION (32 BaseField columns total)
// Column set N contains the running sum of all fractions from 0 to N
__global__ void memory_address_to_id_finalize_interaction_col_kernel(
    unsigned trace_size,
    qm31 *denom_inv_ptr,               // [8 * trace_size]
    m31 *numerator0,                   // [8 * trace_size]
    m31 *numerator1,                   // [8 * trace_size]
    m31 *numerator2,                   // [8 * trace_size]
    m31 *numerator3,                   // [8 * trace_size]
    m31 **interaction_traces           // [32][trace_size] - 8 logup cols × 4 BaseField each
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Running sum of all fractions
        qm31 running_sum = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};

        // Write each of the 8 logup fractions with ACCUMULATION
        for (int i = 0; i < N_LOGUP_COLS; ++i) {
            unsigned idx = i * trace_size + row;
            qm31 frac = mul(
                qm31 {
                    cm31{numerator0[idx], numerator1[idx]},
                    cm31{numerator2[idx], numerator3[idx]}
                },
                denom_inv_ptr[idx]
            );

            // Accumulate the fraction
            running_sum = add(running_sum, frac);

            // Each logup column occupies 4 consecutive columns
            // Write the accumulated sum (not just the fraction)
            int base_col = i * 4;
            interaction_traces[base_col + 0][row] = running_sum.a.a;
            interaction_traces[base_col + 1][row] = running_sum.a.b;
            interaction_traces[base_col + 2][row] = running_sum.b.a;
            interaction_traces[base_col + 3][row] = running_sum.b.b;
        }
    }
}

// Compute cumulative sum for claimed_sum (parallel reduction)
// Only sums the LAST column (28-31) since with accumulation it contains the sum of all fractions
__global__ void memory_address_to_id_cumsum_kernel(
    unsigned trace_size,
    m31 **interaction_traces,          // [32][trace_size]
    m31 *coordinate_sums               // [4] - total sum
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = m31{0};
    m31 sum1 = m31{0};
    m31 sum2 = m31{0};
    m31 sum3 = m31{0};

    // Only sum the LAST column (index 7 = columns 28-31) which contains accumulated total
    int last_base_col = (N_LOGUP_COLS - 1) * 4;
    for (int i = tid; i < trace_size; i += gridSize) {
        sum0 = add(sum0, interaction_traces[last_base_col + 0][i]);
        sum1 = add(sum1, interaction_traces[last_base_col + 1][i]);
        sum2 = add(sum2, interaction_traces[last_base_col + 2][i]);
        sum3 = add(sum3, interaction_traces[last_base_col + 3][i]);
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

// Apply cumsum shift to interaction trace - only applies to the LAST column (28-31)
__global__ void memory_address_to_id_apply_cumsum_shift_kernel(
    m31 *coordinate_sums,
    unsigned trace_size,
    m31 **interaction_traces            // [32][trace_size]
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        // Shift is claimed_sum / trace_size (only for the last column)
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        // Only apply shift to the LAST column (28-31)
        int last_base_col = (N_LOGUP_COLS - 1) * 4;
        interaction_traces[last_base_col + 0][row] = sub(interaction_traces[last_base_col + 0][row], cumsum_shift.a.a);
        interaction_traces[last_base_col + 1][row] = sub(interaction_traces[last_base_col + 1][row], cumsum_shift.a.b);
        interaction_traces[last_base_col + 2][row] = sub(interaction_traces[last_base_col + 2][row], cumsum_shift.b.a);
        interaction_traces[last_base_col + 3][row] = sub(interaction_traces[last_base_col + 3][row], cumsum_shift.b.b);
    }
}

// C wrapper for full CUDA interaction trace generation
extern "C"
void memory_address_to_id_generate_interaction_trace(
    void *lookup_element_ptr,          // MemoryAddressToId relation pointer
    m31 *padded_ids,                   // device pointer - padded IDs
    m31 *padded_mults,                 // device pointer - padded multiplicities
    unsigned total_size_log,           // log of padded total size
    unsigned log_size,                 // log size of each trace chunk
    m31 **interaction_traces,          // [32] device pointers for output (8 logup × 4 BaseField)
    m31 *claimed_sum                   // [4] device pointer for output
) {
    unsigned trace_size = 1u << log_size;

    // Copy lookup elements to device
    MemoryAddressToId *lookup_elements = (MemoryAddressToId *)lookup_element_ptr;
    MemoryAddressToId *device_lookup_elements = clone_to_device<MemoryAddressToId>(lookup_elements, 1);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, N_INTERACTION_COLS);

    // Allocate temporary buffers for 8 fractions
    size_t total_fracs = N_LOGUP_COLS * trace_size;
    qm31 *device_logup_denom = cuda_malloc<qm31>(total_fracs);
    m31 *device_numerator0 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator1 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator2 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator3 = cuda_malloc<m31>(total_fracs);
    qm31 *denom_inv = cuda_malloc<qm31>(total_fracs);

    // Use 256 to match __launch_bounds__(256, 2) on related kernels
    int block_dim = trace_size < 256 ? trace_size : 256;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Step 1: Compute all 8 logup fractions per row
    memory_address_to_id_generate_interaction_frac_kernel<<<num_blocks, block_dim>>>(
        device_lookup_elements,
        padded_ids,
        padded_mults,
        total_size_log,
        trace_size,
        device_logup_denom,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 2: Batch inverse all denominators
    batch_inverse_secure_field(device_logup_denom, denom_inv, total_fracs);

    // Step 3: Finalize interaction columns (write 8 separate logup columns = 32 BaseField columns)
    memory_address_to_id_finalize_interaction_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 4: Compute claimed_sum via parallel reduction (sum last column)
    size_t shared_size = 4 * block_dim * sizeof(m31);
    memory_address_to_id_cumsum_kernel<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 5: Apply cumsum shift to the LAST column (28-31)
    memory_address_to_id_apply_cumsum_shift_kernel<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 6: Apply inclusive prefix sum ONLY to the LAST column (28-31)
    int last_base_col = (N_LOGUP_COLS - 1) * 4;
    inclusive_prefix_sum(interaction_traces[last_base_col + 0], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 1], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 2], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 3], trace_size);

    // Cleanup
    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}
