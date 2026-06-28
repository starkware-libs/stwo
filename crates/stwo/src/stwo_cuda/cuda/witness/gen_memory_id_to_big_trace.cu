#include "relations.cuh"

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

#include "gen_memory_id_to_big_trace.cuh"

__global__ void memory_id_to_big_deduce_kernel(
    unsigned **transpose_big_value_ptr,
    unsigned *small_value_ptr,
    unsigned id,
    m31 * felt252_out
) {
    unsigned out_values[8];
    EncodedMemoryValueId emv;
    emv.encoded = id;

    MemoryValueId mv = decode_memory_value_id(&emv);

    for (int j = 0; j < 8; ++j) {
        switch (mv.tag) {
            case MEMORY_VALUE_ID_F252: {
                out_values[j] = transpose_big_value_ptr[j][mv.value];
                break;
            }
            case MEMORY_VALUE_ID_SMALL: {
                if (j >= 4) {
                    out_values[j] = 0;
                } else {
                    uint32_t limbs[4];
                    u128_to_4_limbs(((u128 *)small_value_ptr)[mv.value], limbs);
                    out_values[j] = limbs[j];
                }
                break;
            }
            case MEMORY_VALUE_ID_EMPTY: {
                printf("Attempted deduce_output on empty memory cell.\\n");
                return;
            }
            default: {
                printf("Invalid MemoryValueId tag: %d\\n", mv.tag);
                return;
            }
        }
    }

    printf("value: ");
    for (int i = 0; i < 8; i++) {
           printf("%d ", out_values[i]);
    }
    printf("\n");

    split_f252(out_values, felt252_out);
}


void memory_id_to_big_deduce_finese_cuda(
    unsigned **transpose_big_value_ptr,
    unsigned *small_value_ptr,
    unsigned id,
    m31 * felt252_out
) {

    unsigned **device_transpose_big_value_ptr = clone_to_device<m31 *>(transpose_big_value_ptr, 8);

    // m31 *felt252_out_dev = clone_to_device<m31 *>(felt252_out, 1);
    printf("run memory_id_to_big_deduce_finese_cuda\n");


    memory_id_to_big_deduce_kernel<<<1, 1>>>(
        device_transpose_big_value_ptr,
        small_value_ptr,
        id,
        felt252_out
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());


    cuda_free_memory(device_transpose_big_value_ptr);
    // cuda_free_memory(felt252_out_dev);
}



__global__ void memory_id_to_big_add_inputs_kernel(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *big_mults,
    unsigned big_mults_row_log_size,
    m31 *small_mults,
    unsigned small_mults_row_log_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input_row_sizes) {
        // Process all input columns (e.g., 24 for add_mod_builtin)
        for (unsigned col = 0; col < input_col_sizes; col++) {
            EncodedMemoryValueId emv;
            emv.encoded = inputs[col][row];
            MemoryValueId mv = decode_memory_value_id(&emv);

            switch (mv.tag) {
                case MEMORY_VALUE_ID_F252: {
                    atomicAdd(&big_mults[mv.value], 1);
                    break;
                }
                case MEMORY_VALUE_ID_SMALL: {
                    atomicAdd(&small_mults[mv.value], 1);
                    break;
                }
                case MEMORY_VALUE_ID_EMPTY: {
                    // 0 indicates padding row, skip silently
                    break;
                }
                default: {
                    printf("Invalid MemoryValueId tag: %d\\n", mv.tag);
                    break;
                }
            }
        }
    }
}


void memory_id_to_big_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *big_mults,
    unsigned big_mults_row_log_size,
    m31 *small_mults,
    unsigned small_mults_row_log_size
) {
    m31 **device_inputs = clone_to_device<m31*>(inputs, 1 * input_col_sizes);

    int block_dim = input_row_sizes < THREAD_COUNT_MAX ? input_row_sizes : THREAD_COUNT_MAX;
    int num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (input_row_sizes + block_dim - 1) / block_dim;

    memory_id_to_big_add_inputs_kernel<<<num_blocks, block_dim>>>(
        device_inputs,
        input_col_sizes,
        input_row_sizes,
        big_mults,
        big_mults_row_log_size,
        small_mults,
        small_mults_row_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_inputs);
}


// =============================================================================
// Phase 1: Base Trace Generation Kernels
// =============================================================================

#define MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX 256

// Kernel to generate the big memory trace.
// Reads transposed big values [8][n_values], applies split_f252, writes to 28 value columns + 1 mult column.
__global__ void memory_id_to_big_generate_big_trace_kernel(
    unsigned **transposed_big_values,  // [8][n_values] - column-major storage
    unsigned *multiplicities,          // [n_values]
    unsigned trace_size,               // padded to power of 2
    unsigned n_values,                 // actual number of values (before padding)
    m31 **trace_columns                // output: [29][trace_size] - 28 value cols + 1 mult col
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Read 8 u32 limbs from transposed storage
        unsigned limbs[8];
        if (row < n_values) {
            for (int j = 0; j < 8; ++j) {
                limbs[j] = transposed_big_values[j][row];
            }
        } else {
            // Padding rows: all zeros
            for (int j = 0; j < 8; ++j) {
                limbs[j] = 0;
            }
        }

        // Split 8 u32 limbs into 28 M31 values (9-bit packing)
        m31 split_values[N_M31_IN_FELT252];
        split_f252(limbs, split_values);

        // Write to value columns
        for (int j = 0; j < N_M31_IN_FELT252; ++j) {
            trace_columns[j][row] = split_values[j];
        }

        // Write multiplicity
        if (row < n_values) {
            trace_columns[N_M31_IN_FELT252][row] = m31{multiplicities[row]};
        } else {
            trace_columns[N_M31_IN_FELT252][row] = m31{0};
        }
    }
}

// C wrapper for big trace generation
extern "C"
void memory_id_to_big_generate_big_trace(
    unsigned **transposed_big_values,  // [8] device pointers
    unsigned *multiplicities,          // device pointer
    unsigned trace_size,               // padded to power of 2
    unsigned n_values,                 // actual number of values
    m31 **trace_columns                // [29] device pointers for output
) {
    unsigned **device_transposed = clone_to_device<unsigned*>(transposed_big_values, 8);
    m31 **device_trace_columns = clone_to_device<m31*>(trace_columns, N_M31_IN_FELT252 + 1);

    int block_dim = trace_size < MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX
                  ? trace_size : MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    memory_id_to_big_generate_big_trace_kernel<<<num_blocks, block_dim>>>(
        device_transposed,
        multiplicities,
        trace_size,
        n_values,
        device_trace_columns
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_transposed);
    cuda_free_memory(device_trace_columns);
}

// Kernel to generate the small memory trace.
// Reads small values as u128, extracts 4 u32 limbs, pads with zeros, applies split_f252 for 8 M31 values.
__global__ void memory_id_to_big_generate_small_trace_kernel(
    u128 *small_values,                // [n_values] as u128
    unsigned *multiplicities,          // [n_values]
    unsigned trace_size,               // padded to power of 2
    unsigned n_values,                 // actual number of values
    m31 **trace_columns                // output: [9][trace_size] - 8 value cols + 1 mult col
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Extract 4 u32 limbs from u128, pad with zeros for total of 8
        unsigned limbs[8];
        if (row < n_values) {
            uint32_t extracted[4];
            u128_to_4_limbs(small_values[row], extracted);
            limbs[0] = extracted[0];
            limbs[1] = extracted[1];
            limbs[2] = extracted[2];
            limbs[3] = extracted[3];
        } else {
            limbs[0] = 0;
            limbs[1] = 0;
            limbs[2] = 0;
            limbs[3] = 0;
        }
        limbs[4] = 0;
        limbs[5] = 0;
        limbs[6] = 0;
        limbs[7] = 0;

        // Split into 28 M31 values, but we only need first 8
        m31 split_values[N_M31_IN_FELT252];
        split_f252(limbs, split_values);

        // Write to value columns (only first 8)
        for (int j = 0; j < N_M31_IN_SMALL_FELT252; ++j) {
            trace_columns[j][row] = split_values[j];
        }

        // Write multiplicity
        if (row < n_values) {
            trace_columns[N_M31_IN_SMALL_FELT252][row] = m31{multiplicities[row]};
        } else {
            trace_columns[N_M31_IN_SMALL_FELT252][row] = m31{0};
        }
    }
}

// C wrapper for small trace generation
extern "C"
void memory_id_to_big_generate_small_trace(
    u128 *small_values,                // device pointer (as u128)
    unsigned *multiplicities,          // device pointer
    unsigned trace_size,               // padded to power of 2
    unsigned n_values,                 // actual number of values
    m31 **trace_columns                // [9] device pointers for output
) {
    m31 **device_trace_columns = clone_to_device<m31*>(trace_columns, N_M31_IN_SMALL_FELT252 + 1);

    int block_dim = trace_size < MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX
                  ? trace_size : MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    memory_id_to_big_generate_small_trace_kernel<<<num_blocks, block_dim>>>(
        small_values,
        multiplicities,
        trace_size,
        n_values,
        device_trace_columns
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_trace_columns);
}


// =============================================================================
// Phase 2: Interaction Trace Generation Kernels
// =============================================================================

// Number of logup columns for big memory: 7 range check + 1 memory value = 8
#define N_BIG_LOGUP_COLS 8

// Kernel to compute ALL logup fractions for big memory interaction trace.
// For each row, computes:
//   - 7 range check fractions: (denom0 + denom1) / (denom0 * denom1) for each 4-limb group
//   - 1 memory value fraction: -mult / lookup.combine([id, limbs])
// All 8 fractions are accumulated into the same output.
__global__ void memory_id_to_big_generate_big_interaction_frac_kernel(
    MemoryIdToBig *lookup_elements,
    RangeCheck_9_9 *rc_9_9,
    RangeCheck_9_9_B *rc_9_9_b,
    RangeCheck_9_9_C *rc_9_9_c,
    RangeCheck_9_9_D *rc_9_9_d,
    RangeCheck_9_9_E *rc_9_9_e,
    RangeCheck_9_9_F *rc_9_9_f,
    RangeCheck_9_9_G *rc_9_9_g,
    RangeCheck_9_9_H *rc_9_9_h,
    m31 **value_columns,               // [28][trace_size] - already split M31 values
    unsigned *multiplicities,          // [trace_size]
    unsigned trace_size,
    unsigned id_offset,                // offset for split tables
    qm31 *denom_ptr,                   // [8 * trace_size] - 8 denominators per row
    m31 *numerator0,                   // [8 * trace_size]
    m31 *numerator1,                   // [8 * trace_size]
    m31 *numerator2,                   // [8 * trace_size]
    m31 *numerator3                    // [8 * trace_size]
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Read all 28 limbs for this row
        m31 limbs[N_M31_IN_FELT252];
        for (int j = 0; j < N_M31_IN_FELT252; ++j) {
            limbs[j] = value_columns[j][row];
        }

        // Compute 7 range check fractions
        // Each group of 4 limbs produces one fraction: (d0+d1)/(d0*d1)
        for (int i = 0; i < 7; ++i) {
            int base_idx = i * 4;
            m31 limb0 = limbs[base_idx];
            m31 limb1 = limbs[base_idx + 1];
            m31 limb2 = limbs[base_idx + 2];
            m31 limb3 = limbs[base_idx + 3];

            m31 rc_input0[2] = {limb0, limb1};
            m31 rc_input1[2] = {limb2, limb3};

            qm31 denom0, denom1;
            switch (i % 4) {
                case 0:
                    denom0 = rc_9_9->combine(rc_input0, 2);
                    denom1 = rc_9_9_b->combine(rc_input1, 2);
                    break;
                case 1:
                    denom0 = rc_9_9_c->combine(rc_input0, 2);
                    denom1 = rc_9_9_d->combine(rc_input1, 2);
                    break;
                case 2:
                    denom0 = rc_9_9_e->combine(rc_input0, 2);
                    denom1 = rc_9_9_f->combine(rc_input1, 2);
                    break;
                case 3:
                    denom0 = rc_9_9_g->combine(rc_input0, 2);
                    denom1 = rc_9_9_h->combine(rc_input1, 2);
                    break;
            }

            // Write frac: (denom0 + denom1) / (denom0 * denom1)
            qm31 numer = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            unsigned idx = i * trace_size + row;
            logup_col_write_frac(idx, numer, denom,
                                denom_ptr, numerator0, numerator1, numerator2, numerator3);
        }

        // Compute memory value fraction: -mult / lookup.combine([id, limbs])
        m31 combine_input[29];
        combine_input[0] = m31{row + id_offset};  // id with offset
        for (int j = 0; j < N_M31_IN_FELT252; ++j) {
            combine_input[j + 1] = limbs[j];
        }
        qm31 mem_denom = lookup_elements->combine(combine_input, 29);
        qm31 mem_numer = qm31{cm31{neg(m31{multiplicities[row]}), m31{0}}, cm31{m31{0}, m31{0}}};
        unsigned mem_idx = 7 * trace_size + row;
        logup_col_write_frac(mem_idx, mem_numer, mem_denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Finalize interaction columns - write 8 logup columns with ACCUMULATION (32 BaseField columns total)
// Column set N contains the running sum of all fractions from 0 to N
__global__ void memory_id_to_big_finalize_interaction_col_kernel(
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
        for (int i = 0; i < N_BIG_LOGUP_COLS; ++i) {
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
__global__ void memory_id_to_big_cumsum_kernel(
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
    int last_base_col = (N_BIG_LOGUP_COLS - 1) * 4;
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
__global__ void memory_id_to_big_apply_cumsum_shift_kernel(
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
        int last_base_col = (N_BIG_LOGUP_COLS - 1) * 4;
        interaction_traces[last_base_col + 0][row] = sub(interaction_traces[last_base_col + 0][row], cumsum_shift.a.a);
        interaction_traces[last_base_col + 1][row] = sub(interaction_traces[last_base_col + 1][row], cumsum_shift.a.b);
        interaction_traces[last_base_col + 2][row] = sub(interaction_traces[last_base_col + 2][row], cumsum_shift.b.a);
        interaction_traces[last_base_col + 3][row] = sub(interaction_traces[last_base_col + 3][row], cumsum_shift.b.b);
    }
}

// Number of BaseField columns for big memory: 8 logup columns × 4 = 32
#define N_BIG_INTERACTION_COLS 32

// C wrapper for big memory interaction trace generation
extern "C"
void memory_id_to_big_generate_big_interaction_trace(
    void *lookup_element_ptr,          // MemoryIdToBig relation pointer
    void *rc_9_9_ptr,                  // RangeCheck_9_9 relation pointer
    void *rc_9_9_b_ptr,                // RangeCheck_9_9_B relation pointer
    void *rc_9_9_c_ptr,                // RangeCheck_9_9_C relation pointer
    void *rc_9_9_d_ptr,                // RangeCheck_9_9_D relation pointer
    void *rc_9_9_e_ptr,                // RangeCheck_9_9_E relation pointer
    void *rc_9_9_f_ptr,                // RangeCheck_9_9_F relation pointer
    void *rc_9_9_g_ptr,                // RangeCheck_9_9_G relation pointer
    void *rc_9_9_h_ptr,                // RangeCheck_9_9_H relation pointer
    m31 **value_columns,               // [28] device pointers - already split M31 values
    unsigned *multiplicities,          // device pointer
    unsigned trace_size,
    unsigned id_offset,                // offset for split tables
    m31 **interaction_traces,          // [32] device pointers for output (8 logup × 4 BaseField)
    m31 *claimed_sum                   // [4] device pointer for output
) {
    // Copy all relation elements to device
    MemoryIdToBig *lookup_elements = (MemoryIdToBig *)lookup_element_ptr;
    MemoryIdToBig *device_lookup_elements = clone_to_device<MemoryIdToBig>(lookup_elements, 1);

    RangeCheck_9_9 *device_rc_9_9 = clone_to_device<RangeCheck_9_9>((RangeCheck_9_9 *)rc_9_9_ptr, 1);
    RangeCheck_9_9_B *device_rc_9_9_b = clone_to_device<RangeCheck_9_9_B>((RangeCheck_9_9_B *)rc_9_9_b_ptr, 1);
    RangeCheck_9_9_C *device_rc_9_9_c = clone_to_device<RangeCheck_9_9_C>((RangeCheck_9_9_C *)rc_9_9_c_ptr, 1);
    RangeCheck_9_9_D *device_rc_9_9_d = clone_to_device<RangeCheck_9_9_D>((RangeCheck_9_9_D *)rc_9_9_d_ptr, 1);
    RangeCheck_9_9_E *device_rc_9_9_e = clone_to_device<RangeCheck_9_9_E>((RangeCheck_9_9_E *)rc_9_9_e_ptr, 1);
    RangeCheck_9_9_F *device_rc_9_9_f = clone_to_device<RangeCheck_9_9_F>((RangeCheck_9_9_F *)rc_9_9_f_ptr, 1);
    RangeCheck_9_9_G *device_rc_9_9_g = clone_to_device<RangeCheck_9_9_G>((RangeCheck_9_9_G *)rc_9_9_g_ptr, 1);
    RangeCheck_9_9_H *device_rc_9_9_h = clone_to_device<RangeCheck_9_9_H>((RangeCheck_9_9_H *)rc_9_9_h_ptr, 1);

    m31 **device_value_columns = clone_to_device<m31*>(value_columns, N_M31_IN_FELT252);
    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, N_BIG_INTERACTION_COLS);

    // Allocate temporary buffers for 8 fractions (7 range check + 1 memory value)
    size_t total_fracs = N_BIG_LOGUP_COLS * trace_size;
    qm31 *device_logup_denom = cuda_malloc<qm31>(total_fracs);
    m31 *device_numerator0 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator1 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator2 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator3 = cuda_malloc<m31>(total_fracs);
    qm31 *denom_inv = cuda_malloc<qm31>(total_fracs);

    int block_dim = trace_size < MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX
                  ? trace_size : MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Step 1: Compute all 8 logup fractions per row
    memory_id_to_big_generate_big_interaction_frac_kernel<<<num_blocks, block_dim>>>(
        device_lookup_elements,
        device_rc_9_9,
        device_rc_9_9_b,
        device_rc_9_9_c,
        device_rc_9_9_d,
        device_rc_9_9_e,
        device_rc_9_9_f,
        device_rc_9_9_g,
        device_rc_9_9_h,
        device_value_columns,
        multiplicities,
        trace_size,
        id_offset,
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
    memory_id_to_big_finalize_interaction_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 4: Compute claimed_sum via parallel reduction (sum all fractions before shift)
    size_t shared_size = 4 * block_dim * sizeof(m31);
    memory_id_to_big_cumsum_kernel<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 5: Apply cumsum shift to the LAST column (28-31)
    memory_id_to_big_apply_cumsum_shift_kernel<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 6: Apply inclusive prefix sum ONLY to the LAST column (28-31)
    int last_base_col = (N_BIG_LOGUP_COLS - 1) * 4;
    inclusive_prefix_sum(interaction_traces[last_base_col + 0], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 1], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 2], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 3], trace_size);

    // Cleanup
    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_rc_9_9);
    cuda_free_memory(device_rc_9_9_b);
    cuda_free_memory(device_rc_9_9_c);
    cuda_free_memory(device_rc_9_9_d);
    cuda_free_memory(device_rc_9_9_e);
    cuda_free_memory(device_rc_9_9_f);
    cuda_free_memory(device_rc_9_9_g);
    cuda_free_memory(device_rc_9_9_h);
    cuda_free_memory(device_value_columns);
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}

// Number of logup columns for small memory: 2 range check + 1 memory value = 3
#define N_SMALL_LOGUP_COLS 3

// Kernel to compute ALL logup fractions for small memory interaction trace.
// For each row, computes:
//   - 2 range check fractions: (denom0 + denom1) / (denom0 * denom1) for each 4-limb group
//   - 1 memory value fraction: -mult / lookup.combine([id, limbs, zeros])
// All 3 fractions are accumulated into the same output.
__global__ void memory_id_to_big_generate_small_interaction_frac_kernel(
    MemoryIdToBig *lookup_elements,
    RangeCheck_9_9 *rc_9_9,
    RangeCheck_9_9_B *rc_9_9_b,
    RangeCheck_9_9_C *rc_9_9_c,
    RangeCheck_9_9_D *rc_9_9_d,
    m31 **value_columns,               // [8][trace_size] - already split M31 values
    unsigned *multiplicities,          // [trace_size]
    unsigned trace_size,
    qm31 *denom_ptr,                   // [3 * trace_size] - 3 denominators per row
    m31 *numerator0,                   // [3 * trace_size]
    m31 *numerator1,                   // [3 * trace_size]
    m31 *numerator2,                   // [3 * trace_size]
    m31 *numerator3                    // [3 * trace_size]
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Read all 8 limbs for this row
        m31 limbs[N_M31_IN_SMALL_FELT252];
        for (int j = 0; j < N_M31_IN_SMALL_FELT252; ++j) {
            limbs[j] = value_columns[j][row];
        }

        // Compute 2 range check fractions
        // Each group of 4 limbs produces one fraction: (d0+d1)/(d0*d1)
        for (int i = 0; i < 2; ++i) {
            int base_idx = i * 4;
            m31 limb0 = limbs[base_idx];
            m31 limb1 = limbs[base_idx + 1];
            m31 limb2 = limbs[base_idx + 2];
            m31 limb3 = limbs[base_idx + 3];

            m31 rc_input0[2] = {limb0, limb1};
            m31 rc_input1[2] = {limb2, limb3};

            qm31 denom0, denom1;
            switch (i % 2) {
                case 0:
                    denom0 = rc_9_9->combine(rc_input0, 2);
                    denom1 = rc_9_9_b->combine(rc_input1, 2);
                    break;
                case 1:
                    denom0 = rc_9_9_c->combine(rc_input0, 2);
                    denom1 = rc_9_9_d->combine(rc_input1, 2);
                    break;
            }

            // Write frac: (denom0 + denom1) / (denom0 * denom1)
            qm31 numer = add(denom0, denom1);
            qm31 denom = mul(denom0, denom1);
            unsigned idx = i * trace_size + row;
            logup_col_write_frac(idx, numer, denom,
                                denom_ptr, numerator0, numerator1, numerator2, numerator3);
        }

        // Compute memory value fraction: -mult / lookup.combine([id, limbs, zeros])
        m31 combine_input[29];
        combine_input[0] = m31{row};  // id (no offset for small)
        for (int j = 0; j < N_M31_IN_SMALL_FELT252; ++j) {
            combine_input[j + 1] = limbs[j];
        }
        // Pad remaining with zeros
        for (int j = N_M31_IN_SMALL_FELT252; j < N_M31_IN_FELT252; ++j) {
            combine_input[j + 1] = m31{0};
        }
        qm31 mem_denom = lookup_elements->combine(combine_input, 29);
        qm31 mem_numer = qm31{cm31{neg(m31{multiplicities[row]}), m31{0}}, cm31{m31{0}, m31{0}}};
        unsigned mem_idx = 2 * trace_size + row;
        logup_col_write_frac(mem_idx, mem_numer, mem_denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// Number of BaseField columns for small memory: 3 logup columns × 4 = 12
#define N_SMALL_INTERACTION_COLS 12

// Finalize interaction columns for small memory - write 3 logup columns with ACCUMULATION (12 BaseField columns total)
// Column set N contains the running sum of all fractions from 0 to N
__global__ void memory_id_to_big_finalize_small_interaction_col_kernel(
    unsigned trace_size,
    qm31 *denom_inv_ptr,               // [3 * trace_size]
    m31 *numerator0,                   // [3 * trace_size]
    m31 *numerator1,                   // [3 * trace_size]
    m31 *numerator2,                   // [3 * trace_size]
    m31 *numerator3,                   // [3 * trace_size]
    m31 **interaction_traces           // [12][trace_size] - 3 logup cols × 4 BaseField each
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // Running sum of all fractions
        qm31 running_sum = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};

        // Write each of the 3 logup fractions with ACCUMULATION
        for (int i = 0; i < N_SMALL_LOGUP_COLS; ++i) {
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

// Compute cumulative sum for small memory claimed_sum (parallel reduction)
// Only sums the LAST column (8-11) since with accumulation it contains the sum of all fractions
__global__ void memory_id_to_big_cumsum_small_kernel(
    unsigned trace_size,
    m31 **interaction_traces,          // [12][trace_size]
    m31 *coordinate_sums               // [4] - total sum
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = m31{0};
    m31 sum1 = m31{0};
    m31 sum2 = m31{0};
    m31 sum3 = m31{0};

    // Only sum the LAST column (index 2 = columns 8-11) which contains accumulated total
    int last_base_col = (N_SMALL_LOGUP_COLS - 1) * 4;
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

// Apply cumsum shift to small memory interaction trace - only applies to the LAST column (8-11)
__global__ void memory_id_to_big_apply_cumsum_shift_small_kernel(
    m31 *coordinate_sums,
    unsigned trace_size,
    m31 **interaction_traces            // [12][trace_size]
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        qm31 claimed_sum = qm31 {
            cm31{coordinate_sums[0], coordinate_sums[1]},
            cm31{coordinate_sums[2], coordinate_sums[3]}
        };
        // Shift is claimed_sum / trace_size (only for the last column)
        qm31 cumsum_shift = div(claimed_sum, m31(trace_size));

        // Only apply shift to the LAST column (8-11)
        int last_base_col = (N_SMALL_LOGUP_COLS - 1) * 4;
        interaction_traces[last_base_col + 0][row] = sub(interaction_traces[last_base_col + 0][row], cumsum_shift.a.a);
        interaction_traces[last_base_col + 1][row] = sub(interaction_traces[last_base_col + 1][row], cumsum_shift.a.b);
        interaction_traces[last_base_col + 2][row] = sub(interaction_traces[last_base_col + 2][row], cumsum_shift.b.a);
        interaction_traces[last_base_col + 3][row] = sub(interaction_traces[last_base_col + 3][row], cumsum_shift.b.b);
    }
}

// C wrapper for small memory interaction trace generation
extern "C"
void memory_id_to_big_generate_small_interaction_trace(
    void *lookup_element_ptr,          // MemoryIdToBig relation pointer
    void *rc_9_9_ptr,                  // RangeCheck_9_9 relation pointer
    void *rc_9_9_b_ptr,                // RangeCheck_9_9_B relation pointer
    void *rc_9_9_c_ptr,                // RangeCheck_9_9_C relation pointer
    void *rc_9_9_d_ptr,                // RangeCheck_9_9_D relation pointer
    m31 **value_columns,               // [8] device pointers - already split M31 values
    unsigned *multiplicities,          // device pointer
    unsigned trace_size,
    m31 **interaction_traces,          // [12] device pointers for output (3 logup × 4 BaseField)
    m31 *claimed_sum                   // [4] device pointer for output
) {
    // Copy all relation elements to device
    MemoryIdToBig *lookup_elements = (MemoryIdToBig *)lookup_element_ptr;
    MemoryIdToBig *device_lookup_elements = clone_to_device<MemoryIdToBig>(lookup_elements, 1);

    RangeCheck_9_9 *device_rc_9_9 = clone_to_device<RangeCheck_9_9>((RangeCheck_9_9 *)rc_9_9_ptr, 1);
    RangeCheck_9_9_B *device_rc_9_9_b = clone_to_device<RangeCheck_9_9_B>((RangeCheck_9_9_B *)rc_9_9_b_ptr, 1);
    RangeCheck_9_9_C *device_rc_9_9_c = clone_to_device<RangeCheck_9_9_C>((RangeCheck_9_9_C *)rc_9_9_c_ptr, 1);
    RangeCheck_9_9_D *device_rc_9_9_d = clone_to_device<RangeCheck_9_9_D>((RangeCheck_9_9_D *)rc_9_9_d_ptr, 1);

    m31 **device_value_columns = clone_to_device<m31*>(value_columns, N_M31_IN_SMALL_FELT252);
    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, N_SMALL_INTERACTION_COLS);

    // Allocate temporary buffers for 3 fractions (2 range check + 1 memory value)
    size_t total_fracs = N_SMALL_LOGUP_COLS * trace_size;
    qm31 *device_logup_denom = cuda_malloc<qm31>(total_fracs);
    m31 *device_numerator0 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator1 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator2 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator3 = cuda_malloc<m31>(total_fracs);
    qm31 *denom_inv = cuda_malloc<qm31>(total_fracs);

    int block_dim = trace_size < MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX
                  ? trace_size : MEMORY_ID_TO_BIG_TRACE_THREAD_COUNT_MAX;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Step 1: Compute all 3 logup fractions per row
    memory_id_to_big_generate_small_interaction_frac_kernel<<<num_blocks, block_dim>>>(
        device_lookup_elements,
        device_rc_9_9,
        device_rc_9_9_b,
        device_rc_9_9_c,
        device_rc_9_9_d,
        device_value_columns,
        multiplicities,
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

    // Step 3: Finalize interaction columns (write 3 separate logup columns = 12 BaseField columns)
    memory_id_to_big_finalize_small_interaction_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 4: Compute claimed_sum via parallel reduction (sum across all 3 logup columns)
    size_t shared_size = 4 * block_dim * sizeof(m31);
    memory_id_to_big_cumsum_small_kernel<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 5: Apply cumsum shift to the LAST column (8-11)
    memory_id_to_big_apply_cumsum_shift_small_kernel<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 6: Apply inclusive prefix sum ONLY to the LAST column (8-11)
    int last_base_col = (N_SMALL_LOGUP_COLS - 1) * 4;
    inclusive_prefix_sum(interaction_traces[last_base_col + 0], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 1], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 2], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 3], trace_size);

    // Cleanup
    cuda_free_memory(device_lookup_elements);
    cuda_free_memory(device_rc_9_9);
    cuda_free_memory(device_rc_9_9_b);
    cuda_free_memory(device_rc_9_9_c);
    cuda_free_memory(device_rc_9_9_d);
    cuda_free_memory(device_value_columns);
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}
