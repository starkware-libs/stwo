#include "gen_range_check_felt_252_width_27_trace.cuh"
#include "relations.cuh"
#include "logup.cuh"
#include "utils.cuh"
#include "fields.cuh"
#include "batch_inverse.cuh"
#include "prefix_sum.cuh"
#include "cuda_mem_pool.cuh"

// Constants
#define M31_262144 262144     // 2^18
#define M31_4194304 4194304   // 2^22

// Width27 to Felt252 conversion
// Extract specific 9-bit parts from Width27 limbs
// Width27 limb has 27 bits, we extract high part (bits 18-26) and low part (bits 0-8)
__device__ void width27_to_felt252_parts(
    m31 limb_0_w27,
    m31 limb_1_w27,
    m31 limb_2_w27,
    m31 limb_3_w27,
    m31 limb_4_w27,
    m31 limb_5_w27,
    m31 limb_6_w27,
    m31 limb_7_w27,
    m31 limb_8_w27,
    // Output: specific 9-bit limbs
    m31* limb_0_high,   // bits 18-26 of limb_0_w27
    m31* limb_1_low,    // bits 0-8 of limb_1_w27
    m31* limb_2_high,   // bits 18-26 of limb_2_w27
    m31* limb_3_low,    // bits 0-8 of limb_3_w27
    m31* limb_4_high,   // bits 18-26 of limb_4_w27
    m31* limb_5_low,    // bits 0-8 of limb_5_w27
    m31* limb_6_high,   // bits 18-26 of limb_6_w27
    m31* limb_7_low,    // bits 0-8 of limb_7_w27
    m31* limb_8_high    // bits 18-26 of limb_8_w27
) {
    // High part: (limb >> 18) & 0x1FF (9 bits)
    // Low part: limb & 0x1FF (9 bits)
    *limb_0_high = (limb_0_w27 >> 18) & 0x1FF;
    *limb_1_low = limb_1_w27 & 0x1FF;
    *limb_2_high = (limb_2_w27 >> 18) & 0x1FF;
    *limb_3_low = limb_3_w27 & 0x1FF;
    *limb_4_high = (limb_4_w27 >> 18) & 0x1FF;
    *limb_5_low = limb_5_w27 & 0x1FF;
    *limb_6_high = (limb_6_w27 >> 18) & 0x1FF;
    *limb_7_low = limb_7_w27 & 0x1FF;
    *limb_8_high = (limb_8_w27 >> 18) & 0x1FF;
}

// =============================================================================
// Phase 1: Base Trace Generation
// =============================================================================

__global__ void range_check_felt_252_width_27_generate_trace_kernel(
    m31** input_limbs,      // 10 input columns (Width27 format)
    unsigned int n_rows,    // Padded size (power of 2)
    unsigned int actual_n_rows,  // Actual data rows (before padding)
    m31** trace_columns     // 20 output trace columns
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Read Width27 input limbs
    m31 limb_0 = input_limbs[0][row];
    m31 limb_1 = input_limbs[1][row];
    m31 limb_2 = input_limbs[2][row];
    m31 limb_3 = input_limbs[3][row];
    m31 limb_4 = input_limbs[4][row];
    m31 limb_5 = input_limbs[5][row];
    m31 limb_6 = input_limbs[6][row];
    m31 limb_7 = input_limbs[7][row];
    m31 limb_8 = input_limbs[8][row];
    m31 limb_9 = input_limbs[9][row];

    // Write input limbs to trace (columns 0-9)
    trace_columns[0][row] = limb_0;
    trace_columns[1][row] = limb_1;
    trace_columns[2][row] = limb_2;
    trace_columns[3][row] = limb_3;
    trace_columns[4][row] = limb_4;
    trace_columns[5][row] = limb_5;
    trace_columns[6][row] = limb_6;
    trace_columns[7][row] = limb_7;
    trace_columns[8][row] = limb_8;
    trace_columns[9][row] = limb_9;

    // Extract high/low parts for range checks
    m31 limb_0_high, limb_1_low, limb_2_high, limb_3_low;
    m31 limb_4_high, limb_5_low, limb_6_high, limb_7_low;
    m31 limb_8_high;

    width27_to_felt252_parts(
        limb_0, limb_1, limb_2, limb_3, limb_4,
        limb_5, limb_6, limb_7, limb_8,
        &limb_0_high, &limb_1_low, &limb_2_high, &limb_3_low,
        &limb_4_high, &limb_5_low, &limb_6_high, &limb_7_low,
        &limb_8_high
    );

    // Write extracted parts to trace (columns 10-18)
    trace_columns[10][row] = limb_0_high;
    trace_columns[11][row] = limb_1_low;
    trace_columns[12][row] = limb_2_high;
    trace_columns[13][row] = limb_3_low;
    trace_columns[14][row] = limb_4_high;
    trace_columns[15][row] = limb_5_low;
    trace_columns[16][row] = limb_6_high;
    trace_columns[17][row] = limb_7_low;
    trace_columns[18][row] = limb_8_high;

    // Column 19: Enabler (1 for actual data rows, 0 for padding rows)
    trace_columns[19][row] = (row < actual_n_rows) ? 1 : 0;
}

// =============================================================================
// Phase 2: Compute LogUp Fractions (store numerator/denominator separately)
// =============================================================================

// Number of logup columns for this component
#define RC_FELT252_W27_N_LOGUP_COLS 8

__global__ void range_check_felt_252_width_27_compute_fractions_kernel(
    m31** trace_columns,
    unsigned int trace_size,
    // Lookup elements
    LookupElementsBasic<2>* rc_9_9,
    LookupElementsBasic<1>* rc_18,
    LookupElementsBasic<2>* rc_9_9_b,
    LookupElementsBasic<1>* rc_18_b,
    LookupElementsBasic<2>* rc_9_9_c,
    LookupElementsBasic<2>* rc_9_9_d,
    LookupElementsBasic<2>* rc_9_9_e,
    LookupElementsBasic<10>* rc_felt252_w27,
    // Output arrays for fractions
    qm31* denom_ptr,          // [8 * trace_size]
    m31* numerator0,          // [8 * trace_size]
    m31* numerator1,
    m31* numerator2,
    m31* numerator3
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Read trace values
    m31 limbs[10];
    for (int i = 0; i < 10; i++) {
        limbs[i] = trace_columns[i][row];
    }

    m31 limb_0_high = trace_columns[10][row];
    m31 limb_1_low = trace_columns[11][row];
    m31 limb_2_high = trace_columns[12][row];
    m31 limb_3_low = trace_columns[13][row];
    m31 limb_4_high = trace_columns[14][row];
    m31 limb_5_low = trace_columns[15][row];
    m31 limb_6_high = trace_columns[16][row];
    m31 limb_7_low = trace_columns[17][row];
    m31 limb_8_high = trace_columns[18][row];
    m31 enabler = trace_columns[19][row];

    // Compute range check values for rc_18
    // rc_18[0]: input_limb_0 - limb_0_high * 262144
    // rc_18[1]: (input_limb_1 - limb_1_low) * 4194304  (mod P handling via mul/sub)
    // etc.

    // Using modular arithmetic helpers
    m31 rc_18_vals[7];
    rc_18_vals[0] = sub((m31)limbs[0], mul((m31)limb_0_high, (m31)M31_262144));
    rc_18_vals[1] = mul(sub((m31)limbs[1], (m31)limb_1_low), (m31)M31_4194304);
    rc_18_vals[2] = mul(sub((m31)limbs[3], (m31)limb_3_low), (m31)M31_4194304);
    rc_18_vals[3] = sub((m31)limbs[4], mul((m31)limb_4_high, (m31)M31_262144));
    rc_18_vals[4] = mul(sub((m31)limbs[5], (m31)limb_5_low), (m31)M31_4194304);
    rc_18_vals[5] = mul(sub((m31)limbs[7], (m31)limb_7_low), (m31)M31_4194304);
    rc_18_vals[6] = sub((m31)limbs[8], mul((m31)limb_8_high, (m31)M31_262144));

    m31 rc_18_b_vals[2];
    rc_18_b_vals[0] = sub((m31)limbs[2], mul((m31)limb_2_high, (m31)M31_262144));
    rc_18_b_vals[1] = sub((m31)limbs[6], mul((m31)limb_6_high, (m31)M31_262144));

    // LogUp column 0: range_check_9_9[limb_0_high, limb_1_low] + range_check_18[rc_18_vals[0]]
    {
        m31 input0[2] = {limb_0_high, limb_1_low};
        m31 input1[1] = {rc_18_vals[0]};
        qm31 denom0 = rc_9_9->combine(input0, 2);
        qm31 denom1 = rc_18->combine(input1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 0 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 1: range_check_18[rc_18_vals[1]] + range_check_9_9_b[limb_2_high, limb_3_low]
    {
        m31 input0[1] = {rc_18_vals[1]};
        m31 input1[2] = {limb_2_high, limb_3_low};
        qm31 denom0 = rc_18->combine(input0, 1);
        qm31 denom1 = rc_9_9_b->combine(input1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 1 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 2: range_check_18_b[rc_18_b_vals[0]] + range_check_18[rc_18_vals[2]]
    {
        m31 input0[1] = {rc_18_b_vals[0]};
        m31 input1[1] = {rc_18_vals[2]};
        qm31 denom0 = rc_18_b->combine(input0, 1);
        qm31 denom1 = rc_18->combine(input1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 2 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 3: range_check_9_9_c[limb_4_high, limb_5_low] + range_check_18[rc_18_vals[3]]
    {
        m31 input0[2] = {limb_4_high, limb_5_low};
        m31 input1[1] = {rc_18_vals[3]};
        qm31 denom0 = rc_9_9_c->combine(input0, 2);
        qm31 denom1 = rc_18->combine(input1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 3 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 4: range_check_18[rc_18_vals[4]] + range_check_9_9_d[limb_6_high, limb_7_low]
    {
        m31 input0[1] = {rc_18_vals[4]};
        m31 input1[2] = {limb_6_high, limb_7_low};
        qm31 denom0 = rc_18->combine(input0, 1);
        qm31 denom1 = rc_9_9_d->combine(input1, 2);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 4 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 5: range_check_18_b[rc_18_b_vals[1]] + range_check_18[rc_18_vals[5]]
    {
        m31 input0[1] = {rc_18_b_vals[1]};
        m31 input1[1] = {rc_18_vals[5]};
        qm31 denom0 = rc_18_b->combine(input0, 1);
        qm31 denom1 = rc_18->combine(input1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 5 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 6: range_check_9_9_e[limb_8_high, limb_9] + range_check_18[rc_18_vals[6]]
    {
        m31 input0[2] = {limb_8_high, limbs[9]};
        m31 input1[1] = {rc_18_vals[6]};
        qm31 denom0 = rc_9_9_e->combine(input0, 2);
        qm31 denom1 = rc_18->combine(input1, 1);
        qm31 numer = add(denom0, denom1);
        qm31 denom = mul(denom0, denom1);
        unsigned int idx = 6 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }

    // LogUp column 7: Self-lookup with -enabler multiplier
    {
        m31 input[10];
        for (int i = 0; i < 10; i++) {
            input[i] = limbs[i];
        }
        qm31 denom = rc_felt252_w27->combine(input, 10);
        // Numerator is -enabler (providing the lookup)
        qm31 numer = qm31{cm31{neg(enabler), 0}, cm31{0, 0}};
        unsigned int idx = 7 * trace_size + row;
        logup_col_write_frac(idx, numer, denom, denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

// =============================================================================
// Phase 3: Finalize interaction columns with accumulation
// =============================================================================

__global__ void range_check_felt_252_width_27_finalize_interaction_kernel(
    unsigned int trace_size,
    qm31* denom_inv_ptr,
    m31* numerator0,
    m31* numerator1,
    m31* numerator2,
    m31* numerator3,
    m31** interaction_traces   // [32][trace_size] - 8 logup cols × 4 BaseField each
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    // Running sum of all fractions
    qm31 running_sum = qm31{cm31{0, 0}, cm31{0, 0}};

    // Write each of the 8 logup fractions with ACCUMULATION
    for (int i = 0; i < RC_FELT252_W27_N_LOGUP_COLS; ++i) {
        unsigned int idx = i * trace_size + row;
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
        int base_col = i * 4;
        interaction_traces[base_col + 0][row] = running_sum.a.a;
        interaction_traces[base_col + 1][row] = running_sum.a.b;
        interaction_traces[base_col + 2][row] = running_sum.b.a;
        interaction_traces[base_col + 3][row] = running_sum.b.b;
    }
}

// =============================================================================
// Phase 4: Compute cumulative sum (only sum last column)
// =============================================================================

__global__ void range_check_felt_252_width_27_cumsum_kernel(
    unsigned int trace_size,
    m31** interaction_traces,
    m31* coordinate_sums      // [4] - total sum
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    m31 sum0 = 0;
    m31 sum1 = 0;
    m31 sum2 = 0;
    m31 sum3 = 0;

    // Only sum the LAST column (index 7 = columns 28-31)
    int last_base_col = (RC_FELT252_W27_N_LOGUP_COLS - 1) * 4;
    for (int i = tid; i < (int)trace_size; i += gridSize) {
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

// =============================================================================
// Phase 5: Apply cumsum shift and prefix sum to last column
// =============================================================================

__global__ void range_check_felt_252_width_27_apply_cumsum_shift_kernel(
    m31* coordinate_sums,
    unsigned int trace_size,
    m31** interaction_traces
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= trace_size) return;

    qm31 claimed_sum = qm31 {
        cm31{coordinate_sums[0], coordinate_sums[1]},
        cm31{coordinate_sums[2], coordinate_sums[3]}
    };
    qm31 cumsum_shift = div(claimed_sum, (m31)trace_size);

    // Only apply shift to the LAST column (28-31)
    int last_base_col = (RC_FELT252_W27_N_LOGUP_COLS - 1) * 4;
    qm31 current = qm31 {
        cm31{interaction_traces[last_base_col + 0][row], interaction_traces[last_base_col + 1][row]},
        cm31{interaction_traces[last_base_col + 2][row], interaction_traces[last_base_col + 3][row]}
    };
    qm31 shifted = sub(current, cumsum_shift);
    interaction_traces[last_base_col + 0][row] = shifted.a.a;
    interaction_traces[last_base_col + 1][row] = shifted.a.b;
    interaction_traces[last_base_col + 2][row] = shifted.b.a;
    interaction_traces[last_base_col + 3][row] = shifted.b.b;
}

// =============================================================================
// Host wrapper functions
// =============================================================================

extern "C" {

void range_check_felt_252_width_27_generate_trace(
    m31** input_limbs,
    unsigned int n_rows,        // Padded size (power of 2)
    unsigned int actual_n_rows, // Actual data rows (before padding)
    m31** trace_columns
) {
    m31** device_input_limbs = clone_to_device<m31*>(input_limbs, 10);
    m31** device_trace_columns = clone_to_device<m31*>(trace_columns, RANGE_CHECK_FELT_252_WIDTH_27_N_TRACE_COLUMNS);

    unsigned int block_size = 256;
    unsigned int grid_size = (n_rows + block_size - 1) / block_size;

    range_check_felt_252_width_27_generate_trace_kernel<<<grid_size, block_size>>>(
        device_input_limbs,
        n_rows,
        actual_n_rows,
        device_trace_columns
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_input_limbs);
    cuda_free_memory(device_trace_columns);
}

// Kernel to add multiplicities for range_check_felt_252_width_27 component
__global__ void range_check_felt_252_width_27_add_to_multiplicities_kernel(
    m31** trace_columns,
    unsigned int n_rows,
    m31* rc_18_mults,
    unsigned int rc_18_log_size,
    m31* rc_18_b_mults,
    unsigned int rc_18_b_log_size,
    m31* rc_9_9_mults,
    unsigned int rc_9_9_log_size,
    m31* rc_9_9_b_mults,
    unsigned int rc_9_9_b_log_size,
    m31* rc_9_9_c_mults,
    unsigned int rc_9_9_c_log_size,
    m31* rc_9_9_d_mults,
    unsigned int rc_9_9_d_log_size,
    m31* rc_9_9_e_mults,
    unsigned int rc_9_9_e_log_size
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Trace layout:
    // Columns 0-9: Input Width27 limbs
    // Columns 10-18: Extracted 9-bit high/low parts for rc_9_9 lookups
    //   Col 10: limb_0_high (bits 18-26 of limb_0)
    //   Col 11: limb_1_low  (bits 0-8 of limb_1)
    //   Col 12: limb_2_high (bits 18-26 of limb_2)
    //   Col 13: limb_3_low  (bits 0-8 of limb_3)
    //   Col 14: limb_4_high (bits 18-26 of limb_4)
    //   Col 15: limb_5_low  (bits 0-8 of limb_5)
    //   Col 16: limb_6_high (bits 18-26 of limb_6)
    //   Col 17: limb_7_low  (bits 0-8 of limb_7)
    //   Col 18: limb_8_high (bits 18-26 of limb_8)

    // Read input values
    m31 limbs[10];
    for (int i = 0; i < 10; i++) {
        limbs[i] = trace_columns[i][row];
    }
    m31 limb_0_high = trace_columns[10][row];
    m31 limb_1_low = trace_columns[11][row];
    m31 limb_2_high = trace_columns[12][row];
    m31 limb_3_low = trace_columns[13][row];
    m31 limb_4_high = trace_columns[14][row];
    m31 limb_5_low = trace_columns[15][row];
    m31 limb_6_high = trace_columns[16][row];
    m31 limb_7_low = trace_columns[17][row];
    m31 limb_8_high = trace_columns[18][row];

    // Compute rc_18 lookup values (must match interaction trace logic)
    // rc_18[0] = input_limb_0 - limb_0_high * 262144
    // rc_18[1] = (input_limb_1 - limb_1_low) * 4194304
    // rc_18[2] = (input_limb_3 - limb_3_low) * 4194304
    // rc_18[3] = input_limb_4 - limb_4_high * 262144
    // rc_18[4] = (input_limb_5 - limb_5_low) * 4194304
    // rc_18[5] = (input_limb_7 - limb_7_low) * 4194304
    // rc_18[6] = input_limb_8 - limb_8_high * 262144
    m31 rc_18_vals[7];
    rc_18_vals[0] = sub(limbs[0], mul(limb_0_high, (m31)M31_262144));
    rc_18_vals[1] = mul(sub(limbs[1], limb_1_low), (m31)M31_4194304);
    rc_18_vals[2] = mul(sub(limbs[3], limb_3_low), (m31)M31_4194304);
    rc_18_vals[3] = sub(limbs[4], mul(limb_4_high, (m31)M31_262144));
    rc_18_vals[4] = mul(sub(limbs[5], limb_5_low), (m31)M31_4194304);
    rc_18_vals[5] = mul(sub(limbs[7], limb_7_low), (m31)M31_4194304);
    rc_18_vals[6] = sub(limbs[8], mul(limb_8_high, (m31)M31_262144));

    // Compute rc_18_b lookup values
    // rc_18_b[0] = input_limb_2 - limb_2_high * 262144
    // rc_18_b[1] = input_limb_6 - limb_6_high * 262144
    m31 rc_18_b_vals[2];
    rc_18_b_vals[0] = sub(limbs[2], mul(limb_2_high, (m31)M31_262144));
    rc_18_b_vals[1] = sub(limbs[6], mul(limb_6_high, (m31)M31_262144));

    // rc_18 lookups (7 lookups using computed values)
    for (int i = 0; i < 7; i++) {
        uint32_t val = rc_18_vals[i] & 0x3FFFF; // 18-bit mask
        if (val < (1u << rc_18_log_size)) {
            atomicAdd(&rc_18_mults[val], 1);
        }
    }

    // rc_18_b lookups (2 lookups using computed values)
    for (int i = 0; i < 2; i++) {
        uint32_t val = rc_18_b_vals[i] & 0x3FFFF; // 18-bit mask
        if (val < (1u << rc_18_b_log_size)) {
            atomicAdd(&rc_18_b_mults[val], 1);
        }
    }

    // rc_9_9 lookup: (limb_0_high, limb_1_low) = (col[10], col[11])
    {
        uint32_t high = limb_0_high & 0x1FF; // 9-bit
        uint32_t low = limb_1_low & 0x1FF;  // 9-bit
        uint32_t addr = (high << 9) | low; // 18-bit index
        if (addr < (1u << rc_9_9_log_size)) {
            atomicAdd(&rc_9_9_mults[addr], 1);
        }
    }

    // rc_9_9_b lookup: (limb_2_high, limb_3_low) = (col[12], col[13])
    {
        uint32_t high = limb_2_high & 0x1FF;
        uint32_t low = limb_3_low & 0x1FF;
        uint32_t addr = (high << 9) | low;
        if (addr < (1u << rc_9_9_b_log_size)) {
            atomicAdd(&rc_9_9_b_mults[addr], 1);
        }
    }

    // rc_9_9_c lookup: (limb_4_high, limb_5_low) = (col[14], col[15])
    {
        uint32_t high = limb_4_high & 0x1FF;
        uint32_t low = limb_5_low & 0x1FF;
        uint32_t addr = (high << 9) | low;
        if (addr < (1u << rc_9_9_c_log_size)) {
            atomicAdd(&rc_9_9_c_mults[addr], 1);
        }
    }

    // rc_9_9_d lookup: (limb_6_high, limb_7_low) = (col[16], col[17])
    {
        uint32_t high = limb_6_high & 0x1FF;
        uint32_t low = limb_7_low & 0x1FF;
        uint32_t addr = (high << 9) | low;
        if (addr < (1u << rc_9_9_d_log_size)) {
            atomicAdd(&rc_9_9_d_mults[addr], 1);
        }
    }

    // rc_9_9_e lookup: (limb_8_high, limb_9) = (col[18], col[9])
    {
        uint32_t high = limb_8_high & 0x1FF;
        uint32_t low = limbs[9] & 0x1FF;  // limb_9 is the second part
        uint32_t addr = (high << 9) | low;
        if (addr < (1u << rc_9_9_e_log_size)) {
            atomicAdd(&rc_9_9_e_mults[addr], 1);
        }
    }
}

void range_check_felt_252_width_27_add_to_multiplicities(
    m31** trace_columns,
    unsigned int n_rows,
    m31* rc_18_mults,
    unsigned int rc_18_log_size,
    m31* rc_18_b_mults,
    unsigned int rc_18_b_log_size,
    m31* rc_9_9_mults,
    unsigned int rc_9_9_log_size,
    m31* rc_9_9_b_mults,
    unsigned int rc_9_9_b_log_size,
    m31* rc_9_9_c_mults,
    unsigned int rc_9_9_c_log_size,
    m31* rc_9_9_d_mults,
    unsigned int rc_9_9_d_log_size,
    m31* rc_9_9_e_mults,
    unsigned int rc_9_9_e_log_size
) {
    if (n_rows == 0) return;

    // Copy trace column pointers to device
    m31** device_trace_columns = clone_to_device<m31*>(trace_columns, RANGE_CHECK_FELT_252_WIDTH_27_N_TRACE_COLUMNS);

    // Launch kernel
    unsigned int block_size = 256;
    unsigned int grid_size = (n_rows + block_size - 1) / block_size;

    range_check_felt_252_width_27_add_to_multiplicities_kernel<<<grid_size, block_size>>>(
        device_trace_columns,
        n_rows,
        rc_18_mults,
        rc_18_log_size,
        rc_18_b_mults,
        rc_18_b_log_size,
        rc_9_9_mults,
        rc_9_9_log_size,
        rc_9_9_b_mults,
        rc_9_9_b_log_size,
        rc_9_9_c_mults,
        rc_9_9_c_log_size,
        rc_9_9_d_mults,
        rc_9_9_d_log_size,
        rc_9_9_e_mults,
        rc_9_9_e_log_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    cuda_free_memory(device_trace_columns);
}

void range_check_felt_252_width_27_generate_interaction_trace(
    m31** trace_columns,
    unsigned int trace_size,
    void* range_check_9_9_lookup_elements,
    void* range_check_18_lookup_elements,
    void* range_check_9_9_b_lookup_elements,
    void* range_check_18_b_lookup_elements,
    void* range_check_9_9_c_lookup_elements,
    void* range_check_9_9_d_lookup_elements,
    void* range_check_9_9_e_lookup_elements,
    void* range_check_felt_252_width_27_lookup_elements,
    m31** interaction_trace_columns,
    qm31* claimed_sum
) {
    m31** device_trace_columns = clone_to_device<m31*>(trace_columns, RANGE_CHECK_FELT_252_WIDTH_27_N_TRACE_COLUMNS);
    m31** device_interaction_traces = clone_to_device<m31*>(interaction_trace_columns, 4 * RC_FELT252_W27_N_LOGUP_COLS);

    unsigned int n_fractions = RC_FELT252_W27_N_LOGUP_COLS * trace_size;

    // Allocate temporary arrays for fraction computation
    qm31* denom_ptr;
    qm31* denom_inv;
    m31* numerator0;
    m31* numerator1;
    m31* numerator2;
    m31* numerator3;

    denom_ptr = cuda_mem_pool_allocate<qm31>(n_fractions);
    denom_inv = cuda_mem_pool_allocate<qm31>(n_fractions);
    numerator0 = cuda_mem_pool_allocate<m31>(n_fractions);
    numerator1 = cuda_mem_pool_allocate<m31>(n_fractions);
    numerator2 = cuda_mem_pool_allocate<m31>(n_fractions);
    numerator3 = cuda_mem_pool_allocate<m31>(n_fractions);

    // Clone lookup elements to device
    LookupElementsBasic<2>* d_rc_9_9 = clone_to_device<LookupElementsBasic<2>>((LookupElementsBasic<2>*)range_check_9_9_lookup_elements, 1);
    LookupElementsBasic<1>* d_rc_18 = clone_to_device<LookupElementsBasic<1>>((LookupElementsBasic<1>*)range_check_18_lookup_elements, 1);
    LookupElementsBasic<2>* d_rc_9_9_b = clone_to_device<LookupElementsBasic<2>>((LookupElementsBasic<2>*)range_check_9_9_b_lookup_elements, 1);
    LookupElementsBasic<1>* d_rc_18_b = clone_to_device<LookupElementsBasic<1>>((LookupElementsBasic<1>*)range_check_18_b_lookup_elements, 1);
    LookupElementsBasic<2>* d_rc_9_9_c = clone_to_device<LookupElementsBasic<2>>((LookupElementsBasic<2>*)range_check_9_9_c_lookup_elements, 1);
    LookupElementsBasic<2>* d_rc_9_9_d = clone_to_device<LookupElementsBasic<2>>((LookupElementsBasic<2>*)range_check_9_9_d_lookup_elements, 1);
    LookupElementsBasic<2>* d_rc_9_9_e = clone_to_device<LookupElementsBasic<2>>((LookupElementsBasic<2>*)range_check_9_9_e_lookup_elements, 1);
    LookupElementsBasic<10>* d_rc_felt252_w27 = clone_to_device<LookupElementsBasic<10>>((LookupElementsBasic<10>*)range_check_felt_252_width_27_lookup_elements, 1);

    unsigned int block_size = 256;
    unsigned int grid_size = (trace_size + block_size - 1) / block_size;

    // Phase 1: Compute fractions
    range_check_felt_252_width_27_compute_fractions_kernel<<<grid_size, block_size>>>(
        device_trace_columns,
        trace_size,
        d_rc_9_9, d_rc_18, d_rc_9_9_b, d_rc_18_b,
        d_rc_9_9_c, d_rc_9_9_d, d_rc_9_9_e, d_rc_felt252_w27,
        denom_ptr, numerator0, numerator1, numerator2, numerator3
    );

    // Phase 2: Batch inverse on denominators
    batch_inverse_secure_field(denom_ptr, denom_inv, n_fractions);

    // Phase 3: Finalize with accumulation
    range_check_felt_252_width_27_finalize_interaction_kernel<<<grid_size, block_size>>>(
        trace_size,
        denom_inv,
        numerator0, numerator1, numerator2, numerator3,
        device_interaction_traces
    );

    // Phase 4: Compute cumulative sum (for claimed_sum)
    m31* d_coordinate_sums;
    d_coordinate_sums = cuda_mem_pool_allocate<m31>(4);
    cudaMemsetAsync(d_coordinate_sums, 0, 4 * sizeof(m31), 0);

    int cumsum_block_size = 256;
    int cumsum_grid_size = (trace_size + cumsum_block_size - 1) / cumsum_block_size;
    if (cumsum_grid_size > 256) cumsum_grid_size = 256;

    range_check_felt_252_width_27_cumsum_kernel<<<cumsum_grid_size, cumsum_block_size, 4 * cumsum_block_size * sizeof(m31)>>>(
        trace_size,
        device_interaction_traces,
        d_coordinate_sums
    );

    // Read claimed_sum from device and write to device output pointer
    m31 h_sums[4];
    cudaMemcpyAsync(h_sums, d_coordinate_sums, 4 * sizeof(m31), cudaMemcpyDeviceToHost, 0);
    qm31 h_claimed_sum = qm31{cm31{h_sums[0], h_sums[1]}, cm31{h_sums[2], h_sums[3]}};
    cudaMemcpyAsync(claimed_sum, &h_claimed_sum, sizeof(qm31), cudaMemcpyHostToDevice, 0);

    // Phase 5: Apply cumsum shift to last column
    range_check_felt_252_width_27_apply_cumsum_shift_kernel<<<grid_size, block_size>>>(
        d_coordinate_sums,
        trace_size,
        device_interaction_traces
    );

    // Phase 6: Inclusive prefix sum on last column (columns 28-31)
    int last_base_col = (RC_FELT252_W27_N_LOGUP_COLS - 1) * 4;
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 0], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 1], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 2], trace_size);
    inclusive_prefix_sum(interaction_trace_columns[last_base_col + 3], trace_size);

    // Cleanup
    cuda_free_memory(device_trace_columns);
    cuda_free_memory(device_interaction_traces);
    cuda_mem_pool_free(denom_ptr);
    cuda_mem_pool_free(denom_inv);
    cuda_mem_pool_free(numerator0);
    cuda_mem_pool_free(numerator1);
    cuda_mem_pool_free(numerator2);
    cuda_mem_pool_free(numerator3);
    cuda_free_memory(d_rc_9_9);
    cuda_free_memory(d_rc_18);
    cuda_free_memory(d_rc_9_9_b);
    cuda_free_memory(d_rc_18_b);
    cuda_free_memory(d_rc_9_9_c);
    cuda_free_memory(d_rc_9_9_d);
    cuda_free_memory(d_rc_9_9_e);
    cuda_free_memory(d_rc_felt252_w27);
    cuda_mem_pool_free(d_coordinate_sums);
}

} // extern "C"
