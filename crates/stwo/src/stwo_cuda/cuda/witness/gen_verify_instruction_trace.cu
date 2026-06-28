// CUDA trace generation for verify_instruction component
// Generates 17 trace columns for instruction verification

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

#include "gen_verify_instruction_trace.cuh"
#include "gen_memory_address_to_id_trace.cuh"

// Lookup element types are already defined in relations.cuh:
// - VerifyInstruction (LookupElementsBasic<7>)
// - MemoryAddressToId (LookupElementsBasic<2>)
// - MemoryIdToBig (LookupElementsBasic<29>)
// - RangeCheck_7_2_5 (LookupElementsBasic<3>)
// - RangeCheck_4_3 (LookupElementsBasic<2>)

// Bit masks for offset encoding
#define MASK_9_BITS 0x1FF   // 511
#define MASK_7_BITS 0x7F    // 127
#define MASK_5_BITS 0x1F    // 31
#define MASK_4_BITS 0x0F    // 15
#define MASK_3_BITS 0x07    // 7
#define MASK_2_BITS 0x03    // 3

__launch_bounds__(256, 2)
__global__ void generate_verify_instruction_trace_kernel(
    m31 **traces,

    // Lookup data outputs (for interaction trace)
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_4_3_0,
    m31 **lookup_verify_instruction_0,

    // Sub-component inputs
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_range_check_7_2_5,
    m31 **sub_component_inputs_range_check_4_3,

    // Input data: 7 input limbs per row
    m31 **verify_instruction_inputs,

    // Multiplicities
    m31 *multiplicities,

    // Memory address to ID state for deduce_output
    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned n_rows,
    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_0   = {0};
    const m31 M31_32  = {32};
    const m31 M31_128 = {128};

    if (row < trace_size) {
        // Get input values - only process valid rows
        m31 input_limb_0_col0, input_limb_1_col1, input_limb_2_col2, input_limb_3_col3;
        m31 input_limb_4_col4, input_limb_5_col5, input_limb_6_col6;
        m31 mult;

        if (row < n_rows) {
            input_limb_0_col0 = verify_instruction_inputs[0][row];
            input_limb_1_col1 = verify_instruction_inputs[1][row];
            input_limb_2_col2 = verify_instruction_inputs[2][row];
            input_limb_3_col3 = verify_instruction_inputs[3][row];
            input_limb_4_col4 = verify_instruction_inputs[4][row];
            input_limb_5_col5 = verify_instruction_inputs[5][row];
            input_limb_6_col6 = verify_instruction_inputs[6][row];
            mult = multiplicities[row];
        } else {
            // Padding rows - all zeros
            input_limb_0_col0 = M31_0;
            input_limb_1_col1 = M31_0;
            input_limb_2_col2 = M31_0;
            input_limb_3_col3 = M31_0;
            input_limb_4_col4 = M31_0;
            input_limb_5_col5 = M31_0;
            input_limb_6_col6 = M31_0;
            mult = M31_0;
        }

        // Write input limbs to columns 0-6
        traces[0][row] = input_limb_0_col0;
        traces[1][row] = input_limb_1_col1;
        traces[2][row] = input_limb_2_col2;
        traces[3][row] = input_limb_3_col3;
        traces[4][row] = input_limb_4_col4;
        traces[5][row] = input_limb_5_col5;
        traces[6][row] = input_limb_6_col6;

        // Encode Offsets (columns 7-14)
        // Extract from input_limb_1 (col 1):
        //   offset0_low: bits 0-8 (9 bits) -> & 511
        //   offset0_mid: bits 9-15 (7 bits) -> >> 9
        uint16_t limb1_val = (uint16_t)(input_limb_1_col1);
        m31 offset0_low_col7 = (m31)(limb1_val & MASK_9_BITS);
        m31 offset0_mid_col8 = (m31)(limb1_val >> 9);
        traces[7][row] = offset0_low_col7;
        traces[8][row] = offset0_mid_col8;

        // Extract from input_limb_2 (col 2):
        //   offset1_low: bits 0-1 (2 bits) -> & 3
        //   offset1_mid: bits 2-10 (9 bits) -> (>> 2) & 511
        //   offset1_high: bits 11-15 (5 bits) -> >> 11
        uint16_t limb2_val = (uint16_t)(input_limb_2_col2);
        m31 offset1_low_col9 = (m31)(limb2_val & MASK_2_BITS);
        m31 offset1_mid_col10 = (m31)((limb2_val >> 2) & MASK_9_BITS);
        m31 offset1_high_col11 = (m31)(limb2_val >> 11);
        traces[9][row] = offset1_low_col9;
        traces[10][row] = offset1_mid_col10;
        traces[11][row] = offset1_high_col11;

        // Extract from input_limb_3 (col 3):
        //   offset2_low: bits 0-3 (4 bits) -> & 15
        //   offset2_mid: bits 4-12 (9 bits) -> (>> 4) & 511
        //   offset2_high: bits 13-15 (3 bits) -> >> 13
        uint16_t limb3_val = (uint16_t)(input_limb_3_col3);
        m31 offset2_low_col12 = (m31)(limb3_val & MASK_4_BITS);
        m31 offset2_mid_col13 = (m31)((limb3_val >> 4) & MASK_9_BITS);
        m31 offset2_high_col14 = (m31)(limb3_val >> 13);
        traces[12][row] = offset2_low_col12;
        traces[13][row] = offset2_mid_col13;
        traces[14][row] = offset2_high_col14;

        // Range check sub-component inputs
        // range_check_7_2_5: [offset0_mid, offset1_low, offset1_high]
        sub_component_inputs_range_check_7_2_5[0][row] = offset0_mid_col8;
        sub_component_inputs_range_check_7_2_5[1][row] = offset1_low_col9;
        sub_component_inputs_range_check_7_2_5[2][row] = offset1_high_col11;
        lookup_range_check_7_2_5_0[0][row] = offset0_mid_col8;
        lookup_range_check_7_2_5_0[1][row] = offset1_low_col9;
        lookup_range_check_7_2_5_0[2][row] = offset1_high_col11;

        // range_check_4_3: [offset2_low, offset2_high]
        sub_component_inputs_range_check_4_3[0][row] = offset2_low_col12;
        sub_component_inputs_range_check_4_3[1][row] = offset2_high_col14;
        lookup_range_check_4_3_0[0][row] = offset2_low_col12;
        lookup_range_check_4_3_0[1][row] = offset2_high_col14;

        // Mem Verify - Read Id
        // Get instruction_id from memory_address_to_id using input_limb_0 as address
        m31 instruction_id_col15 = M31_0;
        if (row < n_rows && input_limb_0_col0 != 0) {
            memory_address_to_id_deduce_output(
                memory_address_to_id_address_to_raw_id,
                input_limb_0_col0,
                &instruction_id_col15
            );
        }
        traces[15][row] = instruction_id_col15;

        // memory_address_to_id sub-component input
        sub_component_inputs_memory_address_to_id[0][row] = input_limb_0_col0;
        lookup_memory_address_to_id_0[0][row] = input_limb_0_col0;
        lookup_memory_address_to_id_0[1][row] = instruction_id_col15;

        // Compute encode_offsets outputs for memory_id_to_big lookup
        // encode_offsets[1] = offset0_mid + offset1_low * 128
        // encode_offsets[3] = offset1_high + offset2_low * 32
        m31 encode_offsets_1 = add(offset0_mid_col8, mul(offset1_low_col9, M31_128));
        m31 encode_offsets_3 = add(offset1_high_col11, mul(offset2_low_col12, M31_32));

        // memory_id_to_big sub-component input and lookup
        // lookup values: [instruction_id, offset0_low, encode[1], offset1_mid, encode[3],
        //                 offset2_mid, offset2_high + input_limb_4, input_limb_5, input_limb_6, 0, 0, ...]
        sub_component_inputs_memory_id_to_big[0][row] = instruction_id_col15;

        lookup_memory_id_to_big_0[0][row] = instruction_id_col15;
        lookup_memory_id_to_big_0[1][row] = offset0_low_col7;
        lookup_memory_id_to_big_0[2][row] = encode_offsets_1;
        lookup_memory_id_to_big_0[3][row] = offset1_mid_col10;
        lookup_memory_id_to_big_0[4][row] = encode_offsets_3;
        lookup_memory_id_to_big_0[5][row] = offset2_mid_col13;
        lookup_memory_id_to_big_0[6][row] = add(offset2_high_col14, input_limb_4_col4);
        lookup_memory_id_to_big_0[7][row] = input_limb_5_col5;
        lookup_memory_id_to_big_0[8][row] = input_limb_6_col6;
        // Remaining 20 values are zeros
        for (int i = 9; i < 29; i++) {
            lookup_memory_id_to_big_0[i][row] = M31_0;
        }

        // verify_instruction lookup (for interaction trace)
        lookup_verify_instruction_0[0][row] = input_limb_0_col0;
        lookup_verify_instruction_0[1][row] = input_limb_1_col1;
        lookup_verify_instruction_0[2][row] = input_limb_2_col2;
        lookup_verify_instruction_0[3][row] = input_limb_3_col3;
        lookup_verify_instruction_0[4][row] = input_limb_4_col4;
        lookup_verify_instruction_0[5][row] = input_limb_5_col5;
        lookup_verify_instruction_0[6][row] = input_limb_6_col6;

        // Multiplicity column (column 16)
        traces[16][row] = mult;
    }
}

// Host function to generate verify_instruction traces
extern "C"
void generate_verify_instruction_trace(
    m31 **traces,
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_4_3_0,
    m31 **lookup_verify_instruction_0,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_range_check_7_2_5,
    m31 **sub_component_inputs_range_check_4_3,
    m31 **verify_instruction_inputs,
    m31 *multiplicities,
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned n_rows,
    unsigned log_size
) {
    unsigned trace_size = 1u << log_size;
    timer global_timer;

    // Copy traces pointer array to device
    m31 **device_traces = clone_to_device<m31*>(traces, N_VERIFY_INSTRUCTION_TRACE_COLUMNS);

    // Copy lookup data arrays to device
    m31 **device_lookup_addr_to_id = clone_to_device<m31*>(lookup_memory_address_to_id_0, 2);
    m31 **device_lookup_id_to_big = clone_to_device<m31*>(lookup_memory_id_to_big_0, 29);
    m31 **device_lookup_rc_7_2_5 = clone_to_device<m31*>(lookup_range_check_7_2_5_0, 3);
    m31 **device_lookup_rc_4_3 = clone_to_device<m31*>(lookup_range_check_4_3_0, 2);
    m31 **device_lookup_verify_instr = clone_to_device<m31*>(lookup_verify_instruction_0, 7);

    // Copy sub-component input arrays to device
    m31 **device_sub_addr_to_id = clone_to_device<m31*>(sub_component_inputs_memory_address_to_id, 1);
    m31 **device_sub_id_to_big = clone_to_device<m31*>(sub_component_inputs_memory_id_to_big, 1);
    m31 **device_sub_rc_7_2_5 = clone_to_device<m31*>(sub_component_inputs_range_check_7_2_5, 3);
    m31 **device_sub_rc_4_3 = clone_to_device<m31*>(sub_component_inputs_range_check_4_3, 2);

    // Copy input arrays to device
    m31 **device_inputs = clone_to_device<m31*>(verify_instruction_inputs, 7);

    // Generate main trace
    // Use 256 threads to match __launch_bounds__(256, 2)
    const int VERIFY_INSTR_TRACE_THREAD_MAX = 256;
    int block_dim = trace_size < VERIFY_INSTR_TRACE_THREAD_MAX ? trace_size : VERIFY_INSTR_TRACE_THREAD_MAX;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    global_timer.start("generate verify_instruction base trace");
    generate_verify_instruction_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,
        device_lookup_addr_to_id,
        device_lookup_id_to_big,
        device_lookup_rc_7_2_5,
        device_lookup_rc_4_3,
        device_lookup_verify_instr,
        device_sub_addr_to_id,
        device_sub_id_to_big,
        device_sub_rc_7_2_5,
        device_sub_rc_4_3,
        device_inputs,
        multiplicities,
        memory_address_to_id_address_to_raw_id,
        n_rows,
        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    global_timer.start("generate verify_instruction base trace");

    // Free device memory
    cuda_free_memory(device_traces);
    cuda_free_memory(device_lookup_addr_to_id);
    cuda_free_memory(device_lookup_id_to_big);
    cuda_free_memory(device_lookup_rc_7_2_5);
    cuda_free_memory(device_lookup_rc_4_3);
    cuda_free_memory(device_lookup_verify_instr);
    cuda_free_memory(device_sub_addr_to_id);
    cuda_free_memory(device_sub_id_to_big);
    cuda_free_memory(device_sub_rc_7_2_5);
    cuda_free_memory(device_sub_rc_4_3);
    cuda_free_memory(device_inputs);
}

// ============================================================================
// Full CUDA interaction trace generation with proper logup accumulation
// ============================================================================

#define N_VI_LOGUP_COLS 3       // 3 logup columns (2 paired + 1 final)
#define N_VI_INTERACTION_COLS 12 // 3 logup × 4 BaseField
// Use smaller thread count for interaction kernel due to large local variables (29-element array)
#define VI_INTERACTION_THREAD_COUNT 256

// Compute 3 logup fractions per row:
//   Column 0: paired (range_check_7_2_5 + range_check_4_3)
//   Column 1: paired (memory_address_to_id + memory_id_to_big)
//   Column 2: final (verify_instruction with -mult)
__global__ void verify_instruction_generate_interaction_frac_kernel(
    m31 **lookup_rc_7_2_5,       // [3][trace_size]
    m31 **lookup_rc_4_3,         // [2][trace_size]
    m31 **lookup_addr_to_id,     // [2][trace_size]
    m31 **lookup_id_to_big,      // [29][trace_size]
    m31 **lookup_verify_instr,   // [7][trace_size]
    m31 *multiplicities,         // [trace_size]
    RangeCheck_7_2_5 *rc_7_2_5_lookup,
    RangeCheck_4_3 *rc_4_3_lookup,
    MemoryAddressToId *addr_to_id_lookup,
    MemoryIdToBig *id_to_big_lookup,
    VerifyInstruction *verify_instr_lookup,
    unsigned trace_size,
    qm31 *denom_ptr,             // [3 * trace_size] - 3 denominators per row
    m31 *numerator0,             // [3 * trace_size]
    m31 *numerator1,             // [3 * trace_size]
    m31 *numerator2,             // [3 * trace_size]
    m31 *numerator3              // [3 * trace_size]
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < trace_size) {
        // ==== Column 0: paired (range_check_7_2_5 + range_check_4_3) ====
        // Numerator: denom0 + denom1, Denominator: denom0 * denom1
        {
            m31 rc_7_2_5_inputs[3] = {
                lookup_rc_7_2_5[0][row],
                lookup_rc_7_2_5[1][row],
                lookup_rc_7_2_5[2][row]
            };
            qm31 denom0 = rc_7_2_5_lookup->combine(rc_7_2_5_inputs, 3);

            m31 rc_4_3_inputs[2] = {
                lookup_rc_4_3[0][row],
                lookup_rc_4_3[1][row]
            };
            qm31 denom1 = rc_4_3_lookup->combine(rc_4_3_inputs, 2);

            qm31 numer = add(denom0, denom1);  // +1 for each lookup
            qm31 denom = mul(denom0, denom1);

            unsigned idx = 0 * trace_size + row;
            logup_col_write_frac(idx, numer, denom,
                                denom_ptr, numerator0, numerator1, numerator2, numerator3);
        }

        // ==== Column 1: paired (memory_address_to_id + memory_id_to_big) ====
        {
            m31 addr_to_id_inputs[2] = {
                lookup_addr_to_id[0][row],
                lookup_addr_to_id[1][row]
            };
            qm31 denom0 = addr_to_id_lookup->combine(addr_to_id_inputs, 2);

            m31 id_to_big_inputs[29];
            for (int i = 0; i < 29; i++) {
                id_to_big_inputs[i] = lookup_id_to_big[i][row];
            }
            qm31 denom1 = id_to_big_lookup->combine(id_to_big_inputs, 29);

            qm31 numer = add(denom0, denom1);  // +1 for each lookup
            qm31 denom = mul(denom0, denom1);

            unsigned idx = 1 * trace_size + row;
            logup_col_write_frac(idx, numer, denom,
                                denom_ptr, numerator0, numerator1, numerator2, numerator3);
        }

        // ==== Column 2: final (verify_instruction with -mult) ====
        {
            m31 verify_instr_inputs[7];
            for (int i = 0; i < 7; i++) {
                verify_instr_inputs[i] = lookup_verify_instr[i][row];
            }
            qm31 denom = verify_instr_lookup->combine(verify_instr_inputs, 7);

            // Numerator: -mult (providing values)
            m31 mult = multiplicities[row];
            qm31 zero = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};
            qm31 qm31_mult = qm31{cm31{mult, m31{0}}, cm31{m31{0}, m31{0}}};
            qm31 numer = sub(zero, qm31_mult);

            unsigned idx = 2 * trace_size + row;
            logup_col_write_frac(idx, numer, denom,
                                denom_ptr, numerator0, numerator1, numerator2, numerator3);
        }
    }
}

// Finalize interaction columns - write 3 logup columns WITH ACCUMULATION (12 BaseField columns)
// Column set N contains the running sum of all fractions from 0 to N
// This matches the memory_address_to_id pattern
__global__ void verify_instruction_finalize_interaction_col_kernel(
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
        // Running sum of all fractions (cross-column accumulation)
        qm31 running_sum = qm31{cm31{m31{0}, m31{0}}, cm31{m31{0}, m31{0}}};

        for (int i = 0; i < N_VI_LOGUP_COLS; ++i) {
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

            // Write the accumulated sum (not just the fraction)
            int base_col = i * 4;
            interaction_traces[base_col + 0][row] = running_sum.a.a;
            interaction_traces[base_col + 1][row] = running_sum.a.b;
            interaction_traces[base_col + 2][row] = running_sum.b.a;
            interaction_traces[base_col + 3][row] = running_sum.b.b;
        }
    }
}

// Compute cumulative sum for claimed_sum (parallel reduction on LAST column only)
// With cross-column accumulation, the last column contains the sum of ALL fractions
__global__ void verify_instruction_cumsum_kernel(
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

    // Only sum the LAST column (8-11) which contains accumulated total
    int last_base_col = (N_VI_LOGUP_COLS - 1) * 4;
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

// Apply cumsum shift to last column (columns 8-11)
// Shift is claimed_sum / trace_size (matching memory_address_to_id pattern)
__global__ void verify_instruction_apply_cumsum_shift_kernel(
    m31 *coordinate_sums,              // [4] - cumulative sum
    unsigned trace_size,
    m31 **interaction_traces           // [12][trace_size]
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
        int last_base_col = (N_VI_LOGUP_COLS - 1) * 4;
        interaction_traces[last_base_col + 0][row] = sub(interaction_traces[last_base_col + 0][row], cumsum_shift.a.a);
        interaction_traces[last_base_col + 1][row] = sub(interaction_traces[last_base_col + 1][row], cumsum_shift.a.b);
        interaction_traces[last_base_col + 2][row] = sub(interaction_traces[last_base_col + 2][row], cumsum_shift.b.a);
        interaction_traces[last_base_col + 3][row] = sub(interaction_traces[last_base_col + 3][row], cumsum_shift.b.b);
    }
}

// Host function for full CUDA interaction trace generation
extern "C"
void generate_verify_instruction_interaction_trace(
    m31 **interaction_traces,          // [12][trace_size] - output
    m31 **lookup_rc_7_2_5,             // [3][trace_size]
    m31 **lookup_rc_4_3,               // [2][trace_size]
    m31 **lookup_addr_to_id,           // [2][trace_size]
    m31 **lookup_id_to_big,            // [29][trace_size]
    m31 **lookup_verify_instr,         // [7][trace_size]
    m31 *multiplicities,               // [trace_size]
    RangeCheck_7_2_5 *rc_7_2_5_lookup,
    RangeCheck_4_3 *rc_4_3_lookup,
    MemoryAddressToId *addr_to_id_lookup,
    MemoryIdToBig *id_to_big_lookup,
    VerifyInstruction *verify_instr_lookup,
    unsigned log_size,
    m31 *claimed_sum                   // [4] - output
) {
    unsigned trace_size = 1u << log_size;
    timer global_timer;

    // Copy pointer arrays to device
    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, N_VI_INTERACTION_COLS);
    m31 **device_lookup_rc_7_2_5 = clone_to_device<m31*>(lookup_rc_7_2_5, 3);
    m31 **device_lookup_rc_4_3 = clone_to_device<m31*>(lookup_rc_4_3, 2);
    m31 **device_lookup_addr_to_id = clone_to_device<m31*>(lookup_addr_to_id, 2);
    m31 **device_lookup_id_to_big = clone_to_device<m31*>(lookup_id_to_big, 29);
    m31 **device_lookup_verify_instr = clone_to_device<m31*>(lookup_verify_instr, 7);

    // Copy lookup elements to device
    RangeCheck_7_2_5 *device_rc_7_2_5_lookup = clone_to_device<RangeCheck_7_2_5>(rc_7_2_5_lookup, 1);
    RangeCheck_4_3 *device_rc_4_3_lookup = clone_to_device<RangeCheck_4_3>(rc_4_3_lookup, 1);
    MemoryAddressToId *device_addr_to_id_lookup = clone_to_device<MemoryAddressToId>(addr_to_id_lookup, 1);
    MemoryIdToBig *device_id_to_big_lookup = clone_to_device<MemoryIdToBig>(id_to_big_lookup, 1);
    VerifyInstruction *device_verify_instr_lookup = clone_to_device<VerifyInstruction>(verify_instr_lookup, 1);

    // Allocate temporary buffers for 3 fractions
    size_t total_fracs = N_VI_LOGUP_COLS * trace_size;
    qm31 *device_logup_denom = cuda_malloc<qm31>(total_fracs);
    m31 *device_numerator0 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator1 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator2 = cuda_malloc<m31>(total_fracs);
    m31 *device_numerator3 = cuda_malloc<m31>(total_fracs);
    qm31 *denom_inv = cuda_malloc<qm31>(total_fracs);

    // Use smaller thread count for frac kernel due to large local variables
    int frac_block_dim = trace_size < VI_INTERACTION_THREAD_COUNT ? trace_size : VI_INTERACTION_THREAD_COUNT;
    int frac_num_blocks = (trace_size + frac_block_dim - 1) / frac_block_dim;

    global_timer.start("generate verify_instruction interaction trace");
    // Step 1: Compute all 3 logup fractions per row
    verify_instruction_generate_interaction_frac_kernel<<<frac_num_blocks, frac_block_dim>>>(
        device_lookup_rc_7_2_5,
        device_lookup_rc_4_3,
        device_lookup_addr_to_id,
        device_lookup_id_to_big,
        device_lookup_verify_instr,
        multiplicities,
        device_rc_7_2_5_lookup,
        device_rc_4_3_lookup,
        device_addr_to_id_lookup,
        device_id_to_big_lookup,
        device_verify_instr_lookup,
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

    // Use standard thread count for lighter-weight kernels
    int block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    int num_blocks = (trace_size + block_dim - 1) / block_dim;

    // Step 3: Finalize interaction columns with accumulation
    verify_instruction_finalize_interaction_col_kernel<<<num_blocks, block_dim>>>(
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 4: Compute claimed_sum via parallel reduction
    size_t shared_size = 4 * block_dim * sizeof(m31);
    verify_instruction_cumsum_kernel<<<num_blocks, block_dim, shared_size>>>(
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 5: Apply cumsum shift to last column (8-11)
    verify_instruction_apply_cumsum_shift_kernel<<<num_blocks, block_dim>>>(
        claimed_sum,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // Step 6: Apply inclusive prefix sum ONLY to the LAST column (8-11)
    int last_base_col = (N_VI_LOGUP_COLS - 1) * 4;
    inclusive_prefix_sum(interaction_traces[last_base_col + 0], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 1], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 2], trace_size);
    inclusive_prefix_sum(interaction_traces[last_base_col + 3], trace_size);

    // Cleanup
    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_lookup_rc_7_2_5);
    cuda_free_memory(device_lookup_rc_4_3);
    cuda_free_memory(device_lookup_addr_to_id);
    cuda_free_memory(device_lookup_id_to_big);
    cuda_free_memory(device_lookup_verify_instr);
    cuda_free_memory(device_rc_7_2_5_lookup);
    cuda_free_memory(device_rc_4_3_lookup);
    cuda_free_memory(device_addr_to_id_lookup);
    cuda_free_memory(device_id_to_big_lookup);
    cuda_free_memory(device_verify_instr_lookup);
    cuda_free_memory(device_logup_denom);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
    global_timer.end("generate verify_instruction interaction trace");
}
