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

#include "gen_triple_xor_32_trace.cuh"
#define TRIPLE_XOR_32_N_TRACE_COLUMNS 21

#define GEN_TRACE_TRIPLE_XOR_32_THREAD_COUNT_MAX 256

#define TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS 5

__launch_bounds__(256, 2)
__global__ void generate_triple_xor_32_trace_kernel(
    m31 **traces,
    m31 **lookup_triple_xor_32_0,

    m31 **lookup_verify_bitwise_xor_8_0,
    m31 **lookup_verify_bitwise_xor_8_1,
    m31 **lookup_verify_bitwise_xor_8_2,
    m31 **lookup_verify_bitwise_xor_8_3,
    m31 **lookup_verify_bitwise_xor_8_b_0,
    m31 **lookup_verify_bitwise_xor_8_b_1,
    m31 **lookup_verify_bitwise_xor_8_b_2,
    m31 **lookup_verify_bitwise_xor_8_b_3,

    m31 **sub_component_inputs_verify_bitwise_xor_8,
    m31 **sub_component_inputs_verify_bitwise_xor_8_b,

    uint32_t **triple_xor_32_input,

    unsigned trace_size
) {
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    const m31 M31_256 = {256};

    const uint16_t UInt16_8 = 8;

    if (row < trace_size) {
        m31 input_limb_0_col0 = low_as_m31(triple_xor_32_input[0][row]);
        traces[0][row] = input_limb_0_col0;
        m31 input_limb_1_col1 = high_as_m31(triple_xor_32_input[0][row]);
        traces[1][row] = input_limb_1_col1;
        m31 input_limb_2_col2 = low_as_m31(triple_xor_32_input[1][row]);
        traces[2][row] = input_limb_2_col2;
        m31 input_limb_3_col3 = high_as_m31(triple_xor_32_input[1][row]);
        traces[3][row] = input_limb_3_col3;
        m31 input_limb_4_col4 = low_as_m31(triple_xor_32_input[2][row]);
        traces[4][row] = input_limb_4_col4;
        m31 input_limb_5_col5 = high_as_m31(triple_xor_32_input[2][row]);
        traces[5][row] = input_limb_5_col5;
        Enabler enabler_col = Enabler(trace_size);

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_298db_0 = ((triple_xor_32_input[0][row] & 0xFFFF) >> (UInt16_8));
        m31 ms_8_bits_col6 = m31 {ms_8_bits_tmp_298db_0};
        traces[6][row] = ms_8_bits_col6;
        m31 split_16_low_part_size_8_output_tmp_298db_1_0 = sub((input_limb_0_col0), mul((ms_8_bits_col6), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_298db_1_1 = ms_8_bits_col6;


        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_298db_2 = ((triple_xor_32_input[0][row] >> 16) >> (UInt16_8));
        m31 ms_8_bits_col7 = m31 {ms_8_bits_tmp_298db_2};
        traces[7][row] = ms_8_bits_col7;
        m31 split_16_low_part_size_8_output_tmp_298db_3_0 = sub((input_limb_1_col1), mul((ms_8_bits_col7), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_298db_3_1 = ms_8_bits_col7;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_298db_4 = ((triple_xor_32_input[1][row] & 0xFFFF) >> (UInt16_8));
        m31 ms_8_bits_col8 = m31 {ms_8_bits_tmp_298db_4};
        traces[8][row] = ms_8_bits_col8;
        m31 split_16_low_part_size_8_output_tmp_298db_5_0 = sub((input_limb_2_col2), mul((ms_8_bits_col8), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_298db_5_1 = ms_8_bits_col8;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_298db_6 = ((triple_xor_32_input[1][row] >> 16) >> (UInt16_8));
        m31 ms_8_bits_col9 = m31 {ms_8_bits_tmp_298db_6};
        traces[9][row] = ms_8_bits_col9;
        m31 split_16_low_part_size_8_output_tmp_298db_7_0 = sub((input_limb_3_col3), mul((ms_8_bits_col9), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_298db_7_1 = ms_8_bits_col9;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_298db_8 = ((triple_xor_32_input[2][row] & 0xFFFF) >> (UInt16_8));
        m31 ms_8_bits_col10 = m31 {ms_8_bits_tmp_298db_8};
        traces[10][row] = ms_8_bits_col10;
        m31 split_16_low_part_size_8_output_tmp_298db_9_0 = sub((input_limb_4_col4), mul((ms_8_bits_col10), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_298db_9_1 = ms_8_bits_col10;

        // Split 16 Low Part Size 8.

        uint16_t ms_8_bits_tmp_298db_10 = ((triple_xor_32_input[2][row] >> 16) >> (UInt16_8));
        m31 ms_8_bits_col11 = m31 {ms_8_bits_tmp_298db_10};
        traces[11][row] = ms_8_bits_col11;
        m31 split_16_low_part_size_8_output_tmp_298db_11_0 = sub((input_limb_5_col5), mul((ms_8_bits_col11), (M31_256)));
        m31 split_16_low_part_size_8_output_tmp_298db_11_1 = ms_8_bits_col11;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_298db_12 = split_16_low_part_size_8_output_tmp_298db_1_0 ^ split_16_low_part_size_8_output_tmp_298db_5_0;
        m31 xor_col12 = m31 {xor_tmp_298db_12};
        traces[12][row] = xor_col12;
        sub_component_inputs_verify_bitwise_xor_8[0 * 3 + 0][row] = split_16_low_part_size_8_output_tmp_298db_1_0;
        sub_component_inputs_verify_bitwise_xor_8[0 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_298db_5_0;
        sub_component_inputs_verify_bitwise_xor_8[0 * 3 + 2][row] = xor_col12;
        lookup_verify_bitwise_xor_8_0[0][row] = split_16_low_part_size_8_output_tmp_298db_1_0;
        lookup_verify_bitwise_xor_8_0[1][row] = split_16_low_part_size_8_output_tmp_298db_5_0;
        lookup_verify_bitwise_xor_8_0[2][row] = xor_col12;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_298db_14 = xor_col12 ^ split_16_low_part_size_8_output_tmp_298db_9_0;
        m31 xor_col13 = m31 {xor_tmp_298db_14};
        traces[13][row] = xor_col13;
        sub_component_inputs_verify_bitwise_xor_8[1 * 3 + 0][row] = xor_col12;
        sub_component_inputs_verify_bitwise_xor_8[1 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_298db_9_0;
        sub_component_inputs_verify_bitwise_xor_8[1 * 3 + 2][row] = xor_col13;
        lookup_verify_bitwise_xor_8_1[0][row] = xor_col12;
        lookup_verify_bitwise_xor_8_1[1][row] = split_16_low_part_size_8_output_tmp_298db_9_0;
        lookup_verify_bitwise_xor_8_1[2][row] = xor_col13;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_298db_16 = ms_8_bits_col6 ^ ms_8_bits_col8;
        m31 xor_col14 = m31 {xor_tmp_298db_16};
        traces[14][row] = xor_col14;
        sub_component_inputs_verify_bitwise_xor_8[2 * 3 + 0][row] = ms_8_bits_col6;
        sub_component_inputs_verify_bitwise_xor_8[2 * 3 + 1][row] = ms_8_bits_col8;
        sub_component_inputs_verify_bitwise_xor_8[2 * 3 + 2][row] = xor_col14;
        lookup_verify_bitwise_xor_8_2[0][row] = ms_8_bits_col6;
        lookup_verify_bitwise_xor_8_2[1][row] = ms_8_bits_col8;
        lookup_verify_bitwise_xor_8_2[2][row] = xor_col14;

        // Bitwise Xor Num Bits 8.

        uint16_t xor_tmp_298db_18 = xor_col14 ^ ms_8_bits_col10;
        m31 xor_col15 = m31 {xor_tmp_298db_18};
        traces[15][row] = xor_col15;
        sub_component_inputs_verify_bitwise_xor_8[3 * 3 + 0][row] = xor_col14;
        sub_component_inputs_verify_bitwise_xor_8[3 * 3 + 1][row] = ms_8_bits_col10;
        sub_component_inputs_verify_bitwise_xor_8[3 * 3 + 2][row] = xor_col15;
        lookup_verify_bitwise_xor_8_3[0][row] = xor_col14;
        lookup_verify_bitwise_xor_8_3[1][row] = ms_8_bits_col10;
        lookup_verify_bitwise_xor_8_3[2][row] = xor_col15;

        // Bitwise Xor Num Bits 8 B.

        uint16_t xor_tmp_298db_20 = split_16_low_part_size_8_output_tmp_298db_3_0 ^ split_16_low_part_size_8_output_tmp_298db_7_0;
        m31 xor_col16 = m31 {xor_tmp_298db_20};
        traces[16][row] = xor_col16;
        sub_component_inputs_verify_bitwise_xor_8_b[0 * 3 + 0][row] = split_16_low_part_size_8_output_tmp_298db_3_0;
        sub_component_inputs_verify_bitwise_xor_8_b[0 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_298db_7_0;
        sub_component_inputs_verify_bitwise_xor_8_b[0 * 3 + 2][row] = xor_col16;
        lookup_verify_bitwise_xor_8_b_0[0][row] = split_16_low_part_size_8_output_tmp_298db_3_0;
        lookup_verify_bitwise_xor_8_b_0[1][row] = split_16_low_part_size_8_output_tmp_298db_7_0;
        lookup_verify_bitwise_xor_8_b_0[2][row] = xor_col16;

        // Bitwise Xor Num Bits 8 B.

        uint16_t xor_tmp_298db_22 = xor_col16 ^ split_16_low_part_size_8_output_tmp_298db_11_0;
        m31 xor_col17 = m31 {xor_tmp_298db_22};
        traces[17][row] = xor_col17;
        sub_component_inputs_verify_bitwise_xor_8_b[1 * 3 + 0][row] = xor_col16;
        sub_component_inputs_verify_bitwise_xor_8_b[1 * 3 + 1][row] = split_16_low_part_size_8_output_tmp_298db_11_0;
        sub_component_inputs_verify_bitwise_xor_8_b[1 * 3 + 2][row] = xor_col17;
        lookup_verify_bitwise_xor_8_b_1[0][row] = xor_col16;
        lookup_verify_bitwise_xor_8_b_1[1][row] = split_16_low_part_size_8_output_tmp_298db_11_0;
        lookup_verify_bitwise_xor_8_b_1[2][row] = xor_col17;

        // Bitwise Xor Num Bits 8 B.

        uint16_t xor_tmp_298db_24 = ms_8_bits_col7 ^ ms_8_bits_col9;
        m31 xor_col18 = m31 {xor_tmp_298db_24};
        traces[18][row] = xor_col18;
        sub_component_inputs_verify_bitwise_xor_8_b[2 * 3 + 0][row] = ms_8_bits_col7;
        sub_component_inputs_verify_bitwise_xor_8_b[2 * 3 + 1][row] = ms_8_bits_col9;
        sub_component_inputs_verify_bitwise_xor_8_b[2 * 3 + 2][row] = xor_col18;
        lookup_verify_bitwise_xor_8_b_2[0][row] = ms_8_bits_col7;
        lookup_verify_bitwise_xor_8_b_2[1][row] = ms_8_bits_col9;
        lookup_verify_bitwise_xor_8_b_2[2][row] = xor_col18;

        // Bitwise Xor Num Bits 8 B.

        uint16_t xor_tmp_298db_26 = xor_col18 ^ ms_8_bits_col11;
        m31 xor_col19 = m31 {xor_tmp_298db_26};
        traces[19][row] = xor_col19;
        sub_component_inputs_verify_bitwise_xor_8_b[3 * 3 + 0][row] = xor_col18;
        sub_component_inputs_verify_bitwise_xor_8_b[3 * 3 + 1][row] = ms_8_bits_col11;
        sub_component_inputs_verify_bitwise_xor_8_b[3 * 3 + 2][row] = xor_col19;
        lookup_verify_bitwise_xor_8_b_3[0][row] = xor_col18;
        lookup_verify_bitwise_xor_8_b_3[1][row] = ms_8_bits_col11;
        lookup_verify_bitwise_xor_8_b_3[2][row] = xor_col19;

        uint32_t triple_xor32_output_tmp_298db_28 =  add((xor_col13), mul((xor_col15), (M31_256))) + (add((xor_col17), mul((xor_col19), (M31_256))) << 16);

        lookup_triple_xor_32_0[0][row] = input_limb_0_col0;
        lookup_triple_xor_32_0[1][row] = input_limb_1_col1;
        lookup_triple_xor_32_0[2][row] = input_limb_2_col2;
        lookup_triple_xor_32_0[3][row] = input_limb_3_col3;
        lookup_triple_xor_32_0[4][row] = input_limb_4_col4;
        lookup_triple_xor_32_0[5][row] = input_limb_5_col5;
        lookup_triple_xor_32_0[6][row] = low_as_m31(triple_xor32_output_tmp_298db_28);
        lookup_triple_xor_32_0[7][row] = high_as_m31(triple_xor32_output_tmp_298db_28);
        traces[20][row] = 1;
    }
}


void generate_triple_xor_32_traces(
    m31 **traces,
    m31 **lookup_triple_xor_32_0,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_b_0 ,
    m31 **lookup_verify_bitwise_xor_8_b_1 ,
    m31 **lookup_verify_bitwise_xor_8_b_2 ,
    m31 **lookup_verify_bitwise_xor_8_b_3 ,

    m31 **sub_component_inputs_verify_bitwise_xor_8,
    m31 **sub_component_inputs_verify_bitwise_xor_8_b,

    uint32_t **triple_xor_32_input,

    unsigned trace_log_size
) {
    m31 **device_traces = clone_to_device<m31*>(traces, TRIPLE_XOR_32_N_TRACE_COLUMNS);
    m31 **device_lookup_triple_xor_32_0 = clone_to_device<m31 *>(lookup_triple_xor_32_0, 8);
    m31 **device_lookup_verify_bitwise_xor_8_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_2  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_2 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_3  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_3 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_2  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_2 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_3  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_3 , 3);

    m31 **device_sub_component_inputs_verify_bitwise_xor_8  = clone_to_device<m31 *>(sub_component_inputs_verify_bitwise_xor_8 , 3 * 4);
    m31 **device_sub_component_inputs_verify_bitwise_xor_8_b  = clone_to_device<m31 *>(sub_component_inputs_verify_bitwise_xor_8_b , 3 * 4);

    uint32_t **device_triple_xor_32_input = clone_to_device<uint32_t *>(triple_xor_32_input, 3);


    timer global_timer;
    global_timer.start("generate triple_xor_32 base trace");

    unsigned trace_size = 1 << trace_log_size;
    int block_dim = trace_size < GEN_TRACE_TRIPLE_XOR_32_THREAD_COUNT_MAX ? trace_size : GEN_TRACE_TRIPLE_XOR_32_THREAD_COUNT_MAX;
    int num_blocks = block_dim < GEN_TRACE_TRIPLE_XOR_32_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    generate_triple_xor_32_trace_kernel<<<num_blocks, block_dim>>>(
        device_traces,

        device_lookup_triple_xor_32_0,

        device_lookup_verify_bitwise_xor_8_0,
        device_lookup_verify_bitwise_xor_8_1,
        device_lookup_verify_bitwise_xor_8_2,
        device_lookup_verify_bitwise_xor_8_3,
        device_lookup_verify_bitwise_xor_8_b_0,
        device_lookup_verify_bitwise_xor_8_b_1,
        device_lookup_verify_bitwise_xor_8_b_2,
        device_lookup_verify_bitwise_xor_8_b_3,

        device_sub_component_inputs_verify_bitwise_xor_8,
        device_sub_component_inputs_verify_bitwise_xor_8_b,

        device_triple_xor_32_input,

        trace_size
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    global_timer.end("generate triple_xor_32 base trace");

    cuda_free_memory(device_triple_xor_32_input);

    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_8_b  );
    cuda_free_memory(device_sub_component_inputs_verify_bitwise_xor_8  );

    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_3);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_2);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_1);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_0);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_3);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_2  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_1  );
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_0  );
    cuda_free_memory(device_lookup_triple_xor_32_0);

    cuda_free_memory(device_traces);
}

template <int N, int M>
__launch_bounds__(256, 2)
__global__ void generate_triple_xor_32_interaction_trace_col_gen_kernel(
    LookupElementsBasic<N>  *lookup_elements_n,
    LookupElementsBasic<M>  *lookup_elements_m,
    m31 **lookup_state_0,
    m31 **lookup_state_1,
    unsigned trace_size,
    qm31 *denom_ptr,
    m31 *numerator0,
    m31 *numerator1,
    m31 *numerator2,
    m31 *numerator3
) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    m31 init_combine_reg[N] = {};
    m31 final_combine_reg[M] = {};

    for (int i = 0; i < N; i++) {
        init_combine_reg[i] = lookup_state_0[i][vec_index];
    }
    for (int i = 0; i < M; i++) {
        final_combine_reg[i] = lookup_state_1[i][vec_index];
    }
    if (vec_index < trace_size) {
        qm31 denom0 = lookup_elements_n->combine(init_combine_reg, N);
        qm31 denom1 = lookup_elements_m->combine(final_combine_reg, M);
        logup_col_write_frac(vec_index, add(denom1, denom0), mul(denom0, denom1),
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}

template <int N>
__launch_bounds__(256, 2)
__global__ void generate_triple_xor_32_interaction_trace_col_single_gen_kernel(
    LookupElementsBasic<N>  *lookup_elements_n,
    m31 **lookup_state_0,
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
       logup_col_write_frac(vec_index, qm31{P-1, 0, 0, 0}, denom,
                            denom_ptr, numerator0, numerator1, numerator2, numerator3);
    }
}


__global__ void generate_triple_xor_32_interaction_trace_finalize_col_kernel(
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

__global__ void generate_triple_xor_32_interaction_trace_cumsum_shift(
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

__global__ void generate_triple_xor_32_interaction_trace_coord_prefix_sum(
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

void generate_triple_xor_32_interaction_traces(
    void *triple_xor_32,
    void *verify_bitwise_xor_8,
    void *verify_bitwise_xor_8_b,

    m31 **lookup_triple_xor_32_0,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_b_0 ,
    m31 **lookup_verify_bitwise_xor_8_b_1 ,
    m31 **lookup_verify_bitwise_xor_8_b_2 ,
    m31 **lookup_verify_bitwise_xor_8_b_3 ,

    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
) {
    unsigned trace_size = 1 << log_size;

    TripleXor32 *triple_xor_32_lookup_elements = (TripleXor32 *)triple_xor_32;
    VerifyBitwiseXor_8  *verify_bitwise_xor_8_lookup_elements  = (VerifyBitwiseXor_8 *)verify_bitwise_xor_8;
    VerifyBitwiseXor_8_B  *verify_bitwise_xor_8_b_lookup_elements  = (VerifyBitwiseXor_8_B *)verify_bitwise_xor_8_b;

    TripleXor32 *device_triple_xor_32_lookup_elements = cuda_malloc<TripleXor32>(1);
    VerifyBitwiseXor_8 *device_verify_bitwise_xor_8_lookup_elements = cuda_malloc<VerifyBitwiseXor_8>(1);
    VerifyBitwiseXor_8_B *device_verify_bitwise_xor_8_b_lookup_elements = cuda_malloc<VerifyBitwiseXor_8_B>(1);

    cuda_mem_copy_host_to_device<TripleXor32>(triple_xor_32_lookup_elements, device_triple_xor_32_lookup_elements, 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_8> (verify_bitwise_xor_8_lookup_elements , device_verify_bitwise_xor_8_lookup_elements , 1);
    cuda_mem_copy_host_to_device<VerifyBitwiseXor_8_B> (verify_bitwise_xor_8_b_lookup_elements , device_verify_bitwise_xor_8_b_lookup_elements , 1);

    qm31 *device_logup_denom = cuda_malloc<qm31>(trace_size);

    m31 **device_lookup_triple_xor_32_0 = clone_to_device<m31 *>(lookup_triple_xor_32_0, 8);
    m31 **device_lookup_verify_bitwise_xor_8_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_2  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_2 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_3  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_3 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_0  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_0 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_1  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_1 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_2  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_2 , 3);
    m31 **device_lookup_verify_bitwise_xor_8_b_3  = clone_to_device<m31 *>(lookup_verify_bitwise_xor_8_b_3 , 3);

    m31 **device_interaction_traces = clone_to_device<m31*>(interaction_traces, 4 * TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS);

    m31 *device_numerator0 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator1 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator2 = cuda_malloc<m31>(trace_size);
    m31 *device_numerator3 = cuda_malloc<m31>(trace_size);
    qm31 * denom_inv = cuda_malloc<qm31>(trace_size);

    // dump_lookup_data(lookup_verify_bitwise_xor_8_0, 3, trace_size);
    // dump_lookup_data(lookup_verify_bitwise_xor_8_1, 3, trace_size);

    timer global_timer;
    global_timer.start("generate triple_xor_32 interaction trace");

    int block_dim = 0;
    int num_blocks = 0;

    block_dim = trace_size < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;

    // #0 Interaction trace For verify_bitwise_xor_8_0 & verify_bitwise_xor_8_1
    generate_triple_xor_32_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_lookup_elements,
        device_verify_bitwise_xor_8_lookup_elements,

        device_lookup_verify_bitwise_xor_8_0,
        device_lookup_verify_bitwise_xor_8_1,

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
    generate_triple_xor_32_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
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

    // #1 Interaction trace For verify_bitwise_xor_8_2 & verify_bitwise_xor_8_3
    block_dim = trace_size < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_triple_xor_32_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_lookup_elements,
        device_verify_bitwise_xor_8_lookup_elements,

        device_lookup_verify_bitwise_xor_8_2,
        device_lookup_verify_bitwise_xor_8_3,

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
    generate_triple_xor_32_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        1,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_traces, 1, trace_size);

    // #2 Interaction trace For verify_bitwise_xor_8_b_0 & verify_bitwise_xor_8_b_1
    block_dim = trace_size < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_triple_xor_32_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_b_lookup_elements,
        device_verify_bitwise_xor_8_b_lookup_elements,

        device_lookup_verify_bitwise_xor_8_b_0,
        device_lookup_verify_bitwise_xor_8_b_1,

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
    generate_triple_xor_32_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        2,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_traces, 2, trace_size);
    // #3 Interaction trace For verify_bitwise_xor_8_b_2 & verify_bitwise_xor_8_b_3
    block_dim = trace_size < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_triple_xor_32_interaction_trace_col_gen_kernel<3, 3><<<num_blocks, block_dim>>>(
        device_verify_bitwise_xor_8_b_lookup_elements,
        device_verify_bitwise_xor_8_b_lookup_elements,

        device_lookup_verify_bitwise_xor_8_b_2,
        device_lookup_verify_bitwise_xor_8_b_3,

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
    generate_triple_xor_32_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        3,
        trace_size,
        denom_inv,
        device_numerator0,
        device_numerator1,
        device_numerator2,
        device_numerator3,
        device_interaction_traces
    );

    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    // dump_interaction_traces(interaction_traces, 3, trace_size);
    // #4 Interaction trace For lookup_triple_xor_32_0
    block_dim = trace_size < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? trace_size : TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX;
    num_blocks = block_dim < TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_triple_xor_32_interaction_trace_col_single_gen_kernel<8><<<num_blocks, block_dim>>>(
        device_triple_xor_32_lookup_elements,

        device_lookup_triple_xor_32_0,

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
    generate_triple_xor_32_interaction_trace_finalize_col_kernel<<<num_blocks, block_dim>>>(
        4,
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
    generate_triple_xor_32_interaction_trace_cumsum_shift<<<num_blocks, block_dim, shared_size>>>(
        TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces,
        claimed_sum
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    block_dim = trace_size < THREAD_COUNT_MAX ? trace_size : THREAD_COUNT_MAX;
    num_blocks = block_dim < THREAD_COUNT_MAX ? 1 : (trace_size + block_dim - 1) / block_dim;
    generate_triple_xor_32_interaction_trace_coord_prefix_sum<<<num_blocks, block_dim>>>(
        claimed_sum,
        TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS,
        trace_size,
        device_interaction_traces
    );
    ASSERT_CUDA_SUCCESS(cudaGetLastError());

    inclusive_prefix_sum(interaction_traces[4 * TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS - 4], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS - 3], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS - 2], trace_size);
    inclusive_prefix_sum(interaction_traces[4 * TRIPLE_XOR_32_N_INTERACTION_TRACE_COLUMNS - 1], trace_size);

    global_timer.end("generate triple_xor_32 interaction trace");

    cuda_free_memory(device_verify_bitwise_xor_8_b_lookup_elements);
    cuda_free_memory(device_verify_bitwise_xor_8_lookup_elements);
    cuda_free_memory(device_triple_xor_32_lookup_elements);

    // dump_interaction_traces(interaction_traces, 8, trace_size);

    cuda_free_memory(device_logup_denom);

    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_3);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_2);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_1);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_b_0);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_3);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_2);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_1);
    cuda_free_memory(device_lookup_verify_bitwise_xor_8_0);
    cuda_free_memory(device_lookup_triple_xor_32_0);

    cuda_free_memory(device_interaction_traces);
    cuda_free_memory(device_numerator0);
    cuda_free_memory(device_numerator1);
    cuda_free_memory(device_numerator2);
    cuda_free_memory(device_numerator3);
    cuda_free_memory(denom_inv);
}