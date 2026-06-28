#ifndef EVALUATE_XOR_ROT_32R12_CONSTRAINT_H
#define EVALUATE_XOR_ROT_32R12_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

DEVICE_FORCEINLINE void splite16_low_part_size12_evaluate(
    m31 split_16_low_part_size_12_input_limb_0,
    m31 ms_4_bits_col0,
    m31 *output_limb_0,
    m31 *output_limb_1
) {
    const m31 M31_4096 = {4096};
    *output_limb_0 = sub(split_16_low_part_size_12_input_limb_0, mul(ms_4_bits_col0, M31_4096));
    *output_limb_1 = ms_4_bits_col0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void bitwise_xor_numbits12_evaluate(
    m31 bitwise_xor_num_bits_12_input_limb_0,
    m31 bitwise_xor_num_bits_12_input_limb_1,
    m31 xor_col0,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 values[4] = {
        VERIFY_BITWISE_XOR_12_RELATION_ID,
        bitwise_xor_num_bits_12_input_limb_0,
        bitwise_xor_num_bits_12_input_limb_1,
        xor_col0
    };
    cuda_evaluator->add_to_relation<4>(common_lookup_elements, qm31{{1,0}, {0,0}}, values);
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void bitwise_xor_numbits4_evaluate(
    m31 bitwise_xor_num_bits_4_input_limb_0,
    m31 bitwise_xor_num_bits_4_input_limb_1,
    m31 xor_col0,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 values[4] = {
        VERIFY_BITWISE_XOR_4_RELATION_ID,
        bitwise_xor_num_bits_4_input_limb_0,
        bitwise_xor_num_bits_4_input_limb_1,
        xor_col0
    };
    cuda_evaluator->add_to_relation<4>(common_lookup_elements, qm31{{1,0}, {0,0}}, values);
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void xor_rot_32R12_evaluate(
    m31 xor_rot_32_r_12_input_limb_0,
    m31 xor_rot_32_r_12_input_limb_1,
    m31 xor_rot_32_r_12_input_limb_2,
    m31 xor_rot_32_r_12_input_limb_3,
    m31 ms_4_bits_col0,
    m31 ms_4_bits_col1,
    m31 ms_4_bits_col2,
    m31 ms_4_bits_col3,
    m31 xor_col4,
    m31 xor_col5,
    m31 xor_col6,
    m31 xor_col7,

    m31 *xor_rot_12_output_tmp_cf62f_16_limb_0,
    m31 *xor_rot_12_output_tmp_cf62f_16_limb_1,

    const CommonLookupElements& common_lookup_elements,

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_16 = {16};
    m31 split_16_low_part_size_12_output_tmp_cf62f_1_limb_0 = {0};
    m31 split_16_low_part_size_12_output_tmp_cf62f_1_limb_1 = {0};
    m31 split_16_low_part_size_12_output_tmp_cf62f_3_limb_0 = {0};
    m31 split_16_low_part_size_12_output_tmp_cf62f_3_limb_1 = {0};
    m31 split_16_low_part_size_12_output_tmp_cf62f_5_limb_0 = {0};
    m31 split_16_low_part_size_12_output_tmp_cf62f_5_limb_1 = {0};
    m31 split_16_low_part_size_12_output_tmp_cf62f_7_limb_0 = {0};
    m31 split_16_low_part_size_12_output_tmp_cf62f_7_limb_1 = {0};


    splite16_low_part_size12_evaluate(
        xor_rot_32_r_12_input_limb_0,
        ms_4_bits_col0,
        &split_16_low_part_size_12_output_tmp_cf62f_1_limb_0,
        &split_16_low_part_size_12_output_tmp_cf62f_1_limb_1
    );

    splite16_low_part_size12_evaluate(
        xor_rot_32_r_12_input_limb_1,
        ms_4_bits_col1,
        &split_16_low_part_size_12_output_tmp_cf62f_3_limb_0,
        &split_16_low_part_size_12_output_tmp_cf62f_3_limb_1
    );

    // if ((threadIdx.x + blockDim.x * blockIdx.x) == 0) {
    // }
    splite16_low_part_size12_evaluate(
        xor_rot_32_r_12_input_limb_2,
        ms_4_bits_col2,
        &split_16_low_part_size_12_output_tmp_cf62f_5_limb_0,
        &split_16_low_part_size_12_output_tmp_cf62f_5_limb_1
    );

    splite16_low_part_size12_evaluate(
        xor_rot_32_r_12_input_limb_3,
        ms_4_bits_col3,
        &split_16_low_part_size_12_output_tmp_cf62f_7_limb_0,
        &split_16_low_part_size_12_output_tmp_cf62f_7_limb_1
    );

    // if ((threadIdx.x + blockDim.x * blockIdx.x) == 0) {
    // }
    bitwise_xor_numbits12_evaluate(
        split_16_low_part_size_12_output_tmp_cf62f_1_limb_0,
        split_16_low_part_size_12_output_tmp_cf62f_5_limb_0,
        xor_col4,
        common_lookup_elements,
        cuda_evaluator
    );

    bitwise_xor_numbits4_evaluate(
        ms_4_bits_col0,
        ms_4_bits_col2,
        xor_col5,
        common_lookup_elements,
        cuda_evaluator
    );

    bitwise_xor_numbits12_evaluate(
        split_16_low_part_size_12_output_tmp_cf62f_3_limb_0,
        split_16_low_part_size_12_output_tmp_cf62f_7_limb_0,
        xor_col6,
        common_lookup_elements,
        cuda_evaluator
    );

    bitwise_xor_numbits4_evaluate(
        ms_4_bits_col1,
        ms_4_bits_col3,
        xor_col7,
        common_lookup_elements,
        cuda_evaluator
    );

    *xor_rot_12_output_tmp_cf62f_16_limb_0 = mul(xor_col6, M31_16);
    *xor_rot_12_output_tmp_cf62f_16_limb_0 = add(*xor_rot_12_output_tmp_cf62f_16_limb_0, xor_col5);

    *xor_rot_12_output_tmp_cf62f_16_limb_1 = mul(xor_col4, M31_16);
    *xor_rot_12_output_tmp_cf62f_16_limb_1 = add(*xor_rot_12_output_tmp_cf62f_16_limb_1, xor_col7);
}

#endif // EVALUATE_XOR_ROT_32R12_CONSTRAINT_H
