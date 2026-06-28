#ifndef EVALUATE_XOR_ROT_32R8_CONSTRAINT_H
#define EVALUATE_XOR_ROT_32R8_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"


DEVICE_FORCEINLINE void splite16_low_part_size8_32R8_evaluate(
    m31 split_16_low_part_size_8_input_limb_0,
    m31 ms_8_bits_col0,
    m31 *output_limb_0,
    m31 *output_limb_1
) {
    const m31 M31_256 = {256};
    *output_limb_0 = sub(split_16_low_part_size_8_input_limb_0, mul(ms_8_bits_col0, M31_256));
    *output_limb_1 = ms_8_bits_col0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void bitwise_xor_numbits8_32R8_evaluate(
    m31 bitwise_xor_num_bits_8_input_limb_0,
    m31 bitwise_xor_num_bits_8_input_limb_1,
    m31 xor_col0,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 values[4] = {
        VERIFY_BITWISE_XOR_8_RELATION_ID,
        bitwise_xor_num_bits_8_input_limb_0,
        bitwise_xor_num_bits_8_input_limb_1,
        xor_col0
    };
    cuda_evaluator->add_to_relation<4>(common_lookup_elements, qm31{{1,0}, {0,0}}, values);
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void bitwise_xor_numbits8_b_32R8_evaluate(
    m31 bitwise_xor_num_bits_8_input_limb_0,
    m31 bitwise_xor_num_bits_8_input_limb_1,
    m31 xor_col0,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 values[4] = {
        VERIFY_BITWISE_XOR_8_B_RELATION_ID,
        bitwise_xor_num_bits_8_input_limb_0,
        bitwise_xor_num_bits_8_input_limb_1,
        xor_col0
    };
    cuda_evaluator->add_to_relation<4>(common_lookup_elements, qm31{{1,0}, {0,0}}, values);
}


template<typename EvaluatorT>
DEVICE_FORCEINLINE void xor_rot_32R8_evaluate(
    m31 xor_rot_32_r_8_input_limb_0,
    m31 xor_rot_32_r_8_input_limb_1,
    m31 xor_rot_32_r_8_input_limb_2,
    m31 xor_rot_32_r_8_input_limb_3,
    m31 ms_8_bits_col0,
    m31 ms_8_bits_col1,
    m31 ms_8_bits_col2,
    m31 ms_8_bits_col3,
    m31 xor_col4,
    m31 xor_col5,
    m31 xor_col6,
    m31 xor_col7,

    m31 *xor_rot_8_output_tmp_aa6bd_16_limb_0,
    m31 *xor_rot_8_output_tmp_aa6bd_16_limb_1,

    const CommonLookupElements& common_lookup_elements,

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_256 = {256};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_1_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_1_limb_1 = {0};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_3_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_3_limb_1 = {0};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_5_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_5_limb_1 = {0};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_7_limb_0 = {0};
    m31 split_16_low_part_size_8_output_tmp_aa6bd_7_limb_1 = {0};

    splite16_low_part_size8_32R8_evaluate(
        xor_rot_32_r_8_input_limb_0,
        ms_8_bits_col0,
        &split_16_low_part_size_8_output_tmp_aa6bd_1_limb_0,
        &split_16_low_part_size_8_output_tmp_aa6bd_1_limb_1
    );

    splite16_low_part_size8_32R8_evaluate(
        xor_rot_32_r_8_input_limb_1,
        ms_8_bits_col1,
        &split_16_low_part_size_8_output_tmp_aa6bd_3_limb_0,
        &split_16_low_part_size_8_output_tmp_aa6bd_3_limb_1
    );

    splite16_low_part_size8_32R8_evaluate(
        xor_rot_32_r_8_input_limb_2,
        ms_8_bits_col2,
        &split_16_low_part_size_8_output_tmp_aa6bd_5_limb_0,
        &split_16_low_part_size_8_output_tmp_aa6bd_5_limb_1
    );

    splite16_low_part_size8_32R8_evaluate(
        xor_rot_32_r_8_input_limb_3,
        ms_8_bits_col3,
        &split_16_low_part_size_8_output_tmp_aa6bd_7_limb_0,
        &split_16_low_part_size_8_output_tmp_aa6bd_7_limb_1
    );

    bitwise_xor_numbits8_32R8_evaluate(
        split_16_low_part_size_8_output_tmp_aa6bd_1_limb_0,
        split_16_low_part_size_8_output_tmp_aa6bd_5_limb_0,
        xor_col4,
        common_lookup_elements,
        cuda_evaluator
    );

    bitwise_xor_numbits8_32R8_evaluate(
        ms_8_bits_col0,
        ms_8_bits_col2,
        xor_col5,
        common_lookup_elements,
        cuda_evaluator
    );

    bitwise_xor_numbits8_b_32R8_evaluate(
        split_16_low_part_size_8_output_tmp_aa6bd_3_limb_0,
        split_16_low_part_size_8_output_tmp_aa6bd_7_limb_0,
        xor_col6,
        common_lookup_elements,
        cuda_evaluator
    );

    bitwise_xor_numbits8_b_32R8_evaluate(
        ms_8_bits_col1,
        ms_8_bits_col3,
        xor_col7,
        common_lookup_elements,
        cuda_evaluator
    );

    *xor_rot_8_output_tmp_aa6bd_16_limb_0 = mul(xor_col6, M31_256);
    *xor_rot_8_output_tmp_aa6bd_16_limb_0 = add(*xor_rot_8_output_tmp_aa6bd_16_limb_0, xor_col5);

    *xor_rot_8_output_tmp_aa6bd_16_limb_1 = mul(xor_col4, M31_256);
    *xor_rot_8_output_tmp_aa6bd_16_limb_1 = add(*xor_rot_8_output_tmp_aa6bd_16_limb_1, xor_col7);
}

#endif // EVALUATE_XOR_ROT_32R8_CONSTRAINT_H