#ifndef EVALUATE_CREATE_BLAKE_INPUT_CONSTRAINT_H
#define EVALUATE_CREATE_BLAKE_INPUT_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"
#include "evaluate_read_blake_word.cuh"
#include "evaluate_xor_rot_32R8.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_create_blake_round_input(
    // 4 input limbs
    const m31 create_blake_round_input_input_limb_0,
    const m31 create_blake_round_input_input_limb_1,
    const m31 create_blake_round_input_input_limb_2,
    const m31 create_blake_round_input_input_limb_3,
    // 48 columns
    const m31 low_16_bits_col0,
    const m31 high_16_bits_col1,
    const m31 low_7_ms_bits_col2,
    const m31 high_14_ms_bits_col3,
    const m31 high_5_ms_bits_col4,
    const m31 state_0_id_col5,
    const m31 low_16_bits_col6,
    const m31 high_16_bits_col7,
    const m31 low_7_ms_bits_col8,
    const m31 high_14_ms_bits_col9,
    const m31 high_5_ms_bits_col10,
    const m31 state_1_id_col11,
    const m31 low_16_bits_col12,
    const m31 high_16_bits_col13,
    const m31 low_7_ms_bits_col14,
    const m31 high_14_ms_bits_col15,
    const m31 high_5_ms_bits_col16,
    const m31 state_2_id_col17,
    const m31 low_16_bits_col18,
    const m31 high_16_bits_col19,
    const m31 low_7_ms_bits_col20,
    const m31 high_14_ms_bits_col21,
    const m31 high_5_ms_bits_col22,
    const m31 state_3_id_col23,
    const m31 low_16_bits_col24,
    const m31 high_16_bits_col25,
    const m31 low_7_ms_bits_col26,
    const m31 high_14_ms_bits_col27,
    const m31 high_5_ms_bits_col28,
    const m31 state_4_id_col29,
    const m31 low_16_bits_col30,
    const m31 high_16_bits_col31,
    const m31 low_7_ms_bits_col32,
    const m31 high_14_ms_bits_col33,
    const m31 high_5_ms_bits_col34,
    const m31 state_5_id_col35,
    const m31 low_16_bits_col36,
    const m31 high_16_bits_col37,
    const m31 low_7_ms_bits_col38,
    const m31 high_14_ms_bits_col39,
    const m31 high_5_ms_bits_col40,
    const m31 state_6_id_col41,
    const m31 low_16_bits_col42,
    const m31 high_16_bits_col43,
    const m31 low_7_ms_bits_col44,
    const m31 high_14_ms_bits_col45,
    const m31 high_5_ms_bits_col46,
    const m31 state_7_id_col47,
    const m31 ms_8_bits_col48,
    const m31 ms_8_bits_col49,
    const m31 xor_col50,
    const m31 xor_col51,
    const m31 xor_col52,
    const m31 xor_col53,
    // output
    m31* out_limbs, // [32]

    // Lookup tables
    const CommonLookupElements& common_lookup_elements,
    // evaluator
    EvaluatorT* cuda_evaluator
) {
    // const
    const m31 M31_1 = m31(1);
    const m31 M31_127 = m31(127);
    const m31 M31_14 = m31(14);
    const m31 M31_15470 = m31(15470);
    const m31 M31_2 = m31(2);
    const m31 M31_23520 = m31(23520);
    const m31 M31_256 = m31(256);
    const m31 M31_26764 = m31(26764);
    const m31 M31_27145 = m31(27145);
    const m31 M31_3 = m31(3);
    const m31 M31_39685 = m31(39685);
    const m31 M31_4 = m31(4);
    const m31 M31_42319 = m31(42319);
    const m31 M31_44677 = m31(44677);
    const m31 M31_47975 = m31(47975);
    const m31 M31_5 = m31(5);
    const m31 M31_52505 = m31(52505);
    const m31 M31_55723 = m31(55723);
    const m31 M31_57468 = m31(57468);
    const m31 M31_58983 = m31(58983);
    const m31 M31_6 = m31(6);
    const m31 M31_62322 = m31(62322);
    const m31 M31_62778 = m31(62778);
    const m31 M31_7 = m31(7);
    const m31 M31_8067 = m31(8067);
    const m31 M31_81 = m31(81);
    const m31 M31_82 = m31(82);
    const m31 M31_9812 = m31(9812);

    // ReadBlakeWord
    m31 read_blake_word_output_tmp_f95c3_8_limb_0, read_blake_word_output_tmp_f95c3_8_limb_1;
    read_blake_word_evaluate(
        create_blake_round_input_input_limb_0,
        low_16_bits_col0, high_16_bits_col1, low_7_ms_bits_col2, high_14_ms_bits_col3, high_5_ms_bits_col4,
        state_0_id_col5,
        &read_blake_word_output_tmp_f95c3_8_limb_0, &read_blake_word_output_tmp_f95c3_8_limb_1,

        common_lookup_elements,
        cuda_evaluator
    );
    m31 read_blake_word_output_tmp_f95c3_17_limb_0, read_blake_word_output_tmp_f95c3_17_limb_1;
    read_blake_word_evaluate(
        add(create_blake_round_input_input_limb_0, M31_1),
        low_16_bits_col6, high_16_bits_col7, low_7_ms_bits_col8, high_14_ms_bits_col9, high_5_ms_bits_col10,
        state_1_id_col11,
        &read_blake_word_output_tmp_f95c3_17_limb_0, &read_blake_word_output_tmp_f95c3_17_limb_1,
        common_lookup_elements,
        cuda_evaluator
    );
    m31 read_blake_word_output_tmp_f95c3_26_limb_0, read_blake_word_output_tmp_f95c3_26_limb_1;
    read_blake_word_evaluate(
        add(create_blake_round_input_input_limb_0, M31_2),
        low_16_bits_col12, high_16_bits_col13, low_7_ms_bits_col14, high_14_ms_bits_col15, high_5_ms_bits_col16,
        state_2_id_col17,
        &read_blake_word_output_tmp_f95c3_26_limb_0, &read_blake_word_output_tmp_f95c3_26_limb_1,
        common_lookup_elements,
        cuda_evaluator
    );
    m31 read_blake_word_output_tmp_f95c3_35_limb_0, read_blake_word_output_tmp_f95c3_35_limb_1;
    read_blake_word_evaluate(
        add(create_blake_round_input_input_limb_0, M31_3),
        low_16_bits_col18, high_16_bits_col19, low_7_ms_bits_col20, high_14_ms_bits_col21, high_5_ms_bits_col22,
        state_3_id_col23,
        &read_blake_word_output_tmp_f95c3_35_limb_0, &read_blake_word_output_tmp_f95c3_35_limb_1,

        common_lookup_elements,
        cuda_evaluator
    );
    m31 read_blake_word_output_tmp_f95c3_44_limb_0, read_blake_word_output_tmp_f95c3_44_limb_1;
    read_blake_word_evaluate(
        add(create_blake_round_input_input_limb_0, M31_4),
        low_16_bits_col24, high_16_bits_col25, low_7_ms_bits_col26, high_14_ms_bits_col27, high_5_ms_bits_col28,
        state_4_id_col29,
        &read_blake_word_output_tmp_f95c3_44_limb_0, &read_blake_word_output_tmp_f95c3_44_limb_1,

        common_lookup_elements,
        cuda_evaluator
    );
    m31 read_blake_word_output_tmp_f95c3_53_limb_0, read_blake_word_output_tmp_f95c3_53_limb_1;
    read_blake_word_evaluate(
        add(create_blake_round_input_input_limb_0, M31_5),
        low_16_bits_col30, high_16_bits_col31, low_7_ms_bits_col32, high_14_ms_bits_col33, high_5_ms_bits_col34,
        state_5_id_col35,
        &read_blake_word_output_tmp_f95c3_53_limb_0, &read_blake_word_output_tmp_f95c3_53_limb_1,

        common_lookup_elements,
        cuda_evaluator
    );
    m31 read_blake_word_output_tmp_f95c3_62_limb_0, read_blake_word_output_tmp_f95c3_62_limb_1;
    read_blake_word_evaluate(
        add(create_blake_round_input_input_limb_0, M31_6),
        low_16_bits_col36, high_16_bits_col37, low_7_ms_bits_col38, high_14_ms_bits_col39, high_5_ms_bits_col40,
        state_6_id_col41,
        &read_blake_word_output_tmp_f95c3_62_limb_0, &read_blake_word_output_tmp_f95c3_62_limb_1,

        common_lookup_elements,
        cuda_evaluator
    );
    m31 read_blake_word_output_tmp_f95c3_71_limb_0, read_blake_word_output_tmp_f95c3_71_limb_1;
    read_blake_word_evaluate(
        add(create_blake_round_input_input_limb_0, M31_7),
        low_16_bits_col42, high_16_bits_col43, low_7_ms_bits_col44, high_14_ms_bits_col45, high_5_ms_bits_col46,
        state_7_id_col47,
        &read_blake_word_output_tmp_f95c3_71_limb_0, &read_blake_word_output_tmp_f95c3_71_limb_1,

        common_lookup_elements,
        cuda_evaluator
    );

    // Split16LowPartSize8
    m31 split_16_low_part_size_8_output_tmp_f95c3_73_limb_0, split_16_low_part_size_8_output_tmp_f95c3_73_limb_1;
    splite16_low_part_size8_32R8_evaluate(
        create_blake_round_input_input_limb_1,
        ms_8_bits_col48,
        &split_16_low_part_size_8_output_tmp_f95c3_73_limb_0, &split_16_low_part_size_8_output_tmp_f95c3_73_limb_1
    );
    m31 split_16_low_part_size_8_output_tmp_f95c3_75_limb_0, split_16_low_part_size_8_output_tmp_f95c3_75_limb_1;
    splite16_low_part_size8_32R8_evaluate(
        create_blake_round_input_input_limb_2,
        ms_8_bits_col49,
        &split_16_low_part_size_8_output_tmp_f95c3_75_limb_0, &split_16_low_part_size_8_output_tmp_f95c3_75_limb_1
    );

    // BitwiseXorNumBits8
    bitwise_xor_numbits8_32R8_evaluate(
        split_16_low_part_size_8_output_tmp_f95c3_73_limb_0, M31_127,
        xor_col50,
        common_lookup_elements,
        cuda_evaluator
    );
    bitwise_xor_numbits8_32R8_evaluate(
        ms_8_bits_col48, M31_82,
        xor_col51,
        common_lookup_elements,
        cuda_evaluator
    );
    bitwise_xor_numbits8_32R8_evaluate(
        split_16_low_part_size_8_output_tmp_f95c3_75_limb_0, M31_14,
        xor_col52,
        common_lookup_elements,
        cuda_evaluator
    );
    bitwise_xor_numbits8_32R8_evaluate(
        ms_8_bits_col49, M31_81,
        xor_col53,
        common_lookup_elements,
        cuda_evaluator
    );

    // output
    out_limbs[0] = low_16_bits_col0;
    out_limbs[1] = high_16_bits_col1;
    out_limbs[2] = low_16_bits_col6;
    out_limbs[3] = high_16_bits_col7;
    out_limbs[4] = low_16_bits_col12;
    out_limbs[5] = high_16_bits_col13;
    out_limbs[6] = low_16_bits_col18;
    out_limbs[7] = high_16_bits_col19;
    out_limbs[8] = low_16_bits_col24;
    out_limbs[9] = high_16_bits_col25;
    out_limbs[10] = low_16_bits_col30;
    out_limbs[11] = high_16_bits_col31;
    out_limbs[12] = low_16_bits_col36;
    out_limbs[13] = high_16_bits_col37;
    out_limbs[14] = low_16_bits_col42;
    out_limbs[15] = high_16_bits_col43;
    out_limbs[16] = M31_58983;
    out_limbs[17] = M31_27145;
    out_limbs[18] = M31_44677;
    out_limbs[19] = M31_47975;
    out_limbs[20] = M31_62322;
    out_limbs[21] = M31_15470;
    out_limbs[22] = M31_62778;
    out_limbs[23] = M31_42319;
    out_limbs[24] = add(xor_col50, mul(xor_col51, M31_256));
    out_limbs[25] = add(xor_col52, mul(xor_col53, M31_256));
    out_limbs[26] = M31_26764;
    out_limbs[27] = M31_39685;
    out_limbs[28] = add(mul(create_blake_round_input_input_limb_3, M31_9812), mul(sub(M31_1, create_blake_round_input_input_limb_3), M31_55723));
    out_limbs[29] = add(mul(create_blake_round_input_input_limb_3, M31_57468), mul(sub(M31_1, create_blake_round_input_input_limb_3), M31_8067));
    out_limbs[30] = M31_52505;
    out_limbs[31] = M31_23520;
}

#endif // EVALUATE_CREATE_BLAKE_INPUT_CONSTRAINT_H