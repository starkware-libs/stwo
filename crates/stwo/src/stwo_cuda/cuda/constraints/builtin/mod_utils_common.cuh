#ifndef MOD_UTILS_COMMON_H
#define MOD_UTILS_COMMON_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include "evaluate_read_positive_num_bits.cuh"
#include "evaluate_read_small.cuh"

// CUDA version ModUtils::evaluate.
// Directly translated from cairo-air/src/components/subroutines/mod_utils.rs,
// Only depends on two types of lookup relations: MemoryAddressToId / MemoryIdToBig.

template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_mem_cond_verify_equal_known_id(
    m31 addr,
    m31 expected_id,
    m31 cond,
    m31 id_col0,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    // ReadId::evaluate: memory_address_to_id lookup.
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, addr, id_col0};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // (id_col0 - expected_id) * cond = 0
    cuda_evaluator->add_constraint(
        mul(sub(id_col0, expected_id), cond)
    );
}

template <typename EvaluatorT>
DEVICE_FORCEINLINE void mod_utils_evaluate(
    m31 mod_utils_input_limb_0,
    m31 mod_utils_input_limb_1,
    m31 is_instance_0_col0,
    m31 p0_id_col1,
    m31 p0_limb_0_col2,
    m31 p0_limb_1_col3,
    m31 p0_limb_2_col4,
    m31 p0_limb_3_col5,
    m31 p0_limb_4_col6,
    m31 p0_limb_5_col7,
    m31 p0_limb_6_col8,
    m31 p0_limb_7_col9,
    m31 p0_limb_8_col10,
    m31 p0_limb_9_col11,
    m31 p0_limb_10_col12,
    m31 p1_id_col13,
    m31 p1_limb_0_col14,
    m31 p1_limb_1_col15,
    m31 p1_limb_2_col16,
    m31 p1_limb_3_col17,
    m31 p1_limb_4_col18,
    m31 p1_limb_5_col19,
    m31 p1_limb_6_col20,
    m31 p1_limb_7_col21,
    m31 p1_limb_8_col22,
    m31 p1_limb_9_col23,
    m31 p1_limb_10_col24,
    m31 p2_id_col25,
    m31 p2_limb_0_col26,
    m31 p2_limb_1_col27,
    m31 p2_limb_2_col28,
    m31 p2_limb_3_col29,
    m31 p2_limb_4_col30,
    m31 p2_limb_5_col31,
    m31 p2_limb_6_col32,
    m31 p2_limb_7_col33,
    m31 p2_limb_8_col34,
    m31 p2_limb_9_col35,
    m31 p2_limb_10_col36,
    m31 p3_id_col37,
    m31 p3_limb_0_col38,
    m31 p3_limb_1_col39,
    m31 p3_limb_2_col40,
    m31 p3_limb_3_col41,
    m31 p3_limb_4_col42,
    m31 p3_limb_5_col43,
    m31 p3_limb_6_col44,
    m31 p3_limb_7_col45,
    m31 p3_limb_8_col46,
    m31 p3_limb_9_col47,
    m31 p3_limb_10_col48,
    m31 values_ptr_id_col49,
    m31 values_ptr_limb_0_col50,
    m31 values_ptr_limb_1_col51,
    m31 values_ptr_limb_2_col52,
    m31 values_ptr_limb_3_col53,
    m31 partial_limb_msb_col54,
    m31 offsets_ptr_id_col55,
    m31 offsets_ptr_limb_0_col56,
    m31 offsets_ptr_limb_1_col57,
    m31 offsets_ptr_limb_2_col58,
    m31 offsets_ptr_limb_3_col59,
    m31 partial_limb_msb_col60,
    m31 offsets_ptr_prev_id_col61,
    m31 offsets_ptr_prev_limb_0_col62,
    m31 offsets_ptr_prev_limb_1_col63,
    m31 offsets_ptr_prev_limb_2_col64,
    m31 offsets_ptr_prev_limb_3_col65,
    m31 partial_limb_msb_col66,
    m31 n_id_col67,
    m31 n_limb_0_col68,
    m31 n_limb_1_col69,
    m31 n_limb_2_col70,
    m31 n_limb_3_col71,
    m31 partial_limb_msb_col72,
    m31 n_prev_id_col73,
    m31 n_prev_limb_0_col74,
    m31 n_prev_limb_1_col75,
    m31 n_prev_limb_2_col76,
    m31 n_prev_limb_3_col77,
    m31 partial_limb_msb_col78,
    m31 values_ptr_prev_id_col79,
    m31 p_prev0_id_col80,
    m31 p_prev1_id_col81,
    m31 p_prev2_id_col82,
    m31 p_prev3_id_col83,
    m31 offsets_a_id_col84,
    m31 msb_col85,
    m31 mid_limbs_set_col86,
    m31 offsets_a_limb_0_col87,
    m31 offsets_a_limb_1_col88,
    m31 offsets_a_limb_2_col89,
    m31 remainder_bits_col90,
    m31 partial_limb_msb_col91,
    m31 offsets_b_id_col92,
    m31 msb_col93,
    m31 mid_limbs_set_col94,
    m31 offsets_b_limb_0_col95,
    m31 offsets_b_limb_1_col96,
    m31 offsets_b_limb_2_col97,
    m31 remainder_bits_col98,
    m31 partial_limb_msb_col99,
    m31 offsets_c_id_col100,
    m31 msb_col101,
    m31 mid_limbs_set_col102,
    m31 offsets_c_limb_0_col103,
    m31 offsets_c_limb_1_col104,
    m31 offsets_c_limb_2_col105,
    m31 remainder_bits_col106,
    m31 partial_limb_msb_col107,
    m31 a0_id_col108,
    m31 a0_limb_0_col109,
    m31 a0_limb_1_col110,
    m31 a0_limb_2_col111,
    m31 a0_limb_3_col112,
    m31 a0_limb_4_col113,
    m31 a0_limb_5_col114,
    m31 a0_limb_6_col115,
    m31 a0_limb_7_col116,
    m31 a0_limb_8_col117,
    m31 a0_limb_9_col118,
    m31 a0_limb_10_col119,
    m31 a1_id_col120,
    m31 a1_limb_0_col121,
    m31 a1_limb_1_col122,
    m31 a1_limb_2_col123,
    m31 a1_limb_3_col124,
    m31 a1_limb_4_col125,
    m31 a1_limb_5_col126,
    m31 a1_limb_6_col127,
    m31 a1_limb_7_col128,
    m31 a1_limb_8_col129,
    m31 a1_limb_9_col130,
    m31 a1_limb_10_col131,
    m31 a2_id_col132,
    m31 a2_limb_0_col133,
    m31 a2_limb_1_col134,
    m31 a2_limb_2_col135,
    m31 a2_limb_3_col136,
    m31 a2_limb_4_col137,
    m31 a2_limb_5_col138,
    m31 a2_limb_6_col139,
    m31 a2_limb_7_col140,
    m31 a2_limb_8_col141,
    m31 a2_limb_9_col142,
    m31 a2_limb_10_col143,
    m31 a3_id_col144,
    m31 a3_limb_0_col145,
    m31 a3_limb_1_col146,
    m31 a3_limb_2_col147,
    m31 a3_limb_3_col148,
    m31 a3_limb_4_col149,
    m31 a3_limb_5_col150,
    m31 a3_limb_6_col151,
    m31 a3_limb_7_col152,
    m31 a3_limb_8_col153,
    m31 a3_limb_9_col154,
    m31 a3_limb_10_col155,
    m31 b0_id_col156,
    m31 b0_limb_0_col157,
    m31 b0_limb_1_col158,
    m31 b0_limb_2_col159,
    m31 b0_limb_3_col160,
    m31 b0_limb_4_col161,
    m31 b0_limb_5_col162,
    m31 b0_limb_6_col163,
    m31 b0_limb_7_col164,
    m31 b0_limb_8_col165,
    m31 b0_limb_9_col166,
    m31 b0_limb_10_col167,
    m31 b1_id_col168,
    m31 b1_limb_0_col169,
    m31 b1_limb_1_col170,
    m31 b1_limb_2_col171,
    m31 b1_limb_3_col172,
    m31 b1_limb_4_col173,
    m31 b1_limb_5_col174,
    m31 b1_limb_6_col175,
    m31 b1_limb_7_col176,
    m31 b1_limb_8_col177,
    m31 b1_limb_9_col178,
    m31 b1_limb_10_col179,
    m31 b2_id_col180,
    m31 b2_limb_0_col181,
    m31 b2_limb_1_col182,
    m31 b2_limb_2_col183,
    m31 b2_limb_3_col184,
    m31 b2_limb_4_col185,
    m31 b2_limb_5_col186,
    m31 b2_limb_6_col187,
    m31 b2_limb_7_col188,
    m31 b2_limb_8_col189,
    m31 b2_limb_9_col190,
    m31 b2_limb_10_col191,
    m31 b3_id_col192,
    m31 b3_limb_0_col193,
    m31 b3_limb_1_col194,
    m31 b3_limb_2_col195,
    m31 b3_limb_3_col196,
    m31 b3_limb_4_col197,
    m31 b3_limb_5_col198,
    m31 b3_limb_6_col199,
    m31 b3_limb_7_col200,
    m31 b3_limb_8_col201,
    m31 b3_limb_9_col202,
    m31 b3_limb_10_col203,
    m31 c0_id_col204,
    m31 c0_limb_0_col205,
    m31 c0_limb_1_col206,
    m31 c0_limb_2_col207,
    m31 c0_limb_3_col208,
    m31 c0_limb_4_col209,
    m31 c0_limb_5_col210,
    m31 c0_limb_6_col211,
    m31 c0_limb_7_col212,
    m31 c0_limb_8_col213,
    m31 c0_limb_9_col214,
    m31 c0_limb_10_col215,
    m31 c1_id_col216,
    m31 c1_limb_0_col217,
    m31 c1_limb_1_col218,
    m31 c1_limb_2_col219,
    m31 c1_limb_3_col220,
    m31 c1_limb_4_col221,
    m31 c1_limb_5_col222,
    m31 c1_limb_6_col223,
    m31 c1_limb_7_col224,
    m31 c1_limb_8_col225,
    m31 c1_limb_9_col226,
    m31 c1_limb_10_col227,
    m31 c2_id_col228,
    m31 c2_limb_0_col229,
    m31 c2_limb_1_col230,
    m31 c2_limb_2_col231,
    m31 c2_limb_3_col232,
    m31 c2_limb_4_col233,
    m31 c2_limb_5_col234,
    m31 c2_limb_6_col235,
    m31 c2_limb_7_col236,
    m31 c2_limb_8_col237,
    m31 c2_limb_9_col238,
    m31 c2_limb_10_col239,
    m31 c3_id_col240,
    m31 c3_limb_0_col241,
    m31 c3_limb_1_col242,
    m31 c3_limb_2_col243,
    m31 c3_limb_3_col244,
    m31 c3_limb_4_col245,
    m31 c3_limb_5_col246,
    m31 c3_limb_6_col247,
    m31 c3_limb_7_col248,
    m31 c3_limb_8_col249,
    m31 c3_limb_9_col250,
    m31 c3_limb_10_col251,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_1 = m31(1);
    m31 M31_2 = m31(2);
    m31 M31_3 = m31(3);
    m31 M31_4 = m31(4);
    m31 M31_5 = m31(5);
    m31 M31_6 = m31(6);
    m31 M31_7 = m31(7);
    m31 M31_512 = m31(512);
    m31 M31_262144 = m31(262144);
    m31 M31_134217728 = m31(134217728);

    // is_instance_0 is 0 or 1
    cuda_evaluator->add_constraint(
        mul(is_instance_0_col0, sub(is_instance_0_col0, M31_1))
    );
    // is_instance_0 is 0 when instance_num is not 0
    cuda_evaluator->add_constraint(
        mul(is_instance_0_col0, mod_utils_input_limb_1)
    );

    // prev_instance_addr = base + 7 * ((seq - 1) + is_instance_0)
    m31 tmp_seq_minus_1 = sub(mod_utils_input_limb_1, M31_1);
    m31 inner = add(tmp_seq_minus_1, is_instance_0_col0);
    m31 seven_inner = mul(M31_7, inner);
    m31 prev_instance_addr = add(mod_utils_input_limb_0, seven_inner);

    // instance_addr = base + 7 * seq
    m31 seven_seq = mul(M31_7, mod_utils_input_limb_1);
    m31 instance_addr = add(mod_utils_input_limb_0, seven_seq);

    // p0..p3: ReadPositiveNumBits99
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        instance_addr,
        p0_id_col1,
        p0_limb_0_col2,
        p0_limb_1_col3,
        p0_limb_2_col4,
        p0_limb_3_col5,
        p0_limb_4_col6,
        p0_limb_5_col7,
        p0_limb_6_col8,
        p0_limb_7_col9,
        p0_limb_8_col10,
        p0_limb_9_col11,
        p0_limb_10_col12,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(instance_addr, M31_1),
        p1_id_col13,
        p1_limb_0_col14,
        p1_limb_1_col15,
        p1_limb_2_col16,
        p1_limb_3_col17,
        p1_limb_4_col18,
        p1_limb_5_col19,
        p1_limb_6_col20,
        p1_limb_7_col21,
        p1_limb_8_col22,
        p1_limb_9_col23,
        p1_limb_10_col24,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(instance_addr, M31_2),
        p2_id_col25,
        p2_limb_0_col26,
        p2_limb_1_col27,
        p2_limb_2_col28,
        p2_limb_3_col29,
        p2_limb_4_col30,
        p2_limb_5_col31,
        p2_limb_6_col32,
        p2_limb_7_col33,
        p2_limb_8_col34,
        p2_limb_9_col35,
        p2_limb_10_col36,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(instance_addr, M31_3),
        p3_id_col37,
        p3_limb_0_col38,
        p3_limb_1_col39,
        p3_limb_2_col40,
        p3_limb_3_col41,
        p3_limb_4_col42,
        p3_limb_5_col43,
        p3_limb_6_col44,
        p3_limb_7_col45,
        p3_limb_8_col46,
        p3_limb_9_col47,
        p3_limb_10_col48,
        common_lookup_elements,
        cuda_evaluator
    );

    // values_ptr, offsets_ptr, offsets_ptr_prev, n, n_prev : ReadPositiveNumBits29
    m31 dummy_vec_29[29];
    evaluate_read_positive_num_bits_29<EvaluatorT>(
        add(instance_addr, M31_4),
        values_ptr_id_col49,
        values_ptr_limb_0_col50,
        values_ptr_limb_1_col51,
        values_ptr_limb_2_col52,
        values_ptr_limb_3_col53,
        partial_limb_msb_col54,
        dummy_vec_29,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_29<EvaluatorT>(
        add(instance_addr, M31_5),
        offsets_ptr_id_col55,
        offsets_ptr_limb_0_col56,
        offsets_ptr_limb_1_col57,
        offsets_ptr_limb_2_col58,
        offsets_ptr_limb_3_col59,
        partial_limb_msb_col60,
        dummy_vec_29,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_29<EvaluatorT>(
        add(prev_instance_addr, M31_5),
        offsets_ptr_prev_id_col61,
        offsets_ptr_prev_limb_0_col62,
        offsets_ptr_prev_limb_1_col63,
        offsets_ptr_prev_limb_2_col64,
        offsets_ptr_prev_limb_3_col65,
        partial_limb_msb_col66,
        dummy_vec_29,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_29<EvaluatorT>(
        add(instance_addr, M31_6),
        n_id_col67,
        n_limb_0_col68,
        n_limb_1_col69,
        n_limb_2_col70,
        n_limb_3_col71,
        partial_limb_msb_col72,
        dummy_vec_29,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_29<EvaluatorT>(
        add(prev_instance_addr, M31_6),
        n_prev_id_col73,
        n_prev_limb_0_col74,
        n_prev_limb_1_col75,
        n_prev_limb_2_col76,
        n_prev_limb_3_col77,
        partial_limb_msb_col78,
        dummy_vec_29,
        common_lookup_elements,
        cuda_evaluator
    );

    // block_reset_condition
    m31 n_prev_as =
        add(
            add(
                add(n_prev_limb_0_col74,
                    mul(n_prev_limb_1_col75, M31_512)),
                mul(n_prev_limb_2_col76, M31_262144)),
            mul(n_prev_limb_3_col77, M31_134217728)
        );
    m31 n_as =
        add(
            add(
                add(n_limb_0_col68,
                    mul(n_limb_1_col69, M31_512)),
                mul(n_limb_2_col70, M31_262144)),
            mul(n_limb_3_col71, M31_134217728)
        );

    m31 block_reset_condition =
        mul(
            sub(n_prev_as, M31_1),
            sub(is_instance_0_col0, M31_1)
        );

    // Progression of n between instances..
    cuda_evaluator->add_constraint(
        mul(
            block_reset_condition,
            sub(sub(n_prev_as, M31_1), n_as)
        )
    );

    // Progression of offsets_ptr between instances..
    m31 offsets_ptr_as =
        add(
            add(
                add(offsets_ptr_limb_0_col56,
                    mul(offsets_ptr_limb_1_col57, M31_512)),
                mul(offsets_ptr_limb_2_col58, M31_262144)),
            mul(offsets_ptr_limb_3_col59, M31_134217728)
        );
    m31 offsets_ptr_prev_as =
        add(
            add(
                add(offsets_ptr_prev_limb_0_col62,
                    mul(offsets_ptr_prev_limb_1_col63, M31_512)),
                mul(offsets_ptr_prev_limb_2_col64, M31_262144)),
            mul(offsets_ptr_prev_limb_3_col65, M31_134217728)
        );

    cuda_evaluator->add_constraint(
        mul(
            block_reset_condition,
            sub(sub(offsets_ptr_as, M31_3), offsets_ptr_prev_as)
        )
    );

    // MemCondVerifyEqualKnownId checks
    evaluate_mem_cond_verify_equal_known_id<EvaluatorT>(
        add(prev_instance_addr, M31_4),
        values_ptr_id_col49,
        block_reset_condition,
        values_ptr_prev_id_col79,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_mem_cond_verify_equal_known_id<EvaluatorT>(
        prev_instance_addr,
        p0_id_col1,
        block_reset_condition,
        p_prev0_id_col80,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_mem_cond_verify_equal_known_id<EvaluatorT>(
        add(prev_instance_addr, M31_1),
        p1_id_col13,
        block_reset_condition,
        p_prev1_id_col81,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_mem_cond_verify_equal_known_id<EvaluatorT>(
        add(prev_instance_addr, M31_2),
        p2_id_col25,
        block_reset_condition,
        p_prev2_id_col82,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_mem_cond_verify_equal_known_id<EvaluatorT>(
        add(prev_instance_addr, M31_3),
        p3_id_col37,
        block_reset_condition,
        p_prev3_id_col83,
        common_lookup_elements,
        cuda_evaluator
    );

    // ReadSmall for offsets_a / offsets_b / offsets_c
    m31 offsets_ptr_as_u32 =
        add(
            add(
                add(offsets_ptr_limb_0_col56,
                    mul(offsets_ptr_limb_1_col57, M31_512)),
                mul(offsets_ptr_limb_2_col58, M31_262144)),
            mul(offsets_ptr_limb_3_col59, M31_134217728)
        );

    m31 read_small_output_tmp_78[2];
    m31 read_small_output_tmp_88[2];
    m31 read_small_output_tmp_98[2];

    evaluate_read_small<EvaluatorT>(
        offsets_ptr_as_u32,
        offsets_a_id_col84,
        msb_col85,
        mid_limbs_set_col86,
        offsets_a_limb_0_col87,
        offsets_a_limb_1_col88,
        offsets_a_limb_2_col89,
        remainder_bits_col90,
        partial_limb_msb_col91,
        read_small_output_tmp_78,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_small<EvaluatorT>(
        add(offsets_ptr_as_u32, M31_1),
        offsets_b_id_col92,
        msb_col93,
        mid_limbs_set_col94,
        offsets_b_limb_0_col95,
        offsets_b_limb_1_col96,
        offsets_b_limb_2_col97,
        remainder_bits_col98,
        partial_limb_msb_col99,
        read_small_output_tmp_88,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_small<EvaluatorT>(
        add(offsets_ptr_as_u32, M31_2),
        offsets_c_id_col100,
        msb_col101,
        mid_limbs_set_col102,
        offsets_c_limb_0_col103,
        offsets_c_limb_1_col104,
        offsets_c_limb_2_col105,
        remainder_bits_col106,
        partial_limb_msb_col107,
        read_small_output_tmp_98,
        common_lookup_elements,
        cuda_evaluator
    );

    m31 values_ptr_as =
        add(
            add(
                add(values_ptr_limb_0_col50,
                    mul(values_ptr_limb_1_col51, M31_512)),
                mul(values_ptr_limb_2_col52, M31_262144)),
            mul(values_ptr_limb_3_col53, M31_134217728)
        );

    m31 off_a = read_small_output_tmp_78[0];
    m31 off_b = read_small_output_tmp_88[0];
    m31 off_c = read_small_output_tmp_98[0];

    // Read a0..a3, b0..b3, c0..c3 via ReadPositiveNumBits99
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(values_ptr_as, off_a),
        a0_id_col108,
        a0_limb_0_col109,
        a0_limb_1_col110,
        a0_limb_2_col111,
        a0_limb_3_col112,
        a0_limb_4_col113,
        a0_limb_5_col114,
        a0_limb_6_col115,
        a0_limb_7_col116,
        a0_limb_8_col117,
        a0_limb_9_col118,
        a0_limb_10_col119,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_a), M31_1),
        a1_id_col120,
        a1_limb_0_col121,
        a1_limb_1_col122,
        a1_limb_2_col123,
        a1_limb_3_col124,
        a1_limb_4_col125,
        a1_limb_5_col126,
        a1_limb_6_col127,
        a1_limb_7_col128,
        a1_limb_8_col129,
        a1_limb_9_col130,
        a1_limb_10_col131,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_a), M31_2),
        a2_id_col132,
        a2_limb_0_col133,
        a2_limb_1_col134,
        a2_limb_2_col135,
        a2_limb_3_col136,
        a2_limb_4_col137,
        a2_limb_5_col138,
        a2_limb_6_col139,
        a2_limb_7_col140,
        a2_limb_8_col141,
        a2_limb_9_col142,
        a2_limb_10_col143,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_a), M31_3),
        a3_id_col144,
        a3_limb_0_col145,
        a3_limb_1_col146,
        a3_limb_2_col147,
        a3_limb_3_col148,
        a3_limb_4_col149,
        a3_limb_5_col150,
        a3_limb_6_col151,
        a3_limb_7_col152,
        a3_limb_8_col153,
        a3_limb_9_col154,
        a3_limb_10_col155,
        common_lookup_elements,
        cuda_evaluator
    );

    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(values_ptr_as, off_b),
        b0_id_col156,
        b0_limb_0_col157,
        b0_limb_1_col158,
        b0_limb_2_col159,
        b0_limb_3_col160,
        b0_limb_4_col161,
        b0_limb_5_col162,
        b0_limb_6_col163,
        b0_limb_7_col164,
        b0_limb_8_col165,
        b0_limb_9_col166,
        b0_limb_10_col167,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_b), M31_1),
        b1_id_col168,
        b1_limb_0_col169,
        b1_limb_1_col170,
        b1_limb_2_col171,
        b1_limb_3_col172,
        b1_limb_4_col173,
        b1_limb_5_col174,
        b1_limb_6_col175,
        b1_limb_7_col176,
        b1_limb_8_col177,
        b1_limb_9_col178,
        b1_limb_10_col179,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_b), M31_2),
        b2_id_col180,
        b2_limb_0_col181,
        b2_limb_1_col182,
        b2_limb_2_col183,
        b2_limb_3_col184,
        b2_limb_4_col185,
        b2_limb_5_col186,
        b2_limb_6_col187,
        b2_limb_7_col188,
        b2_limb_8_col189,
        b2_limb_9_col190,
        b2_limb_10_col191,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_b), M31_3),
        b3_id_col192,
        b3_limb_0_col193,
        b3_limb_1_col194,
        b3_limb_2_col195,
        b3_limb_3_col196,
        b3_limb_4_col197,
        b3_limb_5_col198,
        b3_limb_6_col199,
        b3_limb_7_col200,
        b3_limb_8_col201,
        b3_limb_9_col202,
        b3_limb_10_col203,
        common_lookup_elements,
        cuda_evaluator
    );

    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(values_ptr_as, off_c),
        c0_id_col204,
        c0_limb_0_col205,
        c0_limb_1_col206,
        c0_limb_2_col207,
        c0_limb_3_col208,
        c0_limb_4_col209,
        c0_limb_5_col210,
        c0_limb_6_col211,
        c0_limb_7_col212,
        c0_limb_8_col213,
        c0_limb_9_col214,
        c0_limb_10_col215,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_c), M31_1),
        c1_id_col216,
        c1_limb_0_col217,
        c1_limb_1_col218,
        c1_limb_2_col219,
        c1_limb_3_col220,
        c1_limb_4_col221,
        c1_limb_5_col222,
        c1_limb_6_col223,
        c1_limb_7_col224,
        c1_limb_8_col225,
        c1_limb_9_col226,
        c1_limb_10_col227,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_c), M31_2),
        c2_id_col228,
        c2_limb_0_col229,
        c2_limb_1_col230,
        c2_limb_2_col231,
        c2_limb_3_col232,
        c2_limb_4_col233,
        c2_limb_5_col234,
        c2_limb_6_col235,
        c2_limb_7_col236,
        c2_limb_8_col237,
        c2_limb_9_col238,
        c2_limb_10_col239,
        common_lookup_elements,
        cuda_evaluator
    );
    evaluate_read_positive_num_bits_99<EvaluatorT>(
        add(add(values_ptr_as, off_c), M31_3),
        c3_id_col240,
        c3_limb_0_col241,
        c3_limb_1_col242,
        c3_limb_2_col243,
        c3_limb_3_col244,
        c3_limb_4_col245,
        c3_limb_5_col246,
        c3_limb_6_col247,
        c3_limb_7_col248,
        c3_limb_8_col249,
        c3_limb_9_col250,
        c3_limb_10_col251,
        common_lookup_elements,
        cuda_evaluator
    );
}

#endif // MOD_UTILS_COMMON_H

