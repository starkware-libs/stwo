/*
============================================
PoseidonHadesPermutation CUDA Subroutine
============================================

Subroutine: PoseidonHadesPermutation
Translated from: cairo-air/src/components/subroutines/poseidon_hades_permutation.rs
AIR version: 54d95c0d

Functionality:
- Implements the Hades permutation - the core operation of Poseidon hash function
- Input: 30 limbs representing 3 field elements (10 limbs each)
- Performs full rounds, partial rounds, and state transformations
- 167 intermediate trace columns
- Multiple relation lookups for verification

Structure:
1. Three LinearCombinationN2Coefs11 calls for input preparation
2. First PoseidonFullRoundChain lookup (4 full rounds)
3. Two RangeCheckFelt252Width27 lookups
4. First Cube252 lookup for first element cubing
5. LinearCombinationN4Coefs11M21 for combining elements
6. Second Cube252 lookup
7. LinearCombinationN4Coefs42M21 for combining cubed elements
8. Poseidon3PartialRoundsChain lookup (partial rounds)
9. Two LinearCombinationN4Coefs4211 calls for output preparation
10. Second PoseidonFullRoundChain lookup (final 4 full rounds)

Input:
- poseidon_hades_permutation_input_limb_0..29: 30 input limbs (3 field elements)
- combination_limb_0..9 (col0..9): First combination output
- p_coef (col10): First combination coefficient
- combination_limb_0..9 (col11..20): Second combination output
- p_coef (col21): Second combination coefficient
- combination_limb_0..9 (col22..31): Third combination output
- p_coef (col32): Third combination coefficient
- poseidon_full_round_chain_output_limb_0..29 (col33..62): First full round chain output
- cube_252_output_limb_0..9 (col63..72): First cube output
- combination_limb_0..9 (col73..82): Fourth combination output
- p_coef (col83): Fourth combination coefficient
- cube_252_output_limb_0..9 (col84..93): Second cube output
- combination_limb_0..9 (col94..103): Fifth combination output
- p_coef (col104): Fifth combination coefficient
- poseidon_3_partial_rounds_chain_output_limb_0..39 (col105..144): Partial rounds output
- combination_limb_0..9 (col145..154): Sixth combination output
- p_coef (col155): Sixth combination coefficient
- combination_limb_0..9 (col156..165): Seventh combination output
- p_coef (col166): Seventh combination coefficient
- poseidon_full_round_chain_output_limb_0..29 (col167..196): Second full round chain output

Total columns: 197 (col0 to col196)

Constraint count: Determined by subroutine calls
============================================
*/

#ifndef EVALUATE_POSEIDON_HADES_PERMUTATION_H
#define EVALUATE_POSEIDON_HADES_PERMUTATION_H

#include "fields.cuh"
#include "utils.cuh"
#include "eval_at_row.cuh"
#include "logup.cuh"
#include "relations.cuh"
#include "evaluate_linear_combination_n_2_coefs_1_1.cuh"
#include "evaluate_linear_combination_n_4_coefs_1_1_m2_1.cuh"
#include "evaluate_linear_combination_n_4_coefs_4_2_1_1.cuh"
#include "evaluate_linear_combination_n_4_coefs_4_2_m2_1.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void poseidon_hades_permutation_evaluate(
    // Input limbs (30 total - representing 3 field elements of 10 limbs each)
    m31 poseidon_hades_permutation_input_limb_0,
    m31 poseidon_hades_permutation_input_limb_1,
    m31 poseidon_hades_permutation_input_limb_2,
    m31 poseidon_hades_permutation_input_limb_3,
    m31 poseidon_hades_permutation_input_limb_4,
    m31 poseidon_hades_permutation_input_limb_5,
    m31 poseidon_hades_permutation_input_limb_6,
    m31 poseidon_hades_permutation_input_limb_7,
    m31 poseidon_hades_permutation_input_limb_8,
    m31 poseidon_hades_permutation_input_limb_9,
    m31 poseidon_hades_permutation_input_limb_10,
    m31 poseidon_hades_permutation_input_limb_11,
    m31 poseidon_hades_permutation_input_limb_12,
    m31 poseidon_hades_permutation_input_limb_13,
    m31 poseidon_hades_permutation_input_limb_14,
    m31 poseidon_hades_permutation_input_limb_15,
    m31 poseidon_hades_permutation_input_limb_16,
    m31 poseidon_hades_permutation_input_limb_17,
    m31 poseidon_hades_permutation_input_limb_18,
    m31 poseidon_hades_permutation_input_limb_19,
    m31 poseidon_hades_permutation_input_limb_20,
    m31 poseidon_hades_permutation_input_limb_21,
    m31 poseidon_hades_permutation_input_limb_22,
    m31 poseidon_hades_permutation_input_limb_23,
    m31 poseidon_hades_permutation_input_limb_24,
    m31 poseidon_hades_permutation_input_limb_25,
    m31 poseidon_hades_permutation_input_limb_26,
    m31 poseidon_hades_permutation_input_limb_27,
    m31 poseidon_hades_permutation_input_limb_28,
    m31 poseidon_hades_permutation_input_limb_29,

    // First combination (col0..10)
    m31 combination_limb_0_col0,
    m31 combination_limb_1_col1,
    m31 combination_limb_2_col2,
    m31 combination_limb_3_col3,
    m31 combination_limb_4_col4,
    m31 combination_limb_5_col5,
    m31 combination_limb_6_col6,
    m31 combination_limb_7_col7,
    m31 combination_limb_8_col8,
    m31 combination_limb_9_col9,
    m31 p_coef_col10,

    // Second combination (col11..21)
    m31 combination_limb_0_col11,
    m31 combination_limb_1_col12,
    m31 combination_limb_2_col13,
    m31 combination_limb_3_col14,
    m31 combination_limb_4_col15,
    m31 combination_limb_5_col16,
    m31 combination_limb_6_col17,
    m31 combination_limb_7_col18,
    m31 combination_limb_8_col19,
    m31 combination_limb_9_col20,
    m31 p_coef_col21,

    // Third combination (col22..32)
    m31 combination_limb_0_col22,
    m31 combination_limb_1_col23,
    m31 combination_limb_2_col24,
    m31 combination_limb_3_col25,
    m31 combination_limb_4_col26,
    m31 combination_limb_5_col27,
    m31 combination_limb_6_col28,
    m31 combination_limb_7_col29,
    m31 combination_limb_8_col30,
    m31 combination_limb_9_col31,
    m31 p_coef_col32,

    // First full round chain output (col33..62)
    m31 poseidon_full_round_chain_output_limb_0_col33,
    m31 poseidon_full_round_chain_output_limb_1_col34,
    m31 poseidon_full_round_chain_output_limb_2_col35,
    m31 poseidon_full_round_chain_output_limb_3_col36,
    m31 poseidon_full_round_chain_output_limb_4_col37,
    m31 poseidon_full_round_chain_output_limb_5_col38,
    m31 poseidon_full_round_chain_output_limb_6_col39,
    m31 poseidon_full_round_chain_output_limb_7_col40,
    m31 poseidon_full_round_chain_output_limb_8_col41,
    m31 poseidon_full_round_chain_output_limb_9_col42,
    m31 poseidon_full_round_chain_output_limb_10_col43,
    m31 poseidon_full_round_chain_output_limb_11_col44,
    m31 poseidon_full_round_chain_output_limb_12_col45,
    m31 poseidon_full_round_chain_output_limb_13_col46,
    m31 poseidon_full_round_chain_output_limb_14_col47,
    m31 poseidon_full_round_chain_output_limb_15_col48,
    m31 poseidon_full_round_chain_output_limb_16_col49,
    m31 poseidon_full_round_chain_output_limb_17_col50,
    m31 poseidon_full_round_chain_output_limb_18_col51,
    m31 poseidon_full_round_chain_output_limb_19_col52,
    m31 poseidon_full_round_chain_output_limb_20_col53,
    m31 poseidon_full_round_chain_output_limb_21_col54,
    m31 poseidon_full_round_chain_output_limb_22_col55,
    m31 poseidon_full_round_chain_output_limb_23_col56,
    m31 poseidon_full_round_chain_output_limb_24_col57,
    m31 poseidon_full_round_chain_output_limb_25_col58,
    m31 poseidon_full_round_chain_output_limb_26_col59,
    m31 poseidon_full_round_chain_output_limb_27_col60,
    m31 poseidon_full_round_chain_output_limb_28_col61,
    m31 poseidon_full_round_chain_output_limb_29_col62,

    // First cube output (col63..72)
    m31 cube_252_output_limb_0_col63,
    m31 cube_252_output_limb_1_col64,
    m31 cube_252_output_limb_2_col65,
    m31 cube_252_output_limb_3_col66,
    m31 cube_252_output_limb_4_col67,
    m31 cube_252_output_limb_5_col68,
    m31 cube_252_output_limb_6_col69,
    m31 cube_252_output_limb_7_col70,
    m31 cube_252_output_limb_8_col71,
    m31 cube_252_output_limb_9_col72,

    // Fourth combination (col73..83)
    m31 combination_limb_0_col73,
    m31 combination_limb_1_col74,
    m31 combination_limb_2_col75,
    m31 combination_limb_3_col76,
    m31 combination_limb_4_col77,
    m31 combination_limb_5_col78,
    m31 combination_limb_6_col79,
    m31 combination_limb_7_col80,
    m31 combination_limb_8_col81,
    m31 combination_limb_9_col82,
    m31 p_coef_col83,

    // Second cube output (col84..93)
    m31 cube_252_output_limb_0_col84,
    m31 cube_252_output_limb_1_col85,
    m31 cube_252_output_limb_2_col86,
    m31 cube_252_output_limb_3_col87,
    m31 cube_252_output_limb_4_col88,
    m31 cube_252_output_limb_5_col89,
    m31 cube_252_output_limb_6_col90,
    m31 cube_252_output_limb_7_col91,
    m31 cube_252_output_limb_8_col92,
    m31 cube_252_output_limb_9_col93,

    // Fifth combination (col94..104)
    m31 combination_limb_0_col94,
    m31 combination_limb_1_col95,
    m31 combination_limb_2_col96,
    m31 combination_limb_3_col97,
    m31 combination_limb_4_col98,
    m31 combination_limb_5_col99,
    m31 combination_limb_6_col100,
    m31 combination_limb_7_col101,
    m31 combination_limb_8_col102,
    m31 combination_limb_9_col103,
    m31 p_coef_col104,

    // Partial rounds chain output (col105..144) - 40 limbs for 4 field elements
    m31 poseidon_3_partial_rounds_chain_output_limb_0_col105,
    m31 poseidon_3_partial_rounds_chain_output_limb_1_col106,
    m31 poseidon_3_partial_rounds_chain_output_limb_2_col107,
    m31 poseidon_3_partial_rounds_chain_output_limb_3_col108,
    m31 poseidon_3_partial_rounds_chain_output_limb_4_col109,
    m31 poseidon_3_partial_rounds_chain_output_limb_5_col110,
    m31 poseidon_3_partial_rounds_chain_output_limb_6_col111,
    m31 poseidon_3_partial_rounds_chain_output_limb_7_col112,
    m31 poseidon_3_partial_rounds_chain_output_limb_8_col113,
    m31 poseidon_3_partial_rounds_chain_output_limb_9_col114,
    m31 poseidon_3_partial_rounds_chain_output_limb_10_col115,
    m31 poseidon_3_partial_rounds_chain_output_limb_11_col116,
    m31 poseidon_3_partial_rounds_chain_output_limb_12_col117,
    m31 poseidon_3_partial_rounds_chain_output_limb_13_col118,
    m31 poseidon_3_partial_rounds_chain_output_limb_14_col119,
    m31 poseidon_3_partial_rounds_chain_output_limb_15_col120,
    m31 poseidon_3_partial_rounds_chain_output_limb_16_col121,
    m31 poseidon_3_partial_rounds_chain_output_limb_17_col122,
    m31 poseidon_3_partial_rounds_chain_output_limb_18_col123,
    m31 poseidon_3_partial_rounds_chain_output_limb_19_col124,
    m31 poseidon_3_partial_rounds_chain_output_limb_20_col125,
    m31 poseidon_3_partial_rounds_chain_output_limb_21_col126,
    m31 poseidon_3_partial_rounds_chain_output_limb_22_col127,
    m31 poseidon_3_partial_rounds_chain_output_limb_23_col128,
    m31 poseidon_3_partial_rounds_chain_output_limb_24_col129,
    m31 poseidon_3_partial_rounds_chain_output_limb_25_col130,
    m31 poseidon_3_partial_rounds_chain_output_limb_26_col131,
    m31 poseidon_3_partial_rounds_chain_output_limb_27_col132,
    m31 poseidon_3_partial_rounds_chain_output_limb_28_col133,
    m31 poseidon_3_partial_rounds_chain_output_limb_29_col134,
    m31 poseidon_3_partial_rounds_chain_output_limb_30_col135,
    m31 poseidon_3_partial_rounds_chain_output_limb_31_col136,
    m31 poseidon_3_partial_rounds_chain_output_limb_32_col137,
    m31 poseidon_3_partial_rounds_chain_output_limb_33_col138,
    m31 poseidon_3_partial_rounds_chain_output_limb_34_col139,
    m31 poseidon_3_partial_rounds_chain_output_limb_35_col140,
    m31 poseidon_3_partial_rounds_chain_output_limb_36_col141,
    m31 poseidon_3_partial_rounds_chain_output_limb_37_col142,
    m31 poseidon_3_partial_rounds_chain_output_limb_38_col143,
    m31 poseidon_3_partial_rounds_chain_output_limb_39_col144,

    // Sixth combination (col145..155)
    m31 combination_limb_0_col145,
    m31 combination_limb_1_col146,
    m31 combination_limb_2_col147,
    m31 combination_limb_3_col148,
    m31 combination_limb_4_col149,
    m31 combination_limb_5_col150,
    m31 combination_limb_6_col151,
    m31 combination_limb_7_col152,
    m31 combination_limb_8_col153,
    m31 combination_limb_9_col154,
    m31 p_coef_col155,

    // Seventh combination (col156..166)
    m31 combination_limb_0_col156,
    m31 combination_limb_1_col157,
    m31 combination_limb_2_col158,
    m31 combination_limb_3_col159,
    m31 combination_limb_4_col160,
    m31 combination_limb_5_col161,
    m31 combination_limb_6_col162,
    m31 combination_limb_7_col163,
    m31 combination_limb_8_col164,
    m31 combination_limb_9_col165,
    m31 p_coef_col166,

    // Second full round chain output (col167..196)
    m31 poseidon_full_round_chain_output_limb_0_col167,
    m31 poseidon_full_round_chain_output_limb_1_col168,
    m31 poseidon_full_round_chain_output_limb_2_col169,
    m31 poseidon_full_round_chain_output_limb_3_col170,
    m31 poseidon_full_round_chain_output_limb_4_col171,
    m31 poseidon_full_round_chain_output_limb_5_col172,
    m31 poseidon_full_round_chain_output_limb_6_col173,
    m31 poseidon_full_round_chain_output_limb_7_col174,
    m31 poseidon_full_round_chain_output_limb_8_col175,
    m31 poseidon_full_round_chain_output_limb_9_col176,
    m31 poseidon_full_round_chain_output_limb_10_col177,
    m31 poseidon_full_round_chain_output_limb_11_col178,
    m31 poseidon_full_round_chain_output_limb_12_col179,
    m31 poseidon_full_round_chain_output_limb_13_col180,
    m31 poseidon_full_round_chain_output_limb_14_col181,
    m31 poseidon_full_round_chain_output_limb_15_col182,
    m31 poseidon_full_round_chain_output_limb_16_col183,
    m31 poseidon_full_round_chain_output_limb_17_col184,
    m31 poseidon_full_round_chain_output_limb_18_col185,
    m31 poseidon_full_round_chain_output_limb_19_col186,
    m31 poseidon_full_round_chain_output_limb_20_col187,
    m31 poseidon_full_round_chain_output_limb_21_col188,
    m31 poseidon_full_round_chain_output_limb_22_col189,
    m31 poseidon_full_round_chain_output_limb_23_col190,
    m31 poseidon_full_round_chain_output_limb_24_col191,
    m31 poseidon_full_round_chain_output_limb_25_col192,
    m31 poseidon_full_round_chain_output_limb_26_col193,
    m31 poseidon_full_round_chain_output_limb_27_col194,
    m31 poseidon_full_round_chain_output_limb_28_col195,
    m31 poseidon_full_round_chain_output_limb_29_col196,

    // Lookup elements
    const CommonLookupElements& common_lookup_elements,

    m31 seq,

    EvaluatorT *cuda_evaluator
) {
    // Constants
    const m31 M31_0 = m31(0);
    const m31 M31_1 = m31(1);
    const m31 M31_4 = m31(4);
    const m31 M31_20 = m31(20);
    const m31 M31_31 = m31(31);
    const m31 M31_35 = m31(35);
    const m31 M31_99 = m31(99);
    const m31 M31_112 = m31(112);
    const m31 M31_116 = m31(116);
    const m31 M31_154 = m31(154);
    const m31 M31_208 = m31(208);
    const m31 M31_248 = m31(248);
    const m31 M31_4883209 = m31(4883209);
    const m31 M31_4974792 = m31(4974792);
    const m31 M31_16173996 = m31(16173996);
    const m31 M31_18765944 = m31(18765944);
    const m31 M31_19292069 = m31(19292069);
    const m31 M31_22899501 = m31(22899501);
    const m31 M31_28820206 = m31(28820206);
    const m31 M31_33413160 = m31(33413160);
    const m31 M31_33439011 = m31(33439011);
    const m31 M31_36279186 = m31(36279186);
    const m31 M31_40454143 = m31(40454143);
    const m31 M31_41224388 = m31(41224388);
    const m31 M31_41320857 = m31(41320857);
    const m31 M31_44781849 = m31(44781849);
    const m31 M31_44848225 = m31(44848225);
    const m31 M31_45351266 = m31(45351266);
    const m31 M31_45553283 = m31(45553283);
    const m31 M31_48193339 = m31(48193339);
    const m31 M31_48383197 = m31(48383197);
    const m31 M31_48945103 = m31(48945103);
    const m31 M31_49157069 = m31(49157069);
    const m31 M31_49554771 = m31(49554771);
    const m31 M31_50468641 = m31(50468641);
    const m31 M31_50758155 = m31(50758155);
    const m31 M31_54415179 = m31(54415179);
    const m31 M31_55508188 = m31(55508188);
    const m31 M31_55955004 = m31(55955004);
    const m31 M31_58475513 = m31(58475513);
    const m31 M31_59852719 = m31(59852719);
    const m31 M31_60124463 = m31(60124463);
    const m31 M31_60709090 = m31(60709090);
    const m31 M31_62360091 = m31(62360091);
    const m31 M31_62439890 = m31(62439890);
    const m31 M31_65659846 = m31(65659846);
    const m31 M31_68491350 = m31(68491350);
    const m31 M31_72285071 = m31(72285071);
    const m31 M31_74972783 = m31(74972783);
    const m31 M31_75104388 = m31(75104388);
    const m31 M31_77099918 = m31(77099918);
    const m31 M31_78826183 = m31(78826183);
    const m31 M31_79012328 = m31(79012328);
    const m31 M31_86573645 = m31(86573645);
    const m31 M31_88680813 = m31(88680813);
    const m31 M31_90391646 = m31(90391646);
    const m31 M31_90842759 = m31(90842759);
    const m31 M31_91013252 = m31(91013252);
    const m31 M31_94624323 = m31(94624323);
    const m31 M31_95050340 = m31(95050340);
    const m31 M31_102193642 = m31(102193642);
    const m31 M31_103094260 = m31(103094260);
    const m31 M31_108487870 = m31(108487870);
    const m31 M31_112479959 = m31(112479959);
    const m31 M31_112795138 = m31(112795138);
    const m31 M31_116986206 = m31(116986206);
    const m31 M31_117420501 = m31(117420501);
    const m31 M31_119023582 = m31(119023582);
    const m31 M31_120369218 = m31(120369218);
    const m31 M31_121146754 = m31(121146754);
    const m31 M31_121657377 = m31(121657377);
    const m31 M31_122233508 = m31(122233508);
    const m31 M31_129717753 = m31(129717753);
    const m31 M31_130418270 = m31(130418270);
    const m31 M31_133303902 = m31(133303902);

    // ===================== First LinearCombinationN2Coefs11 =====================
    // Combines input limbs 0-9 with constants into combination col0-10
    linear_combination_n_2_coefs_1_1_evaluate(
        poseidon_hades_permutation_input_limb_0,
        poseidon_hades_permutation_input_limb_1,
        poseidon_hades_permutation_input_limb_2,
        poseidon_hades_permutation_input_limb_3,
        poseidon_hades_permutation_input_limb_4,
        poseidon_hades_permutation_input_limb_5,
        poseidon_hades_permutation_input_limb_6,
        poseidon_hades_permutation_input_limb_7,
        poseidon_hades_permutation_input_limb_8,
        poseidon_hades_permutation_input_limb_9,
        M31_74972783,
        M31_117420501,
        M31_112795138,
        M31_91013252,
        M31_60709090,
        M31_44848225,
        M31_108487870,
        M31_44781849,
        M31_102193642,
        M31_208,
        combination_limb_0_col0,
        combination_limb_1_col1,
        combination_limb_2_col2,
        combination_limb_3_col3,
        combination_limb_4_col4,
        combination_limb_5_col5,
        combination_limb_6_col6,
        combination_limb_7_col7,
        combination_limb_8_col8,
        combination_limb_9_col9,
        p_coef_col10,
        cuda_evaluator
    );

    // ===================== Second LinearCombinationN2Coefs11 =====================
    // Combines input limbs 10-19 with constants into combination col11-21
    linear_combination_n_2_coefs_1_1_evaluate(
        poseidon_hades_permutation_input_limb_10,
        poseidon_hades_permutation_input_limb_11,
        poseidon_hades_permutation_input_limb_12,
        poseidon_hades_permutation_input_limb_13,
        poseidon_hades_permutation_input_limb_14,
        poseidon_hades_permutation_input_limb_15,
        poseidon_hades_permutation_input_limb_16,
        poseidon_hades_permutation_input_limb_17,
        poseidon_hades_permutation_input_limb_18,
        poseidon_hades_permutation_input_limb_19,
        M31_41224388,
        M31_90391646,
        M31_36279186,
        M31_129717753,
        M31_94624323,
        M31_75104388,
        M31_133303902,
        M31_48945103,
        M31_41320857,
        M31_112,
        combination_limb_0_col11,
        combination_limb_1_col12,
        combination_limb_2_col13,
        combination_limb_3_col14,
        combination_limb_4_col15,
        combination_limb_5_col16,
        combination_limb_6_col17,
        combination_limb_7_col18,
        combination_limb_8_col19,
        combination_limb_9_col20,
        p_coef_col21,
        cuda_evaluator
    );

    // ===================== Third LinearCombinationN2Coefs11 =====================
    // Combines input limbs 20-29 with constants into combination col22-32
    linear_combination_n_2_coefs_1_1_evaluate(
        poseidon_hades_permutation_input_limb_20,
        poseidon_hades_permutation_input_limb_21,
        poseidon_hades_permutation_input_limb_22,
        poseidon_hades_permutation_input_limb_23,
        poseidon_hades_permutation_input_limb_24,
        poseidon_hades_permutation_input_limb_25,
        poseidon_hades_permutation_input_limb_26,
        poseidon_hades_permutation_input_limb_27,
        poseidon_hades_permutation_input_limb_28,
        poseidon_hades_permutation_input_limb_29,
        M31_4883209,
        M31_28820206,
        M31_79012328,
        M31_49157069,
        M31_78826183,
        M31_72285071,
        M31_33413160,
        M31_90842759,
        M31_60124463,
        M31_116,
        combination_limb_0_col22,
        combination_limb_1_col23,
        combination_limb_2_col24,
        combination_limb_3_col25,
        combination_limb_4_col26,
        combination_limb_5_col27,
        combination_limb_6_col28,
        combination_limb_7_col29,
        combination_limb_8_col30,
        combination_limb_9_col31,
        p_coef_col32,
        cuda_evaluator
    );

    // ===================== First PoseidonFullRoundChain Lookup (Input) =====================
    // Lookup input state (negative multiplicity)
    m31 poseidon_full_round_chain_chain_tmp_tmp_7d028_66 = add(seq, seq);  // seq * 2

    m31 input_values[33];
    input_values[0] = POSEIDON_FULL_ROUND_CHAIN_RELATION_ID;
    input_values[1] = poseidon_full_round_chain_chain_tmp_tmp_7d028_66;
    input_values[2] = M31_0;
    input_values[3] = combination_limb_0_col0;
    input_values[4] = combination_limb_1_col1;
    input_values[5] = combination_limb_2_col2;
    input_values[6] = combination_limb_3_col3;
    input_values[7] = combination_limb_4_col4;
    input_values[8] = combination_limb_5_col5;
    input_values[9] = combination_limb_6_col6;
    input_values[10] = combination_limb_7_col7;
    input_values[11] = combination_limb_8_col8;
    input_values[12] = combination_limb_9_col9;
    input_values[13] = combination_limb_0_col11;
    input_values[14] = combination_limb_1_col12;
    input_values[15] = combination_limb_2_col13;
    input_values[16] = combination_limb_3_col14;
    input_values[17] = combination_limb_4_col15;
    input_values[18] = combination_limb_5_col16;
    input_values[19] = combination_limb_6_col17;
    input_values[20] = combination_limb_7_col18;
    input_values[21] = combination_limb_8_col19;
    input_values[22] = combination_limb_9_col20;
    input_values[23] = combination_limb_0_col22;
    input_values[24] = combination_limb_1_col23;
    input_values[25] = combination_limb_2_col24;
    input_values[26] = combination_limb_3_col25;
    input_values[27] = combination_limb_4_col26;
    input_values[28] = combination_limb_5_col27;
    input_values[29] = combination_limb_6_col28;
    input_values[30] = combination_limb_7_col29;
    input_values[31] = combination_limb_8_col30;
    input_values[32] = combination_limb_9_col31;

    cuda_evaluator->template add_to_relation<33>(common_lookup_elements, qm31{{sub(M31_0, M31_1), 0}, {0, 0}}, input_values);

    // ===================== First PoseidonFullRoundChain Lookup (Output) =====================
    // Lookup output state (positive multiplicity)
    m31 output_values[33];
    output_values[0] = POSEIDON_FULL_ROUND_CHAIN_RELATION_ID;
    output_values[1] = poseidon_full_round_chain_chain_tmp_tmp_7d028_66;
    output_values[2] = M31_4;
    output_values[3] = poseidon_full_round_chain_output_limb_0_col33;
    output_values[4] = poseidon_full_round_chain_output_limb_1_col34;
    output_values[5] = poseidon_full_round_chain_output_limb_2_col35;
    output_values[6] = poseidon_full_round_chain_output_limb_3_col36;
    output_values[7] = poseidon_full_round_chain_output_limb_4_col37;
    output_values[8] = poseidon_full_round_chain_output_limb_5_col38;
    output_values[9] = poseidon_full_round_chain_output_limb_6_col39;
    output_values[10] = poseidon_full_round_chain_output_limb_7_col40;
    output_values[11] = poseidon_full_round_chain_output_limb_8_col41;
    output_values[12] = poseidon_full_round_chain_output_limb_9_col42;
    output_values[13] = poseidon_full_round_chain_output_limb_10_col43;
    output_values[14] = poseidon_full_round_chain_output_limb_11_col44;
    output_values[15] = poseidon_full_round_chain_output_limb_12_col45;
    output_values[16] = poseidon_full_round_chain_output_limb_13_col46;
    output_values[17] = poseidon_full_round_chain_output_limb_14_col47;
    output_values[18] = poseidon_full_round_chain_output_limb_15_col48;
    output_values[19] = poseidon_full_round_chain_output_limb_16_col49;
    output_values[20] = poseidon_full_round_chain_output_limb_17_col50;
    output_values[21] = poseidon_full_round_chain_output_limb_18_col51;
    output_values[22] = poseidon_full_round_chain_output_limb_19_col52;
    output_values[23] = poseidon_full_round_chain_output_limb_20_col53;
    output_values[24] = poseidon_full_round_chain_output_limb_21_col54;
    output_values[25] = poseidon_full_round_chain_output_limb_22_col55;
    output_values[26] = poseidon_full_round_chain_output_limb_23_col56;
    output_values[27] = poseidon_full_round_chain_output_limb_24_col57;
    output_values[28] = poseidon_full_round_chain_output_limb_25_col58;
    output_values[29] = poseidon_full_round_chain_output_limb_26_col59;
    output_values[30] = poseidon_full_round_chain_output_limb_27_col60;
    output_values[31] = poseidon_full_round_chain_output_limb_28_col61;
    output_values[32] = poseidon_full_round_chain_output_limb_29_col62;

    cuda_evaluator->template add_to_relation<33>(common_lookup_elements, qm31{{M31_1, 0}, {0, 0}}, output_values);

    // ===================== RangeCheckFelt252Width27 Lookups =====================
    // First range check (limbs 0-9)
    m31 range_check_1_values[10];
    range_check_1_values[0] = poseidon_full_round_chain_output_limb_0_col33;
    range_check_1_values[1] = poseidon_full_round_chain_output_limb_1_col34;
    range_check_1_values[2] = poseidon_full_round_chain_output_limb_2_col35;
    range_check_1_values[3] = poseidon_full_round_chain_output_limb_3_col36;
    range_check_1_values[4] = poseidon_full_round_chain_output_limb_4_col37;
    range_check_1_values[5] = poseidon_full_round_chain_output_limb_5_col38;
    range_check_1_values[6] = poseidon_full_round_chain_output_limb_6_col39;
    range_check_1_values[7] = poseidon_full_round_chain_output_limb_7_col40;
    range_check_1_values[8] = poseidon_full_round_chain_output_limb_8_col41;
    range_check_1_values[9] = poseidon_full_round_chain_output_limb_9_col42;

    m31 range_check_1_values_ext[11];
    range_check_1_values_ext[0] = RANGE_CHECK_252_WIDTH_27_RELATION_ID;
    for (int i = 0; i < 10; i++) range_check_1_values_ext[i+1] = range_check_1_values[i];
    cuda_evaluator->add_to_relation<11>(common_lookup_elements, qm31{{M31_1, 0}, {0, 0}}, range_check_1_values_ext);

    // Second range check (limbs 10-19)
    m31 range_check_2_values[10];
    range_check_2_values[0] = poseidon_full_round_chain_output_limb_10_col43;
    range_check_2_values[1] = poseidon_full_round_chain_output_limb_11_col44;
    range_check_2_values[2] = poseidon_full_round_chain_output_limb_12_col45;
    range_check_2_values[3] = poseidon_full_round_chain_output_limb_13_col46;
    range_check_2_values[4] = poseidon_full_round_chain_output_limb_14_col47;
    range_check_2_values[5] = poseidon_full_round_chain_output_limb_15_col48;
    range_check_2_values[6] = poseidon_full_round_chain_output_limb_16_col49;
    range_check_2_values[7] = poseidon_full_round_chain_output_limb_17_col50;
    range_check_2_values[8] = poseidon_full_round_chain_output_limb_18_col51;
    range_check_2_values[9] = poseidon_full_round_chain_output_limb_19_col52;

    m31 range_check_2_values_ext[11];
    range_check_2_values_ext[0] = RANGE_CHECK_252_WIDTH_27_RELATION_ID;
    for (int i = 0; i < 10; i++) range_check_2_values_ext[i+1] = range_check_2_values[i];
    cuda_evaluator->add_to_relation<11>(common_lookup_elements, qm31{{M31_1, 0}, {0, 0}}, range_check_2_values_ext);

    // ===================== First Cube252 Lookup =====================
    // Cube the third element (limbs 20-29)
    m31 cube_1_values[20];
    cube_1_values[0] = poseidon_full_round_chain_output_limb_20_col53;
    cube_1_values[1] = poseidon_full_round_chain_output_limb_21_col54;
    cube_1_values[2] = poseidon_full_round_chain_output_limb_22_col55;
    cube_1_values[3] = poseidon_full_round_chain_output_limb_23_col56;
    cube_1_values[4] = poseidon_full_round_chain_output_limb_24_col57;
    cube_1_values[5] = poseidon_full_round_chain_output_limb_25_col58;
    cube_1_values[6] = poseidon_full_round_chain_output_limb_26_col59;
    cube_1_values[7] = poseidon_full_round_chain_output_limb_27_col60;
    cube_1_values[8] = poseidon_full_round_chain_output_limb_28_col61;
    cube_1_values[9] = poseidon_full_round_chain_output_limb_29_col62;
    cube_1_values[10] = cube_252_output_limb_0_col63;
    cube_1_values[11] = cube_252_output_limb_1_col64;
    cube_1_values[12] = cube_252_output_limb_2_col65;
    cube_1_values[13] = cube_252_output_limb_3_col66;
    cube_1_values[14] = cube_252_output_limb_4_col67;
    cube_1_values[15] = cube_252_output_limb_5_col68;
    cube_1_values[16] = cube_252_output_limb_6_col69;
    cube_1_values[17] = cube_252_output_limb_7_col70;
    cube_1_values[18] = cube_252_output_limb_8_col71;
    cube_1_values[19] = cube_252_output_limb_9_col72;

    m31 cube_1_values_ext[21];
    cube_1_values_ext[0] = CUBE_252_RELATION_ID;
    for (int i = 0; i < 20; i++) cube_1_values_ext[i+1] = cube_1_values[i];
    cuda_evaluator->add_to_relation<21>(common_lookup_elements, qm31{{M31_1, 0}, {0, 0}}, cube_1_values_ext);

    // ===================== LinearCombinationN4Coefs11M21 =====================
    // Combines elements for partial rounds input
    linear_combination_n_4_coefs_1_1_m2_1_evaluate(
        poseidon_full_round_chain_output_limb_0_col33,
        poseidon_full_round_chain_output_limb_1_col34,
        poseidon_full_round_chain_output_limb_2_col35,
        poseidon_full_round_chain_output_limb_3_col36,
        poseidon_full_round_chain_output_limb_4_col37,
        poseidon_full_round_chain_output_limb_5_col38,
        poseidon_full_round_chain_output_limb_6_col39,
        poseidon_full_round_chain_output_limb_7_col40,
        poseidon_full_round_chain_output_limb_8_col41,
        poseidon_full_round_chain_output_limb_9_col42,
        poseidon_full_round_chain_output_limb_10_col43,
        poseidon_full_round_chain_output_limb_11_col44,
        poseidon_full_round_chain_output_limb_12_col45,
        poseidon_full_round_chain_output_limb_13_col46,
        poseidon_full_round_chain_output_limb_14_col47,
        poseidon_full_round_chain_output_limb_15_col48,
        poseidon_full_round_chain_output_limb_16_col49,
        poseidon_full_round_chain_output_limb_17_col50,
        poseidon_full_round_chain_output_limb_18_col51,
        poseidon_full_round_chain_output_limb_19_col52,
        cube_252_output_limb_0_col63,
        cube_252_output_limb_1_col64,
        cube_252_output_limb_2_col65,
        cube_252_output_limb_3_col66,
        cube_252_output_limb_4_col67,
        cube_252_output_limb_5_col68,
        cube_252_output_limb_6_col69,
        cube_252_output_limb_7_col70,
        cube_252_output_limb_8_col71,
        cube_252_output_limb_9_col72,
        M31_103094260,
        M31_121146754,
        M31_95050340,
        M31_16173996,
        M31_50758155,
        M31_54415179,
        M31_19292069,
        M31_45351266,
        M31_122233508,
        M31_248,
        combination_limb_0_col73,
        combination_limb_1_col74,
        combination_limb_2_col75,
        combination_limb_3_col76,
        combination_limb_4_col77,
        combination_limb_5_col78,
        combination_limb_6_col79,
        combination_limb_7_col80,
        combination_limb_8_col81,
        combination_limb_9_col82,
        p_coef_col83,
        common_lookup_elements,
        cuda_evaluator
    );

    // ===================== Second Cube252 Lookup =====================
    // Cube the combined element
    m31 cube_2_values[20];
    cube_2_values[0] = combination_limb_0_col73;
    cube_2_values[1] = combination_limb_1_col74;
    cube_2_values[2] = combination_limb_2_col75;
    cube_2_values[3] = combination_limb_3_col76;
    cube_2_values[4] = combination_limb_4_col77;
    cube_2_values[5] = combination_limb_5_col78;
    cube_2_values[6] = combination_limb_6_col79;
    cube_2_values[7] = combination_limb_7_col80;
    cube_2_values[8] = combination_limb_8_col81;
    cube_2_values[9] = combination_limb_9_col82;
    cube_2_values[10] = cube_252_output_limb_0_col84;
    cube_2_values[11] = cube_252_output_limb_1_col85;
    cube_2_values[12] = cube_252_output_limb_2_col86;
    cube_2_values[13] = cube_252_output_limb_3_col87;
    cube_2_values[14] = cube_252_output_limb_4_col88;
    cube_2_values[15] = cube_252_output_limb_5_col89;
    cube_2_values[16] = cube_252_output_limb_6_col90;
    cube_2_values[17] = cube_252_output_limb_7_col91;
    cube_2_values[18] = cube_252_output_limb_8_col92;
    cube_2_values[19] = cube_252_output_limb_9_col93;

    m31 cube_2_values_ext[21];
    cube_2_values_ext[0] = CUBE_252_RELATION_ID;
    for (int i = 0; i < 20; i++) cube_2_values_ext[i+1] = cube_2_values[i];
    cuda_evaluator->add_to_relation<21>(common_lookup_elements, qm31{{M31_1, 0}, {0, 0}}, cube_2_values_ext);

    // ===================== LinearCombinationN4Coefs42M21 =====================
    // Combines all cubed elements
    linear_combination_n_4_coefs_4_2_m2_1_evaluate(
        poseidon_full_round_chain_output_limb_0_col33,
        poseidon_full_round_chain_output_limb_1_col34,
        poseidon_full_round_chain_output_limb_2_col35,
        poseidon_full_round_chain_output_limb_3_col36,
        poseidon_full_round_chain_output_limb_4_col37,
        poseidon_full_round_chain_output_limb_5_col38,
        poseidon_full_round_chain_output_limb_6_col39,
        poseidon_full_round_chain_output_limb_7_col40,
        poseidon_full_round_chain_output_limb_8_col41,
        poseidon_full_round_chain_output_limb_9_col42,
        cube_252_output_limb_0_col63,
        cube_252_output_limb_1_col64,
        cube_252_output_limb_2_col65,
        cube_252_output_limb_3_col66,
        cube_252_output_limb_4_col67,
        cube_252_output_limb_5_col68,
        cube_252_output_limb_6_col69,
        cube_252_output_limb_7_col70,
        cube_252_output_limb_8_col71,
        cube_252_output_limb_9_col72,
        cube_252_output_limb_0_col84,
        cube_252_output_limb_1_col85,
        cube_252_output_limb_2_col86,
        cube_252_output_limb_3_col87,
        cube_252_output_limb_4_col88,
        cube_252_output_limb_5_col89,
        cube_252_output_limb_6_col90,
        cube_252_output_limb_7_col91,
        cube_252_output_limb_8_col92,
        cube_252_output_limb_9_col93,
        M31_121657377,
        M31_112479959,
        M31_130418270,
        M31_4974792,
        M31_59852719,
        M31_120369218,
        M31_62439890,
        M31_50468641,
        M31_86573645,
        M31_154,
        combination_limb_0_col94,
        combination_limb_1_col95,
        combination_limb_2_col96,
        combination_limb_3_col97,
        combination_limb_4_col98,
        combination_limb_5_col99,
        combination_limb_6_col100,
        combination_limb_7_col101,
        combination_limb_8_col102,
        combination_limb_9_col103,
        p_coef_col104,
        common_lookup_elements,
        cuda_evaluator
    );

    // ===================== Poseidon3PartialRoundsChain Lookup (Input) =====================
    m31 partial_input_values[43];
    partial_input_values[0] = POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID;
    partial_input_values[1] = seq;
    partial_input_values[2] = M31_4;
    partial_input_values[3] = cube_252_output_limb_0_col63;
    partial_input_values[4] = cube_252_output_limb_1_col64;
    partial_input_values[5] = cube_252_output_limb_2_col65;
    partial_input_values[6] = cube_252_output_limb_3_col66;
    partial_input_values[7] = cube_252_output_limb_4_col67;
    partial_input_values[8] = cube_252_output_limb_5_col68;
    partial_input_values[9] = cube_252_output_limb_6_col69;
    partial_input_values[10] = cube_252_output_limb_7_col70;
    partial_input_values[11] = cube_252_output_limb_8_col71;
    partial_input_values[12] = cube_252_output_limb_9_col72;
    partial_input_values[13] = combination_limb_0_col73;
    partial_input_values[14] = combination_limb_1_col74;
    partial_input_values[15] = combination_limb_2_col75;
    partial_input_values[16] = combination_limb_3_col76;
    partial_input_values[17] = combination_limb_4_col77;
    partial_input_values[18] = combination_limb_5_col78;
    partial_input_values[19] = combination_limb_6_col79;
    partial_input_values[20] = combination_limb_7_col80;
    partial_input_values[21] = combination_limb_8_col81;
    partial_input_values[22] = combination_limb_9_col82;
    partial_input_values[23] = cube_252_output_limb_0_col84;
    partial_input_values[24] = cube_252_output_limb_1_col85;
    partial_input_values[25] = cube_252_output_limb_2_col86;
    partial_input_values[26] = cube_252_output_limb_3_col87;
    partial_input_values[27] = cube_252_output_limb_4_col88;
    partial_input_values[28] = cube_252_output_limb_5_col89;
    partial_input_values[29] = cube_252_output_limb_6_col90;
    partial_input_values[30] = cube_252_output_limb_7_col91;
    partial_input_values[31] = cube_252_output_limb_8_col92;
    partial_input_values[32] = cube_252_output_limb_9_col93;
    partial_input_values[33] = combination_limb_0_col94;
    partial_input_values[34] = combination_limb_1_col95;
    partial_input_values[35] = combination_limb_2_col96;
    partial_input_values[36] = combination_limb_3_col97;
    partial_input_values[37] = combination_limb_4_col98;
    partial_input_values[38] = combination_limb_5_col99;
    partial_input_values[39] = combination_limb_6_col100;
    partial_input_values[40] = combination_limb_7_col101;
    partial_input_values[41] = combination_limb_8_col102;
    partial_input_values[42] = combination_limb_9_col103;

    cuda_evaluator->template add_to_relation<43>(common_lookup_elements, qm31{{sub(M31_0, M31_1), 0}, {0, 0}}, partial_input_values);

    // ===================== Poseidon3PartialRoundsChain Lookup (Output) =====================
    m31 partial_output_values[43];
    partial_output_values[0] = POSEIDON_3_PARTIAL_ROUNDS_CHAIN_RELATION_ID;
    partial_output_values[1] = seq;
    partial_output_values[2] = M31_31;
    partial_output_values[3] = poseidon_3_partial_rounds_chain_output_limb_0_col105;
    partial_output_values[4] = poseidon_3_partial_rounds_chain_output_limb_1_col106;
    partial_output_values[5] = poseidon_3_partial_rounds_chain_output_limb_2_col107;
    partial_output_values[6] = poseidon_3_partial_rounds_chain_output_limb_3_col108;
    partial_output_values[7] = poseidon_3_partial_rounds_chain_output_limb_4_col109;
    partial_output_values[8] = poseidon_3_partial_rounds_chain_output_limb_5_col110;
    partial_output_values[9] = poseidon_3_partial_rounds_chain_output_limb_6_col111;
    partial_output_values[10] = poseidon_3_partial_rounds_chain_output_limb_7_col112;
    partial_output_values[11] = poseidon_3_partial_rounds_chain_output_limb_8_col113;
    partial_output_values[12] = poseidon_3_partial_rounds_chain_output_limb_9_col114;
    partial_output_values[13] = poseidon_3_partial_rounds_chain_output_limb_10_col115;
    partial_output_values[14] = poseidon_3_partial_rounds_chain_output_limb_11_col116;
    partial_output_values[15] = poseidon_3_partial_rounds_chain_output_limb_12_col117;
    partial_output_values[16] = poseidon_3_partial_rounds_chain_output_limb_13_col118;
    partial_output_values[17] = poseidon_3_partial_rounds_chain_output_limb_14_col119;
    partial_output_values[18] = poseidon_3_partial_rounds_chain_output_limb_15_col120;
    partial_output_values[19] = poseidon_3_partial_rounds_chain_output_limb_16_col121;
    partial_output_values[20] = poseidon_3_partial_rounds_chain_output_limb_17_col122;
    partial_output_values[21] = poseidon_3_partial_rounds_chain_output_limb_18_col123;
    partial_output_values[22] = poseidon_3_partial_rounds_chain_output_limb_19_col124;
    partial_output_values[23] = poseidon_3_partial_rounds_chain_output_limb_20_col125;
    partial_output_values[24] = poseidon_3_partial_rounds_chain_output_limb_21_col126;
    partial_output_values[25] = poseidon_3_partial_rounds_chain_output_limb_22_col127;
    partial_output_values[26] = poseidon_3_partial_rounds_chain_output_limb_23_col128;
    partial_output_values[27] = poseidon_3_partial_rounds_chain_output_limb_24_col129;
    partial_output_values[28] = poseidon_3_partial_rounds_chain_output_limb_25_col130;
    partial_output_values[29] = poseidon_3_partial_rounds_chain_output_limb_26_col131;
    partial_output_values[30] = poseidon_3_partial_rounds_chain_output_limb_27_col132;
    partial_output_values[31] = poseidon_3_partial_rounds_chain_output_limb_28_col133;
    partial_output_values[32] = poseidon_3_partial_rounds_chain_output_limb_29_col134;
    partial_output_values[33] = poseidon_3_partial_rounds_chain_output_limb_30_col135;
    partial_output_values[34] = poseidon_3_partial_rounds_chain_output_limb_31_col136;
    partial_output_values[35] = poseidon_3_partial_rounds_chain_output_limb_32_col137;
    partial_output_values[36] = poseidon_3_partial_rounds_chain_output_limb_33_col138;
    partial_output_values[37] = poseidon_3_partial_rounds_chain_output_limb_34_col139;
    partial_output_values[38] = poseidon_3_partial_rounds_chain_output_limb_35_col140;
    partial_output_values[39] = poseidon_3_partial_rounds_chain_output_limb_36_col141;
    partial_output_values[40] = poseidon_3_partial_rounds_chain_output_limb_37_col142;
    partial_output_values[41] = poseidon_3_partial_rounds_chain_output_limb_38_col143;
    partial_output_values[42] = poseidon_3_partial_rounds_chain_output_limb_39_col144;

    cuda_evaluator->template add_to_relation<43>(common_lookup_elements, qm31{{M31_1, 0}, {0, 0}}, partial_output_values);

    // ===================== First LinearCombinationN4Coefs4211 =====================
    // Combines partial rounds output for final full rounds
    linear_combination_n_4_coefs_4_2_1_1_evaluate(
        poseidon_3_partial_rounds_chain_output_limb_0_col105,
        poseidon_3_partial_rounds_chain_output_limb_1_col106,
        poseidon_3_partial_rounds_chain_output_limb_2_col107,
        poseidon_3_partial_rounds_chain_output_limb_3_col108,
        poseidon_3_partial_rounds_chain_output_limb_4_col109,
        poseidon_3_partial_rounds_chain_output_limb_5_col110,
        poseidon_3_partial_rounds_chain_output_limb_6_col111,
        poseidon_3_partial_rounds_chain_output_limb_7_col112,
        poseidon_3_partial_rounds_chain_output_limb_8_col113,
        poseidon_3_partial_rounds_chain_output_limb_9_col114,
        poseidon_3_partial_rounds_chain_output_limb_10_col115,
        poseidon_3_partial_rounds_chain_output_limb_11_col116,
        poseidon_3_partial_rounds_chain_output_limb_12_col117,
        poseidon_3_partial_rounds_chain_output_limb_13_col118,
        poseidon_3_partial_rounds_chain_output_limb_14_col119,
        poseidon_3_partial_rounds_chain_output_limb_15_col120,
        poseidon_3_partial_rounds_chain_output_limb_16_col121,
        poseidon_3_partial_rounds_chain_output_limb_17_col122,
        poseidon_3_partial_rounds_chain_output_limb_18_col123,
        poseidon_3_partial_rounds_chain_output_limb_19_col124,
        poseidon_3_partial_rounds_chain_output_limb_20_col125,
        poseidon_3_partial_rounds_chain_output_limb_21_col126,
        poseidon_3_partial_rounds_chain_output_limb_22_col127,
        poseidon_3_partial_rounds_chain_output_limb_23_col128,
        poseidon_3_partial_rounds_chain_output_limb_24_col129,
        poseidon_3_partial_rounds_chain_output_limb_25_col130,
        poseidon_3_partial_rounds_chain_output_limb_26_col131,
        poseidon_3_partial_rounds_chain_output_limb_27_col132,
        poseidon_3_partial_rounds_chain_output_limb_28_col133,
        poseidon_3_partial_rounds_chain_output_limb_29_col134,
        M31_40454143,
        M31_49554771,
        M31_55508188,
        M31_116986206,
        M31_88680813,
        M31_45553283,
        M31_62360091,
        M31_77099918,
        M31_22899501,
        M31_99,
        combination_limb_0_col145,
        combination_limb_1_col146,
        combination_limb_2_col147,
        combination_limb_3_col148,
        combination_limb_4_col149,
        combination_limb_5_col150,
        combination_limb_6_col151,
        combination_limb_7_col152,
        combination_limb_8_col153,
        combination_limb_9_col154,
        p_coef_col155,
        common_lookup_elements,
        cuda_evaluator
    );

    // ===================== Second LinearCombinationN4Coefs4211 =====================
    // Combines remaining partial rounds output
    linear_combination_n_4_coefs_4_2_1_1_evaluate(
        poseidon_3_partial_rounds_chain_output_limb_20_col125,
        poseidon_3_partial_rounds_chain_output_limb_21_col126,
        poseidon_3_partial_rounds_chain_output_limb_22_col127,
        poseidon_3_partial_rounds_chain_output_limb_23_col128,
        poseidon_3_partial_rounds_chain_output_limb_24_col129,
        poseidon_3_partial_rounds_chain_output_limb_25_col130,
        poseidon_3_partial_rounds_chain_output_limb_26_col131,
        poseidon_3_partial_rounds_chain_output_limb_27_col132,
        poseidon_3_partial_rounds_chain_output_limb_28_col133,
        poseidon_3_partial_rounds_chain_output_limb_29_col134,
        poseidon_3_partial_rounds_chain_output_limb_30_col135,
        poseidon_3_partial_rounds_chain_output_limb_31_col136,
        poseidon_3_partial_rounds_chain_output_limb_32_col137,
        poseidon_3_partial_rounds_chain_output_limb_33_col138,
        poseidon_3_partial_rounds_chain_output_limb_34_col139,
        poseidon_3_partial_rounds_chain_output_limb_35_col140,
        poseidon_3_partial_rounds_chain_output_limb_36_col141,
        poseidon_3_partial_rounds_chain_output_limb_37_col142,
        poseidon_3_partial_rounds_chain_output_limb_38_col143,
        poseidon_3_partial_rounds_chain_output_limb_39_col144,
        combination_limb_0_col145,
        combination_limb_1_col146,
        combination_limb_2_col147,
        combination_limb_3_col148,
        combination_limb_4_col149,
        combination_limb_5_col150,
        combination_limb_6_col151,
        combination_limb_7_col152,
        combination_limb_8_col153,
        combination_limb_9_col154,
        M31_48383197,
        M31_48193339,
        M31_55955004,
        M31_65659846,
        M31_68491350,
        M31_119023582,
        M31_33439011,
        M31_58475513,
        M31_18765944,
        M31_20,
        combination_limb_0_col156,
        combination_limb_1_col157,
        combination_limb_2_col158,
        combination_limb_3_col159,
        combination_limb_4_col160,
        combination_limb_5_col161,
        combination_limb_6_col162,
        combination_limb_7_col163,
        combination_limb_8_col164,
        combination_limb_9_col165,
        p_coef_col166,
        common_lookup_elements,
        cuda_evaluator
    );

    // ===================== Second PoseidonFullRoundChain Lookup (Input) =====================
    m31 poseidon_full_round_chain_chain_id_tmp_7d028_149 = add(poseidon_full_round_chain_chain_tmp_tmp_7d028_66, M31_1);

    m31 final_input_values[33];
    final_input_values[0] = POSEIDON_FULL_ROUND_CHAIN_RELATION_ID;
    final_input_values[1] = poseidon_full_round_chain_chain_id_tmp_7d028_149;
    final_input_values[2] = M31_31;
    final_input_values[3] = combination_limb_0_col156;
    final_input_values[4] = combination_limb_1_col157;
    final_input_values[5] = combination_limb_2_col158;
    final_input_values[6] = combination_limb_3_col159;
    final_input_values[7] = combination_limb_4_col160;
    final_input_values[8] = combination_limb_5_col161;
    final_input_values[9] = combination_limb_6_col162;
    final_input_values[10] = combination_limb_7_col163;
    final_input_values[11] = combination_limb_8_col164;
    final_input_values[12] = combination_limb_9_col165;
    final_input_values[13] = combination_limb_0_col145;
    final_input_values[14] = combination_limb_1_col146;
    final_input_values[15] = combination_limb_2_col147;
    final_input_values[16] = combination_limb_3_col148;
    final_input_values[17] = combination_limb_4_col149;
    final_input_values[18] = combination_limb_5_col150;
    final_input_values[19] = combination_limb_6_col151;
    final_input_values[20] = combination_limb_7_col152;
    final_input_values[21] = combination_limb_8_col153;
    final_input_values[22] = combination_limb_9_col154;
    final_input_values[23] = poseidon_3_partial_rounds_chain_output_limb_30_col135;
    final_input_values[24] = poseidon_3_partial_rounds_chain_output_limb_31_col136;
    final_input_values[25] = poseidon_3_partial_rounds_chain_output_limb_32_col137;
    final_input_values[26] = poseidon_3_partial_rounds_chain_output_limb_33_col138;
    final_input_values[27] = poseidon_3_partial_rounds_chain_output_limb_34_col139;
    final_input_values[28] = poseidon_3_partial_rounds_chain_output_limb_35_col140;
    final_input_values[29] = poseidon_3_partial_rounds_chain_output_limb_36_col141;
    final_input_values[30] = poseidon_3_partial_rounds_chain_output_limb_37_col142;
    final_input_values[31] = poseidon_3_partial_rounds_chain_output_limb_38_col143;
    final_input_values[32] = poseidon_3_partial_rounds_chain_output_limb_39_col144;

    cuda_evaluator->template add_to_relation<33>(common_lookup_elements, qm31{{sub(M31_0, M31_1), 0}, {0, 0}}, final_input_values);

    // ===================== Second PoseidonFullRoundChain Lookup (Output) =====================
    m31 final_output_values[33];
    final_output_values[0] = POSEIDON_FULL_ROUND_CHAIN_RELATION_ID;
    final_output_values[1] = poseidon_full_round_chain_chain_id_tmp_7d028_149;
    final_output_values[2] = M31_35;
    final_output_values[3] = poseidon_full_round_chain_output_limb_0_col167;
    final_output_values[4] = poseidon_full_round_chain_output_limb_1_col168;
    final_output_values[5] = poseidon_full_round_chain_output_limb_2_col169;
    final_output_values[6] = poseidon_full_round_chain_output_limb_3_col170;
    final_output_values[7] = poseidon_full_round_chain_output_limb_4_col171;
    final_output_values[8] = poseidon_full_round_chain_output_limb_5_col172;
    final_output_values[9] = poseidon_full_round_chain_output_limb_6_col173;
    final_output_values[10] = poseidon_full_round_chain_output_limb_7_col174;
    final_output_values[11] = poseidon_full_round_chain_output_limb_8_col175;
    final_output_values[12] = poseidon_full_round_chain_output_limb_9_col176;
    final_output_values[13] = poseidon_full_round_chain_output_limb_10_col177;
    final_output_values[14] = poseidon_full_round_chain_output_limb_11_col178;
    final_output_values[15] = poseidon_full_round_chain_output_limb_12_col179;
    final_output_values[16] = poseidon_full_round_chain_output_limb_13_col180;
    final_output_values[17] = poseidon_full_round_chain_output_limb_14_col181;
    final_output_values[18] = poseidon_full_round_chain_output_limb_15_col182;
    final_output_values[19] = poseidon_full_round_chain_output_limb_16_col183;
    final_output_values[20] = poseidon_full_round_chain_output_limb_17_col184;
    final_output_values[21] = poseidon_full_round_chain_output_limb_18_col185;
    final_output_values[22] = poseidon_full_round_chain_output_limb_19_col186;
    final_output_values[23] = poseidon_full_round_chain_output_limb_20_col187;
    final_output_values[24] = poseidon_full_round_chain_output_limb_21_col188;
    final_output_values[25] = poseidon_full_round_chain_output_limb_22_col189;
    final_output_values[26] = poseidon_full_round_chain_output_limb_23_col190;
    final_output_values[27] = poseidon_full_round_chain_output_limb_24_col191;
    final_output_values[28] = poseidon_full_round_chain_output_limb_25_col192;
    final_output_values[29] = poseidon_full_round_chain_output_limb_26_col193;
    final_output_values[30] = poseidon_full_round_chain_output_limb_27_col194;
    final_output_values[31] = poseidon_full_round_chain_output_limb_28_col195;
    final_output_values[32] = poseidon_full_round_chain_output_limb_29_col196;

    cuda_evaluator->template add_to_relation<33>(common_lookup_elements, qm31{{M31_1, 0}, {0, 0}}, final_output_values);
}

#endif // EVALUATE_POSEIDON_HADES_PERMUTATION_H
