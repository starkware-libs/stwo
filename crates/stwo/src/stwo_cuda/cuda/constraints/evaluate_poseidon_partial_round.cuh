/*
============================================
PoseidonPartialRound CUDA Subroutine
============================================

Subroutine: PoseidonPartialRound
translated from: cairo-air/src/comptogethernts/subroutines/poseidon_partial_round.rs
AIR version: 54d95c0d

Functionality:
- Evaluates one partial round of Poseidon permutation
- Processes 3 field elements (z0_3, z1, z2) with cubing and linear combinations
- Applies round key and computes output state

Inputs:
- input: 50 limbs (5 groups of 10 limbs)
  - z0_3 (10 limbs): z0 cubed
  - z1 (10 limbs): z1 value
  - z1_3 (10 limbs): z1 cubed
  - z2 (10 limbs): z2 value
  - half_key (10 limbs): Round key
- cube_252_output (10 limbs): z2 cubed output
- combination (10 limbs + p_coef): First linear combination result
- combination (10 limbs + p_coef): Second linear combination result

Constraint Logic:
- 1 Cube252 lookup (z2 → cube_output)
- 1 LinearCombinationN6Coefs4231M11 call (coeffs [4,2,3,1,-1,1])
- 1 RangeCheckFelt252Width27 lookup (checking first combination)
- 1 LinearCombinationN1Coefs2 call (coeffs [2])

Relation Lookups:
- Cube252: 1 use
- RangeCheck_4_4_4_4: 2 uses (via LinearCombinationN6Coefs4231M11)
- RangeCheck_4_4: 1 use (via LinearCombinationN6Coefs4231M11)
- RangeCheckFelt252Width27: 1 use
- Plus 11 carry constraints (via LinearCombinationN1Coefs2)

============================================
*/

#ifndef EVALUATE_POSEIDON_PARTIAL_ROUND_H
#define EVALUATE_POSEIDON_PARTIAL_ROUND_H

#include "fields.cuh"
#include "utils.cuh"
#include "eval_at_row.cuh"
#include "evaluate_linear_combination_n_6_coefs_4_2_3_1_m1_1.cuh"
#include "evaluate_linear_combination_n_1_coefs_2.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void poseidon_partial_round_evaluate(
    // 50 input limbs (5 groups of 10 limbs each)
    // z0_3: input[0-9]
    m31 input_z0_3_limb_0, m31 input_z0_3_limb_1, m31 input_z0_3_limb_2,
    m31 input_z0_3_limb_3, m31 input_z0_3_limb_4, m31 input_z0_3_limb_5,
    m31 input_z0_3_limb_6, m31 input_z0_3_limb_7, m31 input_z0_3_limb_8,
    m31 input_z0_3_limb_9,
    // z1: input[10-19]
    m31 input_z1_limb_0, m31 input_z1_limb_1, m31 input_z1_limb_2,
    m31 input_z1_limb_3, m31 input_z1_limb_4, m31 input_z1_limb_5,
    m31 input_z1_limb_6, m31 input_z1_limb_7, m31 input_z1_limb_8,
    m31 input_z1_limb_9,
    // z1_3: input[20-29]
    m31 input_z1_3_limb_0, m31 input_z1_3_limb_1, m31 input_z1_3_limb_2,
    m31 input_z1_3_limb_3, m31 input_z1_3_limb_4, m31 input_z1_3_limb_5,
    m31 input_z1_3_limb_6, m31 input_z1_3_limb_7, m31 input_z1_3_limb_8,
    m31 input_z1_3_limb_9,
    // z2: input[30-39]
    m31 input_z2_limb_0, m31 input_z2_limb_1, m31 input_z2_limb_2,
    m31 input_z2_limb_3, m31 input_z2_limb_4, m31 input_z2_limb_5,
    m31 input_z2_limb_6, m31 input_z2_limb_7, m31 input_z2_limb_8,
    m31 input_z2_limb_9,
    // half_key: input[40-49]
    m31 input_half_key_limb_0, m31 input_half_key_limb_1, m31 input_half_key_limb_2,
    m31 input_half_key_limb_3, m31 input_half_key_limb_4, m31 input_half_key_limb_5,
    m31 input_half_key_limb_6, m31 input_half_key_limb_7, m31 input_half_key_limb_8,
    m31 input_half_key_limb_9,

    // Cube252 output limbs (10 limbs)
    m31 cube_252_output_limb_0, m31 cube_252_output_limb_1, m31 cube_252_output_limb_2,
    m31 cube_252_output_limb_3, m31 cube_252_output_limb_4, m31 cube_252_output_limb_5,
    m31 cube_252_output_limb_6, m31 cube_252_output_limb_7, m31 cube_252_output_limb_8,
    m31 cube_252_output_limb_9,

    // First combination result limbs (10 limbs + p_coef)
    m31 combination_limb_0_col10, m31 combination_limb_1_col11, m31 combination_limb_2_col12,
    m31 combination_limb_3_col13, m31 combination_limb_4_col14, m31 combination_limb_5_col15,
    m31 combination_limb_6_col16, m31 combination_limb_7_col17, m31 combination_limb_8_col18,
    m31 combination_limb_9_col19,
    m31 p_coef_col20,

    // Second combination result limbs (10 limbs + p_coef)
    m31 combination_limb_0_col21, m31 combination_limb_1_col22, m31 combination_limb_2_col23,
    m31 combination_limb_3_col24, m31 combination_limb_4_col25, m31 combination_limb_5_col26,
    m31 combination_limb_6_col27, m31 combination_limb_7_col28, m31 combination_limb_8_col29,
    m31 combination_limb_9_col30,
    m31 p_coef_col31,

    // Lookup elements
    const CommonLookupElements& common_lookup_elements,

    // Evaluator
    EvaluatorT *cuda_evaluator
) {
    // Step 1: Cube252 lookup (z2 → cube_output)
    {
        m31 values[20];
        values[0] = input_z2_limb_0;
        values[1] = input_z2_limb_1;
        values[2] = input_z2_limb_2;
        values[3] = input_z2_limb_3;
        values[4] = input_z2_limb_4;
        values[5] = input_z2_limb_5;
        values[6] = input_z2_limb_6;
        values[7] = input_z2_limb_7;
        values[8] = input_z2_limb_8;
        values[9] = input_z2_limb_9;
        values[10] = cube_252_output_limb_0;
        values[11] = cube_252_output_limb_1;
        values[12] = cube_252_output_limb_2;
        values[13] = cube_252_output_limb_3;
        values[14] = cube_252_output_limb_4;
        values[15] = cube_252_output_limb_5;
        values[16] = cube_252_output_limb_6;
        values[17] = cube_252_output_limb_7;
        values[18] = cube_252_output_limb_8;
        values[19] = cube_252_output_limb_9;

        m31 values_ext[21];
        values_ext[0] = CUBE_252_RELATION_ID;
        for (int i = 0; i < 20; i++) values_ext[i+1] = values[i];
        cuda_evaluator->add_to_relation<21>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values_ext);
    }

    // Step 2: LinearCombinationN6Coefs4231M11
    // Combines: z0_3 (coef 4), z1 (coef 2), z1_3 (coef 3), z2 (coef 1), cube_output (coef -1), half_key (coef 1)
    linear_combination_n_6_coefs_4_2_3_1_m1_1_evaluate(
        // 60 input limbs for LinearCombination
        input_z0_3_limb_0, input_z0_3_limb_1, input_z0_3_limb_2, input_z0_3_limb_3,
        input_z0_3_limb_4, input_z0_3_limb_5, input_z0_3_limb_6, input_z0_3_limb_7,
        input_z0_3_limb_8, input_z0_3_limb_9,
        input_z1_limb_0, input_z1_limb_1, input_z1_limb_2, input_z1_limb_3,
        input_z1_limb_4, input_z1_limb_5, input_z1_limb_6, input_z1_limb_7,
        input_z1_limb_8, input_z1_limb_9,
        input_z1_3_limb_0, input_z1_3_limb_1, input_z1_3_limb_2, input_z1_3_limb_3,
        input_z1_3_limb_4, input_z1_3_limb_5, input_z1_3_limb_6, input_z1_3_limb_7,
        input_z1_3_limb_8, input_z1_3_limb_9,
        input_z2_limb_0, input_z2_limb_1, input_z2_limb_2, input_z2_limb_3,
        input_z2_limb_4, input_z2_limb_5, input_z2_limb_6, input_z2_limb_7,
        input_z2_limb_8, input_z2_limb_9,
        cube_252_output_limb_0, cube_252_output_limb_1, cube_252_output_limb_2,
        cube_252_output_limb_3, cube_252_output_limb_4, cube_252_output_limb_5,
        cube_252_output_limb_6, cube_252_output_limb_7, cube_252_output_limb_8,
        cube_252_output_limb_9,
        input_half_key_limb_0, input_half_key_limb_1, input_half_key_limb_2,
        input_half_key_limb_3, input_half_key_limb_4, input_half_key_limb_5,
        input_half_key_limb_6, input_half_key_limb_7, input_half_key_limb_8,
        input_half_key_limb_9,
        // Output combination
        combination_limb_0_col10, combination_limb_1_col11, combination_limb_2_col12,
        combination_limb_3_col13, combination_limb_4_col14, combination_limb_5_col15,
        combination_limb_6_col16, combination_limb_7_col17, combination_limb_8_col18,
        combination_limb_9_col19,
        p_coef_col20,
        // Lookup elements
        common_lookup_elements,
        // Evaluator
        cuda_evaluator
    );

    // Step 3: RangeCheckFelt252Width27 lookup (checking first combination)
    {
        m31 values[10];
        values[0] = combination_limb_0_col10;
        values[1] = combination_limb_1_col11;
        values[2] = combination_limb_2_col12;
        values[3] = combination_limb_3_col13;
        values[4] = combination_limb_4_col14;
        values[5] = combination_limb_5_col15;
        values[6] = combination_limb_6_col16;
        values[7] = combination_limb_7_col17;
        values[8] = combination_limb_8_col18;
        values[9] = combination_limb_9_col19;

        m31 values_ext[11];
        values_ext[0] = RANGE_CHECK_252_WIDTH_27_RELATION_ID;
        for (int i = 0; i < 10; i++) values_ext[i+1] = values[i];
        cuda_evaluator->add_to_relation<11>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values_ext);
    }

    // Step 4: LinearCombinationN1Coefs2
    // Computes: 2 * first_combination = second_combination
    linear_combination_n_1_coefs_2_evaluate(
        // Input: first combination
        combination_limb_0_col10, combination_limb_1_col11, combination_limb_2_col12,
        combination_limb_3_col13, combination_limb_4_col14, combination_limb_5_col15,
        combination_limb_6_col16, combination_limb_7_col17, combination_limb_8_col18,
        combination_limb_9_col19,
        // Output: second combination
        combination_limb_0_col21, combination_limb_1_col22, combination_limb_2_col23,
        combination_limb_3_col24, combination_limb_4_col25, combination_limb_5_col26,
        combination_limb_6_col27, combination_limb_7_col28, combination_limb_8_col29,
        combination_limb_9_col30,
        p_coef_col31,
        // Evaluator
        cuda_evaluator
    );
}

#endif // EVALUATE_POSEIDON_PARTIAL_ROUND_H
