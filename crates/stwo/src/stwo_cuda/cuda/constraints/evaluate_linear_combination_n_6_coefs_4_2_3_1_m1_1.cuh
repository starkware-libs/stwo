/*
============================================
LinearCombinationN6Coefs4231M11 CUDA Subroutine
============================================

Subroutine: LinearCombinationN6Coefs4231M11
translated from: cairo-air/src/comptogethernts/subroutines/linear_combination_n_6_coefs_4_2_3_1_m1_1.rs
AIR version: 54d95c0d

Functionality:
- Computes linear combination of 6 inputs with coefficients [4, 2, 3, 1, -1, 1]
- Input: 60 limbs (6 groups of 10 limbs each)
- Output: Verified sum in 10 combination limbs + p_coef
- Formula: 4*input_0 + 2*input_1 + 3*input_2 + input_3 - input_4 + input_5 = combination + p_coef * p (mod p)

Inputs:
- input_limb_0..59: Six 10-limb numbers (limbs 0-9, 10-19, 20-29, 30-39, 40-49, 50-59)
- combination_limb_0..9: Result limbs
- p_coef: Coefficient for modular reduction

Constraint Logic:
- 9 carry computations with modular arithmetic
- 1 final limb constraint
- 3 RangeCheck lookups (2× RangeCheck_4_4_4_4 + 1× RangeCheck_4_4)

Relation Lookups:
- RangeCheck_4_4_4_4: 2 uses (checking p_coef, carry_0..6 in groups of 4)
- RangeCheck_4_4: 1 use (checking carry_7..8)

Coefficients [4, 2, 3, 1, -1, 1]:
- limb_i = 4*input[i] + 2*input[i+10] + 3*input[i+20] + input[i+30] - input[i+40] + input[i+50]

Note: Offset is +2 (not +1) for all range check lookups
============================================
*/

#ifndef EVALUATE_LINEAR_COMBINATION_N_6_COEFS_4_2_3_1_M1_1_H
#define EVALUATE_LINEAR_COMBINATION_N_6_COEFS_4_2_3_1_M1_1_H

#include "fields.cuh"
#include "utils.cuh"
#include "eval_at_row.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void linear_combination_n_6_coefs_4_2_3_1_m1_1_evaluate(
    // 60 input limbs (6 numbers × 10 limbs each)
    m31 input_limb_0, m31 input_limb_1, m31 input_limb_2, m31 input_limb_3,
    m31 input_limb_4, m31 input_limb_5, m31 input_limb_6, m31 input_limb_7,
    m31 input_limb_8, m31 input_limb_9,
    m31 input_limb_10, m31 input_limb_11, m31 input_limb_12, m31 input_limb_13,
    m31 input_limb_14, m31 input_limb_15, m31 input_limb_16, m31 input_limb_17,
    m31 input_limb_18, m31 input_limb_19,
    m31 input_limb_20, m31 input_limb_21, m31 input_limb_22, m31 input_limb_23,
    m31 input_limb_24, m31 input_limb_25, m31 input_limb_26, m31 input_limb_27,
    m31 input_limb_28, m31 input_limb_29,
    m31 input_limb_30, m31 input_limb_31, m31 input_limb_32, m31 input_limb_33,
    m31 input_limb_34, m31 input_limb_35, m31 input_limb_36, m31 input_limb_37,
    m31 input_limb_38, m31 input_limb_39,
    m31 input_limb_40, m31 input_limb_41, m31 input_limb_42, m31 input_limb_43,
    m31 input_limb_44, m31 input_limb_45, m31 input_limb_46, m31 input_limb_47,
    m31 input_limb_48, m31 input_limb_49,
    m31 input_limb_50, m31 input_limb_51, m31 input_limb_52, m31 input_limb_53,
    m31 input_limb_54, m31 input_limb_55, m31 input_limb_56, m31 input_limb_57,
    m31 input_limb_58, m31 input_limb_59,

    // 10 combination result limbs
    m31 combination_limb_0, m31 combination_limb_1, m31 combination_limb_2,
    m31 combination_limb_3, m31 combination_limb_4, m31 combination_limb_5,
    m31 combination_limb_6, m31 combination_limb_7, m31 combination_limb_8,
    m31 combination_limb_9,

    // Modular reduction coefficient
    m31 p_coef,

    // Lookup elements
    const CommonLookupElements& common_lookup_elements,

    // Evaluator
    EvaluatorT *cuda_evaluator
) {
    // Constants
    const m31 M31_2 = m31(2);
    const m31 M31_3 = m31(3);
    const m31 M31_4 = m31(4);
    const m31 M31_16 = m31(16);
    const m31 M31_136 = m31(136);
    const m31 M31_256 = m31(256);

    // Carry computations
    // carry_i = ((4*input[i] + 2*input[i+10] + 3*input[i+20] + input[i+30] - input[i+40] + input[i+50] - combination[i] - p_adjustment) * 16)

    // carry_0: includes p_coef term
    m31 carry_0 = mul(
        sub(
            sub(
                add(
                    sub(
                        add(
                            add(
                                add(
                                    mul(M31_4, input_limb_0),
                                    mul(M31_2, input_limb_10)
                                ),
                                mul(M31_3, input_limb_20)
                            ),
                            input_limb_30
                        ),
                        input_limb_40
                    ),
                    input_limb_50
                ),
                combination_limb_0
            ),
            p_coef
        ),
        M31_16
    );

    // carry_1
    m31 carry_1 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_0,
                                    mul(M31_4, input_limb_1)
                                ),
                                mul(M31_2, input_limb_11)
                            ),
                            mul(M31_3, input_limb_21)
                        ),
                        input_limb_31
                    ),
                    input_limb_41
                ),
                input_limb_51
            ),
            combination_limb_1
        ),
        M31_16
    );

    // carry_2
    m31 carry_2 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_1,
                                    mul(M31_4, input_limb_2)
                                ),
                                mul(M31_2, input_limb_12)
                            ),
                            mul(M31_3, input_limb_22)
                        ),
                        input_limb_32
                    ),
                    input_limb_42
                ),
                input_limb_52
            ),
            combination_limb_2
        ),
        M31_16
    );

    // carry_3
    m31 carry_3 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_2,
                                    mul(M31_4, input_limb_3)
                                ),
                                mul(M31_2, input_limb_13)
                            ),
                            mul(M31_3, input_limb_23)
                        ),
                        input_limb_33
                    ),
                    input_limb_43
                ),
                input_limb_53
            ),
            combination_limb_3
        ),
        M31_16
    );

    // carry_4
    m31 carry_4 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_3,
                                    mul(M31_4, input_limb_4)
                                ),
                                mul(M31_2, input_limb_14)
                            ),
                            mul(M31_3, input_limb_24)
                        ),
                        input_limb_34
                    ),
                    input_limb_44
                ),
                input_limb_54
            ),
            combination_limb_4
        ),
        M31_16
    );

    // carry_5
    m31 carry_5 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_4,
                                    mul(M31_4, input_limb_5)
                                ),
                                mul(M31_2, input_limb_15)
                            ),
                            mul(M31_3, input_limb_25)
                        ),
                        input_limb_35
                    ),
                    input_limb_45
                ),
                input_limb_55
            ),
            combination_limb_5
        ),
        M31_16
    );

    // carry_6
    m31 carry_6 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_5,
                                    mul(M31_4, input_limb_6)
                                ),
                                mul(M31_2, input_limb_16)
                            ),
                            mul(M31_3, input_limb_26)
                        ),
                        input_limb_36
                    ),
                    input_limb_46
                ),
                input_limb_56
            ),
            combination_limb_6
        ),
        M31_16
    );

    // carry_7: includes p_coef * 136 term (for limb 7)
    m31 carry_7 = mul(
        sub(
            sub(
                add(
                    sub(
                        add(
                            add(
                                add(
                                    add(
                                        carry_6,
                                        mul(M31_4, input_limb_7)
                                    ),
                                    mul(M31_2, input_limb_17)
                                ),
                                mul(M31_3, input_limb_27)
                            ),
                            input_limb_37
                        ),
                        input_limb_47
                    ),
                    input_limb_57
                ),
                combination_limb_7
            ),
            mul(p_coef, M31_136)
        ),
        M31_16
    );

    // carry_8
    m31 carry_8 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_7,
                                    mul(M31_4, input_limb_8)
                                ),
                                mul(M31_2, input_limb_18)
                            ),
                            mul(M31_3, input_limb_28)
                        ),
                        input_limb_38
                    ),
                    input_limb_48
                ),
                input_limb_58
            ),
            combination_limb_8
        ),
        M31_16
    );

    // Final limb constraint (carry_9 must be zero)
    // carry_9 = (carry_8 + 4*input[9] + 2*input[19] + 3*input[29] + input[39] - input[49] + input[59] - combination[9] - p_coef * 256)
    m31 final_limb_constraint = sub(
        sub(
            add(
                sub(
                    add(
                        add(
                            add(
                                add(
                                    carry_8,
                                    mul(M31_4, input_limb_9)
                                ),
                                mul(M31_2, input_limb_19)
                            ),
                            mul(M31_3, input_limb_29)
                        ),
                        input_limb_39
                    ),
                    input_limb_49
                ),
                input_limb_59
            ),
            combination_limb_9
        ),
        mul(p_coef, M31_256)
    );

    cuda_evaluator->add_constraint(final_limb_constraint);

    // RangeCheck_4_4_4_4 lookups for carry values (offset +2)
    // First lookup: p_coef, carry_0, carry_1, carry_2
    {
        m31 values[5] = {RANGE_CHECK_4_4_4_4_RELATION_ID, add(p_coef, M31_2), add(carry_0, M31_2), add(carry_1, M31_2), add(carry_2, M31_2)};
        cuda_evaluator->template add_to_relation<5>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Second lookup: carry_3, carry_4, carry_5, carry_6
    {
        m31 values[5] = {RANGE_CHECK_4_4_4_4_RELATION_ID, add(carry_3, M31_2), add(carry_4, M31_2), add(carry_5, M31_2), add(carry_6, M31_2)};
        cuda_evaluator->template add_to_relation<5>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // RangeCheck_4_4 lookup for carry_7, carry_8
    {
        m31 values[3] = {RANGE_CHECK_4_4_RELATION_ID, add(carry_7, M31_2), add(carry_8, M31_2)};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }
}

#endif // EVALUATE_LINEAR_COMBINATION_N_6_COEFS_4_2_3_1_M1_1_H
