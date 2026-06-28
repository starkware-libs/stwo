/*
============================================
LinearCombinationN4Coefs1M111 CUDA Subroutine
============================================

Subroutine: LinearCombinationN4Coefs1M111
translated from: cairo-air/src/comptogethernts/subroutines/linear_combination_n_4_coefs_1_m1_1_1.rs
AIR version: 54d95c0d

Functionality:
- Computes linear combination of 4 inputs with coefficients [1, -1, 1, 1]
- Input: 40 limbs (4 groups of 10 limbs each)
- Output: Verified sum in 10 combination limbs + p_coef
- Formula: input_0 - input_1 + input_2 + input_3 = combination + p_coef * p (mod p)

Inputs:
- input_limb_0..39: Four 10-limb numbers (limbs 0-9, 10-19, 20-29, 30-39)
- combination_limb_0..9: Result limbs
- p_coef: Coefficient for modular reduction

Constraint Logic:
- 9 carry computations with modular arithmetic
- 1 final limb constraint
- 2 RangeCheck_3_3_3_3_3 lookups (for carry range checks)

Relation Lookups:
- RangeCheck_3_3_3_3_3: 2 uses (10 total carries checked in groups of 5)

Coefficients [1, -1, 1, 1]:
- limb_i = input[i] - input[i+10] + input[i+20] + input[i+30]
============================================
*/

#ifndef EVALUATE_LINEAR_COMBINATION_N_4_COEFS_1_M1_1_1_H
#define EVALUATE_LINEAR_COMBINATION_N_4_COEFS_1_M1_1_1_H

#include "fields.cuh"
#include "utils.cuh"
#include "eval_at_row.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void linear_combination_n_4_coefs_1_m1_1_1_evaluate(
    // 40 input limbs (4 numbers × 10 limbs each)
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
    const m31 M31_1 = m31(1);
    const m31 M31_2 = m31(2);
    const m31 M31_16 = m31(16);
    const m31 M31_136 = m31(136);
    const m31 M31_256 = m31(256);

    // Carry computations
    // carry_i = ((input[i] - input[i+10] + input[i+20] + input[i+30] - combination[i] - p_adjustment) * 16)

    // carry_0: includes p_coef term
    m31 carry_0 = mul(
        sub(
            sub(
                sub(
                    add(
                        add(
                            sub(
                                input_limb_0,
                                input_limb_10
                            ),
                            input_limb_20
                        ),
                        input_limb_30
                    ),
                    combination_limb_0
                ),
                p_coef
            ),
            m31(0)  // No additional p term for limb 0
        ),
        M31_16
    );

    // carry_1
    m31 carry_1 = mul(
        sub(
            add(
                add(
                    add(
                        sub(
                            carry_0,
                            input_limb_11
                        ),
                        input_limb_1
                    ),
                    input_limb_21
                ),
                input_limb_31
            ),
            combination_limb_1
        ),
        M31_16
    );

    // carry_2
    m31 carry_2 = mul(
        sub(
            add(
                add(
                    add(
                        sub(
                            carry_1,
                            input_limb_12
                        ),
                        input_limb_2
                    ),
                    input_limb_22
                ),
                input_limb_32
            ),
            combination_limb_2
        ),
        M31_16
    );

    // carry_3
    m31 carry_3 = mul(
        sub(
            add(
                add(
                    add(
                        sub(
                            carry_2,
                            input_limb_13
                        ),
                        input_limb_3
                    ),
                    input_limb_23
                ),
                input_limb_33
            ),
            combination_limb_3
        ),
        M31_16
    );

    // carry_4
    m31 carry_4 = mul(
        sub(
            add(
                add(
                    add(
                        sub(
                            carry_3,
                            input_limb_14
                        ),
                        input_limb_4
                    ),
                    input_limb_24
                ),
                input_limb_34
            ),
            combination_limb_4
        ),
        M31_16
    );

    // carry_5
    m31 carry_5 = mul(
        sub(
            add(
                add(
                    add(
                        sub(
                            carry_4,
                            input_limb_15
                        ),
                        input_limb_5
                    ),
                    input_limb_25
                ),
                input_limb_35
            ),
            combination_limb_5
        ),
        M31_16
    );

    // carry_6
    m31 carry_6 = mul(
        sub(
            add(
                add(
                    add(
                        sub(
                            carry_5,
                            input_limb_16
                        ),
                        input_limb_6
                    ),
                    input_limb_26
                ),
                input_limb_36
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
                    add(
                        add(
                            sub(
                                carry_6,
                                input_limb_17
                            ),
                            input_limb_7
                        ),
                        input_limb_27
                    ),
                    input_limb_37
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
                add(
                    add(
                        sub(
                            carry_7,
                            input_limb_18
                        ),
                        input_limb_8
                    ),
                    input_limb_28
                ),
                input_limb_38
            ),
            combination_limb_8
        ),
        M31_16
    );

    // Final limb constraint (carry_9 must be zero)
    // carry_9 = (carry_8 + input[9] - input[19] + input[29] + input[39] - combination[9] - p_coef * 256)
    m31 final_limb_constraint = sub(
        sub(
            add(
                add(
                    add(
                        sub(
                            carry_8,
                            input_limb_19
                        ),
                        input_limb_9
                    ),
                    input_limb_29
                ),
                input_limb_39
            ),
            combination_limb_9
        ),
        mul(p_coef, M31_256)
    );

    cuda_evaluator->add_constraint(final_limb_constraint);

    // RangeCheck_3_3_3_3_3 lookups for carry values
    // Rust uses M31_2 offset for ALL carries (verified from linear_combination_n_4_coefs_1_m1_1_1.rs)
    // First lookup: p_coef, carry_0, carry_1, carry_2, carry_3
    {
        m31 values[6] = {
            RANGE_CHECK_3_3_3_3_3_RELATION_ID,
            add(p_coef, M31_2),
            add(carry_0, M31_2),
            add(carry_1, M31_2),
            add(carry_2, M31_2),
            add(carry_3, M31_2)
        };
        cuda_evaluator->add_to_relation<6>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Second lookup: carry_4, carry_5, carry_6, carry_7, carry_8
    {
        m31 values[6] = {
            RANGE_CHECK_3_3_3_3_3_RELATION_ID,
            add(carry_4, M31_2),
            add(carry_5, M31_2),
            add(carry_6, M31_2),
            add(carry_7, M31_2),
            add(carry_8, M31_2)
        };
        cuda_evaluator->add_to_relation<6>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }
}

#endif // EVALUATE_LINEAR_COMBINATION_N_4_COEFS_1_M1_1_1_H
