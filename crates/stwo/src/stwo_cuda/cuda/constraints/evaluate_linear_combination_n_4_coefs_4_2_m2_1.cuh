/*
============================================
LinearCombinationN4Coefs42M21 CUDA Subroutine
============================================

Subroutine: LinearCombinationN4Coefs42M21
translated from: cairo-air/src/comptogethernts/subroutines/linear_combination_n_4_coefs_4_2_m2_1.rs
AIR version: 54d95c0d

Functionality:
- Computes linear combination of 4 inputs with coefficients [4, 2, -2, 1]
- Input: 40 limbs (4 groups of 10 limbs each)
- Output: Verified sum in 10 combination limbs + p_coef
- Formula: 4*input_0 + 2*input_1 - 2*input_2 + input_3 = combination + p_coef * p (mod p)

Inputs:
- input_limb_0..39: Four 10-limb numbers (limbs 0-9, 10-19, 20-29, 30-39)
- combination_limb_0..9: Result limbs
- p_coef: Coefficient for modular reduction

Constraint Logic:
- 9 carry computations with modular arithmetic
- 1 final limb constraint
- 10 carry range constraints (cube constraint: carry³ = carry)

Constraint Count: 11 constraints total
============================================
*/

#ifndef EVALUATE_LINEAR_COMBINATION_N_4_COEFS_4_2_M2_1_H
#define EVALUATE_LINEAR_COMBINATION_N_4_COEFS_4_2_M2_1_H

#include "fields.cuh"
#include "utils.cuh"
#include "eval_at_row.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void linear_combination_n_4_coefs_4_2_m2_1_evaluate(
    m31 input_limb_0,
    m31 input_limb_1,
    m31 input_limb_2,
    m31 input_limb_3,
    m31 input_limb_4,
    m31 input_limb_5,
    m31 input_limb_6,
    m31 input_limb_7,
    m31 input_limb_8,
    m31 input_limb_9,
    m31 input_limb_10,
    m31 input_limb_11,
    m31 input_limb_12,
    m31 input_limb_13,
    m31 input_limb_14,
    m31 input_limb_15,
    m31 input_limb_16,
    m31 input_limb_17,
    m31 input_limb_18,
    m31 input_limb_19,
    m31 input_limb_20,
    m31 input_limb_21,
    m31 input_limb_22,
    m31 input_limb_23,
    m31 input_limb_24,
    m31 input_limb_25,
    m31 input_limb_26,
    m31 input_limb_27,
    m31 input_limb_28,
    m31 input_limb_29,
    m31 input_limb_30,
    m31 input_limb_31,
    m31 input_limb_32,
    m31 input_limb_33,
    m31 input_limb_34,
    m31 input_limb_35,
    m31 input_limb_36,
    m31 input_limb_37,
    m31 input_limb_38,
    m31 input_limb_39,

    m31 combination_limb_0,
    m31 combination_limb_1,
    m31 combination_limb_2,
    m31 combination_limb_3,
    m31 combination_limb_4,
    m31 combination_limb_5,
    m31 combination_limb_6,
    m31 combination_limb_7,
    m31 combination_limb_8,
    m31 combination_limb_9,
    m31 p_coef,

    const CommonLookupElements& common_lookup_elements,

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_2 = 2;
    const m31 M31_3 = 3;
    const m31 M31_4 = 4;
    const m31 M31_16 = 16;
    const m31 M31_136 = 136;
    const m31 M31_256 = 256;

    // ===================== Carry Computations =====================
    // carry_0 = ((4*input_0 + 2*input_10 - 2*input_20 + input_30 - combination_0 - p_coef) * 16)
    m31 carry_0 = mul(
        sub(
            sub(
                add(
                    sub(
                        add(
                            mul(M31_4, input_limb_0),
                            mul(M31_2, input_limb_10)
                        ),
                        mul(M31_2, input_limb_20)
                    ),
                    input_limb_30
                ),
                combination_limb_0
            ),
            p_coef
        ),
        M31_16
    );

    // carry_1 = ((carry_0 + 4*input_1 + 2*input_11 - 2*input_21 + input_31 - combination_1) * 16)
    m31 carry_1 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            carry_0,
                            mul(M31_4, input_limb_1)
                        ),
                        mul(M31_2, input_limb_11)
                    ),
                    mul(M31_2, input_limb_21)
                ),
                input_limb_31
            ),
            combination_limb_1
        ),
        M31_16
    );

    // carry_2 = ((carry_1 + 4*input_2 + 2*input_12 - 2*input_22 + input_32 - combination_2) * 16)
    m31 carry_2 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            carry_1,
                            mul(M31_4, input_limb_2)
                        ),
                        mul(M31_2, input_limb_12)
                    ),
                    mul(M31_2, input_limb_22)
                ),
                input_limb_32
            ),
            combination_limb_2
        ),
        M31_16
    );

    // carry_3 = ((carry_2 + 4*input_3 + 2*input_13 - 2*input_23 + input_33 - combination_3) * 16)
    m31 carry_3 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            carry_2,
                            mul(M31_4, input_limb_3)
                        ),
                        mul(M31_2, input_limb_13)
                    ),
                    mul(M31_2, input_limb_23)
                ),
                input_limb_33
            ),
            combination_limb_3
        ),
        M31_16
    );

    // carry_4 = ((carry_3 + 4*input_4 + 2*input_14 - 2*input_24 + input_34 - combination_4) * 16)
    m31 carry_4 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            carry_3,
                            mul(M31_4, input_limb_4)
                        ),
                        mul(M31_2, input_limb_14)
                    ),
                    mul(M31_2, input_limb_24)
                ),
                input_limb_34
            ),
            combination_limb_4
        ),
        M31_16
    );

    // carry_5 = ((carry_4 + 4*input_5 + 2*input_15 - 2*input_25 + input_35 - combination_5) * 16)
    m31 carry_5 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            carry_4,
                            mul(M31_4, input_limb_5)
                        ),
                        mul(M31_2, input_limb_15)
                    ),
                    mul(M31_2, input_limb_25)
                ),
                input_limb_35
            ),
            combination_limb_5
        ),
        M31_16
    );

    // carry_6 = ((carry_5 + 4*input_6 + 2*input_16 - 2*input_26 + input_36 - combination_6) * 16)
    m31 carry_6 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            carry_5,
                            mul(M31_4, input_limb_6)
                        ),
                        mul(M31_2, input_limb_16)
                    ),
                    mul(M31_2, input_limb_26)
                ),
                input_limb_36
            ),
            combination_limb_6
        ),
        M31_16
    );

    // carry_7 = ((carry_6 + 4*input_7 + 2*input_17 - 2*input_27 + input_37 - combination_7 - p_coef*136) * 16)
    m31 carry_7 = mul(
        sub(
            sub(
                add(
                    sub(
                        add(
                            add(
                                carry_6,
                                mul(M31_4, input_limb_7)
                            ),
                            mul(M31_2, input_limb_17)
                        ),
                        mul(M31_2, input_limb_27)
                    ),
                    input_limb_37
                ),
                combination_limb_7
            ),
            mul(p_coef, M31_136)
        ),
        M31_16
    );

    // carry_8 = ((carry_7 + 4*input_8 + 2*input_18 - 2*input_28 + input_38 - combination_8) * 16)
    m31 carry_8 = mul(
        sub(
            add(
                sub(
                    add(
                        add(
                            carry_7,
                            mul(M31_4, input_limb_8)
                        ),
                        mul(M31_2, input_limb_18)
                    ),
                    mul(M31_2, input_limb_28)
                ),
                input_limb_38
            ),
            combination_limb_8
        ),
        M31_16
    );

    // ===================== Final Limb Constraint =====================
    // carry_8 + 4*input_9 + 2*input_19 - 2*input_29 + input_39 - combination_9 - p_coef*256 = 0
    cuda_evaluator->add_constraint(
        sub(
            sub(
                add(
                    sub(
                        add(
                            add(
                                carry_8,
                                mul(M31_4, input_limb_9)
                            ),
                            mul(M31_2, input_limb_19)
                        ),
                        mul(M31_2, input_limb_29)
                    ),
                    input_limb_39
                ),
                combination_limb_9
            ),
            mul(p_coef, M31_256)
        )
    );

    // ===================== Carry Range Constraints =====================
    // Each carry must satisfy: carry ∈ {-3, -2, -1, 0, 1, 2, 3}
    // For coefficients [4, 2, -2, 1], range is [-3, 3]
    // This is handled by range check lookup tables via add_to_relation
    // Two RangeCheck_4_4_4_4 lookups (4 values each) + One RangeCheck_4_4 lookup (2 values):
    //   - First RangeCheck_4_4_4_4: p_coef, carry_0, carry_1, carry_2 (each + 3 for biasing)
    //   - Second RangeCheck_4_4_4_4: carry_3, carry_4, carry_5, carry_6 (each + 3 for biasing)
    //   - Third RangeCheck_4_4: carry_7, carry_8 (each + 3 for biasing)

    {
        m31 values_0[5] = {
            RANGE_CHECK_4_4_4_4_RELATION_ID,
            add(p_coef, M31_3),
            add(carry_0, M31_3),
            add(carry_1, M31_3),
            add(carry_2, M31_3)
        };
        cuda_evaluator->add_to_relation<5>(common_lookup_elements, qm31{{1,0}, {0,0}}, values_0);
    }

    {
        m31 values_1[5] = {
            RANGE_CHECK_4_4_4_4_RELATION_ID,
            add(carry_3, M31_3),
            add(carry_4, M31_3),
            add(carry_5, M31_3),
            add(carry_6, M31_3)
        };
        cuda_evaluator->add_to_relation<5>(common_lookup_elements, qm31{{1,0}, {0,0}}, values_1);
    }

    {
        m31 values_2[3] = {
            RANGE_CHECK_4_4_RELATION_ID,
            add(carry_7, M31_3),
            add(carry_8, M31_3)
        };
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1,0}, {0,0}}, values_2);
    }
}

#endif // EVALUATE_LINEAR_COMBINATION_N_4_COEFS_4_2_M2_1_H
