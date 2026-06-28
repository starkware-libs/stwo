/*
============================================
LinearCombinationN2Coefs11 CUDA Subroutine
============================================

Subroutine: LinearCombinationN2Coefs11
translated from: cairo-air/src/comptogethernts/subroutines/linear_combination_n_2_coefs_1_1.rs
AIR version: 54d95c0d

Functionality:
- Computes linear combination of 2 inputs with coefficients [1, 1]
- Input: 20 limbs (2 groups of 10 limbs each)
- Output: Verified sum in 10 combination limbs + p_coef
- Formula: input_0 + input_1 = combination + p_coef * p (mod p)

Inputs:
- input_limb_0..19: Two 10-limb numbers (limbs 0-9 and 10-19)
- combination_limb_0..9: Result limbs
- p_coef: Coefficient for modular reduction

Constraint Logic:
- 10 carry computations with modular arithmetic
- 1 final limb constraint
- 10 carry range constraints (cube constraint: carry³ = carry)

Constraint Count: 11 constraints total
============================================
*/

#ifndef EVALUATE_LINEAR_COMBINATION_N_2_COEFS_1_1_H
#define EVALUATE_LINEAR_COMBINATION_N_2_COEFS_1_1_H

#include "fields.cuh"
#include "utils.cuh"
#include "eval_at_row.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void linear_combination_n_2_coefs_1_1_evaluate(
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

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_16 = 16;
    const m31 M31_136 = 136;
    const m31 M31_256 = 256;

    // ===================== Carry Computations =====================
    // carry_0 = ((input_0 + input_10 - combination_0 - p_coef) * 16)
    m31 carry_0 = mul(
        sub(sub(add(input_limb_0, input_limb_10), combination_limb_0), p_coef),
        M31_16
    );

    // carry_1 = ((carry_0 + input_1 + input_11 - combination_1) * 16)
    m31 carry_1 = mul(
        sub(add(add(carry_0, input_limb_1), input_limb_11), combination_limb_1),
        M31_16
    );

    // carry_2 = ((carry_1 + input_2 + input_12 - combination_2) * 16)
    m31 carry_2 = mul(
        sub(add(add(carry_1, input_limb_2), input_limb_12), combination_limb_2),
        M31_16
    );

    // carry_3 = ((carry_2 + input_3 + input_13 - combination_3) * 16)
    m31 carry_3 = mul(
        sub(add(add(carry_2, input_limb_3), input_limb_13), combination_limb_3),
        M31_16
    );

    // carry_4 = ((carry_3 + input_4 + input_14 - combination_4) * 16)
    m31 carry_4 = mul(
        sub(add(add(carry_3, input_limb_4), input_limb_14), combination_limb_4),
        M31_16
    );

    // carry_5 = ((carry_4 + input_5 + input_15 - combination_5) * 16)
    m31 carry_5 = mul(
        sub(add(add(carry_4, input_limb_5), input_limb_15), combination_limb_5),
        M31_16
    );

    // carry_6 = ((carry_5 + input_6 + input_16 - combination_6) * 16)
    m31 carry_6 = mul(
        sub(add(add(carry_5, input_limb_6), input_limb_16), combination_limb_6),
        M31_16
    );

    // carry_7 = ((carry_6 + input_7 + input_17 - combination_7 - p_coef*136) * 16)
    m31 carry_7 = mul(
        sub(sub(add(add(carry_6, input_limb_7), input_limb_17), combination_limb_7),
            mul(p_coef, M31_136)),
        M31_16
    );

    // carry_8 = ((carry_7 + input_8 + input_18 - combination_8) * 16)
    m31 carry_8 = mul(
        sub(add(add(carry_7, input_limb_8), input_limb_18), combination_limb_8),
        M31_16
    );

    // ===================== Final Limb Constraint =====================
    // carry_8 + input_9 + input_19 - combination_9 - p_coef*256 = 0
    cuda_evaluator->add_constraint(
        sub(sub(add(add(carry_8, input_limb_9), input_limb_19), combination_limb_9),
            mul(p_coef, M31_256))
    );

    // ===================== Carry Range Constraints =====================
    // Each carry must satisfy: carry³ = carry (i.e., carry ∈ {-1, 0, 1})
    // biased_carry = (carry + 1) - 1 = carry
    // Constraint: biased_carry³ - biased_carry = 0

    // Carry 0 constraint: p_coef³ = p_coef
    m31 biased_carry_0 = p_coef;  // (p_coef + 1) - 1 = p_coef
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_0, biased_carry_0), biased_carry_0), biased_carry_0)
    );

    // Carry 1 constraint
    m31 biased_carry_1 = carry_0;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_1, biased_carry_1), biased_carry_1), biased_carry_1)
    );

    // Carry 2 constraint
    m31 biased_carry_2 = carry_1;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_2, biased_carry_2), biased_carry_2), biased_carry_2)
    );

    // Carry 3 constraint
    m31 biased_carry_3 = carry_2;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_3, biased_carry_3), biased_carry_3), biased_carry_3)
    );

    // Carry 4 constraint
    m31 biased_carry_4 = carry_3;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_4, biased_carry_4), biased_carry_4), biased_carry_4)
    );

    // Carry 5 constraint
    m31 biased_carry_5 = carry_4;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_5, biased_carry_5), biased_carry_5), biased_carry_5)
    );

    // Carry 6 constraint
    m31 biased_carry_6 = carry_5;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_6, biased_carry_6), biased_carry_6), biased_carry_6)
    );

    // Carry 7 constraint
    m31 biased_carry_7 = carry_6;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_7, biased_carry_7), biased_carry_7), biased_carry_7)
    );

    // Carry 8 constraint
    m31 biased_carry_8 = carry_7;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_8, biased_carry_8), biased_carry_8), biased_carry_8)
    );

    // Carry 9 constraint
    m31 biased_carry_9 = carry_8;
    cuda_evaluator->add_constraint(
        sub(mul(mul(biased_carry_9, biased_carry_9), biased_carry_9), biased_carry_9)
    );
}

#endif // EVALUATE_LINEAR_COMBINATION_N_2_COEFS_1_1_H
