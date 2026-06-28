/*
============================================
LinearCombinationN1Coefs2 CUDA Subroutine
============================================

Subroutine: LinearCombinationN1Coefs2
translated from: cairo-air/src/comptogethernts/subroutines/linear_combination_n_1_coefs_2.rs
AIR version: 54d95c0d

Functionality:
- Computes linear combination of 1 input with coefficient [2]
- Input: 10 limbs (1 number of 10 limbs)
- Output: Verified result in 10 combination limbs + p_coef
- Formula: 2*input_0 = combination + p_coef * p (mod p)

Inputs:
- input_limb_0..9: One 10-limb number
- combination_limb_0..9: Result limbs
- p_coef: Coefficient for modular reduction

Constraint Logic:
- 9 carry computations with modular arithmetic
- 1 final limb constraint
- 10 carry constraints (cubic: carry^3 - carry = 0)

Relation Lookups:
- None (uses algebraic constraints instead)

Coefficients [2]:
- limb_i = 2*input[i]

Carry Constraints:
- Each carry must satisfy: carry^3 - carry = 0
- This ensures carry ∈ {-1, 0, 1}

============================================
*/

#ifndef EVALUATE_LINEAR_COMBINATION_N_1_COEFS_2_H
#define EVALUATE_LINEAR_COMBINATION_N_1_COEFS_2_H

#include "fields.cuh"
#include "utils.cuh"
#include "eval_at_row.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void linear_combination_n_1_coefs_2_evaluate(
    // 10 input limbs (1 number × 10 limbs)
    m31 input_limb_0, m31 input_limb_1, m31 input_limb_2, m31 input_limb_3,
    m31 input_limb_4, m31 input_limb_5, m31 input_limb_6, m31 input_limb_7,
    m31 input_limb_8, m31 input_limb_9,

    // 10 combination result limbs
    m31 combination_limb_0, m31 combination_limb_1, m31 combination_limb_2,
    m31 combination_limb_3, m31 combination_limb_4, m31 combination_limb_5,
    m31 combination_limb_6, m31 combination_limb_7, m31 combination_limb_8,
    m31 combination_limb_9,

    // Modular reduction coefficient
    m31 p_coef,

    // Evaluator
    EvaluatorT *cuda_evaluator
) {
    // Constants
    const m31 M31_2 = m31(2);
    const m31 M31_16 = m31(16);
    const m31 M31_136 = m31(136);
    const m31 M31_256 = m31(256);

    // Carry computations
    // carry_i = ((2*input[i] - combination[i] - p_adjustment) * 16)

    // carry_0: includes p_coef term
    m31 carry_0 = mul(
        sub(
            sub(
                sub(
                    mul(M31_2, input_limb_0),
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
                carry_0,
                mul(M31_2, input_limb_1)
            ),
            combination_limb_1
        ),
        M31_16
    );

    // carry_2
    m31 carry_2 = mul(
        sub(
            add(
                carry_1,
                mul(M31_2, input_limb_2)
            ),
            combination_limb_2
        ),
        M31_16
    );

    // carry_3
    m31 carry_3 = mul(
        sub(
            add(
                carry_2,
                mul(M31_2, input_limb_3)
            ),
            combination_limb_3
        ),
        M31_16
    );

    // carry_4
    m31 carry_4 = mul(
        sub(
            add(
                carry_3,
                mul(M31_2, input_limb_4)
            ),
            combination_limb_4
        ),
        M31_16
    );

    // carry_5
    m31 carry_5 = mul(
        sub(
            add(
                carry_4,
                mul(M31_2, input_limb_5)
            ),
            combination_limb_5
        ),
        M31_16
    );

    // carry_6
    m31 carry_6 = mul(
        sub(
            add(
                carry_5,
                mul(M31_2, input_limb_6)
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
                    carry_6,
                    mul(M31_2, input_limb_7)
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
                carry_7,
                mul(M31_2, input_limb_8)
            ),
            combination_limb_8
        ),
        M31_16
    );

    // Final limb constraint (carry_9 must be zero)
    // carry_9 = (carry_8 + 2*input[9] - combination[9] - p_coef * 256)
    m31 final_limb_constraint = sub(
        sub(
            add(
                carry_8,
                mul(M31_2, input_limb_9)
            ),
            combination_limb_9
        ),
        mul(p_coef, M31_256)
    );

    cuda_evaluator->add_constraint(final_limb_constraint);

    // Carry constraints: carry^3 - carry = 0
    // This ensures each carry ∈ {-1, 0, 1}

    // Note: CPU code has biased_carry = (carry + 1) - 1 = carry
    // So the constraint is just carry^3 - carry = 0

    // Carry constraint for p_coef
    {
        m31 biased_p_coef = p_coef;  // (p_coef + 1) - 1 cancels out
        m31 p_coef_cubed = mul(mul(biased_p_coef, biased_p_coef), biased_p_coef);
        m31 p_coef_constraint = sub(p_coef_cubed, biased_p_coef);
        cuda_evaluator->add_constraint(p_coef_constraint);
    }

    // Carry constraint 0
    {
        m31 biased_carry_0 = carry_0;
        m31 carry_0_cubed = mul(mul(biased_carry_0, biased_carry_0), biased_carry_0);
        m31 carry_0_constraint = sub(carry_0_cubed, biased_carry_0);
        cuda_evaluator->add_constraint(carry_0_constraint);
    }

    // Carry constraint 1
    {
        m31 biased_carry_1 = carry_1;
        m31 carry_1_cubed = mul(mul(biased_carry_1, biased_carry_1), biased_carry_1);
        m31 carry_1_constraint = sub(carry_1_cubed, biased_carry_1);
        cuda_evaluator->add_constraint(carry_1_constraint);
    }

    // Carry constraint 2
    {
        m31 biased_carry_2 = carry_2;
        m31 carry_2_cubed = mul(mul(biased_carry_2, biased_carry_2), biased_carry_2);
        m31 carry_2_constraint = sub(carry_2_cubed, biased_carry_2);
        cuda_evaluator->add_constraint(carry_2_constraint);
    }

    // Carry constraint 3
    {
        m31 biased_carry_3 = carry_3;
        m31 carry_3_cubed = mul(mul(biased_carry_3, biased_carry_3), biased_carry_3);
        m31 carry_3_constraint = sub(carry_3_cubed, biased_carry_3);
        cuda_evaluator->add_constraint(carry_3_constraint);
    }

    // Carry constraint 4
    {
        m31 biased_carry_4 = carry_4;
        m31 carry_4_cubed = mul(mul(biased_carry_4, biased_carry_4), biased_carry_4);
        m31 carry_4_constraint = sub(carry_4_cubed, biased_carry_4);
        cuda_evaluator->add_constraint(carry_4_constraint);
    }

    // Carry constraint 5
    {
        m31 biased_carry_5 = carry_5;
        m31 carry_5_cubed = mul(mul(biased_carry_5, biased_carry_5), biased_carry_5);
        m31 carry_5_constraint = sub(carry_5_cubed, biased_carry_5);
        cuda_evaluator->add_constraint(carry_5_constraint);
    }

    // Carry constraint 6
    {
        m31 biased_carry_6 = carry_6;
        m31 carry_6_cubed = mul(mul(biased_carry_6, biased_carry_6), biased_carry_6);
        m31 carry_6_constraint = sub(carry_6_cubed, biased_carry_6);
        cuda_evaluator->add_constraint(carry_6_constraint);
    }

    // Carry constraint 7
    {
        m31 biased_carry_7 = carry_7;
        m31 carry_7_cubed = mul(mul(biased_carry_7, biased_carry_7), biased_carry_7);
        m31 carry_7_constraint = sub(carry_7_cubed, biased_carry_7);
        cuda_evaluator->add_constraint(carry_7_constraint);
    }

    // Carry constraint 8
    {
        m31 biased_carry_8 = carry_8;
        m31 carry_8_cubed = mul(mul(biased_carry_8, biased_carry_8), biased_carry_8);
        m31 carry_8_constraint = sub(carry_8_cubed, biased_carry_8);
        cuda_evaluator->add_constraint(carry_8_constraint);
    }
}

#endif // EVALUATE_LINEAR_COMBINATION_N_1_COEFS_2_H
