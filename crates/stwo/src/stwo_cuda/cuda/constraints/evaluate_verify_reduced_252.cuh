/*
============================================
VerifyReduced252 CUDA Subroutine
============================================

Subroutine: VerifyReduced252
translated from: cairo-air/src/comptogethernts/subroutines/verify_reduced_252.rs
AIR version: 54d95c0d

Functionality:
- Verifies that a 252-bit value is reduced (canonicalized) to the field's range
- Ensures the value is strictly less than the prime p (252-bit field modulus)
- Uses conditional constraints based on the most significant limbs

Inputs:
- verify_reduced_252_input_limb_0..27: 28 limbs of the 252-bit value (9 bits each)
- ms_limb_is_max: Boolean flag (1 if limb_27 is at maximum value)
- ms_and_mid_limbs_are_max: Boolean flag (1 if both MS and mid limbs are max)
- rc_input: Range check input value

Constraint Logic:
1. ms_limb_is_max is boolean: ms_limb_is_max * (1 - ms_limb_is_max) = 0
2. ms_and_mid_limbs_are_max is boolean: ms_and_mid_limbs_are_max * (1 - ms_and_mid_limbs_are_max) = 0
3. RangeCheck_8(limb_27 - ms_limb_is_max): Verify limb_27 < max if not flagged
4. If MS limb is max (ms_limb_is_max=1), then limbs 22-26 must be 0
5. rc_input = ms_limb_is_max * (120 + limb_21 - ms_and_mid_limbs_are_max)
6. RangeCheck_8(rc_input): Verify the computed range check input
7. If both MS and mid limbs are max (ms_and_mid_limbs_are_max=1), then limbs 0-20 must be 0

Relations:
- RangeCheck_8: 2 uses
  * (limb_27 - ms_limb_is_max)
  * rc_input

Constraint Count:
- 2 boolean constraints
- 1 summed conditional zero constraint (if MS is max, sum of limbs 22-26 = 0)
- 1 rc_input constraint
- 1 summed conditional zero constraint (if both MS and mid are max, sum of limbs 0-20 = 0)
- Total: 5 constraints

Key Algorithm:
This ensures that the 252-bit number is properly reduced modulo the field prime.
The constraints verify that the value doesn't exceed the field size by checking
the most significant limbs and ensuring proper bounds.
============================================
*/

#ifndef EVALUATE_VERIFY_REDUCED_252_CONSTRAINT_H
#define EVALUATE_VERIFY_REDUCED_252_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void verify_reduced_252_evaluate(
    m31 verify_reduced_252_input_limb_0,
    m31 verify_reduced_252_input_limb_1,
    m31 verify_reduced_252_input_limb_2,
    m31 verify_reduced_252_input_limb_3,
    m31 verify_reduced_252_input_limb_4,
    m31 verify_reduced_252_input_limb_5,
    m31 verify_reduced_252_input_limb_6,
    m31 verify_reduced_252_input_limb_7,
    m31 verify_reduced_252_input_limb_8,
    m31 verify_reduced_252_input_limb_9,
    m31 verify_reduced_252_input_limb_10,
    m31 verify_reduced_252_input_limb_11,
    m31 verify_reduced_252_input_limb_12,
    m31 verify_reduced_252_input_limb_13,
    m31 verify_reduced_252_input_limb_14,
    m31 verify_reduced_252_input_limb_15,
    m31 verify_reduced_252_input_limb_16,
    m31 verify_reduced_252_input_limb_17,
    m31 verify_reduced_252_input_limb_18,
    m31 verify_reduced_252_input_limb_19,
    m31 verify_reduced_252_input_limb_20,
    m31 verify_reduced_252_input_limb_21,
    m31 verify_reduced_252_input_limb_22,
    m31 verify_reduced_252_input_limb_23,
    m31 verify_reduced_252_input_limb_24,
    m31 verify_reduced_252_input_limb_25,
    m31 verify_reduced_252_input_limb_26,
    m31 verify_reduced_252_input_limb_27,

    m31 ms_limb_is_max_col0,
    m31 ms_and_mid_limbs_are_max_col1,
    m31 rc_input_col2,

    const CommonLookupElements& common_lookup_elements,

    EvaluatorT *cuda_evaluator
) {
    const m31 M31_1 = 1;
    const m31 M31_120 = 120;

    // ===================== Boolean Constraint 1: ms_limb_is_max is bit =====================
    // Constraint: ms_limb_is_max * (1 - ms_limb_is_max) = 0
    // This ensures ms_limb_is_max ∈ {0, 1}
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0, sub(M31_1, ms_limb_is_max_col0))
    );

    // ===================== Boolean Constraint 2: ms_and_mid_limbs_are_max is bit =====================
    // Constraint: ms_and_mid_limbs_are_max * (1 - ms_and_mid_limbs_are_max) = 0
    // This ensures ms_and_mid_limbs_are_max ∈ {0, 1}
    cuda_evaluator->add_constraint(
        mul(ms_and_mid_limbs_are_max_col1, sub(M31_1, ms_and_mid_limbs_are_max_col1))
    );

    // ===================== RangeCheck_8: limb_27 - ms_limb_is_max =====================
    // Verify that limb_27 is within range when not flagged as max
    m31 rc_value_0[2] = {
        RANGE_CHECK_8_RELATION_ID,
        sub(verify_reduced_252_input_limb_27, ms_limb_is_max_col0)
    };
    cuda_evaluator->add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc_value_0);

    // ===================== If MS limb is max, high limbs (22-26) must be 0 =====================
    // Constraint: ms_limb_is_max * (limb_22 + limb_23 + limb_24 + limb_25 + limb_26) = 0
    cuda_evaluator->add_constraint(
        mul(ms_limb_is_max_col0,
            add(add(add(add(verify_reduced_252_input_limb_22,
                            verify_reduced_252_input_limb_23),
                        verify_reduced_252_input_limb_24),
                    verify_reduced_252_input_limb_25),
                verify_reduced_252_input_limb_26))
    );

    // ===================== rc_input constraint =====================
    // rc_input = ms_limb_is_max * (120 + limb_21 - ms_and_mid_limbs_are_max)
    // Constraint: rc_input - ms_limb_is_max * (120 + limb_21 - ms_and_mid_limbs_are_max) = 0
    m31 expected_rc_input = mul(
        ms_limb_is_max_col0,
        sub(add(M31_120, verify_reduced_252_input_limb_21), ms_and_mid_limbs_are_max_col1)
    );
    cuda_evaluator->add_constraint(
        sub(rc_input_col2, expected_rc_input)
    );

    // ===================== RangeCheck_8: rc_input =====================
    m31 rc_value_1[2] = {RANGE_CHECK_8_RELATION_ID, rc_input_col2};
    cuda_evaluator->add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc_value_1);

    // ===================== If MS and mid limbs are max, low limbs (0-20) must be 0 =====================
    // Constraint: ms_and_mid_limbs_are_max * (limb_0 + limb_1 + ... + limb_20) = 0
    {
        m31 sum_low = add(add(add(add(add(add(add(add(add(add(
                        add(add(add(add(add(add(add(add(add(add(
                            verify_reduced_252_input_limb_0,
                            verify_reduced_252_input_limb_1),
                            verify_reduced_252_input_limb_2),
                            verify_reduced_252_input_limb_3),
                            verify_reduced_252_input_limb_4),
                            verify_reduced_252_input_limb_5),
                            verify_reduced_252_input_limb_6),
                            verify_reduced_252_input_limb_7),
                            verify_reduced_252_input_limb_8),
                            verify_reduced_252_input_limb_9),
                            verify_reduced_252_input_limb_10),
                            verify_reduced_252_input_limb_11),
                            verify_reduced_252_input_limb_12),
                            verify_reduced_252_input_limb_13),
                            verify_reduced_252_input_limb_14),
                            verify_reduced_252_input_limb_15),
                            verify_reduced_252_input_limb_16),
                            verify_reduced_252_input_limb_17),
                            verify_reduced_252_input_limb_18),
                            verify_reduced_252_input_limb_19),
                            verify_reduced_252_input_limb_20);
        cuda_evaluator->add_constraint(
            mul(ms_and_mid_limbs_are_max_col1, sum_low)
        );
    }
}

#endif // EVALUATE_VERIFY_REDUCED_252_CONSTRAINT_H
