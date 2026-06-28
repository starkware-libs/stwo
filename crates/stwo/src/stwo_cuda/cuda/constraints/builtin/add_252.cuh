#ifndef ADD_252_H
#define ADD_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "../evaluate_verify_add_252.cuh"

// CUDAversion Add252::evaluate
// translated from cairo-air/src/comptogethernts/subroutines/add_252.rs
// 252-bitfieldaddition: (a + b) mod p，p = 2^252 + 17*2^192 + 1

// Range check helper for 28 limbs using 8 RangeCheck_9_9 variants
template<typename EvaluatorT>
DEVICE_FORCEINLINE void range_check_mem_value_n_28(
    const m31 limbs[28],  // 28 limbs to range check
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    // Check limbs in pairs using 8 different RangeCheck_9_9 variants
    // Pattern: A,B,C,D,E,F,G,H repeats for 28 limbs (14 pairs)

    // Pairs 0-1: RangeCheck_9_9
    {
        m31 values[3] = {RANGE_CHECK_9_9_RELATION_ID, limbs[0], limbs[1]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 2-3: RangeCheck_9_9_B
    {
        m31 values[3] = {RANGE_CHECK_9_9_B_RELATION_ID, limbs[2], limbs[3]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 4-5: RangeCheck_9_9_C
    {
        m31 values[3] = {RANGE_CHECK_9_9_C_RELATION_ID, limbs[4], limbs[5]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 6-7: RangeCheck_9_9_D
    {
        m31 values[3] = {RANGE_CHECK_9_9_D_RELATION_ID, limbs[6], limbs[7]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 8-9: RangeCheck_9_9_E
    {
        m31 values[3] = {RANGE_CHECK_9_9_E_RELATION_ID, limbs[8], limbs[9]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 10-11: RangeCheck_9_9_F
    {
        m31 values[3] = {RANGE_CHECK_9_9_F_RELATION_ID, limbs[10], limbs[11]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 12-13: RangeCheck_9_9_G
    {
        m31 values[3] = {RANGE_CHECK_9_9_G_RELATION_ID, limbs[12], limbs[13]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 14-15: RangeCheck_9_9_H
    {
        m31 values[3] = {RANGE_CHECK_9_9_H_RELATION_ID, limbs[14], limbs[15]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Repeat pattern for remaining limbs
    // Pairs 16-17: RangeCheck_9_9
    {
        m31 values[3] = {RANGE_CHECK_9_9_RELATION_ID, limbs[16], limbs[17]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 18-19: RangeCheck_9_9_B
    {
        m31 values[3] = {RANGE_CHECK_9_9_B_RELATION_ID, limbs[18], limbs[19]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 20-21: RangeCheck_9_9_C
    {
        m31 values[3] = {RANGE_CHECK_9_9_C_RELATION_ID, limbs[20], limbs[21]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 22-23: RangeCheck_9_9_D
    {
        m31 values[3] = {RANGE_CHECK_9_9_D_RELATION_ID, limbs[22], limbs[23]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 24-25: RangeCheck_9_9_E
    {
        m31 values[3] = {RANGE_CHECK_9_9_E_RELATION_ID, limbs[24], limbs[25]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Pairs 26-27: RangeCheck_9_9_F
    {
        m31 values[3] = {RANGE_CHECK_9_9_F_RELATION_ID, limbs[26], limbs[27]};
        cuda_evaluator->template add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }
}

// Main Add252 function
// Computes: result = (a + b) mod p where p = 2^252 + 17*2^192 + 1
template<typename EvaluatorT>
DEVICE_FORCEINLINE void add_252_evaluate(
    const m31 input_a[28],   // First operand (28 limbs)
    const m31 input_b[28],   // Second operand (28 limbs)
    const m31 result[28],    // Result of addition (28 limbs)
    const m31 sub_p_bit,     // Boolean: whether we subtracted p
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    // Step 1: Range check the result limbs
    range_check_mem_value_n_28(
        result, common_lookup_elements,
        cuda_evaluator
    );

    // Step 2: Verify the addition is correct
    // This checks: a + b = result + sub_p_bit * p
    evaluate_verify_add_252(
        input_a[0], input_a[1], input_a[2], input_a[3], input_a[4], input_a[5], input_a[6], input_a[7],
        input_a[8], input_a[9], input_a[10], input_a[11], input_a[12], input_a[13], input_a[14], input_a[15],
        input_a[16], input_a[17], input_a[18], input_a[19], input_a[20], input_a[21], input_a[22], input_a[23],
        input_a[24], input_a[25], input_a[26], input_a[27],
        input_b[0], input_b[1], input_b[2], input_b[3], input_b[4], input_b[5], input_b[6], input_b[7],
        input_b[8], input_b[9], input_b[10], input_b[11], input_b[12], input_b[13], input_b[14], input_b[15],
        input_b[16], input_b[17], input_b[18], input_b[19], input_b[20], input_b[21], input_b[22], input_b[23],
        input_b[24], input_b[25], input_b[26], input_b[27],
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
        result[8], result[9], result[10], result[11], result[12], result[13], result[14], result[15],
        result[16], result[17], result[18], result[19], result[20], result[21], result[22], result[23],
        result[24], result[25], result[26], result[27],
        sub_p_bit,
        cuda_evaluator
    );
}

#endif // ADD_252_H
