#ifndef MOD_WORDS_TO_12_BIT_ARRAY_H
#define MOD_WORDS_TO_12_BIT_ARRAY_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// CUDAversion ModWordsTo12BitArray::evaluate
// translated from cairo-air/src/comptogethernts/subroutines/mod_words_to_12_bit_array.rs

template <typename EvaluatorT>
DEVICE_FORCEINLINE void mod_words_to_12_bit_array_evaluate(
    // 22 input limbs
    m31 input_limb_0, m31 input_limb_1, m31 input_limb_2, m31 input_limb_3,
    m31 input_limb_4, m31 input_limb_5, m31 input_limb_6, m31 input_limb_7,
    m31 input_limb_8, m31 input_limb_9, m31 input_limb_10,
    m31 input_limb_28, m31 input_limb_29, m31 input_limb_30, m31 input_limb_31,
    m31 input_limb_32, m31 input_limb_33, m31 input_limb_34, m31 input_limb_35,
    m31 input_limb_36, m31 input_limb_37, m31 input_limb_38,
    // 10 trace columns
    m31 limb1b_0_col0, m31 limb2b_0_col1, m31 limb5b_0_col2, m31 limb6b_0_col3, m31 limb9b_0_col4,
    m31 limb1b_1_col5, m31 limb2b_1_col6, m31 limb5b_1_col7, m31 limb6b_1_col8, m31 limb9b_1_col9,
    // Lookup elements
    const CommonLookupElements& common_lookup_elements,
    // Output array (16 elements)
    m31 *output,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_8 = m31(8);
    m31 M31_64 = m31(64);
    m31 M31_512 = m31(512);

    // limb1a_0 = input_limb_1 - limb1b_0 * 8
    m31 limb1a_0 = sub(input_limb_1, mul(limb1b_0_col0, M31_8));
    // limb2a_0 = input_limb_2 - limb2b_0 * 64
    m31 limb2a_0 = sub(input_limb_2, mul(limb2b_0_col1, M31_64));

    // RangeCheck_3_6_6_3 lookup 1
    {
        m31 values[5] = {RANGE_CHECK_3_6_6_3_RELATION_ID, limb1a_0, limb1b_0_col0, limb2a_0, limb2b_0_col1};
        cuda_evaluator->template add_to_relation<5>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // limb5a_0 = input_limb_5 - limb5b_0 * 8
    m31 limb5a_0 = sub(input_limb_5, mul(limb5b_0_col2, M31_8));
    // limb6a_0 = input_limb_6 - limb6b_0 * 64
    m31 limb6a_0 = sub(input_limb_6, mul(limb6b_0_col3, M31_64));

    // RangeCheck_3_6_6_3 lookup 2
    {
        m31 values[5] = {RANGE_CHECK_3_6_6_3_RELATION_ID, limb5a_0, limb5b_0_col2, limb6a_0, limb6b_0_col3};
        cuda_evaluator->template add_to_relation<5>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // limb9a_0 = input_limb_9 - limb9b_0 * 8
    m31 limb9a_0 = sub(input_limb_9, mul(limb9b_0_col4, M31_8));

    // limb1a_1 = input_limb_29 - limb1b_1 * 8
    m31 limb1a_1 = sub(input_limb_29, mul(limb1b_1_col5, M31_8));
    // limb2a_1 = input_limb_30 - limb2b_1 * 64
    m31 limb2a_1 = sub(input_limb_30, mul(limb2b_1_col6, M31_64));

    // RangeCheck_3_6_6_3 lookup 3
    {
        m31 values[5] = {RANGE_CHECK_3_6_6_3_RELATION_ID, limb1a_1, limb1b_1_col5, limb2a_1, limb2b_1_col6};
        cuda_evaluator->template add_to_relation<5>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // limb5a_1 = input_limb_33 - limb5b_1 * 8
    m31 limb5a_1 = sub(input_limb_33, mul(limb5b_1_col7, M31_8));
    // limb6a_1 = input_limb_34 - limb6b_1 * 64
    m31 limb6a_1 = sub(input_limb_34, mul(limb6b_1_col8, M31_64));

    // RangeCheck_3_6_6_3 lookup 4
    {
        m31 values[5] = {RANGE_CHECK_3_6_6_3_RELATION_ID, limb5a_1, limb5b_1_col7, limb6a_1, limb6b_1_col8};
        cuda_evaluator->template add_to_relation<5>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // limb9a_1 = input_limb_37 - limb9b_1 * 8
    m31 limb9a_1 = sub(input_limb_37, mul(limb9b_1_col9, M31_8));

    // RangeCheck_3_6_6_3 lookup 5
    {
        m31 values[5] = {RANGE_CHECK_3_6_6_3_RELATION_ID, limb9a_0, limb9b_0_col4, limb9b_1_col9, limb9a_1};
        cuda_evaluator->template add_to_relation<5>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Compute output limbs
    output[0] = add(input_limb_0, mul(M31_512, limb1a_0));
    output[1] = add(limb1b_0_col0, mul(M31_64, limb2a_0));
    output[2] = add(limb2b_0_col1, mul(M31_8, input_limb_3));
    output[3] = add(input_limb_4, mul(M31_512, limb5a_0));
    output[4] = add(limb5b_0_col2, mul(M31_64, limb6a_0));
    output[5] = add(limb6b_0_col3, mul(M31_8, input_limb_7));
    output[6] = add(input_limb_8, mul(M31_512, limb9a_0));
    output[7] = add(limb9b_0_col4, mul(M31_64, input_limb_10));
    output[8] = add(input_limb_28, mul(M31_512, limb1a_1));
    output[9] = add(limb1b_1_col5, mul(M31_64, limb2a_1));
    output[10] = add(limb2b_1_col6, mul(M31_8, input_limb_31));
    output[11] = add(input_limb_32, mul(M31_512, limb5a_1));
    output[12] = add(limb5b_1_col7, mul(M31_64, limb6a_1));
    output[13] = add(limb6b_1_col8, mul(M31_8, input_limb_35));
    output[14] = add(input_limb_36, mul(M31_512, limb9a_1));
    output[15] = add(limb9b_1_col9, mul(M31_64, input_limb_38));
}

#endif // MOD_WORDS_TO_12_BIT_ARRAY_H
