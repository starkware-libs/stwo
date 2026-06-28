#ifndef EVALUATE_CREATE_BLAKE_OUTPUT_CONSTRAINT_H
#define EVALUATE_CREATE_BLAKE_OUTPUT_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"
#include "constraints/evaluate_verify_blake_word.cuh"


template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_create_blake_output(
    const m31 create_blake_output_input_limb_0,
    const m31 create_blake_output_input_limb_1,
    const m31 create_blake_output_input_limb_2,
    const m31 create_blake_output_input_limb_3,
    const m31 create_blake_output_input_limb_4,
    const m31 create_blake_output_input_limb_5,
    const m31 create_blake_output_input_limb_6,
    const m31 create_blake_output_input_limb_7,
    const m31 create_blake_output_input_limb_8,
    const m31 create_blake_output_input_limb_9,
    const m31 create_blake_output_input_limb_10,
    const m31 create_blake_output_input_limb_11,
    const m31 create_blake_output_input_limb_12,
    const m31 create_blake_output_input_limb_13,
    const m31 create_blake_output_input_limb_14,
    const m31 create_blake_output_input_limb_15,
    const m31 create_blake_output_input_limb_16,
    const m31 create_blake_output_input_limb_17,
    const m31 create_blake_output_input_limb_18,
    const m31 create_blake_output_input_limb_19,
    const m31 create_blake_output_input_limb_20,
    const m31 create_blake_output_input_limb_21,
    const m31 create_blake_output_input_limb_22,
    const m31 create_blake_output_input_limb_23,
    const m31 create_blake_output_input_limb_24,
    const m31 create_blake_output_input_limb_25,
    const m31 create_blake_output_input_limb_26,
    const m31 create_blake_output_input_limb_27,
    const m31 create_blake_output_input_limb_28,
    const m31 create_blake_output_input_limb_29,
    const m31 create_blake_output_input_limb_30,
    const m31 create_blake_output_input_limb_31,
    const m31 create_blake_output_input_limb_32,
    const m31 create_blake_output_input_limb_33,
    const m31 create_blake_output_input_limb_34,
    const m31 create_blake_output_input_limb_35,
    const m31 create_blake_output_input_limb_36,
    const m31 create_blake_output_input_limb_37,
    const m31 create_blake_output_input_limb_38,
    const m31 create_blake_output_input_limb_39,
    const m31 create_blake_output_input_limb_40,
    const m31 create_blake_output_input_limb_41,
    const m31 create_blake_output_input_limb_42,
    const m31 create_blake_output_input_limb_43,
    const m31 create_blake_output_input_limb_44,
    const m31 create_blake_output_input_limb_45,
    const m31 create_blake_output_input_limb_46,
    const m31 create_blake_output_input_limb_47,
    // 16 triple_xor_32 outputs
    const m31 triple_xor_32_output_limb_0_col0,
    const m31 triple_xor_32_output_limb_1_col1,
    const m31 triple_xor_32_output_limb_0_col2,
    const m31 triple_xor_32_output_limb_1_col3,
    const m31 triple_xor_32_output_limb_0_col4,
    const m31 triple_xor_32_output_limb_1_col5,
    const m31 triple_xor_32_output_limb_0_col6,
    const m31 triple_xor_32_output_limb_1_col7,
    const m31 triple_xor_32_output_limb_0_col8,
    const m31 triple_xor_32_output_limb_1_col9,
    const m31 triple_xor_32_output_limb_0_col10,
    const m31 triple_xor_32_output_limb_1_col11,
    const m31 triple_xor_32_output_limb_0_col12,
    const m31 triple_xor_32_output_limb_1_col13,
    const m31 triple_xor_32_output_limb_0_col14,
    const m31 triple_xor_32_output_limb_1_col15,

    const CommonLookupElements& common_lookup_elements,
    // CUDA evaluator
    EvaluatorT* cuda_evaluator,
    // output (16 limbs)
    m31* out_limbs // [16]
) {
    // 1
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_16,
            create_blake_output_input_limb_17,
            create_blake_output_input_limb_32,
            create_blake_output_input_limb_33,
            create_blake_output_input_limb_0,
            create_blake_output_input_limb_1,
            triple_xor_32_output_limb_0_col0,
            triple_xor_32_output_limb_1_col1
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }
    // 2
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_18,
            create_blake_output_input_limb_19,
            create_blake_output_input_limb_34,
            create_blake_output_input_limb_35,
            create_blake_output_input_limb_2,
            create_blake_output_input_limb_3,
            triple_xor_32_output_limb_0_col2,
            triple_xor_32_output_limb_1_col3
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }
    // 3
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_20,
            create_blake_output_input_limb_21,
            create_blake_output_input_limb_36,
            create_blake_output_input_limb_37,
            create_blake_output_input_limb_4,
            create_blake_output_input_limb_5,
            triple_xor_32_output_limb_0_col4,
            triple_xor_32_output_limb_1_col5
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }
    // 4
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_22,
            create_blake_output_input_limb_23,
            create_blake_output_input_limb_38,
            create_blake_output_input_limb_39,
            create_blake_output_input_limb_6,
            create_blake_output_input_limb_7,
            triple_xor_32_output_limb_0_col6,
            triple_xor_32_output_limb_1_col7
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }
    // 5
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_24,
            create_blake_output_input_limb_25,
            create_blake_output_input_limb_40,
            create_blake_output_input_limb_41,
            create_blake_output_input_limb_8,
            create_blake_output_input_limb_9,
            triple_xor_32_output_limb_0_col8,
            triple_xor_32_output_limb_1_col9
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }
    // 6
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_26,
            create_blake_output_input_limb_27,
            create_blake_output_input_limb_42,
            create_blake_output_input_limb_43,
            create_blake_output_input_limb_10,
            create_blake_output_input_limb_11,
            triple_xor_32_output_limb_0_col10,
            triple_xor_32_output_limb_1_col11
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }
    // 7
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_28,
            create_blake_output_input_limb_29,
            create_blake_output_input_limb_44,
            create_blake_output_input_limb_45,
            create_blake_output_input_limb_12,
            create_blake_output_input_limb_13,
            triple_xor_32_output_limb_0_col12,
            triple_xor_32_output_limb_1_col13
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }
    // 8
    {
        m31 values[9] = {
            TRIPLE_XOR_32_RELATION_ID,
            create_blake_output_input_limb_30,
            create_blake_output_input_limb_31,
            create_blake_output_input_limb_46,
            create_blake_output_input_limb_47,
            create_blake_output_input_limb_14,
            create_blake_output_input_limb_15,
            triple_xor_32_output_limb_0_col14,
            triple_xor_32_output_limb_1_col15
        };
        cuda_evaluator->add_to_relation<9>(common_lookup_elements, qm31{1, 0, 0, 0}, values);
    }

    // output
    out_limbs[0] = triple_xor_32_output_limb_0_col0;
    out_limbs[1] = triple_xor_32_output_limb_1_col1;
    out_limbs[2] = triple_xor_32_output_limb_0_col2;
    out_limbs[3] = triple_xor_32_output_limb_1_col3;
    out_limbs[4] = triple_xor_32_output_limb_0_col4;
    out_limbs[5] = triple_xor_32_output_limb_1_col5;
    out_limbs[6] = triple_xor_32_output_limb_0_col6;
    out_limbs[7] = triple_xor_32_output_limb_1_col7;
    out_limbs[8] = triple_xor_32_output_limb_0_col8;
    out_limbs[9] = triple_xor_32_output_limb_1_col9;
    out_limbs[10] = triple_xor_32_output_limb_0_col10;
    out_limbs[11] = triple_xor_32_output_limb_1_col11;
    out_limbs[12] = triple_xor_32_output_limb_0_col12;
    out_limbs[13] = triple_xor_32_output_limb_1_col13;
    out_limbs[14] = triple_xor_32_output_limb_0_col14;
    out_limbs[15] = triple_xor_32_output_limb_1_col15;
}

#endif // EVALUATE_CREATE_BLAKE_OUTPUT_CONSTRAINT_H