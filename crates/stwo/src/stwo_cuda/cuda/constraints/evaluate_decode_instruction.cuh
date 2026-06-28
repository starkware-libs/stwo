#ifndef EVALUATE_DECODE_INSTRUCTION_H
#define EVALUATE_DECODE_INSTRUCTION_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"
#include <cstdio>



template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_2a7a2 (
    m31 decode_instruction_2a7a2f4f5427e720_input,
    m31* output_vec,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_32 = {32};
    m31 M31_32768 = {32768};
    m31 M31_32769 = {32769};
    m31 M31_68 = {68};

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_2a7a2f4f5427e720_input,
        M31_32768,
        M31_32769,
        M31_32769,
        M31_32,
        M31_68,
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0}, {0,0}}, values);

    output_vec[0] = M31_0;
    output_vec[1] = M31_1;
    output_vec[2] = M31_1;
    output_vec[3] = M31_0;
    output_vec[4] = M31_0;
    output_vec[5] = M31_1;
    output_vec[6] = M31_0;
    output_vec[7] = M31_0;
    output_vec[8] = M31_0;
    output_vec[9] = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_1;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = M31_0;
    output_vec[15] = M31_1;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_3b105 (
    m31 decode_instruction_3b1056363058c126_input,
    m31 offset2_col0,
    m31 op1_base_fp_col1,
    m31 op1_base_ap_col2,
    m31 ap_update_add_1_col3,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_2147483646 = {2147483646};
    m31 M31_24 = {24};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};
    m31 M31_4 = {4};
    m31 M31_64 = {64};

    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col1, sub(M31_1, op1_base_fp_col1)));

    // Flag op1_base_ap is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_ap_col2, sub(M31_1, op1_base_ap_col2)));

    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col3, sub(M31_1, ap_update_add_1_col3)));
    m31 value5 = add(
        add(M31_24, mul(op1_base_fp_col1, M31_64)),
        mul(op1_base_ap_col2, M31_128));
    m31 value6 = add(
        M31_4,
        mul(ap_update_add_1_col3, M31_32));

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_3b1056363058c126_input,
        M31_32767,
        M31_32767,
        offset2_col0,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    // Return value: [(offset2_col0 - M31_32768)]
    output_vec[0] = sub(offset2_col0, M31_32768);
    output_vec[1] = M31_2147483646;
    output_vec[2] = sub(offset2_col0, M31_32768);
    output_vec[3] = M31_1;
    output_vec[4] = M31_1;
    output_vec[5] = M31_0;
    output_vec[6] = op1_base_fp_col1;
    output_vec[7] = op1_base_ap_col2;
    output_vec[8] = M31_0;
    output_vec[9] = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_1;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col3;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_4b8cf(
    m31 decode_instruction_4b8cfd7f4c406cba_input,
    m31 offset0_col0,
    m31 offset1_col1,
    m31 offset2_col2,
    m31 dst_base_fp_col3,
    m31 op0_base_fp_col4,
    m31 op1_imm_col5,
    m31 op1_base_fp_col6,
    m31 ap_update_add_1_col7,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_16 = {16};
    m31 M31_256 = {256};
    m31 M31_32 = {32};
    m31 M31_32768 = {32768};
    m31 M31_64 = {64};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    // Rust: flag * (1 - flag)
    cuda_evaluator->add_constraint(mul(dst_base_fp_col3, sub(M31_1, dst_base_fp_col3)));
    // Flag op0_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op0_base_fp_col4, sub(M31_1, op0_base_fp_col4)));
    // Flag op1_imm is a bit.
    cuda_evaluator->add_constraint(mul(op1_imm_col5, sub(M31_1, op1_imm_col5)));
    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col6, sub(M31_1, op1_base_fp_col6)));
    // Flag op1_base_ap is a bit.
    // Rust: tmp = (1 - op1_imm) - op1_base_fp; constraint = tmp * (1 - tmp)
    m31 tmp1 = sub(M31_1, op1_imm_col5);
    m31 tmp2 = sub(tmp1, op1_base_fp_col6);
    m31 tmp3 = sub(M31_1, tmp2);
    cuda_evaluator->add_constraint(mul(tmp2, tmp3));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col7, sub(M31_1, ap_update_add_1_col7)));

    m31 value5 = add(
        add(
            add(
                mul(dst_base_fp_col3, M31_8),
                mul(op0_base_fp_col4, M31_16)
            ),
            mul(op1_imm_col5, M31_32)
        ),
        add(
            mul(op1_base_fp_col6, M31_64),
            mul(tmp2, M31_128)
        )
    );
    m31 value6 = add(
        add(M31_1, mul(ap_update_add_1_col7, M31_32)),
        M31_256
    );

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_4b8cfd7f4c406cba_input,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0] = sub(offset0_col0, M31_32768);
    output_vec[1] = sub(offset1_col1, M31_32768);
    output_vec[2] = sub(offset2_col2, M31_32768);
    output_vec[3] = dst_base_fp_col3;
    output_vec[4] = op0_base_fp_col4;
    output_vec[5] = op1_imm_col5;
    output_vec[6] = op1_base_fp_col6;
    output_vec[7] = tmp2;
    output_vec[8] = M31_0;
    output_vec[9] = M31_1;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col7;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_1;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_7ebc4(
    m31 decode_instruction_7ebc4fb565f52942_input,
    m31 ap_update_add_1_col0,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_2147483646 = {2147483646};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32769 = {32769};
    m31 M31_4 = {4};
    m31 M31_56 = {56};

    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col0, sub(M31_1, ap_update_add_1_col0)));

    m31 value5 = add(M31_4, mul(ap_update_add_1_col0, M31_32));

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_7ebc4fb565f52942_input,
        M31_32767,
        M31_32767,
        M31_32769,
        M31_56,
        value5
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = M31_2147483646;
    output_vec[1]  = M31_2147483646;
    output_vec[2]  = M31_1;
    output_vec[3]  = M31_1;
    output_vec[4]  = M31_1;
    output_vec[5]  = M31_1;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_1;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col0;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_9bd86(
    m31 decode_instruction_9bd8670ed070e5a5_input,
    m31 offset1_col0,
    m31 offset2_col1,
    m31 op0_base_fp_col2,
    m31 ap_update_add_1_col3,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_16 = {16};
    m31 M31_2 = {2};
    m31 M31_2147483646 = {2147483646};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};
    m31 M31_8 = {8};

    // Flag op0_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op0_base_fp_col2, sub(M31_1, op0_base_fp_col2)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col3, sub(M31_1, ap_update_add_1_col3)));

    m31 value5 = add(M31_8, mul(op0_base_fp_col2, M31_16));
    m31 value6 = add(M31_2, mul(ap_update_add_1_col3, M31_32));

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_9bd8670ed070e5a5_input,
        M31_32767,
        offset1_col0,
        offset2_col1,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    // Return values: [(offset1_col0 - M31_32768), (offset2_col1 - M31_32768)]
    output_vec[0]  = sub(offset1_col0, M31_32768);
    output_vec[1]  = sub(offset2_col1, M31_32768);
    output_vec[2]  = sub(offset2_col1, M31_32768);
    output_vec[3]  = M31_1;
    output_vec[4]  = op0_base_fp_col2;
    output_vec[5]  = M31_0;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_1;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col3;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_15a61(
    m31 decode_instruction_15a61c6002c544ec_input,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_130 = {130};
    m31 M31_2147483645 = {2147483645};
    m31 M31_2147483646 = {2147483646};
    m31 M31_32766 = {32766};
    m31 M31_32767 = {32767};
    m31 M31_88 = {88};

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_15a61c6002c544ec_input,
        M31_32766,
        M31_32767,
        M31_32767,
        M31_88,
        M31_130
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = M31_2147483645;
    output_vec[1]  = M31_2147483646;
    output_vec[2]  = M31_2147483646;
    output_vec[3]  = M31_1;
    output_vec[4]  = M31_1;
    output_vec[5]  = M31_0;
    output_vec[6]  = M31_1;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_1;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = M31_0;
    output_vec[15] = M31_0;
    output_vec[16] = M31_1;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_43e1c(
    m31 decode_instruction_43e1c26dca5a217_input,
    m31 offset2_col0,
    m31 op1_base_fp_col1,
    m31 op1_base_ap_col2,
    m31 ap_update_add_1_col3,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_2 = {2};
    m31 M31_2147483646 = {2147483646};
    m31 M31_24 = {24};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};
    m31 M31_64 = {64};

    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col1, sub(M31_1, op1_base_fp_col1)));
    // Flag op1_base_ap is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_ap_col2, sub(M31_1, op1_base_ap_col2)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col3, sub(M31_1, ap_update_add_1_col3)));

    m31 value5 = add(
        add(M31_24, mul(op1_base_fp_col1, M31_64)),
        mul(op1_base_ap_col2, M31_128)
    );
    m31 value6 = add(M31_2, mul(ap_update_add_1_col3, M31_32));

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_43e1c26dca5a217_input,
        M31_32767,
        M31_32767,
        offset2_col0,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    // Return value: offset2 - 32768
    output_vec[0]  = sub(offset2_col0, M31_32768);
    output_vec[1]  = M31_2147483646;
    output_vec[2]  = sub(offset2_col0, M31_32768);
    output_vec[3]  = M31_1;
    output_vec[4]  = M31_1;
    output_vec[5]  = M31_0;
    output_vec[6]  = op1_base_fp_col1;
    output_vec[7]  = op1_base_ap_col2;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_1;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col3;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_161c9(
    m31 decode_instruction_161c97dc78559210_input,
    m31 offset0_col0,
    m31 dst_base_fp_col1,
    m31 ap_update_add_1_col2,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_16 = {16};
    m31 M31_2147483646 = {2147483646};
    m31 M31_256 = {256};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};
    m31 M31_32769 = {32769};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col1, sub(M31_1, dst_base_fp_col1)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col2, sub(M31_1, ap_update_add_1_col2)));

    m31 value5 = add(
        add(mul(dst_base_fp_col1, M31_8), M31_16),
        M31_32
    );
    m31 value6 = add(
        mul(ap_update_add_1_col2, M31_32),
        M31_256
    );

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_161c97dc78559210_input,
        offset0_col0,
        M31_32767,
        M31_32769,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = M31_2147483646;
    output_vec[2]  = M31_1;
    output_vec[3]  = dst_base_fp_col1;
    output_vec[4]  = M31_1;
    output_vec[5]  = M31_1;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col2;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_1;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_3802d(
    m31 decode_instruction_3802d5ea8e8d383b_input,
    m31 offset0_col0,
    m31 offset1_col1,
    m31 offset2_col2,
    m31 dst_base_fp_col3,
    m31 op0_base_fp_col4,
    m31 op1_imm_col5,
    m31 op1_base_fp_col6,
    m31 res_add_col7,
    m31 ap_update_add_1_col8,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_16 = {16};
    m31 M31_256 = {256};
    m31 M31_3 = {3};
    m31 M31_32 = {32};
    m31 M31_32768 = {32768};
    m31 M31_64 = {64};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col3, sub(M31_1, dst_base_fp_col3)));
    // Flag op0_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op0_base_fp_col4, sub(M31_1, op0_base_fp_col4)));
    // Flag op1_imm is a bit.
    cuda_evaluator->add_constraint(mul(op1_imm_col5, sub(M31_1, op1_imm_col5)));
    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col6, sub(M31_1, op1_base_fp_col6)));
    // Flag op1_base_ap is a bit.
    m31 t1 = sub(M31_1, op1_imm_col5);
    m31 t2 = sub(t1, op1_base_fp_col6);
    cuda_evaluator->add_constraint(mul(t2, sub(M31_1, t2)));
    // Flag res_add is a bit.
    cuda_evaluator->add_constraint(mul(res_add_col7, sub(M31_1, res_add_col7)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col8, sub(M31_1, ap_update_add_1_col8)));

    m31 value5 = add(
        add(
            add(
                add(
                    mul(dst_base_fp_col3, M31_8),
                    mul(op0_base_fp_col4, M31_16)
                ),
                mul(op1_imm_col5, M31_32)
            ),
            mul(op1_base_fp_col6, M31_64)
        ),
        add(
            mul(t2, M31_128),
            mul(res_add_col7, M31_256)
        )
    );
    m31 value6 = add(
        add(sub(M31_1, res_add_col7), mul(ap_update_add_1_col8, M31_32)),
        M31_256
    );

    m31 values[8] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_3802d5ea8e8d383b_input,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        value5,
        value6,
        M31_3
    };

    cuda_evaluator->add_to_relation<8>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = sub(offset1_col1, M31_32768);
    output_vec[2]  = sub(offset2_col2, M31_32768);
    output_vec[3]  = dst_base_fp_col3;
    output_vec[4]  = op0_base_fp_col4;
    output_vec[5]  = op1_imm_col5;
    output_vec[6]  = op1_base_fp_col6;
    output_vec[7]  = t2;
    output_vec[8]  = res_add_col7;
    output_vec[9]  = sub(M31_1, res_add_col7);
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col8;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_1;
    output_vec[18] = M31_3;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_64420(
    m31 decode_instruction_64420902f4d72579_input,
    m31 offset0_col0,
    m31 offset1_col1,
    m31 offset2_col2,
    m31 dst_base_fp_col3,
    m31 op0_base_fp_col4,
    m31 op1_base_fp_col5,
    m31 op1_base_ap_col6,
    m31 ap_update_add_1_col7,
    m31 opcode_extension_col8,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_16 = {16};
    m31 M31_32 = {32};
    m31 M31_32768 = {32768};
    m31 M31_64 = {64};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col3, sub(M31_1, dst_base_fp_col3)));
    // Flag op0_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op0_base_fp_col4, sub(M31_1, op0_base_fp_col4)));
    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col5, sub(M31_1, op1_base_fp_col5)));
    // Flag op1_base_ap is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_ap_col6, sub(M31_1, op1_base_ap_col6)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col7, sub(M31_1, ap_update_add_1_col7)));

    m31 value5 = add(
        add(
            add(
                mul(dst_base_fp_col3, M31_8),
                mul(op0_base_fp_col4, M31_16)
            ),
            mul(op1_base_fp_col5, M31_64)
        ),
        mul(op1_base_ap_col6, M31_128)
    );
    m31 value6 = mul(ap_update_add_1_col7, M31_32);

    m31 values[8] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_64420902f4d72579_input,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        value5,
        value6,
        opcode_extension_col8
    };

    cuda_evaluator->add_to_relation<8>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = sub(offset1_col1, M31_32768);
    output_vec[2]  = sub(offset2_col2, M31_32768);
    output_vec[3]  = dst_base_fp_col3;
    output_vec[4]  = op0_base_fp_col4;
    output_vec[5]  = M31_0;
    output_vec[6]  = op1_base_fp_col5;
    output_vec[7]  = op1_base_ap_col6;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col7;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = opcode_extension_col8;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_bc3cd(
    m31 decode_instruction_bc3cd8d59f69b4e6_input,
    m31 offset0_col0,
    m31 offset1_col1,
    m31 offset2_col2,
    m31 dst_base_fp_col3,
    m31 op0_base_fp_col4,
    m31 op1_imm_col5,
    m31 op1_base_fp_col6,
    m31 ap_update_add_1_col7,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_16 = {16};
    m31 M31_256 = {256};
    m31 M31_32 = {32};
    m31 M31_32768 = {32768};
    m31 M31_64 = {64};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col3, sub(M31_1, dst_base_fp_col3)));
    // Flag op0_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op0_base_fp_col4, sub(M31_1, op0_base_fp_col4)));
    // Flag op1_imm is a bit.
    cuda_evaluator->add_constraint(mul(op1_imm_col5, sub(M31_1, op1_imm_col5)));
    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col6, sub(M31_1, op1_base_fp_col6)));
    // Flag op1_base_ap is a bit.
    m31 t1 = sub(M31_1, op1_imm_col5);
    m31 t2 = sub(t1, op1_base_fp_col6);
    cuda_evaluator->add_constraint(mul(t2, sub(M31_1, t2)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col7, sub(M31_1, ap_update_add_1_col7)));

    m31 value5 = add(
        add(
            add(
                add(
                    mul(dst_base_fp_col3, M31_8),
                    mul(op0_base_fp_col4, M31_16)
                ),
                mul(op1_imm_col5, M31_32)
            ),
            mul(op1_base_fp_col6, M31_64)
        ),
        add(
            mul(t2, M31_128),
            M31_256
        )
    );
    m31 value6 = add(mul(ap_update_add_1_col7, M31_32), M31_256);

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_bc3cd8d59f69b4e6_input,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = sub(offset1_col1, M31_32768);
    output_vec[2]  = sub(offset2_col2, M31_32768);
    output_vec[3]  = dst_base_fp_col3;
    output_vec[4]  = op0_base_fp_col4;
    output_vec[5]  = op1_imm_col5;
    output_vec[6]  = op1_base_fp_col6;
    output_vec[7]  = t2;
    output_vec[8]  = M31_1;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col7;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_1;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_cb32b(
    m31 decode_instruction_cb32bef316ee78d5_input,
    m31 offset0_col0,
    m31 offset1_col1,
    m31 offset2_col2,
    m31 dst_base_fp_col3,
    m31 op0_base_fp_col4,
    m31 ap_update_add_1_col5,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_16 = {16};
    m31 M31_256 = {256};
    m31 M31_32 = {32};
    m31 M31_32768 = {32768};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col3, sub(M31_1, dst_base_fp_col3)));
    // Flag op0_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op0_base_fp_col4, sub(M31_1, op0_base_fp_col4)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col5, sub(M31_1, ap_update_add_1_col5)));

    m31 value5 = add(
        mul(dst_base_fp_col3, M31_8),
        mul(op0_base_fp_col4, M31_16)
    );
    m31 value6 = add(mul(ap_update_add_1_col5, M31_32), M31_256);

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_cb32bef316ee78d5_input,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = sub(offset1_col1, M31_32768);
    output_vec[2]  = sub(offset2_col2, M31_32768);
    output_vec[3]  = dst_base_fp_col3;
    output_vec[4]  = op0_base_fp_col4;
    output_vec[5]  = M31_0;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col5;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_1;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_d2a10(
    m31 decode_instruction_d2a10466ff437b2e_input,
    m31 offset2_col0,
    m31 op1_imm_col1,
    m31 op1_base_fp_col2,
    m31 *output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_16 = {16};
    m31 M31_24 = {24};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};
    m31 M31_64 = {64};

    // Flag op1_imm is a bit.
    cuda_evaluator->add_constraint(mul(op1_imm_col1, sub(M31_1, op1_imm_col1)));
    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col2, sub(M31_1, op1_base_fp_col2)));
    // Flag op1_base_ap is a bit.
    cuda_evaluator->add_constraint(mul(sub(sub(M31_1, op1_imm_col1), op1_base_fp_col2),
        sub(M31_1, sub(sub(M31_1, op1_imm_col1), op1_base_fp_col2))));

    m31 value5 = add(
        add(
            add(M31_24, mul(op1_imm_col1, M31_32)),
            mul(op1_base_fp_col2, M31_64)
            ),
        mul(sub(sub(M31_1, op1_imm_col1), op1_base_fp_col2), M31_128)
    );

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_d2a10466ff437b2e_input,
        M31_32767,
        M31_32767,
        offset2_col0,
        value5,
        M31_16
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    // Return values: [(offset2_col0 - M31_32768), ((M31_1 - op1_imm_col1) - op1_base_fp_col2)]
    output_vec[0]  = sub(offset2_col0, M31_32768);
    output_vec[1]  = sub(sub(M31_1, op1_imm_col1), op1_base_fp_col2);
    output_vec[2]  = sub(offset2_col0, M31_32768);
    output_vec[3]  = M31_1;
    output_vec[4]  = M31_1;
    output_vec[5]  = op1_imm_col1;
    output_vec[6]  = op1_base_fp_col2;
    output_vec[7]  = sub(sub(M31_1, op1_imm_col1), op1_base_fp_col2);
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_1;
    output_vec[14] = M31_0;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_de75a(
    m31 decode_instruction_de75ab42b9e8d1d4_input,
    m31 offset0_col0,
    m31 dst_base_fp_col1,
    m31 ap_update_add_1_col2,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_16 = {16};
    m31 M31_2147483646 = {2147483646};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};
    m31 M31_32769 = {32769};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col1, sub(M31_1, dst_base_fp_col1)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col2, sub(M31_1, ap_update_add_1_col2)));

    m31 value5 = add(
        add(mul(dst_base_fp_col1, M31_8), M31_16),
        M31_32
    );
    m31 value6 = add(M31_8, mul(ap_update_add_1_col2, M31_32));

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_de75ab42b9e8d1d4_input,
        offset0_col0,
        M31_32767,
        M31_32769,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = M31_2147483646;
    output_vec[2]  = M31_1;
    output_vec[3]  = dst_base_fp_col1;
    output_vec[4]  = M31_1;
    output_vec[5]  = M31_1;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_1;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col2;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_df7a6(
    m31 decode_instruction_df7a69b85cbf80d5_input,
    m31 offset0_col0,
    m31 offset1_col1,
    m31 offset2_col2,
    m31 dst_base_fp_col3,
    m31 op0_base_fp_col4,
    m31 op1_imm_col5,
    m31 op1_base_fp_col6,
    m31 op1_base_ap_col7,
    m31 res_add_col8,
    m31 res_mul_col9,
    m31 pc_update_jump_col10,
    m31 pc_update_jump_rel_col11,
    m31 pc_update_jnz_col12,
    m31 ap_update_add_col13,
    m31 ap_update_add_1_col14,
    m31 opcode_call_col15,
    m31 opcode_ret_col16,
    m31 opcode_assert_eq_col17,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_16 = {16};
    m31 M31_2 = {2};
    m31 M31_256 = {256};
    m31 M31_32 = {32};
    m31 M31_32768 = {32768};
    m31 M31_4 = {4};
    m31 M31_64 = {64};
    m31 M31_8 = {8};

    // Bit constraints
    cuda_evaluator->add_constraint(mul(dst_base_fp_col3, sub(M31_1, dst_base_fp_col3)));
    cuda_evaluator->add_constraint(mul(op0_base_fp_col4, sub(M31_1, op0_base_fp_col4)));
    cuda_evaluator->add_constraint(mul(op1_imm_col5, sub(M31_1, op1_imm_col5)));
    cuda_evaluator->add_constraint(mul(op1_base_fp_col6, sub(M31_1, op1_base_fp_col6)));
    cuda_evaluator->add_constraint(mul(op1_base_ap_col7, sub(M31_1, op1_base_ap_col7)));
    cuda_evaluator->add_constraint(mul(res_add_col8, sub(M31_1, res_add_col8)));
    cuda_evaluator->add_constraint(mul(res_mul_col9, sub(M31_1, res_mul_col9)));
    cuda_evaluator->add_constraint(mul(pc_update_jump_col10, sub(M31_1, pc_update_jump_col10)));
    cuda_evaluator->add_constraint(mul(pc_update_jump_rel_col11, sub(M31_1, pc_update_jump_rel_col11)));
    cuda_evaluator->add_constraint(mul(pc_update_jnz_col12, sub(M31_1, pc_update_jnz_col12)));
    cuda_evaluator->add_constraint(mul(ap_update_add_col13, sub(M31_1, ap_update_add_col13)));
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col14, sub(M31_1, ap_update_add_1_col14)));
    cuda_evaluator->add_constraint(mul(opcode_call_col15, sub(M31_1, opcode_call_col15)));
    cuda_evaluator->add_constraint(mul(opcode_ret_col16, sub(M31_1, opcode_ret_col16)));
    cuda_evaluator->add_constraint(mul(opcode_assert_eq_col17, sub(M31_1, opcode_assert_eq_col17)));

    m31 value5 = add(
        add(
            add(
                add(
                    add(
                        mul(dst_base_fp_col3, M31_8),
                        mul(op0_base_fp_col4, M31_16)
                    ),
                    mul(op1_imm_col5, M31_32)
                ),
                mul(op1_base_fp_col6, M31_64)
            ),
            mul(op1_base_ap_col7, M31_128)
        ),
        mul(res_add_col8, M31_256)
    );

    m31 value6 = add(
        add(
            add(
                add(
                    add(
                        add(
                            add(
                                res_mul_col9,
                                mul(pc_update_jump_col10, M31_2)
                            ),
                            mul(pc_update_jump_rel_col11, M31_4)
                        ),
                        mul(pc_update_jnz_col12, M31_8)
                    ),
                    mul(ap_update_add_col13, M31_16)
                ),
                mul(ap_update_add_1_col14, M31_32)
            ),
            mul(opcode_call_col15, M31_64)
        ),
        add(
            mul(opcode_ret_col16, M31_128),
            mul(opcode_assert_eq_col17, M31_256)
        )
    );

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_df7a69b85cbf80d5_input,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = sub(offset1_col1, M31_32768);
    output_vec[2]  = sub(offset2_col2, M31_32768);
    output_vec[3]  = dst_base_fp_col3;
    output_vec[4]  = op0_base_fp_col4;
    output_vec[5]  = op1_imm_col5;
    output_vec[6]  = op1_base_fp_col6;
    output_vec[7]  = op1_base_ap_col7;
    output_vec[8]  = res_add_col8;
    output_vec[9]  = res_mul_col9;
    output_vec[10] = pc_update_jump_col10;
    output_vec[11] = pc_update_jump_rel_col11;
    output_vec[12] = pc_update_jnz_col12;
    output_vec[13] = ap_update_add_col13;
    output_vec[14] = ap_update_add_1_col14;
    output_vec[15] = opcode_call_col15;
    output_vec[16] = opcode_ret_col16;
    output_vec[17] = opcode_assert_eq_col17;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_ea769(
    m31 decode_instruction_ea769df2d427981f_input,
    m31 offset2_col0,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_32768 = {32768};
    m31 M31_32769 = {32769};
    m31 M31_64 = {64};
    m31 M31_66 = {66};

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_ea769df2d427981f_input,
        M31_32768,
        M31_32769,
        offset2_col0,
        M31_64,
        M31_66
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = M31_0;
    output_vec[1]  = M31_1;
    output_vec[2]  = sub(offset2_col0, M31_32768);
    output_vec[3]  = M31_0;
    output_vec[4]  = M31_0;
    output_vec[5]  = M31_0;
    output_vec[6]  = M31_1;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_1;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = M31_0;
    output_vec[15] = M31_1;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_fdb6e(
    m31 decode_instruction_fdb6eeac016f2351_input,
    m31 offset2_col0,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_32768 = {32768};
    m31 M31_32769 = {32769};
    m31 M31_66 = {66};

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_fdb6eeac016f2351_input,
        M31_32768,
        M31_32769,
        offset2_col0,
        M31_128,
        M31_66
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = M31_0;
    output_vec[1]  = M31_1;
    output_vec[2]  = sub(offset2_col0, M31_32768);
    output_vec[3]  = M31_0;
    output_vec[4]  = M31_0;
    output_vec[5]  = M31_0;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_1;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_1;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = M31_0;
    output_vec[15] = M31_1;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
}

template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_fe864(
    m31 decode_instruction_fe8642bd3c473132_input,
    m31 offset0_col0,
    m31 offset2_col1,
    m31 dst_base_fp_col2,
    m31 op1_base_fp_col3,
    m31 ap_update_add_1_col4,
    m31* output_vec, // 19 elements
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_0 = {0};
    m31 M31_1 = {1};
    m31 M31_128 = {128};
    m31 M31_16 = {16};
    m31 M31_2147483646 = {2147483646};
    m31 M31_256 = {256};
    m31 M31_32 = {32};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};
    m31 M31_64 = {64};
    m31 M31_8 = {8};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col2, sub(M31_1, dst_base_fp_col2)));
    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col3, sub(M31_1, op1_base_fp_col3)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col4, sub(M31_1, ap_update_add_1_col4)));

    m31 value5 = add(
        add(
            mul(dst_base_fp_col2, M31_8),
            M31_16
        ),
        add(
            mul(op1_base_fp_col3, M31_64),
            mul(sub(M31_1, op1_base_fp_col3), M31_128)
        )
    );
    m31 value6 = add(mul(ap_update_add_1_col4, M31_32), M31_256);

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_fe8642bd3c473132_input,
        offset0_col0,
        M31_32767,
        offset2_col1,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    output_vec[0]  = sub(offset0_col0, M31_32768);
    output_vec[1]  = M31_2147483646;
    output_vec[2]  = sub(offset2_col1, M31_32768);
    output_vec[3]  = dst_base_fp_col2;
    output_vec[4]  = M31_1;
    output_vec[5]  = M31_0;
    output_vec[6]  = op1_base_fp_col3;
    output_vec[7]  = sub(M31_1, op1_base_fp_col3);
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = ap_update_add_1_col4;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_1;
    output_vec[18] = M31_0;
}

// DecodeInstructionB1597 - matches "now" Rust AIR
// Used by jump_opcode_abs. Takes op1_base_fp and ap_update_add_1 only (no separate op1_base_ap).
// op1_base_ap is derived as (1 - op1_base_fp).
// Returns: [offset2 - 32768, (1 - op1_base_fp)]
template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_b1597(
    m31 decode_instruction_b1597_input_pc,
    m31 offset2_col0,
    m31 op1_base_fp_col1,
    m31 ap_update_add_1_col2,
    m31* output_vec, // 2 elements: [offset2 - 32768, op1_base_ap]
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_1 = {1};
    m31 M31_2 = {2};
    m31 M31_24 = {24};
    m31 M31_32 = {32};
    m31 M31_64 = {64};
    m31 M31_128 = {128};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};

    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col1, sub(M31_1, op1_base_fp_col1)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col2, sub(M31_1, ap_update_add_1_col2)));

    // op1_base_ap = 1 - op1_base_fp (not a separate column)
    m31 op1_base_ap = sub(M31_1, op1_base_fp_col1);

    m31 value5 = add(
        add(M31_24, mul(op1_base_fp_col1, M31_64)),
        mul(op1_base_ap, M31_128)
    );
    m31 value6 = add(M31_2, mul(ap_update_add_1_col2, M31_32));

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_b1597_input_pc,
        M31_32767,
        M31_32767,
        offset2_col0,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    // Return: [offset2 - 32768, op1_base_ap]
    output_vec[0] = sub(offset2_col0, M31_32768);
    output_vec[1] = op1_base_ap;
}

// DecodeInstructionBa944 - matches "now" Rust AIR
// Used by jump_opcode_rel. Takes op1_base_fp and ap_update_add_1 only (no separate op1_base_ap).
// op1_base_ap is derived as (1 - op1_base_fp).
// Returns: [offset2 - 32768, (1 - op1_base_fp)]
template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_ba944(
    m31 decode_instruction_ba944_input_pc,
    m31 offset2_col0,
    m31 op1_base_fp_col1,
    m31 ap_update_add_1_col2,
    m31* output_vec, // 2 elements: [offset2 - 32768, op1_base_ap]
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_1 = {1};
    m31 M31_4 = {4};
    m31 M31_24 = {24};
    m31 M31_32 = {32};
    m31 M31_64 = {64};
    m31 M31_128 = {128};
    m31 M31_32767 = {32767};
    m31 M31_32768 = {32768};

    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col1, sub(M31_1, op1_base_fp_col1)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col2, sub(M31_1, ap_update_add_1_col2)));

    // op1_base_ap = 1 - op1_base_fp (not a separate column)
    m31 op1_base_ap = sub(M31_1, op1_base_fp_col1);

    m31 value5 = add(
        add(M31_24, mul(op1_base_fp_col1, M31_64)),
        mul(op1_base_ap, M31_128)
    );
    m31 value6 = add(M31_4, mul(ap_update_add_1_col2, M31_32));

    m31 values[7] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_ba944_input_pc,
        M31_32767,
        M31_32767,
        offset2_col0,
        value5,
        value6
    };

    cuda_evaluator->add_to_relation<7>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    // Return: [offset2 - 32768, op1_base_ap]
    output_vec[0] = sub(offset2_col0, M31_32768);
    output_vec[1] = op1_base_ap;
}

// DecodeInstruction472Fe - matches Rust AIR version c574c96b
// Only 4 constraints (no op1_base_ap bit check)
// Returns: [offset0-32768, offset1-32768, offset2-32768, (1 - op1_base_fp)]
template<typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_decode_instruction_472fe(
    m31 decode_instruction_472fe_input_pc,
    m31 offset0_col0,
    m31 offset1_col1,
    m31 offset2_col2,
    m31 dst_base_fp_col3,
    m31 op0_base_fp_col4,
    m31 op1_base_fp_col5,
    m31 ap_update_add_1_col6,
    m31 opcode_extension_col7,
    m31* output_vec, // 4 elements: [offset0-32768, offset1-32768, offset2-32768, (1-op1_base_fp)]
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    m31 M31_1 = {1};
    m31 M31_8 = {8};
    m31 M31_16 = {16};
    m31 M31_32 = {32};
    m31 M31_64 = {64};
    m31 M31_128 = {128};
    m31 M31_32768 = {32768};

    // Flag dst_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(dst_base_fp_col3, sub(M31_1, dst_base_fp_col3)));
    // Flag op0_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op0_base_fp_col4, sub(M31_1, op0_base_fp_col4)));
    // Flag op1_base_fp is a bit.
    cuda_evaluator->add_constraint(mul(op1_base_fp_col5, sub(M31_1, op1_base_fp_col5)));
    // Flag ap_update_add_1 is a bit.
    cuda_evaluator->add_constraint(mul(ap_update_add_1_col6, sub(M31_1, ap_update_add_1_col6)));

    // Compute the lookup value components
    // value5 = dst_base_fp*8 + op0_base_fp*16 + op1_base_fp*64 + (1-op1_base_fp)*128
    m31 value5 = add(
        add(
            add(
                mul(dst_base_fp_col3, M31_8),
                mul(op0_base_fp_col4, M31_16)
            ),
            mul(op1_base_fp_col5, M31_64)
        ),
        mul(sub(M31_1, op1_base_fp_col5), M31_128)
    );
    m31 value6 = mul(ap_update_add_1_col6, M31_32);

    m31 values[8] = {
        VERIFY_INSTRUCTION_RELATION_ID,
        decode_instruction_472fe_input_pc,
        offset0_col0,
        offset1_col1,
        offset2_col2,
        value5,
        value6,
        opcode_extension_col7
    };

    cuda_evaluator->add_to_relation<8>(common_lookup_elements, qm31{{1,0},{0,0}}, values);

    // Output: [offset0-32768, offset1-32768, offset2-32768, (1-op1_base_fp)]
    output_vec[0] = sub(offset0_col0, M31_32768);
    output_vec[1] = sub(offset1_col1, M31_32768);
    output_vec[2] = sub(offset2_col2, M31_32768);
    output_vec[3] = sub(M31_1, op1_base_fp_col5);
}

#endif // EVALUATE_DECODE_INSTRUCTION_H
