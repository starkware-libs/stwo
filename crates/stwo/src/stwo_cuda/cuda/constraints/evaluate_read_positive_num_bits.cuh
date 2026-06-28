#ifndef EVALUATE_READ_POSITIVE_NUM_BITS_H
#define EVALUATE_READ_POSITIVE_NUM_BITS_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include "evaluate_range_check_last_limb_bits_in_ms_limb.cuh"

// NOTE:
// This file provides the CUDA implementation for ReadPositiveNumBits_* components, for reading signed/unsigned integers from memory.
// It uses MemoryAddressToId / MemoryIdToBig / RangeCheck relations to record logup constraints.

template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_read_positive_num_bits_29(
    m31 read_positive_num_bits_29_input,
    m31 id_col0,
    m31 value_limb_0_col1,
    m31 value_limb_1_col2,
    m31 value_limb_2_col3,
    m31 value_limb_3_col4,
    m31 partial_limb_msb_col5,
    m31 *output_vec, // len: 29
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_0 = m31(0);

    // (1) address-to-id lookup
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, read_positive_num_bits_29_input, id_col0};
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (2) range check for last limb (2 bits) and msb
    evaluate_range_check_last_limb_bits_in_ms_limb_2(
        value_limb_3_col4,
        partial_limb_msb_col5,
        cuda_evaluator
    );

    // (3) id-to-big lookup
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col1, value_limb_1_col2, value_limb_2_col3, value_limb_3_col4
        };
        cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    output_vec[0]  = value_limb_0_col1;
    output_vec[1]  = value_limb_1_col2;
    output_vec[2]  = value_limb_2_col3;
    output_vec[3]  = value_limb_3_col4;
    output_vec[4]  = M31_0;
    output_vec[5]  = M31_0;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = M31_0;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
    output_vec[19] = M31_0;
    output_vec[20] = M31_0;
    output_vec[21] = M31_0;
    output_vec[22] = M31_0;
    output_vec[23] = M31_0;
    output_vec[24] = M31_0;
    output_vec[25] = M31_0;
    output_vec[26] = M31_0;
    output_vec[27] = M31_0;
    output_vec[28] = id_col0;
}

// 99-bit version: corresponds to CPU-side ReadPositiveNumBits99 + ReadPositiveKnownIdNumBits99.
// This function directly inlines two child programs:
// - First: MemoryAddressToId(addr -> id)
// - Then: MemoryIdToBig(id, 11 limbs, remaining filled with 0), total 29 columns.
template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_read_positive_num_bits_99(
    m31 read_positive_num_bits_99_input,
    m31 id_col0,
    m31 value_limb_0_col1,
    m31 value_limb_1_col2,
    m31 value_limb_2_col3,
    m31 value_limb_3_col4,
    m31 value_limb_4_col5,
    m31 value_limb_5_col6,
    m31 value_limb_6_col7,
    m31 value_limb_7_col8,
    m31 value_limb_8_col9,
    m31 value_limb_9_col10,
    m31 value_limb_10_col11,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    // address -> id
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, read_positive_num_bits_99_input, id_col0};
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // id -> big (11 limbs, remaining filled with 0)
    {
        m31 zero = 0;
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col1,
            value_limb_1_col2,
            value_limb_2_col3,
            value_limb_3_col4,
            value_limb_4_col5,
            value_limb_5_col6,
            value_limb_6_col7,
            value_limb_7_col8,
            value_limb_8_col9,
            value_limb_9_col10,
            value_limb_10_col11,
            zero, zero, zero, zero, zero, zero, zero, zero,
            zero, zero, zero, zero, zero, zero, zero, zero
        };
        cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }
}

template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_read_positive_num_bits_27(
    m31 read_positive_num_bits_27_input,
    m31 id_col0,
    m31 value_limb_0_col1,
    m31 value_limb_1_col2,
    m31 value_limb_2_col3,
    m31 *output_vec, // len: 29
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_0 = m31(0);

    // (1) address-to-id lookup
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, read_positive_num_bits_27_input, id_col0};
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (2) id-to-big lookup
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col1, value_limb_1_col2, value_limb_2_col3
        };
        cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    output_vec[0]  = value_limb_0_col1;
    output_vec[1]  = value_limb_1_col2;
    output_vec[2]  = value_limb_2_col3;
    output_vec[3]  = M31_0;
    output_vec[4]  = M31_0;
    output_vec[5]  = M31_0;
    output_vec[6]  = M31_0;
    output_vec[7]  = M31_0;
    output_vec[8]  = M31_0;
    output_vec[9]  = M31_0;
    output_vec[10] = M31_0;
    output_vec[11] = M31_0;
    output_vec[12] = M31_0;
    output_vec[13] = M31_0;
    output_vec[14] = M31_0;
    output_vec[15] = M31_0;
    output_vec[16] = M31_0;
    output_vec[17] = M31_0;
    output_vec[18] = M31_0;
    output_vec[19] = M31_0;
    output_vec[20] = M31_0;
    output_vec[21] = M31_0;
    output_vec[22] = M31_0;
    output_vec[23] = M31_0;
    output_vec[24] = M31_0;
    output_vec[25] = M31_0;
    output_vec[26] = M31_0;
    output_vec[27] = M31_0;
    output_vec[28] = id_col0;
}

template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_read_positive_num_bits_252(
    m31 read_positive_num_bits_252_input,
    m31 id_col0,
    m31 value_limb_0_col1,
    m31 value_limb_1_col2,
    m31 value_limb_2_col3,
    m31 value_limb_3_col4,
    m31 value_limb_4_col5,
    m31 value_limb_5_col6,
    m31 value_limb_6_col7,
    m31 value_limb_7_col8,
    m31 value_limb_8_col9,
    m31 value_limb_9_col10,
    m31 value_limb_10_col11,
    m31 value_limb_11_col12,
    m31 value_limb_12_col13,
    m31 value_limb_13_col14,
    m31 value_limb_14_col15,
    m31 value_limb_15_col16,
    m31 value_limb_16_col17,
    m31 value_limb_17_col18,
    m31 value_limb_18_col19,
    m31 value_limb_19_col20,
    m31 value_limb_20_col21,
    m31 value_limb_21_col22,
    m31 value_limb_22_col23,
    m31 value_limb_23_col24,
    m31 value_limb_24_col25,
    m31 value_limb_25_col26,
    m31 value_limb_26_col27,
    m31 value_limb_27_col28,
    m31 *output_vec, // len: 29
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    // (1) address-to-id lookup
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, read_positive_num_bits_252_input, id_col0};
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{1}, values);
    }
    // (2) id-to-big lookup
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col1,  value_limb_1_col2,  value_limb_2_col3,  value_limb_3_col4,
            value_limb_4_col5,  value_limb_5_col6,  value_limb_6_col7,  value_limb_7_col8,
            value_limb_8_col9,  value_limb_9_col10, value_limb_10_col11, value_limb_11_col12,
            value_limb_12_col13, value_limb_13_col14, value_limb_14_col15, value_limb_15_col16,
            value_limb_16_col17, value_limb_17_col18, value_limb_18_col19, value_limb_19_col20,
            value_limb_20_col21, value_limb_21_col22, value_limb_22_col23, value_limb_23_col24,
            value_limb_24_col25, value_limb_25_col26, value_limb_26_col27, value_limb_27_col28
        };
        cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    output_vec[0]  = value_limb_0_col1;
    output_vec[1]  = value_limb_1_col2;
    output_vec[2]  = value_limb_2_col3;
    output_vec[3]  = value_limb_3_col4;
    output_vec[4]  = value_limb_4_col5;
    output_vec[5]  = value_limb_5_col6;
    output_vec[6]  = value_limb_6_col7;
    output_vec[7]  = value_limb_7_col8;
    output_vec[8]  = value_limb_8_col9;
    output_vec[9]  = value_limb_9_col10;
    output_vec[10] = value_limb_10_col11;
    output_vec[11] = value_limb_11_col12;
    output_vec[12] = value_limb_12_col13;
    output_vec[13] = value_limb_13_col14;
    output_vec[14] = value_limb_14_col15;
    output_vec[15] = value_limb_15_col16;
    output_vec[16] = value_limb_16_col17;
    output_vec[17] = value_limb_17_col18;
    output_vec[18] = value_limb_18_col19;
    output_vec[19] = value_limb_19_col20;
    output_vec[20] = value_limb_20_col21;
    output_vec[21] = value_limb_21_col22;
    output_vec[22] = value_limb_22_col23;
    output_vec[23] = value_limb_23_col24;
    output_vec[24] = value_limb_24_col25;
    output_vec[25] = value_limb_25_col26;
    output_vec[26] = value_limb_26_col27;
    output_vec[27] = value_limb_27_col28;
    output_vec[28] = id_col0;
}

template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_read_positive_num_bits_36(
    m31 read_positive_num_bits_36_input,
    m31 id_col0,
    m31 value_limb_0_col1,
    m31 value_limb_1_col2,
    m31 value_limb_2_col3,
    m31 value_limb_3_col4,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_0 = m31(0);

    // (1) address-to-id lookup
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, read_positive_num_bits_36_input, id_col0};
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (2) id-to-big lookup
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col1, value_limb_1_col2, value_limb_2_col3, value_limb_3_col4,
            M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0,
            M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0
        };
        cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }
}

template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_read_positive_num_bits_72(
    m31 read_positive_num_bits_72_input,
    m31 id_col0,
    m31 value_limb_0_col1,
    m31 value_limb_1_col2,
    m31 value_limb_2_col3,
    m31 value_limb_3_col4,
    m31 value_limb_4_col5,
    m31 value_limb_5_col6,
    m31 value_limb_6_col7,
    m31 value_limb_7_col8,
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_0 = m31(0);

    // (1) address-to-id lookup
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, read_positive_num_bits_72_input, id_col0};
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }

    // (2) id-to-big lookup
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col1, value_limb_1_col2, value_limb_2_col3, value_limb_3_col4,
            value_limb_4_col5, value_limb_5_col6, value_limb_6_col7, value_limb_7_col8,
            M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0, M31_0
        };
        cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{m31(1), m31(0)}, values);
    }
}

#endif // EVALUATE_READ_POSITIVE_NUM_BITS_H
