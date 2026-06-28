#ifndef ENCODE_OFFSETS_COMMON_H
#define ENCODE_OFFSETS_COMMON_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// CUDA version of EncodeOffsets child program.
// Corresponds to CPU-side components/subroutines/encode_offsets.rs::EncodeOffsets::evaluate.
template <typename EvaluatorT>
DEVICE_FORCEINLINE void encode_offsets_evaluate(
    m31 encode_offsets_input_offset0,
    m31 encode_offsets_input_offset1,
    m31 encode_offsets_input_offset2,
    m31 offset0_low_col0,
    m31 offset0_mid_col1,
    m31 offset1_low_col2,
    m31 offset1_mid_col3,
    m31 offset1_high_col4,
    m31 offset2_low_col5,
    m31 offset2_mid_col6,
    m31 offset2_high_col7,
    const CommonLookupElements& common_lookup_elements,
    m31 *out_limb_1,
    m31 *out_limb_3,
    EvaluatorT *cuda_evaluator
) {
    const m31 M31_128 = 128;
    const m31 M31_16 = 16;
    const m31 M31_2048 = 2048;
    const m31 M31_32 = 32;
    const m31 M31_4 = 4;
    const m31 M31_512 = 512;
    const m31 M31_8192 = 8192;

    // Reconstructed offset0 is correct.
    m31 offset0_reconstructed = add(
        offset0_low_col0,
        mul(offset0_mid_col1, M31_512)
    );
    m31 c0 = sub(offset0_reconstructed, encode_offsets_input_offset0);
    cuda_evaluator->add_constraint(c0);

    // Reconstructed offset1 is correct.
    m31 offset1_reconstructed = add(
        add(offset1_low_col2, mul(offset1_mid_col3, M31_4)),
        mul(offset1_high_col4, M31_2048)
    );
    m31 c1 = sub(offset1_reconstructed, encode_offsets_input_offset1);
    cuda_evaluator->add_constraint(c1);

    // Reconstructed offset2 is correct.
    m31 offset2_reconstructed = add(
        add(offset2_low_col5, mul(offset2_mid_col6, M31_16)),
        mul(offset2_high_col7, M31_8192)
    );
    m31 c2 = sub(offset2_reconstructed, encode_offsets_input_offset2);
    cuda_evaluator->add_constraint(c2);

    // RangeCheck_7_2_5 relation.
    {
        m31 values[4] = {
            RANGE_CHECK_7_2_5_RELATION_ID,
            offset0_mid_col1,
            offset1_low_col2,
            offset1_high_col4,
        };
        cuda_evaluator->add_to_relation<4>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // RangeCheck_4_3 relation.
    {
        m31 values[3] = {
            RANGE_CHECK_4_3_RELATION_ID,
            offset2_low_col5,
            offset2_high_col7,
        };
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Return limbs.
    *out_limb_1 = add(offset0_mid_col1, mul(offset1_low_col2, M31_128));
    *out_limb_3 = add(offset1_high_col4, mul(offset2_low_col5, M31_32));
}

#endif // ENCODE_OFFSETS_COMMON_H

