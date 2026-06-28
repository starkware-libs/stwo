#ifndef EVALUATE_READ_SMALL_H
#define EVALUATE_READ_SMALL_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// CUDA version CondDecodeSmallSign::evaluate
template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_cond_decode_small_sign(
    const m31 cond_decode_small_sign_input[29],
    m31 msb_col0,
    m31 mid_limbs_set_col1,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_1 = m31(1);

    // msb is a bit.
    cuda_evaluator->add_constraint(mul(msb_col0, sub(msb_col0, M31_1)));
    // mid_limbs_set is a bit.
    cuda_evaluator->add_constraint(mul(mid_limbs_set_col1, sub(mid_limbs_set_col1, M31_1)));
    // Cannot have msb equals 0 and mid_limbs_set equals 1.
    cuda_evaluator->add_constraint(
        mul(
            mul(cond_decode_small_sign_input[28], mid_limbs_set_col1),
            sub(msb_col0, M31_1)
        )
    );
}

// CUDA version ReadSmall::evaluate，translated from
// cairo-air/src/components/subroutines/read_small.rs
template <typename EvaluatorT>
DEVICE_FORCEINLINE void evaluate_read_small(
    m31 read_small_input,
    m31 id_col0,
    m31 msb_col1,
    m31 mid_limbs_set_col2,
    m31 value_limb_0_col3,
    m31 value_limb_1_col4,
    m31 value_limb_2_col5,
    m31 remainder_bits_col6,
    m31 partial_limb_msb_col7,
    m31 *output_vec, // 2 elements: [decoded_value, id]
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT *cuda_evaluator
) {
    m31 M31_0 = m31(0);
    m31 M31_1 = m31(1);
    m31 M31_2 = m31(2);
    m31 M31_134217728 = m31(134217728);
    m31 M31_136 = m31(136);
    m31 M31_256 = m31(256);
    m31 M31_262144 = m31(262144);
    m31 M31_508 = m31(508);
    m31 M31_511 = m31(511);
    m31 M31_512 = m31(512);
    m31 M31_536870912 = m31(536870912);

    // ReadId::evaluate: memory_address_to_id lookup
    {
        m31 values[3] = {MEMORY_ADDRESS_TO_ID_RELATION_ID, read_small_input, id_col0};
        cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // CondDecodeSmallSign::evaluate([1], msb, mid_limbs_set)
    {
        m31 cond_decode_small_sign_input[29];
        for (int i = 0; i < 28; ++i) {
            cond_decode_small_sign_input[i] = M31_0;
        }
        cond_decode_small_sign_input[28] = M31_1;
        evaluate_cond_decode_small_sign<EvaluatorT>(
            cond_decode_small_sign_input,
            msb_col1,
            mid_limbs_set_col2,
            cuda_evaluator
        );
    }

    // CondRangeCheck2::evaluate([remainder_bits, 1], partial_limb_msb)
    {
        // msb is a bit or condition is 0; here condition is always 1.
        cuda_evaluator->add_constraint(
            mul(
                mul(partial_limb_msb_col7, sub(M31_1, partial_limb_msb_col7)),
                M31_1
            )
        );
        m31 partial_limb_bit_before_msb =
            sub(remainder_bits_col6, mul(partial_limb_msb_col7, M31_2));
        cuda_evaluator->add_constraint(
            mul(
                mul(partial_limb_bit_before_msb,
                    sub(M31_1, partial_limb_bit_before_msb)),
                M31_1
            )
        );
    }

    // memory_id_to_big lookup
    {
        m31 values[30] = {
            MEMORY_ID_TO_BIG_RELATION_ID,
            id_col0,
            value_limb_0_col3,
            value_limb_1_col4,
            value_limb_2_col5,
            add(remainder_bits_col6, mul(mid_limbs_set_col2, M31_508)),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            mul(mid_limbs_set_col2, M31_511),
            sub(mul(M31_136, msb_col1), mid_limbs_set_col2),
            M31_0,
            M31_0,
            M31_0,
            M31_0,
            M31_0,
            mul(msb_col1, M31_256)
        };
        cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Output: decoded value (consistent with CPU version ReadSmall::evaluate return value)
    output_vec[0] =
        sub(
            sub(
                add(
                    add(
                        value_limb_0_col3,
                        mul(value_limb_1_col4, M31_512)
                    ),
                    add(
                        mul(value_limb_2_col5, M31_262144),
                        mul(remainder_bits_col6, M31_134217728)
                    )
                ),
                msb_col1
            ),
            mul(M31_536870912, mid_limbs_set_col2)
        );
    output_vec[1] = id_col0;
}


#endif // EVALUATE_READ_SMALL_H
