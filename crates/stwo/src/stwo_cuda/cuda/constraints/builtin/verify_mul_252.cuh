#ifndef VERIFY_MUL_252_H
#define VERIFY_MUL_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "double_karatsuba_n_7.cuh"

// CUDA version VerifyMul252::evaluate
// translated from cairo-air/src/components/subroutines/verify_mul_252.rs
// Verifies: a * b = c (mod p), where p = 2^251 + 17*2^192 + 1
//
// This implements the full modular reduction verification:
// 1. Compute 55-limb product via DoubleKaratsubaN7
// 2. Compute convolution difference: product - result
// 3. Compute modular reduction values using coefficients derived from p
// 4. Verify via carry chain with RangeCheck_20 lookups
//
// Relation ID pattern for carries (cycling through 8 variants):
//   k:       RANGE_CHECK_20_RELATION_ID
//   carry 0: RANGE_CHECK_20_B_RELATION_ID
//   carry 1: RANGE_CHECK_20_C_RELATION_ID
//   carry 2: RANGE_CHECK_20_D_RELATION_ID
//   carry 3: RANGE_CHECK_20_E_RELATION_ID
//   carry 4: RANGE_CHECK_20_F_RELATION_ID
//   carry 5: RANGE_CHECK_20_G_RELATION_ID
//   carry 6: RANGE_CHECK_20_H_RELATION_ID
//   carry 7: RANGE_CHECK_20_RELATION_ID
//   ... (repeats)
//
// All range check offsets use M31_524288 (= 2^19).

template<typename EvaluatorT>
DEVICE_FORCEINLINE void verify_mul_252_evaluate(
    const m31 input_a[28],   // First operand (28 9-bit limbs)
    const m31 input_b[28],   // Second operand (28 9-bit limbs)
    const m31 input_c[28],   // Expected result (28 9-bit limbs)
    const m31 k,             // Quotient factor
    const m31 carry[27],     // Carry values for verification
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    // Constants matching Rust implementation
    const m31 M31_2 = m31(2);
    const m31 M31_4 = m31(4);
    const m31 M31_8 = m31(8);
    const m31 M31_32 = m31(32);
    const m31 M31_64 = m31(64);
    const m31 M31_136 = m31(136);
    const m31 M31_256 = m31(256);
    const m31 M31_512 = m31(512);
    const m31 M31_524288 = m31(524288);   // 2^19

    // Step 1: Compute 55-limb product via DoubleKaratsubaN7
    m31 dk_input[56];
    for (int i = 0; i < 28; i++) {
        dk_input[i] = input_a[i];
        dk_input[28 + i] = input_b[i];
    }
    m31 product[55];
    double_karatsuba_n_7_evaluate(dk_input, product);

    // Step 2: Compute convolution difference (conv_tmp)
    // conv_tmp[0-27] = product[0-27] - input_c[0-27]
    // conv_tmp[28-54] = product[28-54]
    m31 conv[55];
    for (int i = 0; i < 28; i++) {
        conv[i] = sub(product[i], input_c[i]);
    }
    for (int i = 28; i < 55; i++) {
        conv[i] = product[i];
    }

    // Step 3: Compute modular reduction values (conv_mod)
    // These coefficients come from the modular reduction of p = 2^251 + 17*2^192 + 1
    m31 conv_mod[28];

    // limb 0: (32 * conv[0]) - (4 * conv[21]) + (8 * conv[49])
    conv_mod[0] = add(sub(mul(M31_32, conv[0]), mul(M31_4, conv[21])), mul(M31_8, conv[49]));

    // limb 1: conv[0] + (32 * conv[1]) - (4 * conv[22]) + (8 * conv[50])
    conv_mod[1] = add(sub(add(conv[0], mul(M31_32, conv[1])), mul(M31_4, conv[22])), mul(M31_8, conv[50]));

    // limb 2: conv[1] + (32 * conv[2]) - (4 * conv[23]) + (8 * conv[51])
    conv_mod[2] = add(sub(add(conv[1], mul(M31_32, conv[2])), mul(M31_4, conv[23])), mul(M31_8, conv[51]));

    // limb 3: conv[2] + (32 * conv[3]) - (4 * conv[24]) + (8 * conv[52])
    conv_mod[3] = add(sub(add(conv[2], mul(M31_32, conv[3])), mul(M31_4, conv[24])), mul(M31_8, conv[52]));

    // limb 4: conv[3] + (32 * conv[4]) - (4 * conv[25]) + (8 * conv[53])
    conv_mod[4] = add(sub(add(conv[3], mul(M31_32, conv[4])), mul(M31_4, conv[25])), mul(M31_8, conv[53]));

    // limb 5: conv[4] + (32 * conv[5]) - (4 * conv[26]) + (8 * conv[54])
    conv_mod[5] = add(sub(add(conv[4], mul(M31_32, conv[5])), mul(M31_4, conv[26])), mul(M31_8, conv[54]));

    // limb 6: conv[5] + (32 * conv[6]) - (4 * conv[27])
    conv_mod[6] = sub(add(conv[5], mul(M31_32, conv[6])), mul(M31_4, conv[27]));

    // limbs 7-20: (2 * conv[i-7]) + conv[i-1] + (32 * conv[i]) - (4 * conv[i+21])
    conv_mod[7] = sub(add(add(mul(M31_2, conv[0]), conv[6]), mul(M31_32, conv[7])), mul(M31_4, conv[28]));
    conv_mod[8] = sub(add(add(mul(M31_2, conv[1]), conv[7]), mul(M31_32, conv[8])), mul(M31_4, conv[29]));
    conv_mod[9] = sub(add(add(mul(M31_2, conv[2]), conv[8]), mul(M31_32, conv[9])), mul(M31_4, conv[30]));
    conv_mod[10] = sub(add(add(mul(M31_2, conv[3]), conv[9]), mul(M31_32, conv[10])), mul(M31_4, conv[31]));
    conv_mod[11] = sub(add(add(mul(M31_2, conv[4]), conv[10]), mul(M31_32, conv[11])), mul(M31_4, conv[32]));
    conv_mod[12] = sub(add(add(mul(M31_2, conv[5]), conv[11]), mul(M31_32, conv[12])), mul(M31_4, conv[33]));
    conv_mod[13] = sub(add(add(mul(M31_2, conv[6]), conv[12]), mul(M31_32, conv[13])), mul(M31_4, conv[34]));
    conv_mod[14] = sub(add(add(mul(M31_2, conv[7]), conv[13]), mul(M31_32, conv[14])), mul(M31_4, conv[35]));
    conv_mod[15] = sub(add(add(mul(M31_2, conv[8]), conv[14]), mul(M31_32, conv[15])), mul(M31_4, conv[36]));
    conv_mod[16] = sub(add(add(mul(M31_2, conv[9]), conv[15]), mul(M31_32, conv[16])), mul(M31_4, conv[37]));
    conv_mod[17] = sub(add(add(mul(M31_2, conv[10]), conv[16]), mul(M31_32, conv[17])), mul(M31_4, conv[38]));
    conv_mod[18] = sub(add(add(mul(M31_2, conv[11]), conv[17]), mul(M31_32, conv[18])), mul(M31_4, conv[39]));
    conv_mod[19] = sub(add(add(mul(M31_2, conv[12]), conv[18]), mul(M31_32, conv[19])), mul(M31_4, conv[40]));
    conv_mod[20] = sub(add(add(mul(M31_2, conv[13]), conv[19]), mul(M31_32, conv[20])), mul(M31_4, conv[41]));

    // limb 21: (2 * conv[14]) + conv[20] - (4 * conv[42]) + (64 * conv[49])
    conv_mod[21] = sub(add(add(mul(M31_2, conv[14]), conv[20]), mul(M31_64, conv[49])), mul(M31_4, conv[42]));

    // limb 22: (2 * conv[15]) - (4 * conv[43]) + (2 * conv[49]) + (64 * conv[50])
    conv_mod[22] = add(add(sub(mul(M31_2, conv[15]), mul(M31_4, conv[43])), mul(M31_2, conv[49])), mul(M31_64, conv[50]));

    // limb 23: (2 * conv[16]) - (4 * conv[44]) + (2 * conv[50]) + (64 * conv[51])
    conv_mod[23] = add(add(sub(mul(M31_2, conv[16]), mul(M31_4, conv[44])), mul(M31_2, conv[50])), mul(M31_64, conv[51]));

    // limb 24: (2 * conv[17]) - (4 * conv[45]) + (2 * conv[51]) + (64 * conv[52])
    conv_mod[24] = add(add(sub(mul(M31_2, conv[17]), mul(M31_4, conv[45])), mul(M31_2, conv[51])), mul(M31_64, conv[52]));

    // limb 25: (2 * conv[18]) - (4 * conv[46]) + (2 * conv[52]) + (64 * conv[53])
    conv_mod[25] = add(add(sub(mul(M31_2, conv[18]), mul(M31_4, conv[46])), mul(M31_2, conv[52])), mul(M31_64, conv[53]));

    // limb 26: (2 * conv[19]) - (4 * conv[47]) + (2 * conv[53]) + (64 * conv[54])
    conv_mod[26] = add(add(sub(mul(M31_2, conv[19]), mul(M31_4, conv[47])), mul(M31_2, conv[53])), mul(M31_64, conv[54]));

    // limb 27: (2 * conv[20]) - (4 * conv[48]) + (2 * conv[54])
    conv_mod[27] = add(sub(mul(M31_2, conv[20]), mul(M31_4, conv[48])), mul(M31_2, conv[54]));

    // Step 4: Range check k (k + 524288) with RANGE_CHECK_20_RELATION_ID
    {
        m31 values[2] = {RANGE_CHECK_20_RELATION_ID, add(k, M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Step 5: Carry chain constraints and range checks
    // carry_0 * 512 = conv_mod[0] - k
    cuda_evaluator->add_constraint(sub(mul(carry[0], M31_512), sub(conv_mod[0], k)));
    {
        m31 values[2] = {RANGE_CHECK_20_B_RELATION_ID, add(carry[0], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_1 * 512 = conv_mod[1] + carry_0
    cuda_evaluator->add_constraint(sub(mul(carry[1], M31_512), add(conv_mod[1], carry[0])));
    {
        m31 values[2] = {RANGE_CHECK_20_C_RELATION_ID, add(carry[1], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_2 * 512 = conv_mod[2] + carry_1
    cuda_evaluator->add_constraint(sub(mul(carry[2], M31_512), add(conv_mod[2], carry[1])));
    {
        m31 values[2] = {RANGE_CHECK_20_D_RELATION_ID, add(carry[2], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_3 * 512 = conv_mod[3] + carry_2
    cuda_evaluator->add_constraint(sub(mul(carry[3], M31_512), add(conv_mod[3], carry[2])));
    {
        m31 values[2] = {RANGE_CHECK_20_E_RELATION_ID, add(carry[3], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_4 * 512 = conv_mod[4] + carry_3
    cuda_evaluator->add_constraint(sub(mul(carry[4], M31_512), add(conv_mod[4], carry[3])));
    {
        m31 values[2] = {RANGE_CHECK_20_F_RELATION_ID, add(carry[4], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_5 * 512 = conv_mod[5] + carry_4
    cuda_evaluator->add_constraint(sub(mul(carry[5], M31_512), add(conv_mod[5], carry[4])));
    {
        m31 values[2] = {RANGE_CHECK_20_G_RELATION_ID, add(carry[5], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_6 * 512 = conv_mod[6] + carry_5
    cuda_evaluator->add_constraint(sub(mul(carry[6], M31_512), add(conv_mod[6], carry[5])));
    {
        m31 values[2] = {RANGE_CHECK_20_H_RELATION_ID, add(carry[6], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_7 * 512 = conv_mod[7] + carry_6
    cuda_evaluator->add_constraint(sub(mul(carry[7], M31_512), add(conv_mod[7], carry[6])));
    {
        m31 values[2] = {RANGE_CHECK_20_RELATION_ID, add(carry[7], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_8 * 512 = conv_mod[8] + carry_7
    cuda_evaluator->add_constraint(sub(mul(carry[8], M31_512), add(conv_mod[8], carry[7])));
    {
        m31 values[2] = {RANGE_CHECK_20_B_RELATION_ID, add(carry[8], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_9 * 512 = conv_mod[9] + carry_8
    cuda_evaluator->add_constraint(sub(mul(carry[9], M31_512), add(conv_mod[9], carry[8])));
    {
        m31 values[2] = {RANGE_CHECK_20_C_RELATION_ID, add(carry[9], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_10 * 512 = conv_mod[10] + carry_9
    cuda_evaluator->add_constraint(sub(mul(carry[10], M31_512), add(conv_mod[10], carry[9])));
    {
        m31 values[2] = {RANGE_CHECK_20_D_RELATION_ID, add(carry[10], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_11 * 512 = conv_mod[11] + carry_10
    cuda_evaluator->add_constraint(sub(mul(carry[11], M31_512), add(conv_mod[11], carry[10])));
    {
        m31 values[2] = {RANGE_CHECK_20_E_RELATION_ID, add(carry[11], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_12 * 512 = conv_mod[12] + carry_11
    cuda_evaluator->add_constraint(sub(mul(carry[12], M31_512), add(conv_mod[12], carry[11])));
    {
        m31 values[2] = {RANGE_CHECK_20_F_RELATION_ID, add(carry[12], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_13 * 512 = conv_mod[13] + carry_12
    cuda_evaluator->add_constraint(sub(mul(carry[13], M31_512), add(conv_mod[13], carry[12])));
    {
        m31 values[2] = {RANGE_CHECK_20_G_RELATION_ID, add(carry[13], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_14 * 512 = conv_mod[14] + carry_13
    cuda_evaluator->add_constraint(sub(mul(carry[14], M31_512), add(conv_mod[14], carry[13])));
    {
        m31 values[2] = {RANGE_CHECK_20_H_RELATION_ID, add(carry[14], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_15 * 512 = conv_mod[15] + carry_14
    cuda_evaluator->add_constraint(sub(mul(carry[15], M31_512), add(conv_mod[15], carry[14])));
    {
        m31 values[2] = {RANGE_CHECK_20_RELATION_ID, add(carry[15], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_16 * 512 = conv_mod[16] + carry_15
    cuda_evaluator->add_constraint(sub(mul(carry[16], M31_512), add(conv_mod[16], carry[15])));
    {
        m31 values[2] = {RANGE_CHECK_20_B_RELATION_ID, add(carry[16], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_17 * 512 = conv_mod[17] + carry_16
    cuda_evaluator->add_constraint(sub(mul(carry[17], M31_512), add(conv_mod[17], carry[16])));
    {
        m31 values[2] = {RANGE_CHECK_20_C_RELATION_ID, add(carry[17], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_18 * 512 = conv_mod[18] + carry_17
    cuda_evaluator->add_constraint(sub(mul(carry[18], M31_512), add(conv_mod[18], carry[17])));
    {
        m31 values[2] = {RANGE_CHECK_20_D_RELATION_ID, add(carry[18], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_19 * 512 = conv_mod[19] + carry_18
    cuda_evaluator->add_constraint(sub(mul(carry[19], M31_512), add(conv_mod[19], carry[18])));
    {
        m31 values[2] = {RANGE_CHECK_20_E_RELATION_ID, add(carry[19], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_20 * 512 = conv_mod[20] + carry_19
    cuda_evaluator->add_constraint(sub(mul(carry[20], M31_512), add(conv_mod[20], carry[19])));
    {
        m31 values[2] = {RANGE_CHECK_20_F_RELATION_ID, add(carry[20], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_21 SPECIAL: carry_21 * 512 = (conv_mod[21] - 136*k) + carry_20
    cuda_evaluator->add_constraint(sub(mul(carry[21], M31_512), add(sub(conv_mod[21], mul(M31_136, k)), carry[20])));
    {
        m31 values[2] = {RANGE_CHECK_20_G_RELATION_ID, add(carry[21], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_22 * 512 = conv_mod[22] + carry_21
    cuda_evaluator->add_constraint(sub(mul(carry[22], M31_512), add(conv_mod[22], carry[21])));
    {
        m31 values[2] = {RANGE_CHECK_20_H_RELATION_ID, add(carry[22], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_23 * 512 = conv_mod[23] + carry_22
    cuda_evaluator->add_constraint(sub(mul(carry[23], M31_512), add(conv_mod[23], carry[22])));
    {
        m31 values[2] = {RANGE_CHECK_20_RELATION_ID, add(carry[23], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_24 * 512 = conv_mod[24] + carry_23
    cuda_evaluator->add_constraint(sub(mul(carry[24], M31_512), add(conv_mod[24], carry[23])));
    {
        m31 values[2] = {RANGE_CHECK_20_B_RELATION_ID, add(carry[24], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_25 * 512 = conv_mod[25] + carry_24
    cuda_evaluator->add_constraint(sub(mul(carry[25], M31_512), add(conv_mod[25], carry[24])));
    {
        m31 values[2] = {RANGE_CHECK_20_C_RELATION_ID, add(carry[25], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // carry_26 * 512 = conv_mod[26] + carry_25
    cuda_evaluator->add_constraint(sub(mul(carry[26], M31_512), add(conv_mod[26], carry[25])));
    {
        m31 values[2] = {RANGE_CHECK_20_D_RELATION_ID, add(carry[26], M31_524288)};
        cuda_evaluator->template add_to_relation<2>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, values);
    }

    // Final constraint: (conv_mod[27] - 256*k) + carry_26 = 0
    cuda_evaluator->add_constraint(add(sub(conv_mod[27], mul(M31_256, k)), carry[26]));
}

#endif // VERIFY_MUL_252_H
