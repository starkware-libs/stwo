#ifndef FELT_252_UNPACK_FROM_27_RANGE_CHECK_OUTPUT_H
#define FELT_252_UNPACK_FROM_27_RANGE_CHECK_OUTPUT_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "add_252.cuh"  // For range_check_mem_value_n_28

// CUDA version Felt252UnpackFrom27RangeCheckOutput::evaluate
// translated from cairo-air/src/components/subroutines/felt_252_unpack_from_27_range_check_output.rs
// AIR version 54d95c0d
//
// Unpacks 10 27-bit limbs into 28 9-bit limbs and range checks them.
// Input: 10 limbs (27-bit each) + 18 unpacked columns (9-bit each)
// Output: 10 computed limbs (the missing limbs 2,5,8,11,14,17,20,23,26,27)
//
// The unpacking formula for computed limbs:
// limb_i = (input[i/3] - unpacked[i] - unpacked[i+1] * 512) * 8192
// where i is 0,1,2,... and mapping to actual indices 2,5,8,...

template<typename EvaluatorT>
DEVICE_FORCEINLINE void felt_252_unpack_from_27_range_check_output_evaluate(
    const m31 input[10],           // 10 27-bit input limbs
    const m31 unpacked_limb_0,     // 9-bit unpacked limb 0
    const m31 unpacked_limb_1,     // 9-bit unpacked limb 1
    const m31 unpacked_limb_3,     // 9-bit unpacked limb 3
    const m31 unpacked_limb_4,     // 9-bit unpacked limb 4
    const m31 unpacked_limb_6,     // 9-bit unpacked limb 6
    const m31 unpacked_limb_7,     // 9-bit unpacked limb 7
    const m31 unpacked_limb_9,     // 9-bit unpacked limb 9
    const m31 unpacked_limb_10,    // 9-bit unpacked limb 10
    const m31 unpacked_limb_12,    // 9-bit unpacked limb 12
    const m31 unpacked_limb_13,    // 9-bit unpacked limb 13
    const m31 unpacked_limb_15,    // 9-bit unpacked limb 15
    const m31 unpacked_limb_16,    // 9-bit unpacked limb 16
    const m31 unpacked_limb_18,    // 9-bit unpacked limb 18
    const m31 unpacked_limb_19,    // 9-bit unpacked limb 19
    const m31 unpacked_limb_21,    // 9-bit unpacked limb 21
    const m31 unpacked_limb_22,    // 9-bit unpacked limb 22
    const m31 unpacked_limb_24,    // 9-bit unpacked limb 24
    const m31 unpacked_limb_25,    // 9-bit unpacked limb 25
    const CommonLookupElements& common_lookup_elements,
    m31 output[10],               // Output: computed limbs 2,5,8,11,14,17,20,23,26,27
    EvaluatorT* cuda_evaluator
) {
    const m31 M31_512 = m31(512);
    const m31 M31_8192 = m31(8192);

    // Compute the 10 derived limbs (positions 2,5,8,11,14,17,20,23,26,27)
    // Formula: limb = (input - low_limb - mid_limb * 512) * 8192

    // limb 2: from input[0], unpacked_limb_0, unpacked_limb_1
    output[0] = mul(sub(sub(input[0], unpacked_limb_0), mul(unpacked_limb_1, M31_512)), M31_8192);

    // limb 5: from input[1], unpacked_limb_3, unpacked_limb_4
    output[1] = mul(sub(sub(input[1], unpacked_limb_3), mul(unpacked_limb_4, M31_512)), M31_8192);

    // limb 8: from input[2], unpacked_limb_6, unpacked_limb_7
    output[2] = mul(sub(sub(input[2], unpacked_limb_6), mul(unpacked_limb_7, M31_512)), M31_8192);

    // limb 11: from input[3], unpacked_limb_9, unpacked_limb_10
    output[3] = mul(sub(sub(input[3], unpacked_limb_9), mul(unpacked_limb_10, M31_512)), M31_8192);

    // limb 14: from input[4], unpacked_limb_12, unpacked_limb_13
    output[4] = mul(sub(sub(input[4], unpacked_limb_12), mul(unpacked_limb_13, M31_512)), M31_8192);

    // limb 17: from input[5], unpacked_limb_15, unpacked_limb_16
    output[5] = mul(sub(sub(input[5], unpacked_limb_15), mul(unpacked_limb_16, M31_512)), M31_8192);

    // limb 20: from input[6], unpacked_limb_18, unpacked_limb_19
    output[6] = mul(sub(sub(input[6], unpacked_limb_18), mul(unpacked_limb_19, M31_512)), M31_8192);

    // limb 23: from input[7], unpacked_limb_21, unpacked_limb_22
    output[7] = mul(sub(sub(input[7], unpacked_limb_21), mul(unpacked_limb_22, M31_512)), M31_8192);

    // limb 26: from input[8], unpacked_limb_24, unpacked_limb_25
    output[8] = mul(sub(sub(input[8], unpacked_limb_24), mul(unpacked_limb_25, M31_512)), M31_8192);

    // limb 27: just input[9]
    output[9] = input[9];

    // Build the full 28-limb array for range checking
    m31 full_limbs[28];
    full_limbs[0] = unpacked_limb_0;
    full_limbs[1] = unpacked_limb_1;
    full_limbs[2] = output[0];  // computed limb 2
    full_limbs[3] = unpacked_limb_3;
    full_limbs[4] = unpacked_limb_4;
    full_limbs[5] = output[1];  // computed limb 5
    full_limbs[6] = unpacked_limb_6;
    full_limbs[7] = unpacked_limb_7;
    full_limbs[8] = output[2];  // computed limb 8
    full_limbs[9] = unpacked_limb_9;
    full_limbs[10] = unpacked_limb_10;
    full_limbs[11] = output[3];  // computed limb 11
    full_limbs[12] = unpacked_limb_12;
    full_limbs[13] = unpacked_limb_13;
    full_limbs[14] = output[4];  // computed limb 14
    full_limbs[15] = unpacked_limb_15;
    full_limbs[16] = unpacked_limb_16;
    full_limbs[17] = output[5];  // computed limb 17
    full_limbs[18] = unpacked_limb_18;
    full_limbs[19] = unpacked_limb_19;
    full_limbs[20] = output[6];  // computed limb 20
    full_limbs[21] = unpacked_limb_21;
    full_limbs[22] = unpacked_limb_22;
    full_limbs[23] = output[7];  // computed limb 23
    full_limbs[24] = unpacked_limb_24;
    full_limbs[25] = unpacked_limb_25;
    full_limbs[26] = output[8];  // computed limb 26
    full_limbs[27] = output[9];  // computed limb 27

    // Range check all 28 limbs using RangeCheckMemValueN28
    range_check_mem_value_n_28(
        full_limbs,
        common_lookup_elements,
        cuda_evaluator
    );
}

#endif // FELT_252_UNPACK_FROM_27_RANGE_CHECK_OUTPUT_H
