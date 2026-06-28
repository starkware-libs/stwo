// ============================================================================
// Felt252UnpackFrom27 CUDA Subroutine
// ============================================================================
//
// CUDA version of felt_252_unpack_from_27 subroutine
// Translated from cairo-air/src/components/subroutines/felt_252_unpack_from_27.rs
//
// ## Overview
// This subroutine unpacks 10 input limbs (27-bit each) into 18 smaller limbs
// (some 9-bit, some 18-bit) by splitting them at specific bit boundaries.
//
// ## Input
// - 10 input limbs (27 bits each)
//
// ## Output
// - 18 unpacked limbs via columns (alternating 9-bit and 9-bit pairs)
// - Returns 10 intermediate values for MemVerify (NOT constraints!)
//
// ## Important Note
// This function does NOT add constraints. It only computes intermediate values
// that are passed to MemVerify. The Rust equivalent also just returns these
// values without calling add_constraint.
//
// ============================================================================

#ifndef EVALUATE_FELT_252_UNPACK_FROM_27_CUH
#define EVALUATE_FELT_252_UNPACK_FROM_27_CUH

#include "fields.cuh"
#include "utils.cuh"

// ============================================================================
// Felt252UnpackFrom27 Evaluation Function
// ============================================================================
// Unpacks 27-bit limbs into smaller 9-bit components
//
// For each of the first 9 input limbs:
//   input_limb = unpacked_low (9 bits) + unpacked_high (9 bits) * 512 + remainder * 8192
//   Intermediate value: (input_limb - unpacked_low - unpacked_high * 512) * 8192
//
// The 10th input limb is passed through unchanged
//
// Note: These are intermediate values, NOT constraints!
template <typename EvaluatorT>
__device__ void felt_252_unpack_from_27_evaluate(
    // Input: 10 limbs (27 bits each)
    m31 input_limb_0,
    m31 input_limb_1,
    m31 input_limb_2,
    m31 input_limb_3,
    m31 input_limb_4,
    m31 input_limb_5,
    m31 input_limb_6,
    m31 input_limb_7,
    m31 input_limb_8,
    m31 input_limb_9,
    // Output: 18 unpacked limbs (9 bits each, paired)
    m31 unpacked_limb_0,
    m31 unpacked_limb_1,
    m31 unpacked_limb_3,
    m31 unpacked_limb_4,
    m31 unpacked_limb_6,
    m31 unpacked_limb_7,
    m31 unpacked_limb_9,
    m31 unpacked_limb_10,
    m31 unpacked_limb_12,
    m31 unpacked_limb_13,
    m31 unpacked_limb_15,
    m31 unpacked_limb_16,
    m31 unpacked_limb_18,
    m31 unpacked_limb_19,
    m31 unpacked_limb_21,
    m31 unpacked_limb_22,
    m31 unpacked_limb_24,
    m31 unpacked_limb_25,
    // Output array: 10 computed intermediate values for MemVerify
    m31 *output_limbs,
    EvaluatorT *cuda_evaluator  // Not used for constraints, kept for API consistency
) {
    const m31 M31_512 = m31(512);
    const m31 M31_8192 = m31(8192);

    // Compute intermediate value 0
    // (input_limb_0 - unpacked_0 - unpacked_1 * 512) * 8192
    output_limbs[0] = mul(
        sub(sub(input_limb_0, unpacked_limb_0), mul(unpacked_limb_1, M31_512)),
        M31_8192
    );

    // Compute intermediate value 1
    output_limbs[1] = mul(
        sub(sub(input_limb_1, unpacked_limb_3), mul(unpacked_limb_4, M31_512)),
        M31_8192
    );

    // Compute intermediate value 2
    output_limbs[2] = mul(
        sub(sub(input_limb_2, unpacked_limb_6), mul(unpacked_limb_7, M31_512)),
        M31_8192
    );

    // Compute intermediate value 3
    output_limbs[3] = mul(
        sub(sub(input_limb_3, unpacked_limb_9), mul(unpacked_limb_10, M31_512)),
        M31_8192
    );

    // Compute intermediate value 4
    output_limbs[4] = mul(
        sub(sub(input_limb_4, unpacked_limb_12), mul(unpacked_limb_13, M31_512)),
        M31_8192
    );

    // Compute intermediate value 5
    output_limbs[5] = mul(
        sub(sub(input_limb_5, unpacked_limb_15), mul(unpacked_limb_16, M31_512)),
        M31_8192
    );

    // Compute intermediate value 6
    output_limbs[6] = mul(
        sub(sub(input_limb_6, unpacked_limb_18), mul(unpacked_limb_19, M31_512)),
        M31_8192
    );

    // Compute intermediate value 7
    output_limbs[7] = mul(
        sub(sub(input_limb_7, unpacked_limb_21), mul(unpacked_limb_22, M31_512)),
        M31_8192
    );

    // Compute intermediate value 8
    output_limbs[8] = mul(
        sub(sub(input_limb_8, unpacked_limb_24), mul(unpacked_limb_25, M31_512)),
        M31_8192
    );

    // Limb 9 is passed through unchanged
    output_limbs[9] = input_limb_9;
}

#endif // EVALUATE_FELT_252_UNPACK_FROM_27_CUH
