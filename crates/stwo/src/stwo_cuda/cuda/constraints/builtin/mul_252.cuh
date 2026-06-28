#ifndef MUL_252_H
#define MUL_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "add_252.cuh"  // For range_check_mem_value_n_28
#include "verify_mul_252.cuh"

// CUDA version Mul252::evaluate
// translated from cairo-air/src/components/subroutines/mul_252.rs
// 252-bit field multiplication: (a * b) mod p, p = 2^252 + 17*2^192 + 1
//
// Structure:
// 1. Range check the result using RangeCheckMemValueN28 (with RangeCheck_9_9 variants)
// 2. Verify the multiplication using VerifyMul252 (with RangeCheck_19 variants)

template<typename EvaluatorT>
DEVICE_FORCEINLINE void mul_252_evaluate(
    const m31 input_a[28],   // First operand (28 9-bit limbs)
    const m31 input_b[28],   // Second operand (28 9-bit limbs)
    const m31 result[28],    // Result of multiplication (28 9-bit limbs)
    const m31 k,             // Quotient factor
    const m31 carry[27],     // Carry values for verification
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    // Step 1: Range check result using RangeCheck_9_9 variants
    range_check_mem_value_n_28(
        result, common_lookup_elements,
        cuda_evaluator
    );

    // Step 2: Verify multiplication using VerifyMul252
    verify_mul_252_evaluate(
        input_a, input_b, result, k, carry,
        common_lookup_elements,
        cuda_evaluator
    );
}

#endif // MUL_252_H
