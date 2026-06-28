#ifndef DIV_252_H
#define DIV_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "add_252.cuh"  // For range_check_mem_value_n_28
#include "verify_mul_252.cuh"

// CUDA version Div252::evaluate
// translated from cairo-air/src/components/subroutines/div_252.rs
// 252-bit field division: (a / b) mod p, verified via result * b = a
//
// Structure (matching Rust Div252::evaluate):
// 1. Range check the result (not dividend!) using RangeCheckMemValueN28
// 2. Verify the multiplication: divisor * result = dividend using VerifyMul252

template<typename EvaluatorT>
DEVICE_FORCEINLINE void div_252_evaluate(
    const m31 input_a[28], // Dividend (what we're dividing)
    const m31 input_b[28], // Divisor (what we're dividing by)
    const m31 result[28],  // Result = input_a / input_b
    const m31 k,           // Quotient factor for verification
    const m31 carry[27],   // Carry values for verification
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    // Step 1: Range check the RESULT (matching Rust Div252 line 92-131)
    // This ensures the division result is valid
    range_check_mem_value_n_28(
        result, common_lookup_elements,
        cuda_evaluator
    );

    // Step 2: Verify division via multiplication: divisor * result = dividend
    // Rust Div252 calls VerifyMul252 with (input_a=divisor, div_res=result, input_c=dividend)
    // This verifies: divisor * result = dividend, i.e., result = dividend / divisor
    verify_mul_252_evaluate(
        input_b, result, input_a, k, carry,
        common_lookup_elements,
        cuda_evaluator
    );
}

#endif // DIV_252_H
