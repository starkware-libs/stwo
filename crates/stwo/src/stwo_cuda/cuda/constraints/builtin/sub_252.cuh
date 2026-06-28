#ifndef SUB_252_H
#define SUB_252_H

#include "fields.cuh"
#include "utils.cuh"
#include "../relations.cuh"
#include "../evaluate_verify_add_252.cuh"
#include "add_252.cuh"  // For range_check_mem_value_n_28

// CUDA version of Sub252::evaluate
// Translated from cairo-air/src/components/subroutines/sub_252.rs
// 252-bit field subtraction: (c - a) mod p, p = 2^252 + 17*2^192 + 1
//
// Implementation: verify a + result = c to prove result = c - a
// If c < a, then add p (via sub_p_bit flag)

// Main Sub252 function
// Computes: result = (c - a) mod p where p = 2^252 + 17*2^192 + 1
// Verifies via: a + result + sub_p_bit * p = c
template<typename EvaluatorT>
DEVICE_FORCEINLINE void sub_252_evaluate(
    const m31 input_c[28], // Minuend (minuend)
    const m31 input_a[28], // Subtrahend (subtrahend)
    const m31 result[28],    // Result of subtraction (28 limbs)
    const m31 sub_p_bit,     // Boolean: whether we added p (when c < a)
    const CommonLookupElements& common_lookup_elements,
    EvaluatorT* cuda_evaluator
) {
    // Step 1: Range check the result limbs
    range_check_mem_value_n_28(
        result, common_lookup_elements,
        cuda_evaluator
    );

    // Step 2: Verify the subtraction is correct
    // Verifies: a + result + sub_p_bit * p = c
    // This is equivalent to: result = c - a (mod p)
    // When c < a, we need sub_p_bit=1 to add p to make result positive
    evaluate_verify_add_252(
        // First operand: a
        input_a[0], input_a[1], input_a[2], input_a[3], input_a[4], input_a[5], input_a[6], input_a[7],
        input_a[8], input_a[9], input_a[10], input_a[11], input_a[12], input_a[13], input_a[14], input_a[15],
        input_a[16], input_a[17], input_a[18], input_a[19], input_a[20], input_a[21], input_a[22], input_a[23],
        input_a[24], input_a[25], input_a[26], input_a[27],
        // Second operand: result
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
        result[8], result[9], result[10], result[11], result[12], result[13], result[14], result[15],
        result[16], result[17], result[18], result[19], result[20], result[21], result[22], result[23],
        result[24], result[25], result[26], result[27],
        // Expected sum: c
        input_c[0], input_c[1], input_c[2], input_c[3], input_c[4], input_c[5], input_c[6], input_c[7],
        input_c[8], input_c[9], input_c[10], input_c[11], input_c[12], input_c[13], input_c[14], input_c[15],
        input_c[16], input_c[17], input_c[18], input_c[19], input_c[20], input_c[21], input_c[22], input_c[23],
        input_c[24], input_c[25], input_c[26], input_c[27],
        // Borrow flag (inverted from add: here it means we added p)
        sub_p_bit,
        cuda_evaluator
    );
}

#endif // SUB_252_H
