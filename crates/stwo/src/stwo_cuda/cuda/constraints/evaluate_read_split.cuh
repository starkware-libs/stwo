/*
============================================
ReadSplit CUDA Subroutine
============================================

Subroutine: ReadSplit
translated from: cairo-air/src/comptogethernts/subroutines/read_split.rs
AIR version: 54d95c0d

Functionality:
- Reads a 252-bit value from memory at given address
- Splits the most significant limb into low (5 bits) and high (4 bits) parts
- Verifies the split via RangeCheck_5_4
- Calls MemVerify to validate memory access

Inputs:
- read_split_input_address: Memory address to read from
- value_limb_0..26: The 27 limbs of the 252-bit value (9 bits each)
- ms_limb_low: Low 5 bits of most significant limb
- ms_limb_high: High 4 bits of most significant limb
- id: Memory ID

Output:
- Returns: ms_limb_high * 32 + ms_limb_low (reconstructed MS limb)

Relations:
- RangeCheck_5_4: 1 use (ms_limb_low, ms_limb_high)
- MemoryAddressToId: 1 use (via MemVerify)
- MemoryIdToBig: 1 use (via MemVerify)

Key Logic:
1. Range check the MS limb split (5+4 = 9 bits)
2. Verify memory contents via MemVerify
3. Return reconstructed MS limb value
============================================
*/

#ifndef EVALUATE_READ_SPLIT_CONSTRAINT_H
#define EVALUATE_READ_SPLIT_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"
#include "constraints/evaluate_mem_verify.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE m31 read_split_evaluate(
    m31 read_split_input_address,

    m31 value_limb_0,
    m31 value_limb_1,
    m31 value_limb_2,
    m31 value_limb_3,
    m31 value_limb_4,
    m31 value_limb_5,
    m31 value_limb_6,
    m31 value_limb_7,
    m31 value_limb_8,
    m31 value_limb_9,
    m31 value_limb_10,
    m31 value_limb_11,
    m31 value_limb_12,
    m31 value_limb_13,
    m31 value_limb_14,
    m31 value_limb_15,
    m31 value_limb_16,
    m31 value_limb_17,
    m31 value_limb_18,
    m31 value_limb_19,
    m31 value_limb_20,
    m31 value_limb_21,
    m31 value_limb_22,
    m31 value_limb_23,
    m31 value_limb_24,
    m31 value_limb_25,
    m31 value_limb_26,

    m31 ms_limb_low,
    m31 ms_limb_high,
    m31 id,

    const CommonLookupElements& common_lookup_elements,

    EvaluatorT *cuda_evaluator
) {
    // M31_32 = 2^5 (shift for combining high/low parts)
    const m31 M31_32 = 32;

    // ===================== RangeCheck_5_4: Verify MS limb split =====================
    // Verify that ms_limb_low is 5 bits and ms_limb_high is 4 bits
    m31 rc_values[3] = {
        RANGE_CHECK_5_4_RELATION_ID,
        ms_limb_low,
        ms_limb_high
    };
    cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1, 0}, {0, 0}}, rc_values);

    // ===================== Reconstruct MS limb =====================
    // ms_limb = ms_limb_high * 32 + ms_limb_low
    // This reconstructs the full 9-bit MS limb from the 5+4 split
    m31 reconstructed_ms_limb = add(mul(ms_limb_high, M31_32), ms_limb_low);

    // ===================== MemVerify: Verify memory read =====================
    // Call MemVerify to ensure the value at address matches the provided limbs
    // MemVerify expects 29 limbs: limbs 0-26, reconstructed MS limb (27), and address (28)
    mem_verify_evaluate<EvaluatorT>(
        read_split_input_address,  // limb 0: address
        value_limb_0,              // limb 1
        value_limb_1,              // limb 2
        value_limb_2,              // limb 3
        value_limb_3,              // limb 4
        value_limb_4,              // limb 5
        value_limb_5,              // limb 6
        value_limb_6,              // limb 7
        value_limb_7,              // limb 8
        value_limb_8,              // limb 9
        value_limb_9,              // limb 10
        value_limb_10,             // limb 11
        value_limb_11,             // limb 12
        value_limb_12,             // limb 13
        value_limb_13,             // limb 14
        value_limb_14,             // limb 15
        value_limb_15,             // limb 16
        value_limb_16,             // limb 17
        value_limb_17,             // limb 18
        value_limb_18,             // limb 19
        value_limb_19,             // limb 20
        value_limb_20,             // limb 21
        value_limb_21,             // limb 22
        value_limb_22,             // limb 23
        value_limb_23,             // limb 24
        value_limb_24,             // limb 25
        value_limb_25,             // limb 26
        value_limb_26,             // limb 27
        reconstructed_ms_limb,     // limb 28: reconstructed MS limb
        id,                        // memory ID
        common_lookup_elements,
        cuda_evaluator
    );

    // ===================== Return reconstructed MS limb =====================
    return reconstructed_ms_limb;
}

#endif // EVALUATE_READ_SPLIT_CONSTRAINT_H
