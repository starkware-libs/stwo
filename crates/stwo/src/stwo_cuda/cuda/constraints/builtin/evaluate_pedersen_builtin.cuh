/*
============================================
Pedersen Builtin CUDA AIR Evaluator Header
============================================

Component: PedersenBuiltin
translated from: cairo-air/src/components/pedersen_builtin.rs
AIR version: 54d95c0d

Functionality:
- Top-level Pedersen builtin component for Cairo programs
- Handles memory-mapped Pedersen hash operations
- Calls PartialEcMul for actual EC scalar multiplication
- Verifies memory reads and validates field element bounds

Data Structure:
- 351 trace columns (inputs, outputs, intermediate values)
- 5 relation types with 12 total uses

Relation Lookups:
- MemoryAddressToId: 3 uses
- MemoryIdToBig: 3 uses
- PartialEcMul: 4 uses (calls EC multiplication component)
- RangeCheck_5_4: 2 uses
- RangeCheck_8: 4 uses
============================================
*/

#ifndef EVALUATE_PEDERSEN_BUILTIN_H
#define EVALUATE_PEDERSEN_BUILTIN_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

struct PedersenBuiltin_Claim {
    unsigned log_size;
    unsigned pedersen_builtin_segment_start;
};

struct PedersenBuiltin_Eval {
    unsigned eval_id;
    PedersenBuiltin_Claim claim;
    CommonLookupElements common_lookup_elements;
};

extern "C"
void evaluate_pedersen_builtin(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations,
    unsigned trace0_evaluations_len,
    m31 **trace1_evaluations,
    unsigned trace1_evaluations_len,
    m31 **trace2_evaluations,
    unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size,
    unsigned int eval_domain_log_size,
    unsigned int number_of_columns,
    unsigned int logup_counts,
    void *eval,
    qm31 cumsum_shift,
    bool should_accumulate,
    bool use_assert_evaluator,
    cudaStream_t stream
);

#endif // EVALUATE_PEDERSEN_BUILTIN_H
