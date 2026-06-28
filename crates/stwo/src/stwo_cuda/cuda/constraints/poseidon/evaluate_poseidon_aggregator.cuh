/*
============================================
PoseidonAggregator CUDA Evaluator Header
============================================

Component: PoseidonAggregator
Translated from: cairo-air/src/components/poseidon_aggregator.rs

Trace columns: 342
Preprocessed columns: 1 (seq)

This component performs the core Poseidon aggregation:
1. 3x ReadPositiveKnownIdNumBits252 (reads 3 state words from memory by known ID)
2. Packs the 28 value limbs into 9 packed 27-bit limbs per state word (27 total)
3. PoseidonHadesPermutation on 30 packed inputs (9 + col33, 9 + col61, 9 + col89)
4. 3x Felt252UnpackFrom27 (unpacks output state)
5. 3x MemoryIdToBig lookups for the unpacked output states
6. 1x PoseidonAggregator PROVIDE relation

Relation uses per row:
  Cube252: 2, MemoryIdToBig: 6, Poseidon3PartialRoundsChain: 1,
  PoseidonFullRoundChain: 2, RangeCheck252Width27: 2,
  RangeCheck_3_3_3_3_3: 2, RangeCheck_4_4: 3, RangeCheck_4_4_4_4: 6

interaction_log_sizes = SECURE_EXTENSION_DEGREE * 14 = 56
============================================
*/

#ifndef EVALUATE_POSEIDON_AGGREGATOR_CUH
#define EVALUATE_POSEIDON_AGGREGATOR_CUH

#include "fields.cuh"
#include "logup.cuh"
#include "relations.cuh"

// PoseidonAggregator Evaluation Structure
// Must match the Rust Eval structure layout: eval_id, claim (log_size), common_lookup_elements
struct PoseidonAggregator_Eval {
    unsigned eval_id;
    struct {
        unsigned log_size;
    } claim;
    CommonLookupElements common_lookup_elements;
};

// Host function declaration
extern "C" void evaluate_poseidon_aggregator(
    m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,
    m31 **trace0_evaluations, unsigned trace0_evaluations_len,
    m31 **trace1_evaluations, unsigned trace1_evaluations_len,
    m31 **trace2_evaluations, unsigned trace2_evaluations_len,
    qm31 *random_coeff_powers,
    m31 *denominator_inverses,
    unsigned int domain_log_size, unsigned int eval_domain_log_size,
    unsigned int number_of_columns, unsigned int logup_counts,
    void *eval, qm31 cumsum_shift,
    bool should_accumulate, bool use_assert_evaluator,
    cudaStream_t stream
);

#endif // EVALUATE_POSEIDON_AGGREGATOR_CUH
