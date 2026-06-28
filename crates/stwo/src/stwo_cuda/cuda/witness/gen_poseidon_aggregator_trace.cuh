#pragma once

#include "fields.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void gen_poseidon_aggregator_trace(
    unsigned *traces,
    unsigned log_size,
    // Inputs (6 ID arrays + mults)
    unsigned *input_ids_0,
    unsigned *input_ids_1,
    unsigned *input_ids_2,
    unsigned *input_ids_3,
    unsigned *input_ids_4,
    unsigned *input_ids_5,
    unsigned *mults_in,
    // Memory tables
    unsigned **memory_id_to_big_transposed_big_values,
    unsigned *memory_id_to_big_small_values,
    // Lookup data outputs for interaction trace
    // 6 memory_id_to_big (29 elements each)
    unsigned *lookup_memory_id_to_big_0,
    unsigned *lookup_memory_id_to_big_1,
    unsigned *lookup_memory_id_to_big_2,
    unsigned *lookup_memory_id_to_big_3,
    unsigned *lookup_memory_id_to_big_4,
    unsigned *lookup_memory_id_to_big_5,
    // 2 range_check_3_3_3_3_3 (5 elements each)
    unsigned *lookup_range_check_3_3_3_3_3_0,
    unsigned *lookup_range_check_3_3_3_3_3_1,
    // 6 range_check_4_4_4_4 (4 elements each)
    unsigned *lookup_range_check_4_4_4_4_0,
    unsigned *lookup_range_check_4_4_4_4_1,
    unsigned *lookup_range_check_4_4_4_4_2,
    unsigned *lookup_range_check_4_4_4_4_3,
    unsigned *lookup_range_check_4_4_4_4_4,
    unsigned *lookup_range_check_4_4_4_4_5,
    // 3 range_check_4_4 (2 elements each)
    unsigned *lookup_range_check_4_4_0,
    unsigned *lookup_range_check_4_4_1,
    unsigned *lookup_range_check_4_4_2,
    // 10 poseidon_full_round_chain (32 elements each)
    // pfrc 0-7: sub-component feeds (rounds 0,1,2,3,31,32,33,34)
    // pfrc 8-9: IT boundary entries (round 4, round 35)
    unsigned *lookup_poseidon_full_round_chain_0,
    unsigned *lookup_poseidon_full_round_chain_1,
    unsigned *lookup_poseidon_full_round_chain_2,
    unsigned *lookup_poseidon_full_round_chain_3,
    unsigned *lookup_poseidon_full_round_chain_4,
    unsigned *lookup_poseidon_full_round_chain_5,
    unsigned *lookup_poseidon_full_round_chain_6,
    unsigned *lookup_poseidon_full_round_chain_7,
    unsigned *lookup_poseidon_full_round_chain_8,
    unsigned *lookup_poseidon_full_round_chain_9,
    // 28 poseidon_3_partial_rounds_chain (42 elements each)
    // p3prc 0-26: sub-component feeds, p3prc 27: IT boundary (round 31)
    unsigned *lookup_poseidon_3_partial_rounds_chain_0,
    unsigned *lookup_poseidon_3_partial_rounds_chain_1,
    unsigned *lookup_poseidon_3_partial_rounds_chain_2,
    unsigned *lookup_poseidon_3_partial_rounds_chain_3,
    unsigned *lookup_poseidon_3_partial_rounds_chain_4,
    unsigned *lookup_poseidon_3_partial_rounds_chain_5,
    unsigned *lookup_poseidon_3_partial_rounds_chain_6,
    unsigned *lookup_poseidon_3_partial_rounds_chain_7,
    unsigned *lookup_poseidon_3_partial_rounds_chain_8,
    unsigned *lookup_poseidon_3_partial_rounds_chain_9,
    unsigned *lookup_poseidon_3_partial_rounds_chain_10,
    unsigned *lookup_poseidon_3_partial_rounds_chain_11,
    unsigned *lookup_poseidon_3_partial_rounds_chain_12,
    unsigned *lookup_poseidon_3_partial_rounds_chain_13,
    unsigned *lookup_poseidon_3_partial_rounds_chain_14,
    unsigned *lookup_poseidon_3_partial_rounds_chain_15,
    unsigned *lookup_poseidon_3_partial_rounds_chain_16,
    unsigned *lookup_poseidon_3_partial_rounds_chain_17,
    unsigned *lookup_poseidon_3_partial_rounds_chain_18,
    unsigned *lookup_poseidon_3_partial_rounds_chain_19,
    unsigned *lookup_poseidon_3_partial_rounds_chain_20,
    unsigned *lookup_poseidon_3_partial_rounds_chain_21,
    unsigned *lookup_poseidon_3_partial_rounds_chain_22,
    unsigned *lookup_poseidon_3_partial_rounds_chain_23,
    unsigned *lookup_poseidon_3_partial_rounds_chain_24,
    unsigned *lookup_poseidon_3_partial_rounds_chain_25,
    unsigned *lookup_poseidon_3_partial_rounds_chain_26,
    unsigned *lookup_poseidon_3_partial_rounds_chain_27
);

void gen_poseidon_aggregator_interaction_trace(
    void* lookup_elements,
    // 6 memory_id_to_big (29 elements each)
    unsigned *lookup_memory_id_to_big_0,
    unsigned *lookup_memory_id_to_big_1,
    unsigned *lookup_memory_id_to_big_2,
    unsigned *lookup_memory_id_to_big_3,
    unsigned *lookup_memory_id_to_big_4,
    unsigned *lookup_memory_id_to_big_5,
    // 2 rc_3_3_3_3_3 (5 elements each)
    unsigned *lookup_range_check_3_3_3_3_3_0,
    unsigned *lookup_range_check_3_3_3_3_3_1,
    // 6 rc_4_4_4_4 (4 elements each)
    unsigned *lookup_range_check_4_4_4_4_0,
    unsigned *lookup_range_check_4_4_4_4_1,
    unsigned *lookup_range_check_4_4_4_4_2,
    unsigned *lookup_range_check_4_4_4_4_3,
    unsigned *lookup_range_check_4_4_4_4_4,
    unsigned *lookup_range_check_4_4_4_4_5,
    // 3 rc_4_4 (2 elements each)
    unsigned *lookup_range_check_4_4_0,
    unsigned *lookup_range_check_4_4_1,
    unsigned *lookup_range_check_4_4_2,
    // 4 pfrc (32 elements each, for interaction trace)
    unsigned *lookup_poseidon_full_round_chain_0,
    unsigned *lookup_poseidon_full_round_chain_1,
    unsigned *lookup_poseidon_full_round_chain_2,
    unsigned *lookup_poseidon_full_round_chain_3,
    // 2 p3prc (42 elements each, for interaction trace)
    unsigned *lookup_poseidon_3_partial_rounds_chain_0,
    unsigned *lookup_poseidon_3_partial_rounds_chain_1,
    // Base trace (342 columns)
    unsigned *base_trace,
    unsigned log_size,
    // Output
    unsigned *interaction_trace_columns,
    unsigned *claimed_sum
);

#ifdef __cplusplus
}
#endif
