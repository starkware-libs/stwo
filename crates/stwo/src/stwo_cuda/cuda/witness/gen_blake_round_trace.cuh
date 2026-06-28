
#ifndef GEN_BLAGE_ROUND_TRACE_H
#define GEN_BLAGE_ROUND_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define BLAKE_ROUND_TRACE_GEN_THREAD_COUNT_MAX 256


extern "C"
void generate_blake_round_traces(
    m31 **traces,

    m31 **lookup_blake_g_0,
    m31 **lookup_blake_g_1,
    m31 **lookup_blake_g_2,
    m31 **lookup_blake_g_3,
    m31 **lookup_blake_g_4,
    m31 **lookup_blake_g_5,
    m31 **lookup_blake_g_6,
    m31 **lookup_blake_g_7,
    m31 **lookup_blake_round_0,
    m31 **lookup_blake_round_1,
    m31 **lookup_blake_round_sigma_0,
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,
    m31 **lookup_memory_address_to_id_5,
    m31 **lookup_memory_address_to_id_6,
    m31 **lookup_memory_address_to_id_7,
    m31 **lookup_memory_address_to_id_8,
    m31 **lookup_memory_address_to_id_9,
    m31 **lookup_memory_address_to_id_10,
    m31 **lookup_memory_address_to_id_11,
    m31 **lookup_memory_address_to_id_12,
    m31 **lookup_memory_address_to_id_13,
    m31 **lookup_memory_address_to_id_14,
    m31 **lookup_memory_address_to_id_15,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,
    m31 **lookup_memory_id_to_big_5,
    m31 **lookup_memory_id_to_big_6,
    m31 **lookup_memory_id_to_big_7,
    m31 **lookup_memory_id_to_big_8,
    m31 **lookup_memory_id_to_big_9,
    m31 **lookup_memory_id_to_big_10,
    m31 **lookup_memory_id_to_big_11,
    m31 **lookup_memory_id_to_big_12,
    m31 **lookup_memory_id_to_big_13,
    m31 **lookup_memory_id_to_big_14,
    m31 **lookup_memory_id_to_big_15,
    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_7_2_5_1,
    m31 **lookup_range_check_7_2_5_2,
    m31 **lookup_range_check_7_2_5_3,
    m31 **lookup_range_check_7_2_5_4,
    m31 **lookup_range_check_7_2_5_5,
    m31 **lookup_range_check_7_2_5_6,
    m31 **lookup_range_check_7_2_5_7,
    m31 **lookup_range_check_7_2_5_8,
    m31 **lookup_range_check_7_2_5_9,
    m31 **lookup_range_check_7_2_5_10,
    m31 **lookup_range_check_7_2_5_11,
    m31 **lookup_range_check_7_2_5_12,
    m31 **lookup_range_check_7_2_5_13,
    m31 **lookup_range_check_7_2_5_14,
    m31 **lookup_range_check_7_2_5_15,

    m31 **sub_component_inputs_blake_round_sigma,
    m31 **sub_component_inputs_range_check_7_2_5,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_blake_g,

    m31 **blake_round_input,

    unsigned *memory_address_to_id_address_to_raw_id,

    unsigned **memory_id_to_big_transpose_big_value_ptr,
    unsigned *memory_id_to_big_small_value_ptr,

    unsigned n_rows,
    unsigned trace_log_len
);

extern "C"
void generate_blake_round_interaction_traces(
    void *blake_g,
    void *blake_round,
    void *blake_round_sigma ,
    void *memory_address_to_id ,
    void *memory_id_to_big ,
    void *range_check_7_2_5 ,

    m31 **lookup_blake_g_0,
    m31 **lookup_blake_g_1,
    m31 **lookup_blake_g_2,
    m31 **lookup_blake_g_3,
    m31 **lookup_blake_g_4,
    m31 **lookup_blake_g_5,
    m31 **lookup_blake_g_6,
    m31 **lookup_blake_g_7,

    m31 **lookup_blake_round_0,
    m31 **lookup_blake_round_1,

    m31 **lookup_blake_round_sigma_0,

    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_address_to_id_1,
    m31 **lookup_memory_address_to_id_2,
    m31 **lookup_memory_address_to_id_3,
    m31 **lookup_memory_address_to_id_4,
    m31 **lookup_memory_address_to_id_5,
    m31 **lookup_memory_address_to_id_6,
    m31 **lookup_memory_address_to_id_7,
    m31 **lookup_memory_address_to_id_8,
    m31 **lookup_memory_address_to_id_9,
    m31 **lookup_memory_address_to_id_10,
    m31 **lookup_memory_address_to_id_11,
    m31 **lookup_memory_address_to_id_12,
    m31 **lookup_memory_address_to_id_13,
    m31 **lookup_memory_address_to_id_14,
    m31 **lookup_memory_address_to_id_15,

    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_memory_id_to_big_1,
    m31 **lookup_memory_id_to_big_2,
    m31 **lookup_memory_id_to_big_3,
    m31 **lookup_memory_id_to_big_4,
    m31 **lookup_memory_id_to_big_5,
    m31 **lookup_memory_id_to_big_6,
    m31 **lookup_memory_id_to_big_7,
    m31 **lookup_memory_id_to_big_8,
    m31 **lookup_memory_id_to_big_9,
    m31 **lookup_memory_id_to_big_10,
    m31 **lookup_memory_id_to_big_11,
    m31 **lookup_memory_id_to_big_12,
    m31 **lookup_memory_id_to_big_13,
    m31 **lookup_memory_id_to_big_14,
    m31 **lookup_memory_id_to_big_15,

    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_7_2_5_1,
    m31 **lookup_range_check_7_2_5_2,
    m31 **lookup_range_check_7_2_5_3,
    m31 **lookup_range_check_7_2_5_4,
    m31 **lookup_range_check_7_2_5_5,
    m31 **lookup_range_check_7_2_5_6,
    m31 **lookup_range_check_7_2_5_7,
    m31 **lookup_range_check_7_2_5_8,
    m31 **lookup_range_check_7_2_5_9,
    m31 **lookup_range_check_7_2_5_10,
    m31 **lookup_range_check_7_2_5_11,
    m31 **lookup_range_check_7_2_5_12,
    m31 **lookup_range_check_7_2_5_13,
    m31 **lookup_range_check_7_2_5_14,
    m31 **lookup_range_check_7_2_5_15,

    unsigned n_rows,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

#endif // GEN_BLAGE_ROUND_TRACE_H