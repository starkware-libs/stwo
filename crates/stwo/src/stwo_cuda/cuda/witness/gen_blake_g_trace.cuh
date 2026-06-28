
#ifndef GEN_BLAGE_G_TRACE_H
#define GEN_BLAGE_G_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define BLAKE_G_TRACE_GEN_THREAD_COUNT_MAX 256

typedef struct Enabler {
    unsigned padding_offset;

    HOST_DEVICE_FORCEINLINE Enabler(unsigned padding_offset) : padding_offset(padding_offset) {}


}Enabler;



extern "C"
void generate_blake_g_traces(
    m31 **traces,
    m31 **lookup_blake_g_0,
    m31 **lookup_verify_bitwise_xor_12_0,
    m31 **lookup_verify_bitwise_xor_12_1,
    m31 **lookup_verify_bitwise_xor_4_0 ,
    m31 **lookup_verify_bitwise_xor_4_1 ,
    m31 **lookup_verify_bitwise_xor_7_0 ,
    m31 **lookup_verify_bitwise_xor_7_1 ,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_4 ,
    m31 **lookup_verify_bitwise_xor_8_5 ,
    m31 **lookup_verify_bitwise_xor_8_6 ,
    m31 **lookup_verify_bitwise_xor_8_7 ,
    m31 **lookup_verify_bitwise_xor_9_0 ,
    m31 **lookup_verify_bitwise_xor_9_1 ,

    m31 **sub_component_inputs_verify_bitwise_xor_8,
    m31 **sub_component_inputs_verify_bitwise_xor_12,
    m31 **sub_component_inputs_verify_bitwise_xor_4,
    m31 **sub_component_inputs_verify_bitwise_xor_7,
    m31 **sub_component_inputs_verify_bitwise_xor_9,

    uint32_t **blake_g_input,

    unsigned trace_log_size
);

extern "C"
void generate_blake_g_interaction_traces(
    void *blake_g,
    void *verify_bitwise_xor_12,
    void *verify_bitwise_xor_4 ,
    void *verify_bitwise_xor_7 ,
    void *verify_bitwise_xor_8 ,
    void *verify_bitwise_xor_8_b ,
    void *verify_bitwise_xor_9 ,

    m31 **lookup_blake_g_0,
    m31 **lookup_verify_bitwise_xor_12_0,
    m31 **lookup_verify_bitwise_xor_12_1,
    m31 **lookup_verify_bitwise_xor_4_0 ,
    m31 **lookup_verify_bitwise_xor_4_1 ,
    m31 **lookup_verify_bitwise_xor_7_0 ,
    m31 **lookup_verify_bitwise_xor_7_1 ,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_4 ,
    m31 **lookup_verify_bitwise_xor_8_5 ,
    m31 **lookup_verify_bitwise_xor_8_6 ,
    m31 **lookup_verify_bitwise_xor_8_7 ,
    m31 **lookup_verify_bitwise_xor_9_0 ,
    m31 **lookup_verify_bitwise_xor_9_1 ,

    unsigned n_rows,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);
#endif // GEN_BLAGE_G_TRACE_H