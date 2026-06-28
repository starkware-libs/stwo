
#ifndef GEN_TRIPLE_XOR_32_TRACE_H
#define GEN_TRIPLE_XOR_32_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define TRIPLE_XOR_32_TRACE_GEN_THREAD_COUNT_MAX 256

typedef struct Enabler {
    unsigned padding_offset;

    HOST_DEVICE_FORCEINLINE Enabler(unsigned padding_offset) : padding_offset(padding_offset) {}


}Enabler;


extern "C"
void generate_triple_xor_32_traces(
    m31 **traces,
    m31 **lookup_triple_xor_32_0,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_b_0 ,
    m31 **lookup_verify_bitwise_xor_8_b_1 ,
    m31 **lookup_verify_bitwise_xor_8_b_2 ,
    m31 **lookup_verify_bitwise_xor_8_b_3 ,

    m31 **sub_component_inputs_verify_bitwise_xor_8,
    m31 **sub_component_inputs_verify_bitwise_xor_8_b,

    uint32_t **triple_xor_32_input,

    unsigned trace_log_size
);

extern "C"
void generate_triple_xor_32_interaction_traces(
    void *triple_xor_32,
    void *verify_bitwise_xor_8,
    void *verify_bitwise_xor_8_b,

    m31 **lookup_triple_xor_32_0,
    m31 **lookup_verify_bitwise_xor_8_0 ,
    m31 **lookup_verify_bitwise_xor_8_1 ,
    m31 **lookup_verify_bitwise_xor_8_2 ,
    m31 **lookup_verify_bitwise_xor_8_3 ,
    m31 **lookup_verify_bitwise_xor_8_b_0 ,
    m31 **lookup_verify_bitwise_xor_8_b_1 ,
    m31 **lookup_verify_bitwise_xor_8_b_2 ,
    m31 **lookup_verify_bitwise_xor_8_b_3 ,

    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);
#endif // GEN_TRIPLE_XOR_32_TRACE_H