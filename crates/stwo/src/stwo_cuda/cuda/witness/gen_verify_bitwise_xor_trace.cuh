
#ifndef GEN_VERIFY_BITWISE_XOR_TRACE_H
#define GEN_VERIFY_BITWISE_XOR_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

#define VERIFY_BITWISE_4_N_TRACE_COLUMNS 1
#define VERIFY_BITWISE_4_N_BITS 4
#define VERIFY_BITWISE_4_LOG_SIZE (2 * VERIFY_BITWISE_4_N_BITS)

#define VERIFY_BITWISE_7_N_TRACE_COLUMNS 1
#define VERIFY_BITWISE_7_N_BITS 7
#define VERIFY_BITWISE_7_LOG_SIZE (2 * VERIFY_BITWISE_7_N_BITS)

#define VERIFY_BITWISE_8_N_TRACE_COLUMNS 1
#define VERIFY_BITWISE_8_N_BITS 8
#define VERIFY_BITWISE_8_LOG_SIZE (2 * VERIFY_BITWISE_8_N_BITS)

#define VERIFY_BITWISE_9_N_TRACE_COLUMNS 1
#define VERIFY_BITWISE_9_N_BITS 9
#define VERIFY_BITWISE_9_LOG_SIZE (2 * VERIFY_BITWISE_9_N_BITS)

#define VERIFY_BITWISE_12_ELEM_BITS  12
#define VERIFY_BITWISE_12_EXPAND_BITS  2
#define VERIFY_BITWISE_12_LIMB_BITS  VERIFY_BITWISE_12_ELEM_BITS - VERIFY_BITWISE_12_EXPAND_BITS
#define VERIFY_BITWISE_12_LOG_SIZE  (VERIFY_BITWISE_12_ELEM_BITS - VERIFY_BITWISE_12_EXPAND_BITS) * 2
#define VERIFY_BITWISE_12_N_MULT_COLUMNS  1 << (VERIFY_BITWISE_12_EXPAND_BITS * 2)
#define VERIFY_BITWISE_12_N_TRACE_COLUMNS VERIFY_BITWISE_12_N_MULT_COLUMNS

extern "C"
void verify_bitwise_xor_4_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void verify_bitwise_xor_7_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void verify_bitwise_xor_8_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void verify_bitwise_xor_8_b_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void verify_bitwise_xor_9_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void verify_bitwise_xor_12_mults_init(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 **mults,
    unsigned mults_col_size,
    unsigned mults_row_log_size
);

// Interaction trace generation functions
extern "C"
void verify_bitwise_xor_4_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

extern "C"
void verify_bitwise_xor_7_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

extern "C"
void verify_bitwise_xor_8_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

extern "C"
void verify_bitwise_xor_8_b_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

extern "C"
void verify_bitwise_xor_8_paired_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities_0,
    m31 *multiplicities_1,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

extern "C"
void verify_bitwise_xor_9_interaction_trace(
    void *lookup_elements,
    m31 *multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

extern "C"
void verify_bitwise_xor_12_interaction_trace(
    void *lookup_elements,
    m31 **multiplicities,
    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);

#endif // GEN_VERIFY_BITWISE_XOR_TRACE_H
