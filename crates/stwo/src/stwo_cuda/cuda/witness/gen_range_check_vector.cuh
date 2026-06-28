#ifndef GEN_RANGE_CHECK_VECTOR_TRACE_H
#define GEN_RANGE_CHECK_VECTOR_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

extern "C"
void range_check_vector_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    unsigned n_range,
    unsigned *ranges,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void partition_into_bit_segments_cuda(
    // unsigned *inputs,
    unsigned total_size,
    unsigned n_range,
    unsigned *n_bits_per_segments,
    unsigned **output_value
);

extern "C"
void range_check_vector_generate_interaction_trace(
    void *lookup_element_ptr,
    m31 *multiplicities,

    unsigned n_range,
    unsigned *ranges,

    unsigned log_size,
    m31 **interaction_traces,
    m31 *claimed_sum
);


// Generic macro that works with any N-segment range check.
// All range check relation types inherit from LookupElementsBasic<N> with the same layout.
#define HANDLE_RANGE_CHECK_GENERIC(TEMPLATE_N) \
    { \
        LookupElementsBasic<TEMPLATE_N> *lookup_elements = (LookupElementsBasic<TEMPLATE_N> *)lookup_element_ptr; \
        LookupElementsBasic<TEMPLATE_N> *device_lookup_elements = clone_to_device<LookupElementsBasic<TEMPLATE_N>>(lookup_elements, 1); \
        generate_range_check_interaction_trace_col_single_gen_kernel<TEMPLATE_N><<<num_blocks, block_dim>>>( \
            device_lookup_elements, \
            device_cols_tmp_vec, \
            device_multiplicities, \
            trace_size, \
            device_logup_denom, \
            device_numerator0, \
            device_numerator1, \
            device_numerator2, \
            device_numerator3 \
        ); \
        ASSERT_CUDA_SUCCESS(cudaGetLastError()); \
        cuda_free_memory(device_lookup_elements); \
    }

extern "C"
void range_check_multi_relation_interaction_trace(
    unsigned n_pairs,
    unsigned n_range,
    unsigned *ranges,
    void **lookup_elements,              // 2*n_pairs host pointers to LookupElements
    m31 **multiplicities,                // 2*n_pairs device pointers
    unsigned log_size,
    m31 **interaction_trace_columns,     // 4*n_pairs output column device pointers
    m31 *claimed_sum                     // 4 device m31s for qm31
);

#endif // GEN_RANGE_CHECK_VECTOR_TRACE_H