
#ifndef GEN_MEMORY_ADDRESS_TO_ID_TRACE_H
#define GEN_MEMORY_ADDRESS_TO_ID_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"
#include "gen_blake_round_sigma_trace.cuh"

struct address_to_raw_id {
    unsigned *device_ptr;
    size_t size;
};

struct multiplicities {
    unsigned *device_ptr;
    size_t size;
};

// Note: address_to_raw_id array starts at index 0 for address 1
// So for address N, we access address_to_raw_id[N-1]
// This function does not do bounds checking - caller must ensure valid address
DEVICE_FORCEINLINE void memory_address_to_id_deduce_output(
    unsigned* address_to_raw_id,
    m31 input,
    m31 *output
) {
    unsigned indices = input;
    if (indices == 0) {
        printf("ERROR: memory_address_to_id_deduce_output called with address=0\n");
        *output = {0};
        return;
    }
    *output = (m31) address_to_raw_id[indices - 1];
}

// Bounds-checked version for debugging
DEVICE_FORCEINLINE void memory_address_to_id_deduce_output_checked(
    unsigned* address_to_raw_id,
    unsigned address_to_raw_id_size,
    m31 input,
    m31 *output
) {
    unsigned addr = input;
    if (addr == 0) {
        printf("ERROR: memory_address_to_id_deduce_output_checked: address=0\n");
        *output = {0};
        return;
    }
    if (addr > address_to_raw_id_size) {
        printf("ERROR: memory_address_to_id_deduce_output_checked: address=%u > size=%u\n",
               addr, address_to_raw_id_size);
        *output = {0};
        return;
    }
    *output = (m31) address_to_raw_id[addr - 1];
}

extern "C"
void memory_address_to_id_add_inputs(
    m31 **inputs,
    unsigned input_col_sizes,
    unsigned input_row_sizes,
    m31 *mults,
    unsigned mults_row_log_size
);

extern "C"
void generate_memory_address_to_id_traces(
    m31 **traces,
    m31 **interaction_traces,
    m31 *address_to_raw_id,
    m31 *multiplicities,
    unsigned total_size_log,
    unsigned log_size,
    MemoryAddressToId *lookup_elements
);

#endif // GEN_MEMORY_ADDRESS_TO_ID_TRACE_H