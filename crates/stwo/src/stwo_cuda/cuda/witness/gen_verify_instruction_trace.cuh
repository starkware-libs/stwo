
#ifndef GEN_VERIFY_INSTRUCTION_TRACE_H
#define GEN_VERIFY_INSTRUCTION_TRACE_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// Lookup element types are already defined in relations.cuh:
// - VerifyInstruction (LookupElementsBasic<7>)
// - MemoryAddressToId (LookupElementsBasic<2>)
// - MemoryIdToBig (LookupElementsBasic<29>)
// - RangeCheck_7_2_5 (LookupElementsBasic<3>)
// - RangeCheck_4_3 (LookupElementsBasic<2>)

#define N_VERIFY_INSTRUCTION_TRACE_COLUMNS 17

extern "C"
void generate_verify_instruction_trace(
    m31 **traces,
    m31 **lookup_memory_address_to_id_0,
    m31 **lookup_memory_id_to_big_0,
    m31 **lookup_range_check_7_2_5_0,
    m31 **lookup_range_check_4_3_0,
    m31 **lookup_verify_instruction_0,
    m31 **sub_component_inputs_memory_address_to_id,
    m31 **sub_component_inputs_memory_id_to_big,
    m31 **sub_component_inputs_range_check_7_2_5,
    m31 **sub_component_inputs_range_check_4_3,
    m31 **verify_instruction_inputs,
    m31 *multiplicities,
    unsigned *memory_address_to_id_address_to_raw_id,
    unsigned n_rows,
    unsigned log_size
);

extern "C"
void generate_verify_instruction_interaction_trace(
    m31 **interaction_traces,
    m31 **lookup_rc_7_2_5,
    m31 **lookup_rc_4_3,
    m31 **lookup_addr_to_id,
    m31 **lookup_id_to_big,
    m31 **lookup_verify_instr,
    m31 *multiplicities,
    RangeCheck_7_2_5 *rc_7_2_5_lookup,
    RangeCheck_4_3 *rc_4_3_lookup,
    MemoryAddressToId *addr_to_id_lookup,
    MemoryIdToBig *id_to_big_lookup,
    VerifyInstruction *verify_instr_lookup,
    unsigned log_size,
    m31 *claimed_sum
);

#endif // GEN_VERIFY_INSTRUCTION_TRACE_H
