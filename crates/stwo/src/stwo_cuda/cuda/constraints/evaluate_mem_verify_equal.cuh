#ifndef EVALUATE_MEM_VERIFY_EQUAL_CONSTRAINT_H
#define EVALUATE_MEM_VERIFY_EQUAL_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"


DEVICE_FORCEINLINE void evaluate_mem_verify_equal(
    m31 mem_verify_equal_input_limb_0,
    m31 mem_verify_equal_input_limb_1,
    m31 id_col0,
    const CommonLookupElements& common_lookup_elements,
    auto *cuda_evaluator
) {
    // First relation
    m31 values_0[3] = {
        MEMORY_ADDRESS_TO_ID_RELATION_ID,
        mem_verify_equal_input_limb_0,
        id_col0
    };
    cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1,0}, {0,0}}, values_0);

    // Second relation
    m31 values_1[3] = {
        MEMORY_ADDRESS_TO_ID_RELATION_ID,
        mem_verify_equal_input_limb_1,
        id_col0
    };
    cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1,0}, {0,0}}, values_1);
}

#endif // EVALUATE_MEM_VERIFY_EQUAL_CONSTRAINT_H