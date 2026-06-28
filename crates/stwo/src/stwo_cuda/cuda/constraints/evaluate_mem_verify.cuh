#ifndef EVALUATE_MEM_VERIFY_CONSTRAINT_H
#define EVALUATE_MEM_VERIFY_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"

template<typename EvaluatorT>
DEVICE_FORCEINLINE void mem_verify_evaluate(
    m31 mem_verify_input_limb_0,
    m31 mem_verify_input_limb_1,
    m31 mem_verify_input_limb_2,
    m31 mem_verify_input_limb_3,
    m31 mem_verify_input_limb_4,
    m31 mem_verify_input_limb_5,
    m31 mem_verify_input_limb_6,
    m31 mem_verify_input_limb_7,
    m31 mem_verify_input_limb_8,
    m31 mem_verify_input_limb_9,
    m31 mem_verify_input_limb_10,
    m31 mem_verify_input_limb_11,
    m31 mem_verify_input_limb_12,
    m31 mem_verify_input_limb_13,
    m31 mem_verify_input_limb_14,
    m31 mem_verify_input_limb_15,
    m31 mem_verify_input_limb_16,
    m31 mem_verify_input_limb_17,
    m31 mem_verify_input_limb_18,
    m31 mem_verify_input_limb_19,
    m31 mem_verify_input_limb_20,
    m31 mem_verify_input_limb_21,
    m31 mem_verify_input_limb_22,
    m31 mem_verify_input_limb_23,
    m31 mem_verify_input_limb_24,
    m31 mem_verify_input_limb_25,
    m31 mem_verify_input_limb_26,
    m31 mem_verify_input_limb_27,
    m31 mem_verify_input_limb_28,

    m31 id_col0,

    const CommonLookupElements& common_lookup_elements,

    EvaluatorT *cuda_evaluator
) {

    m31 values_0[3] = {
        MEMORY_ADDRESS_TO_ID_RELATION_ID,
        mem_verify_input_limb_0,
        id_col0
    };
    cuda_evaluator->add_to_relation<3>(common_lookup_elements, qm31{{1,0}, {0,0}}, values_0);

    m31 values_1[30] = {
        MEMORY_ID_TO_BIG_RELATION_ID,
        id_col0,
        mem_verify_input_limb_1,
        mem_verify_input_limb_2,
        mem_verify_input_limb_3,
        mem_verify_input_limb_4,
        mem_verify_input_limb_5,
        mem_verify_input_limb_6,
        mem_verify_input_limb_7,
        mem_verify_input_limb_8,
        mem_verify_input_limb_9,
        mem_verify_input_limb_10,
        mem_verify_input_limb_11,
        mem_verify_input_limb_12,
        mem_verify_input_limb_13,
        mem_verify_input_limb_14,
        mem_verify_input_limb_15,
        mem_verify_input_limb_16,
        mem_verify_input_limb_17,
        mem_verify_input_limb_18,
        mem_verify_input_limb_19,
        mem_verify_input_limb_20,
        mem_verify_input_limb_21,
        mem_verify_input_limb_22,
        mem_verify_input_limb_23,
        mem_verify_input_limb_24,
        mem_verify_input_limb_25,
        mem_verify_input_limb_26,
        mem_verify_input_limb_27,
        mem_verify_input_limb_28
    };
    cuda_evaluator->add_to_relation<30>(common_lookup_elements, qm31{{1,0}, {0,0}}, values_1);
}


#endif // EVALUATE_MEM_VERIFY_CONSTRAINT_H