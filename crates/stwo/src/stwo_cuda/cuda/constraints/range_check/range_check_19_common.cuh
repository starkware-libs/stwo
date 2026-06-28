#ifndef RANGE_CHECK_19_COMMON_H
#define RANGE_CHECK_19_COMMON_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "relations.cuh"

// Common constraint logic for range_check_19_xxx:
// - Read Seq preprocessed column (from trace0/preprocessed)
// - Read multiplicity trace column (from trace1/base)
// - Write (-multiplicity, [relation_id, seq]) to the common lookup elements
template<typename EvaluatorT>
DEVICE_FORCEINLINE void eval_range_check_19_core(
    EvaluatorT &cuda_evaluator0,  // for preprocessed trace (seq)
    EvaluatorT &cuda_evaluator1,  // for base trace (multiplicity)
    CommonLookupElements common_lookup_elements,
    m31 relation_id
) {
    m31 seq_val = cuda_evaluator0.next_trace_mask();  // From trace0 (preprocessed)
    m31 multiplicity = cuda_evaluator1.next_trace_mask();  // From trace1 (base trace)

    m31 values[2] = {relation_id, seq_val};

    m31 neg_mult = neg(multiplicity);
    qm31 multiplicity_ext = {{neg_mult, 0}, {0, 0}};

    cuda_evaluator1.add_to_relation<2>(common_lookup_elements, multiplicity_ext, values);
}

#endif // RANGE_CHECK_19_COMMON_H

