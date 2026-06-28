#ifndef EVALUATE_READ_BLAKE_WORD_CONSTRAINT_H
#define EVALUATE_READ_BLAKE_WORD_CONSTRAINT_H

#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"
#include "eval_at_row.cuh"
#include "constraints/relations.cuh"
#include "constraints/evaluate_verify_blake_word.cuh"


template<typename EvaluatorT>
DEVICE_FORCEINLINE void read_blake_word_evaluate(
    m31 read_blake_word_input,
    m31 low_16_bits_col0,
    m31 high_16_bits_col1,
    m31 low_7_ms_bits_col2,
    m31 high_14_ms_bits_col3,
    m31 high_5_ms_bits_col4,
    m31 id_col5,

    m31 *output_limb_0,
    m31 *output_limb_1,

    const CommonLookupElements& common_lookup_elements,

    EvaluatorT *cuda_evaluator
) {

    verify_blake_word_evaluate(
        read_blake_word_input,
        low_16_bits_col0,
        high_16_bits_col1,
        low_7_ms_bits_col2,
        high_14_ms_bits_col3,
        high_5_ms_bits_col4,
        id_col5,

        common_lookup_elements,

        cuda_evaluator
    );
    *output_limb_0 = low_16_bits_col0;
    *output_limb_1 = high_16_bits_col1;
}

#endif // EVALUATE_READ_BLAKE_WORD_CONSTRAINT_H