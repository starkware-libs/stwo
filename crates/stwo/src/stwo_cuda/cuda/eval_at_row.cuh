#ifndef EVAL_AT_ROW_H
#define EVAL_AT_ROW_H

#include <cassert>
#include "fields.cuh"
#include "utils.cuh"
#include "logup.cuh"

struct Fraction {
    qm31 numerator;
    qm31 denominator;

    HOST_DEVICE_FORCEINLINE Fraction() : numerator(qm31{{0, 0}, {0, 0}}), denominator(qm31{{1, 0}, {0, 0}}) {}

    HOST_DEVICE_FORCEINLINE Fraction(const qm31 numerator, const qm31 denominator)
        : numerator(numerator), denominator(denominator) {}

    HOST_DEVICE_FORCEINLINE static Fraction zero() {
        return Fraction(qm31{{0, 0}, {0, 0}}, qm31{{1, 0}, {0, 0}});
    }

    HOST_DEVICE_FORCEINLINE bool is_zero() const {
        qm31 zero = {{0, 0}, {0, 0}};
        qm31 one = {{1, 0}, {0, 0}};

        bool numerator_is_zero =
            (numerator.a.a == zero.a.a) &&
            (numerator.a.b == zero.a.b) &&
            (numerator.b.a == zero.b.a) &&
            (numerator.b.b == zero.b.b);

        bool denominator_is_one =
            (denominator.a.a == one.a.a) &&
            (denominator.a.b == one.a.b) &&
            (denominator.b.a == one.b.a) &&
            (denominator.b.b == one.b.b);

        return numerator_is_zero && denominator_is_one;
    }

    HOST_DEVICE_FORCEINLINE void dump(const char *description) {
        printf("%s, Fraction: {numerator: (%d + %di) + (%d + %di)u, denominator: (%d + %di) + (%d + %di)u}\n", description, numerator.a.a, numerator.a.b, numerator.b.a, numerator.b.b, denominator.a.a, denominator.a.b, denominator.b.a, denominator.b.b);
    }

    // sum to first element
    HOST_DEVICE_FORCEINLINE static Fraction sum(Fraction *fractions, unsigned len) {
        Fraction result = fractions[0];  // Copy, don't modify original
        for (unsigned i = 1; i < len; i++) {
            result.numerator = add(mul(fractions[i].numerator, result.denominator), mul(fractions[i].denominator, result.numerator));
            result.denominator = mul(fractions[i].denominator, result.denominator);
        }
        return result;
    }
};

template <int N>
struct RelationEntry{
    LookupElementsBasic<N> relation;
    qm31 multiplicity;
    m31 values[N];

    HOST_DEVICE_FORCEINLINE RelationEntry(LookupElementsBasic<N> relation, qm31 multiplicity, m31* values)
    : relation(relation), multiplicity(multiplicity) {
        for (int i = 0; i < N; ++i) {
            this->values[i] = values[i];
        }
    }
};
template<int N>
DEVICE_FORCEINLINE Fraction add_to_relation(
    RelationEntry<N> entry
) {
    Fraction fraction = Fraction(entry.multiplicity, entry.relation.combine(entry.values, N));
    return fraction;
}

typedef struct LogupAtRow {
    unsigned interaction = 0;
    qm31 cumsum_shift = {0};
    Fraction* fractions;
    unsigned log_size;
}LogupAtRow;

typedef struct CudaAssertEvaluator {
    m31 **trace_evaluations;
    qm31 *random_coeff_powers;
    unsigned constraint_index = 0;
    unsigned col_index[3] = {0, 0, 0};
    unsigned row = 0;
    LogupAtRow logup;
    unsigned logup_fraction_index = 0;
    unsigned logup_fraction_counts_per_eval = 0;
    qm31 row_res = { 0 };
    unsigned domain_log_size;
    unsigned eval_domain_log_size;

    DEVICE_FORCEINLINE CudaAssertEvaluator(
        m31 **trace_evaluations,
        qm31 *random_coeff_powers,
        unsigned constraint_index,
        unsigned row,
        qm31 row_res,
        unsigned logup_interaction_index,
        qm31 logup_claim_sum_shift,
        unsigned logup_log_size,
        unsigned eval_domain_log_size,
        Fraction *fractions,
        unsigned logup_counts
    ) {
        this->trace_evaluations = trace_evaluations;
        this->random_coeff_powers = random_coeff_powers;
        this->constraint_index = constraint_index;
        this->row = row;
        this->row_res = row_res;
        this->logup.interaction = logup_interaction_index;
        this->logup.cumsum_shift = logup_claim_sum_shift;
        this->logup.log_size = logup_log_size;
        this->logup.fractions = fractions;
        this->logup_fraction_counts_per_eval = logup_counts;
        this->domain_log_size = logup_log_size;
        this->eval_domain_log_size = eval_domain_log_size;
    }

    template<int N>
    DEVICE_FORCEINLINE void add_to_relation(
        RelationEntry<N> entry
    ) {
        Fraction fraction = Fraction(entry.multiplicity, entry.relation.combine(entry.values, N));
        this->logup.fractions[this->logup_fraction_index + this->row * logup_fraction_counts_per_eval] = fraction;
        // Check for zero denominator (invalid fraction)
        bool denom_is_zero = (fraction.denominator.a.a == 0) &&
                             (fraction.denominator.a.b == 0) &&
                             (fraction.denominator.b.a == 0) &&
                             (fraction.denominator.b.b == 0);
        if (denom_is_zero) {
            printf("CUDA_ASSERT_FRAC ERROR: zero denominator at row=%u, fraction_index=%u\n",
                this->row, this->logup_fraction_index);
            printf("  num=[%u, %u, %u, %u] den=[%u, %u, %u, %u]\n",
                fraction.numerator.a.a, fraction.numerator.a.b,
                fraction.numerator.b.a, fraction.numerator.b.b,
                fraction.denominator.a.a, fraction.denominator.a.b,
                fraction.denominator.b.a, fraction.denominator.b.b);
            assert(false && "Zero denominator in CUDA fraction");
        }
        this->logup_fraction_index = this->logup_fraction_index + 1;
    }

    DEVICE_FORCEINLINE void next_interaction_mask(
        unsigned interaction,
        const int *offsets,
        unsigned N,
        m31 *result
    ) {
        unsigned current_col_index = this->col_index[interaction];
        col_index[interaction] += 1;

        for (unsigned i = 0; i < N; ++i) {
            int off = offsets[i];
            if (off == 0) {
                result[i] = this->trace_evaluations[current_col_index][this->row];
            } else {
                // Use the same offset calculation as CudaEvaluator to correctly handle
                // the case when eval_domain_log_size != domain_log_size (need_to_extend=true)
                int target_row = offset_bit_reversed_circle_domain_index(
                    this->row, this->domain_log_size, this->eval_domain_log_size, off
                );
                result[i] = this->trace_evaluations[current_col_index][target_row];
            }
        }
    }

    DEVICE_FORCEINLINE m31 get_preprocessed_column() {
        m31 result[1];
        int offsets[1] = {0};
        next_interaction_mask(0, offsets, 1, result);
        return result[0];
    }

    // next_trace_mask
    DEVICE_FORCEINLINE m31 next_trace_mask() {
        m31 result[1];
        int offsets[1] = {0};
        next_interaction_mask(0, offsets, 1, result);
        return result[0];
    }

    DEVICE_FORCEINLINE void add_constraint(
        m31 constraint
    ) {
        if (constraint != 0) {
            printf("\033[1;31mASSERT ERROR: cuda thread:%d, constraint_index:%d, expect constraint:0, but actual: %d\033[0m\n", this->row, this->constraint_index, constraint);
            assert(false);
        } else {
        }
        // Debug: Uncomment to print constraint values for row 0
        // if (row == 0)
        (this->constraint_index)++;
    }

    DEVICE_FORCEINLINE m31 add_intermediate(
        m31 constraint
    ) {
        return constraint;
    }

    HOST_DEVICE_FORCEINLINE void add_constraint_ext(
        qm31 constraint
    ) {
        if ((constraint.a.a != 0) || (constraint.a.b != 0) || (constraint.b.a != 0) || (constraint.b.b != 0)) {
            printf("\033[1;31mASSERT ERROR: cuda thread:%d, constraint_index:%d, expect constraint:{0, 0, 0, 0}, but actual: (%d, %d, %d, %d)\033[0m\n",
                this->row, this->constraint_index, constraint.a.a, constraint.a.b, constraint.b.a, constraint.b.b);
            assert(false);
        } else {
            // if (row == 0)
        }
        (this->constraint_index)++;
    }

    DEVICE_FORCEINLINE qm31 combine_ef(m31 values[SECURE_EXTENSION_DEGREE]) {
        qm31 result;
        result.a.a = values[0];
        result.a.b = values[1];
        result.b.a = values[2];
        result.b.b = values[3];

        return result;
    }

    DEVICE_FORCEINLINE void next_extension_interaction_mask(
        unsigned interaction,
        const int *offsets,
        unsigned N,
        qm31 *result
    ) {
        const unsigned N_MAX = 4;
        m31 base_results[SECURE_EXTENSION_DEGREE][N_MAX];
        for (unsigned i = 0; i < SECURE_EXTENSION_DEGREE; ++i) {
            next_interaction_mask(interaction, offsets, N, base_results[i]);
        }

        for (unsigned i = 0; i < N; ++i) {
            m31 values[SECURE_EXTENSION_DEGREE];
            for (unsigned j = 0; j < SECURE_EXTENSION_DEGREE; ++j) {
                values[j] = base_results[j][i];
            }
            result[i] = combine_ef(values);
        }
    }

    // Overload for CommonLookupElements (LookupElementsBasic<128>).
    // N = number of values including RELATION_ID as first element.
    template<int N>
    DEVICE_FORCEINLINE void add_to_relation(
        const LookupElementsBasic<128>& common_elements,
        qm31 multiplicity,
        const m31* values
    ) {
        Fraction fraction = Fraction(multiplicity, common_elements.combine(values, N));
        this->logup.fractions[this->logup_fraction_index + this->row * logup_fraction_counts_per_eval] = fraction;
        // Check for zero denominator (invalid fraction)
        bool denom_is_zero = (fraction.denominator.a.a == 0) &&
                             (fraction.denominator.a.b == 0) &&
                             (fraction.denominator.b.a == 0) &&
                             (fraction.denominator.b.b == 0);
        if (denom_is_zero) {
            printf("CUDA_ASSERT_FRAC ERROR: zero denominator at row=%u, fraction_index=%u\n",
                this->row, this->logup_fraction_index);
            printf("  num=[%u, %u, %u, %u] den=[%u, %u, %u, %u]\n",
                fraction.numerator.a.a, fraction.numerator.a.b,
                fraction.numerator.b.a, fraction.numerator.b.b,
                fraction.denominator.a.a, fraction.denominator.a.b,
                fraction.denominator.b.a, fraction.denominator.b.b);
            assert(false && "Zero denominator in CUDA fraction");
        }
        this->logup_fraction_index = this->logup_fraction_index + 1;
    }
}CudaAssertEvaluator;


typedef struct CudaEvaluator {
    m31 **trace_evaluations;
    qm31 *random_coeff_powers;
    unsigned constraint_index = 0;
    unsigned col_index[3] = {0, 0, 0};
    unsigned row = 0;
    LogupAtRow logup;
    unsigned logup_fraction_index = 0;
    unsigned logup_fraction_counts_per_eval = 0;
    qm31 row_res = { 0 };
    unsigned domain_log_size;
    unsigned eval_domain_log_size;

    DEVICE_FORCEINLINE CudaEvaluator(
        m31 **trace_evaluations,
        qm31 *random_coeff_powers,
        unsigned constraint_index,
        unsigned row,
        qm31 row_res,
        unsigned logup_interaction_index,
        qm31 logup_claim_sum_shift,
        unsigned logup_log_size,
        unsigned eval_domain_log_size,
        Fraction *fractions,
        unsigned logup_counts
    ) {
        this->trace_evaluations = trace_evaluations;
        this->random_coeff_powers = random_coeff_powers;
        this->constraint_index = constraint_index;
        this->row = row;
        this->row_res = row_res;
        this->logup.interaction = logup_interaction_index;
        this->logup.cumsum_shift = logup_claim_sum_shift;
        this->logup.log_size = logup_log_size;
        this->logup.fractions = fractions;
        this->logup_fraction_counts_per_eval = logup_counts;
        this->domain_log_size = logup_log_size;
        this->eval_domain_log_size = eval_domain_log_size;
    }

    template<int N>
    DEVICE_FORCEINLINE void add_to_relation(
        RelationEntry<N> entry
    ) {
        Fraction fraction = Fraction(entry.multiplicity, entry.relation.combine(entry.values, N));
        unsigned idx = this->logup_fraction_index + this->row * logup_fraction_counts_per_eval;
        this->logup.fractions[idx] = fraction;
        this->logup_fraction_index = this->logup_fraction_index + 1;
    }

    DEVICE_FORCEINLINE void next_interaction_mask(
        unsigned interaction,
        const int *offsets,
        unsigned N,
        m31 *result
    ) {
        unsigned current_col_index = this->col_index[interaction];
        col_index[interaction] += 1;

        // unsigned log_size = this->logup.log_size;
        // unsigned domain_size = 1 << log_size;

        for (unsigned i = 0; i < N; ++i) {
            int off = offsets[i];
            if (off == 0) {
                result[i] = this->trace_evaluations[current_col_index][this->row];
            } else {
                int target_row = offset_bit_reversed_circle_domain_index(
                    this->row, this->domain_log_size, this->eval_domain_log_size, off
                );
                result[i] = this->trace_evaluations[current_col_index][target_row];
            }
        }
    }

    DEVICE_FORCEINLINE m31 get_preprocessed_column() {
        m31 result[1];
        int offsets[1] = {0};
        next_interaction_mask(0, offsets, 1, result);
        return result[0];
    }

    // next_trace_mask
    DEVICE_FORCEINLINE m31 next_trace_mask() {
        m31 result[1];
        int offsets[1] = {0};
        next_interaction_mask(0, offsets, 1, result);
        return result[0];
    }

    DEVICE_FORCEINLINE void add_constraint(
        m31 constraint
    ) {
        this->row_res = add(this->row_res, mul(constraint, this->random_coeff_powers[this->constraint_index]));
        (this->constraint_index)++;
    }

    DEVICE_FORCEINLINE m31 add_intermediate(
        m31 constraint
    ) {
        return constraint;
    }

    HOST_DEVICE_FORCEINLINE void add_constraint_ext(
        qm31 constraint
    ) {
        this->row_res = add(this->row_res, mul(constraint, this->random_coeff_powers[this->constraint_index]));
        (this->constraint_index)++;
    }

    DEVICE_FORCEINLINE qm31 combine_ef(m31 values[SECURE_EXTENSION_DEGREE]) {
        qm31 result;
        result.a.a = values[0];
        result.a.b = values[1];
        result.b.a = values[2];
        result.b.b = values[3];

        return result;
    }

    DEVICE_FORCEINLINE void next_extension_interaction_mask(
        unsigned interaction,
        const int *offsets,
        unsigned N,
        qm31 *result
    ) {
        const unsigned N_MAX = 4;
        m31 base_results[SECURE_EXTENSION_DEGREE][N_MAX];
        for (unsigned i = 0; i < SECURE_EXTENSION_DEGREE; ++i) {
            next_interaction_mask(interaction, offsets, N, base_results[i]);
        }

        for (unsigned i = 0; i < N; ++i) {
            m31 values[SECURE_EXTENSION_DEGREE];
            for (unsigned j = 0; j < SECURE_EXTENSION_DEGREE; ++j) {
                values[j] = base_results[j][i];
            }
            result[i] = combine_ef(values);
        }
    }

    // Overload for CommonLookupElements (LookupElementsBasic<128>).
    // N = number of values including RELATION_ID as first element.
    template<int N>
    DEVICE_FORCEINLINE void add_to_relation(
        const LookupElementsBasic<128>& common_elements,
        qm31 multiplicity,
        const m31* values
    ) {
        Fraction fraction = Fraction(multiplicity, common_elements.combine(values, N));
        unsigned idx = this->logup_fraction_index + this->row * logup_fraction_counts_per_eval;
        this->logup.fractions[idx] = fraction;
        this->logup_fraction_index = this->logup_fraction_index + 1;
    }
}CudaEvaluator;

#endif