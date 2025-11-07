use num_traits::{One, Zero};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

use super::FibonacciRelation;

/// Returns (trace columns, target_c_value)
/// trace: 3 columns [a, b, c] where c = a + b
/// target_c_value: the c value at target_element BEFORE bit-reverse
///
/// Generates Fibonacci sequence only up to target_element (inclusive), rest are zeros
pub fn gen_computing_trace(
    log_size: u32,
    initial_a: u32,
    initial_b: u32,
    target_element: usize,
) -> (ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>, BaseField) {
    let n_rows = 1 << log_size;

    let mut col_a = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_b = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    let mut a = BaseField::from_u32_unchecked(initial_a);
    let mut b = BaseField::from_u32_unchecked(initial_b);

    // Generate Fibonacci sequence ONLY up to target_element
    let rows_to_compute = (target_element + 1).min(n_rows);
    let mut target_c_value = BaseField::zero();

    for row in 0..rows_to_compute {
        let c = a + b;

        col_a.set(row, a);
        col_b.set(row, b);
        col_c.set(row, c);

        // Save the c value at target_element (BEFORE bit-reverse)
        if row == target_element {
            target_c_value = c;
        }

        a = b;
        b = c;
    }

    // Rows from rows_to_compute..n_rows remain zeros

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col_a.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_b.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_c.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    (
        vec![
            CircleEvaluation::new(domain, col_a),
            CircleEvaluation::new(domain, col_b),
            CircleEvaluation::new(domain, col_c),
        ],
        target_c_value,
    )
}

/// Generate interaction trace for Computing component using LogUp
///
/// Only the target_element row yields its Fibonacci result c:
///   Adds: +1 / (c - z)  to the LogUp column for that row
///   Other rows contribute 0 (numerator=0)
///
/// This allows Scheduler to verify it's using the correct Fibonacci value
pub fn gen_computing_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    fibonacci_relation: &FibonacciRelation,
    target_element: usize,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let n_rows = 1 << log_size;

    // Create selector column: 1 only for target_element, 0 for rest
    // This gives us proper SIMD masking for the target lane
    let mut selector_col = Col::<SimdBackend, BaseField>::zeros(n_rows);
    selector_col.set(target_element, BaseField::one());

    // IMPORTANT: Apply bit-reverse to match trace column ordering!
    bit_reverse_coset_to_circle_domain_order(selector_col.as_mut_slice());

    let mut logup_gen = LogupTraceGenerator::new(log_size);

    {
        let mut col_gen = logup_gen.new_col();

        // For each vec_row, yield the c value with selector masking
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let c_value = trace[2].data[vec_row]; // Column c

            // Compute denominator: fibonacci_relation.combine([c])
            let denom = fibonacci_relation.combine(&[c_value]);

            // Use selector from helper column - this has proper SIMD masking!
            // Only the lane corresponding to target_element will have numerator=1
            let numerator = selector_col.data[vec_row].into();

            col_gen.write_frac(vec_row, numerator, denom);
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}

/// Trace for Scheduler component: reads one Fibonacci value
///
/// Returns 1 column: [fib_c]
/// where fib_c is the target Fibonacci value from Computing component
/// All rows have the same constant value
pub fn gen_scheduler_trace(
    log_size: u32,
    fib_c_value: BaseField,  // c value from Computing
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;

    println!("  Scheduler trace: fib_c={}", fib_c_value.0);

    let mut col_fib_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // ALL rows have the same constant value
    for row in 0..n_rows {
        col_fib_c.set(row, fib_c_value);
    }

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col_fib_c.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![
        CircleEvaluation::new(domain, col_fib_c),
    ]
}

/// Generate interaction trace for Scheduler component using LogUp
///
/// For each row, uses one Fibonacci value (fib_c):
///   Adds: -1 / (fib_c - z)  to the LogUp column (uses value from Computing)
///
/// The sum of all LogUp entries from Computing and Scheduler should be 0
pub fn gen_scheduler_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    fibonacci_relation: &FibonacciRelation,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    // Create is_first selector column: 1 only for row 0, 0 for rest
    let n_rows = 1 << log_size;
    let mut is_first_col = Col::<SimdBackend, BaseField>::zeros(n_rows);
    is_first_col.set(0, BaseField::one());

    // IMPORTANT: Apply bit-reverse to match trace column ordering!
    bit_reverse_coset_to_circle_domain_order(is_first_col.as_mut_slice());

    {
        let mut col_gen = logup_gen.new_col();

        // For each row, use fib_c (column 0)
        // Multiplicity controlled by is_first selector (only row 0 contributes)
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let fib_c_value = trace[0].data[vec_row]; // Column fib_c

            // Compute denominator
            let denom: PackedSecureField = fibonacci_relation.combine(&[fib_c_value]);

            // Multiplicity is controlled by is_first selector
            let is_first_value = is_first_col.data[vec_row]; // 1 for row 0, 0 for rest
            let is_first_secure: PackedSecureField = is_first_value.into();
            let numerator = -is_first_secure; // -1 for row 0, 0 for rest

            col_gen.write_frac(vec_row, numerator, denom);
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}
