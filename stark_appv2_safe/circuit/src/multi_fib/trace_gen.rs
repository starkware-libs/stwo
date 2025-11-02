use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

use super::FibonacciRelation;

/// Returns 3 columns: [a, b, c] where c = a + b
pub fn gen_computing_trace(
    log_size: u32,
    initial_a: u32,
    initial_b: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;

    let mut col_a = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_b = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    let mut a = BaseField::from_u32_unchecked(initial_a);
    let mut b = BaseField::from_u32_unchecked(initial_b);

    // Generate Fibonacci sequence
    for row in 0..n_rows {
        let c = a + b;

        col_a.set(row, a);
        col_b.set(row, b);
        col_c.set(row, c);

        a = b;
        b = c;
    }

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col_a.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_b.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_c.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![
        CircleEvaluation::new(domain, col_a),
        CircleEvaluation::new(domain, col_b),
        CircleEvaluation::new(domain, col_c),
    ]
}

/// Generate interaction trace for Computing component using LogUp
///
/// For each row, yields the Fibonacci result c:
///   Adds: +1 / (c - z)  to the LogUp column
///
/// This allows Scheduler to verify it's using the correct Fibonacci values
pub fn gen_computing_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    fibonacci_relation: &FibonacciRelation,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    {
        let mut col_gen = logup_gen.new_col();

        // For each row, yield the c value (column index 2)
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let c_value = trace[2].data[vec_row]; // Column c

            // Compute denominator: fibonacci_relation.combine([c])
            let denom = fibonacci_relation.combine(&[c_value]);

            // Write fraction: numerator=1, denominator=(c - z)
            // This represents: +1 / (c - z)
            col_gen.write_frac(
                vec_row,
                PackedM31::broadcast(BaseField::one()).into(), // numerator = 1
                denom,
            );
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}

/// Trace for Scheduler component: reads and sums Fibonacci values
///
/// Returns 3 columns: [fib1_c, fib2_c, sum]
/// where sum = fib1_c + fib2_c
///
/// Values fib1_c and fib2_c are copied from Computing traces (column c)
pub fn gen_scheduler_trace(
    log_size: u32,
    computing1_trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    computing2_trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;

    let mut col_fib1_c = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_fib2_c = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_sum = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Copy c values from both Computing traces and sum them
    for row in 0..n_rows {
        let fib1_c = computing1_trace[2].values.at(row); // c from Computing1
        let fib2_c = computing2_trace[2].values.at(row); // c from Computing2
        let sum = fib1_c + fib2_c;

        col_fib1_c.set(row, fib1_c);
        col_fib2_c.set(row, fib2_c);
        col_sum.set(row, sum);
    }

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col_fib1_c.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_fib2_c.as_mut_slice());
    bit_reverse_coset_to_circle_domain_order(col_sum.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![
        CircleEvaluation::new(domain, col_fib1_c),
        CircleEvaluation::new(domain, col_fib2_c),
        CircleEvaluation::new(domain, col_sum),
    ]
}

/// Generate interaction trace for Scheduler component using LogUp
///
/// For each row, uses two Fibonacci values (fib1_c and fib2_c):
///   Adds: -1 / (fib1_c - z)  to the LogUp column (uses value from Computing1)
///   Adds: -1 / (fib2_c - z)  to the LogUp column (uses value from Computing2)
///
/// The sum of all LogUp entries from Computing1, Computing2, and Scheduler should be 0
pub fn gen_scheduler_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    fibonacci_relation: &FibonacciRelation,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    {
        let mut col_gen = logup_gen.new_col();

        // For each row, use both fib1_c (column 0) and fib2_c (column 1)
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let fib1_c_value = trace[0].data[vec_row]; // Column fib1_c
            let fib2_c_value = trace[1].data[vec_row]; // Column fib2_c

            // Compute denominators for both values
            let denom1: PackedSecureField = fibonacci_relation.combine(&[fib1_c_value]);
            let denom2: PackedSecureField = fibonacci_relation.combine(&[fib2_c_value]);

            // We need to write TWO fractions per row:
            // -1 / (fib1_c - z) and -1 / (fib2_c - z)
            //
            // For finalize_logup_in_pairs(), we combine them:
            // -1/denom1 + -1/denom2 = -(denom1 + denom2) / (denom1 * denom2)

            let sum: PackedSecureField = denom1 + denom2;
            let numerator = -sum;
            let denominator = denom1 * denom2;

            col_gen.write_frac(vec_row, numerator, denominator);
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}
