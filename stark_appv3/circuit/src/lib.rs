use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

/// Simple Fibonacci component: f(n) = f(n-1) + f(n-2)
/// Each row represents one step in the sequence
#[derive(Clone)]
pub struct SimpleFibonacciEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for SimpleFibonacciEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read three consecutive values: a, b, c
        let a = eval.next_trace_mask(); // f(n-2)
        let b = eval.next_trace_mask(); // f(n-1)
        let c = eval.next_trace_mask(); // f(n)

        // Constraint: f(n) = f(n-1) + f(n-2)
        // This is an intra-row constraint (checks values in the same row)
        // Padding with zeros works: 0 = 0 + 0 ✓
        eval.add_constraint(c - (a + b));

        eval
    }
}

pub type SimpleFibonacciComponent = FrameworkComponent<SimpleFibonacciEval>;

/// Calculate the minimum log_size needed to compute f(target_n)
pub fn calculate_log_size(target_n: usize) -> u32 {
    // We need at least target_n - 1 rows to compute f(target_n)
    // (row 0 has f(2), row 1 has f(3), ..., row target_n-2 has f(target_n))
    let min_rows = target_n.saturating_sub(1).max(1);

    // Round up to next power of 2
    let log_size = (min_rows as f64).log2().ceil() as u32;

    // STARK/FRI requires minimum log_size = 2 (4 rows)
    log_size.max(2)
}

/// Generate trace for simple fibonacci sequence up to f(target_n)
/// Remaining rows are padded with zeros
///
/// Structure: 3 columns
/// - Column 0: f(n-2)
/// - Column 1: f(n-1)
/// - Column 2: f(n)
///
/// Returns: (trace, actual_value, log_size_used)
pub fn gen_fibonacci_trace(
    target_n: usize,
    initial_a: u32, // f(0)
    initial_b: u32, // f(1)
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    BaseField,
    u32,
) {
    let log_size = calculate_log_size(target_n);
    let n_rows = 1 << log_size;

    // Create 3 columns
    let mut col_a = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_b = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    let mut a = BaseField::from_u32_unchecked(initial_a);
    let mut b = BaseField::from_u32_unchecked(initial_b);

    // Track the target value
    let mut target_value = BaseField::from_u32_unchecked(0);

    // Compute Fibonacci up to target_n
    // Row i contains: [f(i), f(i+1), f(i+2)]
    let compute_rows = (target_n - 1).min(n_rows);

    for row in 0..compute_rows {
        let c = a + b;

        col_a.set(row, a);
        col_b.set(row, b);
        col_c.set(row, c);

        // Check if this row contains our target
        let current_index = row + 2;
        if current_index == target_n {
            target_value = c;
        }

        // Shift for next row
        a = b;
        b = c;
    }

    // Remaining rows are already zeros (padding)
    // Constraint 0 = 0 + 0 is satisfied ✓

    // Convert to CircleEvaluation
    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = vec![
        CircleEvaluation::new(domain, col_a),
        CircleEvaluation::new(domain, col_b),
        CircleEvaluation::new(domain, col_c),
    ];

    (trace, target_value, log_size)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::pcs::TreeVec;
    use stwo_constraint_framework::assert_constraints_on_polys;

    use super::*;

    #[test]
    fn test_calculate_log_size() {
        assert_eq!(calculate_log_size(2), 2); // 4 rows (minimum for STARK)
        assert_eq!(calculate_log_size(3), 2); // 4 rows (minimum for STARK)
        assert_eq!(calculate_log_size(4), 2); // 4 rows
        assert_eq!(calculate_log_size(5), 2); // 4 rows (f(5) in row 3)
        assert_eq!(calculate_log_size(10), 4); // 16 rows
        assert_eq!(calculate_log_size(50), 6); // 64 rows
        assert_eq!(calculate_log_size(100), 7); // 128 rows
    }

    #[test]
    fn test_fibonacci_trace_with_padding() {
        let target_n = 10;
        let (trace, value, log_size) = gen_fibonacci_trace(target_n, 0, 1);

        // Should use 16 rows (2^4)
        assert_eq!(log_size, 4);

        // Verify f(10) = 55 in standard Fibonacci
        assert_eq!(value, BaseField::from(55));

        // Check that padding rows are zeros
        let n_rows = 1 << log_size;
        for row in 9..n_rows {
            assert_eq!(trace[0].values.at(row), BaseField::from(0));
            assert_eq!(trace[1].values.at(row), BaseField::from(0));
            assert_eq!(trace[2].values.at(row), BaseField::from(0));
        }
    }

    #[test]
    fn test_fibonacci_constraints_with_padding() {
        let target_n = 20;
        let (trace, _value, log_size) = gen_fibonacci_trace(target_n, 0, 1);

        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(log_size),
            |eval| {
                SimpleFibonacciEval {
                    log_n_rows: log_size,
                }
                .evaluate(eval);
            },
            SecureField::zero(),
        );
    }

    #[test]
    fn test_specific_fibonacci_values() {
        // Test f(5) = 5
        let (_, value, _) = gen_fibonacci_trace(5, 0, 1);
        assert_eq!(value, BaseField::from(5));

        // Test f(10) = 55
        let (_, value, _) = gen_fibonacci_trace(10, 0, 1);
        assert_eq!(value, BaseField::from(55));

        // Test f(15) = 610
        let (_, value, _) = gen_fibonacci_trace(15, 0, 1);
        assert_eq!(value, BaseField::from(610));
    }
}
