use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

/// Wide Fibonacci component with normal addition: f(n) = f(n-1) + f(n-2)
/// Each row contains a COMPLETE Fibonacci sequence of length n_columns
///
/// Structure (horizontal):
/// Row 0: [f(0), f(1), f(2), f(3), ..., f(n_columns-1)]
/// Row 1: [f(0), f(1), f(2), f(3), ..., f(n_columns-1)]  (another instance)
/// Row 2: [f(0), f(1), f(2), f(3), ..., f(n_columns-1)]  (another instance)
///
/// This is different from SimpleFibonacci which is vertical (3 cols, many rows)
#[derive(Clone)]
pub struct WideFibonacciEval {
    pub log_n_rows: u32,
    pub n_columns: usize,  // Runtime parameter!
}

impl FrameworkEval for WideFibonacciEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask(); // Column 0: f(0)
        let mut b = eval.next_trace_mask(); // Column 1: f(1)

        // Chain constraints across columns in the same row
        for _ in 2..self.n_columns {
            let c = eval.next_trace_mask(); // Column i: f(i)

            // Constraint: f(i) = f(i-1) + f(i-2)
            // This enforces continuity ACROSS COLUMNS in the same row
            eval.add_constraint(c.clone() - (a.clone() + b.clone()));

            // Shift: next iteration will check f(i+1) = f(i) + f(i-1)
            a = b;
            b = c;
        }

        eval
    }
}

pub type WideFibonacciComponent = FrameworkComponent<WideFibonacciEval>;

/// Generate trace for wide fibonacci
///
/// Returns trace with n_columns columns, where each row is a complete Fibonacci sequence
pub fn gen_wide_fibonacci_trace(
    log_n_rows: u32,
    n_columns: usize,
    initial_a: u32, // f(0)
    initial_b: u32, // f(1)
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    assert!(n_columns >= 2, "Need at least 2 columns for Fibonacci");

    let n_rows = 1 << log_n_rows;

    // Create n_columns columns (one for each element in the sequence)
    let mut trace: Vec<Col<SimdBackend, BaseField>> = (0..n_columns)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(n_rows))
        .collect();

    // Fill each row with a complete Fibonacci sequence
    for row in 0..n_rows {
        let mut a = BaseField::from_u32_unchecked(initial_a);
        let mut b = BaseField::from_u32_unchecked(initial_b);

        // Column 0 and 1: initial values
        trace[0].set(row, a);
        trace[1].set(row, b);

        // Columns 2..n_columns: compute Fibonacci
        for col in 2..n_columns {
            let c = a + b;
            trace[col].set(row, c);
            a = b;
            b = c;
        }
    }

    // Convert to CircleEvaluation
    let domain = CanonicCoset::new(log_n_rows).circle_domain();
    trace
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use num_traits::Zero;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::pcs::TreeVec;
    use stwo_constraint_framework::assert_constraints_on_polys;

    #[test]
    fn test_wide_fibonacci_trace() {
        let log_n_rows = 2; // 4 rows
        let n_columns = 50;
        let trace = gen_wide_fibonacci_trace(log_n_rows, n_columns, 0, 1);

        // Should have n_columns columns
        assert_eq!(trace.len(), n_columns);

        // Check first row has correct Fibonacci sequence
        // Row 0: [0, 1, 1, 2, 3, 5, 8, 13, ...]
        assert_eq!(trace[0].values.at(0), BaseField::from(0)); // f(0)
        assert_eq!(trace[1].values.at(0), BaseField::from(1)); // f(1)
        assert_eq!(trace[2].values.at(0), BaseField::from(1)); // f(2)
        assert_eq!(trace[3].values.at(0), BaseField::from(2)); // f(3)
        assert_eq!(trace[4].values.at(0), BaseField::from(3)); // f(4)
        assert_eq!(trace[5].values.at(0), BaseField::from(5)); // f(5)
        assert_eq!(trace[6].values.at(0), BaseField::from(8)); // f(6)
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        let log_n_rows = 3; // 8 rows
        let n_columns = 50;
        let trace = gen_wide_fibonacci_trace(log_n_rows, n_columns, 0, 1);

        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(log_n_rows),
            |eval| {
                WideFibonacciEval {
                    log_n_rows: log_n_rows,
                    n_columns: n_columns,
                }
                .evaluate(eval);
            },
            SecureField::zero(),
        );
    }

    #[test]
    fn test_different_initial_values() {
        let log_n_rows = 2;
        let n_columns = 50;
        let trace = gen_wide_fibonacci_trace(log_n_rows, n_columns, 2, 3);

        // Row 0 should be: [2, 3, 5, 8, 13, ...]
        assert_eq!(trace[0].values.at(0), BaseField::from(2)); // f(0)
        assert_eq!(trace[1].values.at(0), BaseField::from(3)); // f(1)
        assert_eq!(trace[2].values.at(0), BaseField::from(5)); // f(2)
        assert_eq!(trace[3].values.at(0), BaseField::from(8)); // f(3)
        assert_eq!(trace[4].values.at(0), BaseField::from(13)); // f(4)
    }

    #[test]
    fn test_all_rows_same() {
        let log_n_rows = 3; // 8 rows
        let n_columns = 20;
        let trace = gen_wide_fibonacci_trace(log_n_rows, n_columns, 0, 1);

        // All rows should have the same Fibonacci sequence
        for row in 1..8 {
            for col in 0..n_columns {
                assert_eq!(
                    trace[col].values.at(row),
                    trace[col].values.at(0),
                    "Row {} col {} should match row 0",
                    row,
                    col
                );
            }
        }
    }
}
