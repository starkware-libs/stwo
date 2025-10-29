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
        let a = eval.next_trace_mask();      // f(n-2)
        let b = eval.next_trace_mask();      // f(n-1)
        let c = eval.next_trace_mask();      // f(n)

        // Constraint: f(n) = f(n-1) + f(n-2)
        // Which means: c = a + b
        eval.add_constraint(c - (a + b));

        eval
    }
}

pub type SimpleFibonacciComponent = FrameworkComponent<SimpleFibonacciEval>;

/// Generate trace for simple fibonacci sequence
///
/// Structure: 3 columns
/// - Column 0: f(n-2)
/// - Column 1: f(n-1)
/// - Column 2: f(n)
///
/// Example with initial values a=1, b=1:
/// ```
/// Row 0: [1, 1, 2]    (1 + 1 = 2)
/// Row 1: [1, 2, 3]    (1 + 2 = 3)
/// Row 2: [2, 3, 5]    (2 + 3 = 5)
/// Row 3: [3, 5, 8]    (3 + 5 = 8)
/// Row 4: [5, 8, 13]   (5 + 8 = 13)
/// ...
/// ```
pub fn gen_fibonacci_trace(
    log_size: u32,
    initial_a: u32,  // f(0)
    initial_b: u32,  // f(1)
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;

    // Create 3 columns
    let mut col_a = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_b = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    let mut a = BaseField::from_u32_unchecked(initial_a);
    let mut b = BaseField::from_u32_unchecked(initial_b);

    for row in 0..n_rows {
        let c = a + b;

        col_a.set(row, a);
        col_b.set(row, b);
        col_c.set(row, c);

        // Shift for next row
        a = b;
        b = c;
    }

    // Convert to CircleEvaluation
    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![
        CircleEvaluation::new(domain, col_a),
        CircleEvaluation::new(domain, col_b),
        CircleEvaluation::new(domain, col_c),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::pcs::TreeVec;
    use stwo::core::fields::qm31::SecureField;
    use itertools::Itertools;
    use num_traits::Zero;
    use stwo_constraint_framework::assert_constraints_on_polys;

    #[test]
    fn test_fibonacci_trace() {
        let log_size = 4;  // 16 rows
        let trace = gen_fibonacci_trace(log_size, 1, 1);

        // Check first few values
        assert_eq!(trace[0].values.at(0), BaseField::from(1));  // a = 1
        assert_eq!(trace[1].values.at(0), BaseField::from(1));  // b = 1
        assert_eq!(trace[2].values.at(0), BaseField::from(2));  // c = 2

        assert_eq!(trace[0].values.at(1), BaseField::from(1));  // a = 1
        assert_eq!(trace[1].values.at(1), BaseField::from(2));  // b = 2
        assert_eq!(trace[2].values.at(1), BaseField::from(3));  // c = 3

        assert_eq!(trace[0].values.at(2), BaseField::from(2));  // a = 2
        assert_eq!(trace[1].values.at(2), BaseField::from(3));  // b = 3
        assert_eq!(trace[2].values.at(2), BaseField::from(5));  // c = 5
    }

    #[test]
    fn test_fibonacci_constraints() {
        let log_size = 8;
        let trace = gen_fibonacci_trace(log_size, 1, 1);

        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys = traces.map(|trace| {
            trace.into_iter().map(|c| c.interpolate()).collect_vec()
        });

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(log_size),
            |eval| {
                SimpleFibonacciEval { log_n_rows: log_size }.evaluate(eval);
            },
            SecureField::zero(),
        );
    }

    #[test]
    fn test_different_initial_values() {
        let log_size = 4;
        let trace = gen_fibonacci_trace(log_size, 2, 3);

        // Sequence: 2, 3, 5, 8, 13, 21, ...
        assert_eq!(trace[0].values.at(0), BaseField::from(2));  // a = 2
        assert_eq!(trace[1].values.at(0), BaseField::from(3));  // b = 3
        assert_eq!(trace[2].values.at(0), BaseField::from(5));  // c = 5

        assert_eq!(trace[0].values.at(1), BaseField::from(3));  // a = 3
        assert_eq!(trace[1].values.at(1), BaseField::from(5));  // b = 5
        assert_eq!(trace[2].values.at(1), BaseField::from(8));  // c = 8
    }
}
