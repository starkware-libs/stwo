use std::fs::File;
use std::io::Write;

use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, ORIGINAL_TRACE_IDX};

/// ⚠️ UNSAFE Fibonacci component - Educational example only!
///
/// This implements f(n) = f(n-1) + f(n-2) with ONLY intra-row constraints.
/// Each row is [a, b, c] where c = a + b, but there's NO enforcement that:
/// - row[i+1].a == row[i].b
/// - row[i+1].b == row[i].c
///
/// A malicious prover can generate arbitrary rows that each satisfy c = a + b
/// without forming a continuous Fibonacci sequence.
///
/// For SAFE Fibonacci, see stark_app_wide which uses horizontal layout.
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
        // 🔥 ATTEMPT: Try to read current and next row values using next_interaction_mask
        // This is expected to fail or give garbage values for ORIGINAL_TRACE_IDX
        let [a, a_next] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, 1]);
        let [b, b_next] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, 1]);
        let c = eval.next_trace_mask();
        println!("VALUES EVALUATE");

        // Constraint 1: Intra-row constraint (c = a + b)
        eval.add_constraint(c.clone() - (a.clone() + b.clone()));

        // Constraint 2: Transition constraint row[i+1].a == row[i].b
        eval.add_constraint(a_next - b.clone());

        // Constraint 3: Transition constraint row[i+1].b == row[i].c
        eval.add_constraint(b_next - c);

        eval
    }
}

pub type SimpleFibonacciComponent = FrameworkComponent<SimpleFibonacciEval>;

pub fn gen_fibonacci_trace(
    log_size: u32,
    initial_a: u32, // f(0)
    initial_b: u32, // f(1)
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;

    let mut col_a = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_b = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    let mut a = BaseField::from_u32_unchecked(initial_a);
    let mut b = BaseField::from_u32_unchecked(initial_b);

    // Generate proper Fibonacci sequence
    for row in 0..n_rows {
        let c = a + b;

        col_a.set(row, a);
        col_b.set(row, b);
        col_c.set(row, c);

        a = b;
        b = c;
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![
        CircleEvaluation::new(domain, col_a),
        CircleEvaluation::new(domain, col_b),
        CircleEvaluation::new(domain, col_c),
    ]
}

/// Dump trace to a JSON file for inspection
pub fn dump_trace_to_file(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    filename: &str,
) -> std::io::Result<()> {
    let n_rows = trace[0].values.len();
    let mut file = File::create(filename)?;

    writeln!(file, "{{")?;
    writeln!(file, "  \"n_rows\": {},", n_rows)?;
    writeln!(file, "  \"rows\": [")?;

    for row in 0..n_rows {
        let a = trace[0].values.at(row);
        let b = trace[1].values.at(row);
        let c = trace[2].values.at(row);

        writeln!(
            file,
            "    {{\"row\": {}, \"a\": {}, \"b\": {}, \"c\": {}}}{}",
            row,
            a.0,
            b.0,
            c.0,
            if row < n_rows - 1 { "," } else { "" }
        )?;
    }

    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;

    Ok(())
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
    fn test_fibonacci_trace() {
        let log_size = 4; // 16 rows
        let trace = gen_fibonacci_trace(log_size, 1, 1);

        // Check first few values
        assert_eq!(trace[0].values.at(0), BaseField::from(1)); // a = 1
        assert_eq!(trace[1].values.at(0), BaseField::from(1)); // b = 1
        assert_eq!(trace[2].values.at(0), BaseField::from(2)); // c = 2

        assert_eq!(trace[0].values.at(1), BaseField::from(1)); // a = 1
        assert_eq!(trace[1].values.at(1), BaseField::from(2)); // b = 2
        assert_eq!(trace[2].values.at(1), BaseField::from(3)); // c = 3

        assert_eq!(trace[0].values.at(2), BaseField::from(2)); // a = 2
        assert_eq!(trace[1].values.at(2), BaseField::from(3)); // b = 3
        assert_eq!(trace[2].values.at(2), BaseField::from(5)); // c = 5
    }

    #[test]
    fn test_fibonacci_constraints() {
        let log_size = 4; // Smaller size for easier debugging
        let trace = gen_fibonacci_trace(log_size, 1, 1);

        println!("\n=== Testing Fibonacci Constraints ===");
        println!("First 5 rows of trace:");
        for row in 0..5.min(trace[0].values.len()) {
            println!(
                "Row {}: a={}, b={}, c={}",
                row,
                trace[0].values.at(row).0,
                trace[1].values.at(row).0,
                trace[2].values.at(row).0,
            );
        }

        let traces = TreeVec::new(vec![vec![], trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        println!("\nAttempting constraint verification...");
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
    fn test_different_initial_values() {
        let log_size = 4;
        let trace = gen_fibonacci_trace(log_size, 2, 3);

        // Sequence: 2, 3, 5, 8, 13, 21, ...
        assert_eq!(trace[0].values.at(0), BaseField::from(2)); // a = 2
        assert_eq!(trace[1].values.at(0), BaseField::from(3)); // b = 3
        assert_eq!(trace[2].values.at(0), BaseField::from(5)); // c = 5

        assert_eq!(trace[0].values.at(1), BaseField::from(3)); // a = 3
        assert_eq!(trace[1].values.at(1), BaseField::from(5)); // b = 5
        assert_eq!(trace[2].values.at(1), BaseField::from(8)); // c = 8
    }
}
