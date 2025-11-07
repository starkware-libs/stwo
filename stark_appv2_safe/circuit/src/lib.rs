use std::fs::File;
use std::io::Write;

use num_traits::One;

// Multi-component Fibonacci example with LogUp
pub mod multi_fib;

// Single-component Fibonacci example with LogUp
pub mod single_fib;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, ORIGINAL_TRACE_IDX};

/// ✅ SAFE Fibonacci component with transition constraints!
///
/// This implements f(n) = f(n-1) + f(n-2) with BOTH:
/// - Intra-row constraints: c = a + b
/// - Transition constraints: row[i].a == row[i-1].b AND row[i].b == row[i-1].c
///
/// This ensures the entire trace forms one continuous Fibonacci sequence!
#[derive(Clone)]
pub struct SimpleFibonacciEval {
    pub log_n_rows: u32,
    pub is_first_id: PreProcessedColumnId,
}

impl FrameworkEval for SimpleFibonacciEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read is_first selector (1 for first row, 0 for others)
        let is_first = eval.get_preprocessed_column(self.is_first_id.clone());

        // Read current and previous row values using offsets [0, -1]
        let [a_curr, _a_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [b_curr, b_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [c_curr, c_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);

        // Constraint 1: Intra-row constraint (c = a + b) for current row
        eval.add_constraint(c_curr.clone() - (a_curr.clone() + b_curr.clone()));

        // Constraint 2: Transition constraint a_curr == b_prev
        // Disabled for first row using (1 - is_first) multiplier
        eval.add_constraint((E::F::one() - is_first.clone()) * (a_curr.clone() - b_prev));

        // Constraint 3: Transition constraint b_curr == c_prev
        // Disabled for first row using (1 - is_first) multiplier
        eval.add_constraint((E::F::one() - is_first.clone()) * (b_curr.clone() - c_prev));

        // First row: a = 0
        eval.add_constraint(is_first.clone() * a_curr.clone());
        // First row: b = 1
        eval.add_constraint(is_first.clone() * (b_curr.clone() - E::F::one()));


        eval
    }
}

pub type SimpleFibonacciComponent = FrameworkComponent<SimpleFibonacciEval>;

/// Generate is_first preprocessed column (1 for first row, 0 for others)
pub fn gen_is_first_column(log_size: u32) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Set first row to 1
    col.set(0, BaseField::from_u32_unchecked(1));

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_first_column_id(log_size: u32) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_first_{}", log_size),
    }
}

/// Generate is_active preprocessed column: 1 for rows 0..=target_element, 0 for rest
pub fn gen_is_active_column(
    log_size: u32,
    target_element: usize,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Set 1 for all rows up to and including target_element
    for row in 0..=target_element.min(n_rows - 1) {
        col.set(row, BaseField::one());
    }

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_active_column_id(log_size: u32, target_element: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_active_{}_upto{}", log_size, target_element),
    }
}

/// Generate is_target preprocessed column: 1 ONLY for target_element, 0 for rest (for LogUp)
pub fn gen_is_target_column(
    log_size: u32,
    target_element: usize,
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    let n_rows = 1 << log_size;
    let mut col = Col::<SimdBackend, BaseField>::zeros(n_rows);

    // Set 1 ONLY for target_element
    if target_element < n_rows {
        col.set(target_element, BaseField::one());
    }

    // Convert to bit-reversed circle domain order
    bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());

    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
}

pub fn is_target_column_id(log_size: u32, target_element: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("is_target_{}_row{}", log_size, target_element),
    }
}

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

    // Convert columns to bit-reversed circle domain order
    // This is required for next_interaction_mask with offsets to work correctly!
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

        println!("\n=== Testing Fibonacci Constraints with Transitions ===");
        println!("First 5 rows of trace (bit-reversed order):");
        for row in 0..5.min(trace[0].values.len()) {
            println!(
                "Row {}: a={}, b={}, c={}",
                row,
                trace[0].values.at(row).0,
                trace[1].values.at(row).0,
                trace[2].values.at(row).0,
            );
        }

        // Generate preprocessed trace with is_first column
        let is_first_col = gen_is_first_column(log_size);
        let preprocessed_trace = vec![is_first_col];

        let traces = TreeVec::new(vec![preprocessed_trace, trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        println!("\nAttempting constraint verification with transition constraints...");
        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(log_size),
            |eval| {
                SimpleFibonacciEval {
                    log_n_rows: log_size,
                    is_first_id: is_first_column_id(log_size),
                }
                .evaluate(eval);
            },
            SecureField::zero(),
        );

        println!("✓ All constraints satisfied!");
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