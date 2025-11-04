use num_traits::One;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry, ORIGINAL_TRACE_IDX};

use super::{FibonacciRelation, LOG_CONSTRAINT_DEGREE};

/// Evaluator for Scheduler component
///
/// Trace columns (ORIGINAL_TRACE_IDX):
/// - Column 0: fib1_c
/// - Column 1: fib2_c
/// - Column 2: sum (fib1_c + fib2_c)
///
/// Constraints:
/// 1. sum = fib1_c + fib2_c (for all rows)
/// 2. Values are constant across rows (transition constraints)
/// 3. LogUp uses only first row
#[derive(Clone)]
pub struct FibonacciSchedulerEval {
    pub log_n_rows: u32,
    pub fibonacci_relation: FibonacciRelation,
    pub claimed_sum: SecureField,
    pub is_first_id: PreProcessedColumnId,
}

impl FrameworkEval for FibonacciSchedulerEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_first = eval.get_preprocessed_column(self.is_first_id.clone());

        let [fib1_c_curr, fib1_c_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [fib2_c_curr, fib2_c_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [sum_curr, sum_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);

        // Constraint 1: sum = fib1_c + fib2_c (for all rows)
        eval.add_constraint(sum_curr.clone() - (fib1_c_curr.clone() + fib2_c_curr.clone()));

        // Constraint 2: Transition constraints - values are constant
        // Disabled for first row
        let not_first = E::F::one() - is_first.clone();
        eval.add_constraint(not_first.clone() * (fib1_c_curr.clone() - fib1_c_prev));
        eval.add_constraint(not_first.clone() * (fib2_c_curr.clone() - fib2_c_prev));
        eval.add_constraint(not_first.clone() * (sum_curr.clone() - sum_prev));

        // LogUp: Use values ONLY in first row (multiplicity = -is_first)
        eval.add_to_relation(RelationEntry::new(
            &self.fibonacci_relation,
            (-is_first.clone()).into(), // multiplicity: -1 for row 0, 0 for rest (convert F to EF)
            &[fib1_c_curr],             // value from Computing1
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.fibonacci_relation,
            (-is_first.clone()).into(), // multiplicity: -1 for row 0, 0 for rest (convert F to EF)
            &[fib2_c_curr],             // value from Computing2
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type FibonacciSchedulerComponent = FrameworkComponent<FibonacciSchedulerEval>;
