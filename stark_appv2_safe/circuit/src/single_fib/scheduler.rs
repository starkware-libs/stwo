use num_traits::One;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry, ORIGINAL_TRACE_IDX};

use super::{FibonacciRelation, LOG_CONSTRAINT_DEGREE};

/// Evaluator for Scheduler component (simplified version)
///
/// Trace columns (ORIGINAL_TRACE_IDX):
/// - Column 0: fib_c (the fibonacci value read from computing)
///
/// Constraints:
/// 1. Value is constant across rows (transition constraints)
/// 2. LogUp uses only first row
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

        let [fib_c_curr, fib_c_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);

        // Constraint: Transition constraint - value is constant
        // Disabled for first row
        let not_first = E::F::one() - is_first.clone();
        eval.add_constraint(not_first.clone() * (fib_c_curr.clone() - fib_c_prev));

        // LogUp: Use value ONLY in first row (multiplicity = -is_first)
        eval.add_to_relation(RelationEntry::new(
            &self.fibonacci_relation,
            (-is_first.clone()).into(), // multiplicity: -1 for row 0, 0 for rest (convert F to EF)
            &[fib_c_curr],              // value from Computing
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type FibonacciSchedulerComponent = FrameworkComponent<FibonacciSchedulerEval>;
