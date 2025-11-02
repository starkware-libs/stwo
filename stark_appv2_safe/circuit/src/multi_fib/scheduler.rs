use num_traits::One;
use stwo::core::fields::qm31::SecureField;
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
/// 1. sum = fib1_c + fib2_c
#[derive(Clone)]
pub struct FibonacciSchedulerEval {
    pub log_n_rows: u32,
    pub fibonacci_relation: FibonacciRelation,
    pub claimed_sum: SecureField,
}

impl FrameworkEval for FibonacciSchedulerEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let fib1_c = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0])[0].clone();
        let fib2_c = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0])[0].clone();
        let sum = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0])[0].clone();

        // Constraint: sum = fib1_c + fib2_c
        eval.add_constraint(sum - (fib1_c.clone() + fib2_c.clone()));

        // LogUp: Use fib1_c value
        eval.add_to_relation(RelationEntry::new(
            &self.fibonacci_relation,
            -E::EF::one(), // multiplicity = -1 (use)
            &[fib1_c],     // value from Computing1
        ));

        // LogUp: Use fib2_c value
        eval.add_to_relation(RelationEntry::new(
            &self.fibonacci_relation,
            -E::EF::one(), // multiplicity = -1 (use)
            &[fib2_c],     // value from Computing2
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type FibonacciSchedulerComponent = FrameworkComponent<FibonacciSchedulerEval>;
