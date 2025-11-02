use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry, ORIGINAL_TRACE_IDX};

use super::{FibonacciRelation, LOG_CONSTRAINT_DEGREE};

#[derive(Clone)]
pub struct FibonacciComputingEval {
    pub log_n_rows: u32,
    pub initial_a: u32,
    pub initial_b: u32,
    pub fibonacci_relation: FibonacciRelation,
    pub claimed_sum: SecureField,
    pub is_first_id: PreProcessedColumnId,
}

impl FrameworkEval for FibonacciComputingEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_first = eval.get_preprocessed_column(self.is_first_id.clone());

        let [a_curr, _a_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [b_curr, b_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [c_curr, c_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);

        // Constraint 1: c = a + b 
        eval.add_constraint(c_curr.clone() - (a_curr.clone() + b_curr.clone()));

        // Constraint 2: Transition a_curr = b_prev
        // Disabled for first row
        eval.add_constraint((E::F::one() - is_first.clone()) * (a_curr.clone() - b_prev));

        // Constraint 3: Transition b_curr = c_prev
        // Disabled for first row 
        eval.add_constraint((E::F::one() - is_first.clone()) * (b_curr.clone() - c_prev));

        // Constraint 4: First row initial values
        // First row: a = initial_a
        eval.add_constraint(is_first.clone() * (a_curr.clone() - E::F::from(BaseField::from_u32_unchecked(self.initial_a))));
        // First row: b = initial_b
        eval.add_constraint(is_first.clone() * (b_curr.clone() - E::F::from(BaseField::from_u32_unchecked(self.initial_b))));

        // LogUp
        eval.add_to_relation(RelationEntry::new(
            &self.fibonacci_relation,
            E::EF::one(), // multiplicity = +1 
            &[c_curr],    // value = c
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type FibonacciComputingComponent = FrameworkComponent<FibonacciComputingEval>;
