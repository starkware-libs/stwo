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
    pub is_active_id: PreProcessedColumnId,  // 1 for rows 0..=target_element, 0 for rest
    pub is_target_id: PreProcessedColumnId,  // 1 only for target_element (for LogUp)
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
        let is_active = eval.get_preprocessed_column(self.is_active_id.clone());
        let is_target = eval.get_preprocessed_column(self.is_target_id.clone());

        let [a_curr, _a_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [b_curr, b_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [c_curr, c_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);

        // Constraint 1: c = a + b
        // Multiply by is_active - only works for rows 0..=target_element
        eval.add_constraint(is_active.clone() * (c_curr.clone() - (a_curr.clone() + b_curr.clone())));

        // Constraint 2: Transition a_curr = b_prev
        // Disabled for: first row OR inactive rows
        let not_first = E::F::one() - is_first.clone();
        eval.add_constraint(is_active.clone() * not_first.clone() * (a_curr.clone() - b_prev));

        // Constraint 3: Transition b_curr = c_prev
        // Disabled for: first row OR inactive rows
        eval.add_constraint(is_active.clone() * not_first.clone() * (b_curr.clone() - c_prev));

        // Constraint 4-5: First row initial values
        // These only work for first row, which is always active
        eval.add_constraint(is_first.clone() * (a_curr.clone() - E::F::from(BaseField::from_u32_unchecked(self.initial_a))));
        eval.add_constraint(is_first.clone() * (b_curr.clone() - E::F::from(BaseField::from_u32_unchecked(self.initial_b))));

        // LogUp: yield ONLY for target_element row
        eval.add_to_relation(RelationEntry::new(
            &self.fibonacci_relation,
            is_target.into(),    // multiplicity: 1 only for target_element, 0 for rest (convert F to EF)
            &[c_curr],           // value = c
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type FibonacciComputingComponent = FrameworkComponent<FibonacciComputingEval>;
