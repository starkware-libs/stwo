use num_traits::One;

use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::{EvalAtRow, FrameworkEval};
use crate::core::fields::qm31::QM31;
use crate::core::lookups::utils::Fraction;

pub const N_STATE: usize = 2;
pub type StateMachineElements = LookupElements<N_STATE>;

/// State machine with state of size `N_STATE`.
/// Transition `INDEX` of state increments the state by 1 at that offset.
#[derive(Clone)]
pub struct StateTransitionEval<const INDEX: usize> {
    pub log_n_rows: u32,
    pub lookup_elements: StateMachineElements,
    pub total_sum: QM31,
}

impl<const INDEX: usize> FrameworkEval for StateTransitionEval<INDEX> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let mut logup: LogupAtRow<E> = LogupAtRow::new(1, self.total_sum, None, is_first);

        let input_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());
        let q0: E::EF = self.lookup_elements.combine(&input_state);

        let mut output_state = input_state;
        output_state[INDEX] += E::F::one();
        let q1: E::EF = self.lookup_elements.combine(&output_state);

        // Add to the total sum (1/q0 - 1/q1).
        logup.write_frac(&mut eval, Fraction::new(q1 - q0, q1 * q0));

        logup.finalize(&mut eval);
        eval
    }
}
