use num_traits::One;

use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::{EvalAtRow, FrameworkEval};
use crate::core::fields::qm31::QM31;
use crate::core::lookups::utils::Fraction;

const LOG_CONSTRAINT_DEGREE: u32 = 1;
pub const STATE_SIZE: usize = 2;
/// Random elements to combine the StateMachine state.
pub type StateMachineElements = LookupElements<STATE_SIZE>;

/// State machine with state of size `STATE_SIZE`.
/// Transition `COORDINATE` of state increments the state by 1 at that offset.
#[derive(Clone)]
pub struct StateTransitionEval<const COORDINATE: usize> {
    pub log_n_rows: u32,
    pub lookup_elements: StateMachineElements,
    pub total_sum: QM31,
}

impl<const COORDINATE: usize> FrameworkEval for StateTransitionEval<COORDINATE> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_CONSTRAINT_DEGREE
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let [is_first] = eval.next_interaction_mask(2, [0]);
        let mut logup: LogupAtRow<E> = LogupAtRow::new(1, self.total_sum, None, is_first);

        let input_state: [_; STATE_SIZE] = std::array::from_fn(|_| eval.next_trace_mask());
        let input_denom: E::EF = self.lookup_elements.combine(&input_state);

        let mut output_state = input_state;
        output_state[COORDINATE] += E::F::one();
        let output_denom: E::EF = self.lookup_elements.combine(&output_state);

        // Add to the total sum (1/input_denom - 1/output_denom).
        logup.write_frac(
            &mut eval,
            Fraction::new(output_denom - input_denom, output_denom * input_denom),
        );

        logup.finalize(&mut eval);
        eval
    }
}
