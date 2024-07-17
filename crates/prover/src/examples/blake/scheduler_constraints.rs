use itertools::{chain, Itertools};
use num_traits::One;

use super::{Fu32, N_ROUNDS};
use crate::constraint_framework::logup::{LogupAtRow, LookupElements};
use crate::constraint_framework::EvalAtRow;
use crate::core::vcs::blake2s_ref::SIGMA;

pub struct BlakeSchedulerEval<'a, E: EvalAtRow> {
    pub eval: E,
    pub blake_lookup_elements: &'a LookupElements,
    pub round_lookup_elements: &'a LookupElements,
    pub logup: LogupAtRow<2, E>,
}
impl<'a, E: EvalAtRow> BlakeSchedulerEval<'a, E> {
    pub fn eval(mut self) -> E {
        let messages: [Fu32<E::F>; 16] = std::array::from_fn(|_| self.next_u32());
        let states: [[Fu32<E::F>; 16]; N_ROUNDS + 1] =
            std::array::from_fn(|_| std::array::from_fn(|_| self.next_u32()));

        // Schedule.
        for i in 0..N_ROUNDS {
            let input_state = &states[i];
            let output_state = &states[i + 1];
            let round_messages: [Fu32<E::F>; 16] =
                std::array::from_fn(|j| messages[SIGMA[i][j] as usize]);
            // Use triplet in round lookup.
            self.logup.push_lookup(
                &mut self.eval,
                E::EF::one(),
                &chain![
                    input_state.iter().copied().flat_map(Fu32::to_felts),
                    output_state.iter().copied().flat_map(Fu32::to_felts),
                    round_messages.iter().copied().flat_map(Fu32::to_felts)
                ]
                .collect_vec(),
                self.round_lookup_elements,
            )
        }

        let input_state = &states[0];
        let output_state = &states[N_ROUNDS];

        // TODO: support multiplicities.
        self.logup.push_lookup(
            &mut self.eval,
            E::EF::one() - E::EF::one(),
            &chain![
                input_state.iter().copied().flat_map(Fu32::to_felts),
                output_state.iter().copied().flat_map(Fu32::to_felts),
                messages.iter().copied().flat_map(Fu32::to_felts)
            ]
            .collect_vec(),
            self.blake_lookup_elements,
        );

        self.logup.finalize(&mut self.eval);
        self.eval
    }
    fn next_u32(&mut self) -> Fu32<E::F> {
        let l = self.eval.next_trace_mask();
        let h = self.eval.next_trace_mask();
        Fu32 { l, h }
    }
}
