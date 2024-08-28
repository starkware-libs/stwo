use itertools::{chain, Itertools};
use num_traits::{One, Zero};

use super::BlakeElements;
use crate::constraint_framework::logup::LogupAtRow;
use crate::constraint_framework::EvalAtRow;
use crate::core::vcs::blake2s_ref::SIGMA;
use crate::examples::blake::round::RoundElements;
use crate::examples::blake::{Fu32, N_ROUNDS, STATE_SIZE};

pub fn eval_blake_scheduler_constraints<E: EvalAtRow>(
    eval: &mut E,
    blake_lookup_elements: &BlakeElements,
    round_lookup_elements: &RoundElements,
    mut logup: LogupAtRow<2, E>,
) {
    let messages: [Fu32<E::F>; STATE_SIZE] = std::array::from_fn(|_| eval_next_u32(eval));
    let states: [[Fu32<E::F>; STATE_SIZE]; N_ROUNDS + 1] =
        std::array::from_fn(|_| std::array::from_fn(|_| eval_next_u32(eval)));

    // Schedule.
    for i in 0..N_ROUNDS {
        let input_state = &states[i];
        let output_state = &states[i + 1];
        let round_messages = SIGMA[i].map(|j| messages[j as usize]);
        // Use triplet in round lookup.
        logup.push_lookup(
            eval,
            E::EF::one(),
            &chain![
                input_state.iter().copied().flat_map(Fu32::to_felts),
                output_state.iter().copied().flat_map(Fu32::to_felts),
                round_messages.iter().copied().flat_map(Fu32::to_felts)
            ]
            .collect_vec(),
            round_lookup_elements,
        )
    }

    let input_state = &states[0];
    let output_state = &states[N_ROUNDS];

    // TODO(spapini): Support multiplicities.
    // TODO(spapini): Change to -1.
    logup.push_lookup(
        eval,
        E::EF::zero(),
        &chain![
            input_state.iter().copied().flat_map(Fu32::to_felts),
            output_state.iter().copied().flat_map(Fu32::to_felts),
            messages.iter().copied().flat_map(Fu32::to_felts)
        ]
        .collect_vec(),
        blake_lookup_elements,
    );

    logup.finalize(eval);
}

fn eval_next_u32<E: EvalAtRow>(eval: &mut E) -> Fu32<E::F> {
    let l = eval.next_trace_mask();
    let h = eval.next_trace_mask();
    Fu32 { l, h }
}
