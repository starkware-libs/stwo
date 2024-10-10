use itertools::{chain, Itertools};
use num_traits::Zero;

use super::BlakeElements;
use crate::constraint_framework::logup::LogupAtRow;
use crate::constraint_framework::EvalAtRow;
use crate::core::lookups::utils::{Fraction, Reciprocal};
use crate::core::vcs::blake2s_ref::SIGMA;
use crate::examples::blake::round::RoundElements;
use crate::examples::blake::{Fu32, N_ROUNDS, STATE_SIZE};

pub fn eval_blake_scheduler_constraints<E: EvalAtRow>(
    eval: &mut E,
    blake_lookup_elements: &BlakeElements,
    round_lookup_elements: &RoundElements,
    mut logup: LogupAtRow<E>,
) {
    let messages: [Fu32<E::F>; STATE_SIZE] = std::array::from_fn(|_| eval_next_u32(eval));
    let states: [[Fu32<E::F>; STATE_SIZE]; N_ROUNDS + 1] =
        std::array::from_fn(|_| std::array::from_fn(|_| eval_next_u32(eval)));

    // Schedule.
    for [i, j] in (0..N_ROUNDS).array_chunks::<2>() {
        // Use triplet in round lookup.
        let [denom_i, denom_j] = [i, j].map(|idx| {
            let input_state = &states[idx];
            let output_state = &states[idx + 1];
            let round_messages = SIGMA[idx].map(|k| messages[k as usize].clone());
            round_lookup_elements.combine::<E::F, E::EF>(
                &chain![
                    input_state.iter().cloned().flat_map(Fu32::into_felts),
                    output_state.iter().cloned().flat_map(Fu32::into_felts),
                    round_messages.iter().cloned().flat_map(Fu32::into_felts)
                ]
                .collect_vec(),
            )
        });
        logup.write_frac(eval, Reciprocal::new(denom_i) + Reciprocal::new(denom_j));
    }

    let input_state = &states[0];
    let output_state = &states[N_ROUNDS];

    // TODO(alont): Remove blake interaction.
    logup.write_frac(
        eval,
        Fraction::new(
            E::EF::zero(),
            blake_lookup_elements.combine(
                &chain![
                    input_state.iter().cloned().flat_map(Fu32::into_felts),
                    output_state.iter().cloned().flat_map(Fu32::into_felts),
                    messages.iter().cloned().flat_map(Fu32::into_felts)
                ]
                .collect_vec(),
            ),
        ),
    );

    logup.finalize(eval);
}

fn eval_next_u32<E: EvalAtRow>(eval: &mut E) -> Fu32<E::F> {
    let l = eval.next_trace_mask();
    let h = eval.next_trace_mask();
    Fu32 { l, h }
}
