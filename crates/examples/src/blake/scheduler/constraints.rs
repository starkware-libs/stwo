use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::{chain, Itertools};
use num_traits::{One, Zero};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::FieldExpOps;
use stwo_constraint_framework::{EvalAtRow, RelationEntry};

use crate::blake::blake3::MSG_SCHEDULE;

use super::BlakeElements;
use crate::blake::round::RoundElements;
use crate::blake::{Fu32, N_ROUNDS, STATE_SIZE};

/// Applies Blake3 MSG_SCHEDULE permutation `iterations` times to messages
fn apply_blake3_permutation<F>(messages: &[Fu32<F>; STATE_SIZE], iterations: usize) -> [Fu32<F>; STATE_SIZE]
where
    F: FieldExpOps
        + Clone
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    let mut result = messages.clone();
    for _ in 0..iterations {
        let permuted = MSG_SCHEDULE.map(|i| result[i as usize].clone());
        result = permuted;
    }
    result
}

pub fn eval_blake_scheduler_constraints<E: EvalAtRow>(
    eval: &mut E,
    blake_lookup_elements: &BlakeElements,
    round_lookup_elements: &RoundElements,
) {
    let messages: [Fu32<E::F>; STATE_SIZE] = std::array::from_fn(|_| eval_next_u32(eval));
    let states: [[Fu32<E::F>; STATE_SIZE]; N_ROUNDS + 1] =
        std::array::from_fn(|_| std::array::from_fn(|_| eval_next_u32(eval)));

    // Schedule.
    // Blake3 permutes message BETWEEN rounds. We need to match the permutation
    // that gen.rs applies. For round idx, message has been permuted idx times.
    for [i, j] in (0..N_ROUNDS).array_chunks::<2>() {
        // Use triplet in round lookup.
        let [elems_i, elems_j] = [i, j].map(|idx| {
            let input_state = &states[idx];
            let output_state = &states[idx + 1];
            // Apply MSG_SCHEDULE permutation idx times to get round message
            let round_messages = apply_blake3_permutation(&messages, idx);
            chain![
                input_state.iter().cloned().flat_map(Fu32::into_felts),
                output_state.iter().cloned().flat_map(Fu32::into_felts),
                round_messages.iter().cloned().flat_map(Fu32::into_felts)
            ]
            .collect_vec()
        });
        eval.add_to_relation(RelationEntry::new(
            round_lookup_elements,
            E::EF::one(),
            &elems_i,
        ));
        eval.add_to_relation(RelationEntry::new(
            round_lookup_elements,
            E::EF::one(),
            &elems_j,
        ));
    }

    let input_state = &states[0];
    let output_state = &states[N_ROUNDS];

    // Blake lookup combined with last round if N_ROUNDS is odd
    // This matches the logic in gen_interaction_trace
    if N_ROUNDS % 2 == 1 {
        // Last round (unpaired) combined with blake lookup
        let last_round_idx = N_ROUNDS - 1;
        let last_round_input = &states[last_round_idx];
        let last_round_output = &states[last_round_idx + 1];
        let last_round_messages = apply_blake3_permutation(&messages, last_round_idx);

        // Add last round lookup
        eval.add_to_relation(RelationEntry::new(
            round_lookup_elements,
            E::EF::one(),
            &chain![
                last_round_input.iter().cloned().flat_map(Fu32::into_felts),
                last_round_output.iter().cloned().flat_map(Fu32::into_felts),
                last_round_messages.iter().cloned().flat_map(Fu32::into_felts)
            ]
            .collect_vec(),
        ));
    }

    // Blake lookup
    eval.add_to_relation(RelationEntry::new(
        blake_lookup_elements,
        E::EF::zero(),
        &chain![
            input_state.iter().cloned().flat_map(Fu32::into_felts),
            output_state.iter().cloned().flat_map(Fu32::into_felts),
            messages.iter().cloned().flat_map(Fu32::into_felts)
        ]
        .collect_vec(),
    ));

    eval.finalize_logup_in_pairs();
}

fn eval_next_u32<E: EvalAtRow>(eval: &mut E) -> Fu32<E::F> {
    let l = eval.next_trace_mask();
    let h = eval.next_trace_mask();
    Fu32 { l, h }
}
