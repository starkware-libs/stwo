use num_traits::{One, Zero};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

use super::PoseidonRelation;
use crate::{N_STATE, RATE};

/// Returns (trace columns, target_state_value)
/// trace: 40 columns [message(8), state_in(16), state_out(16)]
/// target_state_value: the state_out at target_element BEFORE bit-reverse
///
/// Generates Poseidon hash only up to target_element (inclusive), rest are zeros
pub fn gen_computing_trace(
    log_size: u32,
    initial_message: [BaseField; RATE],
    target_element: usize,
) -> (ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>, [BaseField; N_STATE]) {
    let n_rows = 1 << log_size;

    const N_COLUMNS: usize = RATE + N_STATE + N_STATE; // message + state_in + state_out = 40

    let mut trace_cols: Vec<Col<SimdBackend, BaseField>> =
        (0..N_COLUMNS).map(|_| Col::<SimdBackend, BaseField>::zeros(n_rows)).collect();

    // Track Poseidon state
    let mut state = [BaseField::zero(); N_STATE];
    let mut target_state_value = [BaseField::zero(); N_STATE];

    // Generate Poseidon hash ONLY up to target_element
    let rows_to_compute = (target_element + 1).min(n_rows);

    for row in 0..rows_to_compute {
        // Message: initial_message for all active rows
        for i in 0..RATE {
            trace_cols[i].set(row, initial_message[i]);
        }

        // First row: state_in = [initial_message, 0, 0, ...]
        // Other rows: state_in = state_out_prev + message (sponge construction)
        if row == 0 {
            for i in 0..RATE {
                state[i] = initial_message[i];
            }
            for i in RATE..N_STATE {
                state[i] = BaseField::zero();
            }
        } else {
            // Rate part: absorb message
            for i in 0..RATE {
                state[i] = state[i] + initial_message[i];
            }
            // Capacity part: unchanged
        }

        // Set state_in in trace
        for i in 0..N_STATE {
            trace_cols[RATE + i].set(row, state[i]);
        }

        // Apply Poseidon permutation: state_out = Poseidon(state_in)
        // For now, using identity (state_out = state_in)
        // This will be replaced with real Poseidon permutation
        let state_out = state;

        // Set state_out in trace
        for i in 0..N_STATE {
            trace_cols[RATE + N_STATE + i].set(row, state_out[i]);
        }

        // Save the state_out at target_element (BEFORE bit-reverse)
        if row == target_element {
            target_state_value = state_out;
        }

        // Update state for next iteration
        state = state_out;
    }

    // Rows from rows_to_compute..n_rows remain zeros

    // Convert to bit-reversed circle domain order
    for col in &mut trace_cols {
        bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = trace_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    (trace, target_state_value)
}

/// Generate interaction trace for Computing component using LogUp
pub fn gen_computing_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    poseidon_relation: &PoseidonRelation,
    target_element: usize,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let n_rows = 1 << log_size;

    // Create selector column: 1 only for target_element, 0 for rest
    let mut selector_col = Col::<SimdBackend, BaseField>::zeros(n_rows);
    selector_col.set(target_element, BaseField::one());

    // IMPORTANT: Apply bit-reverse to match trace column ordering!
    bit_reverse_coset_to_circle_domain_order(selector_col.as_mut_slice());

    let mut logup_gen = LogupTraceGenerator::new(log_size);

    {
        let mut col_gen = logup_gen.new_col();

        // For each vec_row, yield the state_out value with selector masking
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            // Extract state_out (last 16 columns)
            let state_out: [_; N_STATE] = std::array::from_fn(|i| {
                trace[RATE + N_STATE + i].data[vec_row]
            });

            // Compute denominator: poseidon_relation.combine(state_out)
            let denom = poseidon_relation.combine(&state_out);

            // Use selector - only the lane corresponding to target_element will have numerator=1
            let numerator = selector_col.data[vec_row].into();

            col_gen.write_frac(vec_row, numerator, denom);
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}
