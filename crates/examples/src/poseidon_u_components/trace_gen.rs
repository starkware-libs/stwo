use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{LogupTraceGenerator, Relation};

use super::{
    PoseidonElements, RATE, N_STATE, N_COLUMNS, N_HALF_FULL_ROUNDS, N_PARTIAL_ROUNDS,
    EXTERNAL_ROUND_CONSTS, INTERNAL_ROUND_CONSTS,
    apply_external_round_matrix, apply_internal_round_matrix, pow5,
};

/// Lookup data for Poseidon (initial and final states for LogUp)
pub struct LookupData {
    pub initial_state: [Col<SimdBackend, BaseField>; N_STATE],
    pub final_state: [Col<SimdBackend, BaseField>; N_STATE],
}

/// Generate trace for Poseidon Computing component
///
/// Returns (trace columns, lookup_data)
///
/// Generates Poseidon hash ONLY for active rows (with messages).
/// Padding rows are left as zeros with constraints disabled by is_active selector.
///
/// This achieves the optimization: computing 2 out of 128 Poseidon permutations
/// is 64x faster than computing all 128!
pub fn gen_computing_trace(
    log_size: u32,
    messages: Vec<[BaseField; RATE]>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    LookupData,
) {
    use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;

    let n_rows = 1 << log_size;
    let n_messages = messages.len();

    println!("🚀 OPTIMIZATION: Computing Poseidon for {} messages out of {} rows", n_messages, n_rows);
    println!("   Active rows: {} ({:.2}%)", n_messages, (n_messages as f64 / n_rows as f64) * 100.0);
    println!("   Padding rows: {} (filled with zeros)", n_rows - n_messages);

    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(n_rows))
        .collect::<Vec<_>>();
    let mut lookup_data = LookupData {
        initial_state: std::array::from_fn(|_| Col::<SimdBackend, BaseField>::zeros(n_rows)),
        final_state: std::array::from_fn(|_| Col::<SimdBackend, BaseField>::zeros(n_rows)),
    };

    // Generate trace ONLY for active rows (with real messages)
    let mut prev_output: Option<[BaseField; N_STATE]> = None;

    for row in 0..n_messages {
        let mut col_index = 0;
        let message = messages[row];

        // Write message columns (8 elements)
        for i in 0..RATE {
            trace[col_index].set(row, message[i]);
            col_index += 1;
        }

        // Compute initial state
        let mut state: [BaseField; N_STATE] = if let Some(prev) = prev_output {
            // Not first row: state = [prev_rate + message, prev_capacity]
            std::array::from_fn(|i| {
                if i < RATE {
                    prev[i] + message[i]
                } else {
                    prev[i]
                }
            })
        } else {
            // First row: state = [message, zeros]
            std::array::from_fn(|i| {
                if i < RATE {
                    message[i]
                } else {
                    BaseField::from_u32_unchecked(0)
                }
            })
        };

        // Write initial state columns (16 elements)
        for i in 0..N_STATE {
            trace[col_index].set(row, state[i]);
            lookup_data.initial_state[i].set(row, state[i]);
            col_index += 1;
        }


        // Poseidon permutation
        // 4 full rounds
        for round in 0..N_HALF_FULL_ROUNDS {
            for i in 0..N_STATE {
                state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            }
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i]));
            for &s in &state {
                trace[col_index].set(row, s);
                col_index += 1;
            }
        }

        // Partial rounds
        for round in 0..N_PARTIAL_ROUNDS {
            state[0] += INTERNAL_ROUND_CONSTS[round];
            apply_internal_round_matrix(&mut state);
            state[0] = pow5(state[0]);
            trace[col_index].set(row, state[0]);
            col_index += 1;
        }

        // Last 4 full rounds
        for round in N_HALF_FULL_ROUNDS..2 * N_HALF_FULL_ROUNDS {
            for i in 0..N_STATE {
                state[i] += EXTERNAL_ROUND_CONSTS[round][i];
            }
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5(state[i]));
            for &s in &state {
                trace[col_index].set(row, s);
                col_index += 1;
            }
        }

        // Write final state columns (16 elements)
        for i in 0..N_STATE {
            trace[col_index].set(row, state[i]);
            lookup_data.final_state[i].set(row, state[i]);
            col_index += 1;
        }


        // Save output for next row chaining
        prev_output = Some(state);
    }

    println!("✅ Optimization complete: {} Poseidon permutations computed (instead of {})",
             n_messages, n_rows);
    println!("   Padding rows ({}) remain as zeros (constraints disabled by is_active=0)",
             n_rows - n_messages);
    if n_messages > 0 && n_rows > n_messages {
        println!("   Speedup: {}x", n_rows / n_messages);
    }

    // Rows from n_messages..n_rows remain zeros

    // Convert to bit-reversed circle domain order
    for col in &mut trace {
        bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    }
    // NOTE: lookup_data is NOT bit-reversed! We use .data[vec_row] which is already in SIMD-packed order
    // The trace columns are bit-reversed for constraints, but lookup_data stays in coset order for LogUp

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace_evals = trace
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    (trace_evals, lookup_data)
}

/// Generate interaction trace for Computing component using LogUp
///
/// IMPORTANT: This follows the same pattern as stark_appv2_safe/circuit/src/multi_fib/trace_gen.rs
///
/// Key points:
/// 1. Reads state values DIRECTLY from trace columns (already bit-reversed)
/// 2. Uses preprocessed is_active column for masking
/// 3. Combines two LogUp fractions into one column (for finalize_logup_in_pairs)
///
/// Constraints (in computing.rs) have TWO add_to_relation calls:
///   - +is_active / initial_state
///   - -is_active / final_state
///
/// This function combines them using the formula:
///   (+is_active)/initial + (-is_active)/final
///   = is_active * (final - initial) / (initial * final)
///
/// Only active rows contribute to LogUp (is_active=1).
/// Padding rows have is_active=0, so numerator=0 and they don't contribute.
pub fn gen_computing_interaction_trace(
    log_size: u32,
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    lookup_elements: &PoseidonElements,
    is_active_col: &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    // Use preprocessed is_active column (already bit-reversed)
    let is_active_data = &is_active_col.values;

    // Trace column layout:
    // - Columns 0-7: message (RATE=8)
    // - Columns 8-23: initial_state (N_STATE=16)
    // - Columns 24+: intermediate states during permutation
    // - Last 16 columns: final_state (N_STATE=16)
    let initial_state_start = RATE;
    let final_state_start = trace.len() - N_STATE;

    let mut logup_gen = LogupTraceGenerator::new(log_size);

    // Similar to stark_appv2_safe: Use selector approach
    // We have TWO add_to_relation calls in constraints:
    //   1. +is_active / initial_state
    //   2. -is_active / final_state
    // For finalize_logup_in_pairs(), combine them like in scheduler example:
    //   (+is_active)/denom0 + (-is_active)/denom1 = is_active*(denom1-denom0)/(denom0*denom1)
    {
        let mut col_gen = logup_gen.new_col();

        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            // Read from trace columns (already bit-reversed) like stark_appv2_safe does
            let initial_state_packed: [_; N_STATE] =
                std::array::from_fn(|i| trace[initial_state_start + i].values.data[vec_row]);
            let final_state_packed: [_; N_STATE] =
                std::array::from_fn(|i| trace[final_state_start + i].values.data[vec_row]);

            let denom0: PackedSecureField = lookup_elements.combine(&initial_state_packed);
            let denom1: PackedSecureField = lookup_elements.combine(&final_state_packed);
            let is_active_packed: PackedSecureField = is_active_data.data[vec_row].into();

            // Combined formula for two fractions (like scheduler in stark_appv2_safe):
            // +is_active/denom0 + (-is_active)/denom1
            // = is_active * (1/denom0 - 1/denom1)
            // = is_active * (denom1 - denom0) / (denom0 * denom1)
            let numerator = is_active_packed * (denom1 - denom0);
            let denominator = denom0 * denom1;

            col_gen.write_frac(vec_row, numerator, denominator);
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}

/// Generate trace for Scheduler component
///
/// For now, simple trace with just n_messages as constant column
pub fn gen_scheduler_trace(
    log_size: u32,
    n_messages: usize,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;

    println!("  Scheduler trace: n_messages={}", n_messages);

    // Column 0: n_messages (constant across all rows)
    let mut col_n_messages = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let n_msg_field = BaseField::from_u32_unchecked(n_messages as u32);
    for row in 0..n_rows {
        col_n_messages.set(row, n_msg_field);
    }

    bit_reverse_coset_to_circle_domain_order(col_n_messages.as_mut_slice());

    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![CircleEvaluation::new(domain, col_n_messages)]
}

/// Generate interaction trace for Scheduler component
///
/// For now, scheduler doesn't use LogUp (no coordination needed).
/// Returns empty trace and zero claimed_sum.
pub fn gen_scheduler_interaction_trace(
    _log_size: u32,
    _lookup_elements: &PoseidonElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    // No LogUp for scheduler component (for now)
    // Return empty trace and zero claimed_sum
    (vec![], SecureField::zero())
}
