use num_traits::One;
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
    for col in &mut lookup_data.initial_state {
        bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    }
    for col in &mut lookup_data.final_state {
        bit_reverse_coset_to_circle_domain_order(col.as_mut_slice());
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace_evals = trace
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    (trace_evals, lookup_data)
}

/// Generate interaction trace for Computing component using LogUp
///
/// Only active rows contribute to LogUp (with is_active masking).
/// Padding rows have numerator=0, so they don't contribute.
pub fn gen_computing_interaction_trace(
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: &PoseidonElements,
    n_messages: usize,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let n_rows = 1 << log_size;

    // Create is_active selector: 1 for active rows, 0 for padding
    let mut is_active_col = Col::<SimdBackend, BaseField>::zeros(n_rows);
    for row in 0..n_messages.min(n_rows) {
        is_active_col.set(row, BaseField::one());
    }
    bit_reverse_coset_to_circle_domain_order(is_active_col.as_mut_slice());

    let mut logup_gen = LogupTraceGenerator::new(log_size);

    {
        let mut col_gen = logup_gen.new_col();

        // For each vec_row, generate LogUp fraction (masked by is_active)
        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            let initial_state_packed: [_; N_STATE] =
                std::array::from_fn(|i| lookup_data.initial_state[i].data[vec_row]);
            let final_state_packed: [_; N_STATE] =
                std::array::from_fn(|i| lookup_data.final_state[i].data[vec_row]);

            let denom0: PackedSecureField = lookup_elements.combine(&initial_state_packed);
            let denom1: PackedSecureField = lookup_elements.combine(&final_state_packed);

            // Mask by is_active: numerator is (denom1-denom0)*is_active
            let is_active_packed: PackedSecureField = is_active_col.data[vec_row].into();
            let numerator = (denom1 - denom0) * is_active_packed;  // 0 for padding rows
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

/// Generate interaction trace for Scheduler component using LogUp
///
/// For coordination between components (if needed).
/// For now, simple LogUp with is_first selector.
pub fn gen_scheduler_interaction_trace(
    log_size: u32,
    lookup_elements: &PoseidonElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let n_rows = 1 << log_size;
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    // Create is_first selector: 1 only for row 0, 0 for rest
    let mut is_first_col = Col::<SimdBackend, BaseField>::zeros(n_rows);
    is_first_col.set(0, BaseField::one());
    bit_reverse_coset_to_circle_domain_order(is_first_col.as_mut_slice());

    {
        let mut col_gen = logup_gen.new_col();

        for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
            // Simple LogUp: -1/z for first row, 0 for rest
            let is_first_value = is_first_col.data[vec_row];
            let numerator = (-is_first_value).into();
            let denominator = lookup_elements.0.z.into();
            col_gen.write_frac(vec_row, numerator, denominator);
        }

        col_gen.finalize_col();
    }

    logup_gen.finalize_last()
}
