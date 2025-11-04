use num_traits::One;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry, ORIGINAL_TRACE_IDX,
};

use super::{
    apply_external_round_matrix, apply_internal_round_matrix, PoseidonElements,
    EXTERNAL_ROUND_CONSTS, INTERNAL_ROUND_CONSTS, LOG_CONSTRAINT_DEGREE, N_HALF_FULL_ROUNDS,
    N_PARTIAL_ROUNDS, N_STATE, RATE,
};

/// Computing component for Poseidon hash
///
/// This component computes the actual Poseidon permutation for active rows.
/// Inactive (padding) rows are disabled via the is_active selector.
///
/// Trace columns (ORIGINAL_TRACE_IDX):
/// - Columns 0-7: message (8 field elements - RATE)
/// - Columns 8-23: initial_state (16 field elements - N_STATE)
/// - Columns 24+: intermediate states during permutation
/// - Last 16 columns: final_state (16 field elements - N_STATE)
///
/// Key optimization: ALL constraints are multiplied by is_active!
/// This disables constraints for padding rows, allowing zeros.
#[derive(Clone)]
pub struct PoseidonComputingEval {
    pub log_n_rows: u32,
    pub lookup_elements: PoseidonElements,
    pub claimed_sum: SecureField,
    pub is_first_id: PreProcessedColumnId,
    pub is_active_id: PreProcessedColumnId, // 1 for active rows (with messages), 0 for padding
}

impl FrameworkEval for PoseidonComputingEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_first = eval.get_preprocessed_column(self.is_first_id.clone());
        let is_active = eval.get_preprocessed_column(self.is_active_id.clone());

        // Read ALL columns using next_interaction_mask
        // Column layout: [message(8), initial_state(16), intermediate_states, final_state(16)]

        // Read message (8 elements)
        let message: [E::F; RATE] = std::array::from_fn(|_| {
            let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            curr
        });

        // Read initial state (16 elements)
        let initial_state_curr: [E::F; N_STATE] = std::array::from_fn(|_| {
            let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            curr
        });

        // Read intermediate states from first 4 full rounds
        let intermediate_full1: [[E::F; N_STATE]; N_HALF_FULL_ROUNDS] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
                curr
            })
        });

        // Read partial round intermediate states
        let intermediate_partial: [E::F; N_PARTIAL_ROUNDS] = std::array::from_fn(|_| {
            let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            curr
        });

        // Read intermediate states from last 4 full rounds
        let intermediate_full2: [[E::F; N_STATE]; N_HALF_FULL_ROUNDS] = std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
                curr
            })
        });

        // Read final state (16 elements) - current and PREVIOUS row
        let mut final_state_curr_vec = Vec::with_capacity(N_STATE);
        let mut final_state_prev_vec = Vec::with_capacity(N_STATE);
        for _ in 0..N_STATE {
            let [curr, prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            final_state_curr_vec.push(curr);
            final_state_prev_vec.push(prev);
        }
        let final_state_curr: [E::F; N_STATE] =
            std::array::from_fn(|i| final_state_curr_vec[i].clone());
        let final_state_prev: [E::F; N_STATE] =
            std::array::from_fn(|i| final_state_prev_vec[i].clone());

        // ===== CONSTRAINTS (all multiplied by is_active!) =====

        // Constraint 1: First row capacity must be zero (only for active rows)
        for i in RATE..N_STATE {
            eval.add_constraint(
                is_active.clone() * is_first.clone() * initial_state_curr[i].clone(),
            );
        }

        // Constraint 2: Transition constraints (chaining between rows)
        // Only enforced for active rows that are not first row
        let not_first = E::F::one() - is_first.clone();

        // Rate part: initial_state[0..8] = final_state_prev[0..8] + message[0..8]
        for i in 0..RATE {
            let expected = final_state_prev[i].clone() + message[i].clone();
            eval.add_constraint(
                is_active.clone() * not_first.clone() * (initial_state_curr[i].clone() - expected),
            );
        }

        // Capacity part: initial_state[8..16] = final_state_prev[8..16]
        for i in RATE..N_STATE {
            eval.add_constraint(
                is_active.clone()
                    * not_first.clone()
                    * (initial_state_curr[i].clone() - final_state_prev[i].clone()),
            );
        }

        // Constraint 3: Poseidon permutation correctness (only for active rows)
        let mut state = initial_state_curr.clone();

        // 4 full rounds
        for round in 0..N_HALF_FULL_ROUNDS {
            for i in 0..N_STATE {
                state[i] = state[i].clone() + E::F::from(EXTERNAL_ROUND_CONSTS[round][i]);
            }
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5_expr(state[i].clone()));

            // Verify intermediate state matches trace (masked by is_active)
            for i in 0..N_STATE {
                eval.add_constraint(
                    is_active.clone() * (state[i].clone() - intermediate_full1[round][i].clone()),
                );
            }
            state = intermediate_full1[round].clone();
        }

        // Partial rounds
        for round in 0..N_PARTIAL_ROUNDS {
            state[0] = state[0].clone() + E::F::from(INTERNAL_ROUND_CONSTS[round]);
            apply_internal_round_matrix(&mut state);
            state[0] = pow5_expr(state[0].clone());

            // Verify intermediate state matches trace (masked by is_active)
            eval.add_constraint(
                is_active.clone() * (state[0].clone() - intermediate_partial[round].clone()),
            );
            state[0] = intermediate_partial[round].clone();
        }

        // Last 4 full rounds
        for round in 0..N_HALF_FULL_ROUNDS {
            for i in 0..N_STATE {
                state[i] = state[i].clone()
                    + E::F::from(EXTERNAL_ROUND_CONSTS[round + N_HALF_FULL_ROUNDS][i]);
            }
            apply_external_round_matrix(&mut state);
            state = std::array::from_fn(|i| pow5_expr(state[i].clone()));

            // Verify intermediate state matches trace (masked by is_active)
            for i in 0..N_STATE {
                eval.add_constraint(
                    is_active.clone() * (state[i].clone() - intermediate_full2[round][i].clone()),
                );
            }
            state = intermediate_full2[round].clone();
        }

        // Verify final state matches computed state (masked by is_active)
        for i in 0..N_STATE {
            eval.add_constraint(
                is_active.clone() * (state[i].clone() - final_state_curr[i].clone()),
            );
        }

        // LogUp: Provide initial and final state lookups (masked by is_active)
        // This ensures only active rows contribute to LogUp
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            is_active.clone().into(), // multiplicity = is_active (1 for active, 0 for padding)
            &initial_state_curr,
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            (-is_active.clone()).into(), // multiplicity = -is_active
            &final_state_curr,
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

/// Helper function to compute x^5 for constraint expressions
fn pow5_expr<F: Clone + std::ops::Mul<Output = F>>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x
}

pub type PoseidonComputingComponent = FrameworkComponent<PoseidonComputingEval>;
