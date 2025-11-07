use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry, ORIGINAL_TRACE_IDX};

use super::{PoseidonRelation, LOG_CONSTRAINT_DEGREE};
use crate::{N_STATE, RATE};

#[derive(Clone)]
pub struct PoseidonComputingEval {
    pub log_n_rows: u32,
    pub initial_message: [BaseField; RATE],
    pub poseidon_relation: PoseidonRelation,
    pub claimed_sum: SecureField,
    pub is_first_id: PreProcessedColumnId,
    pub is_active_id: PreProcessedColumnId,  // 1 for rows 0..=target_element, 0 for rest
    pub is_target_id: PreProcessedColumnId,  // 1 only for target_element (for LogUp)
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
        let is_target = eval.get_preprocessed_column(self.is_target_id.clone());

        // Read trace columns: message (8) + state_in (16) + state_out (16) = 40 columns

        // Message (8 elements)
        let message: [E::F; RATE] = std::array::from_fn(|_| {
            let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            curr
        });

        // State input (16 elements) - current only
        let state_in_curr: [E::F; N_STATE] = std::array::from_fn(|_| {
            let [curr, _prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            curr
        });

        // State output (16 elements) - need prev for transition
        let mut state_out_curr_vec = Vec::with_capacity(N_STATE);
        let mut state_out_prev_vec = Vec::with_capacity(N_STATE);
        for _ in 0..N_STATE {
            let [curr, prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
            state_out_curr_vec.push(curr);
            state_out_prev_vec.push(prev);
        }
        let state_out_curr: [E::F; N_STATE] = std::array::from_fn(|i| state_out_curr_vec[i].clone());
        let state_out_prev: [E::F; N_STATE] = std::array::from_fn(|i| state_out_prev_vec[i].clone());

        // Constraint 1: First row - capacity must be zero
        // Only enforced on first row (which is always active)
        for i in RATE..N_STATE {
            eval.add_constraint(is_first.clone() * state_in_curr[i].clone());
        }

        // Constraint 2: First row - rate part must match initial message
        // Only enforced on first row (which is always active)
        for i in 0..RATE {
            eval.add_constraint(
                is_first.clone() * (state_in_curr[i].clone() - E::F::from(self.initial_message[i]))
            );
        }

        // Constraint 3: Poseidon permutation - state_out = Poseidon(state_in)
        // For simplicity, using identity (will be replaced with real Poseidon)
        // Multiply by is_active - only works for rows 0..=target_element
        for i in 0..N_STATE {
            eval.add_constraint(
                is_active.clone() * (state_out_curr[i].clone() - state_in_curr[i].clone())
            );
        }

        // Constraint 4: Transition constraints (chaining between rows)
        // Disabled for: first row OR inactive rows
        let not_first = E::F::one() - is_first.clone();
        let enable_chaining = is_active.clone() * not_first;

        // Rate part: state_in[0..8] = state_out_prev[0..8] + message[0..8]
        for i in 0..RATE {
            let expected = state_out_prev[i].clone() + message[i].clone();
            eval.add_constraint(
                enable_chaining.clone() * (state_in_curr[i].clone() - expected)
            );
        }

        // Capacity part: state_in[8..16] = state_out_prev[8..16]
        for i in RATE..N_STATE {
            eval.add_constraint(
                enable_chaining.clone() * (state_in_curr[i].clone() - state_out_prev[i].clone())
            );
        }

        // LogUp: yield ONLY for target_element row
        eval.add_to_relation(RelationEntry::new(
            &self.poseidon_relation,
            is_target.into(),    // multiplicity: 1 only for target_element, 0 for rest
            &state_out_curr,     // yield 16-element state
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type PoseidonComputingComponent = FrameworkComponent<PoseidonComputingEval>;
