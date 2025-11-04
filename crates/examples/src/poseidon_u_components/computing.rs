use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

use super::{PoseidonElements, LOG_CONSTRAINT_DEGREE};

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
#[derive(Clone)]
pub struct PoseidonComputingEval {
    pub log_n_rows: u32,
    pub lookup_elements: PoseidonElements,
    pub claimed_sum: SecureField,
    pub is_first_id: PreProcessedColumnId,
    pub is_active_id: PreProcessedColumnId,  // 1 for active rows (with messages), 0 for padding
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

        // TODO: Implement Poseidon constraints here
        // This will include:
        // 1. Sponge construction constraints (chaining between rows)
        // 2. Poseidon permutation constraints (rounds, S-boxes, MDS)
        // 3. All multiplied by is_active to disable for padding rows
        // 4. LogUp for proving correct permutation

        // For now, placeholder constraint
        eval.add_constraint(is_active.clone() * (is_first.clone() - is_first.clone()));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type PoseidonComputingComponent = FrameworkComponent<PoseidonComputingEval>;
