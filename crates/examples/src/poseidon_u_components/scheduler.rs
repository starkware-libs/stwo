use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

use super::{PoseidonElements, LOG_CONSTRAINT_DEGREE};

/// Scheduler component for Poseidon
///
/// This component manages which rows are active and coordinates between
/// multiple Poseidon computing instances if needed.
///
/// Trace columns (ORIGINAL_TRACE_IDX):
/// - Column 0: n_active_messages (constant across all rows)
/// - Column 1: row_index (increments: 0, 1, 2, ...)
/// - Column 2: is_active_computed (1 if row_index < n_active_messages, else 0)
///
/// Constraints:
/// 1. n_active_messages is constant across rows
/// 2. row_index increments by 1 each row (except first row)
/// 3. is_active_computed = (row_index < n_active_messages)
/// 4. LogUp coordination (if needed)
#[derive(Clone)]
pub struct PoseidonSchedulerEval {
    pub log_n_rows: u32,
    pub lookup_elements: PoseidonElements,
    pub claimed_sum: SecureField,
    pub is_first_id: PreProcessedColumnId,
    pub n_messages: usize,
}

impl FrameworkEval for PoseidonSchedulerEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_first = eval.get_preprocessed_column(self.is_first_id.clone());

        // TODO: Implement scheduler constraints
        // This will manage:
        // 1. Tracking number of active messages
        // 2. Coordinating which rows are active
        // 3. LogUp coordination between computing components

        // For now, placeholder constraint
        eval.add_constraint(is_first.clone() - is_first.clone());

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type PoseidonSchedulerComponent = FrameworkComponent<PoseidonSchedulerEval>;
