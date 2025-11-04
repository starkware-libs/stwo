use num_traits::One;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, ORIGINAL_TRACE_IDX};

use super::{PoseidonElements, LOG_CONSTRAINT_DEGREE};

/// Scheduler component for Poseidon
///
/// This component manages which rows are active and coordinates between
/// multiple Poseidon computing instances if needed.
///
/// Trace columns (ORIGINAL_TRACE_IDX):
/// - Column 0: n_active_messages (constant across all rows)
///
/// Constraints:
/// 1. n_active_messages is constant across rows (simple transition constraint)
///
/// For now, this is a simple component. Could be extended to:
/// - Coordinate between multiple computing components
/// - Verify total number of active rows
/// - LogUp coordination
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

        // Read n_messages column (should be constant)
        let [n_messages_curr, n_messages_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);

        // Constraint: n_messages is constant across rows
        // (disabled for first row)
        let not_first = E::F::one() - is_first.clone();
        eval.add_constraint(
            not_first * (n_messages_curr.clone() - n_messages_prev)
        );

        // No LogUp for scheduler component (for now)
        // If we add LogUp coordination later, uncomment:
        // eval.finalize_logup_in_pairs();

        eval
    }
}

pub type PoseidonSchedulerComponent = FrameworkComponent<PoseidonSchedulerEval>;
