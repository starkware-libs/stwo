use std::ops::Mul;

use num_traits::One;

use super::logup::{LogupAtRow, LogupSums};
use super::{EvalAtRow, INTERACTION_TRACE_IDX};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::TreeVec;

/// Collects information about the constraints.
/// This includes mask offsets and columns at each interaction, and the number of constraints.
#[derive(Default)]
pub struct InfoEvaluator {
    pub mask_offsets: TreeVec<Vec<Vec<isize>>>,
    pub n_constraints: usize,
    pub logup: LogupAtRow<Self>,
}
impl InfoEvaluator {
    pub fn new(log_size: u32, logup_sums: LogupSums) -> Self {
        Self {
            mask_offsets: Default::default(),
            n_constraints: Default::default(),
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, logup_sums.0, logup_sums.1, log_size),
        }
    }

    /// Create an empty `InfoEvaluator`, to measure components before their size and logup sums are
    /// available.
    pub fn empty() -> Self {
        Self::new(16, (SecureField::default(), None))
    }
}
impl EvalAtRow for InfoEvaluator {
    type F = BaseField;
    type EF = SecureField;
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        // Check if requested a mask from a new interaction
        if self.mask_offsets.len() <= interaction {
            // Extend `mask_offsets` so that `interaction` is the last index.
            self.mask_offsets.resize(interaction + 1, vec![]);
        }
        self.mask_offsets[interaction].push(offsets.into_iter().collect());
        [BaseField::one(); N]
    }
    fn add_constraint<G>(&mut self, _constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF>,
    {
        self.n_constraints += 1;
    }

    fn combine_ef(_values: [Self::F; 4]) -> Self::EF {
        SecureField::one()
    }

    super::logup_proxy!();
}
