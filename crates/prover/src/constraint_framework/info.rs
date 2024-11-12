use std::ops::Mul;

use num_traits::One;

use super::preprocessed_columns::PreprocessedColumn;
use super::EvalAtRow;
use crate::constraint_framework::PREPROCESSED_TRACE_IDX;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;

/// Collects information about the constraints.
/// This includes mask offsets and columns at each interaction, and the number of constraints.
#[derive(Default)]
pub struct InfoEvaluator {
    pub mask_offsets: TreeVec<Vec<Vec<isize>>>,
    pub n_constraints: usize,
    pub preprocessed_columns: Vec<PreprocessedColumn>,
}
impl InfoEvaluator {
    pub fn new() -> Self {
        Self::default()
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
        assert!(
            interaction != PREPROCESSED_TRACE_IDX,
            "Preprocessed should be accesses with `next_preprocessed_mask`",
        );

        // Check if requested a mask from a new interaction
        if self.mask_offsets.len() <= interaction {
            // Extend `mask_offsets` so that `interaction` is the last index.
            self.mask_offsets.resize(interaction + 1, vec![]);
        }
        self.mask_offsets[interaction].push(offsets.into_iter().collect());
        [BaseField::one(); N]
    }

    fn next_preprocessed_mask<const N: usize>(
        &mut self,
        column: PreprocessedColumn,
        _offsets: [isize; N],
    ) -> [Self::F; N] {
        self.preprocessed_columns.push(column);
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
}
