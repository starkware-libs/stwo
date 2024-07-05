use std::ops::Mul;

use num_traits::One;

use super::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;

#[derive(Default)]
pub struct InfoEvaluator {
    pub mask_offsets: TreeVec<Vec<Vec<isize>>>,
    pub n_constraints: usize,
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
        if self.mask_offsets.len() <= interaction {
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
}
