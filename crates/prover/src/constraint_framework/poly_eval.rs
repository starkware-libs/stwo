use super::poly::Polynomial;
use super::EvalAtRow;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;

pub struct PolyEvaluator<'a> {}

impl<'a> EvalAtRow for PolyEvaluator<'a> {
    type F = Polynomial<BaseField>;
    type EF = Polynomial<SecureField>;
    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        unimplemented!()
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF>,
    {
        unimplemented!()
    }
    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        unimplemented!()
    }
}
