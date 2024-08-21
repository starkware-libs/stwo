use std::ops::{Deref, DerefMut};

use educe::Educe;

use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;

pub trait MleOps<F: Field>: ColumnOps<F> + Sized {
    /// Returns a transformed [`Mle`] where the first variable is fixed to `assignment`.
    fn fix_first_variable(mle: Mle<Self, F>, assignment: SecureField) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>;
}

/// Multilinear Extension stored as evaluations of a multilinear polynomial over the boolean
/// hypercube in bit-reversed order.
#[derive(Educe)]
#[educe(Debug, Clone)]
pub struct Mle<B: ColumnOps<F>, F: Field> {
    evals: Col<B, F>,
}

impl<B: MleOps<F>, F: Field> Mle<B, F> {
    /// Creates a [`Mle`] from evaluations of a multilinear polynomial on the boolean hypercube.
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations is not a power of two.
    pub fn new(evals: Col<B, F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self { evals }
    }

    pub fn into_evals(self) -> Col<B, F> {
        self.evals
    }

    /// Returns a transformed polynomial where the first variable is fixed to `assignment`.
    pub fn fix_first_variable(self, assignment: SecureField) -> Mle<B, SecureField>
    where
        B: MleOps<SecureField>,
    {
        B::fix_first_variable(self, assignment)
    }

    /// Returns the number of variables in the polynomial.
    pub fn n_variables(&self) -> usize {
        self.evals.len().ilog2() as usize
    }
}

impl<B: ColumnOps<F>, F: Field> Deref for Mle<B, F> {
    type Target = Col<B, F>;

    fn deref(&self) -> &Col<B, F> {
        &self.evals
    }
}

impl<B: ColumnOps<F>, F: Field> DerefMut for Mle<B, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evals
    }
}

#[cfg(test)]
mod test {
    use super::{Mle, MleOps};
    use crate::core::backend::Column;
    use crate::core::fields::qm31::SecureField;
    use crate::core::fields::{ExtensionOf, Field};

    impl<B, F> Mle<B, F>
    where
        F: Field,
        SecureField: ExtensionOf<F>,
        B: MleOps<F>,
    {
        /// Evaluates the multilinear polynomial at `point`.
        pub(crate) fn eval_at_point(&self, point: &[SecureField]) -> SecureField {
            pub fn eval(mle_evals: &[SecureField], p: &[SecureField]) -> SecureField {
                match p {
                    [] => mle_evals[0],
                    &[p_i, ref p @ ..] => {
                        let (lhs, rhs) = mle_evals.split_at(mle_evals.len() / 2);
                        let lhs_eval = eval(lhs, p);
                        let rhs_eval = eval(rhs, p);
                        // Equivalent to `eq(0, p_i) * lhs_eval + eq(1, p_i) * rhs_eval`.
                        p_i * (rhs_eval - lhs_eval) + lhs_eval
                    }
                }
            }

            let mle_evals = self
                .clone()
                .into_evals()
                .to_cpu()
                .into_iter()
                .map(|v| v.into())
                .collect::<Vec<_>>();

            eval(&mle_evals, point)
        }
    }
}
