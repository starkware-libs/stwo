use std::ops::Deref;

use derivative::Derivative;
use stwo_verifier::core::fields::qm31::SecureField;
use stwo_verifier::core::fields::Field;

use crate::core::backend::{Col, Column, ColumnOps};

/// TODO
pub trait MleOps<F: Field>: ColumnOps<F> + Sized {
    /// Returns a transformed [`Mle`] where the first variable is fixed to `assignment`.
    fn fix_first(mle: Mle<Self, F>, assignment: SecureField) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>;
}

/// Multilinear Extension stored as evaluations of a multilinear polynomial over the boolean
/// hypercube in bit-reversed order.
#[derive(Derivative)]
#[derivative(Debug(bound = ""), Clone(bound = ""))]
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
    pub fn fix_first(self, assignment: SecureField) -> Mle<B, SecureField>
    where
        B: MleOps<SecureField>,
    {
        B::fix_first(self, assignment)
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
