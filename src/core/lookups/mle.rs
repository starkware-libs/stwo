use std::ops::Deref;

use derivative::Derivative;

use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;

/// TODO
pub trait MleOps<F: Field>: ColumnOps<F> + Sized {
    /// Evaluates `mle` at `point`.
    fn eval_at_point(mle: &Mle<Self, F>, point: &[F]) -> F;

    /// Returns a transformed [`Mle`] where the first variable is fixed to `assignment`.
    fn fix_first(mle: Mle<Self, F>, assignment: SecureField) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>;
}

/// Multilinear Extension stored as evaluations of a multilinear polynomial over the boolean
/// hypercube.
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

    /// Evaluates the multilinear polynomial at `point`.
    ///
    /// # Panics
    ///
    /// Panics if the the length of point does not match the number variables in the polynomial.
    pub fn eval_at_point(&self, point: &[F]) -> F {
        assert_eq!(self.evals.len(), 1 << point.len());
        B::eval_at_point(self, point)
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
    pub fn num_variables(&self) -> usize {
        self.evals.len().ilog2() as usize
    }
}

impl<B: ColumnOps<F>, F: Field> Deref for Mle<B, F> {
    type Target = Col<B, F>;

    fn deref(&self) -> &Col<B, F> {
        &self.evals
    }
}
