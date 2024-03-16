use std::ops::Deref;

use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Field, FieldOps};

pub trait MleOps<F: Field>: FieldOps<F> + Sized {
    fn eval_at_point(mle: &Mle<Self, F>, point: &[F]) -> F;

    fn fix_first(mle: Mle<Self, F>, assignment: SecureField) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>;
}

/// Multi-Linear extension with values represented in the lagrange basis.
// TODO: Fix docs
#[derive(Debug, Clone)]
pub struct Mle<B: ColumnOps<F>, F> {
    evals: Col<B, F>,
}

impl<B: MleOps<F>, F: Field> Mle<B, F> {
    pub fn new(evals: Col<B, F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self { evals }
    }

    pub fn eval_at_point(&self, point: &[F]) -> F {
        B::eval_at_point(self, point)
    }

    pub fn into_evals(self) -> Col<B, F> {
        self.evals
    }

    pub fn fix_first(self, assignment: SecureField) -> Mle<B, SecureField>
    where
        B: MleOps<SecureField>,
    {
        B::fix_first(self, assignment)
    }

    pub fn num_variables(&self) -> usize {
        self.evals.len().ilog2() as usize
    }
}

impl<B: MleOps<F>, F: Field> Deref for Mle<B, F> {
    type Target = Col<B, F>;

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

/// Rectangular trace where columns are [`Mle`]s
#[derive(Debug, Clone)]
pub struct MleTrace<B: ColumnOps<F>, F> {
    columns: Vec<Mle<B, F>>,
}

impl<B: MleOps<F>, F: Field> MleTrace<B, F> {
    /// # Panics
    ///
    /// Panics if columns is empty or if the columns aren't all the same size.
    pub fn new(columns: Vec<Mle<B, F>>) -> Self {
        let num_variables = columns[0].num_variables();
        assert!(columns.iter().all(|c| c.num_variables() == num_variables));
        Self { columns }
    }

    /// Returns the number of variables that all columns have.
    pub fn num_variables(&self) -> usize {
        self.columns[0].num_variables()
    }

    /// Evaluates each column of the trace at `point`.
    pub fn eval_at_point(&self, point: &[F]) -> Vec<F> {
        self.columns
            .iter()
            .map(|c| c.eval_at_point(point))
            .collect()
    }

    pub fn into_columns(self) -> Vec<Mle<B, F>> {
        self.columns
    }
}

impl<B: ColumnOps<F>, F: Field> Deref for MleTrace<B, F> {
    type Target = [Mle<B, F>];

    fn deref(&self) -> &Self::Target {
        &self.columns
    }
}
