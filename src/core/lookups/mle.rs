use std::fmt::Debug;
use std::ops::Deref;

use crate::core::fields::qm31::SecureField;
use crate::core::fields::Field;

pub trait ColumnOpsV2<F: Field> {
    type Column: ColumnV2<F>;
}

#[allow(clippy::len_without_is_empty)]
pub trait ColumnV2<T>: Debug + Clone + 'static + FromIterator<T> {
    fn to_vec(&self) -> Vec<T>;

    fn len(&self) -> usize;
}

pub type ColV2<B, F> = <B as ColumnOpsV2<F>>::Column;

pub trait MleOps<F: Field>: ColumnOpsV2<F> + Sized {
    fn eval_at_point(mle: &Mle<Self, F>, point: &[F]) -> F;

    fn fix_first(mle: Mle<Self, F>, assignment: SecureField) -> Mle<Self, SecureField>
    where
        Self: MleOps<SecureField>;
}

/// Multilinear extension stored as evaluations of a multilinear polynomial over the boolean
/// hypercube.
pub struct Mle<B: ColumnOpsV2<F>, F: Field> {
    evals: ColV2<B, F>,
}

impl<B: ColumnOpsV2<F>, F: Field> Clone for Mle<B, F> {
    fn clone(&self) -> Self {
        Self {
            evals: self.evals.clone(),
        }
    }
}

impl<B: ColumnOpsV2<F>, F: Field> Debug for Mle<B, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mle").field("evals", &self.evals).finish()
    }
}

impl<B: MleOps<F>, F: Field> Mle<B, F> {
    pub fn new(evals: ColV2<B, F>) -> Self {
        assert!(evals.len().is_power_of_two());
        Self { evals }
    }

    pub fn eval_at_point(&self, point: &[F]) -> F {
        assert_eq!(self.evals.len(), 1 << point.len());
        B::eval_at_point(self, point)
    }

    pub fn into_evals(self) -> ColV2<B, F> {
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

impl<B: ColumnOpsV2<F>, F: Field> Deref for Mle<B, F> {
    type Target = ColV2<B, F>;

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

/// Rectangular trace where columns are [`Mle`]s
pub struct MleTrace<B: ColumnOpsV2<F>, F: Field> {
    columns: Vec<Mle<B, F>>,
}

impl<B: ColumnOpsV2<F>, F: Field> Clone for MleTrace<B, F> {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.clone(),
        }
    }
}

impl<B: ColumnOpsV2<F>, F: Field> Debug for MleTrace<B, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MleTrace")
            .field("columns", &self.columns)
            .finish()
    }
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

impl<B: ColumnOpsV2<F>, F: Field> Deref for MleTrace<B, F> {
    type Target = [Mle<B, F>];

    fn deref(&self) -> &Self::Target {
        &self.columns
    }
}
