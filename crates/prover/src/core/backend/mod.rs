use std::fmt::Debug;

pub use cpu::CpuBackend;

use super::air::accumulation::AccumulationOps;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::FieldOps;
use super::fri::FriOps;
use super::lookups::gkr_prover::GkrOps;
use super::pcs::quotients::QuotientOps;
use super::poly::circle::{CircleEvaluation, PolyOps};
use super::poly::BitReversedOrder;
use super::ColumnVec;
use crate::examples::xor::multilinear_eval_at_point::BatchMultilinearEvalIopProver;

pub mod cpu;
pub mod simd;

pub trait Backend:
    Copy
    + Clone
    + Debug
    + FieldOps<BaseField>
    + FieldOps<SecureField>
    + PolyOps
    + QuotientOps
    + FriOps
    + AccumulationOps
    + GkrOps
    + MultilinearEvalAtPointIopOps
{
}

// TODO: Remove. This is just added to get something working.
pub trait MultilinearEvalAtPointIopOps:
    GkrOps + FieldOps<SecureField> + FieldOps<BaseField> + Sized
{
    fn random_linear_combination(
        _columns: Vec<Col<Self, SecureField>>,
        _random_coeff: SecureField,
    ) -> Col<Self, SecureField> {
        todo!()
    }

    fn write_interaction_trace(
        _prover: &BatchMultilinearEvalIopProver<Self>,
    ) -> ColumnVec<CircleEvaluation<Self, BaseField, BitReversedOrder>> {
        todo!()
    }
}

pub trait ColumnOps<T> {
    type Column: Column<T>;
    fn bit_reverse_column(column: &mut Self::Column);
}

pub type Col<B, T> = <B as ColumnOps<T>>::Column;

// TODO(spapini): Consider removing the generic parameter and only support BaseField.
pub trait Column<T>: Clone + Debug + FromIterator<T> {
    /// Creates a new column of zeros with the given length.
    fn zeros(len: usize) -> Self;
    /// Returns a cpu vector of the column.
    fn to_cpu(&self) -> Vec<T>;
    /// Returns the length of the column.
    fn len(&self) -> usize;
    /// Returns true if the column is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Retrieves the element at the given index.
    fn at(&self, index: usize) -> T;
}
