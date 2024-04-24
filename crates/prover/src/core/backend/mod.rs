use std::fmt::Debug;

pub use cpu::CPUBackend;
use stwo_verifier::core::fields::m31::BaseField;
use stwo_verifier::core::fields::qm31::SecureField;
use stwo_verifier::core::fields::Field;

use super::air::accumulation::AccumulationOps;
use super::commitment_scheme::quotients::QuotientOps;
use super::fri::FriOps;
use super::poly::circle::PolyOps;

#[cfg(target_arch = "x86_64")]
pub mod avx512;
pub mod cpu;

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
{
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

pub trait FieldOps<F: Field>: ColumnOps<F> {
    // TODO(Ohad): change to use a mutable slice.
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column);
}
