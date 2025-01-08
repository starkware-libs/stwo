use std::fmt::Debug;

pub use cpu::CpuBackend;

use super::air::accumulation::AccumulationOps;
use super::channel::MerkleChannel;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fri::FriOps;
use super::lookups::gkr_prover::GkrOps;
use super::pcs::quotients::QuotientOps;
use super::poly::circle::PolyOps;
use super::proof_of_work::GrindOps;
use super::vcs::ops::MerkleOps;

pub mod cpu;
pub mod simd;

pub trait Backend:
    Copy
    + Clone
    + Debug
    + ColumnOps<BaseField>
    + ColumnOps<SecureField>
    + PolyOps
    + QuotientOps
    + FriOps
    + AccumulationOps
    + GkrOps
{
}

pub trait BackendForChannel<MC: MerkleChannel>:
    Backend + MerkleOps<MC::H> + GrindOps<MC::C>
{
}

pub trait ColumnOps<T> {
    type Column: Column<T>;
    fn bit_reverse_column(column: &mut Self::Column);
}

pub type Col<B, T> = <B as ColumnOps<T>>::Column;

// TODO(alont): Consider removing the generic parameter and only support BaseField.
pub trait Column<T>: Clone + Debug + FromIterator<T> {
    /// Creates a new column of zeros with the given length.
    fn zeros(len: usize) -> Self;
    /// Creates a new column of uninitialized values with the given length.
    /// # Safety
    /// The caller must ensure that the column is populated before being used.
    unsafe fn uninitialized(len: usize) -> Self;
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
    /// Sets the element at the given index.
    fn set(&mut self, index: usize, value: T);
}
