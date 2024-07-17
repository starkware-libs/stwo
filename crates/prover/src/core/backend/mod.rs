use std::fmt::Debug;

pub use cpu::CpuBackend;

use super::air::accumulation::AccumulationOps;
use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::FieldOps;
use super::fri::FriOps;
use super::pcs::quotients::QuotientOps;
use super::poly::circle::PolyOps;

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
{
}

pub trait ColumnOps<T> {
    type Column: Buffer<T>;
    fn bit_reverse_column(column: &mut Self::Column);
}

pub type Col<B, T> = <B as ColumnOps<T>>::Column;

/// Given a type T, a buffer of Ts laid out in some canonical way.
pub trait Buffer<T>: Clone + Debug + FromIterator<T> {
    /// Creates a new buffer of zeros with the given length.
    fn zeros(len: usize) -> Self;
    /// Returns a cpu vector of the buffer.
    fn to_cpu(&self) -> Vec<T>;
    /// Returns the length of the buffer.
    fn len(&self) -> usize;
    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Retrieves the element at the given index.
    fn at(&self, index: usize) -> T;
}
