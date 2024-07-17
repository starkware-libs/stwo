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

pub trait BufferOps<T> {
    type Buffer: Buffer<T>;
    fn bit_reverse_column(column: &mut Self::Buffer);
}

pub type Buf<B, T> = <B as BufferOps<T>>::Buffer;

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
