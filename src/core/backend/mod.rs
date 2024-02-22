use std::fmt::Debug;
use std::ops::Index;

pub use cpu::CPUBackend;

use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::Field;
use super::poly::circle::PolyOps;

pub mod cpu;

pub trait Backend:
    Copy
    + Clone
    + Debug
    + FieldOps<BaseField>
    + FieldOps<SecureField>
    + PolyOps<BaseField>
    + PolyOps<SecureField>
{
}

pub trait FieldOps<F: Field> {
    type Column: Column<F>;

    /// Batch inversion of elements.
    fn batch_inverse(column: &[F], dst: &mut [F]);
}

pub type Col<B, F> = <B as FieldOps<F>>::Column;

pub trait Column<F>: Clone + Debug + Index<usize, Output = F> + FromIterator<F> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
