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
    type Column: ColumnTrait<F>;
    fn bit_reverse_column(column: Self::Column) -> Self::Column;
}

pub type Column<B, F> = <B as FieldOps<F>>::Column;

pub trait ColumnTrait<F>: Clone + Debug + Index<usize, Output = F> + FromIterator<F> {
    fn zeros(len: usize) -> Self;
    fn to_vec(&self) -> Vec<F>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
