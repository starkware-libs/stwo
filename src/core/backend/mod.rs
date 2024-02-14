pub mod cpu;
use std::ops::Index;

pub use cpu::CPUBackend;

use super::fields::m31::BaseField;
use super::fields::qm31::SecureField;
use super::fields::Field;
use super::poly::circle::PolyOps;

pub trait Backend:
    Copy
    + Clone
    + std::fmt::Debug
    + FieldOps<BaseField>
    + FieldOps<SecureField>
    + PolyOps<BaseField>
    + PolyOps<SecureField>
{
}

pub trait FieldOps<F: Field> {
    type Column: Clone + std::fmt::Debug + VecLike<F> + Index<usize, Output = F>;
}

pub type Column<B, F> = <B as FieldOps<F>>::Column;

pub trait VecLike<F> {
    fn from_vec(vec: Vec<F>) -> Self;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
