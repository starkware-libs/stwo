mod circle;
mod line;

use super::{Backend, ColumnTrait, FieldOps};
use crate::core::fields::Field;
use crate::core::utils::bit_reverse;

#[derive(Copy, Clone, Debug)]
pub struct CPUBackend;
impl Backend for CPUBackend {}

impl<F: Field> FieldOps<F> for CPUBackend {
    type Column = Vec<F>;

    fn bit_reverse_column(column: Self::Column) -> Self::Column {
        bit_reverse(column)
    }
}

impl<F: Field> ColumnTrait<F> for Vec<F> {
    fn zeros(len: usize) -> Self {
        vec![F::zero(); len]
    }
    fn from_vec(vec: Vec<F>) -> Self {
        vec
    }
    fn to_vec(&self) -> Vec<F> {
        self.clone()
    }
    fn len(&self) -> usize {
        self.len()
    }
}
