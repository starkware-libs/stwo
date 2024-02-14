use std::fmt::Debug;

use super::{Backend, ColumnTrait, FieldOps};
use crate::core::fields::Field;

mod poly;

#[derive(Copy, Clone, Debug)]
pub struct CPUBackend;
impl Backend for CPUBackend {}

impl<F: Field> FieldOps<F> for CPUBackend {
    type Column = Vec<F>;
}

impl<F: Clone + Debug> ColumnTrait<F> for Vec<F> {
    fn len(&self) -> usize {
        self.len()
    }
}
