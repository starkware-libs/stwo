mod poly;

use super::{Backend, FieldOps, VecLike};
use crate::core::fields::Field;

#[derive(Copy, Clone, Debug)]
pub struct CPUBackend;
impl Backend for CPUBackend {}

impl<F: Field> FieldOps<F> for CPUBackend {
    type Column = Vec<F>;
}

impl<F> VecLike<F> for Vec<F> {
    fn from_vec(vec: Vec<F>) -> Self {
        vec
    }
    fn len(&self) -> usize {
        self.len()
    }
}
