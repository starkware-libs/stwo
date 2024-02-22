mod circle;
mod line;

use std::fmt::Debug;

use super::{Backend, Column, FieldOps};
use crate::core::fields::Field;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::poly::line::{LineEvaluation, LinePoly};
use crate::core::poly::NaturalOrder;
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

impl<F: Clone + Debug + Field> Column<F> for Vec<F> {
    fn zeros(len: usize) -> Self {
        vec![F::zero(); len]
    }
    fn to_vec(&self) -> Vec<F> {
        self.clone()
    }
    fn len(&self) -> usize {
        self.len()
    }
}

pub type CPUCirclePoly<F> = CirclePoly<CPUBackend, F>;
pub type CPUCircleEvaluation<F, EvalOrder = NaturalOrder> =
    CircleEvaluation<CPUBackend, F, EvalOrder>;
pub type CPULinePoly<F> = LinePoly<CPUBackend, F>;
pub type CPULineEvaluation<F, EvalOrder = NaturalOrder> = LineEvaluation<CPUBackend, F, EvalOrder>;
