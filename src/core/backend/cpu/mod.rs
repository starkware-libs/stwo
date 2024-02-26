mod circle;

use std::fmt::Debug;

use super::{Backend, FieldOps};
use crate::core::fields::{Column, Field};
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::poly::line::LineEvaluation;
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
    type Ref<'a> = &'a [F]
    where
        Self: 'a;

    type Mut<'a> = &'a mut [F]
    where
        Self: 'a;

    fn as_ref(&self) -> Self::Ref<'_> {
        self.as_slice()
    }

    fn as_mut(&mut self) -> Self::Mut<'_> {
        self.as_mut_slice()
    }

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
// TODO(spapini): Remove the EvalOrder on LineEvaluation.
pub type CPULineEvaluation<F, EvalOrder = NaturalOrder> = LineEvaluation<CPUBackend, F, EvalOrder>;
