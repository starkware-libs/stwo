mod circle;
mod fri;

use std::fmt::Debug;

use super::{Backend, FieldOps};
use crate::core::fields::{Column, Field};
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::poly::line::LineEvaluation;
use crate::core::utils::bit_reverse;

#[derive(Copy, Clone, Debug)]
pub struct CPUBackend;

impl Backend for CPUBackend {}

impl<F: Field> FieldOps<F> for CPUBackend {
    type Column = Vec<F>;

    fn bit_reverse_column(column: &mut Self::Column) {
        bit_reverse(column)
    }

    /// Batch inversion using the Montgomery's trick.
    // TODO(Ohad): Benchmark this function.
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        F::batch_inverse(column, &mut dst[..]);
    }
}

impl<F: Field> Column<F> for Vec<F> {
    fn zeros(len: usize) -> Self {
        vec![F::zero(); len]
    }
    fn to_vec(&self) -> Vec<F> {
        self.clone()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn at(&self, index: usize) -> F {
        self[index]
    }
}

pub type CPUCirclePoly<F> = CirclePoly<CPUBackend, F>;
pub type CPUCircleEvaluation<F, EvalOrder> = CircleEvaluation<CPUBackend, F, EvalOrder>;
// TODO(spapini): Remove the EvalOrder on LineEvaluation.
pub type CPULineEvaluation<F, EvalOrder> = LineEvaluation<CPUBackend, F, EvalOrder>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use crate::core::backend::{CPUBackend, FieldOps};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::{Column, FieldExpOps};

    #[test]
    fn batch_inverse_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let column: Vec<QM31> = (0..16)
            .map(|_| {
                QM31::from_u32_unchecked(
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                )
            })
            .collect();
        let expected = column.iter().map(|e| e.inverse()).collect_vec();
        let mut dst = Column::zeros(column.len());

        CPUBackend::batch_inverse(&column, &mut dst);

        assert_eq!(expected, dst);
    }

    // TODO(Ohad): remove this test.
    #[test]
    fn batch_inverse_reused_vec_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let column: Vec<QM31> = (0..16)
            .map(|_| {
                QM31::from_u32_unchecked(
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                )
            })
            .collect();
        let expected = column.iter().map(|e| e.inverse()).collect_vec();
        let mut dst = Column::zeros(column.len());

        CPUBackend::batch_inverse(&column, &mut dst);

        assert_eq!(expected, dst);
    }
}
