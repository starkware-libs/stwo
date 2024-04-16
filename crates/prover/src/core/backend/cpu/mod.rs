mod accumulation;
mod blake2s;
mod circle;
mod fri;
mod lookups;
pub mod quotients;

use std::fmt::Debug;

use super::{Backend, Column, ColumnOps, FieldOps};
use crate::core::fields::Field;
use crate::core::lookups::mle::Mle;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::utils::bit_reverse;

#[derive(Copy, Clone, Debug)]
pub struct CPUBackend;

impl Backend for CPUBackend {}

impl<T: Debug + Clone + Default> ColumnOps<T> for CPUBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        bit_reverse(column)
    }
}

impl<F: Field> FieldOps<F> for CPUBackend {
    /// Batch inversion using the Montgomery's trick.
    // TODO(Ohad): Benchmark this function.
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        F::batch_inverse(column, &mut dst[..]);
    }
}

impl<T: Debug + Clone + Default> Column<T> for Vec<T> {
    fn zeros(len: usize) -> Self {
        vec![T::default(); len]
    }
    fn to_cpu(&self) -> Vec<T> {
        self.clone()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn at(&self, index: usize) -> T {
        self[index].clone()
    }
}

pub type CPUCirclePoly = CirclePoly<CPUBackend>;
pub type CPUCircleEvaluation<F, EvalOrder> = CircleEvaluation<CPUBackend, F, EvalOrder>;
pub type CPUMle<F> = Mle<CPUBackend, F>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use crate::core::backend::{CPUBackend, Column, FieldOps};
    use crate::core::fields::m31::P;
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::FieldExpOps;

    #[test]
    fn batch_inverse_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let column: Vec<QM31> = (0..16)
            .map(|_| {
                QM31::from_u32_unchecked(
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
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
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                    rng.gen::<u32>() % P,
                )
            })
            .collect();
        let expected = column.iter().map(|e| e.inverse()).collect_vec();
        let mut dst = Column::zeros(column.len());

        CPUBackend::batch_inverse(&column, &mut dst);

        assert_eq!(expected, dst);
    }
}
