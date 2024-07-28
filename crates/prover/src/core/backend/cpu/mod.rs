mod accumulation;
mod blake2s;
mod circle;
mod fri;
pub mod lookups;
pub mod quotients;

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::{Backend, Column, ColumnOps, FieldOps};
use crate::core::fields::Field;
use crate::core::lookups::mle::Mle;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::utils::bit_reverse;

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct CpuBackend;

impl Backend for CpuBackend {}

impl<T: Debug + Clone + Default> ColumnOps<T> for CpuBackend {
    type Column = Vec<T>;

    fn bit_reverse_column(column: &mut Self::Column) {
        bit_reverse(column)
    }
}

impl<F: Field> FieldOps<F> for CpuBackend {
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
    #[allow(clippy::uninit_vec)]
    unsafe fn uninitialized(length: usize) -> Self {
        let mut data = Vec::with_capacity(length);
        data.set_len(length);
        data
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
    fn set(&mut self, index: usize, value: T) {
        self[index] = value;
    }
}

pub type CpuCirclePoly = CirclePoly<CpuBackend>;
pub type CpuCircleEvaluation<F, EvalOrder> = CircleEvaluation<CpuBackend, F, EvalOrder>;
pub type CpuMle<F> = Mle<CpuBackend, F>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use crate::core::backend::{Column, CpuBackend, FieldOps};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::FieldExpOps;

    #[test]
    fn batch_inverse_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let column = rng.gen::<[QM31; 16]>().to_vec();
        let expected = column.iter().map(|e| e.inverse()).collect_vec();
        let mut dst = Column::zeros(column.len());

        CpuBackend::batch_inverse(&column, &mut dst);

        assert_eq!(expected, dst);
    }
}
