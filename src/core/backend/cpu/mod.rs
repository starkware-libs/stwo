use std::fmt::Debug;

use super::{Backend, Column, FieldOps};
use crate::core::fields::Field;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::poly::NaturalOrder;

mod poly;

#[derive(Copy, Clone, Debug)]
pub struct CPUBackend;

impl Backend for CPUBackend {}

impl<F: Field> FieldOps<F> for CPUBackend {
    type Column = Vec<F>;

    /// Batch inversion using the Montgomery's trick.
    // TODO(Ohad): Benchmark this function.
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        let n = column.len();
        dst.clear();

        dst.push(column[0]);
        // First pass.
        for i in 1..n {
            dst.push(dst[i - 1] * column[i]);
        }

        // Inverse cumulative product.
        let mut curr_inverse = dst[n - 1].inverse();

        // Second pass.
        for i in (1..n).rev() {
            dst[i] = dst[i - 1] * curr_inverse;
            curr_inverse *= column[i];
        }
        dst[0] = curr_inverse;
    }
}

impl<F: Clone + Debug> Column<F> for Vec<F> {
    fn len(&self) -> usize {
        self.len()
    }
}

pub type CPUCirclePoly<F> = CirclePoly<CPUBackend, F>;
pub type CPUCircleEvaluation<F, EvalOrder = NaturalOrder> =
    CircleEvaluation<CPUBackend, F, EvalOrder>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use crate::core::backend::{CPUBackend, FieldOps};
    use crate::core::fields::qm31::QM31;
    use crate::core::fields::Field;

    #[test]
    fn batch_inverse_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let elements: Vec<QM31> = (0..10)
            .map(|_| {
                QM31::from_u32_unchecked(
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                )
            })
            .collect();
        let expected = elements.iter().map(|e| e.inverse()).collect_vec();

        let mut dst = vec![];
        CPUBackend::batch_inverse(&elements, &mut dst);

        assert_eq!(expected, dst);
    }

    #[test]
    fn batch_inverse_reused_vec_test() {
        let mut rng = SmallRng::seed_from_u64(0);
        let elements: Vec<QM31> = (0..10)
            .map(|_| {
                QM31::from_u32_unchecked(
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                )
            })
            .collect();
        let mut dst = vec![];
        CPUBackend::batch_inverse(&elements, &mut dst);
        let elements: Vec<QM31> = (0..10)
            .map(|_| {
                QM31::from_u32_unchecked(
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                    rng.gen::<u32>(),
                )
            })
            .collect();
        let expected = elements.iter().map(|e| e.inverse()).collect_vec();

        CPUBackend::batch_inverse(&elements, &mut dst);

        assert_eq!(expected, dst);
    }
}
