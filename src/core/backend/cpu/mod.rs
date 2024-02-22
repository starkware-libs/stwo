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

    // TODO(Ohad): Optimize.
    fn batch_inverse(elements: &[F]) -> Vec<F> {
        let n = elements.len();
        let mut r_i = Vec::with_capacity(n);

        // First pass.
        r_i.push(elements[0]);
        for i in 1..n {
            r_i.push(r_i[i - 1] * elements[i]);
        }

        // Inverse cumulative mul.
        let mut inverse = r_i[n - 1].inverse();

        // Second pass.
        for i in (1..n).rev() {
            r_i[i] = r_i[i - 1] * inverse;
            inverse *= elements[i];
        }
        r_i[0] = inverse;
        r_i
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
    fn test_batch_inverse() {
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

        let inversed = CPUBackend::batch_inverse(&elements);

        assert_eq!(expected, inversed);
    }
}
