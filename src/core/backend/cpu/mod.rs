use std::fmt::Debug;

use itertools::Itertools;

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

    fn batch_inverse(elements: &mut Self::Column) {
        let n = elements.len();

        // First pass.
        let r_i = elements
            .iter()
            .scan(F::one(), |cumulative_product, x| {
                *cumulative_product *= *x;
                Some(*cumulative_product)
            })
            .collect_vec();

        // Inverse cumulative mul.
        let mut inverse = r_i[n - 1].inverse();

        // Second pass.
        for i in (1..n).rev() {
            let temp = r_i[i - 1] * inverse;
            inverse *= elements[i];
            elements[i] = temp;
        }
        elements[0] = inverse;
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
        let mut elements: Vec<QM31> = (0..10)
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

        CPUBackend::batch_inverse(&mut elements);

        assert_eq!(expected, elements);
    }
}
