use std::fmt::Debug;

use super::{Backend, Column, FieldOps};
use crate::core::fields::Field;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::poly::NaturalOrder;

mod poly;

#[derive(Copy, Clone, Debug)]
pub struct CPUBackend;

impl CPUBackend {
    pub fn batch_inverse_optimized<F: Field, const W: usize>(column: &[F], dst: &mut [F]) {
        let n = column.len();
        if n < W {
            Self::batch_inverse(column, dst);
            return;
        }
        debug_assert!(n.is_power_of_two());

        // First pass.
        let mut cum_prod: [F; W] = column[..W].try_into().unwrap();
        dst[..W].copy_from_slice(&cum_prod);
        for i in W..n {
            cum_prod[i % W] *= column[i];
            dst[i] = cum_prod[i % W];
        }
        debug_assert_eq!(dst.len(), n);

        // Inverse cumulative products.
        // Use classic batch inversion.
        let mut tail_inverses = [F::one(); W];
        Self::batch_inverse(&dst[n - W..], &mut tail_inverses);

        // Second pass.
        for i in (W..n).rev() {
            dst[i] = dst[i - W] * tail_inverses[i % W];
            tail_inverses[i % W] *= column[i];
        }
        dst[0..W].copy_from_slice(&tail_inverses);
    }
}

impl Backend for CPUBackend {}

impl<F: Field> FieldOps<F> for CPUBackend {
    type Column = Vec<F>;

    /// Batch inversion using Montgomery's trick.
    // TODO(Ohad): Benchmark this function.
    fn batch_inverse(column: &[F], dst: &mut [F]) {
        let n = column.len();
        assert_eq!(n, dst.len());

        dst[0] = column[0];
        // First pass.
        for i in 1..n {
            dst[i] = dst[i - 1] * column[i];
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
    use num_traits::One;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    use crate::core::backend::{CPUBackend, FieldOps};
    use crate::core::fields::m31::M31;
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

        let mut dst = vec![QM31::one(); 10];
        CPUBackend::batch_inverse(&elements, &mut dst);

        assert_eq!(expected, dst);
    }

    #[test]
    fn batch_inverse_optimized_test() {
        const N_ELEMENTS: usize = 1 << 8;
        let mut rng = SmallRng::seed_from_u64(0);
        let elements: Vec<M31> = (0..N_ELEMENTS)
            .map(|_| M31::from_u32_unchecked(rng.gen::<u32>()))
            .collect();
        let expected = elements.iter().map(|e| e.inverse()).collect_vec();

        let mut dst = vec![M31::one(); N_ELEMENTS];
        CPUBackend::batch_inverse_optimized::<M31, 16>(&elements, &mut dst);

        assert_eq!(expected, dst);
    }
}
