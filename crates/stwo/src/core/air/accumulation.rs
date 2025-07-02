//! Accumulators for a random linear combination of circle polynomial evaluations.
//!
//! Given N evaluations, u_0(P0), ... u_{N-1}(P0), and a random alpha, the combined evaluation is
//! defined as
//!   f(P0) = sum_i alpha^{N-1-i} u_i(P0).

use crate::core::fields::qm31::SecureField;

/// Accumulates N evaluations of u_i(P0) at a single point.
/// Computes f(P0), the combined polynomial at that point.
/// For n accumulated evaluations, the i'th evaluation is multiplied by alpha^(N-1-i).
#[derive(Debug, Clone)]
pub struct PointEvaluationAccumulator {
    random_coeff: SecureField,
    accumulation: SecureField,
}

impl PointEvaluationAccumulator {
    /// Creates a new accumulator.
    /// `random_coeff` should be a secure random field element, drawn from the channel.
    pub fn new(random_coeff: SecureField) -> Self {
        Self {
            random_coeff,
            accumulation: SecureField::default(),
        }
    }

    /// Accumulates u_i(P0), a polynomial evaluation at a P0 in reverse order.
    pub fn accumulate(&mut self, evaluation: SecureField) {
        self.accumulation = self.accumulation * self.random_coeff + evaluation;
    }

    pub const fn finalize(self) -> SecureField {
        self.accumulation
    }
}

#[cfg(test)]
mod tests {

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std_shims::Vec;

    use super::*;
    use crate::core::fields::m31::{M31, P};
    use crate::qm31;

    #[test]
    fn test_point_evaluation_accumulator() {
        // Generate a vector of random sizes with a constant seed.
        let mut rng = SmallRng::seed_from_u64(0);
        const MAX_LOG_SIZE: u32 = 10;
        const MASK: u32 = P;
        let log_sizes = (0..100)
            .map(|_| rng.gen_range(4..MAX_LOG_SIZE))
            .collect::<Vec<_>>();

        // Generate random evaluations.
        let evaluations = log_sizes
            .iter()
            .map(|_| M31::from_u32_unchecked(rng.gen::<u32>() & MASK))
            .collect::<Vec<_>>();
        let alpha = qm31!(2, 3, 4, 5);

        // Use accumulator.
        let mut accumulator = PointEvaluationAccumulator::new(alpha);
        for (_, evaluation) in log_sizes.iter().zip(evaluations.iter()) {
            accumulator.accumulate((*evaluation).into());
        }
        let accumulator_res = accumulator.finalize();

        // Use direct computation.
        let mut res = SecureField::default();
        for evaluation in evaluations.iter() {
            res = res * alpha + *evaluation;
        }

        assert_eq!(accumulator_res, res);
    }
}
