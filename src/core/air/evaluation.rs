//! Accumulators for a random linear combination of circle polynomials.
//! Given N polynomials, sort them by size: u_0(P), ... u_{N-1}(P).
//! Given a random alpha, the combined polynomial is defined as
//!   f(p) = sum_i alpha^{N-1-i} u_i (P).

use crate::core::fields::qm31::QM31;
use crate::core::fields::Field;

/// Accumulates evaluations of u_i(P0) at a single point.
/// Computes f(P0), the combined polynomial at that point.
pub struct PointEvaluationAccumulator {
    random_coeff: QM31,
    // Accumulated evaluations for each log_size.
    // Each `sub_accumulation` holds `sum_{i=0}^{n-1} evaluation_i * alpha^(n-1-i)`,
    // where `n` is the number of accumulated evaluations for this log_size.
    sub_accumulations: Vec<QM31>,
    // Number of accumulated evaluations for each log_size.
    n_accumulated: Vec<usize>,
}
impl PointEvaluationAccumulator {
    /// Creates a new accumulator.
    /// `random_coeff` should be a secure random field element, drawn from the channel.
    /// `max_log_size` is the maximum log_size of the accumulated evaluations.
    pub fn new(random_coeff: QM31, max_log_size: u32) -> Self {
        // TODO(spapini): Consider making all log_sizes usize.
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff,
            sub_accumulations: vec![QM31::default(); max_log_size + 1],
            n_accumulated: vec![0; max_log_size + 1],
        }
    }

    /// Accumulates u_i(P0), a polynomial evaluation at a P0.
    pub fn accumulate(&mut self, log_size: u32, evaluation: QM31) {
        let sub_accumulation = &mut self.sub_accumulations[log_size as usize];
        *sub_accumulation = *sub_accumulation * self.random_coeff + evaluation;

        self.n_accumulated[log_size as usize] += 1;
    }

    /// Computes f(P0), the evaluation of the combined polynomial at P0.
    pub fn finalize(self) -> QM31 {
        // Each `sub_accumulation` holds a linear combination of a consecutive slice of
        // u_0(P0), ... u_{N-1}(P0):
        //   alpha^k u_i(P0) + alpha^{k-1} u_{i+1}(P0) + ... + alpha^0 u_{i+k-1}(P0).
        // To combine all these slices, multiply an accumulator by alpha^k, and add the next slice.
        self.sub_accumulations
            .iter()
            .zip(self.n_accumulated.iter())
            .fold(QM31::default(), |total, (sub_accumulation, n_i)| {
                total * self.random_coeff.pow(*n_i as u128) + *sub_accumulation
            })
    }
}

pub struct DomainEvaluationAccumulator;

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::fields::m31::M31;
    use crate::m31;

    #[test]
    fn test_point_evaluation_accumulator() {
        // Generate a vector of random sizes with a constant seed.
        let rng = &mut StdRng::seed_from_u64(0);
        const MAX_LOG_SIZE: u32 = 10;
        const MASK: u32 = (1 << 30) - 1;
        let log_sizes = (0..100)
            .map(|_| rng.gen_range(4..MAX_LOG_SIZE))
            .collect::<Vec<_>>();

        // Generate random evaluations.
        let evaluations = log_sizes
            .iter()
            .map(|_| M31::from_u32_unchecked(rng.gen::<u32>() & MASK))
            .collect::<Vec<_>>();
        let alpha = m31!(2).into();

        // Use accumulator.
        let mut accumulator = PointEvaluationAccumulator::new(alpha, MAX_LOG_SIZE);
        for (log_size, evaluation) in log_sizes.iter().zip(evaluations.iter()) {
            accumulator.accumulate(*log_size, (*evaluation).into());
        }
        let accumulator_res = accumulator.finalize();

        // Use direct computation.
        let mut res = QM31::default();
        // Sort evaluations by log_size.
        let mut pairs = log_sizes.into_iter().zip(evaluations).collect::<Vec<_>>();
        pairs.sort_by_key(|(log_size, _)| *log_size);
        for (_, evaluation) in pairs.iter() {
            res = res * alpha + *evaluation;
        }

        assert_eq!(accumulator_res, res);
    }
}
