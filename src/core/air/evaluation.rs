//! Accumulators for a random linear combination of circle polynomials.
//! Given N polynomials, sort them by size: u_0(P), ... u_{N-1}(P).
//! Given a random alpha, the combined polynomial is defined as
//!   f(p) = sum_i alpha^{N-1-i} u_i (P).

use crate::core::backend::{Backend, CPUBackend, Column, VecLike};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::QM31;
use crate::core::fields::Field;
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};

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
        //   alpha^n_k u_i(P0) + alpha^{n_k-1} u_{i+1}(P0) + ... + alpha^0 u_{i+n_k-1}(P0).
        // To combine all these slices, multiply an accumulator by alpha^k, and add the next slice.
        self.sub_accumulations
            .iter()
            .zip(self.n_accumulated.iter())
            .fold(QM31::default(), |total, (sub_accumulation, n_i)| {
                total * self.random_coeff.pow(*n_i as u128) + *sub_accumulation
            })
    }
}

/// Accumulates evaluations of u_i(P), each at an evaluation domain of the size of that polynomial.
/// Computes the coefficients of f(P).
pub struct DomainEvaluationAccumulator<B: Backend> {
    random_coeff: QM31,
    // Accumulated evaluations for each log_size.
    // Each `sub_accumulation` holds `sum_{i=0}^{n-1} evaluation_i * alpha^(n-1-i)`,
    // where `n` is the number of accumulated evaluations for this log_size.
    sub_accumulations: Vec<Column<B, QM31>>,
    // Number of accumulated evaluations for each log_size.
    n_cols_per_size: Vec<usize>,
}
impl<B: Backend> DomainEvaluationAccumulator<B> {
    /// Creates a new accumulator.
    /// `random_coeff` should be a secure random field element, drawn from the channel.
    /// `max_log_size` is the maximum log_size of the accumulated evaluations.
    pub fn new(random_coeff: QM31, max_log_size: u32) -> Self {
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff,
            sub_accumulations: (0..(max_log_size + 1))
                .map(|n| Column::<B, QM31>::from_vec(vec![QM31::default(); 1 << n]))
                .collect(),
            n_cols_per_size: vec![0; max_log_size + 1],
        }
    }

    /// Gets accumulators for some sizes.
    /// `n_cols_per_size` is an array of pairs (log_size, n_cols).
    /// For each entry, a [ColumnAccumulator] is returned, expecting to accumulate `n_cols`
    /// evaluations of size `log_size`.
    /// The array size, `N`, is the number of different sizes.
    pub fn columns<const N: usize>(
        &mut self,
        n_cols_per_size: [(u32, usize); N],
    ) -> [ColumnAccumulator<'_, B>; N] {
        n_cols_per_size.iter().for_each(|(log_size, n_col)| {
            self.n_cols_per_size[*log_size as usize] += n_col;
        });
        self.sub_accumulations
            .get_many_mut(n_cols_per_size.map(|(log_size, _)| log_size as usize))
            .unwrap_or_else(|e| panic!("invalid log_sizes: {}", e))
            .map(|c| ColumnAccumulator {
                random_coeff: self.random_coeff,
                col: c,
            })
    }

    /// Returns the log_size of the resulting polynomial.
    pub fn log_size(&self) -> u32 {
        (self.sub_accumulations.len() - 1) as u32
    }
}

impl DomainEvaluationAccumulator<CPUBackend> {
    /// Computes f(P) as coefficients.
    pub fn finalize(self) -> CirclePoly<CPUBackend, QM31> {
        let mut res_coeffs = vec![QM31::default(); 1 << self.log_size()];
        let res_log_size = self.log_size();
        for (coeffs, n_cols) in self
            .sub_accumulations
            .into_iter()
            .enumerate()
            .map(|(log_size, values)| {
                if log_size == 0 {
                    return values;
                }
                CircleEvaluation::<CPUBackend, QM31>::new(
                    CircleDomain::constraint_evaluation_domain(log_size as u32),
                    values,
                )
                .interpolate()
                .extend(res_log_size)
                .coeffs
            })
            .zip(self.n_cols_per_size.iter())
        {
            // Add poly.coeffs into coeffs, elementwise, inplace.
            let multiplier = self.random_coeff.pow(*n_cols as u128);
            res_coeffs
                .iter_mut()
                .zip(coeffs.iter())
                .for_each(|(res_coeff, current_coeff)| {
                    *res_coeff = *res_coeff * multiplier + *current_coeff
                });
        }

        CirclePoly::new(res_coeffs)
    }
}

/// An domain accumulator for polynomials of a single size.
pub struct ColumnAccumulator<'a, B: Backend> {
    random_coeff: QM31,
    col: &'a mut Column<B, QM31>,
}
impl<'a> ColumnAccumulator<'a, CPUBackend> {
    pub fn accumulate(&mut self, index: usize, evaluation: BaseField) {
        let accum = &mut self.col[index];
        *accum = *accum * self.random_coeff + evaluation;
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::backend::CPUBackend;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::{M31, P};
    use crate::qm31;

    type B = CPUBackend;

    #[test]
    fn test_point_evaluation_accumulator() {
        // Generate a vector of random sizes with a constant seed.
        let rng = &mut StdRng::seed_from_u64(0);
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

    #[test]
    fn test_domain_evaluation_accumulator() {
        // Generate a vector of random sizes with a constant seed.
        let rng = &mut StdRng::seed_from_u64(0);
        const LOG_SIZE_MIN: u32 = 4;
        const LOG_SIZE_BOUND: u32 = 10;
        const MASK: u32 = P;
        let log_sizes = (0..100)
            .map(|_| rng.gen_range(LOG_SIZE_MIN..LOG_SIZE_BOUND))
            .collect::<Vec<_>>();

        // Generate random evaluations.
        let evaluations = log_sizes
            .iter()
            .map(|log_size| {
                (0..(1 << *log_size))
                    .map(|_| M31::from_u32_unchecked(rng.gen::<u32>() & MASK))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let alpha = qm31!(2, 3, 4, 5);

        // Use accumulator.
        let mut accumulator = DomainEvaluationAccumulator::<B>::new(alpha, LOG_SIZE_BOUND);
        let n_cols_per_size: [(u32, usize); (LOG_SIZE_BOUND - LOG_SIZE_MIN) as usize] =
            array::from_fn(|i| {
                let current_log_size = LOG_SIZE_MIN + i as u32;
                let n_cols = log_sizes
                    .iter()
                    .copied()
                    .filter(|&log_size| log_size == current_log_size)
                    .count();
                (current_log_size, n_cols)
            });
        let mut cols = accumulator.columns(n_cols_per_size);
        for (log_size, evaluation) in log_sizes.iter().zip(evaluations.iter()) {
            for (index, evaluation) in evaluation.iter().enumerate() {
                cols[(log_size - LOG_SIZE_MIN) as usize].accumulate(index, *evaluation);
            }
        }
        let accumulator_poly = accumulator.finalize();

        // Pick an arbitrary sample point.
        let point = CirclePoint::<QM31>::get_point(98989892);
        let accumulator_res = accumulator_poly.eval_at_point(point);

        // Sort evaluations by log_size.
        let mut pairs = log_sizes.into_iter().zip(evaluations).collect::<Vec<_>>();
        pairs.sort_by_key(|(log_size, _)| *log_size);

        // Use direct computation.
        let mut res = QM31::default();
        for (log_size, values) in pairs.into_iter() {
            res = res * alpha
                + CircleEvaluation::<B, _>::new(
                    CircleDomain::constraint_evaluation_domain(log_size),
                    values,
                )
                .interpolate()
                .eval_at_point(point);
        }

        assert_eq!(accumulator_res, res);
    }
}
