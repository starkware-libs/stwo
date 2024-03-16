//! Accumulators for a random linear combination of circle polynomials.
//! Given N polynomials, sort them by size: u_0(P), ... u_{N-1}(P).
//! Given a random alpha, the combined polynomial is defined as
//!   f(p) = sum_i alpha^{N-1-i} u_i (P).
use crate::core::backend::{Backend, CPUBackend};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure::{SecureCirclePoly, SecureColumn};
use crate::core::fields::{FieldExpOps, FieldOps};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use crate::core::poly::BitReversedOrder;

/// Accumulates evaluations of u_i(P0) at a single point.
/// Computes f(P0), the combined polynomial at that point.
pub struct PointEvaluationAccumulator {
    random_coeff: SecureField,
    /// Accumulated evaluations for each log_size.
    /// Each `sub_accumulation` holds `sum_{i=0}^{n-1} evaluation_i * alpha^(n-1-i)`,
    /// where `n` is the number of accumulated evaluations for this log_size.
    sub_accumulations: Vec<SecureField>,
    /// Number of accumulated evaluations for each log_size.
    n_accumulated: Vec<usize>,
}
impl PointEvaluationAccumulator {
    /// Creates a new accumulator.
    /// `random_coeff` should be a secure random field element, drawn from the channel.
    /// `max_log_size` is the maximum log_size of the accumulated evaluations.
    pub fn new(random_coeff: SecureField, max_log_size: u32) -> Self {
        // TODO(spapini): Consider making all log_sizes usize.
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff,
            sub_accumulations: vec![SecureField::default(); max_log_size + 1],
            n_accumulated: vec![0; max_log_size + 1],
        }
    }

    /// Accumulates u_i(P0), a polynomial evaluation at a P0.
    pub fn accumulate(&mut self, log_size: u32, evaluation: SecureField) {
        assert!(log_size > 0 && log_size < self.sub_accumulations.len() as u32);
        let sub_accumulation = &mut self.sub_accumulations[log_size as usize];
        *sub_accumulation = *sub_accumulation * self.random_coeff + evaluation;

        self.n_accumulated[log_size as usize] += 1;
    }

    /// Computes f(P0), the evaluation of the combined polynomial at P0.
    pub fn finalize(self) -> SecureField {
        // Each `sub_accumulation` holds a linear combination of a consecutive slice of
        // u_0(P0), ... u_{N-1}(P0):
        //   alpha^n_k u_i(P0) + alpha^{n_k-1} u_{i+1}(P0) + ... + alpha^0 u_{i+n_k-1}(P0).
        // To combine all these slices, multiply an accumulator by alpha^k, and add the next slice.
        self.sub_accumulations
            .iter()
            .zip(self.n_accumulated.iter())
            .fold(SecureField::default(), |total, (sub_accumulation, n_i)| {
                total * self.random_coeff.pow(*n_i as u128) + *sub_accumulation
            })
    }
}

/// Accumulates evaluations of u_i(P), each at an evaluation domain of the size of that polynomial.
/// Computes the coefficients of f(P).
pub struct DomainEvaluationAccumulator<B: Backend> {
    pub random_coeff: SecureField,
    /// Accumulated evaluations for each log_size.
    /// Each `sub_accumulation` holds `sum_{i=0}^{n-1} evaluation_i * alpha^(n-1-i)`,
    /// where `n` is the number of accumulated evaluations for this log_size.
    sub_accumulations: Vec<SecureColumn<B>>,
    /// Number of accumulated evaluations for each log_size.
    n_cols_per_size: Vec<usize>,
}

impl<B: Backend> DomainEvaluationAccumulator<B> {
    /// Creates a new accumulator.
    /// `random_coeff` should be a secure random field element, drawn from the channel.
    /// `max_log_size` is the maximum log_size of the accumulated evaluations.
    pub fn new(random_coeff: SecureField, max_log_size: u32) -> Self {
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff,
            sub_accumulations: (0..(max_log_size + 1))
                .map(|n| SecureColumn::zeros(1 << n))
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
            assert!(*log_size > 0 && *log_size < self.sub_accumulations.len() as u32);
            self.n_cols_per_size[*log_size as usize] += n_col;
        });
        let mut res = self
            .sub_accumulations
            .get_many_mut(n_cols_per_size.map(|(log_size, _)| log_size as usize))
            .unwrap_or_else(|e| panic!("invalid log_sizes: {}", e))
            .map(|c| ColumnAccumulator {
                random_coeff_pow: self.random_coeff,
                col: c,
            });
        for i in 0..N {
            res[i].random_coeff_pow = self.random_coeff.pow(n_cols_per_size[i].1 as u128);
        }
        res
    }

    /// Returns the log size of the resulting polynomial.
    pub fn log_size(&self) -> u32 {
        (self.sub_accumulations.len() - 1) as u32
    }
}

pub trait AccumulationOps: FieldOps<BaseField> + Sized {
    /// Accumulates other into colum:
    ///   column = column * alpha + other.
    fn accumulate(column: &mut SecureColumn<Self>, alpha: SecureField, other: &SecureColumn<Self>);
}

impl<B: Backend> DomainEvaluationAccumulator<B> {
    /// Computes f(P) as coefficients.
    pub fn finalize(self) -> SecureCirclePoly<B> {
        let mut res_coeffs = SecureColumn::<B>::zeros(1 << self.log_size());
        let res_log_size = self.log_size();

        for ((log_size, values), n_cols) in self
            .sub_accumulations
            .into_iter()
            .enumerate()
            .zip(self.n_cols_per_size.iter())
            .skip(1)
        {
            if *n_cols == 0 {
                continue;
            }
            let coeffs = SecureColumn::<B> {
                cols: values.cols.map(|c| {
                    CircleEvaluation::<B, BaseField, BitReversedOrder>::new(
                        CanonicCoset::new(log_size as u32).circle_domain(),
                        c,
                    )
                    .interpolate()
                    .extend(res_log_size)
                    .coeffs
                }),
            };
            // Add column coefficients into result coefficients, element-wise, in-place.
            let multiplier = self.random_coeff.pow(*n_cols as u128);
            B::accumulate(&mut res_coeffs, multiplier, &coeffs);
        }

        SecureCirclePoly(res_coeffs.cols.map(CirclePoly::new))
    }
}

/// An domain accumulator for polynomials of a single size.
pub struct ColumnAccumulator<'a, B: Backend> {
    pub random_coeff_pow: SecureField,
    pub col: &'a mut SecureColumn<B>,
}
impl<'a> ColumnAccumulator<'a, CPUBackend> {
    pub fn accumulate(&mut self, index: usize, evaluation: SecureField) {
        // TODO(spapini): Multiplying QM31 by QM31 is not the best way to do this.
        // It's probably better to cache all the coefficient powers and multiply QM31 by M31,
        // and only add in QM31.
        let val = self.col.at(index) * self.random_coeff_pow + evaluation;
        self.col.set(index, val);
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use num_traits::Zero;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::backend::cpu::CPUCircleEvaluation;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::{M31, P};
    use crate::qm31;

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
        let mut res = SecureField::default();
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
        let mut accumulator = DomainEvaluationAccumulator::<CPUBackend>::new(alpha, LOG_SIZE_BOUND);
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
        for log_size in n_cols_per_size.iter().map(|(log_size, _)| *log_size) {
            for index in 0..(1 << log_size) {
                let mut val = SecureField::zero();
                for (col_log_size, evaluation) in log_sizes.iter().zip(evaluations.iter()) {
                    if log_size != *col_log_size {
                        continue;
                    }
                    val = val * alpha + evaluation[index];
                }
                cols[(log_size - LOG_SIZE_MIN) as usize].accumulate(index, val);
            }
        }
        let accumulator_poly = accumulator.finalize();

        // Pick an arbitrary sample point.
        let point = CirclePoint::<SecureField>::get_point(98989892);
        let accumulator_res = accumulator_poly.eval_at_point(point);

        // Sort evaluations by log_size.
        let mut pairs = log_sizes.into_iter().zip(evaluations).collect::<Vec<_>>();
        pairs.sort_by_key(|(log_size, _)| *log_size);

        // Use direct computation.
        let mut res = SecureField::default();
        for (log_size, values) in pairs.into_iter() {
            res = res * alpha
                + CPUCircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), values)
                    .interpolate()
                    .eval_at_point(point);
        }

        assert_eq!(accumulator_res, res);
    }
}
