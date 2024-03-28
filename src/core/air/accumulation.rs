//! Accumulators for a random linear combination of circle polynomials.
//! Given N polynomials, u_0(P), ... u_{N-1}(P), and a random alpha, the combined polynomial is
//! defined as
//!   f(p) = sum_i alpha^{N-1-i} u_i(P).

use itertools::Itertools;

use crate::core::backend::cpu::CPUCircleEvaluation;
use crate::core::backend::{Backend, CPUBackend};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::poly::circle::{CanonicCoset, CirclePoly, SecureCirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::generate_secure_powers;

/// Accumulates N evaluations of u_i(P0) at a single point.
/// Computes f(P0), the combined polynomial at that point.
/// For n accumulated evaluations, the i'th evaluation is multiplied by alpha^(N-1-i).
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

    pub fn finalize(self) -> SecureField {
        self.accumulation
    }
}

/// Accumulates evaluations of u_i(P), each at an evaluation domain of the size of that polynomial.
/// Computes the coefficients of f(P).
pub struct DomainEvaluationAccumulator<B: Backend> {
    random_coeff_powers: Vec<SecureField>,
    /// Accumulated evaluations for each log_size.
    /// Each `sub_accumulation` holds `sum_{i=0}^{N-1} evaluation_i * alpha^(N - 1 - i)`
    /// for all the evaluation for this log size and `N` is the total number of evaluations.
    sub_accumulations: Vec<SecureColumn<B>>,
}

impl<B: Backend> DomainEvaluationAccumulator<B> {
    /// Creates a new accumulator.
    /// `random_coeff` should be a secure random field element, drawn from the channel.
    /// `max_log_size` is the maximum log_size of the accumulated evaluations.
    pub fn new(random_coeff: SecureField, max_log_size: u32, total_columns: usize) -> Self {
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff_powers: generate_secure_powers(random_coeff, total_columns),
            sub_accumulations: (0..(max_log_size + 1))
                .map(|n| SecureColumn::zeros(1 << n))
                .collect(),
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
        self.sub_accumulations
            .get_many_mut(n_cols_per_size.map(|(log_size, _)| log_size as usize))
            .unwrap_or_else(|e| panic!("invalid log_sizes: {}", e))
            .into_iter()
            .zip(n_cols_per_size)
            .map(|(col, (_, n_cols))| {
                let random_coeffs = self
                    .random_coeff_powers
                    .split_off(self.random_coeff_powers.len() - n_cols);
                ColumnAccumulator {
                    random_coeff_powers: random_coeffs,
                    col,
                }
            })
            .collect_vec()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    /// Returns the log size of the resulting polynomial.
    pub fn log_size(&self) -> u32 {
        (self.sub_accumulations.len() - 1) as u32
    }
}

impl DomainEvaluationAccumulator<CPUBackend> {
    /// Computes f(P) as coefficients.
    pub fn finalize(self) -> SecureCirclePoly {
        assert_eq!(
            self.random_coeff_powers.len(),
            0,
            "not all random coefficients were used"
        );
        let mut res_coeffs = SecureColumn::<CPUBackend>::zeros(1 << self.log_size());
        let res_log_size = self.log_size();
        let res_size = 1 << res_log_size;

        for (log_size, values) in self.sub_accumulations.into_iter().enumerate().skip(1) {
            let coeffs = SecureColumn {
                columns: values.columns.map(|c| {
                    CPUCircleEvaluation::<_, BitReversedOrder>::new(
                        CanonicCoset::new(log_size as u32).circle_domain(),
                        c,
                    )
                    .interpolate()
                    .extend(res_log_size)
                    .coeffs
                }),
            };
            // Add column coefficients into result coefficients, element-wise, in-place.
            for i in 0..res_size {
                let res_coeff = res_coeffs.at(i) + coeffs.at(i);
                res_coeffs.set(i, res_coeff);
            }
        }

        SecureCirclePoly(res_coeffs.columns.map(CirclePoly::new))
    }
}

/// A domain accumulator for polynomials of a single size.
pub struct ColumnAccumulator<'a, B: Backend> {
    pub random_coeff_powers: Vec<SecureField>,
    col: &'a mut SecureColumn<B>,
}
impl<'a> ColumnAccumulator<'a, CPUBackend> {
    pub fn accumulate(&mut self, index: usize, evaluation: SecureField) {
        let val = self.col.at(index) + evaluation;
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

    #[test]
    fn test_domain_evaluation_accumulator() {
        // Generate a vector of random sizes with a constant seed.
        let rng = &mut StdRng::seed_from_u64(0);
        const LOG_SIZE_MIN: u32 = 4;
        const LOG_SIZE_BOUND: u32 = 10;
        const MASK: u32 = P;
        let mut log_sizes = (0..100)
            .map(|_| rng.gen_range(LOG_SIZE_MIN..LOG_SIZE_BOUND))
            .collect::<Vec<_>>();
        log_sizes.sort();

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
        let mut accumulator = DomainEvaluationAccumulator::<CPUBackend>::new(
            alpha,
            LOG_SIZE_BOUND,
            evaluations.len(),
        );
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
        let mut eval_chunk_offset = 0;
        for (log_size, n_cols) in n_cols_per_size.iter() {
            for index in 0..(1 << log_size) {
                let mut val = SecureField::zero();
                for (eval_index, (col_log_size, evaluation)) in
                    log_sizes.iter().zip(evaluations.iter()).enumerate()
                {
                    if *log_size != *col_log_size {
                        continue;
                    }

                    // The random coefficient powers chunk is in regular order.
                    let random_coeff_chunk =
                        &cols[(log_size - LOG_SIZE_MIN) as usize].random_coeff_powers;
                    val += random_coeff_chunk
                        [random_coeff_chunk.len() - 1 - (eval_index - eval_chunk_offset)]
                        * evaluation[index];
                }
                cols[(log_size - LOG_SIZE_MIN) as usize].accumulate(index, val);
            }
            eval_chunk_offset += n_cols;
        }
        let accumulator_poly = accumulator.finalize();

        // Pick an arbitrary sample point.
        let point = CirclePoint::<SecureField>::get_point(98989892);
        let accumulator_res = accumulator_poly.eval_at_point(point);

        // Use direct computation.
        let mut res = SecureField::default();
        for (log_size, values) in log_sizes.into_iter().zip(evaluations) {
            res = res * alpha
                + CPUCircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), values)
                    .interpolate()
                    .eval_at_point(point);
        }

        assert_eq!(accumulator_res, res);
    }
}
