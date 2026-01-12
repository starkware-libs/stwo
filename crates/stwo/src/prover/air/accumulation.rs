//! Accumulators for a random linear combination of circle polynomials.
//!
//! Given N polynomials, u_0(P), ... u_{N-1}(P), and a random alpha, the combined polynomial is
//! defined as
//!   f(p) = sum_i alpha^{N-1-i} u_i(P).

use itertools::Itertools;
use tracing::{span, Level};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CanonicCoset;
use crate::prover::backend::{Backend, Col, Column, ColumnOps, CpuBackend};
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, SecureCirclePoly};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

// TODO(ShaharS), rename terminology to constraints instead of columns.
/// Accumulates evaluations of u_i(P), each at an evaluation domain of the size of that polynomial.
/// Computes the coefficients of f(P).
pub struct DomainEvaluationAccumulator<B: Backend> {
    random_coeff_powers: Vec<SecureField>,
    /// Accumulated evaluations for each log_size.
    /// Each `sub_accumulation` holds the sum over all columns i of that log_size, of
    /// `evaluation_i * alpha^(N - 1 - i)`
    /// where `N` is the total number of evaluations.
    sub_accumulations: Vec<Option<SecureColumnByCoords<B>>>,
}

impl<B: Backend> DomainEvaluationAccumulator<B> {
    /// Creates a new accumulator.
    /// `random_coeff` should be a secure random field element, drawn from the channel.
    /// `max_log_size` is the maximum log_size of the accumulated evaluations.
    pub fn new(random_coeff: SecureField, max_log_size: u32, total_columns: usize) -> Self {
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff_powers: B::generate_secure_powers(random_coeff, total_columns),
            sub_accumulations: (0..(max_log_size + 1)).map(|_| None).collect(),
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
            .get_disjoint_mut(n_cols_per_size.map(|(log_size, _)| log_size as usize))
            .unwrap_or_else(|e| panic!("invalid log_sizes: {e}"))
            .into_iter()
            .zip(n_cols_per_size)
            .map(|(col, (log_size, n_cols))| {
                let random_coeffs = self
                    .random_coeff_powers
                    .split_off(self.random_coeff_powers.len() - n_cols);
                ColumnAccumulator {
                    random_coeff_powers: random_coeffs,
                    col: col.get_or_insert_with(|| SecureColumnByCoords::zeros(1 << log_size)),
                }
            })
            .collect_vec()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    /// Returns the log size of the resulting polynomial.
    pub const fn log_size(&self) -> u32 {
        (self.sub_accumulations.len() - 1) as u32
    }

    /// Computes f(P) as coefficients.
    pub fn finalize(self) -> SecureCirclePoly<B> {
        assert_eq!(
            self.random_coeff_powers.len(),
            0,
            "not all random coefficients were used"
        );
        let log_size = self.log_size();
        let _span = span!(
            Level::INFO,
            "Constraints interpolation",
            class = "ConstraintInterpolation"
        )
        .entered();

        let sub_accumulations = self.sub_accumulations.into_iter().flatten().collect_vec();
        let lifted_accumulation = B::lift_and_accumulate(sub_accumulations);

        if let Some(eval) = lifted_accumulation {
            // `lifted_accumulation` must be of size `log_size`, i.e. there must at least one sub
            // accumulation of size `log_size`.
            let twiddles =
                B::precompute_twiddles(CanonicCoset::new(log_size).circle_domain().half_coset);

            SecureCirclePoly(eval.columns.map(|c| {
                CircleEvaluation::<B, BaseField, BitReversedOrder>::new(
                    CanonicCoset::new(log_size).circle_domain(),
                    c,
                )
                .interpolate_with_twiddles(&twiddles)
            }))
        } else {
            SecureCirclePoly(std::array::from_fn(|_| {
                CircleCoefficients::new(Col::<B, BaseField>::zeros(1 << log_size))
            }))
        }
    }
}

pub trait AccumulationOps: ColumnOps<BaseField> + Sized {
    /// Accumulates other into column:
    ///   column = column + other.
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>);

    /// Generates the first `n_powers` powers of `felt`.
    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField>;

    /// Receives a possibly empty vector of columns, sorted in strictly ascending order by column
    /// length, and returns a column which is the coordinate-wise sum of the lifts of the columns
    /// (see also [`crate::prover::backend::simd::blake2s_lifted::to_lifted_simd`] for the
    /// definition of the lift of a column). The size of the output column is equal to the size of
    /// the largest column (i.e. the size the last one).
    ///
    /// If `cols` is empty, returns `None`.
    fn lift_and_accumulate(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>>;
}

/// A domain accumulator for polynomials of a single size.
pub struct ColumnAccumulator<'a, B: Backend> {
    pub random_coeff_powers: Vec<SecureField>,
    pub col: &'a mut SecureColumnByCoords<B>,
}
impl ColumnAccumulator<'_, CpuBackend> {
    pub fn accumulate(&mut self, index: usize, evaluation: SecureField) {
        let val = self.col.at(index) + evaluation;
        self.col.set(index, val);
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::M31;
    use crate::prover::backend::cpu::CpuCircleEvaluation;
    use crate::qm31;

    #[test]
    fn test_domain_evaluation_accumulator_lifted() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE_MIN: u32 = 4;
        const LOG_SIZE_BOUND: u32 = 10;
        let mut log_sizes = (0..100)
            .map(|_| rng.gen_range(LOG_SIZE_MIN..LOG_SIZE_BOUND))
            .collect::<Vec<_>>();
        log_sizes.sort();

        // Generate random evaluations.
        let evaluations = log_sizes
            .iter()
            .map(|log_size| {
                (0..(1 << *log_size))
                    .map(|_| M31::from(rng.gen::<u32>()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let alpha = qm31!(2, 3, 4, 5);

        let mut accumulator = DomainEvaluationAccumulator::<CpuBackend>::new(
            alpha,
            LOG_SIZE_BOUND - 1,
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

        // Use direct computation: first interpolate each evaluation to obtain a polynomial,
        // evaluate its lift at `point`, and accumulate over the evaluations.
        let mut res = SecureField::default();
        for (log_size, values) in log_sizes.into_iter().zip(evaluations) {
            res = res * alpha
                + CpuCircleEvaluation::<BaseField, BitReversedOrder>::new(
                    CanonicCoset::new(log_size).circle_domain(),
                    values,
                )
                .interpolate()
                // The max log domain size is LOG_SIZE_BOUND - 1.
                .eval_at_point(point.repeated_double(LOG_SIZE_BOUND - 1 - log_size));
        }

        assert_eq!(accumulator_res, res);
    }
}
