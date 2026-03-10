//! Accumulators for a random linear combination of circle polynomials.
//!
//! Given N polynomials, u_0(P), ... u_{N-1}(P), and a random alpha, the combined polynomial is
//! defined as
//!   f(p) = sum_i alpha^{N-1-i} u_i(P).

use itertools::Itertools;
use tracing::{span, Level};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::prover::backend::{Backend, Col, Column, ColumnOps, CpuBackend};
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, SecureCirclePoly};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

// TODO(ShaharS), rename terminology to constraints instead of columns.
/// Accumulates evaluations of u_i(P), each at an evaluation domain of the size of that polynomial.
/// Computes the coefficients of f(P).
pub struct DomainEvaluationAccumulator<B: Backend> {
    random_coeff_powers: Vec<SecureField>,
    /// Accumulated evaluations for each log_size.
    /// Each entry holds an optional (domain, column) pair for evaluations at that log_size.
    /// Each `sub_accumulation` holds the sum over all columns i of that log_size, of
    /// `evaluation_i * alpha^(N - 1 - i)` on the corresponding domain.
    sub_accumulations: Vec<Option<(CircleDomain, SecureColumnByCoords<B>)>>,
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

    /// Gets accumulators for some domains.
    /// `n_cols_per_domain` is an array of pairs (domain, n_cols).
    /// For each entry, a [ColumnAccumulator] is returned, expecting to accumulate `n_cols`
    /// evaluations on `domain`.
    /// The array size, `N`, is the number of different domains.
    pub fn columns<const N: usize>(
        &mut self,
        n_cols_per_domain: [(CircleDomain, usize); N],
    ) -> [ColumnAccumulator<'_, B>; N] {
        let log_sizes = n_cols_per_domain.map(|(domain, _)| domain.log_size() as usize);
        let slots = self
            .sub_accumulations
            .get_disjoint_mut(log_sizes)
            .unwrap_or_else(|e| panic!("invalid log_sizes: {e}"));

        slots
            .into_iter()
            .zip(n_cols_per_domain)
            .map(|(slot, (domain, n_cols))| {
                let random_coeffs = self
                    .random_coeff_powers
                    .split_off(self.random_coeff_powers.len() - n_cols);
                if let Some((existing_domain, _)) = slot.as_ref() {
                    assert_eq!(
                        *existing_domain,
                        domain,
                        "Domain mismatch for log_size {}: existing domain differs from requested",
                        domain.log_size()
                    );
                }
                let (_, col) = slot.get_or_insert_with(|| {
                    (domain, SecureColumnByCoords::zeros(1 << domain.log_size()))
                });
                ColumnAccumulator {
                    random_coeff_powers: random_coeffs,
                    col,
                }
            })
            .collect_vec()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    /// Skips the last `n_coeffs` random coefficients.
    ///
    /// This is useful when the component is disabled and its random coefficients are not used.
    ///
    /// We skip the last coefficients because the verifier combines constraints via
    /// `acc = acc * rand_coeff + new_constraint`. As a result, the first constraint uses the
    /// last random coefficient, the second constraint uses the second-to-last random
    /// coefficient, and so on.
    pub fn skip_coeffs(&mut self, n_coeffs: usize) {
        self.random_coeff_powers
            .truncate(self.random_coeff_powers.len() - n_coeffs);
    }

    /// Returns the log size of the resulting polynomial.
    pub const fn log_size(&self) -> u32 {
        (self.sub_accumulations.len() - 1) as u32
    }

    /// Computes f(P) as coefficients.
    /// `twiddles` must be precomputed for the max-size canonical domain's half coset.
    pub fn finalize(self, twiddles: &TwiddleTree<B>) -> SecureCirclePoly<B> {
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

        let sub_accumulations = self
            .sub_accumulations
            .into_iter()
            .filter_map(|entry| {
                entry.map(|(domain, col)| {
                    assert!(
                        domain.is_canonic(),
                        "non-canonical domain are not supported"
                    );
                    col
                })
            })
            .collect_vec();
        let lifted_accumulation = B::lift_and_accumulate(sub_accumulations);

        if let Some(eval) = lifted_accumulation {
            // `lifted_accumulation` must be of size `log_size`, i.e. there must at least one sub
            // accumulation of size `log_size`.
            SecureCirclePoly(eval.columns.map(|c| {
                CircleEvaluation::<B, BaseField, BitReversedOrder>::new(
                    CanonicCoset::new(log_size).circle_domain(),
                    c,
                )
                .interpolate_with_twiddles(twiddles)
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
    use crate::prover::poly::circle::PolyOps;
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
        let n_cols_per_domain: [(CircleDomain, usize); (LOG_SIZE_BOUND - LOG_SIZE_MIN) as usize] =
            array::from_fn(|i| {
                let current_log_size = LOG_SIZE_MIN + i as u32;
                let n_cols = log_sizes
                    .iter()
                    .copied()
                    .filter(|&log_size| log_size == current_log_size)
                    .count();
                (CanonicCoset::new(current_log_size).circle_domain(), n_cols)
            });

        let mut cols = accumulator.columns(n_cols_per_domain);
        let mut eval_chunk_offset = 0;
        for (domain, n_cols) in n_cols_per_domain.iter() {
            let log_size = domain.log_size();
            for index in 0..(1 << log_size) {
                let mut val = SecureField::zero();
                for (eval_index, (col_log_size, evaluation)) in
                    log_sizes.iter().zip(evaluations.iter()).enumerate()
                {
                    if log_size != *col_log_size {
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
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(LOG_SIZE_BOUND - 1)
                .circle_domain()
                .half_coset,
        );
        let accumulator_poly = accumulator.finalize(&twiddles);

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

    /// Tests that calling `columns()` twice for the same log_size accumulates correctly
    /// (the second call must not zero out the first component's values).
    #[test]
    fn test_accumulate_two_components_with_the_same_size() {
        const LOG_SIZE: u32 = 5;
        let alpha = qm31!(2, 3, 4, 5);
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();

        // Two components, each with 1 constraint, both at the same log_size.
        let mut accumulator = DomainEvaluationAccumulator::<CpuBackend>::new(alpha, LOG_SIZE, 2);

        // First component accumulates.
        let [mut col1] = accumulator.columns([(domain, 1)]);
        for i in 0..(1 << LOG_SIZE) {
            col1.accumulate(i, col1.random_coeff_powers[0] * M31::from(i as u32 + 1));
        }

        // Second component accumulates into the same log_size.
        let [mut col2] = accumulator.columns([(domain, 1)]);
        for i in 0..(1 << LOG_SIZE) {
            col2.accumulate(i, col2.random_coeff_powers[0] * M31::from(100));
        }

        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let poly = accumulator.finalize(&twiddles);

        // Verify at a sample point using direct computation.
        let point = CirclePoint::<SecureField>::get_point(12345);
        let actual = poly.eval_at_point(point);

        // Direct: alpha^1 * eval1(point) + alpha^0 * eval2(point)
        let eval1 = CpuCircleEvaluation::<BaseField, BitReversedOrder>::new(
            domain,
            (0..(1 << LOG_SIZE))
                .map(|i| M31::from(i as u32 + 1))
                .collect(),
        )
        .interpolate()
        .eval_at_point(point);

        let eval2 = CpuCircleEvaluation::<BaseField, BitReversedOrder>::new(
            domain,
            vec![M31::from(100); 1 << LOG_SIZE],
        )
        .interpolate()
        .eval_at_point(point);

        let expected = alpha * eval1 + eval2;
        assert_eq!(actual, expected);
    }
}
