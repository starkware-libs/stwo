use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::QM31;
use crate::core::fields::Field;
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};

/// Accumulates evaluations of constraint polynomials at a single point.
/// The result is an evaluation of the composition polynomial at that point.
pub struct PointEvaluationAccumulator {
    random_coeff: QM31,
    // Accumulated evaluations for each log_size.
    // Each `sub_accumulation` holds `sum_{i=0}^{n-1} evaluation_i *
    // random_coeff^(n_accumulated-1-i)`,
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

    /// Accumulates a constraint evaluation at a point.
    pub fn accumulate(&mut self, log_size: u32, evaluation: QM31) {
        let sub_accumulation = &mut self.sub_accumulations[log_size as usize];
        *sub_accumulation = *sub_accumulation * self.random_coeff + evaluation;

        self.n_accumulated[log_size as usize] += 1;
    }

    /// Finalizes the accumulation and returns the evaluation of the composition polynomial at the
    /// point.
    pub fn finalize(self) -> QM31 {
        self.sub_accumulations
            .iter()
            .zip(self.n_accumulated.iter())
            .fold(QM31::default(), |total, (sub_accumulation, n)| {
                // `total` holds the accumulation of evaluations of smaller log_sizes, with
                // coefficients ranging from `random_coeff^0` to `random_coeff^(N-1)`.
                // `sub_accumulation` holds the accumulation of evaluations of the current log_size,
                // with coefficients ranging from `random_coeff^0` to `random_coeff^(n-1)`.
                // We need to multiply `total` by `random_coeff^n` and add `sub_accumulation`.
                total * self.random_coeff.pow(*n as u128) + *sub_accumulation
            })
    }
}

/// Accumulates evalaution of the composition polynomial on the entire evaluation domain.
pub struct DomainEvaluationAccumulator {
    random_coeff: QM31,
    evaluations: Vec<Vec<QM31>>,
    evaluation_counts: Vec<usize>,
}
impl DomainEvaluationAccumulator {
    pub fn new(random_coeff: QM31, max_log_size: u32) -> Self {
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff,
            evaluations: (0..(max_log_size + 1))
                .map(|n| vec![QM31::default(); 1 << n])
                .collect(),
            evaluation_counts: vec![0; max_log_size + 1],
        }
    }
    // TODO(spapini): Think how to couple the count increment with the accumulation, without
    // overcomplicating the API.
    pub fn increment_count(&mut self, log_size: u32) {
        self.evaluation_counts[log_size as usize] += 1;
    }
    pub fn accumulate(&mut self, log_size: u32, index: usize, evaluation: BaseField) {
        let accum = &mut self.evaluations[log_size as usize][index];
        *accum = *accum * self.random_coeff + evaluation;
    }

    pub fn finalize(self) -> CirclePoly<QM31> {
        let mut res_coeffs = vec![QM31::default(); 1 << self.log_size()];
        let res_log_size = self.log_size();
        for (coeffs, count) in self
            .evaluations
            .into_iter()
            .enumerate()
            .map(|(log_size, values)| {
                if log_size == 0 {
                    return values;
                }
                CircleEvaluation::new(
                    CircleDomain::constraint_evaluation_domain(log_size as u32),
                    values,
                )
                .interpolate()
                .extend(res_log_size)
                .coeffs
            })
            .zip(self.evaluation_counts.iter())
        {
            // Add poly.coeffs into coeffs, elementwise, inplace.
            let multiplier = self.random_coeff.pow(*count as u128);
            res_coeffs
                .iter_mut()
                .zip(coeffs.iter())
                .for_each(|(a, b)| *a = *a * multiplier + *b);
        }

        CirclePoly::new(res_coeffs)
    }

    fn log_size(&self) -> u32 {
        (self.evaluations.len() - 1) as u32
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::circle::CirclePoint;
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

    #[test]
    fn test_domain_evaluation_accumulator() {
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
            .map(|log_size| {
                (0..(1 << *log_size))
                    .map(|_| M31::from_u32_unchecked(rng.gen::<u32>() & MASK))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let alpha = m31!(2).into();

        // Use accumulator.
        let mut accumulator = DomainEvaluationAccumulator::new(alpha, MAX_LOG_SIZE);
        for (log_size, evaluation) in log_sizes.iter().zip(evaluations.iter()) {
            accumulator.increment_count(*log_size);
            for (index, evaluation) in evaluation.iter().enumerate() {
                accumulator.accumulate(*log_size, index, *evaluation);
            }
        }
        let accumulator_poly = accumulator.finalize();

        // Pick an arbitrary sample point.
        let point = CirclePoint::<QM31>::get_point(98989892);

        let accumulator_res = accumulator_poly.eval_at_point(point);

        // Use direct computation.
        let mut res = QM31::default();
        // Sort evaluations by log_size.
        let mut pairs = log_sizes.into_iter().zip(evaluations).collect::<Vec<_>>();
        pairs.sort_by_key(|(log_size, _)| *log_size);
        for (log_size, values) in pairs.into_iter() {
            res = res * alpha
                + CircleEvaluation::new(
                    CircleDomain::constraint_evaluation_domain(log_size),
                    values,
                )
                .interpolate()
                .eval_at_point(point);
        }

        assert_eq!(accumulator_res, res);
    }
}
