use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::QM31;
use crate::core::fields::Field;
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};

/// Accumulates evaluation of the composition polynomial on a single point.
pub struct PointEvaluationAccumulator {
    pub random_coeff: QM31,
    pub evaluations: Vec<QM31>,
    pub evaluation_counts: Vec<usize>,
}
impl PointEvaluationAccumulator {
    pub fn new(random_coeff: QM31, max_log_size: u32) -> Self {
        // TODO(spapini): Consider making all log_sizes usize.
        let max_log_size = max_log_size as usize;
        Self {
            random_coeff,
            evaluations: vec![QM31::default(); max_log_size + 1],
            evaluation_counts: vec![0; max_log_size + 1],
        }
    }

    pub fn accumulate(&mut self, log_size: u32, evaluation: QM31) {
        let accum = &mut self.evaluations[log_size as usize];
        *accum = *accum * self.random_coeff + evaluation;

        self.evaluation_counts[log_size as usize] += 1;
    }

    pub fn finalize(self) -> QM31 {
        self.evaluations
            .iter()
            .zip(self.evaluation_counts.iter())
            .fold(QM31::default(), |acc, (evaluation, count)| {
                acc * self.random_coeff.pow(*count as u128) + *evaluation
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
