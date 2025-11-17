use std::iter::zip;

use itertools::Itertools;
use num_traits::One;
use tracing::{span, Level};

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{ColumnSampleBatch, PointSample};
use crate::prover::backend::ColumnOps;
use crate::prover::poly::circle::{CircleEvaluation, PolyOps, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

pub trait QuotientOps: PolyOps {
    /// Receives a non-empty set of columns of the *same* size, and populates the vector
    /// `accumulated_numerators_vec` with their FRI numerators accumulations, across
    /// `sample_batches`.
    ///
    /// For each sample batch in sample_batches, accumulates the numerators of the columns involved
    /// in this batch and pushes an `AccumulatedNumerators` object into
    /// `accumulated_numerators_vec`.
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        // The random coefficient received by the pcs for accumulating the quotients.
        random_coeff: SecureField,
        // A pointer to the power of the random coefficient that must be used as the starting
        // power for the accumulation performed by this function.
        curr_coeff_power: &mut SecureField,
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    );

    /// Given a vector of `AccumulatedNumerators`, the function iterates over the points of the
    /// largest domain of the accumulated numerators, and:
    /// * for each sample point, computes the denominator of the quotient for that (domain point,
    ///   sample point).
    /// * multiplies it by the accumulated numerator for that domain point and sample point.
    /// * sums across sample points.
    fn accumulate_denominators(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder>;
}

/// Helper struct that keeps track of the accumulation of the numerators involved in the FRI
/// quotients.
pub struct AccumulatedNumerators<B: ColumnOps<BaseField>> {
    /// One of the sample points received by the pcs.
    pub sample_point: CirclePoint<SecureField>,
    /// Stores a circle evaluation of the form:
    ///     p -> ∑ α^{k_i} * (cᵢ * f̃ᵢ(p) - bᵢ)
    /// where
    /// * p ∈ canonic coset of log size = l (where l is the log size of the column).
    /// * i runs over some column indices of the trace.
    /// * α is the random coefficient for the accumulation.
    /// * k_i is the randomness exponent for column i and sample point `sample_point`.
    /// * f̃ᵢ is the lift of the trace poly fᵢ to log size l.
    /// * (bᵢ, cᵢ) are the `b` and `c` line coefficients for column i and sample point
    ///   `sample_point`.
    pub partial_numerators_acc: SecureColumnByCoords<B>,
    /// Stores an accumulation of the form
    ///      ∑ α^{k_i} * aᵢ
    /// where
    /// * i runs over some column indices of the trace.
    /// * α and k_i are as in the previous docstring for `partial_numerators_acc`.
    /// * aᵢ is the `a` line coefficient for column i and sample point `sample_point`.
    ///
    /// The index set of the summation is equal to the index set of the summation referred in the
    /// previous docstring for `partial_numerators_acc`.
    pub first_linear_term_acc: SecureField,
}

pub fn compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    _log_blowup_factor: u32,
) -> SecureEvaluation<B, BitReversedOrder> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();

    let mut accumulated_numerators_vec: Vec<AccumulatedNumerators<B>> = vec![];
    let mut curr_coeff_power = SecureField::one();

    // Populate `accumulated_numerators_vec`, per (log_size, sample_point). After this iteration,
    // `accumulated_numerators_vec` will have length equal to
    //
    //   ∑_k (# of distinct sample points per log size k).
    //
    zip(columns, samples)
        .sorted_by_key(|(c, _)| c.domain.log_size())
        .group_by(|(c, _)| c.domain.log_size())
        .into_iter()
        .for_each(|(_, tuples)| {
            let (columns, samples): (Vec<_>, Vec<_>) = tuples.unzip();
            // TODO: slice.
            let sample_batches = ColumnSampleBatch::new_vec(&samples);
            B::accumulate_numerators(
                &columns,
                random_coeff,
                &mut curr_coeff_power,
                &sample_batches,
                &mut accumulated_numerators_vec,
            )
        });

    // Group and accumulate the numerators per sample point: the accumulations (of different
    // lengths) get lifted and accumulated to a single vector. After this step, there is a single
    // accumulation per sample point.
    let accumulations_per_sample_point = accumulated_numerators_vec
        .into_iter()
        .sorted_by_key(|c| c.sample_point.x)
        .group_by(|c| c.sample_point)
        .into_iter()
        .map(|(sample_point, accumulations_per_log_size)| {
            let accumulations_per_log_size = accumulations_per_log_size.collect_vec();
            // Accumulate the `a` coefficients.
            let first_linear_term_acc: SecureField = accumulations_per_log_size
                .iter()
                .map(|x| x.first_linear_term_acc)
                .sum();
            // Lift and accumulate the partial numerators vectors.
            // `partial_numerators_acc` is already sorted increasingly by size as required by
            // `B::lift_and_accumulate`.
            let partial_numerators_acc = accumulations_per_log_size
                .into_iter()
                .map(|x| x.partial_numerators_acc)
                .collect_vec();
            let res = B::lift_and_accumulate(partial_numerators_acc).unwrap();

            AccumulatedNumerators {
                sample_point,
                partial_numerators_acc: res,
                first_linear_term_acc,
            }
        })
        .collect_vec();

    // Finally, compute the denominators and compute the lifted quotients.
    B::accumulate_denominators(accumulations_per_sample_point)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::M31;
    use crate::core::pcs::quotients::PointSample;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::pcs::quotient_ops::compute_fri_quotients;
    use crate::prover::SecureField;

    #[test]
    fn test_quotients_are_low_degree() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 4;

        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(M31::from).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let eval = polynomial.evaluate(eval_domain);

        let sample_points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = sample_points
            .into_iter()
            .map(|x| PointSample {
                point: x,
                value: polynomial.eval_at_point(x),
            })
            .collect_vec();
        let rand_coeff =
            SecureField::from_m31_array(std::array::from_fn(|_| M31::from(rng.gen::<u32>())));
        let quot_eval = compute_fri_quotients(&[&eval], &[samples], rand_coeff, LOG_BLOWUP_FACTOR);
        let mut coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.clone()).interpolate())
            .collect_vec();
        let zeros = coeffs[0].coeffs.split_off((1 << LOG_SIZE) - 1);

        assert!(zeros.iter().all(|c| c.is_zero()));
    }
}
