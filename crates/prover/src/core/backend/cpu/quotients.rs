use itertools::{izip, zip_eq, Itertools};
use num_traits::{One, Zero};

use super::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::constraints::complex_conjugate_line_coeffs;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumnByCoords;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::quotients::{ColumnSampleBatch, PointSample, QuotientOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

impl QuotientOps for CpuBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let mut values = unsafe { SecureColumnByCoords::uninitialized(domain.size()) };
        let quotient_constants = quotient_constants(sample_batches, random_coeff);

        for row in 0..domain.size() {
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let query_values_at_row = columns.iter().map(|col| col[row]).collect_vec();
            let row_value = accumulate_row_quotients(
                sample_batches,
                &query_values_at_row,
                &quotient_constants,
                domain_point,
            );
            values.set(row, row_value);
        }
        SecureEvaluation::new(domain, values)
    }
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    queried_values_at_row: &[BaseField],
    quotient_constants: &QuotientConstants,
    domain_point: CirclePoint<BaseField>,
) -> SecureField {
    let denominator_inverses = denominator_inverses(sample_batches, domain_point);
    let mut row_accumulator = SecureField::zero();
    for (sample_batch, line_coeffs, batch_coeff, denominator_inverse) in izip!(
        sample_batches,
        &quotient_constants.line_coeffs,
        &quotient_constants.batch_random_coeffs,
        denominator_inverses
    ) {
        let mut numerator = SecureField::zero();
        for ((column_index, _), (a, b, c)) in zip_eq(&sample_batch.columns_and_values, line_coeffs)
        {
            let value = queried_values_at_row[*column_index] * *c;
            // The numerator is a line equation passing through
            //   (sample_point.y, sample_value), (conj(sample_point), conj(sample_value))
            // evaluated at (domain_point.y, value).
            // When substituting a polynomial in this line equation, we get a polynomial with a root
            // at sample_point and conj(sample_point) if the original polynomial had the values
            // sample_value and conj(sample_value) at these points.
            let linear_term = *a * domain_point.y + *b;
            numerator += value - linear_term;
        }

        row_accumulator = row_accumulator * *batch_coeff + numerator.mul_cm31(denominator_inverse);
    }
    row_accumulator
}

/// Precomputes the complex conjugate line coefficients for each column in each sample batch.
///
/// For the `i`-th (in a sample batch) column's numerator term `alpha^i * (c * F(p) - (a * p.y +
/// b))`, we precompute and return the constants: (`alpha^i * a`, `alpha^i * b`, `alpha^i * c`).
pub fn column_line_coeffs(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> Vec<Vec<(SecureField, SecureField, SecureField)>> {
    sample_batches
        .iter()
        .map(|sample_batch| {
            let mut alpha = SecureField::one();
            sample_batch
                .columns_and_values
                .iter()
                .map(|(_, sampled_value)| {
                    alpha *= random_coeff;
                    let sample = PointSample {
                        point: sample_batch.point,
                        value: *sampled_value,
                    };
                    complex_conjugate_line_coeffs(&sample, alpha)
                })
                .collect()
        })
        .collect()
}

/// Precomputes the random coefficients used to linearly combine the batched quotients.
///
/// For each sample batch we compute random_coeff^(number of columns in the batch),
/// which is used to linearly combine the batch with the next one.
pub fn batch_random_coeffs(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> Vec<SecureField> {
    sample_batches
        .iter()
        .map(|sb| random_coeff.pow(sb.columns_and_values.len() as u128))
        .collect()
}

fn denominator_inverses(
    sample_batches: &[ColumnSampleBatch],
    domain_point: CirclePoint<M31>,
) -> Vec<CM31> {
    let mut denominators = Vec::new();

    // We want a P to be on a line that passes through a point Pr + uPi in QM31^2, and its conjugate
    // Pr - uPi. Thus, Pr - P is parallel to Pi. Or, (Pr - P).x * Pi.y - (Pr - P).y * Pi.x = 0.
    for sample_batch in sample_batches {
        // Extract Pr, Pi.
        let prx = sample_batch.point.x.0;
        let pry = sample_batch.point.y.0;
        let pix = sample_batch.point.x.1;
        let piy = sample_batch.point.y.1;
        denominators.push((prx - domain_point.x) * piy - (pry - domain_point.y) * pix);
    }

    CM31::batch_inverse(&denominators)
}

pub fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> QuotientConstants {
    QuotientConstants {
        line_coeffs: column_line_coeffs(sample_batches, random_coeff),
        batch_random_coeffs: batch_random_coeffs(sample_batches, random_coeff),
    }
}

/// Holds the precomputed constant values used in each quotient evaluation.
pub struct QuotientConstants {
    /// The line coefficients for each quotient numerator term. For more details see
    /// [self::column_line_coeffs].
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
    /// The random coefficients used to linearly combine the batched quotients For more details see
    /// [self::batch_random_coeffs].
    pub batch_random_coeffs: Vec<SecureField>,
}

#[cfg(test)]
mod tests {
    use crate::core::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::core::backend::CpuBackend;
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::poly::circle::CanonicCoset;
    use crate::{m31, qm31};

    #[test]
    fn test_quotients_are_low_degree() {
        const LOG_SIZE: u32 = 7;
        const LOG_BLOWUP_FACTOR: u32 = 1;
        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + 1).circle_domain();
        let eval = polynomial.evaluate(eval_domain);
        let point = SECURE_FIELD_CIRCLE_GEN;
        let value = polynomial.eval_at_point(point);
        let coeff = qm31!(1, 2, 3, 4);
        let quot_eval = CpuBackend::accumulate_quotients(
            eval_domain,
            &[&eval],
            coeff,
            &[ColumnSampleBatch {
                point,
                columns_and_values: vec![(0, value)],
            }],
            LOG_BLOWUP_FACTOR,
        );
        let quot_poly_base_field =
            CpuCircleEvaluation::new(eval_domain, quot_eval.columns[0].clone()).interpolate();
        assert!(quot_poly_base_field.is_in_fri_space(LOG_SIZE));
    }
}
