use itertools::zip_eq;
use num_traits::{One, Zero};

use super::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::quotients::{ColumnSampleBatch, PointSample, QuotientOps};
use crate::core::constraints::{complex_conjugate_line_coeffs, pair_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::{ComplexConjugate, FieldExpOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

impl QuotientOps for CPUBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        let mut values = SecureColumn::zeros(domain.size());
        let quotient_constants = quotient_constants(sample_batches, random_coeff);

        for row in 0..domain.size() {
            // TODO(alonh): Make an efficient bit reverse domain iterator, possibly for AVX backend.
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let row_value = accumulate_row_quotients(
                sample_batches,
                columns,
                &quotient_constants,
                row,
                random_coeff,
                domain_point,
            );
            values.set(row, row_value);
        }
        SecureEvaluation { domain, values }
    }
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<CPUBackend, BaseField, BitReversedOrder>],
    quotient_constants: &QuotientConstants,
    row: usize,
    random_coeff: SecureField,
    domain_point: CirclePoint<BaseField>,
) -> SecureField {
    let mut row_accumulator = SecureField::zero();
    for (sample_batch, sample_constants) in zip_eq(sample_batches, &quotient_constants.line_coeffs)
    {
        let mut numerator = SecureField::zero();
        for ((column_index, _), (a, b, c)) in
            zip_eq(&sample_batch.columns_and_values, sample_constants)
        {
            let column = &columns[*column_index];
            let value = column[row] * *c;
            let linear_term = *a * domain_point.y + *b;
            numerator += value - linear_term;
        }

        let denominator = pair_vanishing(
            sample_batch.point,
            sample_batch.point.complex_conjugate(),
            domain_point.into_ef(),
        );

        row_accumulator = row_accumulator
            * random_coeff.pow(sample_batch.columns_and_values.len() as u128)
            + numerator / denominator;
    }
    row_accumulator
}

/// Precompute the complex conjugate line coefficients for each column in each sample batch.
/// Specifically, for the i-th (in a sample batch) column's numerator term
/// `alpha^i * (c * F(p) - (a * p.y + b))`, we precompute the constants `alpha^i * a`, alpha^i * `b`
/// and alpha^i * `c`.
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

pub fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
) -> QuotientConstants {
    let line_coeffs = column_line_coeffs(sample_batches, random_coeff);
    QuotientConstants { line_coeffs }
}

/// Holds the constant values used in the quotient evaluation.
pub struct QuotientConstants {
    /// The precomputed line coefficients for each quotient numerator term.
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
}

#[cfg(test)]
mod tests {
    use crate::core::backend::cpu::{CPUCircleEvaluation, CPUCirclePoly};
    use crate::core::backend::CPUBackend;
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::commitment_scheme::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::poly::circle::CanonicCoset;
    use crate::{m31, qm31};

    #[test]
    fn test_quotients_are_low_degree() {
        const LOG_SIZE: u32 = 7;
        let polynomial = CPUCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + 1).circle_domain();
        let eval = polynomial.evaluate(eval_domain);
        let point = SECURE_FIELD_CIRCLE_GEN;
        let value = polynomial.eval_at_point(point);
        let coeff = qm31!(1, 2, 3, 4);
        let quot_eval = CPUBackend::accumulate_quotients(
            eval_domain,
            &[&eval],
            coeff,
            &[ColumnSampleBatch {
                point,
                columns_and_values: vec![(0, value)],
            }],
        );
        let quot_poly_base_field =
            CPUCircleEvaluation::new(eval_domain, quot_eval.columns[0].clone()).interpolate();
        assert!(quot_poly_base_field.is_in_fft_space(LOG_SIZE));
    }
}
