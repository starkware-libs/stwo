use num_traits::Zero;

use super::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::constraints::{complex_conjugate_line, pair_vanishing};
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
        samples: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        let mut values = SecureColumn::<CPUBackend>::zeros(domain.size());
        // TODO(spapini): bit reverse iterator.
        for row in 0..domain.size() {
            // TODO(alonh): Make an efficient bit reverse domain iterator, possibly for AVX backend.
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let row_value =
                accumulate_row_quotients(samples, columns, row, random_coeff, domain_point);
            values.set(row, row_value);
        }
        SecureEvaluation { domain, values }
    }
}

pub fn accumulate_row_quotients(
    samples: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<CPUBackend, BaseField, BitReversedOrder>],
    row: usize,
    random_coeff: SecureField,
    domain_point: CirclePoint<BaseField>,
) -> SecureField {
    let mut row_accumlator = SecureField::zero();
    for sample in samples {
        let mut numerator = SecureField::zero();
        for (column_index, sample_value) in &sample.column_indices_and_values {
            let column = &columns[*column_index];
            let value = column[row];
            let current_numerator =
                complex_conjugate_line(domain_point, value, sample.point, *sample_value);
            numerator = numerator * random_coeff + current_numerator;
        }

        let denominator = pair_vanishing(
            sample.point,
            sample.point.complex_conjugate(),
            domain_point.into_ef(),
        );

        row_accumlator = row_accumlator
            * random_coeff.pow(sample.column_indices_and_values.len() as u128)
            + numerator / denominator;
    }
    row_accumlator
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
                column_indices_and_values: vec![(0, value)],
            }],
        );
        let quot_poly_base_field =
            CPUCircleEvaluation::new(eval_domain, quot_eval.values.columns[0].clone())
                .interpolate();
        assert!(quot_poly_base_field.is_in_fft_space(LOG_SIZE));
    }
}
