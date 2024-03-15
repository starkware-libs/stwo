use num_traits::Zero;

use super::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::constraints::{complex_conjugate_line, pair_vanishing};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::{ComplexConjugate, FieldExpOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

impl QuotientOps for CPUBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        openings: &[ColumnSampleBatch],
    ) -> SecureColumn<Self> {
        let mut res = SecureColumn::zeros(domain.size());
        for row in 0..domain.size() {
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let row_value =
                accumulate_row_quotients(openings, columns, row, random_coeff, domain_point);
            res.set(row, row_value);
        }
        res
    }
}

pub fn accumulate_row_quotients(
    openings: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<CPUBackend, BaseField, BitReversedOrder>],
    row: usize,
    random_coeff: SecureField,
    domain_point: CirclePoint<BaseField>,
) -> SecureField {
    let mut row_accumlator = SecureField::zero();
    for opening in openings {
        let mut numerator = SecureField::zero();
        for (column_index, open_value) in &opening.column_indices_and_values {
            let column = &columns[*column_index];
            let value = column[row];
            let linear_term = complex_conjugate_line(opening.point, *open_value, domain_point);
            numerator = numerator * random_coeff + value - linear_term;
        }

        let denominator = pair_vanishing(
            opening.point,
            opening.point.complex_conjugate(),
            domain_point.into_ef(),
        );

        row_accumlator = row_accumlator
            * random_coeff.pow(opening.column_indices_and_values.len() as u128)
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
            CPUCircleEvaluation::new(eval_domain, quot_eval.columns[0].clone()).interpolate();
        assert!(quot_poly_base_field.is_in_fft_space(LOG_SIZE));
    }
}
