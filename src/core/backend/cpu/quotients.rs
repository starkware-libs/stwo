use num_traits::Zero;

use super::CPUBackend;
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::quotients::{BatchedColumnOpenings, QuotientOps};
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
        openings: &[BatchedColumnOpenings],
    ) -> SecureColumn<Self> {
        let mut res = SecureColumn::zeros(domain.size());
        for row in 0..domain.size() {
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let row_accumlator =
                accumulate_row_quotients(openings, columns, row, random_coeff, domain_point);
            res.set(row, row_accumlator);
        }
        res
    }
}

pub fn accumulate_row_quotients(
    openings: &[BatchedColumnOpenings],
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
            let current_numerator =
                complex_conjugate_line(domain_point, value, opening.point, *open_value);
            numerator = numerator * random_coeff + current_numerator;
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
