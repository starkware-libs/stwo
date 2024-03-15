use num_traits::Zero;

use super::CPUBackend;
use crate::core::air::accumulation::ColumnAccumulator;
use crate::core::backend::Col;
use crate::core::commitment_scheme::quotients::{BatchedColumnOpenings, QuotientOps};
use crate::core::constraints::pair_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{ComplexConjugate, FieldExpOps};
use crate::core::poly::circle::CircleDomain;
use crate::core::utils::bit_reverse_index;

impl QuotientOps for CPUBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        mut accum: ColumnAccumulator<'_, Self>,
        columns: &[Col<Self, BaseField>],
        random_coeff: SecureField,
        openings: &[BatchedColumnOpenings],
    ) {
        for row in 0..domain.size() {
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let mut row_accumlator = SecureField::zero();
            for opening in openings {
                let mut numerator = SecureField::zero();
                for (column_index, open_value) in &opening.column_indices_and_values {
                    let column = &columns[*column_index];
                    let value = column[row];
                    numerator = numerator * random_coeff + (value - *open_value);
                }

                let denominator = pair_vanishing(
                    opening.point,
                    opening.point.complex_conjugate(),
                    domain_point.into_ef(),
                );

                row_accumlator *= random_coeff.pow(opening.column_indices_and_values.len() as u128)
                    + numerator / denominator;
            }
            accum.accumulate(row, row_accumlator);
        }
    }
}
