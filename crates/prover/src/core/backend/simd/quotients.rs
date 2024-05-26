use std::iter::zip;

use itertools::izip;
use num_traits::{One, Zero};

use super::column::SecureFieldVec;
use super::m31::{PackedBaseField, LOG_N_LANES, N_LANES};
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::backend::cpu::quotients::{batch_random_coeffs, QuotientConstants};
use crate::core::backend::{Col, Column};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::FieldOps;
use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

impl QuotientOps for SimdBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        assert!(domain.log_size() >= LOG_N_LANES);
        let mut values = SecureColumn::<Self>::zeros(domain.size());
        let quotient_constants = quotient_constants(sample_batches, random_coeff, domain);

        // TODO(spapini): bit reverse iterator.
        for vec_row in 0..1 << (domain.log_size() - LOG_N_LANES) {
            // TODO(spapini): Optimize this, for the small number of columns case.
            let points = std::array::from_fn(|i| {
                domain.at(bit_reverse_index(
                    (vec_row << LOG_N_LANES) + i,
                    domain.log_size(),
                ))
            });
            let domain_points_x = PackedBaseField::from_array(points.map(|p| p.x));
            let domain_points_y = PackedBaseField::from_array(points.map(|p| p.y));
            let row_accumulator = accumulate_row_quotients(
                sample_batches,
                columns,
                &quotient_constants,
                vec_row,
                (domain_points_x, domain_points_y),
            );
            unsafe { values.set_packed(vec_row, row_accumulator) };
        }
        SecureEvaluation { domain, values }
    }
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>],
    quotient_constants: &QuotientConstants<SimdBackend>,
    vec_row: usize,
    _domain_point_vec: (PackedBaseField, PackedBaseField),
) -> PackedSecureField {
    let mut row_accumulator = PackedSecureField::zero();
    for (sample_batch, batch_coeff, denominator_inverses) in izip!(
        sample_batches,
        &quotient_constants.batch_random_coeffs,
        &quotient_constants.denominator_inverses
    ) {
        let mut numerator = PackedSecureField::zero();
        for (column_index, sampled_value) in sample_batch.columns_and_values.iter() {
            let column = &columns[*column_index];
            let value = column.data[vec_row];
            numerator += PackedSecureField::broadcast(-*sampled_value) + value;
        }

        row_accumulator = row_accumulator * PackedSecureField::broadcast(*batch_coeff)
            + numerator * denominator_inverses.data[vec_row];
    }
    row_accumulator
}

/// Point vanishing for the packed representation of the points. skips the division.
/// See [crate::core::constraints::point_vanishing_fraction] for more details.
fn packed_point_vanishing_fraction(
    excluded: CirclePoint<SecureField>,
    p: (PackedBaseField, PackedBaseField),
) -> (PackedSecureField, PackedSecureField) {
    let e_conjugate = excluded.conjugate();
    let h_x = PackedSecureField::broadcast(e_conjugate.x) * p.0
        - PackedSecureField::broadcast(e_conjugate.y) * p.1;
    let h_y = PackedSecureField::broadcast(e_conjugate.y) * p.0
        + PackedSecureField::broadcast(e_conjugate.x) * p.1;
    (h_y, PackedSecureField::one() + h_x)
}

fn denominator_inverses(
    sample_batches: &[ColumnSampleBatch],
    domain: CircleDomain,
) -> Vec<Col<SimdBackend, SecureField>> {
    let mut numerators = Vec::new();
    let mut denominators = Vec::new();

    for sample_batch in sample_batches {
        for vec_row in 0..1 << (domain.log_size() - LOG_N_LANES) {
            // TODO(spapini): Optimize this, for the small number of columns case.
            let points = std::array::from_fn(|i| {
                domain.at(bit_reverse_index(
                    (vec_row << LOG_N_LANES) + i,
                    domain.log_size(),
                ))
            });
            let domain_points_x = PackedBaseField::from_array(points.map(|p| p.x));
            let domain_points_y = PackedBaseField::from_array(points.map(|p| p.y));
            let domain_point_vec = (domain_points_x, domain_points_y);
            let (denominator, numerator) =
                packed_point_vanishing_fraction(sample_batch.point, domain_point_vec);
            denominators.push(denominator);
            numerators.push(numerator);
        }
    }

    let denominators = SecureFieldVec {
        length: denominators.len() * N_LANES,
        data: denominators,
    };

    let numerators = SecureFieldVec {
        length: numerators.len() * N_LANES,
        data: numerators,
    };

    let mut flat_denominator_inverses = SecureFieldVec::zeros(denominators.len());
    <SimdBackend as FieldOps<SecureField>>::batch_inverse(
        &denominators,
        &mut flat_denominator_inverses,
    );

    zip(&mut flat_denominator_inverses.data, &numerators.data)
        .for_each(|(inv, denom_denom)| *inv *= *denom_denom);

    flat_denominator_inverses
        .data
        .chunks(domain.size() / N_LANES)
        .map(|denominator_inverses| denominator_inverses.iter().copied().collect())
        .collect()
}

fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
    domain: CircleDomain,
) -> QuotientConstants<SimdBackend> {
    let batch_random_coeffs = batch_random_coeffs(sample_batches, random_coeff);
    let denominator_inverses = denominator_inverses(sample_batches, domain);
    QuotientConstants {
        batch_random_coeffs,
        denominator_inverses,
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::simd::column::BaseFieldVec;
    use crate::core::backend::simd::SimdBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::BaseField;
    use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::qm31;

    #[test]
    fn test_accumulate_quotients() {
        const LOG_SIZE: u32 = 8;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let e0: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();
        let e1: BaseFieldVec = (0..domain.size()).map(|i| BaseField::from(2 * i)).collect();
        let columns = vec![
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, e0),
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, e1),
        ];
        let random_coeff = qm31!(1, 2, 3, 4);
        let a = qm31!(3, 6, 9, 12);
        let b = qm31!(4, 8, 12, 16);
        let samples = vec![ColumnSampleBatch {
            point: SECURE_FIELD_CIRCLE_GEN,
            columns_and_values: vec![(0, a), (1, b)],
        }];
        let cpu_columns = columns
            .iter()
            .map(|c| {
                CircleEvaluation::<CpuBackend, _, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            })
            .collect::<Vec<_>>();
        let cpu_result = CpuBackend::accumulate_quotients(
            domain,
            &cpu_columns.iter().collect_vec(),
            random_coeff,
            &samples,
        )
        .values
        .to_vec();

        let res = SimdBackend::accumulate_quotients(
            domain,
            &columns.iter().collect_vec(),
            random_coeff,
            &samples,
        )
        .values
        .to_vec();

        assert_eq!(res, cpu_result);
    }
}
