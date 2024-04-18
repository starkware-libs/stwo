use itertools::{izip, Itertools};
use num_traits::One;

use super::qm31::PackedSecureField;
use super::{AVX512Backend, SecureFieldVec, K_BLOCK_SIZE, VECS_LOG_SIZE};
use crate::core::backend::avx512::PackedBaseField;
use crate::core::backend::cpu::quotients::{
    batch_random_coeffs, column_line_coeffs, QuotientConstants,
};
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

impl QuotientOps for AVX512Backend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        assert!(domain.log_size() >= VECS_LOG_SIZE as u32);
        let mut values = SecureColumn::<AVX512Backend>::zeros(domain.size());
        let quotient_constants = quotient_constants(sample_batches, random_coeff, domain);

        // TODO(spapini): bit reverse iterator.
        for vec_row in 0..(1 << (domain.log_size() - VECS_LOG_SIZE as u32)) {
            // TODO(spapini): Optimize this, for the small number of columns case.
            let points = std::array::from_fn(|i| {
                domain.at(bit_reverse_index(
                    (vec_row << VECS_LOG_SIZE) + i,
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

// TODO(Ohad): no longer using pair_vanishing, remove domain_point_vec and line_coeffs, or write a
// function that deals with quotients over pair_vanishing polynomials.
pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<AVX512Backend, BaseField, BitReversedOrder>],
    quotient_constants: &QuotientConstants<AVX512Backend>,
    vec_row: usize,
    _domain_point_vec: (PackedBaseField, PackedBaseField),
) -> PackedSecureField {
    let mut row_accumulator = PackedSecureField::zero();
    for (sample_batch, _, batch_coeff, denominator_inverses) in izip!(
        sample_batches,
        &quotient_constants.line_coeffs,
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

/// Pair vanishing for the packed representation of the points. See
/// [crate::core::constraints::pair_vanishing] for more details.
/// TODO: remove _ when in use.
fn _packed_pair_vanishing(
    excluded0: CirclePoint<SecureField>,
    excluded1: CirclePoint<SecureField>,
    packed_p: (PackedBaseField, PackedBaseField),
) -> PackedSecureField {
    PackedSecureField::broadcast(excluded0.y - excluded1.y) * packed_p.0
        + PackedSecureField::broadcast(excluded1.x - excluded0.x) * packed_p.1
        + PackedSecureField::broadcast(excluded0.x * excluded1.y - excluded0.y * excluded1.x)
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
    (h_y, (PackedSecureField::one() + h_x))
}

fn denominator_inverses(
    sample_batches: &[ColumnSampleBatch],
    domain: CircleDomain,
) -> Vec<Col<AVX512Backend, SecureField>> {
    let mut numerator_terms = Vec::with_capacity(sample_batches.len() * domain.size());
    let flat_denominators: SecureFieldVec = sample_batches
        .iter()
        .flat_map(|sample_batch| {
            (0..(1 << (domain.log_size() - VECS_LOG_SIZE as u32)))
                .map(|vec_row| {
                    // TODO(spapini): Optimize this, for the small number of columns case.
                    let points = std::array::from_fn(|i| {
                        domain.at(bit_reverse_index(
                            (vec_row << VECS_LOG_SIZE) + i,
                            domain.log_size(),
                        ))
                    });
                    let domain_points_x = PackedBaseField::from_array(points.map(|p| p.x));
                    let domain_points_y = PackedBaseField::from_array(points.map(|p| p.y));
                    let domain_point_vec = (domain_points_x, domain_points_y);
                    let (num, denom) =
                        packed_point_vanishing_fraction(sample_batch.point, domain_point_vec);
                    numerator_terms.push(denom);
                    num
                })
                .collect_vec()
        })
        .collect();

    let mut flat_denominator_inverses = SecureFieldVec::zeros(flat_denominators.len());
    <AVX512Backend as FieldOps<SecureField>>::batch_inverse(
        &flat_denominators,
        &mut flat_denominator_inverses,
    );

    flat_denominator_inverses
        .data
        .iter_mut()
        .zip(&numerator_terms)
        .for_each(|(inv, denom_denom)| *inv *= *denom_denom);

    flat_denominator_inverses
        .data
        .chunks(domain.size() / K_BLOCK_SIZE)
        .map(|denominator_inverses| denominator_inverses.iter().copied().collect())
        .collect()
}

fn quotient_constants(
    sample_batches: &[ColumnSampleBatch],
    random_coeff: SecureField,
    domain: CircleDomain,
) -> QuotientConstants<AVX512Backend> {
    let line_coeffs = column_line_coeffs(sample_batches, random_coeff);
    let batch_random_coeffs = batch_random_coeffs(sample_batches, random_coeff);
    let denominator_inverses = denominator_inverses(sample_batches, domain);
    QuotientConstants {
        line_coeffs,
        batch_random_coeffs,
        denominator_inverses,
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec};
    use crate::core::backend::{CPUBackend, Column};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::BaseField;
    use crate::core::pcs::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;
    use crate::qm31;

    #[test]
    fn test_avx_accumulate_quotients() {
        const LOG_SIZE: u32 = 8;
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let e0: BaseFieldVec = (0..domain.size()).map(BaseField::from).collect();
        let e1: BaseFieldVec = (0..domain.size()).map(|i| BaseField::from(2 * i)).collect();
        let columns = vec![
            CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(domain, e0),
            CircleEvaluation::<AVX512Backend, _, BitReversedOrder>::new(domain, e1),
        ];
        let random_coeff = qm31!(1, 2, 3, 4);
        let a = qm31!(3, 6, 9, 12);
        let b = qm31!(4, 8, 12, 16);
        let samples = vec![ColumnSampleBatch {
            point: SECURE_FIELD_CIRCLE_GEN,
            columns_and_values: vec![(0, a), (1, b)],
        }];
        let avx_result = AVX512Backend::accumulate_quotients(
            domain,
            &columns.iter().collect_vec(),
            random_coeff,
            &samples,
        )
        .values
        .to_vec();

        let cpu_columns = columns
            .iter()
            .map(|c| {
                CircleEvaluation::<CPUBackend, _, BitReversedOrder>::new(
                    c.domain,
                    c.values.to_cpu(),
                )
            })
            .collect::<Vec<_>>();

        let cpu_result = CPUBackend::accumulate_quotients(
            domain,
            &cpu_columns.iter().collect_vec(),
            random_coeff,
            &samples,
        )
        .values
        .to_vec();

        assert_eq!(avx_result, cpu_result);
    }
}
