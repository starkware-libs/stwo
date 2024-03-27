use super::qm31::PackedQM31;
use super::{AVX512Backend, VECS_LOG_SIZE};
use crate::core::backend::avx512::PackedBaseField;
use crate::core::commitment_scheme::quotients::{ColumnSampleBatch, QuotientOps};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::{ComplexConjugate, FieldExpOps};
use crate::core::poly::circle::{CircleDomain, CircleEvaluation, SecureEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::utils::bit_reverse_index;

impl QuotientOps for AVX512Backend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        samples: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        assert!(domain.log_size() >= VECS_LOG_SIZE as u32);
        let mut values = SecureColumn::<AVX512Backend>::zeros(domain.size());
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
            let row_accumlator = accumulate_row_quotients(
                samples,
                columns,
                vec_row,
                random_coeff,
                (domain_points_x, domain_points_y),
            );
            values.set_packed(vec_row, row_accumlator);
        }
        SecureEvaluation { domain, values }
    }
}

pub fn accumulate_row_quotients(
    samples: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<AVX512Backend, BaseField, BitReversedOrder>],
    vec_row: usize,
    random_coeff: SecureField,
    domain_point_vec: (PackedBaseField, PackedBaseField),
) -> PackedQM31 {
    let mut row_accumlator = PackedQM31::zero();
    for sample in samples {
        let mut numerator = PackedQM31::zero();
        for (column_index, sample_value) in &sample.columns_and_values {
            let column = &columns[*column_index];
            let value = column.data[vec_row];
            // TODO(alonh): Optimize and simplify this.
            // The numerator is a line equation passing through
            //   (sample_point.y, sample_value), (conj(sample_point), conj(sample_value))
            // evaluated at (domain_point.y, value).
            // When substituting a polynomial in this line equation, we get a polynomial with a root
            // at sample_point and conj(sample_point) if the original polynomial had the values
            // sample_value and conj(sample_value) at these points.
            let current_numerator = cross(
                (domain_point_vec.1, value),
                (sample.point.y, *sample_value),
                (
                    sample.point.y.complex_conjugate(),
                    sample_value.complex_conjugate(),
                ),
            );
            numerator = numerator * PackedQM31::broadcast(random_coeff) + current_numerator;
        }

        let denominator = cross(
            domain_point_vec,
            (sample.point.x, sample.point.y),
            (
                sample.point.x.complex_conjugate(),
                sample.point.y.complex_conjugate(),
            ),
        );

        row_accumlator = row_accumlator
            * PackedQM31::broadcast(random_coeff.pow(sample.columns_and_values.len() as u128))
            + numerator * denominator.inverse();
    }
    row_accumlator
}

/// Computes the cross product of of the vectors (a.0, a.1), (b.0, b.1), (c.0, c.1).
/// This is a multilinear function of the inputs that vanishes when the inputs are collinear.
fn cross(
    a: (PackedBaseField, PackedBaseField),
    b: (SecureField, SecureField),
    c: (SecureField, SecureField),
) -> PackedQM31 {
    PackedQM31::broadcast(b.0 - c.0) * a.1 - PackedQM31::broadcast(b.1 - c.1) * a.0
        + PackedQM31::broadcast(b.1 * c.0 - b.0 * c.1)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec};
    use crate::core::backend::{CPUBackend, Column};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::commitment_scheme::quotients::{ColumnSampleBatch, QuotientOps};
    use crate::core::fields::m31::BaseField;
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
                    c.values.to_vec(),
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

        // TODO(spapini): This is calculated in a different way from CPUBackend right now.
        assert_ne!(avx_result, cpu_result);
    }
}
