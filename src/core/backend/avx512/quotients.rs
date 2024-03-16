use itertools::zip_eq;

use super::qm31::PackedQM31;
use super::{AVX512Backend, VECS_LOG_SIZE};
use crate::core::backend::avx512::PackedBaseField;
use crate::core::backend::cpu::quotients::column_constants;
use crate::core::circle::CirclePoint;
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
        sample_batches: &[ColumnSampleBatch],
    ) -> SecureEvaluation<Self> {
        assert!(domain.log_size() >= VECS_LOG_SIZE as u32);
        let mut values = SecureColumn::<AVX512Backend>::zeros(domain.size());
        let column_constants = column_constants(sample_batches, random_coeff);

        // TODO(spapini): bit reverse iterator.
        for pack_index in 0..(1 << (domain.log_size() - VECS_LOG_SIZE as u32)) {
            // TODO(spapini): Optimized this, for the small number of columns case.
            let points = std::array::from_fn(|i| {
                domain.at(bit_reverse_index(
                    (pack_index << VECS_LOG_SIZE) + i,
                    domain.log_size(),
                ))
            });
            let domain_points_x = PackedBaseField::from_array(points.map(|p| p.x));
            let domain_points_y = PackedBaseField::from_array(points.map(|p| p.y));
            let row_accumulator = accumulate_row_quotients(
                sample_batches,
                columns,
                &column_constants,
                pack_index,
                random_coeff,
                (domain_points_x, domain_points_y),
            );
            values.set_packed(pack_index, row_accumulator);
        }
        SecureEvaluation { domain, values }
    }
}

pub fn accumulate_row_quotients(
    sample_batches: &[ColumnSampleBatch],
    columns: &[&CircleEvaluation<AVX512Backend, BaseField, BitReversedOrder>],
    column_constants: &[Vec<(SecureField, SecureField, SecureField)>],
    vec_row: usize,
    random_coeff: SecureField,
    domain_point_vec: (PackedBaseField, PackedBaseField),
) -> PackedQM31 {
    let mut row_accumulator = PackedQM31::zero();
    for (sample_batch, sample_constants) in zip_eq(sample_batches, column_constants) {
        let mut numerator = PackedQM31::zero();
        for ((column_index, _), (a, b, c)) in
            zip_eq(&sample_batch.columns_and_values, sample_constants)
        {
            let column = &columns[*column_index];
            let value = PackedQM31::broadcast(*c) * column.data[vec_row];
            // The numerator is a line equation passing through
            //   (sample_point.y, sample_value), (conj(sample_point), conj(sample_value))
            // evaluated at (domain_point.y, value).
            // When substituting a polynomial in this line equation, we get a polynomial with a root
            // at sample_point and conj(sample_point) if the original polynomial had the values
            // sample_value and conj(sample_value) at these points.
            // TODO(AlonH): Use single point vanishing to save a multiplication.
            let linear_term =
                PackedQM31::broadcast(*a) * domain_point_vec.1 + PackedQM31::broadcast(*b);
            numerator += value - linear_term;
        }

        let denominator = packed_pair_vanishing(
            sample_batch.point,
            sample_batch.point.complex_conjugate(),
            domain_point_vec,
        );

        row_accumulator = row_accumulator
            * PackedQM31::broadcast(
                random_coeff.pow(sample_batch.columns_and_values.len() as u128),
            )
            + numerator * denominator.inverse();
    }
    row_accumulator
}

/// Pair vanishing for the packed representation of the points. See
/// [crate::core::constraints::pair_vanishing] for more details.
fn packed_pair_vanishing(
    excluded0: CirclePoint<SecureField>,
    excluded1: CirclePoint<SecureField>,
    packed_p: (PackedBaseField, PackedBaseField),
) -> PackedQM31 {
    PackedQM31::broadcast(excluded0.y - excluded1.y) * packed_p.0
        + PackedQM31::broadcast(excluded1.x - excluded0.x) * packed_p.1
        + PackedQM31::broadcast(excluded0.x * excluded1.y - excluded0.y * excluded1.x)
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
