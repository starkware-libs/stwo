use super::qm31::PackedQM31;
use super::{AVX512Backend, VECS_LOG_SIZE};
use crate::core::backend::avx512::PackedBaseField;
use crate::core::commitment_scheme::quotients::{BatchedColumnOpenings, QuotientOps};
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
        openings: &[BatchedColumnOpenings],
    ) -> SecureEvaluation<Self> {
        assert!(domain.log_size() >= VECS_LOG_SIZE as u32);
        let mut values = SecureColumn::<AVX512Backend>::zeros(domain.size());
        // TODO(spapini): bit reverse iterator.
        for vec_row in 0..(1 << (domain.log_size() - VECS_LOG_SIZE as u32)) {
            // TODO(spapini): Optimized this, for the small number of columns case.
            let points = std::array::from_fn(|i| {
                domain.at(bit_reverse_index(
                    (vec_row << VECS_LOG_SIZE) + i,
                    domain.log_size(),
                ))
            });
            let domain_points_x = PackedBaseField::from_array(points.map(|p| p.x));
            let domain_points_y = PackedBaseField::from_array(points.map(|p| p.y));
            let row_accumlator = accumulate_row_quotients(
                openings,
                columns,
                vec_row,
                random_coeff,
                (domain_points_x, domain_points_y),
            );
            values.set(vec_row, row_accumlator);
        }
        SecureEvaluation { domain, values }
    }
}

pub fn accumulate_row_quotients(
    openings: &[BatchedColumnOpenings],
    columns: &[&CircleEvaluation<AVX512Backend, BaseField, BitReversedOrder>],
    vec_row: usize,
    random_coeff: SecureField,
    domain_point_vec: (PackedBaseField, PackedBaseField),
) -> PackedQM31 {
    let mut row_accumlator = PackedQM31::zero();
    for opening in openings {
        let mut numerator = PackedQM31::zero();
        for (column_index, open_value) in &opening.column_indices_and_values {
            let column = &columns[*column_index];
            let value = column.data[vec_row];
            let current_numerator = cross(
                (domain_point_vec.1, value),
                (opening.point.y, *open_value),
                (
                    opening.point.y.complex_conjugate(),
                    open_value.complex_conjugate(),
                ),
            );
            numerator = numerator * PackedQM31::broadcast(random_coeff) + current_numerator;
        }

        let denominator = cross(
            domain_point_vec,
            (opening.point.x, opening.point.y),
            (
                opening.point.x.complex_conjugate(),
                opening.point.y.complex_conjugate(),
            ),
        );

        row_accumlator = row_accumlator
            * PackedQM31::broadcast(
                random_coeff.pow(opening.column_indices_and_values.len() as u128),
            )
            + numerator * denominator.inverse();
    }
    row_accumlator
}

fn cross(
    c: (PackedBaseField, PackedBaseField),
    b: (SecureField, SecureField),
    a: (SecureField, SecureField),
) -> PackedQM31 {
    PackedQM31::broadcast(b.0 - a.0) * c.1 - PackedQM31::broadcast(b.1 - a.1) * c.0
        + PackedQM31::broadcast(b.1 * a.0 - b.0 * a.1)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::avx512::{AVX512Backend, BaseFieldVec};
    use crate::core::backend::{CPUBackend, Column};
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::commitment_scheme::quotients::{BatchedColumnOpenings, QuotientOps};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
    use crate::core::poly::BitReversedOrder;

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
        let random_coeff = SecureField::from_m31_array(std::array::from_fn(BaseField::from));
        let a = SecureField::from_m31_array(std::array::from_fn(|i| BaseField::from(3 * i)));
        let b = SecureField::from_m31_array(std::array::from_fn(|i| BaseField::from(4 * i)));
        let openings = vec![BatchedColumnOpenings {
            point: SECURE_FIELD_CIRCLE_GEN,
            column_indices_and_values: vec![(0, a), (0, b)],
        }];
        let avx_result = AVX512Backend::accumulate_quotients(
            domain,
            &columns.iter().collect_vec(),
            random_coeff,
            &openings,
        )
        .values
        .to_cpu();

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
            &openings,
        );

        assert_eq!(avx_result, cpu_result.values.to_cpu());
    }
}
