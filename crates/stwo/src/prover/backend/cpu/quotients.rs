use std::collections::HashMap;
use std::iter::zip;

use itertools::Itertools;
use num_traits::Zero;

use super::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    accumulate_row_partial_numerators, accumulate_row_quotients, denominator_inverses_,
    quotient_constants, quotient_constants_, ColumnSampleBatch,
};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::utils::bit_reverse_index;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

impl QuotientOps for CpuBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let mut values = unsafe { SecureColumnByCoords::uninitialized(domain.size()) };
        let quotient_constants = quotient_constants(sample_batches, random_coeff);

        for row in 0..domain.size() {
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let query_values_at_row = columns.iter().map(|col| col[row]).collect_vec();
            let row_value = accumulate_row_quotients(
                sample_batches,
                &query_values_at_row,
                &quotient_constants,
                domain_point,
            );
            values.set(row, row_value);
        }
        SecureEvaluation::new(domain, values)
    }

    /// Receives a slice of evaluations, all of the same size.
    /// This also needs to return the accumulated a's.
    fn accumulate_numerators(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        start_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
        a_accumulation_dict: &mut HashMap<CirclePoint<SecureField>, SecureField>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let mut values = unsafe { SecureColumnByCoords::uninitialized(domain.size()) };
        let quotient_constants = quotient_constants_(sample_batches, random_coeff, start_coeff);

        for row in 0..domain.size() {
            let domain_point = domain.at(bit_reverse_index(row, domain.log_size()));
            let query_values_at_row = columns.iter().map(|col| col[row]).collect_vec();
            let row_value = accumulate_row_partial_numerators(
                sample_batches,
                &query_values_at_row,
                &quotient_constants,
                domain_point,
            );
            values.set(row, row_value);
        }
        // Compute the a accumulation.
        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            let val = a_accumulation_dict.entry(batch.point).or_default();
            *val += coeffs
                .iter()
                .fold(SecureField::zero(), |acc, (a, ..)| acc + *a);
        }

        SecureEvaluation::new(domain, values)
    }

    fn accumulate_denominators(
        numerators: &mut SecureEvaluation<Self, BitReversedOrder>,
        _log_blowup_factor: u32,
        a_accumulation_dict: &HashMap<CirclePoint<SecureField>, SecureField>,
    ) {
        // TODO(Leo): to modify. This assumes that there is only one OOD point.
        assert_eq!(a_accumulation_dict.keys().len(), 1);
        let (sample_point, acc) = a_accumulation_dict.iter().next().unwrap();

        let domain = CanonicCoset::new(numerators.len().ilog2()).circle_domain();
        for i in 0..domain.size() {
            let domain_point = domain.at(bit_reverse_index(i, domain.log_size()));
            let den_inv = denominator_inverses_(&[*sample_point], domain_point)[0];
            let res = numerators.values.at(i) - *acc * domain_point.y;
            numerators.values.set(i, res.mul_cm31(den_inv));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::pcs::quotients::ColumnSampleBatch;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::CpuBackend;
    use crate::prover::QuotientOps;
    use crate::{m31, qm31};

    #[test]
    fn test_quotients_are_low_degree() {
        const LOG_SIZE: u32 = 7;
        const LOG_BLOWUP_FACTOR: u32 = 1;
        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + 1).circle_domain();
        let eval = polynomial.evaluate(eval_domain);
        let point = SECURE_FIELD_CIRCLE_GEN;
        let value = polynomial.eval_at_point(point);
        let coeff = qm31!(1, 2, 3, 4);
        let quot_eval = CpuBackend::accumulate_quotients(
            eval_domain,
            &[&eval],
            coeff,
            &[ColumnSampleBatch {
                point,
                columns_and_values: vec![(0, value)],
            }],
            LOG_BLOWUP_FACTOR,
        );
        let quot_poly_base_field =
            CpuCircleEvaluation::new(eval_domain, quot_eval.columns[0].clone()).interpolate();
        assert!(quot_poly_base_field.is_in_fri_space(LOG_SIZE));
    }
}
