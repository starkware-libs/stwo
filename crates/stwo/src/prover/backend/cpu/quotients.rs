use std::iter::zip;

use itertools::Itertools;
use num_traits::Zero;

use super::CpuBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    accumulate_row_partial_numerators, accumulate_row_quotients, denominator_inverses_,
    quotient_constants, quotient_constants_, ColumnSampleBatch,
};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::utils::bit_reverse_index;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
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

    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        curr_coeff: &mut SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        let size = columns[0].values.len();
        let quotient_constants = quotient_constants_(sample_batches, random_coeff, curr_coeff);

        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            let mut liftable_numerators = unsafe { SecureColumnByCoords::uninitialized(size) };
            for row in 0..size {
                /////////// TODO(Leo): delete
                let query_values_at_row = columns.iter().map(|col| col[row]).collect_vec();
                let row_value =
                    accumulate_row_partial_numerators(batch, &query_values_at_row, &coeffs);
                liftable_numerators.set(row, row_value);
            }
            let linear_term: SecureField = coeffs.iter().map(|(a, ..)| a).sum();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                liftable_numerators,
                linear_term,
            })
        }
    }

    fn accumulate_denominators(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let log_size = accs
            .iter()
            .map(|x| x.liftable_numerators.len())
            .max()
            .unwrap()
            .ilog2();

        let domain = CanonicCoset::new(log_size).circle_domain();
        let mut res: SecureColumnByCoords<CpuBackend> = SecureColumnByCoords::zeros(1 << log_size);
        let sample_points = accs.iter().map(|x| x.sample_point).collect_vec();
        // Populate `res` with the quotients.
        for i in 0..res.len() {
            let domain_point = domain.at(bit_reverse_index(i, domain.log_size()));
            let inverses = denominator_inverses_(&sample_points, domain_point);
            let mut val = SecureField::zero();
            for (acc, den_inv) in accs.iter().zip_eq(inverses) {
                let mut local = SecureField::zero();
                let log_ratio = log_size - acc.liftable_numerators.len().ilog2();
                local += acc
                    .liftable_numerators
                    .at((i >> (log_ratio + 1) << 1) + (i & 1))
                    - acc.linear_term * domain_point.y;
                val += local.mul_cm31(den_inv)
            }
            res.set(i, val);
        }
        SecureEvaluation::new(domain, res)
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
