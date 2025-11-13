use std::collections::HashMap;
use std::iter::zip;

use itertools::Itertools;
use num_traits::Zero;

use super::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    accumulate_row_partial_numerators, accumulate_row_partial_numerators_v2,
    accumulate_row_quotients, denominator_inverses_, quotient_constants, quotient_constants_,
    ColumnSampleBatch,
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

    /// Receives a **nonempty** slice of evaluations, all of the same size.
    /// This also needs to return the accumulated a's.
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        start_coeff: &mut SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
        a_accumulation_dict: &mut HashMap<CirclePoint<SecureField>, SecureField>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let size = columns[0].len();
        let mut values = unsafe { SecureColumnByCoords::uninitialized(size) };
        let quotient_constants = quotient_constants_(sample_batches, random_coeff, start_coeff);
        // TODO(Leo): for accumulate_numerators_v2: take the iteration in the a accumulation and
        // move it here.
        for row in 0..size {
            let query_values_at_row = columns.iter().map(|col| col[row]).collect_vec();
            let row_value = accumulate_row_partial_numerators(
                sample_batches,
                &query_values_at_row,
                &quotient_constants,
            );
            values.set(row, row_value);
        }
        // Compute the a accumulation.
        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            let val = a_accumulation_dict.entry(batch.point).or_default();
            *val += coeffs.iter().map(|(a, ..)| a).sum::<SecureField>();
        }

        SecureEvaluation::new(CanonicCoset::new(size.ilog2()).circle_domain(), values)
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

    #[allow(unused_variables)]
    fn accumulate_numerators_v2(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        curr_coeff: &mut SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        let size = columns[0].len();
        let quotient_constants = quotient_constants_(sample_batches, random_coeff, curr_coeff);

        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            let mut liftable_numerators = unsafe { SecureColumnByCoords::uninitialized(size) };
            for row in 0..size {
                let query_values_at_row = columns.iter().map(|col| col[row]).collect_vec();
                let row_value =
                    accumulate_row_partial_numerators_v2(batch, &query_values_at_row, &coeffs);
                liftable_numerators.set(row, row_value);
            }
            let linear_term = coeffs.iter().map(|(a, ..)| a).sum::<SecureField>();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                liftable_numerators,
                linear_term,
            })
        }
    }
    // TODO(Leo): maybe needs to receive log size?
    fn accumulate_denominators_v2(
        accs: Vec<AccumulatedNumerators<Self>>,
        log_size: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let mut res: SecureColumnByCoords<CpuBackend> = SecureColumnByCoords::zeros(1 << log_size);
        // let mut x_coords_inv = batch_inverse(&domain.iter().map(|point| point.x).collect_vec());
        // CpuBackend::bit_reverse_column(&mut x_coords_inv);
        let sample_points = accs.iter().map(|x| x.sample_point).collect_vec();
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
