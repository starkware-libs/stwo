use std::iter::zip;

use itertools::Itertools;
use num_traits::Zero;

use super::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    accumulate_row_partial_numerators, denominator_inverses, quotient_constants, ColumnSampleBatch,
};
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

impl QuotientOps for CpuBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        let size = columns[0].len();
        let quotient_constants = quotient_constants(sample_batches);

        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            let mut partial_numerators_acc = unsafe { SecureColumnByCoords::uninitialized(size) };
            for row in 0..size {
                let query_values_at_row = columns.iter().map(|col| col[row]).collect_vec();
                let row_value =
                    accumulate_row_partial_numerators(batch, &query_values_at_row, &coeffs);
                partial_numerators_acc.set(row, row_value);
            }
            let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, ..)| a).sum();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc,
                first_linear_term_acc,
            })
        }
    }

    fn compute_quotients_and_combine(
        accumulations: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let max_log_size = accumulations
            .iter()
            .map(|x| x.partial_numerators_acc.len())
            .max()
            .unwrap()
            .ilog2();

        let domain = CanonicCoset::new(max_log_size).circle_domain();
        let mut quotients: SecureColumnByCoords<CpuBackend> =
            unsafe { SecureColumnByCoords::uninitialized(1 << max_log_size) };
        let sample_points: Vec<CirclePoint<SecureField>> =
            accumulations.iter().map(|x| x.sample_point).collect();
        // Populate `quotients`.
        for row in 0..quotients.len() {
            let domain_point = domain.at(bit_reverse_index(row, max_log_size));
            let inverses = denominator_inverses(&sample_points, domain_point);
            let mut quotient = SecureField::zero();
            for (acc, den_inv) in accumulations.iter().zip_eq(inverses) {
                let mut full_numerator = SecureField::zero();
                let log_ratio = max_log_size - acc.partial_numerators_acc.len().ilog2();
                let lifted_idx = (row >> (log_ratio + 1) << 1) + (row & 1);

                full_numerator += acc.partial_numerators_acc.at(lifted_idx)
                    - acc.first_linear_term_acc * domain_point.y;
                // Note that `den_inv` is an element of CM31 (see the docs and comments in the
                // function [`crates::core::pcs::quotients::denominator_inverses`]).
                quotient += full_numerator.mul_cm31(den_inv)
            }
            quotients.set(row, quotient);
        }
        SecureEvaluation::new(domain, quotients)
    }
}
