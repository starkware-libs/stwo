use std::iter::zip;

use itertools::Itertools;
use num_traits::Zero;

use super::CpuBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::cm31::CM31;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::quotients::{
    accumulate_row_partial_numerators, quotient_constants, ColumnSampleBatch,
};
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse_index;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

impl QuotientOps for CpuBackend {
    type DenominatorInverses = Vec<Vec<CM31>>;

    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        _twiddles: &TwiddleTree<Self>,
        _log_blowup_factor: u32,
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

    fn compute_denominator_inverses(
        sample_points: &[CirclePoint<SecureField>],
        lifting_log_size: u32,
    ) -> Self::DenominatorInverses {
        let domain = CanonicCoset::new(lifting_log_size).circle_domain();
        sample_points
            .iter()
            .map(|sample_point| {
                let prx = sample_point.x.0;
                let pry = sample_point.y.0;
                let pix = sample_point.x.1;
                let piy = sample_point.y.1;
                let denominators: Vec<CM31> = (0..1 << lifting_log_size)
                    .map(|row| {
                        let domain_point = domain.at(bit_reverse_index(row, lifting_log_size));
                        (prx - domain_point.x) * piy - (pry - domain_point.y) * pix
                    })
                    .collect();
                CM31::batch_inverse(&denominators)
            })
            .collect()
    }

    fn compute_quotients_and_combine(
        accumulations: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        denominator_inverses: Self::DenominatorInverses,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let domain = CanonicCoset::new(lifting_log_size).circle_domain();
        let mut quotients: SecureColumnByCoords<CpuBackend> =
            unsafe { SecureColumnByCoords::uninitialized(1 << lifting_log_size) };
        // Populate `quotients`.
        for row in 0..quotients.len() {
            let domain_point = domain.at(bit_reverse_index(row, lifting_log_size));
            let mut quotient = SecureField::zero();
            for (acc_idx, acc) in accumulations.iter().enumerate() {
                let mut full_numerator = SecureField::zero();
                let log_ratio = lifting_log_size - acc.partial_numerators_acc.len().ilog2();
                let lifted_idx = (row >> (log_ratio + 1) << 1) + (row & 1);

                full_numerator += acc.partial_numerators_acc.at(lifted_idx)
                    - acc.first_linear_term_acc * domain_point.y;
                // Note that `den_inv` is an element of CM31 (see the docs and comments in the
                // function [`crates::core::pcs::quotients::denominator_inverses`]).
                quotient += full_numerator.mul_cm31(denominator_inverses[acc_idx][row])
            }
            quotients.set(row, quotient);
        }
        SecureEvaluation::new(domain, quotients)
    }
}
