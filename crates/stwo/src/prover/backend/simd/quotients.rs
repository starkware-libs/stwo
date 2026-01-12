use std::iter::zip;

use itertools::{zip_eq, Itertools};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use super::column::CM31Column;
use super::domain::CircleDomainBitRevIterator;
use super::m31::PackedBaseField;
use super::qm31::PackedSecureField;
use super::SimdBackend;
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::quotients::{quotient_constants, ColumnSampleBatch, NumeratorData};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::prover::backend::simd::cm31::PackedCM31;
use crate::prover::backend::simd::utils::to_lifted_simd;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

pub struct QuotientConstants {
    pub line_coeffs: Vec<Vec<(SecureField, SecureField, SecureField)>>,
    pub denominator_inverses: Vec<CM31Column>,
}

impl QuotientOps for SimdBackend {
    // TODO(Leo): optimize.
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        let size = columns[0].length;
        let quotient_constants = quotient_constants(sample_batches);

        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            let mut partial_numerators_acc = unsafe { SecureColumnByCoords::uninitialized(size) };

            #[cfg(not(feature = "parallel"))]
            let iter = partial_numerators_acc.chunks_mut(1);

            // TODO(Leo): make chunk size configurable.
            #[cfg(feature = "parallel")]
            let iter = partial_numerators_acc.par_chunks_mut(1);

            iter.enumerate().for_each(|(chunk_idx, mut values_dst)| {
                let query_values_at_row = batch.cols_vals_randpows.iter().map(
                    |NumeratorData {
                         column_index: idx, ..
                     }| columns[*idx].data[chunk_idx],
                );
                let row_value = accumulate_row_partial_numerators(query_values_at_row, &coeffs);
                unsafe {
                    values_dst.set_packed(0, row_value);
                }
            });
            let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, ..)| a).sum();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc,
                first_linear_term_acc,
            })
        }
    }

    // TODO(Leo): optimize.
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
        let domain_points: Vec<CirclePoint<PackedBaseField>> =
            CircleDomainBitRevIterator::new(domain).collect();
        let mut quotients: SecureColumnByCoords<SimdBackend> =
            unsafe { SecureColumnByCoords::uninitialized(1 << max_log_size) };
        let sample_points: Vec<CirclePoint<SecureField>> =
            accumulations.iter().map(|x| x.sample_point).collect();
        let denominators_inverses = denominator_inverses(&sample_points, domain);

        // Populate `quotients`.
        // TODO(Leo): make chunk size configurable.
        #[cfg(not(feature = "parallel"))]
        let iter = quotients.chunks_mut(1).enumerate();

        #[cfg(feature = "parallel")]
        let iter = quotients.par_chunks_mut(1).enumerate();

        iter.for_each(|(domain_idx, mut value_dst)| {
            let mut quotient = PackedSecureField::zero();
            for (acc, den_inv) in accumulations.iter().zip_eq(denominators_inverses.iter()) {
                let mut full_numerator = PackedSecureField::zero();

                let log_ratio = max_log_size - acc.partial_numerators_acc.len().ilog2();
                let lifted_partial_numerator =
                    PackedSecureField::from_packed_m31s(std::array::from_fn(|j| {
                        let lifted_simd = to_lifted_simd(
                            acc.partial_numerators_acc.columns[j].data[domain_idx >> log_ratio]
                                .into_simd(),
                            log_ratio,
                            domain_idx,
                        );
                        unsafe { PackedBaseField::from_simd_unchecked(lifted_simd) }
                    }));

                full_numerator += lifted_partial_numerator
                    - PackedSecureField::broadcast(acc.first_linear_term_acc)
                        * domain_points[domain_idx].y;
                quotient += full_numerator * den_inv[domain_idx];
            }
            unsafe {
                value_dst.set_packed(0, quotient);
            }
        });
        SecureEvaluation::new(domain, quotients)
    }
}

fn accumulate_row_partial_numerators(
    queried_values_at_row: impl Iterator<Item = PackedBaseField>,
    coeffs: &Vec<(SecureField, SecureField, SecureField)>,
) -> PackedSecureField {
    let mut numerator = PackedSecureField::zero();
    for (val_at_row, (_, b, c)) in zip_eq(queried_values_at_row, coeffs) {
        let value = PackedSecureField::broadcast(*c) * val_at_row;
        numerator += value - PackedSecureField::broadcast(*b);
    }
    numerator
}

fn denominator_inverses(
    sample_points: &[CirclePoint<SecureField>],
    domain: CircleDomain,
) -> Vec<Vec<PackedCM31>> {
    let domain_points = CircleDomainBitRevIterator::new(domain);

    #[cfg(not(feature = "parallel"))]
    let (domain_points_iter, sample_points_iter) = (domain_points, sample_points.iter());
    #[cfg(feature = "parallel")]
    let (domain_points_iter, sample_points_iter) =
        (domain_points.par_iter(), sample_points.par_iter());

    sample_points_iter
        .map(|sample_point| {
            // Extract Pr, Pi.
            let prx = PackedCM31::broadcast(sample_point.x.0);
            let pry = PackedCM31::broadcast(sample_point.y.0);
            let pix = PackedCM31::broadcast(sample_point.x.1);
            let piy = PackedCM31::broadcast(sample_point.y.1);

            // The iter itself is cloned for each sample batch.
            let denominators = domain_points_iter
                .clone()
                .map(|points| (prx - points.x) * piy - (pry - points.y) * pix)
                .collect::<Vec<_>>();
            PackedCM31::batch_inverse(&denominators)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::quotients::{
        build_samples_with_randomness_and_periodicity, ColumnSampleBatch, PointSample,
    };
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::CpuBackend;
    use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
    use crate::prover::poly::circle::CircleEvaluation;
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::QuotientOps;
    use crate::qm31;

    #[test]
    fn test_simd_and_cpu_numerators_are_consistent() {
        const LOG_SIZE: u32 = 10;
        const N_COLS: usize = 100;
        let mut rng = SmallRng::seed_from_u64(0);
        let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
        let values = BaseColumn::from_cpu((0..1 << LOG_SIZE).map(BaseField::from).collect());
        let columns =
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, values);

        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let points = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let samples = (0..N_COLS)
            .zip(mask_structure.iter())
            .map(|(_, i)| {
                points
                    .into_iter()
                    .zip_eq([
                        SecureField::from(rng.gen::<u32>()),
                        SecureField::from(rng.gen::<u32>()),
                    ])
                    .take(*i)
                    .map(|(point, value)| PointSample { point, value })
                    .collect_vec()
            })
            .collect_vec();
        let random_coeff = qm31!(98, 76, 54, 32);
        let sample_batches = ColumnSampleBatch::new_vec(
            &build_samples_with_randomness_and_periodicity(
                &TreeVec(vec![samples]),
                vec![vec![LOG_SIZE; N_COLS].into_iter()],
                LOG_SIZE,
                random_coeff,
            )
            .iter()
            .flatten()
            .collect_vec(),
        );
        // SIMD
        let mut accumulated_numerators_vec_simd: Vec<AccumulatedNumerators<SimdBackend>> = vec![];
        let columns_simd: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
            (0..N_COLS).map(|_| columns.clone()).collect();

        SimdBackend::accumulate_numerators(
            &columns_simd.iter().collect_vec(),
            &sample_batches,
            &mut accumulated_numerators_vec_simd,
        );
        // CPU
        let mut accumulated_numerators_vec_cpu: Vec<AccumulatedNumerators<CpuBackend>> = vec![];
        let columns_cpu: Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> =
            (0..N_COLS).map(|_| columns.to_cpu().clone()).collect();
        CpuBackend::accumulate_numerators(
            &columns_cpu.iter().collect_vec(),
            &sample_batches,
            &mut accumulated_numerators_vec_cpu,
        );

        accumulated_numerators_vec_simd
            .iter()
            .zip_eq(accumulated_numerators_vec_cpu)
            .for_each(|(acc_simd, acc_cpu)| {
                assert_eq!(
                    acc_simd.first_linear_term_acc,
                    acc_cpu.first_linear_term_acc
                );
                assert_eq!(
                    acc_simd.partial_numerators_acc.to_cpu().columns,
                    acc_cpu.partial_numerators_acc.columns
                );
            });
    }
}
