use std::iter::zip;

use itertools::{zip_eq, Itertools};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use super::column::CM31Column;
use super::SimdBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{quotient_constants, ColumnSampleBatch};
use crate::prover::backend::simd::m31::PackedBaseField;
use crate::prover::backend::simd::qm31::PackedSecureField;
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
        random_coeff: SecureField,
        curr_coeff_power: &mut SecureField,
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        let size = columns[0].length;
        let quotient_constants = quotient_constants(sample_batches, random_coeff, curr_coeff_power);

        for (batch, coeffs) in zip(sample_batches, quotient_constants.line_coeffs) {
            let mut partial_numerators_acc = unsafe { SecureColumnByCoords::uninitialized(size) };

            #[cfg(not(feature = "parallel"))]
            let iter = partial_numerators_acc.chunks_mut(1);

            // TODO(Leo): make chunk size configurable.
            #[cfg(feature = "parallel")]
            let iter = partial_numerators_acc.par_chunks_mut(1);

            iter.enumerate().for_each(|(chunk_idx, mut values_dst)| {
                let query_values_at_row = batch
                    .columns_and_values
                    .iter()
                    .map(|(idx, _)| columns[*idx].data[chunk_idx])
                    .collect_vec();
                let row_value = accumulate_row_partial_numerators(&query_values_at_row, &coeffs);
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

    #[allow(unused_variables)]
    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        unimplemented!()
    }
}

fn accumulate_row_partial_numerators(
    queried_values_at_row: &[PackedBaseField],
    coeffs: &Vec<(SecureField, SecureField, SecureField)>,
) -> PackedSecureField {
    let mut numerator = PackedSecureField::zero();
    for (val_at_row, (_, b, c)) in zip_eq(queried_values_at_row, coeffs) {
        let value = PackedSecureField::broadcast(*c) * *val_at_row;
        numerator += value - PackedSecureField::broadcast(*b);
    }
    numerator
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::quotients::{ColumnSampleBatch, PointSample};
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
        let sample_batches = ColumnSampleBatch::new_vec(&samples.iter().collect_vec());
        let random_coeff = qm31!(98, 76, 54, 32);
        // SIMD
        let mut curr_coeff_power = SecureField::one();
        let mut accumulated_numerators_vec_simd: Vec<AccumulatedNumerators<SimdBackend>> = vec![];
        let columns_simd: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
            (0..N_COLS).map(|_| columns.clone()).collect();

        SimdBackend::accumulate_numerators(
            &columns_simd.iter().collect_vec(),
            random_coeff,
            &mut curr_coeff_power,
            &sample_batches,
            &mut accumulated_numerators_vec_simd,
        );
        // CPU
        let mut curr_coeff_power = SecureField::one();
        let mut accumulated_numerators_vec_cpu: Vec<AccumulatedNumerators<CpuBackend>> = vec![];
        let columns_cpu: Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> =
            (0..N_COLS).map(|_| columns.to_cpu().clone()).collect();
        CpuBackend::accumulate_numerators(
            &columns_cpu.iter().collect_vec(),
            random_coeff,
            &mut curr_coeff_power,
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
