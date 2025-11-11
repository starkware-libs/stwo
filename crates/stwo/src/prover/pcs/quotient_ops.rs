#![allow(warnings)]
use core::cmp::Reverse;
use std::collections::HashMap;
use std::iter::zip;

use itertools::Itertools;
use num_traits::One;
use tracing::{span, Level};

use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::quotients::{ColumnSampleBatch, PointSample};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::utils::bit_reverse_index;
use crate::prover::backend::{Backend, Col, ColumnOps, CpuBackend};
use crate::prover::poly::circle::{CircleEvaluation, PolyOps, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

pub trait QuotientOps: PolyOps {
    /// Accumulates the quotients of the columns at the given domain.
    /// For a column f(x), and a point sample (p,v), the quotient is
    ///   (f(x) - V0(x))/V1(x)
    /// where V0(p)=v, V0(conj(p))=conj(v), and V1 is a vanishing polynomial for p,conj(p).
    /// This ensures that if f(p)=v, then the quotient is a polynomial.
    /// The result is a linear combination of the quotients using powers of random_coeff.
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder>;

    fn accumulate_numerators(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        start_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
        a_accumulation_dict: &mut HashMap<CirclePoint<SecureField>, SecureField>,
    ) -> SecureEvaluation<Self, BitReversedOrder>;
}

#[allow(dead_code, unused_variables)]
pub fn compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    log_blowup_factor: u32,
) -> Vec<SecureEvaluation<B, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    let mut a_accumulation_dict = HashMap::<CirclePoint<SecureField>, SecureField>::default();
    let mut start_coeff = SecureField::one();
    let unlifted = zip(columns, samples)
        .sorted_by_key(|(c, _)| c.domain.log_size())
        .group_by(|(c, _)| c.domain.log_size())
        .into_iter()
        .map(|(log_size, tuples)| {
            let (columns, samples): (Vec<_>, Vec<_>) = tuples.unzip();
            let domain = CanonicCoset::new(log_size).circle_domain();
            // TODO: slice.
            let sample_batches = ColumnSampleBatch::new_vec(&samples);
            B::accumulate_numerators(
                domain,
                &columns,
                random_coeff,
                start_coeff,
                &sample_batches,
                log_blowup_factor,
                &mut a_accumulation_dict,
            )
        })
        .collect_vec();

    let mut curr_eval: Option<SecureEvaluation<B, BitReversedOrder>> = None;
    for mut col in unlifted.into_iter() {
        if let Some(prev_eval) = curr_eval {
            B::lift_and_accumulate(&mut col, &prev_eval);
        }
        curr_eval = Some(col);
    }

    // TODO(Leo): to modify. This assumes that there is only one OOD point.
    assert_eq!(a_accumulation_dict.keys().len(), 1);
    let (point, acc) = a_accumulation_dict.iter().next().unwrap();

    let mut curr_eval = curr_eval.unwrap();
    let max_log_size = curr_eval.len().ilog2();
    let domain = CanonicCoset::new(max_log_size).circle_domain();
    let bitrev_y_coords = (0..curr_eval.len())
        .map(|i| *acc * domain.at(bit_reverse_index(i, max_log_size)).y)
        .collect_vec();

    unimplemented!()
    // TODO(Leo): compute denoms and divide
    // vec![curr_eval]
}

#[allow(dead_code, unused_variables)]
pub fn _compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    log_blowup_factor: u32,
) -> Vec<SecureEvaluation<CpuBackend, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    let mut a_accumulation_dict = HashMap::<CirclePoint<SecureField>, SecureField>::default();
    let mut start_coeff = SecureField::one();
    let unlifted = zip(columns, samples)
        .sorted_by_key(|(c, _)| c.domain.log_size())
        .group_by(|(c, _)| c.domain.log_size())
        .into_iter()
        .map(|(log_size, tuples)| {
            dbg!(start_coeff);
            let (columns, samples): (Vec<_>, Vec<_>) = tuples.unzip();
            let domain = CanonicCoset::new(log_size).circle_domain();
            // TODO: slice.
            let sample_batches = ColumnSampleBatch::new_vec(&samples);
            let res = B::accumulate_numerators(
                domain,
                &columns,
                random_coeff,
                start_coeff,
                &sample_batches,
                log_blowup_factor,
                &mut a_accumulation_dict,
            );
            start_coeff *= random_coeff.pow(sample_batches.iter().fold(0u128, |acc, batch| {
                acc + batch.columns_and_values.len() as u128
            }));
            res
        })
        .collect_vec();

    let mut curr_eval: Option<SecureEvaluation<B, BitReversedOrder>> = None;
    for mut col in unlifted.into_iter() {
        if let Some(prev_eval) = curr_eval {
            B::lift_and_accumulate(&mut col, &prev_eval);
        }
        curr_eval = Some(col);
    }

    // TODO(Leo): to modify. This assumes that there is only one OOD point.
    assert_eq!(a_accumulation_dict.keys().len(), 1);
    let (point, acc) = a_accumulation_dict.iter().next().unwrap();

    let mut curr_eval = curr_eval.unwrap().to_cpu();
    let max_log_size = curr_eval.len().ilog2();
    let domain = CanonicCoset::new(max_log_size).circle_domain();
    let bitrev_y_coords = SecureColumnByCoords::from_iter(
        (0..curr_eval.len()).map(|i| -*acc * domain.at(bit_reverse_index(i, max_log_size)).y),
    );
    CpuBackend::accumulate(&mut curr_eval.values, &bitrev_y_coords);
    // TODO(Leo): compute denoms and divide
    vec![curr_eval]
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::{SecureField, QM31};
    use crate::core::pcs::quotients::{
        column_line_coeffs, denominator_inverses, ColumnSampleBatch, PointSample,
    };
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse_index;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::CpuBackend;
    use crate::prover::pcs::quotient_ops::{_compute_fri_quotients, compute_fri_quotients};
    use crate::prover::poly::circle::SecureEvaluation;
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::{m31, qm31};

    #[allow(unused_variables, dead_code)]
    #[test]
    fn test_quotients_are_correct() {
        let mut rng = SmallRng::seed_from_u64(0);
        const LOG_SIZE_SHORT: u32 = 2;
        const LOG_SIZE_LONG: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 1;

        let log_sizes: Vec<u32> = vec![LOG_SIZE_SHORT, LOG_SIZE_LONG];
        // Generate random polys.
        let polys: Vec<CpuCirclePoly> = log_sizes
            .iter()
            .map(|log_size| {
                CpuCirclePoly::new(
                    (0..(1 << *log_size))
                        .map(|_| M31::from(rng.gen::<u32>()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let evals = polys
            .iter()
            .map(|p| {
                p.evaluate(CanonicCoset::new(p.log_size() + LOG_BLOWUP_FACTOR).circle_domain())
            })
            .collect_vec();
        let alpha = qm31!(2, 0, 1, 0);
        let z = CirclePoint::<SecureField>::get_point(98989892);
        let max_log_size = log_sizes.last().unwrap() + LOG_BLOWUP_FACTOR;
        let lifted_samples = polys
            .iter()
            .zip(&evals)
            .map(|(p, e)| {
                let value = p.eval_at_point(z.repeated_double(max_log_size - e.domain.log_size()));
                vec![PointSample { point: z, value }]
            })
            .collect_vec();

        let mut expected: Vec<SecureField> = vec![QM31::zero(); 1 << max_log_size as usize];
        let max_domain = CanonicCoset::new(max_log_size).circle_domain();

        // Only test samples at a single OOD point.
        let sample_batches = ColumnSampleBatch::new_vec(&lifted_samples.iter().collect_vec());
        assert_eq!(sample_batches.len(), 1);

        // Compute the quotients in the most naive way possible.
        for (idx, val) in expected.iter_mut().enumerate() {
            let domain_point = max_domain.at(bit_reverse_index(idx, max_log_size));
            let line_coeffs = &column_line_coeffs(&sample_batches, alpha, SecureField::one())[0];

            // First poly.
            let (a, b, c) = line_coeffs[0];
            let poly = &polys[0];
            let num = c * poly.eval_at_point(
                domain_point
                    .repeated_double(LOG_SIZE_LONG - LOG_SIZE_SHORT)
                    .into_ef(),
            ) - b
                - a * domain_point.y;
            let den_inv = denominator_inverses(&sample_batches, domain_point)[0];
            let quotient0 = num.mul_cm31(CM31::one()); // TODO(Leo): put correct den

            // Second poly.
            let (a, b, c) = line_coeffs[1];
            let poly = &polys[1];
            let num = c * poly.eval_at_point(domain_point.repeated_double(0).into_ef())
                - b
                - a * domain_point.y;
            let den_inv = denominator_inverses(&sample_batches, domain_point)[0];
            let quotient1 = num.mul_cm31(CM31::one());

            *val = quotient0 + quotient1;
        }
        let expected = SecureEvaluation::<_, BitReversedOrder>::new(
            max_domain.clone(),
            SecureColumnByCoords::<CpuBackend>::from_iter(expected.into_iter()),
        );

        let actual = _compute_fri_quotients::<CpuBackend>(
            &evals.iter().collect_vec(),
            &lifted_samples,
            alpha,
            LOG_BLOWUP_FACTOR,
        );
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].columns.len(), expected.columns.len());
        assert_eq!(actual[0].columns, expected.columns);
    }

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
        let quot_eval = compute_fri_quotients(
            &[&eval],
            &[vec![PointSample { point, value }]],
            coeff,
            LOG_BLOWUP_FACTOR,
        )
        .pop()
        .unwrap();
        let quot_poly_base_field =
            CpuCircleEvaluation::new(eval_domain, quot_eval.values.columns[0].clone())
                .interpolate();
        assert!(quot_poly_base_field.is_in_fri_space(LOG_SIZE));
    }
}
