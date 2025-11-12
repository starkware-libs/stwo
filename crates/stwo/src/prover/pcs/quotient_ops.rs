use std::collections::{HashMap, HashSet};
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
use crate::prover::poly::circle::{CircleEvaluation, PolyOps, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
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

    fn accumulate_denominators(
        numerators: &mut SecureEvaluation<Self, BitReversedOrder>,
        log_blowup_factor: u32,
        a_accumulation_dict: &HashMap<CirclePoint<SecureField>, SecureField>,
    );
}

#[allow(dead_code, unused_variables)]
pub fn compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    log_blowup_factor: u32,
) -> Vec<SecureEvaluation<B, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    unimplemented!()
}

pub fn _compute_fri_quotients<B: QuotientOps + AccumulationOps>(
    columns: &[&CircleEvaluation<B, BaseField, BitReversedOrder>],
    samples: &[Vec<PointSample>],
    random_coeff: SecureField,
    log_blowup_factor: u32,
) -> SecureEvaluation<B, BitReversedOrder> {
    let _span = span!(Level::INFO, "Compute FRI quotients", class = "FRIQuotients").entered();
    // TODO(Leo): support multiple sample points.
    let mut sample_points = HashSet::new();
    samples.iter().flatten().for_each(|v| {
        sample_points.insert(v.point);
    });
    assert_eq!(sample_points.len(), 1);

    let mut a_accumulation_dict = HashMap::<CirclePoint<SecureField>, SecureField>::default();
    let mut start_coeff = SecureField::one();

    // Accumulate the numerators, for each domain log size.
    let unlifted = zip(columns, samples)
        .sorted_by_key(|(c, _)| c.domain.log_size())
        .group_by(|(c, _)| c.domain.log_size())
        .into_iter()
        .map(|(log_size, tuples)| {
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

    // Lift the numerators.
    let mut curr_eval: Option<SecureEvaluation<B, BitReversedOrder>> = None;
    for mut col in unlifted.into_iter() {
        if let Some(prev_eval) = curr_eval {
            B::lift_and_accumulate(&mut col, &prev_eval);
        }
        curr_eval = Some(col);
    }
    let mut curr_eval = curr_eval.unwrap();

    // Deal with the denominators.
    B::accumulate_denominators(&mut curr_eval, log_blowup_factor, &a_accumulation_dict);
    curr_eval
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;
    use num_traits::{One, Zero};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::circle::{CirclePoint, SECURE_FIELD_CIRCLE_GEN};
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::{SecureField, QM31};
    use crate::core::pcs::quotients::{
        column_line_coeffs, denominator_inverses, ColumnSampleBatch, PointSample,
    };
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse_index;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::CpuBackend;
    use crate::prover::pcs::quotient_ops::_compute_fri_quotients;
    use crate::prover::poly::circle::SecureEvaluation;
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;
    use crate::{m31, qm31};

    #[test]
    fn test_quotients_are_correct() {
        let mut rng = SmallRng::seed_from_u64(0);

        const LOG_SIZE_SHORT: u32 = 5;
        const LOG_SIZE_LONG: u32 = 8;
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

        let alpha = qm31!(2, 15, 1, 94);
        // Draw a sample point.
        let z = CirclePoint::<SecureField>::get_point(98989892);
        let max_log_size = log_sizes.last().unwrap() + LOG_BLOWUP_FACTOR;
        // TODO(Leo): test multiple sample points when supported.
        let lifted_samples = polys
            .iter()
            .zip(&evals)
            .map(|(p, e)| {
                let value = p.eval_at_point(z.repeated_double(max_log_size - e.domain.log_size()));
                vec![PointSample { point: z, value }]
            })
            .collect_vec();

        let mut expected: Vec<SecureField> = vec![QM31::zero(); 1 << max_log_size as usize];
        let domain = CanonicCoset::new(max_log_size).circle_domain();

        let sample_batches = ColumnSampleBatch::new_vec(&lifted_samples.iter().collect_vec());
        assert_eq!(sample_batches.len(), 1);

        // Compute the quotients in the most naive way possible.
        for (idx, val) in expected.iter_mut().enumerate() {
            let domain_point = domain.at(bit_reverse_index(idx, max_log_size));
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
            let num0 = num;

            // Second poly.
            let (a, b, c) = line_coeffs[1];
            let poly = &polys[1];
            let num = c * poly.eval_at_point(domain_point.repeated_double(0).into_ef())
                - b
                - a * domain_point.y;
            let num1 = num;

            // Deal with the denominator.
            let den_inv = denominator_inverses(&sample_batches, domain_point)[0];
            *val = (num0 + num1).mul_cm31(den_inv);
        }

        let expected = SecureEvaluation::<_, BitReversedOrder>::new(
            domain,
            SecureColumnByCoords::<CpuBackend>::from_iter(expected),
        );

        let actual = _compute_fri_quotients::<CpuBackend>(
            &evals.iter().collect_vec(),
            &lifted_samples,
            alpha,
            LOG_BLOWUP_FACTOR,
        );

        assert_eq!(actual.columns, expected.columns);
    }

    #[test]
    fn test_quotients_are_low_degree() {
        const LOG_SIZE: u32 = 3;
        const LOG_BLOWUP_FACTOR: u32 = 1;
        let polynomial = CpuCirclePoly::new((0..1 << LOG_SIZE).map(|i| m31!(i)).collect());
        let eval_domain = CanonicCoset::new(LOG_SIZE + LOG_BLOWUP_FACTOR).circle_domain();
        let eval = polynomial.evaluate(eval_domain);
        let point = SECURE_FIELD_CIRCLE_GEN;
        let value = polynomial.eval_at_point(point);
        let rand_coeff = qm31!(1, 2, 5, 9876);
        let quot_eval = _compute_fri_quotients(
            &[&eval],
            &[vec![PointSample { point, value }]],
            rand_coeff,
            LOG_BLOWUP_FACTOR,
        );
        let coeffs = quot_eval
            .values
            .columns
            .iter()
            .map(|c| CpuCircleEvaluation::new(eval_domain, c.clone()).interpolate())
            .collect_vec();
        assert!(coeffs.iter().all(|c| c.is_in_fri_space(LOG_SIZE)));
        ///////////////////////////////////////////////////////////////////
        // let config = PcsConfig::default();
        // // Precompute twiddles.
        // let twiddles = CpuBackend::precompute_twiddles(
        //     CanonicCoset::new(LOG_SIZE + 1 + LOG_BLOWUP_FACTOR)
        //         .circle_domain()
        //         .half_coset,
        // );

        // // Setup protocol.
        // let prover_channel = &mut Blake2sM31Channel::default();
        // let mut commitment_scheme =
        //     CommitmentSchemeProver::<CpuBackend, Blake2sM31MerkleChannel>::new(config,
        // &twiddles); FriProver::<CpuBackend,
        // Blake2sM31MerkleChannel>::commit(prover_channel, config.fri_config, &vec![quot_eval],
        // &twiddles);

        // let mut tree = commitment_scheme.tree_builder();
        // tree.extend_polys([polynomial]);
        // tree.commit(prover_channel);
        // commitment_scheme.prove_lifted_values(TreeVec::new(vec![vec![vec!
        // [SECURE_FIELD_CIRCLE_GEN]]]), prover_channel);
    }
}
