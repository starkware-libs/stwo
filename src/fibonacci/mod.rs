use std::iter::zip;

use itertools::{enumerate, Itertools};
use num_traits::One;

use self::air::FibonacciAir;
use self::component::FibonacciComponent;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::hasher::Hasher;
use crate::core::air::AirExt;
use crate::core::backend::cpu::CPUCircleEvaluation;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::circle::CirclePoint;
use crate::core::commitment_scheme::CommitmentSchemeVerifier;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{FieldExpOps, IntoSlice};
use crate::core::fri::{FriConfig, FriVerifier, SparseCircleEvaluation};
use crate::core::oods::get_pair_oods_quotient;
use crate::core::poly::circle::{combine_secure_value, CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::proof_of_work::ProofOfWork;
use crate::core::prover::{
    prove, StarkProof, LOG_BLOWUP_FACTOR, LOG_LAST_LAYER_DEGREE_BOUND, N_QUERIES,
    PROOF_OF_WORK_BITS,
};

mod air;
mod component;

pub struct Fibonacci {
    pub air: FibonacciAir,
    pub trace_commitment_domain: CanonicCoset,
    pub composition_polynomial_commitment_domain: CanonicCoset,
    pub claim: BaseField,
}

impl Fibonacci {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        let trace_commitment_domain = CanonicCoset::new(log_size + LOG_BLOWUP_FACTOR);
        let composition_polynomial_commitment_domain =
            CanonicCoset::new(log_size + 1 + LOG_BLOWUP_FACTOR);
        let component = FibonacciComponent::new(log_size, claim);
        Self {
            air: FibonacciAir::new(component),
            trace_commitment_domain,
            composition_polynomial_commitment_domain,
            claim,
        }
    }

    fn get_trace(&self) -> CPUCircleEvaluation<BaseField, BitReversedOrder> {
        // Trace.
        let trace_domain = CanonicCoset::new(self.air.component.log_size);
        // TODO(AlonH): Consider using Vec::new instead of Vec::with_capacity throughout file.
        let mut trace = Vec::with_capacity(trace_domain.size());

        // Fill trace with fibonacci squared.
        let mut a = BaseField::one();
        let mut b = BaseField::one();
        for _ in 0..trace_domain.size() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(trace_domain, trace)
    }

    pub fn prove(&self) -> StarkProof {
        let trace = self.get_trace();
        let channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[self.claim])));
        prove(&self.air, channel, vec![trace])
    }
}

pub fn verify_proof<const N_BITS: u32>(proof: StarkProof, claim: BaseField) -> bool {
    let fib = Fibonacci::new(N_BITS, claim);
    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[claim])));
    let mut commitment_scheme = CommitmentSchemeVerifier::new();
    commitment_scheme.commit(proof.commitments[0], channel);
    let random_coeff = channel.draw_felt();
    commitment_scheme.commit(proof.commitments[1], channel);
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    let trace_oods_points = fib.air.mask_points(oods_point);

    let composition_polynomial_oods_value = fib.air.eval_composition_polynomial_at_point(
        oods_point,
        &proof.trace_oods_values,
        random_coeff,
    );
    assert_eq!(
        composition_polynomial_oods_value,
        combine_secure_value(proof.composition_polynomial_column_oods_values)
    );

    let bounds = fib.air.quotient_log_bounds();
    let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR, N_QUERIES);
    let mut fri_verifier =
        FriVerifier::commit(channel, fri_config, proof.fri_proof, bounds).unwrap();

    ProofOfWork::new(PROOF_OF_WORK_BITS).verify(channel, &proof.proof_of_work);
    let opening_positions = fri_verifier
        .column_opening_positions(channel)
        .into_values()
        .collect_vec();
    commitment_scheme.verify(&proof.decommitments, &opening_positions);

    // An evaluation for each mask item and one for the composition_polynomial.
    let mut sparse_circle_evaluations = Vec::with_capacity(trace_oods_points.len() + 1);
    for (opened_values, oods_value) in zip(
        &proof.opened_values[1],
        proof.composition_polynomial_column_oods_values,
    ) {
        let mut evaluation = Vec::with_capacity(opening_positions[1].len());
        let mut opened_values_iter = opened_values.iter();
        for sub_circle_domain in opening_positions[1].iter() {
            let values = (&mut opened_values_iter)
                .take(1 << sub_circle_domain.log_size)
                .copied()
                .collect();
            let sub_circle_evaluation = CircleEvaluation::new(
                sub_circle_domain.to_circle_domain(
                    &fib.composition_polynomial_commitment_domain.circle_domain(),
                ),
                values,
            );
            evaluation.push(
                get_pair_oods_quotient(oods_point, oods_value, &sub_circle_evaluation)
                    .bit_reverse(),
            );
        }
        assert!(
            opened_values_iter.next().is_none(),
            "Not all values were used."
        );
        sparse_circle_evaluations.push(SparseCircleEvaluation::new(evaluation));
    }
    for (component_points, component_values) in zip(&trace_oods_points, &proof.trace_oods_values) {
        for (i, (column_points, column_values)) in
            enumerate(zip(component_points, component_values))
        {
            for (oods_point, oods_value) in zip(column_points, column_values) {
                let mut evaluation = Vec::with_capacity(opening_positions[0].len());
                let mut opened_values = proof.opened_values[0][i].iter().copied();
                for sub_circle_domain in opening_positions[0].iter() {
                    let values = (&mut opened_values)
                        .take(1 << sub_circle_domain.log_size)
                        .collect();
                    let sub_circle_evaluation = CircleEvaluation::new(
                        sub_circle_domain
                            .to_circle_domain(&fib.trace_commitment_domain.circle_domain()),
                        values,
                    );
                    evaluation.push(
                        get_pair_oods_quotient(*oods_point, *oods_value, &sub_circle_evaluation)
                            .bit_reverse(),
                    );
                }
                assert!(opened_values.next().is_none(), "Not all values were used.");
                sparse_circle_evaluations.push(SparseCircleEvaluation::new(evaluation));
            }
        }
    }

    fri_verifier.decommit(sparse_circle_evaluations).unwrap();

    true
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;

    use super::Fibonacci;
    use crate::commitment_scheme::utils::tests::generate_test_queries;
    use crate::core::air::evaluation::PointEvaluationAccumulator;
    use crate::core::air::{AirExt, Component, ComponentTrace};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::queries::Queries;
    use crate::core::utils::bit_reverse;
    use crate::fibonacci::verify_proof;
    use crate::{m31, qm31};

    #[test]
    fn test_composition_polynomial_is_low_degree() {
        let fib = Fibonacci::new(5, m31!(443693538));
        let trace = fib.get_trace();
        let trace_poly = trace.interpolate();
        let trace = ComponentTrace::new(vec![&trace_poly]);

        // TODO(ShaharS), Change to a channel implementation to retrieve the random
        // coefficients from extension field.
        let random_coeff = qm31!(2213980, 2213981, 2213982, 2213983);
        let component_traces = vec![trace];
        let composition_polynomial_poly = fib
            .air
            .compute_composition_polynomial(random_coeff, &component_traces);

        // Evaluate this polynomial at another point out of the evaluation domain and compare to
        // what we expect.
        let point = CirclePoint::<SecureField>::get_point(98989892);

        let (_, mask_values) = fib
            .air
            .component
            .mask_points_and_values(point, &component_traces[0]);
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(
            random_coeff,
            fib.air.max_constraint_log_degree_bound(),
        );
        fib.air.component.evaluate_quotients_by_mask(
            point,
            &mask_values,
            &mut evaluation_accumulator,
        );
        let oods_value = evaluation_accumulator.finalize();
        assert_eq!(oods_value, composition_polynomial_poly.eval_at_point(point));
    }

    #[test]
    fn test_oods_quotients_are_low_degree() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let proof = fib.prove();
        let (composition_polynomial_quotient, trace_quotients) = proof
            .additional_proof_data
            .oods_quotients
            .split_first()
            .unwrap();

        // Assert that the trace quotients are low degree.
        for quotient in trace_quotients.iter() {
            let interpolated_quotient_poly = quotient.clone().interpolate();
            assert!(interpolated_quotient_poly.is_in_fft_space(FIB_LOG_SIZE));
        }

        // Assert that the composition polynomial quotient is low degree.
        let interpolated_quotient_poly = composition_polynomial_quotient.clone().interpolate();
        assert!(interpolated_quotient_poly.is_in_fft_space(FIB_LOG_SIZE + 1));
    }

    #[test]
    fn test_sparse_circle_points() {
        let log_domain_size = 7;
        let domain = CanonicCoset::new(log_domain_size).circle_domain();
        let mut trace_commitment_points = domain.iter().collect_vec();
        bit_reverse(&mut trace_commitment_points);

        // Generate queries.
        let trace_queries = Queries {
            positions: generate_test_queries(7, 1 << log_domain_size),
            log_domain_size,
        };

        // Get the opening positions and points.
        const FRI_STEP_SIZE: u32 = 3;
        let trace_opening_positions = trace_queries.opening_positions(FRI_STEP_SIZE);
        let trace_decommitment_positions = trace_opening_positions.flatten();
        let mut trace_opened_points = trace_decommitment_positions
            .iter()
            .map(|p| trace_commitment_points[*p]);

        // Assert that we got the correct domain_points.
        for sub_circle_domain in trace_opening_positions.iter() {
            let points = (&mut trace_opened_points)
                .take(1 << sub_circle_domain.log_size)
                .collect_vec();
            let circle_domain = sub_circle_domain.to_circle_domain(&domain);
            // Bit reverse the domain points to match the order of the opened points.
            let mut domain_points = circle_domain.iter().collect_vec();
            bit_reverse(&mut domain_points);
            assert_eq!(points, domain_points);
        }
    }

    #[test]
    fn test_prove() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));
        let trace = fib.get_trace();
        let trace_poly = trace.interpolate();
        let trace = ComponentTrace::new(vec![&trace_poly]);

        let proof = fib.prove();
        let oods_point = proof.additional_proof_data.oods_point;

        let (_, mask_values) = fib.air.component.mask_points_and_values(oods_point, &trace);
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(
            proof
                .additional_proof_data
                .composition_polynomial_random_coeff,
            fib.air.max_constraint_log_degree_bound(),
        );
        fib.air.component.evaluate_quotients_by_mask(
            oods_point,
            &mask_values,
            &mut evaluation_accumulator,
        );
        let hz = evaluation_accumulator.finalize();

        assert_eq!(
            proof
                .additional_proof_data
                .composition_polynomial_oods_value,
            hz
        );
        assert!(verify_proof::<FIB_LOG_SIZE>(proof, fib.claim));
    }

    // TODO(AlonH): Check the correct error occurs after introducing errors instead of
    // #[should_panic].
    #[test]
    #[should_panic]
    fn test_prove_invalid_trace_value() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove();
        invalid_proof.opened_values.0[0][0][4] += BaseField::one();

        verify_proof::<FIB_LOG_SIZE>(invalid_proof, fib.claim);
    }

    // TODO(AlonH): Check the correct error occurs after introducing errors instead of
    // #[should_panic].
    #[test]
    #[should_panic]
    fn test_prove_invalid_trace_oods_values() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove();
        invalid_proof.trace_oods_values.swap(0, 1);

        verify_proof::<FIB_LOG_SIZE>(invalid_proof, fib.claim);
    }

    // TODO(AlonH): Check the correct error occurs after introducing errors instead of
    // #[should_panic].
    #[test]
    #[should_panic]
    fn test_prove_insufficient_trace_values() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove();
        invalid_proof.opened_values.0[0][0].pop();

        verify_proof::<FIB_LOG_SIZE>(invalid_proof, fib.claim);
    }
}
