use std::iter::zip;

use num_traits::One;

use self::component::FibonacciComponent;
use crate::commitment_scheme::blake2_hash::Blake2sHasher;
use crate::commitment_scheme::hasher::Hasher;
use crate::commitment_scheme::merkle_decommitment::MerkleDecommitment;
use crate::core::air::evaluation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use crate::core::air::{Component, ComponentTrace, Mask, MaskItem};
use crate::core::channel::{Blake2sChannel, Channel as ChannelTrait};
use crate::core::circle::{CirclePoint, Coset};
use crate::core::commitment_scheme::{CommitmentSchemeProver, CommitmentSchemeVerifier};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::{Field, IntoSlice};
use crate::core::fri::{
    CirclePolyDegreeBound, FriConfig, FriProof, FriProver, FriVerifier, SparseCircleEvaluation,
};
use crate::core::oods::{get_oods_quotient, get_pair_oods_quotient};
use crate::core::poly::circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly};
use crate::core::poly::BitReversedOrder;
use crate::core::proof_of_work::{ProofOfWork, ProofOfWorkProof};
use crate::core::queries::Queries;

type Channel = Blake2sChannel;
type MerkleHasher = Blake2sHasher;

mod component;

const LOG_BLOWUP_FACTOR: u32 = 1;
// TODO(Andrew): Change to 0 once related bug is fixed.
const LOG_LAST_LAYER_DEGREE_BOUND: u32 = 1;
const PROOF_OF_WORK_BITS: u32 = 12;
const N_QUERIES: usize = 3;

pub struct Fibonacci {
    pub component: FibonacciComponent,
    pub trace_domain: CanonicCoset,
    pub trace_eval_domain: CircleDomain,
    pub trace_commitment_domain: CanonicCoset,
    pub constraint_zero_domain: Coset,
    pub composition_polynomial_eval_domain: CircleDomain,
    pub composition_polynomial_commitment_domain: CanonicCoset,
    pub claim: BaseField,
}

pub struct AdditionalProofData {
    pub composition_polynomial_oods_value: SecureField,
    pub composition_polynomial_random_coeff: SecureField,
    pub oods_point: CirclePoint<SecureField>,
    pub oods_quotients: Vec<CircleEvaluation<SecureField, BitReversedOrder>>,
}

pub struct FibonacciProof {
    pub public_input: BaseField,
    pub trace_commitments: Vec<<MerkleHasher as Hasher>::Hash>,
    pub trace_decommitments: Vec<MerkleDecommitment<BaseField, MerkleHasher>>,
    pub composition_polynomial_commitment: <MerkleHasher as Hasher>::Hash,
    pub composition_polynomial_decommitment: MerkleDecommitment<SecureField, MerkleHasher>,
    pub trace_oods_values: Vec<SecureField>,
    pub composition_polynomial_opened_values: Vec<SecureField>,
    pub trace_opened_values: Vec<BaseField>,
    pub proof_of_work: ProofOfWorkProof,
    pub fri_proof: FriProof<MerkleHasher>,
    pub additional_proof_data: AdditionalProofData,
}

impl Fibonacci {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        let trace_domain = CanonicCoset::new(log_size);
        let trace_eval_domain = trace_domain.evaluation_domain(log_size + 1);
        let trace_commitment_domain = CanonicCoset::new(log_size + LOG_BLOWUP_FACTOR);
        let constraint_zero_domain = Coset::subgroup(log_size);
        let composition_polynomial_eval_domain =
            CircleDomain::constraint_evaluation_domain(log_size + 1);
        let composition_polynomial_commitment_domain =
            CanonicCoset::new(log_size + 1 + LOG_BLOWUP_FACTOR);
        Self {
            component: FibonacciComponent { log_size, claim },
            trace_domain,
            trace_eval_domain,
            trace_commitment_domain,
            constraint_zero_domain,
            composition_polynomial_eval_domain,
            composition_polynomial_commitment_domain,
            claim,
        }
    }

    fn get_trace(&self) -> CircleEvaluation<BaseField> {
        // Trace.
        // TODO(AlonH): Consider usin Vec::new instead of Vec::with_capacity throughout file.
        let mut trace = Vec::with_capacity(self.trace_domain.size());

        // Fill trace with fibonacci squared.
        let mut a = BaseField::one();
        let mut b = BaseField::one();
        for _ in 0..self.trace_domain.size() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(self.trace_domain, trace)
    }

    pub fn get_mask(&self) -> Mask {
        Mask::new(
            (0..3)
                .map(|offset| MaskItem {
                    column_index: 0,
                    offset,
                })
                .collect(),
        )
    }

    /// Returns the composition polynomial evaluations using the trace and a random coefficient.
    fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        trace: &ComponentTrace<'_>,
    ) -> CirclePoly<SecureField> {
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.component.max_constraint_log_degree_bound(),
        );
        self.component
            .evaluate_constraint_quotients_on_domain(trace, &mut accumulator);
        accumulator.finalize()
    }

    pub fn prove(&self) -> FibonacciProof {
        let channel = &mut Channel::new(Blake2sHasher::hash(BaseField::into_slice(&[self.claim])));

        // Evaluate and commit on trace.
        let trace = self.get_trace();
        let trace_poly = trace.interpolate();
        let trace_commitment_scheme = CommitmentSchemeProver::new(
            vec![trace_poly],
            vec![self.trace_commitment_domain],
            channel,
        );

        // Evaluate and commit on composition polynomial.
        let random_coeff = channel.draw_random_secure_felts()[0];
        let component_trace = ComponentTrace::new(vec![&trace_commitment_scheme.polynomials[0]]);
        let composition_polynomial_poly =
            self.compute_composition_polynomial(random_coeff, &component_trace);
        let composition_polynomial_commitment_scheme = CommitmentSchemeProver::new(
            vec![composition_polynomial_poly],
            vec![self.composition_polynomial_commitment_domain],
            channel,
        );

        // Evaluate the trace mask and the composition polynomial on the OODS point.
        let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
        let mask = self.get_mask();
        let (trace_oods_points, trace_oods_values) = self
            .component
            .mask_values_at_point(oods_point, &component_trace);
        let composition_polynomial_oods_value =
            composition_polynomial_commitment_scheme.polynomials[0].eval_at_point(oods_point);

        // Calculate a quotient polynomial for each trace mask item and one for the composition
        // polynomial.
        let mut oods_quotients = Vec::with_capacity(mask.len() + 1);
        oods_quotients.push(
            get_oods_quotient(
                oods_point,
                composition_polynomial_oods_value,
                &composition_polynomial_commitment_scheme.evaluations[0],
            )
            .bit_reverse(),
        );
        for (point, value) in zip(&trace_oods_points, &trace_oods_values) {
            oods_quotients.push(
                get_pair_oods_quotient(*point, *value, &trace_commitment_scheme.evaluations[0])
                    .bit_reverse(),
            );
        }

        let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR);
        let fri_prover = FriProver::commit(channel, fri_config, &oods_quotients);

        let proof_of_work = ProofOfWork::new(PROOF_OF_WORK_BITS).prove(channel);
        // TODO(AlonH): Get opening positions from FRI.
        let composition_polynomial_queries = Queries::generate(
            channel,
            self.composition_polynomial_commitment_domain.log_size(),
            N_QUERIES,
        );
        let trace_queries = composition_polynomial_queries.fold(
            self.composition_polynomial_commitment_domain.log_size()
                - self.trace_commitment_domain.log_size(),
        );
        let fri_proof = fri_prover.decommit(&composition_polynomial_queries);

        const FRI_STEP_SIZE: u32 = 1;
        let composition_polynomial_decommitment_positions = composition_polynomial_queries
            .opening_positions(FRI_STEP_SIZE)
            .flatten();
        let trace_decommitment_positions = trace_queries.opening_positions(FRI_STEP_SIZE).flatten();

        // Decommit and get the values in the opening positions.
        let composition_polynomial_opened_values = composition_polynomial_decommitment_positions
            .iter()
            .map(|p| composition_polynomial_commitment_scheme.evaluations[0].values[*p])
            .collect();
        let trace_opened_values = trace_decommitment_positions
            .iter()
            .map(|p| trace_commitment_scheme.evaluations[0].values[*p])
            .collect();
        let composition_polynomial_decommitment = composition_polynomial_commitment_scheme
            .generate_decommitment(composition_polynomial_decommitment_positions);
        let trace_decommitment =
            trace_commitment_scheme.generate_decommitment(trace_decommitment_positions);

        FibonacciProof {
            public_input: self.claim,
            trace_commitments: vec![trace_commitment_scheme.root()],
            trace_decommitments: vec![trace_decommitment],
            composition_polynomial_commitment: composition_polynomial_commitment_scheme.root(),
            composition_polynomial_decommitment,
            trace_oods_values,
            composition_polynomial_opened_values,
            trace_opened_values,
            proof_of_work,
            fri_proof,
            additional_proof_data: AdditionalProofData {
                composition_polynomial_oods_value,
                composition_polynomial_random_coeff: random_coeff,
                oods_point,
                oods_quotients,
            },
        }
    }
}

pub fn verify_proof<const N_BITS: u32>(proof: FibonacciProof) -> bool {
    let fib = Fibonacci::new(N_BITS, proof.public_input);
    let channel = &mut Channel::new(Blake2sHasher::hash(BaseField::into_slice(&[
        proof.public_input
    ])));
    let trace_commitment_scheme =
        CommitmentSchemeVerifier::new(proof.trace_commitments[0], channel);
    let random_coeff = channel.draw_random_secure_felts()[0];
    let composition_polynomial_commitment_scheme =
        CommitmentSchemeVerifier::new(proof.composition_polynomial_commitment, channel);
    let oods_point = CirclePoint::<SecureField>::get_random_point(channel);
    let mask = fib.get_mask();
    let trace_oods_points = fib.component.mask_points(oods_point);

    let mut evaluation_accumulator = PointEvaluationAccumulator::new(
        random_coeff,
        fib.component.max_constraint_log_degree_bound(),
    );
    fib.component.evaluate_quotients_by_mask(
        oods_point,
        &proof.trace_oods_values,
        &mut evaluation_accumulator,
    );
    let composition_polynomial_oods_value = evaluation_accumulator.finalize();

    let fri_config = FriConfig::new(LOG_LAST_LAYER_DEGREE_BOUND, LOG_BLOWUP_FACTOR);
    // TODO(AlonH): Get bounds as public params of the proof.
    let bounds = vec![
        CirclePolyDegreeBound::new(fib.composition_polynomial_eval_domain.log_size()),
        CirclePolyDegreeBound::new(fib.trace_domain.log_size()),
        CirclePolyDegreeBound::new(fib.trace_domain.log_size()),
        CirclePolyDegreeBound::new(fib.trace_domain.log_size()),
    ];
    let fri_verifier = FriVerifier::commit(channel, fri_config, proof.fri_proof, bounds).unwrap();

    ProofOfWork::new(PROOF_OF_WORK_BITS).verify(channel, &proof.proof_of_work);
    let composition_polynomial_queries = Queries::generate(
        channel,
        fib.composition_polynomial_commitment_domain.log_size(),
        N_QUERIES,
    );
    let trace_queries = composition_polynomial_queries.fold(
        fib.composition_polynomial_commitment_domain.log_size()
            - fib.trace_commitment_domain.log_size(),
    );
    // TODO(AlonH): Get sub circle domains from FRI.
    const FRI_STEP_SIZE: u32 = 1;
    let composition_polynomial_opening_positions =
        composition_polynomial_queries.opening_positions(FRI_STEP_SIZE);
    let trace_opening_positions = trace_queries.opening_positions(FRI_STEP_SIZE);
    assert_eq!(
        trace_opening_positions.len(),
        proof.trace_opened_values.len() >> FRI_STEP_SIZE
    );
    assert_eq!(
        composition_polynomial_opening_positions.len(),
        proof.composition_polynomial_opened_values.len() >> FRI_STEP_SIZE
    );
    assert!(trace_commitment_scheme.verify(
        &proof.trace_decommitments[0],
        &trace_opening_positions.flatten()
    ));
    assert!(composition_polynomial_commitment_scheme.verify(
        &proof.composition_polynomial_decommitment,
        &composition_polynomial_opening_positions.flatten()
    ));

    // An evaluation for each mask item and one for the composition_polynomial.
    let mut sparse_circle_evaluations = Vec::with_capacity(mask.len() + 1);
    let mut evaluation = Vec::with_capacity(composition_polynomial_opening_positions.len());
    let mut opened_values = proof.composition_polynomial_opened_values.into_iter();
    for sub_circle_domain in composition_polynomial_opening_positions.iter() {
        let values = (&mut opened_values)
            .take(1 << sub_circle_domain.log_size)
            .collect();
        let sub_circle_evaluation = CircleEvaluation::new(
            sub_circle_domain
                .to_circle_domain(&fib.composition_polynomial_commitment_domain.circle_domain()),
            values,
        );
        evaluation.push(
            get_oods_quotient(
                oods_point,
                composition_polynomial_oods_value,
                &sub_circle_evaluation,
            )
            .bit_reverse(),
        );
    }
    assert!(opened_values.next().is_none(), "Not all values were used.");
    sparse_circle_evaluations.push(SparseCircleEvaluation::new(evaluation));
    for (oods_point, oods_value) in zip(&trace_oods_points, &proof.trace_oods_values) {
        let mut evaluation = Vec::with_capacity(trace_opening_positions.len());
        let mut opened_values = proof.trace_opened_values.iter().copied();
        for sub_circle_domain in trace_opening_positions.iter() {
            let values = (&mut opened_values)
                .take(1 << sub_circle_domain.log_size)
                .collect();
            let sub_circle_evaluation = CircleEvaluation::new(
                sub_circle_domain.to_circle_domain(&fib.trace_commitment_domain.circle_domain()),
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

    fri_verifier
        .decommit(&composition_polynomial_queries, sparse_circle_evaluations)
        .unwrap();

    true
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;

    use super::Fibonacci;
    use crate::commitment_scheme::utils::tests::generate_test_queries;
    use crate::core::air::evaluation::PointEvaluationAccumulator;
    use crate::core::air::{Component, ComponentTrace};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::{BaseField, M31};
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
        let composition_polynomial_poly = fib.compute_composition_polynomial(random_coeff, &trace);

        // Evaluate this polynomial at another point, out of trace_eval_domain and compare to what
        // we expect.
        let point = CirclePoint::<SecureField>::get_point(98989892);

        let (_, mask_values) = fib.component.mask_values_at_point(point, &trace);
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(
            random_coeff,
            fib.component.max_constraint_log_degree_bound(),
        );
        fib.component
            .evaluate_quotients_by_mask(point, &mask_values, &mut evaluation_accumulator);
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
            let interpolated_quotient_poly = quotient.clone().bit_reverse().interpolate();
            assert!(interpolated_quotient_poly.is_in_fft_space(FIB_LOG_SIZE));
        }

        // Assert that the composition polynomial quotient is low degree.
        let interpolated_quotient_poly = composition_polynomial_quotient
            .clone()
            .bit_reverse()
            .interpolate();
        assert!(interpolated_quotient_poly.is_in_fft_space(FIB_LOG_SIZE + 1));
    }

    #[test]
    fn test_sparse_circle_points() {
        let log_domain_size = 7;
        let domain = CanonicCoset::new(log_domain_size).circle_domain();
        let domain_points = domain.iter().collect_vec();
        let trace_commitment_points = bit_reverse(domain_points);

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
            let domain_points = bit_reverse(circle_domain.iter().collect_vec());
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

        let (_, mask_values) = fib.component.mask_values_at_point(oods_point, &trace);
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(
            proof
                .additional_proof_data
                .composition_polynomial_random_coeff,
            fib.component.max_constraint_log_degree_bound(),
        );
        fib.component.evaluate_quotients_by_mask(
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
        assert!(verify_proof::<FIB_LOG_SIZE>(proof));
    }

    // TODO(AlonH): Check the correct error occurs after introducing errors instead of
    // #[should_panic].
    #[test]
    #[should_panic]
    fn test_prove_invalid_trace_value() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove();
        invalid_proof.trace_opened_values[4] += BaseField::one();

        verify_proof::<FIB_LOG_SIZE>(invalid_proof);
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

        verify_proof::<FIB_LOG_SIZE>(invalid_proof);
    }

    // TODO(AlonH): Check the correct error occurs after introducing errors instead of
    // #[should_panic].
    #[test]
    #[should_panic]
    fn test_prove_insufficient_trace_values() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove();
        invalid_proof.trace_opened_values.pop();

        verify_proof::<FIB_LOG_SIZE>(invalid_proof);
    }
}
