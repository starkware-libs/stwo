use std::iter::zip;

use num_traits::One;

use self::air::{FibonacciAir, MultiFibonacciAir};
use self::component::FibonacciComponent;
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{FieldExpOps, IntoSlice};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, verify, ProvingError, StarkProof, VerificationError};
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::hasher::Hasher;

pub mod air;
mod component;

pub struct Fibonacci {
    pub air: FibonacciAir,
}

impl Fibonacci {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        let component = FibonacciComponent::new(log_size, claim);
        Self {
            air: FibonacciAir::new(component),
        }
    }

    pub fn get_trace(&self) -> CpuCircleEvaluation<BaseField, BitReversedOrder> {
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

    pub fn prove(&self) -> Result<StarkProof, ProvingError> {
        let trace = self.get_trace();
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[self
            .air
            .component
            .claim])));
        prove(&self.air, channel, vec![trace])
    }

    pub fn verify(&self, proof: StarkProof) -> Result<(), VerificationError> {
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[self
            .air
            .component
            .claim])));
        verify(proof, &self.air, channel)
    }
}

pub struct MultiFibonacci {
    pub air: MultiFibonacciAir,
    log_sizes: Vec<u32>,
    claims: Vec<BaseField>,
}

impl MultiFibonacci {
    pub fn new(log_sizes: Vec<u32>, claims: Vec<BaseField>) -> Self {
        assert!(!log_sizes.is_empty());
        assert_eq!(log_sizes.len(), claims.len());
        let air = MultiFibonacciAir::new(&log_sizes, &claims);
        Self {
            air,
            log_sizes,
            claims,
        }
    }

    pub fn get_trace(&self) -> Vec<CpuCircleEvaluation<BaseField, BitReversedOrder>> {
        zip(&self.log_sizes, &self.claims)
            .map(|(log_size, claim)| {
                let fib = Fibonacci::new(*log_size, *claim);
                fib.get_trace()
            })
            .collect()
    }

    pub fn prove(&self) -> Result<StarkProof, ProvingError> {
        let channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&self.claims)));
        prove(&self.air, channel, self.get_trace())
    }

    pub fn verify(&self, proof: StarkProof) -> Result<(), VerificationError> {
        let channel =
            &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&self.claims)));
        verify(proof, &self.air, channel)
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::collections::BTreeMap;
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{Fibonacci, MultiFibonacci};
    use crate::core::air::accumulation::PointEvaluationAccumulator;
    use crate::core::air::{AirExt, AirProverExt, Component, ComponentTrace};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::prover::VerificationError;
    use crate::core::queries::Queries;
    use crate::core::utils::bit_reverse;
    use crate::core::InteractionElements;
    use crate::{m31, qm31};

    pub fn generate_test_queries(n_queries: usize, trace_length: usize) -> Vec<usize> {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut queries: Vec<usize> = (0..n_queries)
            .map(|_| rng.gen_range(0..trace_length))
            .collect();
        queries.sort();
        queries.dedup();
        queries
    }

    #[test]
    fn test_composition_polynomial_is_low_degree() {
        let fib = Fibonacci::new(5, m31!(443693538));
        let trace = fib.get_trace();
        let trace_poly = trace.interpolate();
        let trace_eval =
            trace_poly.evaluate(CanonicCoset::new(trace_poly.log_size() + 1).circle_domain());
        let trace = ComponentTrace::new(
            TreeVec::new(vec![vec![&trace_poly]]),
            TreeVec::new(vec![vec![&trace_eval]]),
        );

        let random_coeff = qm31!(2213980, 2213981, 2213982, 2213983);
        let component_traces = vec![trace];
        let composition_polynomial_poly = fib.air.compute_composition_polynomial(
            random_coeff,
            &component_traces,
            &InteractionElements::new(BTreeMap::new()),
        );

        // Evaluate this polynomial at another point out of the evaluation domain and compare to
        // what we expect.
        let point = CirclePoint::<SecureField>::get_point(98989892);

        let points = fib.air.mask_points(point);
        let mask_values = zip(&component_traces[0].polys[0], &points[0])
            .map(|(poly, points)| {
                points
                    .iter()
                    .map(|point| poly.eval_at_point(*point))
                    .collect_vec()
            })
            .collect_vec();

        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        fib.air.component.evaluate_constraint_quotients_at_point(
            point,
            &mask_values,
            &mut evaluation_accumulator,
            &InteractionElements::new(BTreeMap::new()),
        );
        let oods_value = evaluation_accumulator.finalize();

        assert_eq!(oods_value, composition_polynomial_poly.eval_at_point(point));
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
    fn test_fib_prove() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let proof = fib.prove().unwrap();
        fib.verify(proof).unwrap();
    }

    #[test]
    fn test_prove_invalid_trace_value() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove().unwrap();
        invalid_proof.commitment_scheme_proof.queried_values.0[0][0][3] += BaseField::one();

        let error = fib.verify(invalid_proof).unwrap_err();
        assert_matches!(error, VerificationError::Merkle(_));
    }

    #[test]
    fn test_prove_invalid_trace_oods_values() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove().unwrap();
        invalid_proof
            .commitment_scheme_proof
            .sampled_values
            .swap(0, 1);

        let error = fib.verify(invalid_proof).unwrap_err();
        assert_matches!(error, VerificationError::InvalidStructure(_));
        assert!(error
            .to_string()
            .contains("Unexpected sampled_values structure"));
    }

    #[test]
    fn test_prove_insufficient_trace_values() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

        let mut invalid_proof = fib.prove().unwrap();
        invalid_proof.commitment_scheme_proof.queried_values.0[0][0].pop();

        let error = fib.verify(invalid_proof).unwrap_err();
        assert_matches!(error, VerificationError::Merkle(_));
    }

    #[test]
    fn test_rectangular_multi_fibonacci() {
        let multi_fib = MultiFibonacci::new(vec![5; 16], vec![m31!(443693538); 16]);
        let proof = multi_fib.prove().unwrap();
        multi_fib.verify(proof).unwrap();
    }

    #[test]
    fn test_mixed_degree_multi_fibonacci() {
        let multi_fib = MultiFibonacci::new(
            // TODO(spapini): Change order of log_sizes.
            vec![3, 5, 7],
            vec![m31!(1056169651), m31!(443693538), m31!(722122436)],
        );
        let proof = multi_fib.prove().unwrap();
        multi_fib.verify(proof).unwrap();
    }
}
