use air::RangeCheckAir;

use self::component::RangeCheckComponent;
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::IntoSlice;
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{ProvingError, StarkProof, VerificationError};
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::hasher::Hasher;
use crate::trace_generation::{commit_and_prove, commit_and_verify};

pub mod air;
mod component;

#[derive(Clone)]
pub struct RangeCheck {
    pub air: RangeCheckAir,
}

impl RangeCheck {
    pub fn new(log_size: u32, value: BaseField) -> Self {
        let component = RangeCheckComponent::new(log_size, value);
        Self {
            air: RangeCheckAir::new(component),
        }
    }

    pub fn get_trace(&self) -> CpuCircleEvaluation<BaseField, BitReversedOrder> {
        // Trace.
        let trace_domain = CanonicCoset::new(self.air.component.log_size);
        let mut trace = Vec::with_capacity(trace_domain.size());

        let mut value_bits = self.air.component.value.0;

        // Push the initial value to the trace.
        trace.push(self.air.component.value);

        // Fill trace with range_check.
        for _ in 0..15 {
            trace.push(M31::from(value_bits & 0x1));
            value_bits >>= 1;
        }

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(trace_domain, trace)
    }

    pub fn prove(&self) -> Result<StarkProof, ProvingError> {
        let trace = self.get_trace();
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[self
            .air
            .component
            .value])));
        commit_and_prove(&self.air, channel, vec![trace])
    }

    pub fn verify(&self, proof: StarkProof) -> Result<(), VerificationError> {
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[self
            .air
            .component
            .value])));
        commit_and_verify(proof, &self.air, channel)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;

    use super::RangeCheck;
    use crate::core::air::accumulation::PointEvaluationAccumulator;
    use crate::core::air::{AirExt, AirProverExt, Component, ComponentTrace};
    use crate::core::circle::CirclePoint;
    use crate::core::fields::qm31::SecureField;
    use crate::core::pcs::TreeVec;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::{InteractionElements, LookupValues};
    use crate::trace_generation::BASE_TRACE;
    use crate::{m31, qm31};

    #[test]
    fn test_composition_polynomial_is_low_degree() {
        let range_check = RangeCheck::new(4, m31!(5));
        let trace = range_check.get_trace();
        println!("{:?}", trace);
        let trace_poly = trace.interpolate();
        println!("{:?}", trace_poly);
        let trace_eval =
            trace_poly.evaluate(CanonicCoset::new(trace_poly.log_size() + 1).circle_domain());
        println!("{:?}", trace_eval);
        let trace = ComponentTrace::new(
            TreeVec::new(vec![vec![&trace_poly]]),
            TreeVec::new(vec![vec![&trace_eval]]),
        );

        let random_coeff = qm31!(2213980, 2213981, 2213982, 2213983);
        let component_traces = vec![trace];
        let composition_polynomial_poly = range_check.air.compute_composition_polynomial(
            random_coeff,
            &component_traces,
            &InteractionElements::default(),
            &LookupValues::default(),
        );

        // Evaluate this polynomial at another point out of the evaluation domain and compare to
        // what we expect.
        let point = CirclePoint::<SecureField>::get_point(98989892);

        let points = range_check.air.mask_points(point);
        let mask_values = zip(&component_traces[0].polys[BASE_TRACE], &points[0])
            .map(|(poly, points)| {
                points
                    .iter()
                    .map(|point| poly.eval_at_point(*point))
                    .collect_vec()
            })
            .collect_vec();

        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        range_check
            .air
            .component
            .evaluate_constraint_quotients_at_point(
                point,
                &TreeVec::new(vec![mask_values]),
                &mut evaluation_accumulator,
                &InteractionElements::default(),
                &LookupValues::default(),
            );
        let oods_value = evaluation_accumulator.finalize();

        assert_eq!(oods_value, composition_polynomial_poly.eval_at_point(point));
    }
    #[test]
    fn test_range_check_prove() {
        const RANGE_CHECK_LOG_SIZE: u32 = 4;
        let range_check = RangeCheck::new(RANGE_CHECK_LOG_SIZE, m31!(20));

        let proof = range_check.prove().unwrap();
        range_check.verify(proof).unwrap();
    }
}
