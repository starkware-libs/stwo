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

        // Push the value to the trace.
        trace.push(self.air.component.value);

        // Fill trace with binary representation of value.
        let mut value_bits = self.air.component.value.0;
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
    use proptest::prelude::*;

    use super::RangeCheck;
    use crate::m31;

    const RANGE_CHECK_LOG_SIZE: u32 = 4;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 50, // Number of test cases to generate
            .. ProptestConfig::default()
        })]

        #[test]
        fn test_range_check_prove(valid_value in 0..32768_u32) {
            let range_check = RangeCheck::new(RANGE_CHECK_LOG_SIZE, m31!(valid_value));
            let proof = range_check.prove().unwrap();
            range_check.verify(proof).unwrap();
        }

        #[test]
        #[should_panic]
        fn test_range_check_prove_overflow(invalid_value in 32768..u32::MAX) {
            let range_check = RangeCheck::new(RANGE_CHECK_LOG_SIZE, m31!(invalid_value));
            let proof = range_check.prove().unwrap();
            range_check.verify(proof).unwrap();
        }
    }
}
