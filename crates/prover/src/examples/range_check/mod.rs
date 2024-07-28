use air::RangeCheckAir;

use self::component::RangeCheckComponent;
use crate::core::backend::cpu::CpuCircleEvaluation;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
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
        let trace = Vec::with_capacity(trace_domain.size());

        // Fill trace with range_check.

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
