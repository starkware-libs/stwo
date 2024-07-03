mod round;
#[allow(unused)]
mod scheduler;

use std::ops::{Add, AddAssign, Mul, Sub};

use round::{BlakeRoundComponent, BlakeTraceGenerator};

use crate::core::air::{
    Air, AirProver, AirTraceVerifier, AirTraceWriter, Component, ComponentProver,
};
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements};

pub struct BlakeAir {
    pub component: BlakeRoundComponent,
}

impl Air for BlakeAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }
}

impl AirTraceVerifier for BlakeAir {
    fn interaction_elements(&self, _channel: &mut Blake2sChannel) -> InteractionElements {
        InteractionElements::default()
    }
}

impl AirTraceWriter<SimdBackend> for BlakeAir {
    fn interact(
        &self,
        _trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn to_air_prover(&self) -> &impl AirProver<SimdBackend> {
        self
    }
}

impl AirProver<SimdBackend> for BlakeAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![&self.component]
    }
}

pub fn gen_trace(
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    BlakeTraceGenerator::gen_trace(log_size)
}

#[derive(Clone, Copy)]
struct Fu32<F>
where
    F: FieldExpOps
        + Copy
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    l: F,
    h: F,
}

#[cfg(test)]
mod tests {
    use tracing::{span, Level};

    use crate::core::backend::simd::SimdBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::prover::{prove, verify};
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::hasher::Hasher;
    use crate::examples::blake::round::{BlakeRoundComponent, BlakeTraceGenerator};
    use crate::examples::blake::BlakeAir;

    #[test_log::test]
    fn test_simd_blake_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_blake_prove -- --nocapture

        // Note: 15 means 208MB of trace.
        const LOG_N_ROWS: u32 = 18;
        let component = BlakeRoundComponent {
            log_size: LOG_N_ROWS,
        };
        let span = span!(Level::INFO, "Trace generation").entered();
        let trace = BlakeTraceGenerator::gen_trace(component.log_size);
        println!("trace length: {}", trace.len());
        span.exit();
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let air = BlakeAir { component };
        let proof = prove::<SimdBackend>(&air, channel, trace).unwrap();

        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        verify(proof, &air, channel).unwrap();
    }
}
