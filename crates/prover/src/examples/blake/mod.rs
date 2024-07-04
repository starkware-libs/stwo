mod eval;
mod lookup;
mod round;
mod round_constraints;
#[allow(unused)]
mod round_gen;
#[allow(unused)]
mod scheduler;

use std::ops::{Add, AddAssign, Mul, Sub};

use round::BlakeRoundComponent;

use crate::core::air::{
    Air, AirProver, AirTraceVerifier, AirTraceWriter, Component, ComponentProver,
};
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::CircleEvaluation;
use crate::core::poly::BitReversedOrder;
use crate::core::{ColumnVec, InteractionElements};

#[derive(Copy, Clone, Debug)]
pub struct LookupElements {
    pub z: SecureField,
    pub alpha: SecureField,
}

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
    use crate::core::backend::simd::SimdBackend;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::pcs::CommitmentSchemeProver;
    use crate::core::poly::circle::{CanonicCoset, PolyOps};
    use crate::core::prover::{generate_proof, LOG_BLOWUP_FACTOR};
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::hasher::Hasher;
    use crate::core::InteractionElements;
    use crate::examples::blake::round::{check_constraints_on_trace, BlakeRoundComponent};
    use crate::examples::blake::round_gen::{
        gen_interaction_trace, get_constant_trace, BlakeTraceGenerator,
    };
    use crate::examples::blake::{BlakeAir, LookupElements};

    #[test_log::test]
    fn test_simd_blake_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_blake_prove -- --nocapture

        // Note: 15 means 208MB of trace.
        const LOG_N_ROWS: u32 = 10;
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_ROWS + 1 + LOG_BLOWUP_FACTOR)
                .circle_domain()
                .half_coset,
        );

        // TODO: Add public input.
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let mut commitment_scheme = CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);

        // Generate trace.
        let (trace, lookup_exprs) = BlakeTraceGenerator::gen_trace(LOG_N_ROWS);
        let polys = trace
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(&twiddles))
            .collect();
        commitment_scheme.commit(polys, channel, &twiddles);

        // Draw lookup elements.
        let [z, alpha] = channel.draw_felts(2).try_into().unwrap();
        let lookup_elements = LookupElements { z, alpha };

        let (trace, claimed_xor_sums) =
            gen_interaction_trace(LOG_N_ROWS, lookup_exprs, lookup_elements);
        let polys = trace
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(&twiddles))
            .collect();
        commitment_scheme.commit(polys, channel, &twiddles);

        // Constant trace.
        let trace = get_constant_trace(LOG_N_ROWS);
        let polys = trace
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(&twiddles))
            .collect();
        commitment_scheme.commit(polys, channel, &twiddles);

        // Check constraints - sanity check.
        check_constraints_on_trace(
            LOG_N_ROWS,
            lookup_elements,
            &claimed_xor_sums,
            commitment_scheme.trees.as_ref().map(|t| &t.polynomials[..]),
        );

        // Prove constraints.
        let component = BlakeRoundComponent {
            log_size: LOG_N_ROWS,
            lookup_elements,
            claimed_xor_sums,
        };
        let air = BlakeAir { component };

        let _proof = generate_proof::<SimdBackend>(
            &air,
            channel,
            &InteractionElements::default(),
            &twiddles,
            &mut commitment_scheme,
        )
        .unwrap();
        // TODO: Send the statement:
        //   N_LOGS, claimed_xor_sums, lookup_elements.

        // TODO: Verify.
    }
}
