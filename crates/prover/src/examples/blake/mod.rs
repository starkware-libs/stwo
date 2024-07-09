use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::u32x16;

use itertools::{chain, Itertools};
use round::BlakeRoundComponent;
use scheduler::BlakeSchedulerComponent;
use scheduler_gen::BlakeInput;
use tracing::{span, Level};

use crate::constraint_framework::constant_cols::gen_is_first;
use crate::constraint_framework::logup::LookupElements;
use crate::core::air::{Air, AirProver, Component, ComponentProver};
use crate::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::{Blake2sChannel, Channel};
use crate::core::fields::m31::BaseField;
use crate::core::fields::{FieldExpOps, IntoSlice};
use crate::core::pcs::CommitmentSchemeProver;
use crate::core::poly::circle::{CanonicCoset, PolyOps};
use crate::core::prover::{prove_without_commit, StarkProof, LOG_BLOWUP_FACTOR};
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs::hasher::Hasher;
use crate::core::InteractionElements;

// Blake3.
pub const N_ROUNDS: usize = 7;

mod round;
mod round_constraints;
mod round_gen;
mod scheduler;
mod scheduler_constraints;
mod scheduler_gen;

#[derive(Clone, Copy, Debug)]
struct Fu32<F>
where
    F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    l: F,
    h: F,
}
impl<F> Fu32<F>
where
    F: FieldExpOps
        + Copy
        + Debug
        + AddAssign<F>
        + Add<F, Output = F>
        + Sub<F, Output = F>
        + Mul<BaseField, Output = F>,
{
    fn to_felts(self) -> [F; 2] {
        [self.l, self.h]
    }
}

pub struct BlakeAir {
    pub scheduler_component: BlakeSchedulerComponent,
    pub round_component: BlakeRoundComponent,
}

impl Air for BlakeAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.scheduler_component, &self.round_component]
    }
}

impl AirProver<SimdBackend> for BlakeAir {
    fn prover_components(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![&self.scheduler_component, &self.round_component]
    }
}

fn to_felts(x: u32x16) -> [PackedBaseField; 2] {
    [
        unsafe { PackedBaseField::from_simd_unchecked(x & u32x16::splat(0xffff)) },
        unsafe { PackedBaseField::from_simd_unchecked(x >> 16) },
    ]
}

#[allow(unused)]
pub fn prove_blake(log_size: u32) -> (BlakeAir, StarkProof) {
    assert!(log_size >= LOG_N_LANES);

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 3 + 1 + LOG_BLOWUP_FACTOR)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Prepare inputs.
    let blake_inputs = (0..(1 << (log_size - LOG_N_LANES)))
        .map(|i| {
            let v = [u32x16::from_array(std::array::from_fn(|j| (i + 2 * j) as u32)); 16];
            let m = [u32x16::from_array(std::array::from_fn(|j| (i + 2 * j + 1) as u32)); 16];
            BlakeInput { v, m }
        })
        .collect::<Vec<_>>();

    // Setup protocol.
    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    let commitment_scheme = &mut CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR);

    // Trace.
    let span = span!(Level::INFO, "Scheduler Generation").entered();
    let (scheduler_trace, scheduler_lookup_data, round_inputs) =
        scheduler_gen::gen_trace(log_size, &blake_inputs);

    // Prepare xor multiplicities.
    let mut xor_mults = vec![0, 1 << 24];

    // TODO: Use a cascade of 3 round components.
    span.exit();
    let span = span!(Level::INFO, "Round Generation").entered();
    let (round_trace, round_lookup_data) =
        round_gen::gen_trace(log_size + 3, &round_inputs, &mut xor_mults);
    let n_padded_rounds = (1 << (log_size + 3 - LOG_N_LANES)) - round_trace.len();
    let round_trace0 = round_trace.clone();

    span.exit();
    let span = span!(Level::INFO, "Trace Commitment").entered();
    commitment_scheme.commit_on_evals(
        chain![scheduler_trace, round_trace].collect_vec(),
        channel,
        &twiddles,
    );

    // Draw lookup element.
    let blake_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
    let round_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
    let xor_lookup_elements = LookupElements::draw(channel, 3);

    // Interaction trace.
    span.exit();
    let span = span!(Level::INFO, "Scheduler Interaction Generation").entered();
    let (scheduler_trace, scheduler_claimed_sum) = scheduler_gen::gen_interaction_trace(
        log_size,
        scheduler_lookup_data,
        &round_lookup_elements,
        &blake_lookup_elements,
    );

    span.exit();
    let span = span!(Level::INFO, "Round Interaction Generation").entered();
    let (round_trace, round_claimed_sum) = round_gen::gen_interaction_trace(
        log_size + 3,
        round_lookup_data,
        &xor_lookup_elements,
        &round_lookup_elements,
    );
    let round_trace1 = round_trace.clone();

    span.exit();
    let span = span!(Level::INFO, "Interaction Commitment").entered();
    commitment_scheme.commit_on_evals(
        chain![scheduler_trace, round_trace].collect_vec(),
        channel,
        &twiddles,
    );

    // Constant trace.
    span.exit();
    let span = span!(Level::INFO, "Constant Trace Generation").entered();
    commitment_scheme.commit_on_evals(
        vec![gen_is_first(log_size), gen_is_first(log_size + 3)],
        channel,
        &twiddles,
    );
    span.exit();

    // // Sanity check.
    // let scheduler_traces = TreeVec::new(vec![
    //     scheduler_trace0,
    //     scheduler_trace1,
    //     vec![gen_is_first(log_size)],
    // ]);
    // let scheduler_trace_polys =
    //     scheduler_traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

    // assert_constraints(
    //     &scheduler_trace_polys,
    //     CanonicCoset::new(log_size),
    //     |mut eval| {
    //         let [is_first] = eval.next_interaction_mask(2, [0]);
    //         BlakeSchedulerEval {
    //             eval,
    //             blake_lookup_elements,
    //             round_lookup_elements,
    //             logup: LogupAtRow::new(1, scheduler_claimed_sum, is_first),
    //         }
    //         .eval();
    //     },
    // );

    // let round_traces = TreeVec::new(vec![
    //     round_trace0,
    //     round_trace1,
    //     vec![gen_is_first(log_size + 3)],
    // ]);
    // let round_trace_polys =
    //     round_traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

    // assert_constraints(
    //     &round_trace_polys,
    //     CanonicCoset::new(log_size + 3),
    //     |mut eval| {
    //         let [is_first] = eval.next_interaction_mask(2, [0]);
    //         BlakeRoundEval {
    //             eval,
    //             xor_lookup_elements,
    //             logup: LogupAtRow::new(1, round_claimed_sum, is_first),
    //         }
    //         .eval();
    //     },
    // );

    // Prove constraints.
    let scheduler_component = BlakeSchedulerComponent {
        log_size,
        blake_lookup_elements,
        round_lookup_elements: round_lookup_elements.clone(),
        claimed_sum: scheduler_claimed_sum, // TODO: This is not correct.
    };
    let round_component = BlakeRoundComponent {
        log_size: log_size + 3,
        xor_lookup_elements,
        round_lookup_elements,
        claimed_sum: round_claimed_sum,
    };
    let air = BlakeAir {
        scheduler_component,
        round_component,
    };
    let proof = prove_without_commit::<SimdBackend>(
        &air,
        channel,
        &InteractionElements::default(),
        &twiddles,
        commitment_scheme,
    )
    .unwrap();

    (air, proof)
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::constraint_framework::logup::LookupElements;
    use crate::core::air::AirExt;
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::IntoSlice;
    use crate::core::pcs::CommitmentSchemeVerifier;
    use crate::core::prover::verify_without_commit;
    use crate::core::vcs::blake2_hash::Blake2sHasher;
    use crate::core::vcs::hasher::Hasher;
    use crate::core::InteractionElements;
    use crate::examples::blake::prove_blake;

    #[test_log::test]
    fn test_simd_blake_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f -C opt-level=3" cargo test
        //   test_simd_blake_prove -- --nocapture

        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();

        // Prove.
        let (air, proof) = prove_blake(log_n_instances);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
        let commitment_scheme = &mut CommitmentSchemeVerifier::new();

        // Decommit.
        let sizes = air.column_log_sizes();
        // Trace columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Draw lookup element.
        let _blake_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
        let _round_lookup_elements = LookupElements::draw(channel, 2 * 16 * 3);
        let xor_lookup_elements = LookupElements::draw(channel, 3);
        assert_eq!(xor_lookup_elements, air.round_component.xor_lookup_elements);
        // TODO(spapini): Check claimed sum against first and last instances.
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Constant columns.
        commitment_scheme.commit(
            proof.commitments[2],
            &[
                air.round_component.log_size - 3,
                air.round_component.log_size,
            ],
            channel,
        );

        verify_without_commit(
            &air,
            channel,
            &InteractionElements::default(),
            commitment_scheme,
            proof,
        )
        .unwrap();
    }
}
