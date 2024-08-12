use std::simd::u32x16;

use itertools::{chain, multiunzip, Itertools};
use tracing::{span, Level};

use super::round::BlakeRoundComponent;
use super::scheduler::BlakeSchedulerComponent;
use super::xor_table::XorTableComponent;
use crate::constraint_framework::constant_columns::gen_is_first;
use crate::core::air::{Air, AirProver, Component, ComponentProver};
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::channel::Channel;
use crate::core::pcs::CommitmentSchemeProver;
use crate::core::poly::circle::{CanonicCoset, PolyOps};
use crate::core::prover::{prove, StarkProof, LOG_BLOWUP_FACTOR};
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::InteractionElements;
use crate::examples::blake::round::RoundElements;
use crate::examples::blake::scheduler::{self, BlakeElements, BlakeInput};
use crate::examples::blake::{
    round, xor_table, BlakeXorElements, XorAccums, N_ROUNDS, ROUND_LOG_SPLIT,
};

pub struct BlakeAir {
    pub scheduler_component: BlakeSchedulerComponent,
    pub round_components: Vec<BlakeRoundComponent>,
    pub xor12: XorTableComponent<12, 4>,
    pub xor9: XorTableComponent<9, 2>,
    pub xor8: XorTableComponent<8, 2>,
    pub xor7: XorTableComponent<7, 2>,
    pub xor4: XorTableComponent<4, 0>,
}

impl Air for BlakeAir {
    fn components(&self) -> Vec<&dyn Component> {
        chain![
            [&self.scheduler_component as &dyn Component],
            self.round_components.iter().map(|c| c as &dyn Component),
            [
                &self.xor12 as &dyn Component,
                &self.xor9 as &dyn Component,
                &self.xor8 as &dyn Component,
                &self.xor7 as &dyn Component,
                &self.xor4 as &dyn Component,
            ]
        ]
        .collect()
    }
}

impl AirProver<SimdBackend> for BlakeAir {
    fn component_provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        chain![
            [&self.scheduler_component as &dyn ComponentProver<SimdBackend>],
            self.round_components
                .iter()
                .map(|c| c as &dyn ComponentProver<SimdBackend>),
            [
                &self.xor12 as &dyn ComponentProver<SimdBackend>,
                &self.xor9 as &dyn ComponentProver<SimdBackend>,
                &self.xor8 as &dyn ComponentProver<SimdBackend>,
                &self.xor7 as &dyn ComponentProver<SimdBackend>,
                &self.xor4 as &dyn ComponentProver<SimdBackend>,
            ]
        ]
        .collect()
    }
}

#[allow(unused)]
pub fn prove_blake<C, H>(log_size: u32) -> (BlakeAir, StarkProof<H>)
where
    SimdBackend: MerkleOps<H>,
    C: Channel,
    H: MerkleHasher<Hash = C::Digest>,
{
    assert!(log_size >= LOG_N_LANES);
    assert_eq!(
        ROUND_LOG_SPLIT.map(|x| (1 << x)).into_iter().sum::<u32>() as usize,
        N_ROUNDS
    );

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    const XOR_TABLE_MAX_LOG_SIZE: u32 = 16;
    let log_max_rows =
        (log_size + *ROUND_LOG_SPLIT.iter().max().unwrap()).max(XOR_TABLE_MAX_LOG_SIZE);
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_max_rows + 1 + LOG_BLOWUP_FACTOR)
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
        .collect_vec();

    // Setup protocol.
    let channel = &mut C::new(C::Digest::default());
    let commitment_scheme = &mut CommitmentSchemeProver::new(LOG_BLOWUP_FACTOR, &twiddles);

    let span = span!(Level::INFO, "Trace").entered();

    // Scheduler.
    let (scheduler_trace, scheduler_lookup_data, round_inputs) =
        scheduler::gen_trace(log_size, &blake_inputs);

    // Rounds.
    let mut xor_accums = XorAccums::default();
    let mut rest = &round_inputs[..];
    // Split round inputs to components, according to [ROUND_LOG_SPLIT].
    let (round_traces, round_lookup_datas): (Vec<_>, Vec<_>) =
        multiunzip(ROUND_LOG_SPLIT.map(|l| {
            let (cur_inputs, r) = rest.split_at(1 << (log_size - LOG_N_LANES + l));
            rest = r;
            round::generate_trace(log_size + l, cur_inputs, &mut xor_accums)
        }));

    // Xor tables.
    let (xor_trace12, xor_lookup_data12) = xor_table::generate_trace(xor_accums.xor12);
    let (xor_trace9, xor_lookup_data9) = xor_table::generate_trace(xor_accums.xor9);
    let (xor_trace8, xor_lookup_data8) = xor_table::generate_trace(xor_accums.xor8);
    let (xor_trace7, xor_lookup_data7) = xor_table::generate_trace(xor_accums.xor7);
    let (xor_trace4, xor_lookup_data4) = xor_table::generate_trace(xor_accums.xor4);

    // Trace commitment.
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(
        chain![
            scheduler_trace,
            round_traces.into_iter().flatten(),
            xor_trace12,
            xor_trace9,
            xor_trace8,
            xor_trace7,
            xor_trace4,
        ]
        .collect_vec(),
    );
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let blake_lookup_elements = BlakeElements::draw(channel);
    let round_lookup_elements = RoundElements::draw(channel);
    let xor_lookup_elements = BlakeXorElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (scheduler_trace, scheduler_claimed_sum) = scheduler::gen_interaction_trace(
        log_size,
        scheduler_lookup_data,
        &round_lookup_elements,
        &blake_lookup_elements,
    );

    let (round_traces, round_claimed_sums): (Vec<_>, Vec<_>) = multiunzip(
        ROUND_LOG_SPLIT
            .iter()
            .zip(round_lookup_datas)
            .map(|(l, lookup_data)| {
                round::generate_interaction_trace(
                    log_size + l,
                    lookup_data,
                    &xor_lookup_elements,
                    &round_lookup_elements,
                )
            }),
    );

    let (xor_trace12, xor_claimed_sum12) =
        xor_table::generate_interaction_trace(xor_lookup_data12, &xor_lookup_elements.xor12);
    let (xor_trace9, xor_claimed_sum9) =
        xor_table::generate_interaction_trace(xor_lookup_data9, &xor_lookup_elements.xor9);
    let (xor_trace8, xor_claimed_sum8) =
        xor_table::generate_interaction_trace(xor_lookup_data8, &xor_lookup_elements.xor8);
    let (xor_trace7, xor_claimed_sum7) =
        xor_table::generate_interaction_trace(xor_lookup_data7, &xor_lookup_elements.xor7);
    let (xor_trace4, xor_claimed_sum4) =
        xor_table::generate_interaction_trace(xor_lookup_data4, &xor_lookup_elements.xor4);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(
        chain![
            scheduler_trace,
            round_traces.into_iter().flatten(),
            xor_trace12,
            xor_trace9,
            xor_trace8,
            xor_trace7,
            xor_trace4,
        ]
        .collect_vec(),
    );
    tree_builder.commit(channel);
    span.exit();

    // Constant trace.
    let span = span!(Level::INFO, "Constant Trace").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(
        chain![
            [gen_is_first(log_size)],
            ROUND_LOG_SPLIT.map(|l| gen_is_first(log_size + l)),
            xor_table::generate_constant_trace::<12, 4>(),
            xor_table::generate_constant_trace::<9, 2>(),
            xor_table::generate_constant_trace::<8, 2>(),
            xor_table::generate_constant_trace::<7, 2>(),
            xor_table::generate_constant_trace::<4, 0>(),
        ]
        .collect_vec(),
    );
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let scheduler_component = BlakeSchedulerComponent {
        log_size,
        blake_lookup_elements,
        round_lookup_elements: round_lookup_elements.clone(),
        claimed_sum: scheduler_claimed_sum,
    };
    let round_components = round_claimed_sums
        .into_iter()
        .zip(ROUND_LOG_SPLIT)
        .map(|(claimed_sum, l)| BlakeRoundComponent {
            log_size: log_size + l,
            xor_lookup_elements: xor_lookup_elements.clone(),
            round_lookup_elements: round_lookup_elements.clone(),
            claimed_sum,
        })
        .collect();
    let xor12 = XorTableComponent::<12, 4> {
        lookup_elements: xor_lookup_elements.xor12,
        claimed_sum: xor_claimed_sum12,
    };
    let xor9 = XorTableComponent::<9, 2> {
        lookup_elements: xor_lookup_elements.xor9,
        claimed_sum: xor_claimed_sum9,
    };
    let xor8 = XorTableComponent::<8, 2> {
        lookup_elements: xor_lookup_elements.xor8,
        claimed_sum: xor_claimed_sum8,
    };
    let xor7 = XorTableComponent::<7, 2> {
        lookup_elements: xor_lookup_elements.xor7,
        claimed_sum: xor_claimed_sum7,
    };
    let xor4 = XorTableComponent::<4, 0> {
        lookup_elements: xor_lookup_elements.xor4,
        claimed_sum: xor_claimed_sum4,
    };
    let air = BlakeAir {
        scheduler_component,
        round_components,
        xor12,
        xor9,
        xor8,
        xor7,
        xor4,
    };
    let proof = prove::<SimdBackend, _, _>(
        &air.component_provers(),
        channel,
        &InteractionElements::default(),
        commitment_scheme,
    )
    .unwrap();

    (air, proof)
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::core::air::{Air, Components};
    use crate::core::channel::{Blake2sChannel, Channel};
    use crate::core::pcs::CommitmentSchemeVerifier;
    use crate::core::prover::verify;
    use crate::core::vcs::blake2_hash::Blake2sHash;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::InteractionElements;
    use crate::examples::blake::air::prove_blake;
    use crate::examples::blake::round::RoundElements;
    use crate::examples::blake::xor_table::XorElements;

    // Note: this test is slow. Only run in release.
    #[ignore]
    #[test_log::test]
    fn test_simd_blake_prove() {
        // Note: To see time measurement, run test with
        //   LOG_N_INSTANCES=16 RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUSTFLAGS="
        //   -C target-cpu=native -C target-feature=+avx512f" cargo test --release
        //   test_simd_blake_prove -- --nocapture --ignored

        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "6".to_string())
            .parse::<u32>()
            .unwrap();

        // Prove.
        let (air, proof) = prove_blake::<Blake2sChannel, Blake2sMerkleHasher>(log_n_instances);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::new(Blake2sHash::default());
        let commitment_scheme = &mut CommitmentSchemeVerifier::new();

        // Decommit.
        let sizes = Components(air.components()).column_log_sizes();

        // Trace columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Draw lookup element.
        let blake_lookup_elements = RoundElements::draw(channel);
        let round_lookup_elements = RoundElements::draw(channel);
        let xor_lookup_elements = XorElements::draw(channel);
        assert_eq!(
            blake_lookup_elements,
            air.scheduler_component.blake_lookup_elements
        );
        assert_eq!(
            round_lookup_elements,
            air.scheduler_component.round_lookup_elements
        );
        assert_eq!(xor_lookup_elements, air.xor12.lookup_elements);

        // TODO(spapini): Check claimed sum against first and last instances.
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Constant columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(
            &air.components(),
            channel,
            &InteractionElements::default(), // Not in use.
            commitment_scheme,
            proof,
        )
        .unwrap();
    }
}
