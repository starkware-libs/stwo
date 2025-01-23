use std::simd::u32x16;

use itertools::{chain, multiunzip, Itertools};
use num_traits::Zero;
use serde::Serialize;
use tracing::{span, Level};

use super::preprocessed_columns::XorTable;
use super::round::{blake_round_info, BlakeRoundComponent, BlakeRoundEval};
use super::scheduler::{BlakeSchedulerComponent, BlakeSchedulerEval};
use super::xor_table::{xor12, xor4, xor7, xor8, xor9};
use crate::constraint_framework::preprocessed_columns::{IsFirst, PreProcessedColumnId};
use crate::constraint_framework::{TraceLocationAllocator, PREPROCESSED_TRACE_IDX};
use crate::core::air::{Component, ComponentProver};
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::BackendForChannel;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
use crate::core::poly::circle::{CanonicCoset, PolyOps};
use crate::core::prover::{prove, verify, StarkProof, VerificationError};
use crate::core::vcs::ops::MerkleHasher;
use crate::examples::blake::round::RoundElements;
use crate::examples::blake::scheduler::{self, blake_scheduler_info, BlakeElements, BlakeInput};
use crate::examples::blake::{
    round, xor_table, BlakeXorElements, XorAccums, N_ROUNDS, ROUND_LOG_SPLIT,
};

fn preprocessed_xor_columns() -> [PreProcessedColumnId; 20] {
    [
        XorTable::new(12, 4, 0).id(),
        XorTable::new(12, 4, 1).id(),
        XorTable::new(12, 4, 2).id(),
        IsFirst::new(XorTable::new(12, 4, 0).column_bits()).id(),
        XorTable::new(9, 2, 0).id(),
        XorTable::new(9, 2, 1).id(),
        XorTable::new(9, 2, 2).id(),
        IsFirst::new(XorTable::new(9, 2, 0).column_bits()).id(),
        XorTable::new(8, 2, 0).id(),
        XorTable::new(8, 2, 1).id(),
        XorTable::new(8, 2, 2).id(),
        IsFirst::new(XorTable::new(8, 2, 0).column_bits()).id(),
        XorTable::new(7, 2, 0).id(),
        XorTable::new(7, 2, 1).id(),
        XorTable::new(7, 2, 2).id(),
        IsFirst::new(XorTable::new(7, 2, 0).column_bits()).id(),
        XorTable::new(4, 0, 0).id(),
        XorTable::new(4, 0, 1).id(),
        XorTable::new(4, 0, 2).id(),
        IsFirst::new(XorTable::new(4, 0, 0).column_bits()).id(),
    ]
}

const fn preprocessed_xor_columns_log_sizes() -> [u32; 20] {
    [
        XorTable::new(12, 4, 0).column_bits(),
        XorTable::new(12, 4, 1).column_bits(),
        XorTable::new(12, 4, 2).column_bits(),
        XorTable::new(12, 4, 0).column_bits(),
        XorTable::new(9, 2, 0).column_bits(),
        XorTable::new(9, 2, 1).column_bits(),
        XorTable::new(9, 2, 2).column_bits(),
        XorTable::new(9, 2, 0).column_bits(),
        XorTable::new(8, 2, 0).column_bits(),
        XorTable::new(8, 2, 1).column_bits(),
        XorTable::new(8, 2, 2).column_bits(),
        XorTable::new(8, 2, 0).column_bits(),
        XorTable::new(7, 2, 0).column_bits(),
        XorTable::new(7, 2, 1).column_bits(),
        XorTable::new(7, 2, 2).column_bits(),
        XorTable::new(7, 2, 0).column_bits(),
        XorTable::new(4, 0, 0).column_bits(),
        XorTable::new(4, 0, 1).column_bits(),
        XorTable::new(4, 0, 2).column_bits(),
        XorTable::new(4, 0, 0).column_bits(),
    ]
}

#[derive(Serialize)]
pub struct BlakeStatement0 {
    log_size: u32,
}
impl BlakeStatement0 {
    fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut sizes = vec![];
        sizes.push(
            blake_scheduler_info()
                .mask_offsets
                .as_cols_ref()
                .map_cols(|_| self.log_size),
        );
        for l in ROUND_LOG_SPLIT {
            sizes.push(
                blake_round_info()
                    .mask_offsets
                    .as_cols_ref()
                    .map_cols(|_| self.log_size + l),
            );
        }
        sizes.push(xor_table::xor12::trace_sizes::<12, 4>());
        sizes.push(xor_table::xor9::trace_sizes::<9, 2>());
        sizes.push(xor_table::xor8::trace_sizes::<8, 2>());
        sizes.push(xor_table::xor7::trace_sizes::<7, 2>());
        sizes.push(xor_table::xor4::trace_sizes::<4, 0>());

        let mut log_sizes = TreeVec::concat_cols(sizes.into_iter());

        let log_size = self.log_size;

        let scheduler_is_first_column_log_size = log_size;
        let blake_round_is_first_column_log_sizes = ROUND_LOG_SPLIT.iter().map(|l| log_size + l);

        log_sizes[PREPROCESSED_TRACE_IDX] = chain!(
            [scheduler_is_first_column_log_size],
            blake_round_is_first_column_log_sizes,
            preprocessed_xor_columns_log_sizes(),
        )
        .collect_vec();

        log_sizes
    }
    fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size as u64);
    }
}

pub struct AllElements {
    blake_elements: BlakeElements,
    round_elements: RoundElements,
    xor_elements: BlakeXorElements,
}
impl AllElements {
    pub fn draw(channel: &mut impl Channel) -> Self {
        Self {
            blake_elements: BlakeElements::draw(channel),
            round_elements: RoundElements::draw(channel),
            xor_elements: BlakeXorElements::draw(channel),
        }
    }
}

pub struct BlakeStatement1 {
    scheduler_claimed_sum: SecureField,
    round_claimed_sums: Vec<SecureField>,
    xor12_claimed_sum: SecureField,
    xor9_claimed_sum: SecureField,
    xor8_claimed_sum: SecureField,
    xor7_claimed_sum: SecureField,
    xor4_claimed_sum: SecureField,
}
impl BlakeStatement1 {
    fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(
            &chain![
                [
                    self.scheduler_claimed_sum,
                    self.xor12_claimed_sum,
                    self.xor9_claimed_sum,
                    self.xor8_claimed_sum,
                    self.xor7_claimed_sum,
                    self.xor4_claimed_sum
                ],
                self.round_claimed_sums.clone()
            ]
            .collect_vec(),
        )
    }
}

pub struct BlakeProof<H: MerkleHasher> {
    stmt0: BlakeStatement0,
    stmt1: BlakeStatement1,
    stark_proof: StarkProof<H>,
}

pub struct BlakeComponents {
    scheduler_component: BlakeSchedulerComponent,
    round_components: Vec<BlakeRoundComponent>,
    xor12: xor12::XorTableComponent<12, 4>,
    xor9: xor9::XorTableComponent<9, 2>,
    xor8: xor8::XorTableComponent<8, 2>,
    xor7: xor7::XorTableComponent<7, 2>,
    xor4: xor4::XorTableComponent<4, 0>,
}
impl BlakeComponents {
    fn new(stmt0: &BlakeStatement0, all_elements: &AllElements, stmt1: &BlakeStatement1) -> Self {
        let log_size = stmt0.log_size;

        let scheduler_is_first_column = IsFirst::new(log_size).id();
        let blake_round_is_first_columns_iter: Vec<PreProcessedColumnId> = ROUND_LOG_SPLIT
            .iter()
            .map(|l| IsFirst::new(log_size + l).id())
            .collect_vec();

        let tree_span_provider = &mut TraceLocationAllocator::new_with_preproccessed_columns(
            &chain!(
                [scheduler_is_first_column],
                blake_round_is_first_columns_iter,
                preprocessed_xor_columns(),
            )
            .collect_vec()[..],
        );

        Self {
            scheduler_component: BlakeSchedulerComponent::new(
                tree_span_provider,
                BlakeSchedulerEval {
                    log_size: stmt0.log_size,
                    blake_lookup_elements: all_elements.blake_elements.clone(),
                    round_lookup_elements: all_elements.round_elements.clone(),
                    claimed_sum: stmt1.scheduler_claimed_sum,
                },
                stmt1.scheduler_claimed_sum,
            ),
            round_components: ROUND_LOG_SPLIT
                .iter()
                .zip(stmt1.round_claimed_sums.clone())
                .map(|(l, claimed_sum)| {
                    BlakeRoundComponent::new(
                        tree_span_provider,
                        BlakeRoundEval {
                            log_size: stmt0.log_size + l,
                            xor_lookup_elements: all_elements.xor_elements.clone(),
                            round_lookup_elements: all_elements.round_elements.clone(),
                            claimed_sum,
                        },
                        claimed_sum,
                    )
                })
                .collect(),
            xor12: xor12::XorTableComponent::new(
                tree_span_provider,
                xor12::XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor12.clone(),
                    claimed_sum: stmt1.xor12_claimed_sum,
                },
                stmt1.xor12_claimed_sum,
            ),
            xor9: xor9::XorTableComponent::new(
                tree_span_provider,
                xor9::XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor9.clone(),
                    claimed_sum: stmt1.xor9_claimed_sum,
                },
                stmt1.xor9_claimed_sum,
            ),
            xor8: xor8::XorTableComponent::new(
                tree_span_provider,
                xor8::XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor8.clone(),
                    claimed_sum: stmt1.xor8_claimed_sum,
                },
                stmt1.xor8_claimed_sum,
            ),
            xor7: xor7::XorTableComponent::new(
                tree_span_provider,
                xor7::XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor7.clone(),
                    claimed_sum: stmt1.xor7_claimed_sum,
                },
                stmt1.xor7_claimed_sum,
            ),
            xor4: xor4::XorTableComponent::new(
                tree_span_provider,
                xor4::XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor4.clone(),
                    claimed_sum: stmt1.xor4_claimed_sum,
                },
                stmt1.xor4_claimed_sum,
            ),
        }
    }
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
pub fn prove_blake<MC: MerkleChannel>(log_size: u32, config: PcsConfig) -> (BlakeProof<MC::H>)
where
    SimdBackend: BackendForChannel<MC>,
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
        CanonicCoset::new(log_max_rows + 1 + config.fri_config.log_blowup_factor)
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
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::new(config, &twiddles);

    // Preprocessed trace.
    // TODO(ShaharS): share is_first column between components when constant columns support this.
    let span = span!(Level::INFO, "Preprocessed Trace").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(
        chain![
            vec![IsFirst::new(log_size).gen_column_simd()],
            ROUND_LOG_SPLIT
                .iter()
                .map(|l| IsFirst::new(log_size + l).gen_column_simd()),
            XorTable::new(12, 4, 0).generate_constant_trace(),
            XorTable::new(9, 2, 0).generate_constant_trace(),
            XorTable::new(8, 2, 0).generate_constant_trace(),
            XorTable::new(7, 2, 0).generate_constant_trace(),
            XorTable::new(4, 0, 0).generate_constant_trace(),
        ]
        .collect_vec(),
    );
    tree_builder.commit(channel);
    span.exit();

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
    let (xor_trace12, xor_lookup_data12) = xor_table::xor12::generate_trace(xor_accums.xor12);
    let (xor_trace9, xor_lookup_data9) = xor_table::xor9::generate_trace(xor_accums.xor9);
    let (xor_trace8, xor_lookup_data8) = xor_table::xor8::generate_trace(xor_accums.xor8);
    let (xor_trace7, xor_lookup_data7) = xor_table::xor7::generate_trace(xor_accums.xor7);
    let (xor_trace4, xor_lookup_data4) = xor_table::xor4::generate_trace(xor_accums.xor4);

    // Statement0.
    let stmt0 = BlakeStatement0 { log_size };
    stmt0.mix_into(channel);

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
    let all_elements = AllElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (scheduler_trace, scheduler_claimed_sum) = scheduler::gen_interaction_trace(
        log_size,
        scheduler_lookup_data,
        &all_elements.round_elements,
        &all_elements.blake_elements,
    );

    let (round_traces, round_claimed_sums): (Vec<_>, Vec<_>) = multiunzip(
        ROUND_LOG_SPLIT
            .iter()
            .zip(round_lookup_datas)
            .map(|(l, lookup_data)| {
                round::generate_interaction_trace(
                    log_size + l,
                    lookup_data,
                    &all_elements.xor_elements,
                    &all_elements.round_elements,
                )
            }),
    );

    let (xor_trace12, xor12_claimed_sum) = xor_table::xor12::generate_interaction_trace(
        xor_lookup_data12,
        &all_elements.xor_elements.xor12,
    );
    let (xor_trace9, xor9_claimed_sum) = xor_table::xor9::generate_interaction_trace(
        xor_lookup_data9,
        &all_elements.xor_elements.xor9,
    );
    let (xor_trace8, xor8_claimed_sum) = xor_table::xor8::generate_interaction_trace(
        xor_lookup_data8,
        &all_elements.xor_elements.xor8,
    );
    let (xor_trace7, xor7_claimed_sum) = xor_table::xor7::generate_interaction_trace(
        xor_lookup_data7,
        &all_elements.xor_elements.xor7,
    );
    let (xor_trace4, xor4_claimed_sum) = xor_table::xor4::generate_interaction_trace(
        xor_lookup_data4,
        &all_elements.xor_elements.xor4,
    );

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

    // Statement1.
    let stmt1 = BlakeStatement1 {
        scheduler_claimed_sum,
        round_claimed_sums,
        xor12_claimed_sum,
        xor9_claimed_sum,
        xor8_claimed_sum,
        xor7_claimed_sum,
        xor4_claimed_sum,
    };
    stmt1.mix_into(channel);
    tree_builder.commit(channel);
    span.exit();

    assert_eq!(
        commitment_scheme
            .polynomials()
            .as_cols_ref()
            .map_cols(|c| c.log_size())
            .0,
        stmt0.log_sizes().0
    );

    // Prove constraints.
    let components = BlakeComponents::new(&stmt0, &all_elements, &stmt1);
    let stark_proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();

    BlakeProof {
        stmt0,
        stmt1,
        stark_proof,
    }
}

#[allow(unused)]
pub fn verify_blake<MC: MerkleChannel>(
    BlakeProof {
        stmt0,
        stmt1,
        stark_proof,
    }: BlakeProof<MC::H>,
) -> Result<(), VerificationError> {
    // TODO(alonf): Consider mixing the config into the channel.
    let channel = &mut MC::C::default();
    const REQUIRED_SECURITY_BITS: u32 = 5;
    assert!(stark_proof.config.security_bits() >= REQUIRED_SECURITY_BITS);
    let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(stark_proof.config);

    let log_sizes = stmt0.log_sizes();

    // Preprocessed trace.
    commitment_scheme.commit(stark_proof.commitments[0], &log_sizes[0], channel);

    // Trace.
    stmt0.mix_into(channel);
    commitment_scheme.commit(stark_proof.commitments[1], &log_sizes[1], channel);

    // Draw interaction elements.
    let all_elements = AllElements::draw(channel);

    // Interaction trace.
    stmt1.mix_into(channel);
    commitment_scheme.commit(stark_proof.commitments[2], &log_sizes[2], channel);

    let components = BlakeComponents::new(&stmt0, &all_elements, &stmt1);

    // Check that all sums are correct.
    let claimed_sum = stmt1.scheduler_claimed_sum
        + stmt1.round_claimed_sums.iter().sum::<SecureField>()
        + stmt1.xor12_claimed_sum
        + stmt1.xor9_claimed_sum
        + stmt1.xor8_claimed_sum
        + stmt1.xor7_claimed_sum
        + stmt1.xor4_claimed_sum;

    // TODO(shahars): Add inputs to sum, and constraint them.
    assert_eq!(claimed_sum, SecureField::zero());

    verify(
        &components.components(),
        channel,
        commitment_scheme,
        stark_proof,
    )
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::core::pcs::PcsConfig;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::examples::blake::air::{prove_blake, verify_blake};

    // Note: this test is slow. Only run in release.
    #[cfg_attr(not(feature = "slow-tests"), ignore)]
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
        let config = PcsConfig::default();

        // Prove.
        let proof = prove_blake::<Blake2sMerkleChannel>(log_n_instances, config);

        // Verify.
        verify_blake::<Blake2sMerkleChannel>(proof).unwrap();
    }
}
