use std::array;
use std::simd::u32x16;

use itertools::{chain, multiunzip, Itertools};
use tracing::{span, Level};

use super::gkr_lookups::MleCoeffColumnOracleAccumulator;
use super::round::{BlakeRoundComponent, BlakeRoundEval};
use super::scheduler::BlakeSchedulerComponent;
use super::xor_table::{XorLookupArtifacts, XorTableComponent, XorTableEval};
use crate::constraint_framework::{FrameworkEval, TraceLocationAllocator};
use crate::core::air::ComponentProver;
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::BackendForChannel;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::lookups::gkr_prover::prove_batch;
use crate::core::lookups::gkr_verifier::{
    GkrBatchProof, LookupArtifactInstance, LookupArtifactInstanceIter,
};
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig};
use crate::core::poly::circle::{CanonicCoset, PolyOps};
use crate::core::prover::{prove, StarkProof, VerificationError};
use crate::core::vcs::ops::MerkleHasher;
use crate::examples::blake::air::AllElements;
use crate::examples::blake::scheduler::{self as air_scheduler, BlakeInput};
use crate::examples::blake::{
    round as air_round, xor_table as air_xor_table, XorAccums, N_ROUNDS, ROUND_LOG_SPLIT,
};
use crate::examples::blake_gkr::gkr_lookups::accumulation::{MleClaimAccumulator, MleCollection};
use crate::examples::blake_gkr::gkr_lookups::mle_eval::{self, MleEvalProverComponent};
use crate::examples::blake_gkr::round::RoundLookupArtifact;
use crate::examples::blake_gkr::scheduler::{BlakeSchedulerEval, SchedulerLookupArtifact};
use crate::examples::blake_gkr::{round, scheduler, xor_table};

pub struct BlakeClaim {
    log_size: u32,
}

impl BlakeClaim {
    fn mix_into(&self, channel: &mut impl Channel) {
        // TODO(spapini): Do this better.
        channel.mix_u64(self.log_size as u64);
    }
}

pub struct BlakeProof<H: MerkleHasher> {
    pub claim: BlakeClaim,
    pub gkr_proof: GkrBatchProof,
    pub stark_proof: StarkProof<H>,
}

pub struct BlakeLookupArtifacts {
    scheduler: SchedulerLookupArtifact,
    /// `|ROUND_LOG_SPLIT|` many round artifacts.
    rounds: Vec<RoundLookupArtifact>,
    xor: XorLookupArtifacts,
}

impl BlakeLookupArtifacts {
    pub fn new_from_iter(mut iter: impl Iterator<Item = LookupArtifactInstance>) -> Self {
        Self {
            scheduler: SchedulerLookupArtifact::new_from_iter(&mut iter),
            rounds: ROUND_LOG_SPLIT
                .iter()
                .map(|_| RoundLookupArtifact::new_from_iter(&mut iter))
                .collect(),
            xor: XorLookupArtifacts::new_from_iter(&mut iter),
        }
    }

    pub fn verify_succinct_mle_claims(
        &self,
        lookup_elements: &AllElements,
    ) -> Result<(), InvalidClaimError> {
        let Self {
            scheduler,
            rounds,
            xor,
        } = self;
        scheduler.verify_succinct_mle_claims()?;
        for round in rounds {
            round.verify_succinct_mle_claims()?;
        }
        xor.verify_succinct_mle_claims(&lookup_elements.xor_elements)?;
        Ok(())
    }

    pub fn accumulate_mle_eval_iop_claims(&self, acc: &mut MleClaimAccumulator) {
        let Self {
            scheduler,
            rounds,
            xor,
        } = self;
        scheduler.accumulate_mle_eval_iop_claims(acc);
        rounds
            .iter()
            .for_each(|round| round.accumulate_mle_eval_iop_claims(acc));
        xor.accumulate_mle_eval_iop_claims(acc);
    }
}

#[derive(Debug)]
pub struct InvalidClaimError;

pub struct BlakeComponents {
    scheduler_component: BlakeSchedulerComponent,
    round_components: Vec<BlakeRoundComponent>,
    xor12: XorTableComponent<12, 4>,
    xor9: XorTableComponent<9, 2>,
    xor8: XorTableComponent<8, 2>,
    xor7: XorTableComponent<7, 2>,
    xor4: XorTableComponent<4, 0>,
}

impl BlakeComponents {
    pub fn new(
        trace_location_allocator: &mut TraceLocationAllocator,
        claim: &BlakeClaim,
        all_elements: &AllElements,
    ) -> Self {
        Self {
            scheduler_component: BlakeSchedulerComponent::new(
                trace_location_allocator,
                BlakeSchedulerEval {
                    log_size: claim.log_size,
                    blake_lookup_elements: all_elements.blake_elements.clone(),
                    round_lookup_elements: all_elements.round_elements.clone(),
                },
            ),
            round_components: ROUND_LOG_SPLIT
                .iter()
                .map(|l| {
                    BlakeRoundComponent::new(
                        trace_location_allocator,
                        BlakeRoundEval {
                            log_size: claim.log_size + l,
                            xor_lookup_elements: all_elements.xor_elements.clone(),
                            round_lookup_elements: all_elements.round_elements.clone(),
                        },
                    )
                })
                .collect(),
            xor12: XorTableComponent::new(
                trace_location_allocator,
                XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor12.clone(),
                },
            ),
            xor9: XorTableComponent::new(
                trace_location_allocator,
                XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor12.clone(),
                },
            ),
            xor8: XorTableComponent::new(
                trace_location_allocator,
                XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor12.clone(),
                },
            ),
            xor7: XorTableComponent::new(
                trace_location_allocator,
                XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor12.clone(),
                },
            ),
            xor4: XorTableComponent::new(
                trace_location_allocator,
                XorTableEval {
                    lookup_elements: all_elements.xor_elements.xor12.clone(),
                },
            ),
        }
    }

    pub fn accumulate_mle_coeff_col_oracles<'this: 'acc, 'acc>(
        &'this self,
        acc_by_n_vars: &mut [Option<MleCoeffColumnOracleAccumulator<'acc>>],
    ) {
        let Self {
            scheduler_component,
            round_components,
            xor12,
            xor9,
            xor8,
            xor7,
            xor4,
        } = self;
        acc_by_n_vars[scheduler_component.log_size as usize]
            .as_mut()
            .unwrap()
            .accumulate(scheduler_component);
        for round_component in round_components {
            acc_by_n_vars[round_component.log_size as usize]
                .as_mut()
                .unwrap()
                .accumulate(round_component)
        }
        acc_by_n_vars[xor12.log_size() as usize]
            .as_mut()
            .unwrap()
            .accumulate(xor12);
        acc_by_n_vars[xor9.log_size() as usize]
            .as_mut()
            .unwrap()
            .accumulate(xor9);
        acc_by_n_vars[xor8.log_size() as usize]
            .as_mut()
            .unwrap()
            .accumulate(xor8);
        acc_by_n_vars[xor7.log_size() as usize]
            .as_mut()
            .unwrap()
            .accumulate(xor7);
        acc_by_n_vars[xor4.log_size() as usize]
            .as_mut()
            .unwrap()
            .accumulate(xor4);
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

pub fn prove_blake<MC: MerkleChannel>(log_size: u32, config: PcsConfig) -> BlakeProof<MC::H>
where
    SimdBackend: BackendForChannel<MC>,
{
    assert!(log_size >= LOG_N_LANES);
    assert_eq!(
        ROUND_LOG_SPLIT.map(|x| 1 << x).iter().sum::<usize>(),
        N_ROUNDS
    );

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    const XOR_TABLE_MAX_LOG_SIZE: u32 = 16;
    let max_log_size =
        (log_size + *ROUND_LOG_SPLIT.iter().max().unwrap()).max(XOR_TABLE_MAX_LOG_SIZE);
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Prepare inputs.
    let blake_inputs = (0..1 << (log_size - LOG_N_LANES))
        .map(|i| {
            let v = [u32x16::from_array(array::from_fn(|j| (i + 2 * j) as u32)); 16];
            let m = [u32x16::from_array(array::from_fn(|j| (i + 2 * j + 1) as u32)); 16];
            BlakeInput { v, m }
        })
        .collect_vec();

    // Setup protocol.
    let channel = &mut MC::C::default();
    let commitment_scheme = &mut CommitmentSchemeProver::new(config, &twiddles);

    let span = span!(Level::INFO, "Trace").entered();

    // Scheduler.
    let (scheduler_trace, scheduler_lookup_data, round_inputs) =
        air_scheduler::gen_trace(log_size, &blake_inputs);

    // Rounds.
    let mut xor_accums = XorAccums::default();
    let mut rest = &round_inputs[..];
    // Split round inputs to components, according to [ROUND_LOG_SPLIT].
    let (round_traces, round_lookup_datas): (Vec<_>, Vec<_>) =
        multiunzip(ROUND_LOG_SPLIT.map(|l| {
            let (cur_inputs, r) = rest.split_at(1 << (log_size - LOG_N_LANES + l));
            rest = r;
            air_round::generate_trace(log_size + l, cur_inputs, &mut xor_accums)
        }));

    // Xor tables.
    let (xor_trace12, xor_lookup_data12) = air_xor_table::generate_trace(xor_accums.xor12);
    let (xor_trace9, xor_lookup_data9) = air_xor_table::generate_trace(xor_accums.xor9);
    let (xor_trace8, xor_lookup_data8) = air_xor_table::generate_trace(xor_accums.xor8);
    let (xor_trace7, xor_lookup_data7) = air_xor_table::generate_trace(xor_accums.xor7);
    let (xor_trace4, xor_lookup_data4) = air_xor_table::generate_trace(xor_accums.xor4);

    // Claim.
    let claim = BlakeClaim { log_size };
    claim.mix_into(channel);

    // Trace commitment.
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![
        scheduler_trace,
        round_traces.into_iter().flatten(),
        xor_trace12,
        xor_trace9,
        xor_trace8,
        xor_trace7,
        xor_trace4,
    ]);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let all_elements = AllElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let mut lookup_input_layers = Vec::new();
    let mut mle_eval_at_point_collection = MleCollection::default();

    lookup_input_layers.extend(scheduler::generate_lookup_instances(
        log_size,
        scheduler_lookup_data,
        &all_elements.round_elements,
        &all_elements.blake_elements,
        &mut mle_eval_at_point_collection,
    ));

    ROUND_LOG_SPLIT
        .iter()
        .zip(round_lookup_datas)
        .for_each(|(l, lookup_data)| {
            lookup_input_layers.extend(round::generate_lookup_instances(
                log_size + l,
                lookup_data,
                &all_elements.xor_elements,
                &all_elements.round_elements,
                &mut mle_eval_at_point_collection,
            ));
        });

    lookup_input_layers.extend(xor_table::generate_lookup_instances(
        xor_lookup_data12,
        &all_elements.xor_elements.xor12,
        &mut mle_eval_at_point_collection,
    ));
    lookup_input_layers.extend(xor_table::generate_lookup_instances(
        xor_lookup_data9,
        &all_elements.xor_elements.xor9,
        &mut mle_eval_at_point_collection,
    ));
    lookup_input_layers.extend(xor_table::generate_lookup_instances(
        xor_lookup_data8,
        &all_elements.xor_elements.xor8,
        &mut mle_eval_at_point_collection,
    ));
    lookup_input_layers.extend(xor_table::generate_lookup_instances(
        xor_lookup_data7,
        &all_elements.xor_elements.xor7,
        &mut mle_eval_at_point_collection,
    ));
    lookup_input_layers.extend(xor_table::generate_lookup_instances(
        xor_lookup_data4,
        &all_elements.xor_elements.xor4,
        &mut mle_eval_at_point_collection,
    ));

    let gkr_span = span!(Level::INFO, "GKR proof").entered();
    let (gkr_proof, gkr_artifact) = prove_batch(channel, lookup_input_layers);
    gkr_span.exit();
    let mle_acc_coeff = channel.draw_felt();
    let mles = mle_eval_at_point_collection.random_linear_combine_by_n_variables(mle_acc_coeff);

    // TODO(andrew): Consider unifying new_from_iter, verify_succinct_mle_claims,
    // accumulate_mle_eval_iop_claims.
    let mut lookup_instances_iter = LookupArtifactInstanceIter::new(&gkr_proof, &gkr_artifact);
    let blake_lookup_artifacts = BlakeLookupArtifacts::new_from_iter(&mut lookup_instances_iter);
    assert!(lookup_instances_iter.next().is_none());
    blake_lookup_artifacts
        .verify_succinct_mle_claims(&all_elements)
        .unwrap();
    let mut mle_eval_iop_acc = MleClaimAccumulator::new(mle_acc_coeff);
    blake_lookup_artifacts.accumulate_mle_eval_iop_claims(&mut mle_eval_iop_acc);
    let mut mle_claim_by_n_variables = mle_eval_iop_acc.finalize();

    let max_mle_n_variables = mles.iter().map(|mle| mle.n_variables()).max().unwrap();
    let mut mle_coeff_col_acc_by_n_variables = vec![None; max_mle_n_variables + 1];

    for mle in &mles {
        let n_variables = mle.n_variables();
        mle_coeff_col_acc_by_n_variables[n_variables] =
            Some(MleCoeffColumnOracleAccumulator::new(mle_acc_coeff));
    }

    let trace_location_allocator = &mut TraceLocationAllocator::default();
    let blake_components = BlakeComponents::new(trace_location_allocator, &claim, &all_elements);
    blake_components.accumulate_mle_coeff_col_oracles(&mut mle_coeff_col_acc_by_n_variables);

    let mut tree_builder = commitment_scheme.tree_builder();
    let mle_eval_prover_components = mles
        .into_iter()
        .map(|mle| {
            let n_vars = mle.n_variables();
            let coeff_column_oracle = mle_coeff_col_acc_by_n_variables[n_vars].as_ref().unwrap();
            let claim = mle_claim_by_n_variables[n_vars].take().unwrap();
            let eval_point = gkr_artifact.ood_point(n_vars);

            tree_builder.extend_evals(mle_eval::build_trace(&mle, eval_point, claim));

            // Sanity check the claims.
            #[cfg(test)]
            debug_assert_eq!(claim, mle.eval_at_point(eval_point));

            MleEvalProverComponent::generate(
                trace_location_allocator,
                coeff_column_oracle,
                eval_point,
                mle,
                claim,
                &twiddles,
                1,
            )
        })
        .collect_vec();
    tree_builder.commit(channel);
    span.exit();

    let components = chain![
        blake_components.component_provers(),
        mle_eval_prover_components
            .iter()
            .map(|c| c as &dyn ComponentProver<SimdBackend>)
    ]
    .collect_vec();

    let stark_proof = prove(&components, channel, commitment_scheme).unwrap();

    BlakeProof {
        claim,
        gkr_proof,
        stark_proof,
    }
}

#[allow(unused)]
pub fn verify_blake<MC: MerkleChannel>(
    BlakeProof {
        claim,
        gkr_proof,
        stark_proof,
    }: BlakeProof<MC::H>,
    config: PcsConfig,
) -> Result<(), VerificationError> {
    let channel = &mut MC::C::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(config);

    // let log_sizes = stmt0.log_sizes();

    // // Trace.
    // stmt0.mix_into(channel);
    // commitment_scheme.commit(stark_proof.commitments[0], &log_sizes[0], channel);

    // // Draw interaction elements.
    // let all_elements = AllElements::draw(channel);

    // // Interaction trace.
    // stmt1.mix_into(channel);
    // commitment_scheme.commit(stark_proof.commitments[1], &log_sizes[1], channel);

    // // Constant trace.
    // commitment_scheme.commit(stark_proof.commitments[2], &log_sizes[2], channel);

    // let components = BlakeComponents::new(&stmt0, &all_elements, &stmt1);

    // // Check that all sums are correct.
    // let total_sum = stmt1.scheduler_claimed_sum
    //     + stmt1.round_claimed_sums.iter().sum::<SecureField>()
    //     + stmt1.xor12_claimed_sum
    //     + stmt1.xor9_claimed_sum
    //     + stmt1.xor8_claimed_sum
    //     + stmt1.xor7_claimed_sum
    //     + stmt1.xor4_claimed_sum;

    // // TODO(spapini): Add inputs to sum, and constraint them.
    // assert_eq!(total_sum, SecureField::zero());

    // verify(
    //     &components.components(),
    //     channel,
    //     commitment_scheme,
    //     stark_proof,
    // )

    todo!()
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::core::pcs::PcsConfig;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::examples::blake_gkr::air::prove_blake;

    // Note: this test is slow. Only run in release.
    #[cfg_attr(not(feature = "slow-tests"), ignore)]
    #[test_log::test]
    fn test_simd_blake_gkr_prove() {
        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "6".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig::default();

        // Prove.
        let _proof = prove_blake::<Blake2sMerkleChannel>(log_n_instances, config);

        // Verify.
        // verify_blake::<Blake2sMerkleChannel>(proof, config).unwrap();
    }
}
