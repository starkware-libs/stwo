use stwo_constraint_framework::relation_tracker::RelationSummary;
use stwo_constraint_framework::Relation;
pub mod components;
pub mod gen;

use components::{
    track_state_machine_relations, State, StateMachineComponents, StateMachineElements,
    StateMachineOp0Component, StateMachineOp1Component, StateMachineProof, StateMachineStatement0,
    StateMachineStatement1, StateMachineStatisticalLogupComponents,
    StateMachineStatisticalLogupProof, StateMachineStatisticalLogupStatement1,
    StateMachineStatisticalOp0Component, StateMachineStatisticalOp1Component, StateTransitionEval,
    StatisticalStateTransitionEval, STATE_SIZE,
};
use gen::{gen_interaction_trace, gen_masked_interaction_trace, gen_trace};
use itertools::{chain, Itertools};
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use stwo::core::air::Components as AirComponents;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::{SecureField, QM31, SECURE_EXTENSION_DEGREE};
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::{verify, verify_zk_with_witness_randomization_audit, VerificationError};
use stwo::core::zk::{
    apply_zk_column_degree_bounds, build_zk_config_from_air_privacy_provider,
    mix_zk_public_metadata, ZkAirId, ZkAirPrivacyProvider, ZkColumnRange, ZkDependencyKind,
    ZkDependencyMetadataCompleteness, ZkLogupClaimManifestEntry, ZkLogupClaimMetadataCompleteness,
    ZkLogupClaimPolicy, ZkLogupClaimVisibility, ZkLogupStatisticalAggregateGroup,
    ZkLogupStatisticalAggregateTarget, ZkPrivacyDependency, ZkPrivacyInferenceMode,
    ZkPrivacyReason, ZkPrivateColumnSemanticDomains, ZkPrivateColumnUsage, ZkPrivateRoot,
    ZkTraceTreeScope, ZkVerificationConfig, ZkWitnessRandomizationVerifierAudit,
};
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::zk::{ZkDerivationGate, ZkDerivationReview, ZkProvingConfig};
use stwo::prover::{prove, prove_zk, CommitmentSchemeProver};
use stwo_constraint_framework::logup::LogupClaim;
use stwo_constraint_framework::{
    StatisticalLogupAggregateComponent, StatisticalLogupCorrectionRef, TraceLocationAllocator,
    INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
};

#[allow(unused)]
pub fn prove_state_machine(
    log_n_rows: u32,
    initial_state: State,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
    track_relations: bool,
) -> (
    StateMachineComponents,
    StateMachineProof<Blake2sMerkleHasher>,
    Option<RelationSummary>,
) {
    let (x_axis_log_rows, y_axis_log_rows) = (log_n_rows, log_n_rows - 1);
    assert!(y_axis_log_rows >= LOG_N_LANES && x_axis_log_rows >= LOG_N_LANES);

    let mut intermediate_state = initial_state;
    intermediate_state[0] += M31::from_u32_unchecked(1 << x_axis_log_rows);
    let mut final_state = intermediate_state;
    final_state[1] += M31::from_u32_unchecked(1 << y_axis_log_rows);

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_store_polynomials_coefficients();
    // Trace.
    let trace_op0 = gen_trace(x_axis_log_rows, initial_state, 0);
    let trace_op1 = gen_trace(y_axis_log_rows, intermediate_state, 1);

    // Commitments.
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.commit(channel);

    let stmt0 = StateMachineStatement0 {
        n: x_axis_log_rows,
        m: y_axis_log_rows,
    };
    stmt0.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![trace_op0.clone(), trace_op1.clone()].collect_vec());
    tree_builder.commit(channel);

    // Draw lookup element.
    let lookup_elements = StateMachineElements::draw(channel);

    // Interaction trace.
    let (interaction_trace_op0, claimed_sum_op0) =
        gen_interaction_trace(&trace_op0, 0, &lookup_elements);
    let (interaction_trace_op1, claimed_sum_op1) =
        gen_interaction_trace(&trace_op1, 1, &lookup_elements);

    let stmt1 = StateMachineStatement1 {
        x_axis_claimed_sum: claimed_sum_op0,
        y_axis_claimed_sum: claimed_sum_op1,
    };
    stmt1.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![interaction_trace_op0, interaction_trace_op1].collect_vec());
    tree_builder.commit(channel);

    // Prove constraints.
    let mut tree_span_provider = &mut TraceLocationAllocator::default();
    let component0 = StateMachineOp0Component::new(
        tree_span_provider,
        StateTransitionEval {
            log_n_rows: x_axis_log_rows,
            lookup_elements: lookup_elements.clone(),
            claimed_sum: claimed_sum_op0,
        },
        claimed_sum_op0,
    );
    let component1 = StateMachineOp1Component::new(
        tree_span_provider,
        StateTransitionEval {
            log_n_rows: y_axis_log_rows,
            lookup_elements,
            claimed_sum: claimed_sum_op1,
        },
        claimed_sum_op1,
    );

    let components = StateMachineComponents {
        component0,
        component1,
    };

    let trace = chain![&trace_op0, &trace_op1].collect_vec();

    let relation_summary = match track_relations {
        false => None,
        true => Some(RelationSummary::summarize_relations(
            &track_state_machine_relations(&TreeVec(vec![vec![], trace]), &components),
        )),
    };

    let stark_proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();
    let proof = StateMachineProof {
        public_input: [initial_state, final_state],
        stmt0,
        stmt1,
        stark_proof,
    };
    (components, proof, relation_summary)
}

pub fn verify_state_machine(
    channel: &mut Blake2sChannel,
    components: StateMachineComponents,
    proof: StateMachineProof<Blake2sMerkleHasher>,
) -> Result<(), VerificationError> {
    let pcs_config = proof.stark_proof.config;
    pcs_config.mix_into(channel);
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);
    // Decommit.
    // Retrieve the expected column sizes in each commitment interaction, from the AIR.
    let sizes = proof.stmt0.log_sizes();

    // Preprocessed columns.
    commitment_scheme.commit(proof.stark_proof.commitments[0], &sizes[0], channel);
    // Trace columns.
    proof.stmt0.mix_into(channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &sizes[1], channel);

    // Assert state machine statement.
    let lookup_elements = StateMachineElements::draw(channel);
    let initial_state_comb: QM31 = lookup_elements.combine(&proof.public_input[0]);
    let final_state_comb: QM31 = lookup_elements.combine(&proof.public_input[1]);
    assert_eq!(
        (proof.stmt1.x_axis_claimed_sum + proof.stmt1.y_axis_claimed_sum)
            * initial_state_comb
            * final_state_comb,
        final_state_comb - initial_state_comb
    );

    // Interaction columns.
    proof.stmt1.mix_into(channel);
    commitment_scheme.commit(proof.stark_proof.commitments[2], &sizes[2], channel);

    verify(
        &components.components(),
        channel,
        commitment_scheme,
        proof.stark_proof,
    )
}

const STATE_MACHINE_STATISTICAL_LOGUP_AGGREGATE_ID: u32 = 0;

fn state_machine_public_aggregate_target(
    lookup_elements: &StateMachineElements,
    public_input: &[State; 2],
) -> SecureField {
    let initial_state_comb: QM31 = lookup_elements.combine(&public_input[0]);
    let final_state_comb: QM31 = lookup_elements.combine(&public_input[1]);
    initial_state_comb.inverse() - final_state_comb.inverse()
}

fn state_machine_public_statement_bytes(
    stmt0: &StateMachineStatement0,
    public_input: &[State; 2],
    label: &[u8],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(label);
    bytes.extend_from_slice(&stmt0.n.to_le_bytes());
    bytes.extend_from_slice(&stmt0.m.to_le_bytes());
    for state in public_input {
        for value in state {
            bytes.extend_from_slice(&value.0.to_le_bytes());
        }
    }
    let log_sizes = state_machine_statistical_logup_component_log_sizes(stmt0);
    bytes.extend_from_slice(&(log_sizes.0.len() as u64).to_le_bytes());
    for tree_log_sizes in &log_sizes.0 {
        bytes.extend_from_slice(&(tree_log_sizes.len() as u64).to_le_bytes());
        for log_size in tree_log_sizes {
            bytes.extend_from_slice(&log_size.to_le_bytes());
        }
    }
    bytes
}

fn state_machine_power_of_two_bound(log_size: u32) -> u64 {
    1u64.checked_shl(log_size).unwrap_or(u64::MAX / 4)
}

fn state_machine_private_lookup_term_count_bound(stmt0: &StateMachineStatement0) -> u64 {
    state_machine_power_of_two_bound(stmt0.n)
        .saturating_add(state_machine_power_of_two_bound(stmt0.m))
        .saturating_mul(2)
        .max(1)
}

fn state_machine_zk_review_hash(seed: u8) -> [u8; 32] {
    [seed; 32]
}

fn state_machine_zk_derivation_reviews() -> Vec<ZkDerivationReview> {
    [
        ZkDerivationGate::StwoSplitQueryExpansion,
        ZkDerivationGate::CircleRandomizerSpace,
        ZkDerivationGate::OodsDomainExclusion,
        ZkDerivationGate::ZkAwareDegreeMetadata,
        ZkDerivationGate::FriBatchMaskDegree,
        ZkDerivationGate::PrivateLookupPermutationExclusion,
        ZkDerivationGate::ProofDataSecrecy,
        ZkDerivationGate::ZkPerformanceControls,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, gate)| ZkDerivationReview {
        gate,
        review_hash: state_machine_zk_review_hash(index as u8 + 70),
    })
    .collect()
}

fn build_state_machine_statistical_logup_components(
    stmt0: &StateMachineStatement0,
    lookup_elements: StateMachineElements,
    x_axis_masked_claim: SecureField,
    y_axis_masked_claim: SecureField,
    public_aggregate_target: SecureField,
) -> StateMachineStatisticalLogupComponents {
    let tree_span_provider = &mut TraceLocationAllocator::default();
    let component0 = StateMachineStatisticalOp0Component::new_with_logup_claim(
        tree_span_provider,
        StatisticalStateTransitionEval {
            transition: StateTransitionEval {
                log_n_rows: stmt0.n,
                lookup_elements: lookup_elements.clone(),
                claimed_sum: QM31::zero(),
            },
            constraint_log_degree_bound: stmt0.n + 1,
        },
        LogupClaim::MaskedWithPrivateCorrection(x_axis_masked_claim),
    );
    let component1 = StateMachineStatisticalOp1Component::new_with_logup_claim(
        tree_span_provider,
        StatisticalStateTransitionEval {
            transition: StateTransitionEval {
                log_n_rows: stmt0.m,
                lookup_elements,
                claimed_sum: QM31::zero(),
            },
            constraint_log_degree_bound: stmt0.m + 1,
        },
        LogupClaim::MaskedWithPrivateCorrection(y_axis_masked_claim),
    );

    let mask_offsets = {
        let component_refs = vec![
            &component0 as &dyn stwo::core::air::Component,
            &component1 as &dyn stwo::core::air::Component,
        ];
        AirComponents {
            components: component_refs,
            n_preprocessed_columns: 0,
        }
        .mask_offsets(false)
        .expect("statistical LogUp components must expose structured mask offsets")
    };

    let op0_interaction_span = component0.trace_locations()[INTERACTION_TRACE_IDX];
    let op1_interaction_span = component1.trace_locations()[INTERACTION_TRACE_IDX];
    let op0_correction_start = op0_interaction_span.col_end - SECURE_EXTENSION_DEGREE;
    let op1_correction_start = op1_interaction_span.col_end - SECURE_EXTENSION_DEGREE;
    let op0_opening_index =
        mask_offsets[op0_interaction_span.tree_index][op0_correction_start].len();
    let op1_opening_index =
        mask_offsets[op1_interaction_span.tree_index][op1_correction_start].len();

    let aggregate_component = StatisticalLogupAggregateComponent::new(
        mask_offsets.len(),
        vec![
            StatisticalLogupCorrectionRef {
                tree_index: op0_interaction_span.tree_index,
                column_start: op0_correction_start,
                opening_index: op0_opening_index,
                log_size: stmt0.n,
                masked_claim: x_axis_masked_claim,
            },
            StatisticalLogupCorrectionRef {
                tree_index: op1_interaction_span.tree_index,
                column_start: op1_correction_start,
                opening_index: op1_opening_index,
                log_size: stmt0.m,
                masked_claim: y_axis_masked_claim,
            },
        ],
        public_aggregate_target,
    );

    StateMachineStatisticalLogupComponents {
        component0,
        component1,
        aggregate_component,
    }
}

fn state_machine_statistical_logup_component_log_sizes(
    stmt0: &StateMachineStatement0,
) -> TreeVec<Vec<u32>> {
    let components = build_state_machine_statistical_logup_components(
        stmt0,
        StateMachineElements::dummy(),
        SecureField::zero(),
        SecureField::zero(),
        SecureField::zero(),
    );
    let component_refs = components.components();
    AirComponents {
        components: component_refs,
        n_preprocessed_columns: 0,
    }
    .column_log_sizes()
}

fn state_machine_statistical_logup_max_constraint_log_degree_bound(
    stmt0: &StateMachineStatement0,
) -> u32 {
    let components = build_state_machine_statistical_logup_components(
        stmt0,
        StateMachineElements::dummy(),
        SecureField::zero(),
        SecureField::zero(),
        SecureField::zero(),
    );
    components
        .components()
        .into_iter()
        .map(|component| component.max_constraint_log_degree_bound())
        .max()
        .unwrap_or(0)
}

struct StateMachineStatisticalLogupZkPrivacyProvider<'a> {
    stmt0: &'a StateMachineStatement0,
    public_input: [State; 2],
}

impl<'a> StateMachineStatisticalLogupZkPrivacyProvider<'a> {
    const fn new(stmt0: &'a StateMachineStatement0, public_input: [State; 2]) -> Self {
        Self {
            stmt0,
            public_input,
        }
    }
}

impl ZkAirPrivacyProvider for StateMachineStatisticalLogupZkPrivacyProvider<'_> {
    fn air_id(&self) -> ZkAirId {
        ZkAirId(b"stwo.examples.state-machine.statistical-logup.zk.v1".to_vec())
    }

    fn component_column_log_sizes(&self) -> TreeVec<Vec<u32>> {
        state_machine_statistical_logup_component_log_sizes(self.stmt0)
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        state_machine_statistical_logup_max_constraint_log_degree_bound(self.stmt0)
    }

    fn trace_tree_scopes(&self) -> Vec<ZkTraceTreeScope> {
        vec![
            ZkTraceTreeScope::Preprocessed,
            ZkTraceTreeScope::OriginalTrace,
            ZkTraceTreeScope::InteractionTrace {
                interaction_index: 0,
            },
        ]
    }

    fn public_roots(&self) -> Vec<ZkColumnRange> {
        vec![]
    }

    fn private_roots(&self) -> Vec<ZkPrivateRoot> {
        let log_sizes = self.component_column_log_sizes();
        zk_private_roots_for_tree(
            ORIGINAL_TRACE_IDX,
            log_sizes[ORIGINAL_TRACE_IDX].len(),
            ZkPrivateColumnUsage::OrdinaryWitness,
            ZkPrivacyReason::Witness,
        )
    }

    fn private_column_semantic_domains(&self) -> Vec<ZkPrivateColumnSemanticDomains> {
        let components = build_state_machine_statistical_logup_components(
            self.stmt0,
            StateMachineElements::dummy(),
            SecureField::zero(),
            SecureField::zero(),
            SecureField::zero(),
        );
        let aggregate_domain_log_size = self.stmt0.n.max(self.stmt0.m);
        [
            (
                components.component0.trace_locations()[INTERACTION_TRACE_IDX],
                self.stmt0.n,
            ),
            (
                components.component1.trace_locations()[INTERACTION_TRACE_IDX],
                self.stmt0.m,
            ),
        ]
        .into_iter()
        .flat_map(|(span, component_domain_log_size)| {
            let correction_start = span.col_end - SECURE_EXTENSION_DEGREE;
            (0..SECURE_EXTENSION_DEGREE).map(move |offset| ZkPrivateColumnSemanticDomains {
                range: ZkColumnRange::new(
                    span.tree_index,
                    correction_start + offset,
                    correction_start + offset + 1,
                ),
                semantic_trace_domain_log_sizes: vec![
                    component_domain_log_size,
                    aggregate_domain_log_size,
                ],
            })
        })
        .collect()
    }

    fn dependency_edges(&self) -> Vec<ZkPrivacyDependency> {
        let log_sizes = self.component_column_log_sizes();
        vec![ZkPrivacyDependency {
            from: ZkColumnRange::new(ORIGINAL_TRACE_IDX, 0, log_sizes[ORIGINAL_TRACE_IDX].len()),
            to: ZkColumnRange::new(
                INTERACTION_TRACE_IDX,
                0,
                log_sizes[INTERACTION_TRACE_IDX].len(),
            ),
            kind: ZkDependencyKind::LogUpRunningSum,
        }]
    }

    fn dependency_metadata_completeness(&self) -> ZkDependencyMetadataCompleteness {
        ZkDependencyMetadataCompleteness::CompleteTraceAndInteractionClosure
    }

    fn logup_claim_policies(&self) -> Vec<ZkLogupClaimPolicy> {
        (0..2)
            .map(|claim_index| ZkLogupClaimPolicy {
                interaction_index: 0,
                claim_index,
                visibility: ZkLogupClaimVisibility::StatisticalAggregate {
                    aggregate_id: STATE_MACHINE_STATISTICAL_LOGUP_AGGREGATE_ID,
                },
                semantic_domain: vec![],
                semantic_statement: vec![],
            })
            .collect()
    }

    fn logup_claim_manifest(&self) -> Vec<ZkLogupClaimManifestEntry> {
        vec![ZkLogupClaimManifestEntry {
            interaction_index: 0,
            claim_count: 2,
        }]
    }

    fn logup_statistical_aggregate_groups(&self) -> Vec<ZkLogupStatisticalAggregateGroup> {
        let term_count_bound = state_machine_private_lookup_term_count_bound(self.stmt0);
        vec![ZkLogupStatisticalAggregateGroup {
            aggregate_id: STATE_MACHINE_STATISTICAL_LOGUP_AGGREGATE_ID,
            target: ZkLogupStatisticalAggregateTarget::PublicExpression {
                domain: b"stwo.examples.state-machine.statistical-logup.aggregate-target.v1"
                    .to_vec(),
                statement: state_machine_public_statement_bytes(
                    self.stmt0,
                    &self.public_input,
                    b"state-machine-statistical-logup-public-target-v1",
                ),
            },
            relation_domain: b"stwo.examples.state-machine.statistical-logup.aggregate-relation.v1"
                .to_vec(),
            relation_statement: state_machine_public_statement_bytes(
                self.stmt0,
                &self.public_input,
                b"masked-claims-minus-private-corrections-equals-public-transition-target-v1",
            ),
            private_lookup_term_count_bound: term_count_bound,
            lookup_challenge_count: STATE_SIZE as u32 + 1,
            expected_proof_volume: term_count_bound.saturating_mul(8).max(1),
            min_statistical_security_bits: 80,
            safety_margin_bits: 8,
        }]
    }

    fn logup_claim_metadata_completeness(&self) -> ZkLogupClaimMetadataCompleteness {
        ZkLogupClaimMetadataCompleteness::Complete
    }

    fn application_domain(&self) -> &[u8] {
        b"stwo.examples.state-machine.statistical-logup.zk-public-statement.v1"
    }

    fn application_statement(&self) -> Vec<u8> {
        state_machine_public_statement_bytes(
            self.stmt0,
            &self.public_input,
            b"state-machine-statistical-logup-public-statement-v1",
        )
    }
}

fn zk_private_roots_for_tree(
    tree_index: usize,
    column_count: usize,
    usage: ZkPrivateColumnUsage,
    reason: ZkPrivacyReason,
) -> Vec<ZkPrivateRoot> {
    (0..column_count)
        .map(|column_index| ZkPrivateRoot {
            range: ZkColumnRange::new(tree_index, column_index, column_index + 1),
            usage,
            reason,
        })
        .collect()
}

fn state_machine_statistical_logup_zk_configs(
    stmt0: &StateMachineStatement0,
    public_input: [State; 2],
    fri_log_blowup_factor: u32,
) -> (
    ZkProvingConfig,
    ZkVerificationConfig,
    ZkWitnessRandomizationVerifierAudit,
) {
    let provider = StateMachineStatisticalLogupZkPrivacyProvider::new(stmt0, public_input);
    let artifacts = build_zk_config_from_air_privacy_provider(
        &provider,
        fri_log_blowup_factor,
        ZkPrivacyInferenceMode::FailClosed,
    )
    .expect("StateMachine statistical LogUp provider-derived ZK metadata must be valid");
    let prover_config = ZkProvingConfig {
        metadata: artifacts.metadata.clone(),
        privacy_map: artifacts.privacy_map.clone(),
        private_column_scope: Some(artifacts.verifier_audit.private_column_scope.clone()),
        quotient_split_mask_profile: artifacts.verifier_config.quotient_split_mask_profile,
        logup_statistical_security_budgets: artifacts.logup_statistical_security_budgets.clone(),
        query_closure: None,
        randomizer_rank_profile: None,
        derived_randomizer_metadata: None,
        column_degree_bounds: artifacts.column_degree_bounds.clone(),
        derivation_reviews: state_machine_zk_derivation_reviews(),
    };

    (
        prover_config,
        artifacts.verifier_config,
        artifacts.verifier_audit,
    )
}

fn state_machine_statistical_logup_committed_column_log_sizes(
    stmt0: &StateMachineStatement0,
    zk_verifier_config: &ZkVerificationConfig,
) -> Result<TreeVec<Vec<u32>>, VerificationError> {
    apply_zk_column_degree_bounds(
        state_machine_statistical_logup_component_log_sizes(stmt0),
        &zk_verifier_config
            .metadata
            .witness_randomization
            .private_column_degree_bounds,
    )
    .map_err(|err| {
        VerificationError::InvalidStructure(format!(
            "Invalid StateMachine statistical LogUp ZK column degree bounds: {err:?}"
        ))
    })
}

#[allow(unused)]
pub fn prove_state_machine_statistical_logup(
    log_n_rows: u32,
    initial_state: State,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
    mask_seed: u64,
) -> (
    StateMachineStatisticalLogupComponents,
    StateMachineStatisticalLogupProof<Blake2sMerkleHasher>,
) {
    let (x_axis_log_rows, y_axis_log_rows) = (log_n_rows, log_n_rows - 1);
    assert!(y_axis_log_rows >= LOG_N_LANES && x_axis_log_rows >= LOG_N_LANES);
    let mut config = config;

    let mut intermediate_state = initial_state;
    intermediate_state[0] += M31::from_u32_unchecked(1 << x_axis_log_rows);
    let mut final_state = intermediate_state;
    final_state[1] += M31::from_u32_unchecked(1 << y_axis_log_rows);

    let stmt0 = StateMachineStatement0 {
        n: x_axis_log_rows,
        m: y_axis_log_rows,
    };
    let public_input = [initial_state, final_state];
    let (zk_prover_config, ..) = state_machine_statistical_logup_zk_configs(
        &stmt0,
        public_input,
        config.fri_config.log_blowup_factor,
    );
    config.lifting_log_size = Some(
        zk_prover_config
            .metadata
            .degree_profile
            .fri_first_layer_log_size,
    );

    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(
            zk_prover_config
                .metadata
                .degree_profile
                .fri_first_layer_log_size,
        )
        .circle_domain()
        .half_coset,
    );

    config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_store_polynomials_coefficients();

    let trace_op0 = gen_trace(x_axis_log_rows, initial_state, 0);
    let trace_op1 = gen_trace(y_axis_log_rows, intermediate_state, 1);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.commit(channel);

    stmt0.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![trace_op0.clone(), trace_op1.clone()].collect_vec());
    let mut witness_rng = StdRng::seed_from_u64(mask_seed ^ 0x9e37_79b9_7f4a_7c15);
    tree_builder
        .commit_zk_witness_randomized(&zk_prover_config, &mut witness_rng, channel)
        .unwrap();

    mix_zk_public_metadata(
        channel,
        &zk_prover_config.metadata,
        &zk_prover_config.column_degree_bounds,
    );
    let lookup_elements = StateMachineElements::draw(channel);
    let public_aggregate_target =
        state_machine_public_aggregate_target(&lookup_elements, &public_input);

    let mut mask_rng = StdRng::seed_from_u64(mask_seed);
    let x_axis_masked_claim: SecureField = mask_rng.gen();
    let y_axis_masked_claim = public_aggregate_target - x_axis_masked_claim;

    let (interaction_trace_op0, ..) =
        gen_masked_interaction_trace(&trace_op0, 0, &lookup_elements, x_axis_masked_claim);
    let (interaction_trace_op1, ..) =
        gen_masked_interaction_trace(&trace_op1, 1, &lookup_elements, y_axis_masked_claim);

    let stmt1 = StateMachineStatisticalLogupStatement1 {
        x_axis_masked_claim,
        y_axis_masked_claim,
    };
    stmt1.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![interaction_trace_op0, interaction_trace_op1].collect_vec());
    tree_builder
        .commit_zk_witness_randomized(&zk_prover_config, &mut witness_rng, channel)
        .unwrap();

    let components = build_state_machine_statistical_logup_components(
        &stmt0,
        lookup_elements,
        x_axis_masked_claim,
        y_axis_masked_claim,
        public_aggregate_target,
    );

    let mut proof_rng = StdRng::seed_from_u64(mask_seed ^ 0xd1b5_4a32_d192_ed03);
    let stark_proof = prove_zk(
        &components.component_provers(),
        channel,
        commitment_scheme,
        &zk_prover_config,
        &mut proof_rng,
    )
    .unwrap();
    let proof = StateMachineStatisticalLogupProof {
        public_input,
        stmt0,
        stmt1,
        stark_proof,
    };
    (components, proof)
}

pub fn verify_state_machine_statistical_logup(
    channel: &mut Blake2sChannel,
    proof: StateMachineStatisticalLogupProof<Blake2sMerkleHasher>,
) -> Result<(), VerificationError> {
    let pcs_config = proof.stark_proof.0.randomized_pcs_proof.config;
    let (_, zk_verifier_config, zk_verifier_audit) = state_machine_statistical_logup_zk_configs(
        &proof.stmt0,
        proof.public_input,
        pcs_config.fri_config.log_blowup_factor,
    );
    verify_state_machine_statistical_logup_with_zk_artifacts(
        channel,
        proof,
        zk_verifier_config,
        zk_verifier_audit,
    )
}

fn verify_state_machine_statistical_logup_with_zk_artifacts(
    channel: &mut Blake2sChannel,
    proof: StateMachineStatisticalLogupProof<Blake2sMerkleHasher>,
    zk_verifier_config: ZkVerificationConfig,
    zk_verifier_audit: ZkWitnessRandomizationVerifierAudit,
) -> Result<(), VerificationError> {
    let pcs_config = proof.stark_proof.0.randomized_pcs_proof.config;
    let committed_sizes = state_machine_statistical_logup_committed_column_log_sizes(
        &proof.stmt0,
        &zk_verifier_config,
    )?;

    pcs_config.mix_into(channel);
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);
    let commitments = &proof.stark_proof.0.randomized_pcs_proof.commitments;
    if commitments.len() < 3 {
        return Err(VerificationError::InvalidStructure(String::from(
            "StateMachine statistical LogUp ZK proof is missing trace commitments",
        )));
    }

    commitment_scheme.commit(commitments[0], &committed_sizes[0], channel);
    proof.stmt0.mix_into(channel);
    commitment_scheme.commit(commitments[1], &committed_sizes[1], channel);

    mix_zk_public_metadata(
        channel,
        &zk_verifier_config.metadata,
        &zk_verifier_config.column_degree_bounds,
    );
    let lookup_elements = StateMachineElements::draw(channel);
    let public_aggregate_target =
        state_machine_public_aggregate_target(&lookup_elements, &proof.public_input);
    if proof.stmt1.x_axis_masked_claim + proof.stmt1.y_axis_masked_claim != public_aggregate_target
    {
        return Err(VerificationError::InvalidStructure(String::from(
            "StateMachine statistical LogUp masked claims do not match public aggregate target",
        )));
    }
    let components = build_state_machine_statistical_logup_components(
        &proof.stmt0,
        lookup_elements,
        proof.stmt1.x_axis_masked_claim,
        proof.stmt1.y_axis_masked_claim,
        public_aggregate_target,
    );
    let component_refs = components.components();

    proof.stmt1.mix_into(channel);
    commitment_scheme.commit(commitments[2], &committed_sizes[2], channel);

    verify_zk_with_witness_randomization_audit(
        &component_refs,
        channel,
        commitment_scheme,
        proof.stark_proof,
        &zk_verifier_config,
        &zk_verifier_audit,
    )
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use stwo::core::channel::Blake2sChannel;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;
    use stwo::core::fields::FieldExpOps;
    use stwo::core::pcs::{PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::zk::{
        build_zk_air_metadata_from_privacy_provider, canonical_zk_private_column_scope_hash,
        zk_singleton_column_ranges, ZkAirId, ZkAirMetadataBuildError, ZkAirPrivacyProvider,
        ZkColumnRange, ZkDependencyKind, ZkDependencyMetadataCompleteness, ZkPrivacyDependency,
        ZkPrivacyInferenceMode, ZkPrivacyReason, ZkPrivateColumnUsage, ZkPrivateRoot,
        ZkTraceTreeScope,
    };
    use stwo_constraint_framework::expr::ExprEvaluator;
    use stwo_constraint_framework::{
        assert_constraints_on_polys, FrameworkEval, Relation, TraceLocationAllocator,
        INTERACTION_TRACE_IDX,
    };

    use super::components::{
        State, StateMachineElements, StateMachineOp0Component, StateMachineStatement0,
        StateTransitionEval, STATE_SIZE,
    };
    use super::gen::{gen_interaction_trace, gen_trace};
    use super::{
        prove_state_machine, prove_state_machine_statistical_logup, verify_state_machine,
        verify_state_machine_statistical_logup,
        verify_state_machine_statistical_logup_with_zk_artifacts,
    };

    struct StateMachineZkPrivacyProvider<'a> {
        stmt0: &'a StateMachineStatement0,
        public_input: [State; 2],
    }

    impl<'a> StateMachineZkPrivacyProvider<'a> {
        const fn new(stmt0: &'a StateMachineStatement0, public_input: [State; 2]) -> Self {
            Self {
                stmt0,
                public_input,
            }
        }
    }

    fn state_machine_public_input(stmt0: &StateMachineStatement0) -> [State; 2] {
        let initial_state = [M31::zero(); STATE_SIZE];
        let final_state = [
            M31::from_u32_unchecked(1 << stmt0.n),
            M31::from_u32_unchecked(1 << stmt0.m),
        ];

        [initial_state, final_state]
    }

    fn state_machine_max_constraint_log_degree_bound(stmt0: &StateMachineStatement0) -> u32 {
        let op0_bound = StateTransitionEval::<0> {
            log_n_rows: stmt0.n,
            lookup_elements: StateMachineElements::dummy(),
            claimed_sum: QM31::zero(),
        }
        .max_constraint_log_degree_bound();
        let op1_bound = StateTransitionEval::<1> {
            log_n_rows: stmt0.m,
            lookup_elements: StateMachineElements::dummy(),
            claimed_sum: QM31::zero(),
        }
        .max_constraint_log_degree_bound();

        op0_bound.max(op1_bound)
    }

    impl ZkAirPrivacyProvider for StateMachineZkPrivacyProvider<'_> {
        fn air_id(&self) -> ZkAirId {
            ZkAirId(b"stwo.examples.state-machine.private-logup.blocked.zk.v1".to_vec())
        }

        fn component_column_log_sizes(&self) -> TreeVec<Vec<u32>> {
            self.stmt0.log_sizes()
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            state_machine_max_constraint_log_degree_bound(self.stmt0)
        }

        fn trace_tree_scopes(&self) -> Vec<ZkTraceTreeScope> {
            vec![
                ZkTraceTreeScope::Preprocessed,
                ZkTraceTreeScope::OriginalTrace,
                ZkTraceTreeScope::InteractionTrace {
                    interaction_index: 0,
                },
            ]
        }

        fn public_roots(&self) -> Vec<ZkColumnRange> {
            vec![]
        }

        fn private_roots(&self) -> Vec<ZkPrivateRoot> {
            let log_sizes = self.stmt0.log_sizes();
            zk_singleton_column_ranges(1, log_sizes[1].len())
                .into_iter()
                .map(|range| ZkPrivateRoot {
                    range,
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                    reason: ZkPrivacyReason::Witness,
                })
                .collect()
        }

        fn dependency_edges(&self) -> Vec<ZkPrivacyDependency> {
            let log_sizes = self.stmt0.log_sizes();
            if log_sizes[1].is_empty() || log_sizes[2].is_empty() {
                return vec![];
            }

            vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, log_sizes[1].len()),
                to: ZkColumnRange::new(2, 0, log_sizes[2].len()),
                kind: ZkDependencyKind::LogUpRunningSum,
            }]
        }

        fn dependency_metadata_completeness(&self) -> ZkDependencyMetadataCompleteness {
            ZkDependencyMetadataCompleteness::CompleteTraceAndInteractionClosure
        }

        fn application_domain(&self) -> &[u8] {
            b"stwo.examples.state-machine.zk-public-statement.v1"
        }

        fn application_statement(&self) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"state-machine-public-transition-private-logup-blocked-v2");
            bytes.extend_from_slice(&self.stmt0.n.to_le_bytes());
            bytes.extend_from_slice(&self.stmt0.m.to_le_bytes());
            for state in self.public_input {
                for value in state {
                    bytes.extend_from_slice(&value.0.to_le_bytes());
                }
            }
            let log_sizes = self.stmt0.log_sizes();
            bytes.extend_from_slice(&(log_sizes.0.len() as u64).to_le_bytes());
            for tree_log_sizes in &log_sizes.0 {
                bytes.extend_from_slice(&(tree_log_sizes.len() as u64).to_le_bytes());
                for log_size in tree_log_sizes {
                    bytes.extend_from_slice(&log_size.to_le_bytes());
                }
            }
            bytes
        }
    }

    #[test]
    fn test_state_machine_constraints() {
        let log_n_rows = 8;
        let initial_state = [M31::zero(); STATE_SIZE];

        let trace = gen_trace(log_n_rows, initial_state, 0);
        let lookup_elements = StateMachineElements::draw(&mut Blake2sChannel::default());

        // Interaction trace.
        let (interaction_trace, claimed_sum) = gen_interaction_trace(&trace, 0, &lookup_elements);

        let component = StateMachineOp0Component::new(
            &mut TraceLocationAllocator::default(),
            StateTransitionEval {
                log_n_rows,
                lookup_elements,
                claimed_sum,
            },
            claimed_sum,
        );

        let trace = TreeVec::new(vec![vec![], trace, interaction_trace]);
        let trace_polys = trace.map_cols(|c| c.interpolate());
        let component_eval = component.clone();
        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(log_n_rows),
            |assert_eval| {
                component_eval.evaluate(assert_eval);
            },
            claimed_sum,
        );
    }

    #[test]
    fn test_state_machine_claimed_sum() {
        let log_n_rows = 8;
        let config = PcsConfig::default();

        // Initial and last state.
        let initial_state = [M31::zero(); STATE_SIZE];
        let last_state = [
            M31::from_u32_unchecked(1 << log_n_rows),
            M31::from_u32_unchecked(1 << (log_n_rows - 1)),
        ];

        // Setup protocol.
        let channel = &mut Blake2sChannel::default();
        let (component, ..) =
            prove_state_machine(log_n_rows, initial_state, config, channel, false);

        let interaction_elements = component.component0.lookup_elements.clone();
        let initial_state_comb: QM31 = interaction_elements.combine(&initial_state);
        let last_state_comb: QM31 = interaction_elements.combine(&last_state);

        assert_eq!(
            component.component0.claimed_sum + component.component1.claimed_sum,
            initial_state_comb.inverse() - last_state_comb.inverse()
        );
    }

    #[test]
    fn test_relation_tracker() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];
        let final_state = [
            M31::from_u32_unchecked(1 << log_n_rows),
            M31::from_u32_unchecked(1 << (log_n_rows - 1)),
        ];

        // Summarize `StateMachineElements`.
        let (_, _, summary) = prove_state_machine(
            log_n_rows,
            initial_state,
            config,
            &mut Blake2sChannel::default(),
            true,
        );
        let summary = summary.unwrap();
        let relation_info = summary.get_relation_info("StateMachineElements").unwrap();

        // Check the final state inferred from the summary.
        let mut curr_state = initial_state;
        for entry in relation_info {
            let (x_step, y_step) = match entry.0.len() {
                2 => (entry.0[0], entry.0[1]),
                1 => (entry.0[0], M31::zero()),
                0 => (M31::zero(), M31::zero()),
                _ => unreachable!(),
            };
            let mult = entry.1;
            let next_state = [curr_state[0] - x_step * mult, curr_state[1] - y_step * mult];

            curr_state = next_state;
        }

        assert_eq!(curr_state, final_state);
    }

    #[test]
    fn test_state_machine_prove() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];
        let prover_channel = &mut Blake2sChannel::default();
        let verifier_channel = &mut Blake2sChannel::default();

        let (components, proof, _) =
            prove_state_machine(log_n_rows, initial_state, config, prover_channel, false);

        verify_state_machine(verifier_channel, components, proof).unwrap();
    }

    #[test]
    fn test_state_machine_statistical_logup_prove() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];

        let (_, proof_a) = prove_state_machine_statistical_logup(
            log_n_rows,
            initial_state,
            config,
            &mut Blake2sChannel::default(),
            101,
        );
        let masked_claims_a = (
            proof_a.stmt1.x_axis_masked_claim,
            proof_a.stmt1.y_axis_masked_claim,
        );
        let interaction_samples_a = proof_a.stark_proof.0.randomized_pcs_proof.sampled_values
            [INTERACTION_TRACE_IDX]
            .clone();
        verify_state_machine_statistical_logup(&mut Blake2sChannel::default(), proof_a).unwrap();

        let (_, proof_b) = prove_state_machine_statistical_logup(
            log_n_rows,
            initial_state,
            config,
            &mut Blake2sChannel::default(),
            202,
        );
        let masked_claims_b = (
            proof_b.stmt1.x_axis_masked_claim,
            proof_b.stmt1.y_axis_masked_claim,
        );
        let interaction_samples_b = proof_b.stark_proof.0.randomized_pcs_proof.sampled_values
            [INTERACTION_TRACE_IDX]
            .clone();
        verify_state_machine_statistical_logup(&mut Blake2sChannel::default(), proof_b).unwrap();

        assert_ne!(masked_claims_a, masked_claims_b);
        assert_ne!(interaction_samples_a, interaction_samples_b);
    }

    #[test]
    fn test_state_machine_statistical_logup_rejects_tampered_masked_claim() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];

        let (_, mut proof) = prove_state_machine_statistical_logup(
            log_n_rows,
            initial_state,
            config,
            &mut Blake2sChannel::default(),
            303,
        );
        proof.stmt1.x_axis_masked_claim += QM31::from_u32_unchecked(1, 0, 0, 0);

        assert!(
            verify_state_machine_statistical_logup(&mut Blake2sChannel::default(), proof).is_err()
        );
    }

    #[test]
    fn test_state_machine_statistical_logup_rejects_wrong_private_column_domain_metadata() {
        let log_n_rows = 8;
        let config = PcsConfig::default();
        let initial_state = [M31::zero(); STATE_SIZE];

        let (_, proof) = prove_state_machine_statistical_logup(
            log_n_rows,
            initial_state,
            config,
            &mut Blake2sChannel::default(),
            404,
        );
        let pcs_config = proof.stark_proof.0.randomized_pcs_proof.config;
        let (_, mut zk_verifier_config, mut zk_verifier_audit) =
            super::state_machine_statistical_logup_zk_configs(
                &proof.stmt0,
                proof.public_input,
                pcs_config.fri_config.log_blowup_factor,
            );
        let entry_index = zk_verifier_audit
            .private_column_scope
            .entries
            .iter()
            .position(|entry| entry.trace_domain_log_size == proof.stmt0.m)
            .expect("mixed-domain StateMachine proof must contain a smaller private domain");
        zk_verifier_audit.private_column_scope.entries[entry_index].trace_domain_log_size =
            proof.stmt0.n;
        zk_verifier_audit.private_column_scope.hash =
            canonical_zk_private_column_scope_hash(&zk_verifier_audit.private_column_scope);
        zk_verifier_config
            .metadata
            .witness_randomization
            .private_column_scope_hash = zk_verifier_audit.private_column_scope.hash;

        assert!(verify_state_machine_statistical_logup_with_zk_artifacts(
            &mut Blake2sChannel::default(),
            proof,
            zk_verifier_config,
            zk_verifier_audit,
        )
        .is_err());
    }

    #[test]
    fn test_state_machine_constraint_repr() {
        let log_n_rows = 8;
        let initial_state = [M31::zero(); STATE_SIZE];

        let trace = gen_trace(log_n_rows, initial_state, 0);
        let lookup_elements = StateMachineElements::draw(&mut Blake2sChannel::default());

        let (_, claimed_sum) = gen_interaction_trace(&trace, 0, &lookup_elements);

        let component = StateMachineOp0Component::new(
            &mut TraceLocationAllocator::default(),
            StateTransitionEval {
                log_n_rows,
                lookup_elements,
                claimed_sum,
            },
            claimed_sum,
        );

        let eval = component.evaluate(ExprEvaluator::new());
        let expected = "let intermediate0 = (StateMachineElements_alpha0) * (trace_1_column_0_offset_0) \
            + (StateMachineElements_alpha1) * (trace_1_column_1_offset_0) \
            - (StateMachineElements_z);

\
        let intermediate1 = (StateMachineElements_alpha0) * (trace_1_column_0_offset_0 + m31(1).into()) \
            + (StateMachineElements_alpha1) * (trace_1_column_1_offset_0) \
            - (StateMachineElements_z);

\
        let constraint_0 = (QM31Impl::from_partial_evals([trace_2_column_2_offset_0, trace_2_column_3_offset_0, trace_2_column_4_offset_0, trace_2_column_5_offset_0]) \
            - (QM31Impl::from_partial_evals([trace_2_column_2_offset_neg_1, trace_2_column_3_offset_neg_1, trace_2_column_4_offset_neg_1, trace_2_column_5_offset_neg_1])) \
                + (claimed_sum) * (1 / (column_size))\
            ) \
            * ((intermediate0) * (intermediate1)) \
            - (intermediate1 - (intermediate0));"
            .to_string();

        assert_eq!(eval.format_constraints(), expected);
    }

    #[test]
    fn test_logup_counts() {
        let log_n_rows = 8;
        let initial_state = [M31::zero(); STATE_SIZE];
        let (components, ..) = prove_state_machine(
            log_n_rows,
            initial_state,
            PcsConfig::default(),
            &mut Blake2sChannel::default(),
            false,
        );

        let counts0 = components.component0.logup_counts();
        let counts1 = components.component1.logup_counts();

        assert_eq!(counts0["StateMachineElements"], (1 << log_n_rows) * 2);
        assert_eq!(counts1["StateMachineElements"], (1 << (log_n_rows - 1)) * 2);
    }

    #[test]
    fn test_state_machine_zk_provider_uses_statement_log_sizes_and_air_bounds() {
        let stmt0 = StateMachineStatement0 { n: 8, m: 7 };
        let provider =
            StateMachineZkPrivacyProvider::new(&stmt0, state_machine_public_input(&stmt0));
        let log_sizes = stmt0.log_sizes();

        assert_eq!(provider.component_column_log_sizes().0, log_sizes.0);
        assert_eq!(
            provider.trace_tree_scopes(),
            vec![
                ZkTraceTreeScope::Preprocessed,
                ZkTraceTreeScope::OriginalTrace,
                ZkTraceTreeScope::InteractionTrace {
                    interaction_index: 0,
                },
            ]
        );
        assert!(provider.public_roots().is_empty());
        assert_eq!(
            provider.max_constraint_log_degree_bound(),
            state_machine_max_constraint_log_degree_bound(&stmt0)
        );

        let log_sizes = stmt0.log_sizes();
        let private_roots = provider.private_roots();
        assert_eq!(private_roots.len(), log_sizes[1].len());
        for (column, root) in private_roots.iter().enumerate() {
            assert_eq!(root.range, ZkColumnRange::new(1, column, column + 1));
            assert!(root.range.is_singleton());
            assert_eq!(root.usage, ZkPrivateColumnUsage::OrdinaryWitness);
            assert_eq!(root.reason, ZkPrivacyReason::Witness);
        }

        assert_eq!(
            provider.dependency_edges(),
            vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, log_sizes[1].len()),
                to: ZkColumnRange::new(2, 0, log_sizes[2].len()),
                kind: ZkDependencyKind::LogUpRunningSum,
            }]
        );
    }

    #[test]
    fn test_state_machine_zk_provider_fails_closed_on_private_logup_claims() {
        let stmt0 = StateMachineStatement0 { n: 8, m: 7 };
        let provider =
            StateMachineZkPrivacyProvider::new(&stmt0, state_machine_public_input(&stmt0));

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                PcsConfig::default().fri_config.log_blowup_factor,
                ZkPrivacyInferenceMode::FailClosed,
            ),
            Err(ZkAirMetadataBuildError::IncompleteLogupClaimMetadata)
        ));
    }
}
