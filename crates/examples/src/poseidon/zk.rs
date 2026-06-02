//! Reusable Poseidon ZK metadata and transcript helpers.
//!
//! This module derives Poseidon ZK geometry for private original-trace columns.
//! Full private LogUp is intentionally fail-closed until a reviewed protocol
//! hides witness-derived LogUp claimed-sum scalars.

use stwo::core::air::Component;
use stwo::core::channel::Channel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::TreeVec;
use stwo::core::zk::{
    build_zk_air_metadata_from_privacy_provider, build_zk_config_from_air_privacy_provider,
    derive_stwo_zk_air_degree_bounds, mix_zk_public_metadata,
    zk_masked_private_constraint_log_expansion, zk_public_air_constraint_log_expansion_from_bound,
    zk_singleton_column_ranges, zk_trace_domain_log_size_from_column_bounds,
    ZkAirCanonicalMetadata, ZkAirDegreeBounds, ZkAirId, ZkAirMetadataBuildError,
    ZkAirPrivacyProvider, ZkColumnDegreeBound, ZkColumnRange, ZkDependencyKind,
    ZkDependencyMetadataCompleteness, ZkPrivacyDependency, ZkPrivacyInferenceMode, ZkPrivacyReason,
    ZkPrivateColumnScope, ZkPrivateColumnUsage, ZkPrivateRoot, ZkPublicStatementHash,
    ZkQuotientSplitMaskProfile, ZkTraceTreeScope, ZkTraceTreeScopeBinding, ZkVerificationConfig,
    ZkWitnessRandomizationVerifierAudit,
};
use stwo::core::ColumnVec;
use stwo::prover::zk::{ZkDerivationReview, ZkProvingConfig};
use stwo_constraint_framework::TraceLocationAllocator;

use super::{
    PoseidonComponent, PoseidonElements, PoseidonEval, LOG_EXPAND, N_LOG_INSTANCES_PER_ROW,
};

pub type PoseidonZkDegreeBounds = ZkAirDegreeBounds;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseidonZkMetadataError {
    InvalidLogInstances,
    EmptyTraceMetadata,
    TraceDomainMismatch {
        expected: u32,
        actual: u32,
    },
    MissingTraceTrees {
        actual: usize,
    },
    ConstraintDegreeBound {
        trace_log_degree: u32,
        max_constraint_log_degree_bound: u32,
    },
    DegreeOverflow,
    FriBatchDegree,
    QuotientSplitMaskProfile,
    Air(ZkAirMetadataBuildError),
}

impl From<ZkAirMetadataBuildError> for PoseidonZkMetadataError {
    fn from(error: ZkAirMetadataBuildError) -> Self {
        match error {
            ZkAirMetadataBuildError::EmptyTraceMetadata => Self::EmptyTraceMetadata,
            ZkAirMetadataBuildError::TraceDomainMismatch { expected, actual } => {
                Self::TraceDomainMismatch { expected, actual }
            }
            ZkAirMetadataBuildError::ConstraintDegreeBound {
                trace_log_degree,
                max_constraint_log_degree_bound,
            } => Self::ConstraintDegreeBound {
                trace_log_degree,
                max_constraint_log_degree_bound,
            },
            ZkAirMetadataBuildError::DegreeOverflow => Self::DegreeOverflow,
            ZkAirMetadataBuildError::FriBatchDegree => Self::FriBatchDegree,
            ZkAirMetadataBuildError::QuotientSplitMaskProfile => Self::QuotientSplitMaskProfile,
            error => Self::Air(error),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PoseidonZkCanonicalMetadata {
    pub air_metadata: ZkAirCanonicalMetadata,
    pub component_column_log_sizes: TreeVec<ColumnVec<u32>>,
    pub trace_tree_scope_bindings: Vec<ZkTraceTreeScopeBinding>,
    pub trace_tree_scope_hash: [u8; 32],
    pub preprocessed_public_ranges: Vec<ZkColumnRange>,
    pub original_trace_private_ranges: Vec<ZkColumnRange>,
    pub interaction_trace_private_ranges: Vec<ZkColumnRange>,
    pub composition_split_range: ZkColumnRange,
    pub private_ranges: Vec<ZkColumnRange>,
    pub private_column_scope: ZkPrivateColumnScope,
    pub trace_domain_log_size: u32,
    pub randomized_witness_log_degree: u32,
    pub fri_first_layer_log_size: u32,
    pub quotient_degree_bound: ZkColumnDegreeBound,
    pub quotient_split_mask_profile: ZkQuotientSplitMaskProfile,
    pub public_statement_hash: ZkPublicStatementHash,
}

#[must_use]
pub const fn ceil_log2_usize(value: usize) -> u32 {
    let mut log = 0;
    let mut size = 1usize;
    while size < value {
        size <<= 1;
        log += 1;
    }
    log
}

#[must_use]
pub const fn max_u32(lhs: u32, rhs: u32) -> u32 {
    if lhs > rhs {
        lhs
    } else {
        rhs
    }
}

#[must_use]
pub fn derive_poseidon_zk_degree_bounds(
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
    private_constraint_degree: usize,
    fri_log_blowup_factor: u32,
) -> PoseidonZkDegreeBounds {
    derive_poseidon_zk_degree_bounds_from_log_expansion(
        trace_log_degree,
        randomized_private_column_log_degree,
        ceil_log2_usize(private_constraint_degree),
        fri_log_blowup_factor,
    )
}

#[must_use]
pub fn derive_poseidon_zk_degree_bounds_from_log_expansion(
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
    private_constraint_log_expansion: u32,
    fri_log_blowup_factor: u32,
) -> PoseidonZkDegreeBounds {
    try_derive_poseidon_zk_degree_bounds_from_log_expansion(
        trace_log_degree,
        randomized_private_column_log_degree,
        private_constraint_log_expansion,
        fri_log_blowup_factor,
    )
    .expect("Poseidon ZK degree geometry must fit supported domains")
}

pub fn try_derive_poseidon_zk_degree_bounds_from_log_expansion(
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
    private_constraint_log_expansion: u32,
    fri_log_blowup_factor: u32,
) -> Result<PoseidonZkDegreeBounds, PoseidonZkMetadataError> {
    Ok(derive_stwo_zk_air_degree_bounds(
        trace_log_degree,
        randomized_private_column_log_degree,
        LOG_EXPAND,
        private_constraint_log_expansion,
        fri_log_blowup_factor,
        stwo::core::verifier::COMPOSITION_LOG_SPLIT,
    )?)
}

#[must_use]
pub fn poseidon_zk_masked_private_constraint_log_expansion(
    public_air_constraint_log_expansion: u32,
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
) -> u32 {
    zk_masked_private_constraint_log_expansion(
        public_air_constraint_log_expansion,
        trace_log_degree,
        randomized_private_column_log_degree,
    )
    .expect("Poseidon reviewed private constraint expansion must fit u32")
}

pub fn poseidon_zk_private_constraint_log_expansion(
    component: &PoseidonComponent,
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
) -> Result<u32, PoseidonZkMetadataError> {
    let max_constraint_log_degree_bound = component.max_constraint_log_degree_bound();
    let public_air_constraint_log_expansion = zk_public_air_constraint_log_expansion_from_bound(
        trace_log_degree,
        max_constraint_log_degree_bound,
    )?;

    Ok(poseidon_zk_masked_private_constraint_log_expansion(
        public_air_constraint_log_expansion,
        trace_log_degree,
        randomized_private_column_log_degree,
    ))
}

pub fn poseidon_zk_log_n_rows(log_n_instances: u32) -> Result<u32, PoseidonZkMetadataError> {
    if log_n_instances < N_LOG_INSTANCES_PER_ROW as u32 {
        return Err(PoseidonZkMetadataError::InvalidLogInstances);
    }

    Ok(log_n_instances - N_LOG_INSTANCES_PER_ROW as u32)
}

#[must_use]
pub fn poseidon_zk_randomized_log_degree(trace_log_degree: u32) -> u32 {
    trace_log_degree + 1
}

pub fn poseidon_zk_metadata_component(log_n_rows: u32) -> PoseidonComponent {
    PoseidonComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonEval {
            log_n_rows,
            lookup_elements: PoseidonElements::dummy(),
            claimed_sum: SecureField::default(),
        },
        SecureField::default(),
    )
}

pub struct PoseidonZkPrivacyProvider<'a> {
    component: &'a PoseidonComponent,
}

impl<'a> PoseidonZkPrivacyProvider<'a> {
    #[must_use]
    pub const fn new(component: &'a PoseidonComponent) -> Self {
        Self { component }
    }
}

impl ZkAirPrivacyProvider for PoseidonZkPrivacyProvider<'_> {
    fn air_id(&self) -> ZkAirId {
        ZkAirId(b"stwo.examples.poseidon.component.zk.v1".to_vec())
    }

    fn component_column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        self.component.trace_log_degree_bounds()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.component.max_constraint_log_degree_bound()
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
        let component_column_log_sizes = self.component.trace_log_degree_bounds();
        if component_column_log_sizes[0].is_empty() {
            vec![]
        } else {
            vec![ZkColumnRange::new(
                0,
                0,
                component_column_log_sizes[0].len(),
            )]
        }
    }

    fn private_roots(&self) -> Vec<ZkPrivateRoot> {
        let component_column_log_sizes = self.component.trace_log_degree_bounds();
        zk_singleton_column_ranges(1, component_column_log_sizes[1].len())
            .into_iter()
            .map(|range| ZkPrivateRoot {
                range,
                usage: ZkPrivateColumnUsage::OrdinaryWitness,
                reason: ZkPrivacyReason::Witness,
            })
            .collect()
    }

    fn dependency_edges(&self) -> Vec<ZkPrivacyDependency> {
        let component_column_log_sizes = self.component.trace_log_degree_bounds();
        if component_column_log_sizes[1].is_empty() || component_column_log_sizes[2].is_empty() {
            return vec![];
        }

        vec![ZkPrivacyDependency {
            from: ZkColumnRange::new(1, 0, component_column_log_sizes[1].len()),
            to: ZkColumnRange::new(2, 0, component_column_log_sizes[2].len()),
            kind: ZkDependencyKind::LogUpRunningSum,
        }]
    }

    fn dependency_metadata_completeness(&self) -> ZkDependencyMetadataCompleteness {
        ZkDependencyMetadataCompleteness::CompleteTraceAndInteractionClosure
    }

    fn application_domain(&self) -> &[u8] {
        b"stwo.examples.poseidon.zk-public-statement.v1"
    }

    fn application_statement(&self) -> Vec<u8> {
        b"private-original-trace-and-logup-interaction".to_vec()
    }
}

pub fn poseidon_zk_canonical_metadata(
    component: &PoseidonComponent,
    fri_log_blowup_factor: u32,
) -> Result<PoseidonZkCanonicalMetadata, PoseidonZkMetadataError> {
    let component_column_log_sizes = component.trace_log_degree_bounds();
    if component_column_log_sizes.len() < 3 {
        return Err(PoseidonZkMetadataError::MissingTraceTrees {
            actual: component_column_log_sizes.len(),
        });
    }
    let trace_domain_log_size =
        zk_trace_domain_log_size_from_column_bounds(&component_column_log_sizes)
            .ok_or(PoseidonZkMetadataError::EmptyTraceMetadata)?;
    if trace_domain_log_size != component.log_n_rows {
        return Err(PoseidonZkMetadataError::TraceDomainMismatch {
            expected: component.log_n_rows,
            actual: trace_domain_log_size,
        });
    }

    let original_trace_private_ranges =
        zk_singleton_column_ranges(1, component_column_log_sizes[1].len());
    let interaction_trace_private_ranges =
        zk_singleton_column_ranges(2, component_column_log_sizes[2].len());
    let provider = PoseidonZkPrivacyProvider::new(component);
    let air_metadata = build_zk_air_metadata_from_privacy_provider(
        &provider,
        fri_log_blowup_factor,
        ZkPrivacyInferenceMode::FailClosed,
    )?;

    Ok(PoseidonZkCanonicalMetadata {
        component_column_log_sizes: air_metadata.component_column_log_sizes.clone(),
        trace_tree_scope_bindings: air_metadata.trace_tree_scope_bindings.clone(),
        trace_tree_scope_hash: air_metadata.trace_tree_scope_hash,
        preprocessed_public_ranges: air_metadata.public_ranges.clone(),
        original_trace_private_ranges,
        interaction_trace_private_ranges,
        composition_split_range: air_metadata.composition_split_range,
        private_ranges: air_metadata.private_ranges.clone(),
        private_column_scope: air_metadata.private_column_scope.clone(),
        trace_domain_log_size: air_metadata.trace_domain_log_size,
        randomized_witness_log_degree: air_metadata.randomized_witness_log_degree,
        fri_first_layer_log_size: air_metadata.fri_first_layer_log_size,
        quotient_degree_bound: air_metadata.quotient_degree_bound,
        quotient_split_mask_profile: air_metadata.quotient_split_mask_profile,
        public_statement_hash: air_metadata.public_statement_hash,
        air_metadata,
    })
}

pub fn poseidon_zk_configs(
    component: &PoseidonComponent,
    fri_log_blowup_factor: u32,
    derivation_reviews: Vec<ZkDerivationReview>,
) -> Result<
    (
        ZkProvingConfig,
        ZkVerificationConfig,
        ZkWitnessRandomizationVerifierAudit,
        PoseidonZkCanonicalMetadata,
    ),
    PoseidonZkMetadataError,
> {
    let canonical_metadata = poseidon_zk_canonical_metadata(component, fri_log_blowup_factor)?;
    let provider = PoseidonZkPrivacyProvider::new(component);
    let artifacts = build_zk_config_from_air_privacy_provider(
        &provider,
        fri_log_blowup_factor,
        ZkPrivacyInferenceMode::FailClosed,
    )?;
    let prover_config = ZkProvingConfig {
        metadata: artifacts.metadata.clone(),
        privacy_map: artifacts.privacy_map.clone(),
        private_column_scope: Some(canonical_metadata.private_column_scope.clone()),
        quotient_split_mask_profile: Some(canonical_metadata.quotient_split_mask_profile),
        logup_statistical_security_budgets: Vec::new(),
        query_closure: None,
        randomizer_rank_profile: None,
        derived_randomizer_metadata: None,
        column_degree_bounds: artifacts.column_degree_bounds.clone(),
        derivation_reviews,
    };

    Ok((
        prover_config,
        artifacts.verifier_config,
        artifacts.verifier_audit,
        canonical_metadata,
    ))
}

pub fn mix_poseidon_zk_prover_metadata_before_lookup<C: Channel>(
    channel: &mut C,
    config: &ZkProvingConfig,
) {
    mix_zk_public_metadata(channel, &config.metadata, &config.column_degree_bounds);
}

pub fn mix_poseidon_zk_verifier_metadata_before_lookup<C: Channel>(
    channel: &mut C,
    config: &ZkVerificationConfig,
) {
    mix_zk_public_metadata(channel, &config.metadata, &config.column_degree_bounds);
}

pub fn poseidon_zk_pcs_config(
    log_n_rows: u32,
    fri_config: FriConfig,
) -> Result<stwo::core::pcs::PcsConfig, PoseidonZkMetadataError> {
    let metadata_component = poseidon_zk_metadata_component(log_n_rows);
    let canonical_metadata =
        poseidon_zk_canonical_metadata(&metadata_component, fri_config.log_blowup_factor)?;

    Ok(stwo::core::pcs::PcsConfig {
        fri_config,
        lifting_log_size: Some(canonical_metadata.fri_first_layer_log_size),
        ..stwo::core::pcs::PcsConfig::default()
    })
}
