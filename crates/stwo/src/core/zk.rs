use core::mem;

use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std_shims::{vec, BTreeSet, Vec};

use crate::core::channel::Channel;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::coset_vanishing;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::fields::ComplexConjugate;
use crate::core::fri::{FriConfig, FriProof, FriProofAux};
use crate::core::pcs::quotients::{CommitmentSchemeProof, CommitmentSchemeProofAux};
use crate::core::pcs::utils::{prepare_preprocessed_query_positions, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::poly::utils::get_folding_alphas;
use crate::core::proof::SizeEstimate;
use crate::core::utils::bit_reverse_index;
use crate::core::vcs::blake2_hash::Blake2sHasher;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux, MerkleVerificationError,
    MerkleVerifierLifted,
};
use crate::core::ColumnVec;

/// Version marker for the explicit ZK proof format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkProofVersion(pub u32);

impl ZkProofVersion {
    pub const V1: Self = Self(1);
}

/// Hash of the verifier-owned privacy map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPrivacyMapHash(pub [u8; 32]);

/// Hash binding application-level public statement data into the ZK transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPublicStatementHash(pub [u8; 32]);

/// Stable tree/column range for verifier-owned privacy metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkColumnRange {
    pub tree_index: usize,
    pub column_start: usize,
    pub column_end: usize,
}

impl ZkColumnRange {
    #[must_use]
    pub const fn new(tree_index: usize, column_start: usize, column_end: usize) -> Self {
        Self {
            tree_index,
            column_start,
            column_end,
        }
    }

    #[must_use]
    pub fn is_singleton(self) -> bool {
        self.column_start.checked_add(1) == Some(self.column_end)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkTraceTreeScope {
    Preprocessed,
    OriginalTrace,
    InteractionTrace { interaction_index: u32 },
    CompositionSplit,
    Custom(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkTraceTreeScopeBinding {
    pub tree_index: usize,
    pub scope: ZkTraceTreeScope,
    pub column_log_degree_bounds: ColumnVec<u32>,
}

#[must_use]
pub fn zk_singleton_column_ranges(tree_index: usize, column_count: usize) -> Vec<ZkColumnRange> {
    (0..column_count)
        .map(|column| ZkColumnRange::new(tree_index, column, column + 1))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkAirDegreeBounds {
    pub trace_log_degree: u32,
    pub randomized_private_column_log_degree: u32,
    pub public_air_constraint_log_expansion: u32,
    pub private_constraint_log_expansion: u32,
    pub composition_log_split: u32,
    pub full_composition_log_degree_bound: u32,
    pub split_composition_log_degree_bound: u32,
    pub left_masked_split_log_degree_bound: u32,
    pub right_masked_split_log_degree_bound: u32,
    pub fri_log_blowup_factor: u32,
    pub fri_first_layer_log_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkAirMetadataBuildError {
    EmptyTraceMetadata,
    TraceDomainMismatch {
        expected: u32,
        actual: u32,
    },
    MissingTraceTreeScope {
        tree_index: usize,
    },
    InvalidTraceTreeScope {
        tree_index: usize,
    },
    DuplicateTraceTreeScope {
        tree_index: usize,
    },
    DuplicatePrivateRange {
        range: ZkColumnRange,
    },
    InvalidPublicRange {
        range: ZkColumnRange,
    },
    InvalidPrivateRange {
        range: ZkColumnRange,
    },
    PublicPrivateRangeOverlap {
        public_range: ZkColumnRange,
        private_range: ZkColumnRange,
    },
    IneligiblePrivateColumn {
        range: ZkColumnRange,
        usage: ZkPrivateColumnUsage,
    },
    MissingPrivateInteractionColumn {
        range: ZkColumnRange,
    },
    InvalidPrivateLogupScope {
        range: ZkColumnRange,
    },
    InvalidPrivacyProviderScopeCount {
        expected: usize,
        actual: usize,
    },
    InvalidPrivacyDependencyRange {
        range: ZkColumnRange,
    },
    IncompleteDependencyMetadata,
    IncompleteLogupClaimMetadata,
    MissingPrivateLogupClaimManifest {
        interaction_index: u32,
    },
    InvalidLogupClaimManifest {
        interaction_index: u32,
    },
    MissingPrivateLogupClaimPolicy {
        interaction_index: u32,
        claim_index: u32,
    },
    UnsupportedPrivateLogupClaimPolicy {
        interaction_index: u32,
        claim_index: u32,
    },
    InvalidLogupClaimPolicy {
        interaction_index: u32,
        claim_index: u32,
    },
    DuplicateLogupStatisticalAggregateGroup {
        aggregate_id: u32,
    },
    MissingLogupStatisticalAggregateGroup {
        aggregate_id: u32,
    },
    UnusedLogupStatisticalAggregateGroup {
        aggregate_id: u32,
    },
    InvalidLogupStatisticalAggregateGroup {
        aggregate_id: u32,
    },
    LogupStatisticalSecurityOverflow {
        aggregate_id: u32,
    },
    InsufficientLogupStatisticalSecurity {
        aggregate_id: u32,
        computed_bits: u32,
        min_bits: u32,
    },
    UnclassifiedAirColumn {
        range: ZkColumnRange,
    },
    PublicRangeOverlap {
        lhs: ZkColumnRange,
        rhs: ZkColumnRange,
    },
    ConstraintDegreeBound {
        trace_log_degree: u32,
        max_constraint_log_degree_bound: u32,
    },
    DegreeGeometryMismatch,
    DegreeOverflow,
    FriBatchDegree,
    QuotientSplitMaskProfile,
}

#[derive(Clone, Debug)]
pub struct ZkAirCanonicalMetadata {
    pub component_column_log_sizes: TreeVec<ColumnVec<u32>>,
    pub trace_tree_scope_bindings: Vec<ZkTraceTreeScopeBinding>,
    pub trace_tree_scope_hash: [u8; 32],
    pub public_ranges: Vec<ZkColumnRange>,
    pub private_ranges: Vec<ZkColumnRange>,
    pub private_column_scope: ZkPrivateColumnScope,
    pub logup_claim_metadata_completeness: ZkLogupClaimMetadataCompleteness,
    pub logup_claim_manifest: Vec<ZkLogupClaimManifestEntry>,
    pub logup_claim_policies: Vec<ZkLogupClaimPolicy>,
    pub logup_statistical_aggregate_groups: Vec<ZkLogupStatisticalAggregateGroup>,
    pub logup_statistical_security_budgets: Vec<ZkLogupStatisticalSecurityBudget>,
    pub composition_split_range: ZkColumnRange,
    pub trace_domain_log_size: u32,
    pub randomized_witness_log_degree: u32,
    pub fri_first_layer_log_size: u32,
    pub quotient_degree_bound: ZkColumnDegreeBound,
    pub quotient_split_mask_profile: ZkQuotientSplitMaskProfile,
    pub public_statement_hash: ZkPublicStatementHash,
    pub degree_bounds: ZkAirDegreeBounds,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkAirConfigArtifacts {
    pub privacy_map: ZkPrivacyMap,
    pub metadata: ZkPublicMetadata,
    pub verifier_config: ZkVerificationConfig,
    pub verifier_audit: ZkWitnessRandomizationVerifierAudit,
    pub column_degree_bounds: Vec<ZkColumnDegreeBound>,
    pub logup_statistical_security_budgets: Vec<ZkLogupStatisticalSecurityBudget>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkAirId(pub Vec<u8>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkPrivacyReason {
    Witness,
    DerivedFromPrivateRoot,
    LogUpInteraction,
    ApplicationPrivate,
    Custom(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkPrivateRoot {
    pub range: ZkColumnRange,
    pub usage: ZkPrivateColumnUsage,
    pub reason: ZkPrivacyReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkDependencyKind {
    AirConstraint,
    LogUpInput,
    LogUpRunningSum,
    LookupMultiplicity,
    CompositionQuotient,
    FriOracle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkPrivacyDependency {
    pub from: ZkColumnRange,
    pub to: ZkColumnRange,
    pub kind: ZkDependencyKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkPrivacyInferenceMode {
    FailClosed,
    /// Conservative fallback that marks every interaction column private, but
    /// still requires complete dependency metadata for private roots.
    MarkAllInteractionDerivedPrivate,
    /// Test/review-only mode. Production ZK config construction still requires
    /// complete dependency metadata for private roots.
    UseDeclaredDependenciesOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkDependencyMetadataCompleteness {
    Incomplete,
    CompleteTraceAndInteractionClosure,
}

impl ZkDependencyMetadataCompleteness {
    #[must_use]
    pub const fn is_complete(self) -> bool {
        matches!(self, Self::CompleteTraceAndInteractionClosure)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkLogupClaimMetadataCompleteness {
    Incomplete,
    Complete,
}

impl ZkLogupClaimMetadataCompleteness {
    #[must_use]
    pub const fn is_complete(self) -> bool {
        matches!(self, Self::Complete)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkLogupClaimVisibility {
    /// The scalar is a private witness-derived LogUp fingerprint and this ZK
    /// path must reject until a reviewed private-claim protocol handles it.
    PrivateUnsupported,
    /// The scalar is intentionally public. This is a reviewed semantic
    /// assertion, not a cryptographic hiding transformation. This visibility
    /// must not be used for witness-derived LogUp fingerprints unless a
    /// paper-grounded Math/Crypto review proves that the exposed scalar is part
    /// of the public statement and leaks no private witness information.
    SemanticallyPublic,
    /// The scalar belongs to a private LogUp aggregate group. The verifier must
    /// not receive the raw scalar; it may receive only the reviewed aggregate
    /// representation, and the group must satisfy the statistical leakage
    /// budget declared by verifier-owned metadata.
    StatisticalAggregate { aggregate_id: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkLogupClaimPolicy {
    pub interaction_index: u32,
    pub claim_index: u32,
    pub visibility: ZkLogupClaimVisibility,
    pub semantic_domain: Vec<u8>,
    pub semantic_statement: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkLogupClaimManifestEntry {
    pub interaction_index: u32,
    pub claim_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkLogupStatisticalAggregateTarget {
    Zero,
    PublicExpression { domain: Vec<u8>, statement: Vec<u8> },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkLogupStatisticalAggregateGroup {
    pub aggregate_id: u32,
    pub target: ZkLogupStatisticalAggregateTarget,
    pub relation_domain: Vec<u8>,
    pub relation_statement: Vec<u8>,
    pub private_lookup_term_count_bound: u64,
    pub lookup_challenge_count: u32,
    pub expected_proof_volume: u64,
    pub min_statistical_security_bits: u32,
    pub safety_margin_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkPrivateColumnSemanticDomains {
    pub range: ZkColumnRange,
    pub semantic_trace_domain_log_sizes: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkLogupStatisticalSecurityBudget {
    pub aggregate_id: u32,
    pub extension_field_bits: u32,
    pub private_lookup_term_count_bound: u64,
    pub lookup_challenge_count: u32,
    pub expected_proof_volume: u64,
    pub safety_margin_bits: u32,
    pub computed_security_bits: u32,
    pub min_statistical_security_bits: u32,
}

pub const ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS: u32 = 124;
pub const ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH: [u8; 32] = [0x5a; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkLogupStatisticalAggregatePolicyError {
    NonEmptySecurityBudgets,
    NonEmptySecurityBudgetHash,
}

pub fn reject_zk_logup_statistical_security_budget_activation(
    metadata: &ZkPublicMetadata,
    budgets: &[ZkLogupStatisticalSecurityBudget],
) -> Result<(), ZkLogupStatisticalAggregatePolicyError> {
    if !budgets.is_empty() {
        return Err(ZkLogupStatisticalAggregatePolicyError::NonEmptySecurityBudgets);
    }
    if metadata.logup_statistical_security_budget_hash
        != ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH
    {
        return Err(ZkLogupStatisticalAggregatePolicyError::NonEmptySecurityBudgetHash);
    }

    Ok(())
}

pub trait ZkAirPrivacyProvider {
    fn air_id(&self) -> ZkAirId;

    fn component_column_log_sizes(&self) -> TreeVec<ColumnVec<u32>>;

    fn max_constraint_log_degree_bound(&self) -> u32;

    fn trace_tree_scopes(&self) -> Vec<ZkTraceTreeScope>;

    fn public_roots(&self) -> Vec<ZkColumnRange>;

    fn private_roots(&self) -> Vec<ZkPrivateRoot>;

    fn private_column_semantic_domains(&self) -> Vec<ZkPrivateColumnSemanticDomains> {
        vec![]
    }

    fn dependency_edges(&self) -> Vec<ZkPrivacyDependency>;

    fn dependency_metadata_completeness(&self) -> ZkDependencyMetadataCompleteness {
        ZkDependencyMetadataCompleteness::Incomplete
    }

    fn logup_claim_policies(&self) -> Vec<ZkLogupClaimPolicy> {
        vec![]
    }

    fn logup_claim_manifest(&self) -> Vec<ZkLogupClaimManifestEntry> {
        vec![]
    }

    fn logup_statistical_aggregate_groups(&self) -> Vec<ZkLogupStatisticalAggregateGroup> {
        vec![]
    }

    fn logup_claim_metadata_completeness(&self) -> ZkLogupClaimMetadataCompleteness {
        ZkLogupClaimMetadataCompleteness::Incomplete
    }

    fn application_domain(&self) -> &[u8];

    fn application_statement(&self) -> Vec<u8>;
}

#[must_use]
pub const fn zk_max_u32(lhs: u32, rhs: u32) -> u32 {
    if lhs > rhs {
        lhs
    } else {
        rhs
    }
}

pub fn zk_power_of_two_u64(log_size: u32) -> Result<u64, ZkAirMetadataBuildError> {
    1u64.checked_shl(log_size)
        .ok_or(ZkAirMetadataBuildError::DegreeOverflow)
}

pub fn zk_public_air_constraint_log_expansion_from_bound(
    trace_log_degree: u32,
    max_constraint_log_degree_bound: u32,
) -> Result<u32, ZkAirMetadataBuildError> {
    max_constraint_log_degree_bound
        .checked_sub(trace_log_degree)
        .ok_or(ZkAirMetadataBuildError::ConstraintDegreeBound {
            trace_log_degree,
            max_constraint_log_degree_bound,
        })
}

pub fn zk_masked_private_constraint_log_expansion(
    public_air_constraint_log_expansion: u32,
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
) -> Result<u32, ZkAirMetadataBuildError> {
    let randomized_degree_delta =
        randomized_private_column_log_degree.saturating_sub(trace_log_degree);
    public_air_constraint_log_expansion
        .checked_add(zk_max_u32(
            public_air_constraint_log_expansion,
            randomized_degree_delta,
        ))
        .ok_or(ZkAirMetadataBuildError::DegreeOverflow)
}

pub fn zk_default_randomized_witness_log_degree(
    trace_domain_log_size: u32,
) -> Result<u32, ZkAirMetadataBuildError> {
    trace_domain_log_size
        .checked_add(1)
        .ok_or(ZkAirMetadataBuildError::DegreeOverflow)
}

fn zk_log2_ceil_u64(value: u64) -> Result<u32, ZkAirMetadataBuildError> {
    if value <= 1 {
        return Ok(0);
    }
    value
        .checked_next_power_of_two()
        .ok_or(ZkAirMetadataBuildError::DegreeOverflow)
        .map(u64::ilog2)
}

pub fn zk_private_column_randomizer_dimension(
    entry: &ZkPrivateColumnScopeEntry,
) -> Result<u64, ZkAirMetadataBuildError> {
    zk_power_of_two_u64(entry.trace_domain_log_size)
}

pub fn zk_private_column_semantic_vanishing_degree_bound(
    entry: &ZkPrivateColumnScopeEntry,
) -> Result<u64, ZkAirMetadataBuildError> {
    entry
        .semantic_trace_domain_log_sizes
        .iter()
        .try_fold(0u64, |acc, &log_size| {
            acc.checked_add(zk_power_of_two_u64(log_size)?)
                .ok_or(ZkAirMetadataBuildError::DegreeOverflow)
        })
}

pub fn zk_private_column_randomized_log_degree(
    entry: &ZkPrivateColumnScopeEntry,
) -> Result<u32, ZkAirMetadataBuildError> {
    let vanishing_degree_bound = zk_private_column_semantic_vanishing_degree_bound(entry)?;
    let randomizer_dimension = zk_private_column_randomizer_dimension(entry)?;
    zk_log2_ceil_u64(
        vanishing_degree_bound
            .checked_add(randomizer_dimension)
            .ok_or(ZkAirMetadataBuildError::DegreeOverflow)?,
    )
}

pub fn zk_randomized_witness_log_degree_for_private_scope(
    trace_domain_log_size: u32,
    entries: &[ZkPrivateColumnScopeEntry],
) -> Result<u32, ZkAirMetadataBuildError> {
    if entries.is_empty() {
        return zk_default_randomized_witness_log_degree(trace_domain_log_size);
    }

    entries.iter().try_fold(0, |acc, entry| {
        Ok(acc.max(zk_private_column_randomized_log_degree(entry)?))
    })
}

pub fn derive_stwo_zk_air_degree_bounds(
    trace_log_degree: u32,
    randomized_private_column_log_degree: u32,
    public_air_constraint_log_expansion: u32,
    private_constraint_log_expansion: u32,
    fri_log_blowup_factor: u32,
    composition_log_split: u32,
) -> Result<ZkAirDegreeBounds, ZkAirMetadataBuildError> {
    let private_constraint_log_expansion = zk_max_u32(
        public_air_constraint_log_expansion,
        private_constraint_log_expansion,
    );
    let full_composition_log_degree_bound = randomized_private_column_log_degree
        .checked_add(private_constraint_log_expansion)
        .ok_or(ZkAirMetadataBuildError::DegreeOverflow)?;
    let split_composition_log_degree_bound = full_composition_log_degree_bound
        .checked_sub(composition_log_split)
        .ok_or(ZkAirMetadataBuildError::DegreeOverflow)?;
    let left_masked_split_log_degree_bound = full_composition_log_degree_bound;
    let right_masked_split_log_degree_bound = split_composition_log_degree_bound;
    let max_committed_split_log_degree_bound = zk_max_u32(
        left_masked_split_log_degree_bound,
        zk_max_u32(
            right_masked_split_log_degree_bound,
            randomized_private_column_log_degree,
        ),
    );
    let fri_first_layer_log_size = max_committed_split_log_degree_bound
        .checked_add(fri_log_blowup_factor)
        .ok_or(ZkAirMetadataBuildError::DegreeOverflow)?;

    Ok(ZkAirDegreeBounds {
        trace_log_degree,
        randomized_private_column_log_degree,
        public_air_constraint_log_expansion,
        private_constraint_log_expansion,
        composition_log_split,
        full_composition_log_degree_bound,
        split_composition_log_degree_bound,
        left_masked_split_log_degree_bound,
        right_masked_split_log_degree_bound,
        fri_log_blowup_factor,
        fri_first_layer_log_size,
    })
}

/// Verifier-owned privacy map. Proof metadata may echo its hash, but must not
/// be trusted as the source of privacy policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPrivacyMap {
    pub version: ZkProofVersion,
    pub private_columns: Vec<ZkColumnRange>,
    pub hash: ZkPrivacyMapHash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkPrivateColumnUsage {
    OrdinaryWitness,
    Lookup,
    Permutation,
    Fractional,
    LogUp,
    Memory,
    GrandProduct,
    Multiset,
}

impl ZkPrivateColumnUsage {
    #[must_use]
    pub const fn eligible_for_witness_randomization(self) -> bool {
        matches!(self, Self::OrdinaryWitness | Self::LogUp)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkPrivateColumnScopeEntry {
    pub range: ZkColumnRange,
    pub usage: ZkPrivateColumnUsage,
    pub trace_domain_log_size: u32,
    pub semantic_trace_domain_log_sizes: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPrivateColumnScope {
    pub version: ZkProofVersion,
    pub hash: [u8; 32],
    pub entries: Vec<ZkPrivateColumnScopeEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkPrivateColumnScopeValidationError {
    UnsupportedProofVersion {
        actual: u32,
    },
    VersionMismatch,
    ScopeHashMismatch,
    NonCanonicalEntries,
    MissingPrivateColumn {
        range: ZkColumnRange,
    },
    NonSingletonPrivateRange {
        range: ZkColumnRange,
    },
    EntryForNonPrivateColumn {
        range: ZkColumnRange,
    },
    DuplicatePrivateColumn {
        range: ZkColumnRange,
    },
    IneligiblePrivateColumn {
        range: ZkColumnRange,
        usage: ZkPrivateColumnUsage,
    },
}

impl ZkPrivateColumnScope {
    pub fn canonicalize(&mut self) {
        self.entries.sort_unstable();
        self.entries.dedup();
    }

    #[must_use]
    pub fn canonicalized(mut self) -> Self {
        self.canonicalize();
        self
    }
}

#[must_use]
pub fn canonical_zk_private_column_scope_from_entries(
    entries: Vec<ZkPrivateColumnScopeEntry>,
) -> ZkPrivateColumnScope {
    let mut scope = ZkPrivateColumnScope {
        version: ZkProofVersion::V1,
        hash: [0; 32],
        entries,
    }
    .canonicalized();
    scope.hash = canonical_zk_private_column_scope_hash(&scope);
    scope
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkCirclePointEncoding {
    pub x: BaseField,
    pub y: BaseField,
}

impl From<CirclePoint<BaseField>> for ZkCirclePointEncoding {
    fn from(point: CirclePoint<BaseField>) -> Self {
        Self {
            x: point.x,
            y: point.y,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkCircleCosetEncoding {
    pub log_size: u32,
    pub initial_index: u64,
    pub initial: ZkCirclePointEncoding,
    pub step_size: u64,
    pub step: ZkCirclePointEncoding,
}

impl From<Coset> for ZkCircleCosetEncoding {
    fn from(coset: Coset) -> Self {
        Self {
            log_size: coset.log_size,
            initial_index: u64::try_from(coset.initial_index.0)
                .expect("ZK coset initial index must fit in u64"),
            initial: coset.initial.into(),
            step_size: u64::try_from(coset.step_size.0)
                .expect("ZK coset step size must fit in u64"),
            step: coset.step.into(),
        }
    }
}

#[must_use]
pub fn zk_trace_domain_half_coset(trace_domain: Coset) -> Coset {
    assert!(trace_domain.log_size() > 0);
    Coset::new(trace_domain.initial_index, trace_domain.log_size() - 1)
}

#[must_use]
pub fn zk_trace_domain_circle_domain(trace_domain: Coset) -> CircleDomain {
    CircleDomain::new(zk_trace_domain_half_coset(trace_domain))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkRandomizerSpaceEntry {
    pub range: ZkColumnRange,
    pub trace_domain: ZkCircleCosetEncoding,
    pub semantic_trace_domains: Vec<ZkCircleCosetEncoding>,
    pub randomized_log_degree: u32,
    pub randomizer_dimension: u64,
}

const ZK_PRIVACY_MAP_HASH_DOMAIN: &[u8] = b"stwo.zk.privacy-map.v1";
const ZK_TRACE_TREE_SCOPE_HASH_DOMAIN: &[u8] = b"stwo.zk.trace-tree-scope.v1";
const ZK_PRIVATE_COLUMN_SCOPE_HASH_DOMAIN: &[u8] = b"stwo.zk.private-column-scope.v1";
const ZK_RANDOMIZER_SPACE_HASH_DOMAIN: &[u8] = b"stwo.zk.randomizer-space.v1";
const ZK_SPLIT_DERIVATION_HASH_DOMAIN: &[u8] = b"stwo.zk.split-derivation.v1";
const ZK_QUOTIENT_SPLIT_MASK_PROFILE_HASH_DOMAIN: &[u8] = b"stwo.zk.quotient-split-mask-profile.v1";
const ZK_RANDOMIZER_BASIS_ID: &[u8] = b"circle-fft-bit-reversed";
const ZK_RANDOMIZER_CONSTRUCTION_ID: &[u8] = b"eval-coset-vanishing-times-r-interpolate";
const ZK_RANDOMIZER_RANK_MATRIX_ID: &[u8] = b"base-field-functional-matrix-v1";
const ZK_SPLIT_IDENTITY: &[u8] = b"p(z)=left(z)+pi^(L-2)(z.x)*right(z)";
const ZK_QUOTIENT_SPLIT_MASK_CONSTRUCTION_ID: &[u8] = b"left-plus-pi-t-right-minus-t";
const ZK_QUOTIENT_OPENING_MODEL: &[u8] = b"internal-pcs-fri-only";
const ZK_H_BATCH_RULE: &[u8] = b"fri-first-layer-degree-bound";

fn push_tag(dst: &mut Vec<u8>, tag: &[u8]) {
    push_u64(dst, tag.len() as u64);
    dst.extend_from_slice(tag);
}

fn push_u32(dst: &mut Vec<u8>, value: u32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(dst: &mut Vec<u8>, value: u64) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn push_usize(dst: &mut Vec<u8>, value: usize) {
    push_u64(
        dst,
        u64::try_from(value).expect("ZK metadata usize field must fit in u64"),
    );
}

fn push_hash(dst: &mut Vec<u8>, value: &[u8; 32]) {
    dst.extend_from_slice(value);
}

fn push_base_field(dst: &mut Vec<u8>, value: BaseField) {
    push_u32(dst, value.0);
}

fn push_column_range(dst: &mut Vec<u8>, range: ZkColumnRange) {
    push_usize(dst, range.tree_index);
    push_usize(dst, range.column_start);
    push_usize(dst, range.column_end);
}

fn push_circle_point(dst: &mut Vec<u8>, point: ZkCirclePointEncoding) {
    push_base_field(dst, point.x);
    push_base_field(dst, point.y);
}

fn push_circle_coset(dst: &mut Vec<u8>, coset: ZkCircleCosetEncoding) {
    push_u32(dst, coset.log_size);
    push_u64(dst, coset.initial_index);
    push_circle_point(dst, coset.initial);
    push_u64(dst, coset.step_size);
    push_circle_point(dst, coset.step);
}

fn private_column_usage_tag(usage: ZkPrivateColumnUsage) -> u32 {
    match usage {
        ZkPrivateColumnUsage::OrdinaryWitness => 0,
        ZkPrivateColumnUsage::Lookup => 1,
        ZkPrivateColumnUsage::Permutation => 2,
        ZkPrivateColumnUsage::Fractional => 3,
        ZkPrivateColumnUsage::LogUp => 4,
        ZkPrivateColumnUsage::Memory => 5,
        ZkPrivateColumnUsage::GrandProduct => 6,
        ZkPrivateColumnUsage::Multiset => 7,
    }
}

fn trace_tree_scope_tag(scope: ZkTraceTreeScope) -> (u32, u32) {
    match scope {
        ZkTraceTreeScope::Preprocessed => (0, 0),
        ZkTraceTreeScope::OriginalTrace => (1, 0),
        ZkTraceTreeScope::InteractionTrace { interaction_index } => (2, interaction_index),
        ZkTraceTreeScope::CompositionSplit => (3, 0),
        ZkTraceTreeScope::Custom(value) => (4, value),
    }
}

fn privacy_reason_tag(reason: ZkPrivacyReason) -> (u32, u32) {
    match reason {
        ZkPrivacyReason::Witness => (0, 0),
        ZkPrivacyReason::DerivedFromPrivateRoot => (1, 0),
        ZkPrivacyReason::LogUpInteraction => (2, 0),
        ZkPrivacyReason::ApplicationPrivate => (3, 0),
        ZkPrivacyReason::Custom(value) => (4, value),
    }
}

fn dependency_kind_tag(kind: ZkDependencyKind) -> u32 {
    match kind {
        ZkDependencyKind::AirConstraint => 0,
        ZkDependencyKind::LogUpInput => 1,
        ZkDependencyKind::LogUpRunningSum => 2,
        ZkDependencyKind::LookupMultiplicity => 3,
        ZkDependencyKind::CompositionQuotient => 4,
        ZkDependencyKind::FriOracle => 5,
    }
}

fn privacy_inference_mode_tag(mode: ZkPrivacyInferenceMode) -> u32 {
    match mode {
        ZkPrivacyInferenceMode::FailClosed => 0,
        ZkPrivacyInferenceMode::MarkAllInteractionDerivedPrivate => 1,
        ZkPrivacyInferenceMode::UseDeclaredDependenciesOnly => 2,
    }
}

fn dependency_metadata_completeness_tag(completeness: ZkDependencyMetadataCompleteness) -> u32 {
    match completeness {
        ZkDependencyMetadataCompleteness::Incomplete => 0,
        ZkDependencyMetadataCompleteness::CompleteTraceAndInteractionClosure => 1,
    }
}

fn logup_claim_metadata_completeness_tag(completeness: ZkLogupClaimMetadataCompleteness) -> u32 {
    match completeness {
        ZkLogupClaimMetadataCompleteness::Incomplete => 0,
        ZkLogupClaimMetadataCompleteness::Complete => 1,
    }
}

fn logup_claim_visibility_tag(visibility: ZkLogupClaimVisibility) -> u32 {
    match visibility {
        ZkLogupClaimVisibility::PrivateUnsupported => 0,
        ZkLogupClaimVisibility::SemanticallyPublic => 1,
        ZkLogupClaimVisibility::StatisticalAggregate { .. } => 2,
    }
}

fn logup_statistical_aggregate_target_tag(target: &ZkLogupStatisticalAggregateTarget) -> u32 {
    match target {
        ZkLogupStatisticalAggregateTarget::Zero => 0,
        ZkLogupStatisticalAggregateTarget::PublicExpression { .. } => 1,
    }
}

fn blake2s_hash(bytes: &[u8]) -> [u8; 32] {
    Blake2sHasher::hash(bytes).into()
}

#[must_use]
pub fn canonical_zk_logup_statistical_security_budget_hash(
    budgets: &[ZkLogupStatisticalSecurityBudget],
) -> [u8; 32] {
    if budgets.is_empty() {
        return ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH;
    }

    let mut budgets = budgets.to_vec();
    budgets.sort_unstable();

    let mut bytes = Vec::new();
    push_tag(&mut bytes, b"stwo.zk.logup.statistical.security.budget.v1");
    push_u64(&mut bytes, budgets.len() as u64);
    for budget in budgets {
        push_u32(&mut bytes, budget.aggregate_id);
        push_u32(&mut bytes, budget.extension_field_bits);
        push_u64(&mut bytes, budget.private_lookup_term_count_bound);
        push_u32(&mut bytes, budget.lookup_challenge_count);
        push_u64(&mut bytes, budget.expected_proof_volume);
        push_u32(&mut bytes, budget.safety_margin_bits);
        push_u32(&mut bytes, budget.computed_security_bits);
        push_u32(&mut bytes, budget.min_statistical_security_bits);
    }

    blake2s_hash(&bytes)
}

#[must_use]
pub fn canonical_zk_privacy_map_hash(privacy_map: &ZkPrivacyMap) -> ZkPrivacyMapHash {
    let mut private_columns = privacy_map.private_columns.clone();
    private_columns.sort_unstable();

    let mut bytes = Vec::new();
    push_tag(&mut bytes, ZK_PRIVACY_MAP_HASH_DOMAIN);
    push_u32(&mut bytes, privacy_map.version.0);
    push_u64(&mut bytes, private_columns.len() as u64);
    for range in private_columns {
        push_column_range(&mut bytes, range);
    }

    ZkPrivacyMapHash(blake2s_hash(&bytes))
}

#[must_use]
pub fn canonical_zk_trace_tree_scope_hash(bindings: &[ZkTraceTreeScopeBinding]) -> [u8; 32] {
    let mut bindings = bindings.to_vec();
    bindings.sort_unstable_by_key(|binding| binding.tree_index);

    let mut bytes = Vec::new();
    push_tag(&mut bytes, ZK_TRACE_TREE_SCOPE_HASH_DOMAIN);
    push_u32(&mut bytes, ZkProofVersion::V1.0);
    push_u64(&mut bytes, bindings.len() as u64);
    for binding in bindings {
        let (scope_tag, scope_parameter) = trace_tree_scope_tag(binding.scope);
        push_usize(&mut bytes, binding.tree_index);
        push_u32(&mut bytes, scope_tag);
        push_u32(&mut bytes, scope_parameter);
        push_usize(&mut bytes, binding.column_log_degree_bounds.len());
        for log_degree_bound in binding.column_log_degree_bounds {
            push_u32(&mut bytes, log_degree_bound);
        }
    }

    blake2s_hash(&bytes)
}

#[must_use]
pub fn canonical_zk_private_column_scope_hash(scope: &ZkPrivateColumnScope) -> [u8; 32] {
    let scope = scope.clone().canonicalized();
    let mut bytes = Vec::new();
    push_tag(&mut bytes, ZK_PRIVATE_COLUMN_SCOPE_HASH_DOMAIN);
    push_u32(&mut bytes, scope.version.0);
    push_u64(&mut bytes, scope.entries.len() as u64);
    for entry in scope.entries {
        push_column_range(&mut bytes, entry.range);
        push_u32(&mut bytes, private_column_usage_tag(entry.usage));
        push_u32(&mut bytes, entry.trace_domain_log_size);
        push_u64(
            &mut bytes,
            entry.semantic_trace_domain_log_sizes.len() as u64,
        );
        for log_size in entry.semantic_trace_domain_log_sizes {
            push_u32(&mut bytes, log_size);
        }
    }
    blake2s_hash(&bytes)
}

#[must_use]
pub fn canonical_zk_randomizer_space_hash(
    private_column_scope_hash: [u8; 32],
    entries: &[ZkRandomizerSpaceEntry],
) -> [u8; 32] {
    let mut entries = entries.to_vec();
    entries.sort_unstable();

    let mut bytes = Vec::new();
    push_tag(&mut bytes, ZK_RANDOMIZER_SPACE_HASH_DOMAIN);
    push_u32(&mut bytes, ZkProofVersion::V1.0);
    push_hash(&mut bytes, &private_column_scope_hash);
    push_u64(&mut bytes, entries.len() as u64);
    for entry in entries {
        push_column_range(&mut bytes, entry.range);
        push_circle_coset(&mut bytes, entry.trace_domain);
        push_u64(&mut bytes, entry.semantic_trace_domains.len() as u64);
        for semantic_domain in entry.semantic_trace_domains {
            push_circle_coset(&mut bytes, semantic_domain);
        }
        push_u32(&mut bytes, entry.randomized_log_degree);
        push_u64(&mut bytes, entry.randomizer_dimension);
        push_tag(&mut bytes, ZK_RANDOMIZER_BASIS_ID);
        push_tag(&mut bytes, ZK_RANDOMIZER_CONSTRUCTION_ID);
        push_tag(&mut bytes, ZK_RANDOMIZER_RANK_MATRIX_ID);
    }
    blake2s_hash(&bytes)
}

fn push_column_ranges(dst: &mut Vec<u8>, ranges: &[ZkColumnRange]) {
    push_u64(dst, ranges.len() as u64);
    for &range in ranges {
        push_column_range(dst, range);
    }
}

fn push_tree_column_log_sizes(dst: &mut Vec<u8>, column_log_sizes: &TreeVec<ColumnVec<u32>>) {
    push_u64(dst, column_log_sizes.len() as u64);
    for tree in column_log_sizes.iter() {
        push_u64(dst, tree.len() as u64);
        for &log_size in tree {
            push_u32(dst, log_size);
        }
    }
}

fn push_logup_claim_policies(dst: &mut Vec<u8>, policies: &[ZkLogupClaimPolicy]) {
    let mut policies = policies.to_vec();
    policies.sort();

    push_u64(dst, policies.len() as u64);
    for policy in policies {
        push_u32(dst, policy.interaction_index);
        push_u32(dst, policy.claim_index);
        push_u32(dst, logup_claim_visibility_tag(policy.visibility));
        if let ZkLogupClaimVisibility::StatisticalAggregate { aggregate_id } = policy.visibility {
            push_u32(dst, aggregate_id);
        }
        push_tag(dst, &policy.semantic_domain);
        push_tag(dst, &policy.semantic_statement);
    }
}

fn push_logup_claim_manifest(dst: &mut Vec<u8>, manifest: &[ZkLogupClaimManifestEntry]) {
    let mut manifest = manifest.to_vec();
    manifest.sort();

    push_u64(dst, manifest.len() as u64);
    for entry in manifest {
        push_u32(dst, entry.interaction_index);
        push_u32(dst, entry.claim_count);
    }
}

fn push_logup_statistical_aggregate_groups(
    dst: &mut Vec<u8>,
    groups: &[ZkLogupStatisticalAggregateGroup],
) {
    let mut groups = groups.to_vec();
    groups.sort();

    push_u64(dst, groups.len() as u64);
    for group in groups {
        push_u32(dst, group.aggregate_id);
        push_u32(dst, logup_statistical_aggregate_target_tag(&group.target));
        match group.target {
            ZkLogupStatisticalAggregateTarget::Zero => {}
            ZkLogupStatisticalAggregateTarget::PublicExpression { domain, statement } => {
                push_tag(dst, &domain);
                push_tag(dst, &statement);
            }
        }
        push_tag(dst, &group.relation_domain);
        push_tag(dst, &group.relation_statement);
        push_u64(dst, group.private_lookup_term_count_bound);
        push_u32(dst, group.lookup_challenge_count);
        push_u64(dst, group.expected_proof_volume);
        push_u32(dst, group.min_statistical_security_bits);
        push_u32(dst, group.safety_margin_bits);
    }
}

fn ranges_overlap(lhs: ZkColumnRange, rhs: ZkColumnRange) -> bool {
    lhs.tree_index == rhs.tree_index
        && lhs.column_start < rhs.column_end
        && rhs.column_start < lhs.column_end
}

fn zk_column_range_singletons(
    range: ZkColumnRange,
    error: fn(ZkColumnRange) -> ZkAirMetadataBuildError,
) -> Result<Vec<ZkColumnRange>, ZkAirMetadataBuildError> {
    if range.column_start >= range.column_end {
        return Err(error(range));
    }

    Ok((range.column_start..range.column_end)
        .map(|column| ZkColumnRange::new(range.tree_index, column, column + 1))
        .collect())
}

fn add_zk_private_scope_entry(
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    entries: &mut Vec<ZkPrivateColumnScopeEntry>,
    range: ZkColumnRange,
    usage: ZkPrivateColumnUsage,
    extra_semantic_trace_domain_log_sizes: &[u32],
) -> Result<bool, ZkAirMetadataBuildError> {
    let trace_domain_log_size = component_column_log_sizes[range.tree_index][range.column_start];
    let mut semantic_trace_domain_log_sizes = extra_semantic_trace_domain_log_sizes.to_vec();
    semantic_trace_domain_log_sizes.push(trace_domain_log_size);
    semantic_trace_domain_log_sizes.sort_unstable();
    semantic_trace_domain_log_sizes.dedup();

    if let Some(existing) = entries.iter_mut().find(|entry| entry.range == range) {
        if existing.usage != usage || existing.trace_domain_log_size != trace_domain_log_size {
            return Err(ZkAirMetadataBuildError::DuplicatePrivateRange { range });
        }
        let previous = existing.semantic_trace_domain_log_sizes.clone();
        existing
            .semantic_trace_domain_log_sizes
            .extend(semantic_trace_domain_log_sizes);
        existing.semantic_trace_domain_log_sizes.sort_unstable();
        existing.semantic_trace_domain_log_sizes.dedup();
        return Ok(existing.semantic_trace_domain_log_sizes != previous);
    }

    entries.push(ZkPrivateColumnScopeEntry {
        range,
        usage,
        trace_domain_log_size,
        semantic_trace_domain_log_sizes,
    });
    Ok(true)
}

fn zk_private_usage_for_dependency_target(
    dependency: ZkPrivacyDependency,
    trace_tree_scope_bindings: &[ZkTraceTreeScopeBinding],
) -> ZkPrivateColumnUsage {
    if trace_tree_scope_bindings.iter().any(|binding| {
        binding.tree_index == dependency.to.tree_index
            && matches!(binding.scope, ZkTraceTreeScope::InteractionTrace { .. })
    }) {
        return ZkPrivateColumnUsage::LogUp;
    }

    match dependency.kind {
        ZkDependencyKind::LogUpInput | ZkDependencyKind::LogUpRunningSum => {
            ZkPrivateColumnUsage::LogUp
        }
        ZkDependencyKind::AirConstraint
        | ZkDependencyKind::LookupMultiplicity
        | ZkDependencyKind::CompositionQuotient
        | ZkDependencyKind::FriOracle => ZkPrivateColumnUsage::OrdinaryWitness,
    }
}

fn zk_range_has_private_column(
    entries: &[ZkPrivateColumnScopeEntry],
    range: ZkColumnRange,
) -> Result<bool, ZkAirMetadataBuildError> {
    let singletons = zk_column_range_singletons(range, |range| {
        ZkAirMetadataBuildError::InvalidPrivacyDependencyRange { range }
    })?;
    Ok(singletons
        .iter()
        .any(|range| entries.iter().any(|entry| entry.range == *range)))
}

fn validate_zk_dependency_ranges(
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    dependency_edges: &[ZkPrivacyDependency],
) -> Result<(), ZkAirMetadataBuildError> {
    for dependency in dependency_edges {
        if !range_within_tree_bounds(dependency.from, component_column_log_sizes) {
            return Err(ZkAirMetadataBuildError::InvalidPrivacyDependencyRange {
                range: dependency.from,
            });
        }
        if !range_within_tree_bounds(dependency.to, component_column_log_sizes) {
            return Err(ZkAirMetadataBuildError::InvalidPrivacyDependencyRange {
                range: dependency.to,
            });
        }
    }

    Ok(())
}

fn validate_zk_public_ranges(
    public_ranges: &[ZkColumnRange],
) -> Result<(), ZkAirMetadataBuildError> {
    for (index, &lhs) in public_ranges.iter().enumerate() {
        for &rhs in &public_ranges[index + 1..] {
            if ranges_overlap(lhs, rhs) {
                return Err(ZkAirMetadataBuildError::PublicRangeOverlap { lhs, rhs });
            }
        }
    }

    Ok(())
}

fn validate_zk_air_column_classification_coverage(
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    trace_tree_scope_bindings: &[ZkTraceTreeScopeBinding],
    public_ranges: &[ZkColumnRange],
    private_scope_entries: &[ZkPrivateColumnScopeEntry],
) -> Result<(), ZkAirMetadataBuildError> {
    for binding in trace_tree_scope_bindings {
        if matches!(binding.scope, ZkTraceTreeScope::CompositionSplit) {
            continue;
        }
        for column in 0..component_column_log_sizes[binding.tree_index].len() {
            let range = ZkColumnRange::new(binding.tree_index, column, column + 1);
            let public = public_ranges
                .iter()
                .any(|public_range| ranges_overlap(*public_range, range));
            let private = private_scope_entries
                .iter()
                .any(|entry| entry.range == range);
            if !public && !private {
                return Err(ZkAirMetadataBuildError::UnclassifiedAirColumn { range });
            }
        }
    }

    Ok(())
}

fn private_logup_interaction_indices(
    trace_tree_scope_bindings: &[ZkTraceTreeScopeBinding],
    private_scope_entries: &[ZkPrivateColumnScopeEntry],
) -> Result<BTreeSet<u32>, ZkAirMetadataBuildError> {
    let mut indices = BTreeSet::new();
    for entry in private_scope_entries {
        let Some(binding) = trace_tree_scope_bindings
            .iter()
            .find(|binding| binding.tree_index == entry.range.tree_index)
        else {
            continue;
        };
        if entry.usage == ZkPrivateColumnUsage::LogUp {
            if let ZkTraceTreeScope::InteractionTrace { interaction_index } = binding.scope {
                indices.insert(interaction_index);
            } else {
                return Err(ZkAirMetadataBuildError::InvalidPrivateLogupScope {
                    range: entry.range,
                });
            }
        }
        if matches!(binding.scope, ZkTraceTreeScope::InteractionTrace { .. }) {
            if let ZkTraceTreeScope::InteractionTrace { interaction_index } = binding.scope {
                indices.insert(interaction_index);
            }
        }
    }
    Ok(indices)
}

fn ceil_log2_u128(value: u128) -> u32 {
    if value <= 1 {
        0
    } else {
        u128::BITS - (value - 1).leading_zeros()
    }
}

pub fn derive_zk_logup_statistical_security_budget(
    group: &ZkLogupStatisticalAggregateGroup,
) -> Result<ZkLogupStatisticalSecurityBudget, ZkAirMetadataBuildError> {
    if group.private_lookup_term_count_bound == 0
        || group.lookup_challenge_count == 0
        || group.expected_proof_volume == 0
        || group.min_statistical_security_bits == 0
        || group.relation_domain.is_empty()
        || group.relation_statement.is_empty()
    {
        return Err(
            ZkAirMetadataBuildError::InvalidLogupStatisticalAggregateGroup {
                aggregate_id: group.aggregate_id,
            },
        );
    }
    if let ZkLogupStatisticalAggregateTarget::PublicExpression { domain, statement } = &group.target
    {
        if domain.is_empty() || statement.is_empty() {
            return Err(
                ZkAirMetadataBuildError::InvalidLogupStatisticalAggregateGroup {
                    aggregate_id: group.aggregate_id,
                },
            );
        }
    }

    let exposure = u128::from(group.private_lookup_term_count_bound)
        .checked_mul(u128::from(group.lookup_challenge_count))
        .and_then(|value| value.checked_mul(u128::from(group.expected_proof_volume)))
        .ok_or(ZkAirMetadataBuildError::LogupStatisticalSecurityOverflow {
            aggregate_id: group.aggregate_id,
        })?;
    let exposure_bits = ceil_log2_u128(exposure);
    let computed_security_bits = ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS
        .saturating_sub(exposure_bits)
        .saturating_sub(group.safety_margin_bits);
    if computed_security_bits < group.min_statistical_security_bits {
        return Err(
            ZkAirMetadataBuildError::InsufficientLogupStatisticalSecurity {
                aggregate_id: group.aggregate_id,
                computed_bits: computed_security_bits,
                min_bits: group.min_statistical_security_bits,
            },
        );
    }

    Ok(ZkLogupStatisticalSecurityBudget {
        aggregate_id: group.aggregate_id,
        extension_field_bits: ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS,
        private_lookup_term_count_bound: group.private_lookup_term_count_bound,
        lookup_challenge_count: group.lookup_challenge_count,
        expected_proof_volume: group.expected_proof_volume,
        safety_margin_bits: group.safety_margin_bits,
        computed_security_bits,
        min_statistical_security_bits: group.min_statistical_security_bits,
    })
}

fn validate_zk_private_logup_claim_policies(
    trace_tree_scope_bindings: &[ZkTraceTreeScopeBinding],
    private_scope_entries: &[ZkPrivateColumnScopeEntry],
    logup_claim_metadata_completeness: ZkLogupClaimMetadataCompleteness,
    logup_claim_manifest: &[ZkLogupClaimManifestEntry],
    logup_claim_policies: &[ZkLogupClaimPolicy],
    logup_statistical_aggregate_groups: &[ZkLogupStatisticalAggregateGroup],
) -> Result<Vec<ZkLogupStatisticalSecurityBudget>, ZkAirMetadataBuildError> {
    let required_interaction_indices =
        private_logup_interaction_indices(trace_tree_scope_bindings, private_scope_entries)?;

    let mut manifest_seen = BTreeSet::new();
    for entry in logup_claim_manifest {
        if !manifest_seen.insert(entry.interaction_index)
            || entry.claim_count == 0
            || !required_interaction_indices.contains(&entry.interaction_index)
        {
            return Err(ZkAirMetadataBuildError::InvalidLogupClaimManifest {
                interaction_index: entry.interaction_index,
            });
        }
    }

    let mut policy_seen = BTreeSet::new();
    for policy in logup_claim_policies {
        if !policy_seen.insert((policy.interaction_index, policy.claim_index)) {
            return Err(ZkAirMetadataBuildError::InvalidLogupClaimPolicy {
                interaction_index: policy.interaction_index,
                claim_index: policy.claim_index,
            });
        }
        let Some(manifest_entry) = logup_claim_manifest
            .iter()
            .find(|entry| entry.interaction_index == policy.interaction_index)
        else {
            return Err(ZkAirMetadataBuildError::InvalidLogupClaimPolicy {
                interaction_index: policy.interaction_index,
                claim_index: policy.claim_index,
            });
        };
        if policy.claim_index >= manifest_entry.claim_count {
            return Err(ZkAirMetadataBuildError::InvalidLogupClaimPolicy {
                interaction_index: policy.interaction_index,
                claim_index: policy.claim_index,
            });
        }
        if policy.visibility == ZkLogupClaimVisibility::SemanticallyPublic
            && (policy.semantic_domain.is_empty() || policy.semantic_statement.is_empty())
        {
            return Err(ZkAirMetadataBuildError::InvalidLogupClaimPolicy {
                interaction_index: policy.interaction_index,
                claim_index: policy.claim_index,
            });
        }
    }

    if required_interaction_indices.is_empty() {
        if let Some(group) = logup_statistical_aggregate_groups.first() {
            return Err(
                ZkAirMetadataBuildError::UnusedLogupStatisticalAggregateGroup {
                    aggregate_id: group.aggregate_id,
                },
            );
        }
        return Ok(vec![]);
    }

    if !logup_claim_metadata_completeness.is_complete() {
        return Err(ZkAirMetadataBuildError::IncompleteLogupClaimMetadata);
    }

    let mut group_seen = BTreeSet::new();
    for group in logup_statistical_aggregate_groups {
        if !group_seen.insert(group.aggregate_id) {
            return Err(
                ZkAirMetadataBuildError::DuplicateLogupStatisticalAggregateGroup {
                    aggregate_id: group.aggregate_id,
                },
            );
        }
    }

    let mut used_statistical_groups = BTreeSet::new();
    for policy in logup_claim_policies {
        match policy.visibility {
            ZkLogupClaimVisibility::PrivateUnsupported => {
                return Err(
                    ZkAirMetadataBuildError::UnsupportedPrivateLogupClaimPolicy {
                        interaction_index: policy.interaction_index,
                        claim_index: policy.claim_index,
                    },
                );
            }
            ZkLogupClaimVisibility::SemanticallyPublic => {}
            ZkLogupClaimVisibility::StatisticalAggregate { aggregate_id } => {
                let Some(group) = logup_statistical_aggregate_groups
                    .iter()
                    .find(|group| group.aggregate_id == aggregate_id)
                else {
                    return Err(
                        ZkAirMetadataBuildError::MissingLogupStatisticalAggregateGroup {
                            aggregate_id,
                        },
                    );
                };
                used_statistical_groups.insert(group.aggregate_id);
            }
        }
    }

    for group in logup_statistical_aggregate_groups {
        if !used_statistical_groups.contains(&group.aggregate_id) {
            return Err(
                ZkAirMetadataBuildError::UnusedLogupStatisticalAggregateGroup {
                    aggregate_id: group.aggregate_id,
                },
            );
        }
    }

    for interaction_index in required_interaction_indices {
        let Some(manifest_entry) = logup_claim_manifest
            .iter()
            .find(|entry| entry.interaction_index == interaction_index)
        else {
            return Err(ZkAirMetadataBuildError::MissingPrivateLogupClaimManifest {
                interaction_index,
            });
        };
        for claim_index in 0..manifest_entry.claim_count {
            if !logup_claim_policies.iter().any(|policy| {
                policy.interaction_index == interaction_index && policy.claim_index == claim_index
            }) {
                return Err(ZkAirMetadataBuildError::MissingPrivateLogupClaimPolicy {
                    interaction_index,
                    claim_index,
                });
            }
        }
    }

    let mut budgets = Vec::with_capacity(logup_statistical_aggregate_groups.len());
    for group in logup_statistical_aggregate_groups {
        let member_count = logup_claim_policies
            .iter()
            .filter(|policy| {
                matches!(
                    policy.visibility,
                    ZkLogupClaimVisibility::StatisticalAggregate { aggregate_id }
                        if aggregate_id == group.aggregate_id
                )
            })
            .count();
        if group.private_lookup_term_count_bound < member_count as u64 {
            return Err(
                ZkAirMetadataBuildError::InvalidLogupStatisticalAggregateGroup {
                    aggregate_id: group.aggregate_id,
                },
            );
        }
        budgets.push(derive_zk_logup_statistical_security_budget(group)?);
    }

    Ok(budgets)
}

fn build_zk_trace_tree_scope_bindings_from_scopes(
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    scopes: Vec<ZkTraceTreeScope>,
) -> Result<Vec<ZkTraceTreeScopeBinding>, ZkAirMetadataBuildError> {
    if scopes.len() != component_column_log_sizes.len() {
        return Err(ZkAirMetadataBuildError::InvalidPrivacyProviderScopeCount {
            expected: component_column_log_sizes.len(),
            actual: scopes.len(),
        });
    }

    Ok(scopes
        .into_iter()
        .enumerate()
        .map(|(tree_index, scope)| ZkTraceTreeScopeBinding {
            tree_index,
            scope,
            column_log_degree_bounds: component_column_log_sizes[tree_index].clone(),
        })
        .collect())
}

fn derived_private_scope_entries_from_provider_policy(
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    trace_tree_scope_bindings: &[ZkTraceTreeScopeBinding],
    private_roots: &[ZkPrivateRoot],
    private_column_semantic_domains: &[ZkPrivateColumnSemanticDomains],
    dependency_edges: &[ZkPrivacyDependency],
    dependency_metadata_completeness: ZkDependencyMetadataCompleteness,
    inference_mode: ZkPrivacyInferenceMode,
) -> Result<Vec<ZkPrivateColumnScopeEntry>, ZkAirMetadataBuildError> {
    if !private_roots.is_empty() && !dependency_metadata_completeness.is_complete() {
        return Err(ZkAirMetadataBuildError::IncompleteDependencyMetadata);
    }
    validate_zk_dependency_ranges(component_column_log_sizes, dependency_edges)?;
    for domain_set in private_column_semantic_domains {
        if !domain_set.range.is_singleton()
            || !range_within_tree_bounds(domain_set.range, component_column_log_sizes)
        {
            return Err(ZkAirMetadataBuildError::InvalidPrivateRange {
                range: domain_set.range,
            });
        }
    }

    let mut entries = Vec::new();
    for root in private_roots {
        for range in zk_column_range_singletons(root.range, |range| {
            ZkAirMetadataBuildError::InvalidPrivateRange { range }
        })? {
            let extra_semantic_trace_domain_log_sizes = private_column_semantic_domains
                .iter()
                .filter(|domain_set| domain_set.range == range)
                .flat_map(|domain_set| domain_set.semantic_trace_domain_log_sizes.iter().copied())
                .collect::<Vec<_>>();
            add_zk_private_scope_entry(
                component_column_log_sizes,
                &mut entries,
                range,
                root.usage,
                &extra_semantic_trace_domain_log_sizes,
            )?;
        }
    }

    if matches!(
        inference_mode,
        ZkPrivacyInferenceMode::MarkAllInteractionDerivedPrivate
    ) && !entries.is_empty()
    {
        for binding in trace_tree_scope_bindings {
            if matches!(binding.scope, ZkTraceTreeScope::InteractionTrace { .. }) {
                for column in 0..binding.column_log_degree_bounds.len() {
                    let range = ZkColumnRange::new(binding.tree_index, column, column + 1);
                    let extra_semantic_trace_domain_log_sizes = private_column_semantic_domains
                        .iter()
                        .filter(|domain_set| domain_set.range == range)
                        .flat_map(|domain_set| {
                            domain_set.semantic_trace_domain_log_sizes.iter().copied()
                        })
                        .collect::<Vec<_>>();
                    add_zk_private_scope_entry(
                        component_column_log_sizes,
                        &mut entries,
                        range,
                        ZkPrivateColumnUsage::LogUp,
                        &extra_semantic_trace_domain_log_sizes,
                    )?;
                }
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for dependency in dependency_edges {
            if !zk_range_has_private_column(&entries, dependency.from)? {
                continue;
            }
            let usage =
                zk_private_usage_for_dependency_target(*dependency, trace_tree_scope_bindings);
            for range in zk_column_range_singletons(dependency.to, |range| {
                ZkAirMetadataBuildError::InvalidPrivacyDependencyRange { range }
            })? {
                let extra_semantic_trace_domain_log_sizes = private_column_semantic_domains
                    .iter()
                    .filter(|domain_set| domain_set.range == range)
                    .flat_map(|domain_set| {
                        domain_set.semantic_trace_domain_log_sizes.iter().copied()
                    })
                    .collect::<Vec<_>>();
                changed |= add_zk_private_scope_entry(
                    component_column_log_sizes,
                    &mut entries,
                    range,
                    usage,
                    &extra_semantic_trace_domain_log_sizes,
                )?;
            }
        }
    }

    for domain_set in private_column_semantic_domains {
        if !entries.iter().any(|entry| entry.range == domain_set.range) {
            return Err(ZkAirMetadataBuildError::InvalidPrivateRange {
                range: domain_set.range,
            });
        }
    }

    Ok(entries)
}

fn canonical_zk_air_provider_statement_bytes(
    air_id: ZkAirId,
    application_statement: Vec<u8>,
    inference_mode: ZkPrivacyInferenceMode,
    dependency_metadata_completeness: ZkDependencyMetadataCompleteness,
    private_roots: &[ZkPrivateRoot],
    dependency_edges: &[ZkPrivacyDependency],
    logup_claim_metadata_completeness: ZkLogupClaimMetadataCompleteness,
    logup_claim_manifest: &[ZkLogupClaimManifestEntry],
    logup_claim_policies: &[ZkLogupClaimPolicy],
    logup_statistical_aggregate_groups: &[ZkLogupStatisticalAggregateGroup],
) -> Vec<u8> {
    let mut private_roots = private_roots.to_vec();
    let mut dependency_edges = dependency_edges.to_vec();
    private_roots.sort();
    dependency_edges.sort();

    let mut bytes = Vec::new();
    push_tag(&mut bytes, b"stwo-zk-air-privacy-provider-statement-v3");
    push_tag(&mut bytes, &air_id.0);
    push_tag(&mut bytes, &application_statement);
    push_u32(&mut bytes, privacy_inference_mode_tag(inference_mode));
    push_u32(
        &mut bytes,
        dependency_metadata_completeness_tag(dependency_metadata_completeness),
    );
    push_u64(&mut bytes, private_roots.len() as u64);
    for root in private_roots {
        push_column_range(&mut bytes, root.range);
        push_u32(&mut bytes, private_column_usage_tag(root.usage));
        let (reason_tag, reason_value) = privacy_reason_tag(root.reason);
        push_u32(&mut bytes, reason_tag);
        push_u32(&mut bytes, reason_value);
    }
    push_u64(&mut bytes, dependency_edges.len() as u64);
    for dependency in dependency_edges {
        push_column_range(&mut bytes, dependency.from);
        push_column_range(&mut bytes, dependency.to);
        push_u32(&mut bytes, dependency_kind_tag(dependency.kind));
    }
    push_u32(
        &mut bytes,
        logup_claim_metadata_completeness_tag(logup_claim_metadata_completeness),
    );
    push_logup_claim_manifest(&mut bytes, logup_claim_manifest);
    push_logup_claim_policies(&mut bytes, logup_claim_policies);
    push_logup_statistical_aggregate_groups(&mut bytes, logup_statistical_aggregate_groups);
    bytes
}

fn range_within_tree_bounds(
    range: ZkColumnRange,
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
) -> bool {
    range.column_start < range.column_end
        && component_column_log_sizes
            .get(range.tree_index)
            .is_some_and(|tree| range.column_end <= tree.len())
}

fn validate_zk_air_tree_scope_bindings(
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    trace_tree_scope_bindings: &[ZkTraceTreeScopeBinding],
) -> Result<(), ZkAirMetadataBuildError> {
    let mut seen_tree_indices = BTreeSet::new();
    for binding in trace_tree_scope_bindings {
        if !seen_tree_indices.insert(binding.tree_index) {
            return Err(ZkAirMetadataBuildError::DuplicateTraceTreeScope {
                tree_index: binding.tree_index,
            });
        }
        let Some(expected_column_bounds) = component_column_log_sizes.get(binding.tree_index)
        else {
            return Err(ZkAirMetadataBuildError::InvalidTraceTreeScope {
                tree_index: binding.tree_index,
            });
        };
        if expected_column_bounds != &binding.column_log_degree_bounds {
            return Err(ZkAirMetadataBuildError::InvalidTraceTreeScope {
                tree_index: binding.tree_index,
            });
        }
    }
    for tree_index in 0..component_column_log_sizes.len() {
        if !seen_tree_indices.contains(&tree_index) {
            return Err(ZkAirMetadataBuildError::MissingTraceTreeScope { tree_index });
        }
    }

    Ok(())
}

fn validate_zk_air_public_and_private_ranges(
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    trace_tree_scope_bindings: &[ZkTraceTreeScopeBinding],
    public_ranges: &[ZkColumnRange],
    private_scope_entries: &[ZkPrivateColumnScopeEntry],
) -> Result<(), ZkAirMetadataBuildError> {
    for &public_range in public_ranges {
        if !range_within_tree_bounds(public_range, component_column_log_sizes) {
            return Err(ZkAirMetadataBuildError::InvalidPublicRange {
                range: public_range,
            });
        }
    }

    let mut seen_private_ranges = BTreeSet::new();
    for entry in private_scope_entries {
        if !entry.range.is_singleton()
            || !range_within_tree_bounds(entry.range, component_column_log_sizes)
        {
            return Err(ZkAirMetadataBuildError::InvalidPrivateRange { range: entry.range });
        }
        if !entry.usage.eligible_for_witness_randomization() {
            return Err(ZkAirMetadataBuildError::IneligiblePrivateColumn {
                range: entry.range,
                usage: entry.usage,
            });
        }
        if !seen_private_ranges.insert(entry.range) {
            return Err(ZkAirMetadataBuildError::DuplicatePrivateRange { range: entry.range });
        }
        for &public_range in public_ranges {
            if ranges_overlap(public_range, entry.range) {
                return Err(ZkAirMetadataBuildError::PublicPrivateRangeOverlap {
                    public_range,
                    private_range: entry.range,
                });
            }
        }
    }
    for binding in trace_tree_scope_bindings {
        if matches!(binding.scope, ZkTraceTreeScope::InteractionTrace { .. }) {
            for column in 0..binding.column_log_degree_bounds.len() {
                let range = ZkColumnRange::new(binding.tree_index, column, column + 1);
                if !private_scope_entries
                    .iter()
                    .any(|entry| entry.range == range && entry.usage == ZkPrivateColumnUsage::LogUp)
                {
                    return Err(ZkAirMetadataBuildError::MissingPrivateInteractionColumn { range });
                }
            }
        }
    }

    Ok(())
}

#[must_use]
pub fn canonical_zk_air_public_statement_hash(
    application_domain: &[u8],
    application_statement: &[u8],
    component_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    trace_tree_scope_hash: [u8; 32],
    public_ranges: &[ZkColumnRange],
    private_column_scope: &ZkPrivateColumnScope,
    logup_claim_metadata_completeness: ZkLogupClaimMetadataCompleteness,
    logup_claim_manifest: &[ZkLogupClaimManifestEntry],
    logup_claim_policies: &[ZkLogupClaimPolicy],
    logup_statistical_aggregate_groups: &[ZkLogupStatisticalAggregateGroup],
    composition_split_range: ZkColumnRange,
    trace_domain_log_size: u32,
    randomized_witness_log_degree: u32,
    fri_first_layer_log_size: u32,
    quotient_degree_bound: ZkColumnDegreeBound,
) -> ZkPublicStatementHash {
    let private_column_scope = private_column_scope.clone().canonicalized();
    let mut bytes = Vec::new();
    push_tag(&mut bytes, application_domain);
    push_tag(&mut bytes, application_statement);
    push_u32(&mut bytes, ZkProofVersion::V1.0);
    push_tree_column_log_sizes(&mut bytes, component_column_log_sizes);
    push_hash(&mut bytes, &trace_tree_scope_hash);
    push_u32(&mut bytes, trace_domain_log_size);
    push_u32(&mut bytes, randomized_witness_log_degree);
    push_u32(&mut bytes, fri_first_layer_log_size);
    push_column_ranges(&mut bytes, public_ranges);
    push_u64(&mut bytes, private_column_scope.entries.len() as u64);
    for entry in private_column_scope.entries {
        push_column_range(&mut bytes, entry.range);
        push_u32(&mut bytes, private_column_usage_tag(entry.usage));
        push_u32(&mut bytes, entry.trace_domain_log_size);
        push_u64(
            &mut bytes,
            entry.semantic_trace_domain_log_sizes.len() as u64,
        );
        for log_size in entry.semantic_trace_domain_log_sizes {
            push_u32(&mut bytes, log_size);
        }
    }
    push_u32(
        &mut bytes,
        logup_claim_metadata_completeness_tag(logup_claim_metadata_completeness),
    );
    push_logup_claim_manifest(&mut bytes, logup_claim_manifest);
    push_logup_claim_policies(&mut bytes, logup_claim_policies);
    push_logup_statistical_aggregate_groups(&mut bytes, logup_statistical_aggregate_groups);
    push_column_range(&mut bytes, composition_split_range);
    push_column_range(&mut bytes, quotient_degree_bound.range);
    push_u32(&mut bytes, quotient_degree_bound.log_degree_bound);

    ZkPublicStatementHash(blake2s_hash(&bytes))
}

/// Builds canonical public ZK AIR metadata from actual component tree geometry
/// and an explicit private-column policy.
///
/// Public leakage contract: tree roles, column counts, private/public column
/// positions, private degree bounds, quotient bounds, and stable metadata
/// hashes are public circuit metadata. Do not use this builder when the
/// private-column policy or AIR geometry is intended to be confidential.
#[allow(clippy::too_many_arguments)]
pub fn build_stwo_zk_air_metadata(
    component_column_log_sizes: TreeVec<ColumnVec<u32>>,
    trace_domain_log_size: u32,
    randomized_witness_log_degree: u32,
    degree_bounds: ZkAirDegreeBounds,
    trace_tree_scope_bindings: Vec<ZkTraceTreeScopeBinding>,
    public_ranges: Vec<ZkColumnRange>,
    private_scope_entries: Vec<ZkPrivateColumnScopeEntry>,
    logup_claim_metadata_completeness: ZkLogupClaimMetadataCompleteness,
    logup_claim_manifest: Vec<ZkLogupClaimManifestEntry>,
    logup_claim_policies: Vec<ZkLogupClaimPolicy>,
    application_domain: &[u8],
    application_statement: &[u8],
) -> Result<ZkAirCanonicalMetadata, ZkAirMetadataBuildError> {
    build_stwo_zk_air_metadata_with_logup_aggregates(
        component_column_log_sizes,
        trace_domain_log_size,
        randomized_witness_log_degree,
        degree_bounds,
        trace_tree_scope_bindings,
        public_ranges,
        private_scope_entries,
        logup_claim_metadata_completeness,
        logup_claim_manifest,
        logup_claim_policies,
        vec![],
        application_domain,
        application_statement,
    )
}

/// Builds canonical public ZK AIR metadata with statistical-ZK LogUp aggregate
/// declarations. This is the entry point for private LogUp metadata; callers
/// that do not declare aggregate groups continue to fail closed for private
/// witness-derived LogUp claims unless they are explicitly semantically public.
#[allow(clippy::too_many_arguments)]
pub fn build_stwo_zk_air_metadata_with_logup_aggregates(
    component_column_log_sizes: TreeVec<ColumnVec<u32>>,
    trace_domain_log_size: u32,
    randomized_witness_log_degree: u32,
    degree_bounds: ZkAirDegreeBounds,
    mut trace_tree_scope_bindings: Vec<ZkTraceTreeScopeBinding>,
    mut public_ranges: Vec<ZkColumnRange>,
    private_scope_entries: Vec<ZkPrivateColumnScopeEntry>,
    logup_claim_metadata_completeness: ZkLogupClaimMetadataCompleteness,
    logup_claim_manifest: Vec<ZkLogupClaimManifestEntry>,
    logup_claim_policies: Vec<ZkLogupClaimPolicy>,
    mut logup_statistical_aggregate_groups: Vec<ZkLogupStatisticalAggregateGroup>,
    application_domain: &[u8],
    application_statement: &[u8],
) -> Result<ZkAirCanonicalMetadata, ZkAirMetadataBuildError> {
    let actual_trace_domain_log_size =
        zk_trace_domain_log_size_from_column_bounds(&component_column_log_sizes)
            .ok_or(ZkAirMetadataBuildError::EmptyTraceMetadata)?;
    if actual_trace_domain_log_size != trace_domain_log_size {
        return Err(ZkAirMetadataBuildError::TraceDomainMismatch {
            expected: trace_domain_log_size,
            actual: actual_trace_domain_log_size,
        });
    }
    let expected_degree_bounds = derive_stwo_zk_air_degree_bounds(
        degree_bounds.trace_log_degree,
        degree_bounds.randomized_private_column_log_degree,
        degree_bounds.public_air_constraint_log_expansion,
        degree_bounds.private_constraint_log_expansion,
        degree_bounds.fri_log_blowup_factor,
        degree_bounds.composition_log_split,
    )?;
    if degree_bounds != expected_degree_bounds {
        return Err(ZkAirMetadataBuildError::DegreeGeometryMismatch);
    }
    if degree_bounds.trace_log_degree != trace_domain_log_size
        || degree_bounds.randomized_private_column_log_degree != randomized_witness_log_degree
        || degree_bounds.fri_first_layer_log_size
            < randomized_witness_log_degree
                .checked_add(degree_bounds.fri_log_blowup_factor)
                .ok_or(ZkAirMetadataBuildError::DegreeOverflow)?
    {
        return Err(ZkAirMetadataBuildError::DegreeGeometryMismatch);
    }
    public_ranges.sort_unstable();
    logup_statistical_aggregate_groups.sort();
    validate_zk_air_tree_scope_bindings(&component_column_log_sizes, &trace_tree_scope_bindings)?;
    validate_zk_public_ranges(&public_ranges)?;
    validate_zk_air_public_and_private_ranges(
        &component_column_log_sizes,
        &trace_tree_scope_bindings,
        &public_ranges,
        &private_scope_entries,
    )?;
    validate_zk_air_column_classification_coverage(
        &component_column_log_sizes,
        &trace_tree_scope_bindings,
        &public_ranges,
        &private_scope_entries,
    )?;
    let logup_statistical_security_budgets = validate_zk_private_logup_claim_policies(
        &trace_tree_scope_bindings,
        &private_scope_entries,
        logup_claim_metadata_completeness,
        &logup_claim_manifest,
        &logup_claim_policies,
        &logup_statistical_aggregate_groups,
    )?;

    let composition_tree_index = component_column_log_sizes.len();
    let composition_split_range =
        ZkColumnRange::new(composition_tree_index, 0, 2 * SECURE_EXTENSION_DEGREE);
    let quotient_degree_bound = ZkColumnDegreeBound {
        range: composition_split_range,
        log_degree_bound: degree_bounds.split_composition_log_degree_bound,
    };
    let mut composition_split_column_bounds =
        vec![degree_bounds.left_masked_split_log_degree_bound; SECURE_EXTENSION_DEGREE];
    composition_split_column_bounds.extend(vec![
        degree_bounds.right_masked_split_log_degree_bound;
        SECURE_EXTENSION_DEGREE
    ]);
    trace_tree_scope_bindings.push(ZkTraceTreeScopeBinding {
        tree_index: composition_tree_index,
        scope: ZkTraceTreeScope::CompositionSplit,
        column_log_degree_bounds: composition_split_column_bounds,
    });
    let trace_tree_scope_hash = canonical_zk_trace_tree_scope_hash(&trace_tree_scope_bindings);
    let quotient_split_mask_profile = stwo_composition_quotient_split_mask_profile(
        quotient_degree_bound.range.tree_index,
        degree_bounds.full_composition_log_degree_bound,
        quotient_degree_bound.log_degree_bound,
        zk_power_of_two_u64(quotient_degree_bound.log_degree_bound)?,
        degree_bounds.left_masked_split_log_degree_bound,
        degree_bounds.right_masked_split_log_degree_bound,
    )
    .map_err(|_| ZkAirMetadataBuildError::QuotientSplitMaskProfile)?;

    let private_column_scope =
        canonical_zk_private_column_scope_from_entries(private_scope_entries);
    let private_ranges = private_column_scope
        .entries
        .iter()
        .map(|entry| entry.range)
        .collect::<Vec<_>>();
    let public_statement_hash = canonical_zk_air_public_statement_hash(
        application_domain,
        application_statement,
        &component_column_log_sizes,
        trace_tree_scope_hash,
        &public_ranges,
        &private_column_scope,
        logup_claim_metadata_completeness,
        &logup_claim_manifest,
        &logup_claim_policies,
        &logup_statistical_aggregate_groups,
        composition_split_range,
        trace_domain_log_size,
        randomized_witness_log_degree,
        degree_bounds.fri_first_layer_log_size,
        quotient_degree_bound,
    );

    Ok(ZkAirCanonicalMetadata {
        component_column_log_sizes,
        trace_tree_scope_bindings,
        trace_tree_scope_hash,
        public_ranges,
        private_ranges,
        private_column_scope,
        logup_claim_metadata_completeness,
        logup_claim_manifest,
        logup_claim_policies,
        logup_statistical_aggregate_groups,
        logup_statistical_security_budgets,
        composition_split_range,
        trace_domain_log_size,
        randomized_witness_log_degree,
        fri_first_layer_log_size: degree_bounds.fri_first_layer_log_size,
        quotient_degree_bound,
        quotient_split_mask_profile,
        public_statement_hash,
        degree_bounds,
    })
}

pub fn build_zk_air_metadata_from_privacy_provider<P: ZkAirPrivacyProvider>(
    provider: &P,
    fri_log_blowup_factor: u32,
    inference_mode: ZkPrivacyInferenceMode,
) -> Result<ZkAirCanonicalMetadata, ZkAirMetadataBuildError> {
    let component_column_log_sizes = provider.component_column_log_sizes();
    let trace_domain_log_size =
        zk_trace_domain_log_size_from_column_bounds(&component_column_log_sizes)
            .ok_or(ZkAirMetadataBuildError::EmptyTraceMetadata)?;
    let trace_tree_scope_bindings = build_zk_trace_tree_scope_bindings_from_scopes(
        &component_column_log_sizes,
        provider.trace_tree_scopes(),
    )?;
    let private_roots = provider.private_roots();
    let private_column_semantic_domains = provider.private_column_semantic_domains();
    let dependency_edges = provider.dependency_edges();
    let dependency_metadata_completeness = provider.dependency_metadata_completeness();
    let logup_claim_manifest = provider.logup_claim_manifest();
    let logup_claim_policies = provider.logup_claim_policies();
    let logup_statistical_aggregate_groups = provider.logup_statistical_aggregate_groups();
    let logup_claim_metadata_completeness = provider.logup_claim_metadata_completeness();
    let private_scope_entries = derived_private_scope_entries_from_provider_policy(
        &component_column_log_sizes,
        &trace_tree_scope_bindings,
        &private_roots,
        &private_column_semantic_domains,
        &dependency_edges,
        dependency_metadata_completeness,
        inference_mode,
    )?;
    let randomized_witness_log_degree = zk_randomized_witness_log_degree_for_private_scope(
        trace_domain_log_size,
        &private_scope_entries,
    )?;
    let public_air_constraint_log_expansion = zk_public_air_constraint_log_expansion_from_bound(
        trace_domain_log_size,
        provider.max_constraint_log_degree_bound(),
    )?;
    let private_constraint_log_expansion = zk_masked_private_constraint_log_expansion(
        public_air_constraint_log_expansion,
        trace_domain_log_size,
        randomized_witness_log_degree,
    )?;
    let degree_bounds = derive_stwo_zk_air_degree_bounds(
        trace_domain_log_size,
        randomized_witness_log_degree,
        public_air_constraint_log_expansion,
        private_constraint_log_expansion,
        fri_log_blowup_factor,
        crate::core::verifier::COMPOSITION_LOG_SPLIT,
    )?;
    let provider_statement = canonical_zk_air_provider_statement_bytes(
        provider.air_id(),
        provider.application_statement(),
        inference_mode,
        dependency_metadata_completeness,
        &private_roots,
        &dependency_edges,
        logup_claim_metadata_completeness,
        &logup_claim_manifest,
        &logup_claim_policies,
        &logup_statistical_aggregate_groups,
    );

    build_stwo_zk_air_metadata_with_logup_aggregates(
        component_column_log_sizes,
        trace_domain_log_size,
        randomized_witness_log_degree,
        degree_bounds,
        trace_tree_scope_bindings,
        provider.public_roots(),
        private_scope_entries,
        logup_claim_metadata_completeness,
        logup_claim_manifest,
        logup_claim_policies,
        logup_statistical_aggregate_groups,
        provider.application_domain(),
        &provider_statement,
    )
}

pub fn build_zk_config_from_air_privacy_provider<P: ZkAirPrivacyProvider>(
    provider: &P,
    fri_log_blowup_factor: u32,
    inference_mode: ZkPrivacyInferenceMode,
) -> Result<ZkAirConfigArtifacts, ZkAirMetadataBuildError> {
    let canonical_metadata = build_zk_air_metadata_from_privacy_provider(
        provider,
        fri_log_blowup_factor,
        inference_mode,
    )?;
    build_stwo_zk_air_config_artifacts(&canonical_metadata, fri_log_blowup_factor)
}

pub fn build_stwo_zk_air_config_artifacts(
    canonical_metadata: &ZkAirCanonicalMetadata,
    fri_log_blowup_factor: u32,
) -> Result<ZkAirConfigArtifacts, ZkAirMetadataBuildError> {
    let mut privacy_map = ZkPrivacyMap {
        version: ZkProofVersion::V1,
        private_columns: canonical_metadata.private_ranges.clone(),
        hash: ZkPrivacyMapHash([0; 32]),
    };
    privacy_map.hash = canonical_zk_privacy_map_hash(&privacy_map);

    let h_witness = zk_power_of_two_u64(canonical_metadata.trace_domain_log_size)?;
    let randomizer_space_entries = canonical_metadata
        .private_column_scope
        .entries
        .iter()
        .cloned()
        .map(|entry| {
            let trace_domain = CanonicCoset::new(entry.trace_domain_log_size).coset;
            let semantic_trace_domains = entry
                .semantic_trace_domain_log_sizes
                .iter()
                .map(|&log_size| {
                    let domain = CanonicCoset::new(log_size).coset;
                    ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(domain))
                })
                .collect::<Vec<_>>();
            Ok(ZkRandomizerSpaceEntry {
                range: entry.range,
                trace_domain: ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(trace_domain)),
                semantic_trace_domains,
                randomized_log_degree: zk_private_column_randomized_log_degree(&entry)?,
                randomizer_dimension: zk_private_column_randomizer_dimension(&entry)?,
            })
        })
        .collect::<Result<Vec<_>, ZkAirMetadataBuildError>>()?;
    let randomizer_space_hash = canonical_zk_randomizer_space_hash(
        canonical_metadata.private_column_scope.hash,
        &randomizer_space_entries,
    );
    let private_degree_bounds = canonical_metadata
        .private_column_scope
        .entries
        .iter()
        .map(|entry| {
            Ok(ZkColumnDegreeBound {
                range: entry.range,
                log_degree_bound: zk_private_column_randomized_log_degree(entry)?,
            })
        })
        .collect::<Result<Vec<_>, ZkAirMetadataBuildError>>()?;
    let h_batch = expected_zk_fri_batch_degree_bound(
        canonical_metadata.fri_first_layer_log_size,
        fri_log_blowup_factor,
    )
    .ok_or(ZkAirMetadataBuildError::FriBatchDegree)?;
    let metadata = ZkPublicMetadata {
        version: ZkProofVersion::V1,
        privacy_map_hash: privacy_map.hash,
        public_statement_hash: canonical_metadata.public_statement_hash,
        logup_statistical_security_budget_hash: canonical_zk_logup_statistical_security_budget_hash(
            &canonical_metadata.logup_statistical_security_budgets,
        ),
        degree_profile: ZkDegreeProfile {
            trace_domain_log_size: canonical_metadata.trace_domain_log_size,
            h_witness,
            h_batch,
            fri_first_layer_log_size: canonical_metadata.fri_first_layer_log_size,
        },
        witness_randomization: ZkWitnessRandomizationProfile {
            h_witness,
            randomizer_space_hash,
            private_column_scope_hash: canonical_metadata.private_column_scope.hash,
            private_column_degree_bounds: private_degree_bounds.clone(),
        },
        quotient_integration: ZkQuotientIntegrationProfile {
            h_batch,
            fri_first_layer_log_size: canonical_metadata.fri_first_layer_log_size,
            split_derivation_hash: canonical_zk_split_derivation_hash(
                canonical_metadata.degree_bounds.composition_log_split,
            ),
            quotient_degree_bounds: vec![canonical_metadata.quotient_degree_bound],
        },
    };
    let mut column_degree_bounds = private_degree_bounds;
    column_degree_bounds.push(canonical_metadata.quotient_degree_bound);
    let verifier_config = ZkVerificationConfig {
        metadata: metadata.clone(),
        column_degree_bounds: column_degree_bounds.clone(),
        quotient_split_mask_profile: Some(canonical_metadata.quotient_split_mask_profile),
        logup_statistical_security_budgets: canonical_metadata
            .logup_statistical_security_budgets
            .clone(),
    };
    let verifier_audit = ZkWitnessRandomizationVerifierAudit {
        privacy_map: privacy_map.clone(),
        private_column_scope: canonical_metadata.private_column_scope.clone(),
    };

    Ok(ZkAirConfigArtifacts {
        privacy_map,
        metadata,
        verifier_config,
        verifier_audit,
        column_degree_bounds,
        logup_statistical_security_budgets: canonical_metadata
            .logup_statistical_security_budgets
            .clone(),
    })
}

#[must_use]
pub fn canonical_zk_split_derivation_hash(composition_log_split: u32) -> [u8; 32] {
    let mut bytes = Vec::new();
    push_tag(&mut bytes, ZK_SPLIT_DERIVATION_HASH_DOMAIN);
    push_u32(&mut bytes, ZkProofVersion::V1.0);
    push_u32(&mut bytes, composition_log_split);
    push_tag(&mut bytes, ZK_SPLIT_IDENTITY);
    push_tag(&mut bytes, ZK_QUOTIENT_OPENING_MODEL);
    push_tag(&mut bytes, ZK_H_BATCH_RULE);
    blake2s_hash(&bytes)
}

#[must_use]
pub fn canonical_zk_quotient_split_mask_profile_hash(
    profile: ZkQuotientSplitMaskProfile,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    push_tag(&mut bytes, ZK_QUOTIENT_SPLIT_MASK_PROFILE_HASH_DOMAIN);
    push_u32(&mut bytes, ZkProofVersion::V1.0);
    push_u32(&mut bytes, profile.split_index);
    push_u32(&mut bytes, profile.split_identity_log_degree_bound);
    push_u32(&mut bytes, profile.split_mask_log_degree_bound);
    push_u64(&mut bytes, profile.h_split);
    push_column_range(&mut bytes, profile.left_range);
    push_column_range(&mut bytes, profile.right_range);
    push_u32(&mut bytes, profile.left_masked_log_degree_bound);
    push_u32(&mut bytes, profile.right_masked_log_degree_bound);
    push_tag(&mut bytes, ZK_RANDOMIZER_BASIS_ID);
    push_tag(&mut bytes, ZK_SPLIT_IDENTITY);
    push_tag(&mut bytes, ZK_QUOTIENT_SPLIT_MASK_CONSTRUCTION_ID);
    blake2s_hash(&bytes)
}

pub fn validate_zk_private_column_scope_for_witness_randomization(
    privacy_map: &ZkPrivacyMap,
    expected_scope_hash: [u8; 32],
    scope: &ZkPrivateColumnScope,
) -> Result<(), ZkPrivateColumnScopeValidationError> {
    if scope.version != ZkProofVersion::V1 {
        return Err(
            ZkPrivateColumnScopeValidationError::UnsupportedProofVersion {
                actual: scope.version.0,
            },
        );
    }
    if scope.version != privacy_map.version {
        return Err(ZkPrivateColumnScopeValidationError::VersionMismatch);
    }
    if scope.hash != expected_scope_hash {
        return Err(ZkPrivateColumnScopeValidationError::ScopeHashMismatch);
    }
    if scope.hash != canonical_zk_private_column_scope_hash(scope) {
        return Err(ZkPrivateColumnScopeValidationError::ScopeHashMismatch);
    }
    if scope.entries != scope.clone().canonicalized().entries {
        return Err(ZkPrivateColumnScopeValidationError::NonCanonicalEntries);
    }

    for &range in &privacy_map.private_columns {
        if !range.is_singleton() {
            return Err(ZkPrivateColumnScopeValidationError::NonSingletonPrivateRange { range });
        }
    }

    for entries in scope.entries.windows(2) {
        if entries[0].range == entries[1].range {
            return Err(
                ZkPrivateColumnScopeValidationError::DuplicatePrivateColumn {
                    range: entries[0].range,
                },
            );
        }
    }

    for entry in &scope.entries {
        if !privacy_map.private_columns.contains(&entry.range) {
            return Err(
                ZkPrivateColumnScopeValidationError::EntryForNonPrivateColumn {
                    range: entry.range,
                },
            );
        }
        if !entry.usage.eligible_for_witness_randomization() {
            return Err(
                ZkPrivateColumnScopeValidationError::IneligiblePrivateColumn {
                    range: entry.range,
                    usage: entry.usage,
                },
            );
        }
        if entry.semantic_trace_domain_log_sizes.is_empty()
            || !entry
                .semantic_trace_domain_log_sizes
                .contains(&entry.trace_domain_log_size)
            || entry.semantic_trace_domain_log_sizes != {
                let mut canonical = entry.semantic_trace_domain_log_sizes.clone();
                canonical.sort_unstable();
                canonical.dedup();
                canonical
            }
        {
            return Err(ZkPrivateColumnScopeValidationError::NonCanonicalEntries);
        }
    }

    for &range in &privacy_map.private_columns {
        if !scope.entries.iter().any(|entry| entry.range == range) {
            return Err(ZkPrivateColumnScopeValidationError::MissingPrivateColumn { range });
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ZkQueryClosureKind {
    OodsExtension,
    TranslatedBase,
    FriPosition,
    FutureQuotientComponent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkQueryClosureEntry {
    pub range: ZkColumnRange,
    pub kind: ZkQueryClosureKind,
    pub domain_id: u64,
    pub point_or_position: [u64; 8],
    pub coordinate_index: u8,
}

impl ZkQueryClosureEntry {
    #[must_use]
    pub const fn new(
        range: ZkColumnRange,
        kind: ZkQueryClosureKind,
        domain_id: u64,
        point_or_position: [u64; 8],
        coordinate_index: u8,
    ) -> Self {
        Self {
            range,
            kind,
            domain_id,
            point_or_position,
            coordinate_index,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkQueryClosure {
    pub entries: Vec<ZkQueryClosureEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkQueryClosureValidationError {
    NonCanonicalEntries,
    EntryForNonPrivateColumn {
        range: ZkColumnRange,
    },
    InvalidCoordinateIndex {
        entry: ZkQueryClosureEntry,
        max_exclusive: u8,
    },
    QueryDomainTooLarge {
        entry: ZkQueryClosureEntry,
        log_size: u64,
    },
    QueryDomainMismatch {
        entry: ZkQueryClosureEntry,
        expected: u64,
        actual: u64,
    },
    QueryPositionOutOfDomain {
        entry: ZkQueryClosureEntry,
        position: u64,
        domain_size: u64,
    },
    InsufficientRandomizerDimension {
        range: ZkColumnRange,
        required: u64,
        actual: u64,
    },
}

impl ZkQueryClosure {
    pub fn canonicalize(&mut self) {
        self.entries.sort_unstable();
        self.entries.dedup();
    }

    #[must_use]
    pub fn canonicalized(mut self) -> Self {
        self.canonicalize();
        self
    }

    #[must_use]
    pub fn query_count_for_range(&self, range: ZkColumnRange) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.range == range)
            .count() as u64
    }
}

fn zk_query_closure_max_coordinate_index(kind: ZkQueryClosureKind) -> u8 {
    match kind {
        ZkQueryClosureKind::OodsExtension => 4,
        ZkQueryClosureKind::TranslatedBase
        | ZkQueryClosureKind::FriPosition
        | ZkQueryClosureKind::FutureQuotientComponent => 1,
    }
}

pub fn validate_zk_query_closure_for_witness_randomization(
    privacy_map: &ZkPrivacyMap,
    metadata: &ZkPublicMetadata,
    closure: &ZkQueryClosure,
) -> Result<(), ZkQueryClosureValidationError> {
    if closure.entries != closure.clone().canonicalized().entries {
        return Err(ZkQueryClosureValidationError::NonCanonicalEntries);
    }

    for &entry in &closure.entries {
        if !privacy_map.private_columns.contains(&entry.range) {
            return Err(ZkQueryClosureValidationError::EntryForNonPrivateColumn {
                range: entry.range,
            });
        }

        let max_exclusive = zk_query_closure_max_coordinate_index(entry.kind);
        if entry.coordinate_index >= max_exclusive {
            return Err(ZkQueryClosureValidationError::InvalidCoordinateIndex {
                entry,
                max_exclusive,
            });
        }

        if entry.kind == ZkQueryClosureKind::FriPosition {
            let max_domain_id = metadata.degree_profile.fri_first_layer_log_size as u64;
            if entry.domain_id > max_domain_id {
                return Err(ZkQueryClosureValidationError::QueryDomainMismatch {
                    entry,
                    expected: max_domain_id,
                    actual: entry.domain_id,
                });
            }
            if entry.domain_id >= u64::BITS as u64 {
                return Err(ZkQueryClosureValidationError::QueryDomainTooLarge {
                    entry,
                    log_size: entry.domain_id,
                });
            }
            let domain_size = 1u64 << entry.domain_id;
            let position = entry.point_or_position[0];
            if position >= domain_size {
                return Err(ZkQueryClosureValidationError::QueryPositionOutOfDomain {
                    entry,
                    position,
                    domain_size,
                });
            }
        }
    }

    for &range in &privacy_map.private_columns {
        let required = closure.query_count_for_range(range);
        let actual = metadata.witness_randomization.h_witness;
        if actual < required {
            return Err(
                ZkQueryClosureValidationError::InsufficientRandomizerDimension {
                    range,
                    required,
                    actual,
                },
            );
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkRandomizerRankEntry {
    pub range: ZkColumnRange,
    pub query_count: u64,
    pub randomizer_dimension: u64,
    pub rank: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkRandomizerRankProfile {
    pub entries: Vec<ZkRandomizerRankEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkRandomizerRankValidationError {
    NonCanonicalEntries,
    EntryForNonPrivateColumn {
        range: ZkColumnRange,
    },
    DuplicatePrivateColumn {
        range: ZkColumnRange,
    },
    MissingPrivateColumn {
        range: ZkColumnRange,
    },
    QueryCountMismatch {
        range: ZkColumnRange,
        expected: u64,
        actual: u64,
    },
    RandomizerDimensionMismatch {
        range: ZkColumnRange,
        expected: u64,
        actual: u64,
    },
    InsufficientRandomizerDimension {
        range: ZkColumnRange,
        required: u64,
        actual: u64,
    },
    RankDeficient {
        range: ZkColumnRange,
        required: u64,
        actual: u64,
    },
}

impl ZkRandomizerRankProfile {
    pub fn canonicalize(&mut self) {
        self.entries.sort_unstable();
        self.entries.dedup();
    }

    #[must_use]
    pub fn canonicalized(mut self) -> Self {
        self.canonicalize();
        self
    }

    #[must_use]
    pub fn entry_for_range(&self, range: ZkColumnRange) -> Option<ZkRandomizerRankEntry> {
        self.entries
            .iter()
            .copied()
            .find(|entry| entry.range == range)
    }
}

pub fn validate_zk_randomizer_rank_profile_for_witness_randomization(
    privacy_map: &ZkPrivacyMap,
    metadata: &ZkPublicMetadata,
    closure: &ZkQueryClosure,
    profile: &ZkRandomizerRankProfile,
) -> Result<(), ZkRandomizerRankValidationError> {
    if profile.entries != profile.clone().canonicalized().entries {
        return Err(ZkRandomizerRankValidationError::NonCanonicalEntries);
    }

    for entries in profile.entries.windows(2) {
        if entries[0].range == entries[1].range {
            return Err(ZkRandomizerRankValidationError::DuplicatePrivateColumn {
                range: entries[0].range,
            });
        }
    }

    for &entry in &profile.entries {
        if !privacy_map.private_columns.contains(&entry.range) {
            return Err(ZkRandomizerRankValidationError::EntryForNonPrivateColumn {
                range: entry.range,
            });
        }
    }

    for &range in &privacy_map.private_columns {
        let Some(entry) = profile.entry_for_range(range) else {
            return Err(ZkRandomizerRankValidationError::MissingPrivateColumn { range });
        };
        let expected_query_count = closure.query_count_for_range(range);
        if entry.query_count != expected_query_count {
            return Err(ZkRandomizerRankValidationError::QueryCountMismatch {
                range,
                expected: expected_query_count,
                actual: entry.query_count,
            });
        }

        let max_dimension = metadata.witness_randomization.h_witness;
        if entry.randomizer_dimension > max_dimension {
            return Err(
                ZkRandomizerRankValidationError::RandomizerDimensionMismatch {
                    range,
                    expected: max_dimension,
                    actual: entry.randomizer_dimension,
                },
            );
        }
        if entry.randomizer_dimension < expected_query_count {
            return Err(
                ZkRandomizerRankValidationError::InsufficientRandomizerDimension {
                    range,
                    required: expected_query_count,
                    actual: entry.randomizer_dimension,
                },
            );
        }
        if entry.rank != expected_query_count {
            return Err(ZkRandomizerRankValidationError::RankDeficient {
                range,
                required: expected_query_count,
                actual: entry.rank,
            });
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkRandomizerQueryFunctional {
    pub range: ZkColumnRange,
    pub kind: ZkQueryClosureKind,
    pub domain_id: u64,
    pub point_or_position: [u64; 8],
    pub point: CirclePoint<SecureField>,
    pub coordinate_index: u8,
}

impl ZkRandomizerQueryFunctional {
    #[must_use]
    pub const fn closure_entry(self) -> ZkQueryClosureEntry {
        ZkQueryClosureEntry::new(
            self.range,
            self.kind,
            self.domain_id,
            self.point_or_position,
            self.coordinate_index,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkRandomizerMatrix {
    pub range: ZkColumnRange,
    pub query_count: u64,
    pub randomizer_dimension: u64,
    pub rows: Vec<Vec<BaseField>>,
}

impl ZkRandomizerMatrix {
    #[must_use]
    pub fn rank(&self) -> u64 {
        base_field_matrix_rank(self.rows.clone()) as u64
    }

    #[must_use]
    pub fn rank_entry(&self) -> ZkRandomizerRankEntry {
        ZkRandomizerRankEntry {
            range: self.range,
            query_count: self.query_count,
            randomizer_dimension: self.randomizer_dimension,
            rank: self.rank(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkRandomizerMatrixBuild {
    pub closure: ZkQueryClosure,
    pub matrices: Vec<ZkRandomizerMatrix>,
    pub rank_profile: ZkRandomizerRankProfile,
}

pub const ZK_RANDOMIZER_MATRIX_MAX_DIMENSION: u64 = 1 << 20;
pub const ZK_RANDOMIZER_MATRIX_MAX_ROWS_PER_RANGE: usize = 1 << 16;
pub const ZK_RANDOMIZER_MATRIX_MAX_CELLS_PER_RANGE: u64 = 1 << 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkRandomizerMatrixBuildError {
    FunctionalForNonPrivateColumn {
        range: ZkColumnRange,
    },
    InvalidCoordinateIndex {
        functional: ZkRandomizerQueryFunctional,
        max_exclusive: u8,
    },
    RandomizerDimensionTooLarge {
        actual: u64,
        max: u64,
    },
    RowCountTooLarge {
        range: ZkColumnRange,
        actual: usize,
        max: usize,
    },
    MatrixCellCountTooLarge {
        range: ZkColumnRange,
        rows: usize,
        columns: u64,
        max: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkSampleMetadataBuildError {
    NonSingletonPrivateRange {
        range: ZkColumnRange,
    },
    MissingTree {
        tree_index: usize,
    },
    MissingColumn {
        range: ZkColumnRange,
    },
    MissingOpeningGeometry {
        range: ZkColumnRange,
    },
    DuplicateOpeningGeometry {
        range: ZkColumnRange,
    },
    InvalidOpeningGeometry {
        range: ZkColumnRange,
    },
    MissingPrivateColumnDegreeBound {
        range: ZkColumnRange,
    },
    DuplicatePrivateColumnDegreeBound {
        range: ZkColumnRange,
    },
    InvalidPrivateColumnDegreeBound {
        range: ZkColumnRange,
    },
    PrivateColumnDegreeBoundTooSmall {
        range: ZkColumnRange,
        required: u32,
        actual: u32,
    },
    CommittedColumnLogSizeOverflow {
        range: ZkColumnRange,
        log_degree_bound: u32,
        log_blowup_factor: u32,
    },
    CommittedColumnLogSizeMismatch {
        range: ZkColumnRange,
        expected: u32,
        actual: u32,
    },
    QueryPositionOutOfDomain {
        position: usize,
        domain_size: usize,
    },
    TraceDomainTooLarge {
        log_size: u32,
    },
    LiftingDomainTooLarge {
        log_size: u32,
    },
    RandomizerMatrix(ZkRandomizerMatrixBuildError),
}

fn zk_randomizer_dimension_from_trace_log_size(
    trace_log_size: u32,
) -> Result<u64, ZkSampleMetadataBuildError> {
    1u64.checked_shl(trace_log_size)
        .ok_or(ZkSampleMetadataBuildError::TraceDomainTooLarge {
            log_size: trace_log_size,
        })
}

fn validate_zk_private_opening_geometry_for_witness_randomization(
    privacy_map: &ZkPrivacyMap,
    private_column_scope: &ZkPrivateColumnScope,
    private_column_degree_bounds: &[ZkColumnDegreeBound],
    opening_geometry: &[ZkPrivateColumnOpeningGeometry],
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Result<(), ZkSampleMetadataBuildError> {
    let mut seen_geometry = BTreeSet::new();
    for geometry in opening_geometry {
        if !geometry.range.is_singleton()
            || !privacy_map.private_columns.contains(&geometry.range)
            || geometry.tree_height > lifting_log_size
            || geometry.committed_column_log_size > geometry.tree_height
        {
            return Err(ZkSampleMetadataBuildError::InvalidOpeningGeometry {
                range: geometry.range,
            });
        }
        if !seen_geometry.insert(geometry.range) {
            return Err(ZkSampleMetadataBuildError::DuplicateOpeningGeometry {
                range: geometry.range,
            });
        }
    }

    let mut seen_bounds = BTreeSet::new();
    for bound in private_column_degree_bounds {
        if !bound.range.is_singleton() || !privacy_map.private_columns.contains(&bound.range) {
            return Err(
                ZkSampleMetadataBuildError::InvalidPrivateColumnDegreeBound { range: bound.range },
            );
        }
        if !seen_bounds.insert(bound.range) {
            return Err(
                ZkSampleMetadataBuildError::DuplicatePrivateColumnDegreeBound {
                    range: bound.range,
                },
            );
        }
    }

    for &range in &privacy_map.private_columns {
        if !private_column_scope
            .entries
            .iter()
            .any(|entry| entry.range == range)
        {
            return Err(ZkSampleMetadataBuildError::MissingPrivateColumnDegreeBound { range });
        }
    }

    for entry in &private_column_scope.entries {
        if !privacy_map.private_columns.contains(&entry.range) {
            continue;
        }
        let geometry = opening_geometry
            .iter()
            .find(|geometry| geometry.range == entry.range)
            .ok_or(ZkSampleMetadataBuildError::MissingOpeningGeometry { range: entry.range })?;
        let bound = private_column_degree_bounds
            .iter()
            .find(|bound| bound.range == entry.range)
            .ok_or(
                ZkSampleMetadataBuildError::MissingPrivateColumnDegreeBound { range: entry.range },
            )?;
        let required_log_degree = zk_private_column_randomized_log_degree(entry).map_err(|_| {
            ZkSampleMetadataBuildError::InvalidPrivateColumnDegreeBound { range: entry.range }
        })?;
        if bound.log_degree_bound < required_log_degree {
            return Err(
                ZkSampleMetadataBuildError::PrivateColumnDegreeBoundTooSmall {
                    range: entry.range,
                    required: required_log_degree,
                    actual: bound.log_degree_bound,
                },
            );
        }
        let expected_committed_log_size = bound
            .log_degree_bound
            .checked_add(log_blowup_factor)
            .ok_or(ZkSampleMetadataBuildError::CommittedColumnLogSizeOverflow {
                range: entry.range,
                log_degree_bound: bound.log_degree_bound,
                log_blowup_factor,
            })?;
        if geometry.committed_column_log_size != expected_committed_log_size {
            return Err(ZkSampleMetadataBuildError::CommittedColumnLogSizeMismatch {
                range: entry.range,
                expected: expected_committed_log_size,
                actual: geometry.committed_column_log_size,
            });
        }
    }

    Ok(())
}

fn zk_private_opening_geometry_for_range(
    opening_geometry: &[ZkPrivateColumnOpeningGeometry],
    range: ZkColumnRange,
) -> Result<ZkPrivateColumnOpeningGeometry, ZkSampleMetadataBuildError> {
    opening_geometry
        .iter()
        .copied()
        .find(|geometry| geometry.range == range)
        .ok_or(ZkSampleMetadataBuildError::MissingOpeningGeometry { range })
}

fn zk_project_fri_position_to_committed_column_position(
    range: ZkColumnRange,
    position: usize,
    lifting_log_size: u32,
    geometry: ZkPrivateColumnOpeningGeometry,
) -> Result<usize, ZkSampleMetadataBuildError> {
    let domain_size = 1usize.checked_shl(lifting_log_size).ok_or(
        ZkSampleMetadataBuildError::LiftingDomainTooLarge {
            log_size: lifting_log_size,
        },
    )?;
    if position >= domain_size {
        return Err(ZkSampleMetadataBuildError::QueryPositionOutOfDomain {
            position,
            domain_size,
        });
    }
    let committed_domain_size = 1usize
        .checked_shl(geometry.committed_column_log_size)
        .ok_or(ZkSampleMetadataBuildError::LiftingDomainTooLarge {
            log_size: geometry.committed_column_log_size,
        })?;
    let tree_position = if geometry.tree_height == 0 {
        0
    } else if geometry.tree_height == lifting_log_size {
        position
    } else {
        prepare_preprocessed_query_positions(&[position], lifting_log_size, geometry.tree_height)
            .into_iter()
            .next()
            .ok_or(ZkSampleMetadataBuildError::InvalidOpeningGeometry { range })?
    };
    let column_shift = geometry
        .tree_height
        .checked_sub(geometry.committed_column_log_size)
        .ok_or(ZkSampleMetadataBuildError::InvalidOpeningGeometry { range })?;
    let row_projection_shift = column_shift
        .checked_add(1)
        .ok_or(ZkSampleMetadataBuildError::InvalidOpeningGeometry { range })?;
    let projected_position = (tree_position >> row_projection_shift << 1) + (tree_position & 1);
    if projected_position >= committed_domain_size {
        return Err(ZkSampleMetadataBuildError::QueryPositionOutOfDomain {
            position: projected_position,
            domain_size: committed_domain_size,
        });
    }

    Ok(projected_position)
}

fn zk_committed_column_point_at_position(
    position: usize,
    committed_column_log_size: u32,
) -> Result<CirclePoint<SecureField>, ZkSampleMetadataBuildError> {
    let domain_size = 1usize.checked_shl(committed_column_log_size).ok_or(
        ZkSampleMetadataBuildError::LiftingDomainTooLarge {
            log_size: committed_column_log_size,
        },
    )?;
    if position >= domain_size {
        return Err(ZkSampleMetadataBuildError::QueryPositionOutOfDomain {
            position,
            domain_size,
        });
    }

    let committed_domain = CanonicCoset::try_new(committed_column_log_size)
        .map_err(|_| ZkSampleMetadataBuildError::LiftingDomainTooLarge {
            log_size: committed_column_log_size,
        })?
        .circle_domain();
    let domain_index = bit_reverse_index(position, committed_column_log_size);

    Ok(committed_domain.at(domain_index).into_ef::<SecureField>())
}

pub fn build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
    privacy_map: &ZkPrivacyMap,
    private_column_scope: &ZkPrivateColumnScope,
    private_column_degree_bounds: &[ZkColumnDegreeBound],
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
    opening_geometry: &[ZkPrivateColumnOpeningGeometry],
    log_blowup_factor: u32,
) -> Result<ZkRandomizerMatrixBuild, ZkSampleMetadataBuildError> {
    validate_zk_private_opening_geometry_for_witness_randomization(
        privacy_map,
        private_column_scope,
        private_column_degree_bounds,
        opening_geometry,
        lifting_log_size,
        log_blowup_factor,
    )?;

    let mut closure_entries = Vec::new();
    let mut matrices = Vec::new();
    let mut rank_entries = Vec::new();

    for entry in &private_column_scope.entries {
        if !privacy_map.private_columns.contains(&entry.range) {
            continue;
        }
        let semantic_domains = entry
            .semantic_trace_domain_log_sizes
            .iter()
            .map(|&log_size| {
                CanonicCoset::try_new(log_size)
                    .map(|domain| domain.coset)
                    .map_err(|_| ZkSampleMetadataBuildError::TraceDomainTooLarge { log_size })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let scoped_privacy_map = ZkPrivacyMap {
            version: privacy_map.version,
            private_columns: vec![entry.range],
            hash: privacy_map.hash,
        };
        let geometry = zk_private_opening_geometry_for_range(opening_geometry, entry.range)?;
        let functionals =
            zk_randomizer_query_functionals_from_stwo_sample_metadata_with_opening_geometry(
                &scoped_privacy_map,
                sampled_points,
                fri_query_positions,
                lifting_log_size,
                &[geometry],
            )?;
        let build = build_zk_randomizer_matrices_for_witness_randomization_with_semantic_domains(
            &semantic_domains,
            &scoped_privacy_map,
            zk_randomizer_dimension_from_trace_log_size(entry.trace_domain_log_size)?,
            &functionals,
        )
        .map_err(ZkSampleMetadataBuildError::RandomizerMatrix)?;
        closure_entries.extend(build.closure.entries);
        matrices.extend(build.matrices);
        rank_entries.extend(build.rank_profile.entries);
    }

    let mut closure = ZkQueryClosure {
        entries: closure_entries,
    };
    closure.canonicalize();
    let mut rank_profile = ZkRandomizerRankProfile {
        entries: rank_entries,
    };
    rank_profile.canonicalize();

    Ok(ZkRandomizerMatrixBuild {
        closure,
        matrices,
        rank_profile,
    })
}

pub fn build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope(
    privacy_map: &ZkPrivacyMap,
    private_column_scope: &ZkPrivateColumnScope,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
) -> Result<ZkRandomizerMatrixBuild, ZkSampleMetadataBuildError> {
    let mut closure_entries = Vec::new();
    let mut matrices = Vec::new();
    let mut rank_entries = Vec::new();

    for entry in &private_column_scope.entries {
        if !privacy_map.private_columns.contains(&entry.range) {
            continue;
        }
        let semantic_domains = entry
            .semantic_trace_domain_log_sizes
            .iter()
            .map(|&log_size| {
                CanonicCoset::try_new(log_size)
                    .map(|domain| domain.coset)
                    .map_err(|_| ZkSampleMetadataBuildError::TraceDomainTooLarge { log_size })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let scoped_privacy_map = ZkPrivacyMap {
            version: privacy_map.version,
            private_columns: vec![entry.range],
            hash: privacy_map.hash,
        };
        let build = build_zk_randomizer_matrices_from_stwo_sample_metadata_with_semantic_domains(
            &semantic_domains,
            &scoped_privacy_map,
            zk_randomizer_dimension_from_trace_log_size(entry.trace_domain_log_size)?,
            sampled_points,
            fri_query_positions,
            lifting_log_size,
        )?;
        closure_entries.extend(build.closure.entries);
        matrices.extend(build.matrices);
        rank_entries.extend(build.rank_profile.entries);
    }

    let mut closure = ZkQueryClosure {
        entries: closure_entries,
    };
    closure.canonicalize();
    let mut rank_profile = ZkRandomizerRankProfile {
        entries: rank_entries,
    };
    rank_profile.canonicalize();

    Ok(ZkRandomizerMatrixBuild {
        closure,
        matrices,
        rank_profile,
    })
}

pub fn build_zk_randomizer_matrices_from_stwo_sample_metadata_with_semantic_domains(
    semantic_domains: &[Coset],
    privacy_map: &ZkPrivacyMap,
    randomizer_dimension: u64,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
) -> Result<ZkRandomizerMatrixBuild, ZkSampleMetadataBuildError> {
    let functionals = zk_randomizer_query_functionals_from_stwo_sample_metadata(
        privacy_map,
        sampled_points,
        fri_query_positions,
        lifting_log_size,
    )?;
    build_zk_randomizer_matrices_for_witness_randomization_with_semantic_domains(
        semantic_domains,
        privacy_map,
        randomizer_dimension,
        &functionals,
    )
    .map_err(ZkSampleMetadataBuildError::RandomizerMatrix)
}

pub fn build_zk_randomizer_matrices_from_stwo_sample_metadata(
    trace_domain: Coset,
    privacy_map: &ZkPrivacyMap,
    randomizer_dimension: u64,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
) -> Result<ZkRandomizerMatrixBuild, ZkSampleMetadataBuildError> {
    let functionals = zk_randomizer_query_functionals_from_stwo_sample_metadata(
        privacy_map,
        sampled_points,
        fri_query_positions,
        lifting_log_size,
    )?;

    build_zk_randomizer_matrices_for_witness_randomization(
        trace_domain,
        privacy_map,
        randomizer_dimension,
        &functionals,
    )
    .map_err(ZkSampleMetadataBuildError::RandomizerMatrix)
}

fn zk_randomizer_query_functionals_from_stwo_sample_metadata(
    privacy_map: &ZkPrivacyMap,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
) -> Result<Vec<ZkRandomizerQueryFunctional>, ZkSampleMetadataBuildError> {
    let domain_size = 1usize.checked_shl(lifting_log_size).ok_or(
        ZkSampleMetadataBuildError::LiftingDomainTooLarge {
            log_size: lifting_log_size,
        },
    )?;
    if let Some(&position) = fri_query_positions
        .iter()
        .find(|&&position| position >= domain_size)
    {
        return Err(ZkSampleMetadataBuildError::QueryPositionOutOfDomain {
            position,
            domain_size,
        });
    }

    let lifting_domain = CanonicCoset::new(lifting_log_size).circle_domain();
    let mut functionals = Vec::new();

    for &range in &privacy_map.private_columns {
        if !range.is_singleton() {
            return Err(ZkSampleMetadataBuildError::NonSingletonPrivateRange { range });
        }
        let tree = sampled_points.get(range.tree_index).ok_or(
            ZkSampleMetadataBuildError::MissingTree {
                tree_index: range.tree_index,
            },
        )?;
        let column_points = tree
            .get(range.column_start)
            .ok_or(ZkSampleMetadataBuildError::MissingColumn { range })?;

        for &point in column_points {
            let point_encoding = encode_zk_query_point(point);
            for coordinate_index in 0..SECURE_EXTENSION_DEGREE {
                functionals.push(ZkRandomizerQueryFunctional {
                    range,
                    kind: ZkQueryClosureKind::OodsExtension,
                    domain_id: range.tree_index as u64,
                    point_or_position: point_encoding,
                    point,
                    coordinate_index: coordinate_index as u8,
                });
            }
        }

        for &position in fri_query_positions {
            let domain_index = bit_reverse_index(position, lifting_log_size);
            let point = lifting_domain.at(domain_index).into_ef::<SecureField>();
            functionals.push(ZkRandomizerQueryFunctional {
                range,
                kind: ZkQueryClosureKind::FriPosition,
                domain_id: lifting_log_size as u64,
                point_or_position: encode_zk_query_position(position),
                point,
                coordinate_index: 0,
            });
        }
    }

    Ok(functionals)
}

fn zk_randomizer_query_functionals_from_stwo_sample_metadata_with_opening_geometry(
    privacy_map: &ZkPrivacyMap,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
    opening_geometry: &[ZkPrivateColumnOpeningGeometry],
) -> Result<Vec<ZkRandomizerQueryFunctional>, ZkSampleMetadataBuildError> {
    let domain_size = 1usize.checked_shl(lifting_log_size).ok_or(
        ZkSampleMetadataBuildError::LiftingDomainTooLarge {
            log_size: lifting_log_size,
        },
    )?;
    if let Some(&position) = fri_query_positions
        .iter()
        .find(|&&position| position >= domain_size)
    {
        return Err(ZkSampleMetadataBuildError::QueryPositionOutOfDomain {
            position,
            domain_size,
        });
    }

    let mut functionals = Vec::new();

    for &range in &privacy_map.private_columns {
        if !range.is_singleton() {
            return Err(ZkSampleMetadataBuildError::NonSingletonPrivateRange { range });
        }
        let geometry = zk_private_opening_geometry_for_range(opening_geometry, range)?;
        let tree = sampled_points.get(range.tree_index).ok_or(
            ZkSampleMetadataBuildError::MissingTree {
                tree_index: range.tree_index,
            },
        )?;
        let column_points = tree
            .get(range.column_start)
            .ok_or(ZkSampleMetadataBuildError::MissingColumn { range })?;

        let oods_doubling = lifting_log_size
            .checked_sub(geometry.committed_column_log_size)
            .ok_or(ZkSampleMetadataBuildError::InvalidOpeningGeometry { range })?;
        for &raw_point in column_points {
            let point = raw_point.repeated_double(oods_doubling);
            let point_encoding = encode_zk_query_point(point);
            for coordinate_index in 0..SECURE_EXTENSION_DEGREE {
                functionals.push(ZkRandomizerQueryFunctional {
                    range,
                    kind: ZkQueryClosureKind::OodsExtension,
                    domain_id: geometry.committed_column_log_size as u64,
                    point_or_position: point_encoding,
                    point,
                    coordinate_index: coordinate_index as u8,
                });
            }
        }

        for &position in fri_query_positions {
            let committed_position = zk_project_fri_position_to_committed_column_position(
                range,
                position,
                lifting_log_size,
                geometry,
            )?;
            let point = zk_committed_column_point_at_position(
                committed_position,
                geometry.committed_column_log_size,
            )?;
            functionals.push(ZkRandomizerQueryFunctional {
                range,
                kind: ZkQueryClosureKind::FriPosition,
                domain_id: geometry.committed_column_log_size as u64,
                point_or_position: encode_zk_query_position(committed_position),
                point,
                coordinate_index: 0,
            });
        }
    }

    Ok(functionals)
}

pub fn build_zk_randomizer_matrices_for_witness_randomization(
    trace_domain: Coset,
    privacy_map: &ZkPrivacyMap,
    randomizer_dimension: u64,
    functionals: &[ZkRandomizerQueryFunctional],
) -> Result<ZkRandomizerMatrixBuild, ZkRandomizerMatrixBuildError> {
    build_zk_randomizer_matrices_for_witness_randomization_with_semantic_domains(
        &[trace_domain],
        privacy_map,
        randomizer_dimension,
        functionals,
    )
}

pub fn build_zk_randomizer_matrices_for_witness_randomization_with_semantic_domains(
    semantic_domains: &[Coset],
    privacy_map: &ZkPrivacyMap,
    randomizer_dimension: u64,
    functionals: &[ZkRandomizerQueryFunctional],
) -> Result<ZkRandomizerMatrixBuild, ZkRandomizerMatrixBuildError> {
    if randomizer_dimension > ZK_RANDOMIZER_MATRIX_MAX_DIMENSION {
        return Err(ZkRandomizerMatrixBuildError::RandomizerDimensionTooLarge {
            actual: randomizer_dimension,
            max: ZK_RANDOMIZER_MATRIX_MAX_DIMENSION,
        });
    }
    for &functional in functionals {
        if !privacy_map.private_columns.contains(&functional.range) {
            return Err(
                ZkRandomizerMatrixBuildError::FunctionalForNonPrivateColumn {
                    range: functional.range,
                },
            );
        }
        let max_exclusive = zk_query_closure_max_coordinate_index(functional.kind);
        if functional.coordinate_index >= max_exclusive {
            return Err(ZkRandomizerMatrixBuildError::InvalidCoordinateIndex {
                functional,
                max_exclusive,
            });
        }
    }

    let mut canonical_functionals: Vec<(ZkQueryClosureEntry, ZkRandomizerQueryFunctional)> =
        functionals
            .iter()
            .copied()
            .map(|functional| (functional.closure_entry(), functional))
            .collect();
    canonical_functionals.sort_unstable_by_key(|(entry, _)| *entry);
    canonical_functionals.dedup_by_key(|(entry, _)| *entry);

    let closure = ZkQueryClosure {
        entries: canonical_functionals
            .iter()
            .map(|(entry, _)| *entry)
            .collect(),
    };

    let ambient_log_dimension = log2_ceil_u64(randomizer_dimension);
    let mut matrices = Vec::with_capacity(privacy_map.private_columns.len());

    for &range in &privacy_map.private_columns {
        let row_count = canonical_functionals
            .iter()
            .filter(|(_, functional)| functional.range == range)
            .count();
        if row_count > ZK_RANDOMIZER_MATRIX_MAX_ROWS_PER_RANGE {
            return Err(ZkRandomizerMatrixBuildError::RowCountTooLarge {
                range,
                actual: row_count,
                max: ZK_RANDOMIZER_MATRIX_MAX_ROWS_PER_RANGE,
            });
        }
        let cell_count = (row_count as u128) * (randomizer_dimension as u128);
        if cell_count > ZK_RANDOMIZER_MATRIX_MAX_CELLS_PER_RANGE as u128 {
            return Err(ZkRandomizerMatrixBuildError::MatrixCellCountTooLarge {
                range,
                rows: row_count,
                columns: randomizer_dimension,
                max: ZK_RANDOMIZER_MATRIX_MAX_CELLS_PER_RANGE,
            });
        }

        let rows: Vec<Vec<BaseField>> = canonical_functionals
            .iter()
            .filter_map(|(_, functional)| (functional.range == range).then_some(*functional))
            .map(|functional| {
                build_zk_randomizer_matrix_row(
                    semantic_domains,
                    randomizer_dimension,
                    ambient_log_dimension,
                    functional,
                )
            })
            .collect();
        matrices.push(ZkRandomizerMatrix {
            range,
            query_count: rows.len() as u64,
            randomizer_dimension,
            rows,
        });
    }

    let mut rank_profile = ZkRandomizerRankProfile {
        entries: matrices
            .iter()
            .map(ZkRandomizerMatrix::rank_entry)
            .collect(),
    };
    rank_profile.canonicalize();

    Ok(ZkRandomizerMatrixBuild {
        closure,
        matrices,
        rank_profile,
    })
}

#[must_use]
pub fn encode_zk_query_point(point: CirclePoint<SecureField>) -> [u64; 8] {
    let x = point.x.to_m31_array();
    let y = point.y.to_m31_array();
    [
        x[0].0 as u64,
        x[1].0 as u64,
        x[2].0 as u64,
        x[3].0 as u64,
        y[0].0 as u64,
        y[1].0 as u64,
        y[2].0 as u64,
        y[3].0 as u64,
    ]
}

#[must_use]
pub fn encode_zk_query_position(position: usize) -> [u64; 8] {
    [position as u64, 0, 0, 0, 0, 0, 0, 0]
}

fn build_zk_randomizer_matrix_row(
    semantic_domains: &[Coset],
    randomizer_dimension: u64,
    ambient_log_dimension: usize,
    functional: ZkRandomizerQueryFunctional,
) -> Vec<BaseField> {
    let vanishing = semantic_domains
        .iter()
        .fold(SecureField::one(), |acc, &semantic_domain| {
            let half_domain = zk_trace_domain_half_coset(semantic_domain);
            acc * coset_vanishing(half_domain, functional.point)
                * coset_vanishing(half_domain.conjugate(), functional.point)
        });
    let folding_alphas = get_folding_alphas(functional.point, ambient_log_dimension);
    let coordinate_index = functional.coordinate_index as usize;

    (0..randomizer_dimension)
        .map(|basis_index| {
            let basis_eval =
                fft_basis_element_evaluation(basis_index, ambient_log_dimension, &folding_alphas);
            (vanishing * basis_eval).to_m31_array()[coordinate_index]
        })
        .collect()
}

fn fft_basis_element_evaluation(
    basis_index: u64,
    log_dimension: usize,
    folding_alphas: &[SecureField],
) -> SecureField {
    let mut evaluation = SecureField::one();
    for (level, &alpha) in folding_alphas.iter().enumerate() {
        let bit_index = log_dimension - 1 - level;
        if ((basis_index >> bit_index) & 1) == 1 {
            evaluation *= alpha;
        }
    }
    evaluation
}

fn log2_ceil_u64(value: u64) -> usize {
    if value <= 1 {
        return 0;
    }
    (u64::BITS - (value - 1).leading_zeros()) as usize
}

#[must_use]
pub fn base_field_matrix_rank(mut rows: Vec<Vec<BaseField>>) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let column_count = rows.iter().map(Vec::len).min().unwrap_or(0);
    let mut rank = 0;

    for column in 0..column_count {
        let Some(pivot_offset) = rows[rank..].iter().position(|row| !row[column].is_zero()) else {
            continue;
        };
        let pivot = rank + pivot_offset;
        rows.swap(rank, pivot);

        let pivot_inverse = rows[rank][column].inverse();
        for value in &mut rows[rank][column..] {
            *value *= pivot_inverse;
        }

        let pivot_row = rows[rank].clone();
        for (row_index, row) in rows.iter_mut().enumerate() {
            if row_index == rank || row[column].is_zero() {
                continue;
            }
            let factor = row[column];
            for col in column..column_count {
                row[col] -= factor * pivot_row[col];
            }
        }

        rank += 1;
        if rank == rows.len() {
            break;
        }
    }

    rank
}

/// Public degree profile for a ZK proof.
///
/// `h_witness` is the witness-randomizer degree budget. `h_batch` is the
/// Protocol 2 FRI batch-mask degree budget. `fri_first_layer_log_size` is the
/// committed first-layer domain used for `H_batch = raw_quotient + R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkDegreeProfile {
    pub trace_domain_log_size: u32,
    pub h_witness: u64,
    pub h_batch: u64,
    pub fri_first_layer_log_size: u32,
}

/// Verifier-owned degree bound for a stable tree/column range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkColumnDegreeBound {
    pub range: ZkColumnRange,
    pub log_degree_bound: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkPrivateColumnOpeningGeometry {
    pub range: ZkColumnRange,
    pub tree_height: u32,
    pub committed_column_log_size: u32,
}

/// Public profile for Phase 2 witness randomization.
///
/// `randomizer_space_hash` must bind the reviewed STWO circle randomizer-space
/// derivation, including basis, dimension, `v_H * r_i` construction, and query
/// closure rank argument. The hash is public review evidence, not secret
/// randomness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkWitnessRandomizationProfile {
    pub h_witness: u64,
    pub randomizer_space_hash: [u8; 32],
    pub private_column_scope_hash: [u8; 32],
    pub private_column_degree_bounds: Vec<ZkColumnDegreeBound>,
}

/// Public profile for Phase 3 quotient/OODS/FRI integration.
///
/// `split_derivation_hash` binds the reviewed STWO `split_at_mid`
/// query-expansion derivation. `h_batch` is the reviewed Protocol 2 batch-mask
/// degree budget for `H_batch = raw_quotient + R`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkQuotientIntegrationProfile {
    pub h_batch: u64,
    pub fri_first_layer_log_size: u32,
    pub split_derivation_hash: [u8; 32],
    pub quotient_degree_bounds: Vec<ZkColumnDegreeBound>,
}

/// Passive metadata for the reviewed STWO quotient-split masking design.
///
/// This profile describes the algebraic cancellation
/// `left_hat = left + Pi_L * t`, `right_hat = right - t` for STWO's
/// `split_at_mid` identity. It is deliberately not part of the active proof
/// format yet; activation requires prover and verifier wiring plus separate
/// soundness review.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkQuotientSplitMaskProfile {
    pub split_index: u32,
    pub split_identity_log_degree_bound: u32,
    pub split_mask_log_degree_bound: u32,
    pub h_split: u64,
    pub left_range: ZkColumnRange,
    pub right_range: ZkColumnRange,
    pub left_masked_log_degree_bound: u32,
    pub right_masked_log_degree_bound: u32,
}

pub fn stwo_composition_quotient_split_mask_profile(
    composition_tree_index: usize,
    split_identity_log_degree_bound: u32,
    split_mask_log_degree_bound: u32,
    h_split: u64,
    left_masked_log_degree_bound: u32,
    right_masked_log_degree_bound: u32,
) -> Result<ZkQuotientSplitMaskProfile, ZkQuotientSplitMaskProfileValidationError> {
    let profile = ZkQuotientSplitMaskProfile {
        split_index: 0,
        split_identity_log_degree_bound,
        split_mask_log_degree_bound,
        h_split,
        left_range: ZkColumnRange::new(composition_tree_index, 0, SECURE_EXTENSION_DEGREE),
        right_range: ZkColumnRange::new(
            composition_tree_index,
            SECURE_EXTENSION_DEGREE,
            2 * SECURE_EXTENSION_DEGREE,
        ),
        left_masked_log_degree_bound,
        right_masked_log_degree_bound,
    };
    validate_zk_quotient_split_mask_profile(profile)?;

    Ok(profile)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkQuotientSplitMaskProfileValidationError {
    EmptySplitMask,
    EmptySplitRange {
        range: ZkColumnRange,
    },
    SplitRangeTreeMismatch {
        left_range: ZkColumnRange,
        right_range: ZkColumnRange,
    },
    SplitRangeWidthMismatch {
        left_range: ZkColumnRange,
        right_range: ZkColumnRange,
    },
    NonContiguousSplitRanges {
        left_range: ZkColumnRange,
        right_range: ZkColumnRange,
    },
    SplitIdentityUnderflow {
        split_identity_log_degree_bound: u32,
    },
    SplitMaskDegreeExceedsSplitComponent {
        split_mask_log_degree_bound: u32,
        split_component_log_degree_bound: u32,
    },
    SplitMaskDimensionTooLarge {
        split_mask_log_degree_bound: u32,
    },
    SplitMaskEntropyTooLarge {
        h_split: u64,
        max: u64,
    },
    LeftMaskedBoundBelowCancellationProduct {
        required: u32,
        actual: u32,
    },
    RightMaskedBoundBelowOriginalSplit {
        required: u32,
        actual: u32,
    },
}

/// Validates passive quotient-split masking metadata for STWO's current
/// two-way `split_at_mid` identity.
///
/// Let `L = split_identity_log_degree_bound` and `S = L - 1`. The reviewed
/// conservative bound requires `t` to fit in the original split-component
/// space (`T <= S`), `right_hat` to retain at least the original split bound,
/// and `left_hat` to allow the `Pi_L * t` cancellation product.
pub fn validate_zk_quotient_split_mask_profile(
    profile: ZkQuotientSplitMaskProfile,
) -> Result<(), ZkQuotientSplitMaskProfileValidationError> {
    if profile.h_split == 0 {
        return Err(ZkQuotientSplitMaskProfileValidationError::EmptySplitMask);
    }
    if profile.left_range.column_start >= profile.left_range.column_end {
        return Err(ZkQuotientSplitMaskProfileValidationError::EmptySplitRange {
            range: profile.left_range,
        });
    }
    if profile.right_range.column_start >= profile.right_range.column_end {
        return Err(ZkQuotientSplitMaskProfileValidationError::EmptySplitRange {
            range: profile.right_range,
        });
    }
    if profile.left_range.tree_index != profile.right_range.tree_index {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::SplitRangeTreeMismatch {
                left_range: profile.left_range,
                right_range: profile.right_range,
            },
        );
    }
    let left_range_width = profile.left_range.column_end - profile.left_range.column_start;
    let right_range_width = profile.right_range.column_end - profile.right_range.column_start;
    if left_range_width != right_range_width {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::SplitRangeWidthMismatch {
                left_range: profile.left_range,
                right_range: profile.right_range,
            },
        );
    }
    if profile.left_range.column_end != profile.right_range.column_start {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::NonContiguousSplitRanges {
                left_range: profile.left_range,
                right_range: profile.right_range,
            },
        );
    }

    if profile.split_identity_log_degree_bound < 2 {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::SplitIdentityUnderflow {
                split_identity_log_degree_bound: profile.split_identity_log_degree_bound,
            },
        );
    }
    let split_component_log_degree_bound = profile.split_identity_log_degree_bound - 1;
    if profile.split_mask_log_degree_bound > split_component_log_degree_bound {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::SplitMaskDegreeExceedsSplitComponent {
                split_mask_log_degree_bound: profile.split_mask_log_degree_bound,
                split_component_log_degree_bound,
            },
        );
    }
    let max_h_split = 1u64
        .checked_shl(profile.split_mask_log_degree_bound)
        .ok_or(
            ZkQuotientSplitMaskProfileValidationError::SplitMaskDimensionTooLarge {
                split_mask_log_degree_bound: profile.split_mask_log_degree_bound,
            },
        )?;
    if profile.h_split > max_h_split {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::SplitMaskEntropyTooLarge {
                h_split: profile.h_split,
                max: max_h_split,
            },
        );
    }
    if profile.left_masked_log_degree_bound < profile.split_identity_log_degree_bound {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::LeftMaskedBoundBelowCancellationProduct {
                required: profile.split_identity_log_degree_bound,
                actual: profile.left_masked_log_degree_bound,
            },
        );
    }
    if profile.right_masked_log_degree_bound < split_component_log_degree_bound {
        return Err(
            ZkQuotientSplitMaskProfileValidationError::RightMaskedBoundBelowOriginalSplit {
                required: split_component_log_degree_bound,
                actual: profile.right_masked_log_degree_bound,
            },
        );
    }

    Ok(())
}

/// Public metadata echoed by a ZK proof and compared against verifier-owned
/// configuration before affected Fiat-Shamir challenges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPublicMetadata {
    pub version: ZkProofVersion,
    pub privacy_map_hash: ZkPrivacyMapHash,
    pub public_statement_hash: ZkPublicStatementHash,
    pub logup_statistical_security_budget_hash: [u8; 32],
    pub degree_profile: ZkDegreeProfile,
    pub witness_randomization: ZkWitnessRandomizationProfile,
    pub quotient_integration: ZkQuotientIntegrationProfile,
}

/// Verifier-owned witness-randomization audit inputs.
///
/// Query closure and rank profile are derived during verification from the
/// verifier-constructed sample points and Fiat-Shamir FRI query positions,
/// because those are proof-specific. This audit block carries only stable
/// verifier policy needed to perform that dynamic rank check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkWitnessRandomizationVerifierAudit {
    pub privacy_map: ZkPrivacyMap,
    pub private_column_scope: ZkPrivateColumnScope,
}

/// Verifier-owned ZK configuration. Verification trusts this configuration,
/// not the proof's echoed metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkVerificationConfig {
    pub metadata: ZkPublicMetadata,
    pub column_degree_bounds: Vec<ZkColumnDegreeBound>,
    pub quotient_split_mask_profile: Option<ZkQuotientSplitMaskProfile>,
    pub logup_statistical_security_budgets: Vec<ZkLogupStatisticalSecurityBudget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkWitnessRandomizationVerifierAuditError {
    MissingAudit,
    UnexpectedAuditForPublicOnly,
    PrivacyMapVersionMismatch,
    PrivacyMapHashMismatch,
    PrivateColumnDegreeBoundsMismatch,
    MissingPrivateLogupStatisticalSecurityBudget,
    RandomizerSpaceHashMismatch,
    InvalidTraceDomainLogSize { log_size: u32 },
    PrivateColumnScope(ZkPrivateColumnScopeValidationError),
    QueryClosure(ZkQueryClosureValidationError),
    RandomizerRank(ZkRandomizerRankValidationError),
    SampleMetadata(ZkSampleMetadataBuildError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkMetadataValidationError {
    UnsupportedProofVersion {
        actual: u32,
    },
    InvalidFriBatchDegreeBound,
    FriFirstLayerLogSizeMismatch {
        expected: u32,
        actual: u32,
    },
    QuotientFirstLayerLogSizeMismatch {
        expected: u32,
        actual: u32,
    },
    FriBatchDegreeMismatch {
        expected: u64,
        actual: u64,
    },
    QuotientFriBatchDegreeMismatch {
        expected: u64,
        actual: u64,
    },
    WitnessRandomizationDegreeMismatch {
        expected: u64,
        actual: u64,
    },
    MissingPrivateColumnDegreeBounds,
    MissingQuotientDegreeBounds,
    InvalidColumnDegreeBoundRange {
        range: ZkColumnRange,
    },
    EmptyRandomizerSpaceHash,
    EmptyPrivateColumnScopeHash,
    EmptySplitDerivationHash,
    EmptyWitnessRandomizer,
    WitnessRandomizerDimensionTooLarge {
        dimension: u64,
    },
    WitnessRandomizedDomainNotLarger {
        trace_domain_log_size: u32,
        randomized_log_degree: u32,
    },
    WitnessRandomizedDomainTooSmall {
        trace_domain_size: u64,
        randomizer_coefficient_count: u64,
        randomized_domain_size: u64,
    },
    UnexpectedWitnessRandomizationForPhase1,
    UnexpectedPrivateColumnDegreeBoundsForPhase1,
    UnexpectedQuotientDegreeBoundsForPhase1,
    UnexpectedRandomizerSpaceHashForPhase1,
    UnexpectedPrivateColumnScopeHashForPhase1,
    UnexpectedSplitDerivationHashForPhase1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkVerificationConfigValidationError {
    PublicMetadataMismatch,
    ColumnDegreeBoundsMismatch,
    LogupStatisticalSecurityBudgetHashMismatch,
    UnexpectedQuotientSplitMaskProfileForPublicOnly,
    MissingQuotientSplitMaskProfile,
    QuotientSplitMaskProfile(ZkQuotientSplitMaskProfileBindingError),
    InvalidLogupStatisticalSecurityBudget {
        aggregate_id: u32,
    },
    InsufficientLogupStatisticalSecurityBudget {
        aggregate_id: u32,
        computed_bits: u32,
        min_bits: u32,
    },
    UnexpectedColumnDegreeBoundsForPhase1,
    Metadata(ZkMetadataValidationError),
}

impl From<ZkMetadataValidationError> for ZkVerificationConfigValidationError {
    fn from(error: ZkMetadataValidationError) -> Self {
        Self::Metadata(error)
    }
}

#[must_use]
pub fn expected_zk_column_degree_bounds(metadata: &ZkPublicMetadata) -> Vec<ZkColumnDegreeBound> {
    let mut expected = metadata
        .witness_randomization
        .private_column_degree_bounds
        .clone();
    expected.extend(metadata.quotient_integration.quotient_degree_bounds.clone());
    expected
}

pub fn validate_zk_logup_statistical_security_budgets(
    budgets: &[ZkLogupStatisticalSecurityBudget],
) -> Result<(), ZkVerificationConfigValidationError> {
    let mut seen = BTreeSet::new();
    for budget in budgets {
        if !seen.insert(budget.aggregate_id)
            || budget.extension_field_bits != ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS
            || budget.private_lookup_term_count_bound == 0
            || budget.lookup_challenge_count == 0
            || budget.expected_proof_volume == 0
            || budget.min_statistical_security_bits == 0
        {
            return Err(
                ZkVerificationConfigValidationError::InvalidLogupStatisticalSecurityBudget {
                    aggregate_id: budget.aggregate_id,
                },
            );
        }
        let exposure = u128::from(budget.private_lookup_term_count_bound)
            .checked_mul(u128::from(budget.lookup_challenge_count))
            .and_then(|value| value.checked_mul(u128::from(budget.expected_proof_volume)))
            .ok_or(
                ZkVerificationConfigValidationError::InvalidLogupStatisticalSecurityBudget {
                    aggregate_id: budget.aggregate_id,
                },
            )?;
        let expected_bits = ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS
            .saturating_sub(ceil_log2_u128(exposure))
            .saturating_sub(budget.safety_margin_bits);
        if budget.computed_security_bits != expected_bits {
            return Err(
                ZkVerificationConfigValidationError::InvalidLogupStatisticalSecurityBudget {
                    aggregate_id: budget.aggregate_id,
                },
            );
        }
        if budget.computed_security_bits < budget.min_statistical_security_bits {
            return Err(
                ZkVerificationConfigValidationError::InsufficientLogupStatisticalSecurityBudget {
                    aggregate_id: budget.aggregate_id,
                    computed_bits: budget.computed_security_bits,
                    min_bits: budget.min_statistical_security_bits,
                },
            );
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkQuotientSplitMaskProfileBindingError {
    Profile(ZkQuotientSplitMaskProfileValidationError),
    MissingQuotientDegreeBounds,
    SplitMaskEntropyBelowFullDimension {
        required: u64,
        actual: u64,
    },
    UnexpectedQuotientSplitDegreeBounds {
        expected_range: ZkColumnRange,
        actual_range: Option<ZkColumnRange>,
        actual_count: usize,
    },
    QuotientSplitDegreeBoundOverflow {
        log_degree_bound: u32,
    },
    ProfileMismatch {
        expected: ZkQuotientSplitMaskProfile,
        actual: ZkQuotientSplitMaskProfile,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkQuotientSplitMaskQueryBudgetError {
    QueryCountTooLarge { n_queries: usize },
    InsufficientMaskDimension { required: u64, actual: u64 },
}

pub fn validate_zk_quotient_split_mask_query_budget(
    profile: ZkQuotientSplitMaskProfile,
    n_queries: usize,
) -> Result<(), ZkQuotientSplitMaskQueryBudgetError> {
    let n_queries_u64 = u64::try_from(n_queries)
        .map_err(|_| ZkQuotientSplitMaskQueryBudgetError::QueryCountTooLarge { n_queries })?;
    let required = n_queries_u64
        .checked_mul(2)
        .and_then(|query_openings| query_openings.checked_add(1))
        .ok_or(ZkQuotientSplitMaskQueryBudgetError::QueryCountTooLarge { n_queries })?;
    if profile.h_split < required {
        return Err(
            ZkQuotientSplitMaskQueryBudgetError::InsufficientMaskDimension {
                required,
                actual: profile.h_split,
            },
        );
    }

    Ok(())
}

pub fn validate_zk_quotient_split_mask_profile_for_metadata(
    metadata: &ZkPublicMetadata,
    actual: ZkQuotientSplitMaskProfile,
) -> Result<(), ZkQuotientSplitMaskProfileBindingError> {
    validate_zk_quotient_split_mask_profile(actual)
        .map_err(ZkQuotientSplitMaskProfileBindingError::Profile)?;
    let required_h_split = 1u64.checked_shl(actual.split_mask_log_degree_bound).ok_or(
        ZkQuotientSplitMaskProfileBindingError::Profile(
            ZkQuotientSplitMaskProfileValidationError::SplitMaskDimensionTooLarge {
                split_mask_log_degree_bound: actual.split_mask_log_degree_bound,
            },
        ),
    )?;
    if actual.h_split != required_h_split {
        return Err(
            ZkQuotientSplitMaskProfileBindingError::SplitMaskEntropyBelowFullDimension {
                required: required_h_split,
                actual: actual.h_split,
            },
        );
    }
    let quotient_degree_bounds = &metadata.quotient_integration.quotient_degree_bounds;
    let quotient_bound = quotient_degree_bounds
        .first()
        .copied()
        .ok_or(ZkQuotientSplitMaskProfileBindingError::MissingQuotientDegreeBounds)?;
    let expected_quotient_range = ZkColumnRange::new(
        quotient_bound.range.tree_index,
        0,
        2 * SECURE_EXTENSION_DEGREE,
    );
    if quotient_degree_bounds.len() != 1 || quotient_bound.range != expected_quotient_range {
        return Err(
            ZkQuotientSplitMaskProfileBindingError::UnexpectedQuotientSplitDegreeBounds {
                expected_range: expected_quotient_range,
                actual_range: Some(quotient_bound.range),
                actual_count: quotient_degree_bounds.len(),
            },
        );
    }

    let split_identity_log_degree_bound = quotient_bound.log_degree_bound.checked_add(1).ok_or(
        ZkQuotientSplitMaskProfileBindingError::QuotientSplitDegreeBoundOverflow {
            log_degree_bound: quotient_bound.log_degree_bound,
        },
    )?;
    let expected = ZkQuotientSplitMaskProfile {
        split_index: 0,
        split_identity_log_degree_bound,
        split_mask_log_degree_bound: quotient_bound.log_degree_bound,
        h_split: actual.h_split,
        left_range: ZkColumnRange::new(quotient_bound.range.tree_index, 0, SECURE_EXTENSION_DEGREE),
        right_range: ZkColumnRange::new(
            quotient_bound.range.tree_index,
            SECURE_EXTENSION_DEGREE,
            2 * SECURE_EXTENSION_DEGREE,
        ),
        left_masked_log_degree_bound: split_identity_log_degree_bound,
        right_masked_log_degree_bound: quotient_bound.log_degree_bound,
    };
    if actual != expected {
        return Err(ZkQuotientSplitMaskProfileBindingError::ProfileMismatch { expected, actual });
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkColumnDegreeBoundApplicationError {
    MissingTree {
        range: ZkColumnRange,
    },
    InvalidRange {
        range: ZkColumnRange,
    },
    RangeOutOfBounds {
        range: ZkColumnRange,
        column_count: usize,
    },
    DuplicateColumn {
        tree_index: usize,
        column_index: usize,
    },
    DegreeBoundShrinksColumn {
        range: ZkColumnRange,
        column_index: usize,
        base_log_degree_bound: u32,
        zk_log_degree_bound: u32,
    },
}

/// Applies verifier-owned ZK column degree bounds to a deterministic trace
/// degree profile.
///
/// ZK bounds may only increase an existing committed column degree bound, never
/// shrink it, and overlapping public ranges are rejected. This is a Phase 3
/// building block and does not by itself enable private-witness STARK ZK.
pub fn apply_zk_column_degree_bounds(
    mut base_bounds: TreeVec<ColumnVec<u32>>,
    zk_bounds: &[ZkColumnDegreeBound],
) -> Result<TreeVec<ColumnVec<u32>>, ZkColumnDegreeBoundApplicationError> {
    let mut seen_columns = BTreeSet::new();

    for bound in zk_bounds {
        let Some(tree_bounds) = base_bounds.get_mut(bound.range.tree_index) else {
            return Err(ZkColumnDegreeBoundApplicationError::MissingTree { range: bound.range });
        };
        if bound.range.column_start >= bound.range.column_end {
            return Err(ZkColumnDegreeBoundApplicationError::InvalidRange { range: bound.range });
        }
        if bound.range.column_start > tree_bounds.len() {
            return Err(ZkColumnDegreeBoundApplicationError::RangeOutOfBounds {
                range: bound.range,
                column_count: tree_bounds.len(),
            });
        }
        if bound.range.column_end > tree_bounds.len() {
            return Err(ZkColumnDegreeBoundApplicationError::RangeOutOfBounds {
                range: bound.range,
                column_count: tree_bounds.len(),
            });
        }
        for column_index in bound.range.column_start..bound.range.column_end {
            if !seen_columns.insert((bound.range.tree_index, column_index)) {
                return Err(ZkColumnDegreeBoundApplicationError::DuplicateColumn {
                    tree_index: bound.range.tree_index,
                    column_index,
                });
            }
            let base_log_degree_bound = tree_bounds[column_index];
            if bound.log_degree_bound < base_log_degree_bound {
                return Err(
                    ZkColumnDegreeBoundApplicationError::DegreeBoundShrinksColumn {
                        range: bound.range,
                        column_index,
                        base_log_degree_bound,
                        zk_log_degree_bound: bound.log_degree_bound,
                    },
                );
            }
            tree_bounds[column_index] = bound.log_degree_bound;
        }
    }

    Ok(base_bounds)
}

/// Returns the original trace-domain log size represented by component-owned
/// column degree bounds.
///
/// This intentionally does not inspect committed column sizes. In the
/// paper-shaped private witness path, committed private columns may be larger
/// because they contain `w_hat = w + v_H * r`, while the trace domain `H`
/// remains the original AIR trace domain.
#[must_use]
pub fn zk_trace_domain_log_size_from_column_bounds(
    column_log_degree_bounds: &TreeVec<ColumnVec<u32>>,
) -> Option<u32> {
    column_log_degree_bounds.iter().flatten().copied().max()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkCommittedColumnLogSizeValidationError {
    TreeCountMismatch {
        expected: usize,
        actual: usize,
    },
    ColumnCountMismatch {
        tree_index: usize,
        expected: usize,
        actual: usize,
    },
    LogSizeOverflow {
        tree_index: usize,
        column_index: usize,
        expected_log_degree_bound: u32,
        log_blowup_factor: u32,
    },
    LogSizeMismatch {
        tree_index: usize,
        column_index: usize,
        expected_log_size: u32,
        actual_log_size: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkCompositionColumnLogSizeValidationError {
    ColumnCountMismatch {
        expected: usize,
        actual: usize,
    },
    LogSizeOverflow {
        expected_log_degree_bound: u32,
        log_blowup_factor: u32,
    },
    LogSizeMismatch {
        column_index: usize,
        expected_log_size: u32,
        actual_log_size: u32,
    },
}

pub fn validate_zk_committed_column_log_sizes(
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    expected_log_degree_bounds: &TreeVec<ColumnVec<u32>>,
    log_blowup_factor: u32,
) -> Result<(), ZkCommittedColumnLogSizeValidationError> {
    if committed_column_log_sizes.len() != expected_log_degree_bounds.len() {
        return Err(ZkCommittedColumnLogSizeValidationError::TreeCountMismatch {
            expected: expected_log_degree_bounds.len(),
            actual: committed_column_log_sizes.len(),
        });
    }

    for (tree_index, (committed_tree, expected_tree)) in committed_column_log_sizes
        .iter()
        .zip(expected_log_degree_bounds.iter())
        .enumerate()
    {
        if committed_tree.len() != expected_tree.len() {
            return Err(
                ZkCommittedColumnLogSizeValidationError::ColumnCountMismatch {
                    tree_index,
                    expected: expected_tree.len(),
                    actual: committed_tree.len(),
                },
            );
        }
        for (column_index, (&actual_log_size, &expected_log_degree_bound)) in
            committed_tree.iter().zip(expected_tree).enumerate()
        {
            let expected_log_size = expected_log_degree_bound
                .checked_add(log_blowup_factor)
                .ok_or(ZkCommittedColumnLogSizeValidationError::LogSizeOverflow {
                    tree_index,
                    column_index,
                    expected_log_degree_bound,
                    log_blowup_factor,
                })?;
            if actual_log_size != expected_log_size {
                return Err(ZkCommittedColumnLogSizeValidationError::LogSizeMismatch {
                    tree_index,
                    column_index,
                    expected_log_size,
                    actual_log_size,
                });
            }
        }
    }

    Ok(())
}

pub fn validate_zk_composition_column_log_sizes(
    composition_column_log_sizes: &[u32],
    expected_column_count: usize,
    expected_log_degree_bound: u32,
    log_blowup_factor: u32,
) -> Result<(), ZkCompositionColumnLogSizeValidationError> {
    if composition_column_log_sizes.len() != expected_column_count {
        return Err(
            ZkCompositionColumnLogSizeValidationError::ColumnCountMismatch {
                expected: expected_column_count,
                actual: composition_column_log_sizes.len(),
            },
        );
    }

    let expected_log_size = expected_log_degree_bound
        .checked_add(log_blowup_factor)
        .ok_or(ZkCompositionColumnLogSizeValidationError::LogSizeOverflow {
            expected_log_degree_bound,
            log_blowup_factor,
        })?;
    for (column_index, &actual_log_size) in composition_column_log_sizes.iter().enumerate() {
        if actual_log_size != expected_log_size {
            return Err(ZkCompositionColumnLogSizeValidationError::LogSizeMismatch {
                column_index,
                expected_log_size,
                actual_log_size,
            });
        }
    }

    Ok(())
}

pub fn validate_zk_composition_column_log_sizes_against_bounds(
    composition_column_log_sizes: &[u32],
    expected_log_degree_bounds: &[u32],
    log_blowup_factor: u32,
) -> Result<(), ZkCompositionColumnLogSizeValidationError> {
    if composition_column_log_sizes.len() != expected_log_degree_bounds.len() {
        return Err(
            ZkCompositionColumnLogSizeValidationError::ColumnCountMismatch {
                expected: expected_log_degree_bounds.len(),
                actual: composition_column_log_sizes.len(),
            },
        );
    }

    for (column_index, (&actual_log_size, &expected_log_degree_bound)) in
        composition_column_log_sizes
            .iter()
            .zip(expected_log_degree_bounds)
            .enumerate()
    {
        let expected_log_size = expected_log_degree_bound
            .checked_add(log_blowup_factor)
            .ok_or(ZkCompositionColumnLogSizeValidationError::LogSizeOverflow {
                expected_log_degree_bound,
                log_blowup_factor,
            })?;
        if actual_log_size != expected_log_size {
            return Err(ZkCompositionColumnLogSizeValidationError::LogSizeMismatch {
                column_index,
                expected_log_size,
                actual_log_size,
            });
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkSampledValuesShapeValidationError {
    TreeCountMismatch {
        expected: usize,
        actual: usize,
    },
    ColumnCountMismatch {
        tree_index: usize,
        expected: usize,
        actual: usize,
    },
    SampleCountMismatch {
        tree_index: usize,
        column_index: usize,
        expected: usize,
        actual: usize,
    },
}

pub fn validate_zk_sampled_values_shape(
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    sampled_values: &TreeVec<ColumnVec<Vec<SecureField>>>,
) -> Result<(), ZkSampledValuesShapeValidationError> {
    if sampled_values.len() != sampled_points.len() {
        return Err(ZkSampledValuesShapeValidationError::TreeCountMismatch {
            expected: sampled_points.len(),
            actual: sampled_values.len(),
        });
    }

    for (tree_index, (point_tree, value_tree)) in
        sampled_points.iter().zip(sampled_values.iter()).enumerate()
    {
        if value_tree.len() != point_tree.len() {
            return Err(ZkSampledValuesShapeValidationError::ColumnCountMismatch {
                tree_index,
                expected: point_tree.len(),
                actual: value_tree.len(),
            });
        }
        for (column_index, (points, values)) in point_tree.iter().zip(value_tree).enumerate() {
            if values.len() != points.len() {
                return Err(ZkSampledValuesShapeValidationError::SampleCountMismatch {
                    tree_index,
                    column_index,
                    expected: points.len(),
                    actual: values.len(),
                });
            }
        }
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct ZkStarkDegreeBoundProfile {
    pub column_log_degree_bounds: TreeVec<ColumnVec<u32>>,
    pub trace_log_degree_bound: u32,
    pub composition_log_degree_bound: u32,
    pub split_composition_log_degree_bound: u32,
    pub split_composition_log_degree_bounds: ColumnVec<u32>,
    pub fri_first_layer_log_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkStarkDegreeBoundProfileError {
    VerificationConfig(ZkVerificationConfigValidationError),
    Metadata(ZkMetadataValidationError),
    ColumnDegreeBounds(ZkColumnDegreeBoundApplicationError),
    EmptyTraceDegreeProfile,
    CompositionSplitUnderflow {
        composition_log_degree_bound: u32,
        composition_log_split: u32,
    },
    CompositionDegreeOverflow {
        split_composition_log_degree_bound: u32,
        composition_log_split: u32,
    },
    QuotientColumnCountOverflow {
        composition_log_split: u32,
    },
    UnexpectedQuotientDegreeBoundRange {
        expected_range: ZkColumnRange,
        actual_range: ZkColumnRange,
    },
    NonCanonicalQuotientDegreeBounds {
        expected_range: ZkColumnRange,
        actual_count: usize,
    },
    SplitDerivationHashMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    QuotientDegreeBoundShrinksComposition {
        normal_split_composition_log_degree_bound: u32,
        zk_split_composition_log_degree_bound: u32,
    },
    FriFirstLayerTooSmall {
        required: u32,
        actual: u32,
    },
    FriFirstLayerBelowTraceDegree {
        required: u32,
        actual: u32,
    },
    FriFirstLayerBelowCommittedTraceSize {
        required: u32,
        actual: u32,
    },
    FriFirstLayerMismatch {
        degree_profile: u32,
        quotient_integration: u32,
    },
}

impl From<ZkColumnDegreeBoundApplicationError> for ZkStarkDegreeBoundProfileError {
    fn from(error: ZkColumnDegreeBoundApplicationError) -> Self {
        Self::ColumnDegreeBounds(error)
    }
}

impl From<ZkVerificationConfigValidationError> for ZkStarkDegreeBoundProfileError {
    fn from(error: ZkVerificationConfigValidationError) -> Self {
        Self::VerificationConfig(error)
    }
}

impl From<ZkMetadataValidationError> for ZkStarkDegreeBoundProfileError {
    fn from(error: ZkMetadataValidationError) -> Self {
        Self::Metadata(error)
    }
}

fn zk_metadata_has_private_witness_randomization(metadata: &ZkPublicMetadata) -> bool {
    metadata.degree_profile.h_witness != 0
        || metadata.witness_randomization.h_witness != 0
        || !metadata
            .witness_randomization
            .private_column_degree_bounds
            .is_empty()
        || metadata
            .witness_randomization
            .randomizer_space_hash
            .iter()
            .any(|&byte| byte != 0)
        || metadata
            .witness_randomization
            .private_column_scope_hash
            .iter()
            .any(|&byte| byte != 0)
}

/// Returns true when metadata declares any field that requires private STARK
/// witness or quotient integration activation.
#[must_use]
pub fn zk_metadata_requires_private_stark_activation(metadata: &ZkPublicMetadata) -> bool {
    zk_metadata_has_private_witness_randomization(metadata)
        || !metadata
            .quotient_integration
            .quotient_degree_bounds
            .is_empty()
        || metadata
            .quotient_integration
            .split_derivation_hash
            .iter()
            .any(|&byte| byte != 0)
}

fn validate_zk_quotient_degree_bounds_for_stark_profile(
    bounds: &[ZkColumnDegreeBound],
    expected_range: ZkColumnRange,
) -> Result<(), ZkStarkDegreeBoundProfileError> {
    validate_zk_column_degree_bound_ranges(bounds)?;
    if bounds.is_empty() {
        return Ok(());
    }
    if bounds.len() != 1 {
        return Err(
            ZkStarkDegreeBoundProfileError::NonCanonicalQuotientDegreeBounds {
                expected_range,
                actual_count: bounds.len(),
            },
        );
    }
    if bounds[0].range != expected_range {
        return Err(
            ZkStarkDegreeBoundProfileError::UnexpectedQuotientDegreeBoundRange {
                expected_range,
                actual_range: bounds[0].range,
            },
        );
    }

    Ok(())
}

fn zk_split_composition_log_degree_bounds_for_stark_profile(
    metadata: &ZkPublicMetadata,
    verifier_config: &ZkVerificationConfig,
    expected_range: ZkColumnRange,
    raw_split_composition_log_degree_bound: u32,
) -> Result<ColumnVec<u32>, ZkStarkDegreeBoundProfileError> {
    validate_zk_quotient_degree_bounds_for_stark_profile(
        &metadata.quotient_integration.quotient_degree_bounds,
        expected_range,
    )?;
    let mut split_composition_log_degree_bounds =
        vec![raw_split_composition_log_degree_bound; expected_range.column_end];

    if let Some(profile) = verifier_config.quotient_split_mask_profile {
        validate_zk_quotient_split_mask_profile_for_metadata(metadata, profile)
            .map_err(ZkVerificationConfigValidationError::QuotientSplitMaskProfile)?;
        if profile.left_range.tree_index != expected_range.tree_index
            || profile.right_range.tree_index != expected_range.tree_index
            || profile.left_range.column_start != expected_range.column_start
            || profile.right_range.column_end != expected_range.column_end
        {
            return Err(
                ZkStarkDegreeBoundProfileError::UnexpectedQuotientDegreeBoundRange {
                    expected_range,
                    actual_range: profile.left_range,
                },
            );
        }
        for column_index in profile.left_range.column_start..profile.left_range.column_end {
            split_composition_log_degree_bounds[column_index] =
                profile.left_masked_log_degree_bound;
        }
        for column_index in profile.right_range.column_start..profile.right_range.column_end {
            split_composition_log_degree_bounds[column_index] =
                profile.right_masked_log_degree_bound;
        }
    }

    Ok(split_composition_log_degree_bounds)
}

fn validate_zk_split_derivation_hash_for_stark_profile(
    metadata: &ZkPublicMetadata,
    composition_log_split: u32,
) -> Result<(), ZkStarkDegreeBoundProfileError> {
    if !zk_metadata_has_private_witness_randomization(metadata) {
        return Ok(());
    }

    let expected = canonical_zk_split_derivation_hash(composition_log_split);
    let actual = metadata.quotient_integration.split_derivation_hash;
    if actual != expected {
        return Err(
            ZkStarkDegreeBoundProfileError::SplitDerivationHashMismatch { expected, actual },
        );
    }

    Ok(())
}

fn zk_composition_split_column_count(
    composition_log_split: u32,
) -> Result<usize, ZkStarkDegreeBoundProfileError> {
    let split_count = 1usize.checked_shl(composition_log_split).ok_or(
        ZkStarkDegreeBoundProfileError::QuotientColumnCountOverflow {
            composition_log_split,
        },
    )?;
    split_count.checked_mul(SECURE_EXTENSION_DEGREE).ok_or(
        ZkStarkDegreeBoundProfileError::QuotientColumnCountOverflow {
            composition_log_split,
        },
    )
}

/// Derives the public ZK STARK degree profile used by Phase 3 wiring.
///
/// Public-only metadata returns the normal STARK composition split profile.
/// Private-witness metadata must carry verifier-owned quotient degree bounds;
/// those bounds are treated as the source of truth for the split composition
/// bound and may only increase the normal bound. This helper is deliberately
/// conservative and does not activate private-witness STARK ZK by itself.
pub fn derive_zk_stark_degree_bound_profile(
    base_column_log_degree_bounds: TreeVec<ColumnVec<u32>>,
    normal_composition_log_degree_bound: u32,
    composition_log_split: u32,
    verifier_config: &ZkVerificationConfig,
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Result<ZkStarkDegreeBoundProfile, ZkStarkDegreeBoundProfileError> {
    let metadata = &verifier_config.metadata;
    validate_zk_public_metadata_against_verifier_config(metadata, verifier_config)?;
    if zk_metadata_has_private_witness_randomization(metadata) {
        validate_zk_witness_metadata(metadata, lifting_log_size, log_blowup_factor)?;
    } else {
        validate_zk_public_only_metadata(metadata, lifting_log_size, log_blowup_factor)?;
    }

    let expected_quotient_range = ZkColumnRange::new(
        base_column_log_degree_bounds.0.len(),
        0,
        zk_composition_split_column_count(composition_log_split)?,
    );
    validate_zk_split_derivation_hash_for_stark_profile(metadata, composition_log_split)?;

    let column_log_degree_bounds = apply_zk_column_degree_bounds(
        base_column_log_degree_bounds,
        &metadata.witness_randomization.private_column_degree_bounds,
    )?;
    let trace_log_degree_bound = column_log_degree_bounds
        .iter()
        .flatten()
        .copied()
        .max()
        .ok_or(ZkStarkDegreeBoundProfileError::EmptyTraceDegreeProfile)?;
    let normal_split_composition_log_degree_bound = normal_composition_log_degree_bound
        .checked_sub(composition_log_split)
        .ok_or(ZkStarkDegreeBoundProfileError::CompositionSplitUnderflow {
            composition_log_degree_bound: normal_composition_log_degree_bound,
            composition_log_split,
        })?;

    let zk_split_composition_log_degree_bound = metadata
        .quotient_integration
        .quotient_degree_bounds
        .iter()
        .map(|bound| bound.log_degree_bound)
        .max()
        .unwrap_or(normal_split_composition_log_degree_bound);
    if zk_split_composition_log_degree_bound < normal_split_composition_log_degree_bound {
        return Err(
            ZkStarkDegreeBoundProfileError::QuotientDegreeBoundShrinksComposition {
                normal_split_composition_log_degree_bound,
                zk_split_composition_log_degree_bound,
            },
        );
    }
    let split_composition_log_degree_bounds =
        zk_split_composition_log_degree_bounds_for_stark_profile(
            metadata,
            verifier_config,
            expected_quotient_range,
            zk_split_composition_log_degree_bound,
        )?;
    let composition_log_degree_bound =
        if let Some(profile) = verifier_config.quotient_split_mask_profile {
            let profile_raw_split_bound = profile
                .split_identity_log_degree_bound
                .checked_sub(composition_log_split)
                .ok_or(ZkStarkDegreeBoundProfileError::CompositionSplitUnderflow {
                    composition_log_degree_bound: profile.split_identity_log_degree_bound,
                    composition_log_split,
                })?;
            if profile_raw_split_bound < normal_split_composition_log_degree_bound {
                return Err(
                    ZkStarkDegreeBoundProfileError::QuotientDegreeBoundShrinksComposition {
                        normal_split_composition_log_degree_bound,
                        zk_split_composition_log_degree_bound: profile_raw_split_bound,
                    },
                );
            }
            profile.split_identity_log_degree_bound
        } else {
            zk_split_composition_log_degree_bound
                .checked_add(composition_log_split)
                .ok_or(ZkStarkDegreeBoundProfileError::CompositionDegreeOverflow {
                    split_composition_log_degree_bound: zk_split_composition_log_degree_bound,
                    composition_log_split,
                })?
        };
    if metadata.degree_profile.fri_first_layer_log_size
        != metadata.quotient_integration.fri_first_layer_log_size
    {
        return Err(ZkStarkDegreeBoundProfileError::FriFirstLayerMismatch {
            degree_profile: metadata.degree_profile.fri_first_layer_log_size,
            quotient_integration: metadata.quotient_integration.fri_first_layer_log_size,
        });
    }
    let required_committed_trace_log_size =
        trace_log_degree_bound
            .checked_add(log_blowup_factor)
            .ok_or(ZkStarkDegreeBoundProfileError::FriFirstLayerTooSmall {
                required: u32::MAX,
                actual: metadata.degree_profile.fri_first_layer_log_size,
            })?;
    if metadata.degree_profile.fri_first_layer_log_size < required_committed_trace_log_size {
        return Err(
            ZkStarkDegreeBoundProfileError::FriFirstLayerBelowCommittedTraceSize {
                required: required_committed_trace_log_size,
                actual: metadata.degree_profile.fri_first_layer_log_size,
            },
        );
    }
    let max_committed_split_composition_log_degree_bound = split_composition_log_degree_bounds
        .iter()
        .copied()
        .max()
        .unwrap_or(zk_split_composition_log_degree_bound);
    let required_fri_first_layer_log_size = max_committed_split_composition_log_degree_bound
        .checked_add(log_blowup_factor)
        .ok_or(ZkStarkDegreeBoundProfileError::FriFirstLayerTooSmall {
            required: u32::MAX,
            actual: metadata.degree_profile.fri_first_layer_log_size,
        })?;
    if metadata.degree_profile.fri_first_layer_log_size < required_fri_first_layer_log_size {
        return Err(ZkStarkDegreeBoundProfileError::FriFirstLayerTooSmall {
            required: required_fri_first_layer_log_size,
            actual: metadata.degree_profile.fri_first_layer_log_size,
        });
    }

    Ok(ZkStarkDegreeBoundProfile {
        column_log_degree_bounds,
        trace_log_degree_bound,
        composition_log_degree_bound,
        split_composition_log_degree_bound: zk_split_composition_log_degree_bound,
        split_composition_log_degree_bounds,
        fri_first_layer_log_size: metadata.degree_profile.fri_first_layer_log_size,
    })
}

pub fn validate_zk_public_metadata_against_verifier_config(
    proof_metadata: &ZkPublicMetadata,
    verifier_config: &ZkVerificationConfig,
) -> Result<(), ZkVerificationConfigValidationError> {
    if proof_metadata != &verifier_config.metadata {
        return Err(ZkVerificationConfigValidationError::PublicMetadataMismatch);
    }
    if verifier_config.column_degree_bounds
        != expected_zk_column_degree_bounds(&verifier_config.metadata)
    {
        return Err(ZkVerificationConfigValidationError::ColumnDegreeBoundsMismatch);
    }
    validate_zk_logup_statistical_security_budgets(
        &verifier_config.logup_statistical_security_budgets,
    )?;
    if verifier_config
        .metadata
        .logup_statistical_security_budget_hash
        != canonical_zk_logup_statistical_security_budget_hash(
            &verifier_config.logup_statistical_security_budgets,
        )
    {
        return Err(
            ZkVerificationConfigValidationError::LogupStatisticalSecurityBudgetHashMismatch,
        );
    }
    let requires_quotient_split_mask_profile = !verifier_config
        .metadata
        .witness_randomization
        .private_column_degree_bounds
        .is_empty()
        && !verifier_config
            .metadata
            .quotient_integration
            .quotient_degree_bounds
            .is_empty();
    if requires_quotient_split_mask_profile {
        validate_zk_column_degree_bound_ranges(
            &verifier_config
                .metadata
                .quotient_integration
                .quotient_degree_bounds,
        )
        .map_err(ZkVerificationConfigValidationError::Metadata)?;
    }
    match (
        requires_quotient_split_mask_profile,
        verifier_config.quotient_split_mask_profile,
    ) {
        (false, None) => {}
        (false, Some(_)) => {
            return Err(
                ZkVerificationConfigValidationError::UnexpectedQuotientSplitMaskProfileForPublicOnly,
            );
        }
        (true, None) => {
            return Err(ZkVerificationConfigValidationError::MissingQuotientSplitMaskProfile);
        }
        (true, Some(profile)) => {
            validate_zk_quotient_split_mask_profile_for_metadata(&verifier_config.metadata, profile)
                .map_err(ZkVerificationConfigValidationError::QuotientSplitMaskProfile)?
        }
    }

    Ok(())
}

pub fn validate_zk_public_only_metadata_against_verifier_config(
    proof_metadata: &ZkPublicMetadata,
    verifier_config: &ZkVerificationConfig,
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Result<(), ZkVerificationConfigValidationError> {
    validate_zk_public_metadata_against_verifier_config(proof_metadata, verifier_config)?;
    validate_zk_public_only_metadata(
        &verifier_config.metadata,
        lifting_log_size,
        log_blowup_factor,
    )?;
    if !verifier_config.column_degree_bounds.is_empty() {
        return Err(ZkVerificationConfigValidationError::UnexpectedColumnDegreeBoundsForPhase1);
    }

    Ok(())
}

pub fn validate_zk_witness_randomization_audit_for_verifier(
    verifier_config: &ZkVerificationConfig,
    audit: Option<&ZkWitnessRandomizationVerifierAudit>,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
) -> Result<(), ZkWitnessRandomizationVerifierAuditError> {
    validate_zk_witness_randomization_audit_for_verifier_impl(
        verifier_config,
        audit,
        sampled_points,
        fri_query_positions,
        lifting_log_size,
        None,
    )
}

pub fn validate_zk_witness_randomization_audit_for_verifier_with_opening_geometry(
    verifier_config: &ZkVerificationConfig,
    audit: Option<&ZkWitnessRandomizationVerifierAudit>,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
    opening_geometry: &[ZkPrivateColumnOpeningGeometry],
    log_blowup_factor: u32,
) -> Result<(), ZkWitnessRandomizationVerifierAuditError> {
    validate_zk_witness_randomization_audit_for_verifier_impl(
        verifier_config,
        audit,
        sampled_points,
        fri_query_positions,
        lifting_log_size,
        Some((opening_geometry, log_blowup_factor)),
    )
}

fn validate_zk_witness_randomization_audit_for_verifier_impl(
    verifier_config: &ZkVerificationConfig,
    audit: Option<&ZkWitnessRandomizationVerifierAudit>,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
    opening_geometry: Option<(&[ZkPrivateColumnOpeningGeometry], u32)>,
) -> Result<(), ZkWitnessRandomizationVerifierAuditError> {
    let metadata = &verifier_config.metadata;
    if !zk_metadata_has_private_witness_randomization(metadata) {
        return if audit.is_some() {
            Err(ZkWitnessRandomizationVerifierAuditError::UnexpectedAuditForPublicOnly)
        } else {
            Ok(())
        };
    }

    let audit = audit.ok_or(ZkWitnessRandomizationVerifierAuditError::MissingAudit)?;
    if audit.privacy_map.version != metadata.version {
        return Err(ZkWitnessRandomizationVerifierAuditError::PrivacyMapVersionMismatch);
    }
    if audit.privacy_map.hash != metadata.privacy_map_hash {
        return Err(ZkWitnessRandomizationVerifierAuditError::PrivacyMapHashMismatch);
    }
    let mut expected_private_ranges = audit.privacy_map.private_columns.clone();
    expected_private_ranges.sort_unstable();
    let mut actual_private_ranges = metadata
        .witness_randomization
        .private_column_degree_bounds
        .iter()
        .map(|bound| bound.range)
        .collect::<Vec<_>>();
    actual_private_ranges.sort_unstable();
    if actual_private_ranges != expected_private_ranges {
        return Err(ZkWitnessRandomizationVerifierAuditError::PrivateColumnDegreeBoundsMismatch);
    }
    validate_zk_private_column_scope_for_witness_randomization(
        &audit.privacy_map,
        metadata.witness_randomization.private_column_scope_hash,
        &audit.private_column_scope,
    )
    .map_err(ZkWitnessRandomizationVerifierAuditError::PrivateColumnScope)?;
    let has_private_logup = audit
        .private_column_scope
        .entries
        .iter()
        .any(|entry| entry.usage == ZkPrivateColumnUsage::LogUp);
    if has_private_logup
        && (verifier_config
            .logup_statistical_security_budgets
            .is_empty()
            || metadata.logup_statistical_security_budget_hash
                == ZK_EMPTY_LOGUP_STATISTICAL_SECURITY_BUDGET_HASH)
    {
        return Err(
            ZkWitnessRandomizationVerifierAuditError::MissingPrivateLogupStatisticalSecurityBudget,
        );
    }

    let randomizer_space_entries = audit
        .private_column_scope
        .entries
        .iter()
        .map(|entry| {
            let trace_domain = CanonicCoset::try_new(entry.trace_domain_log_size)
                .map_err(
                    |_| ZkWitnessRandomizationVerifierAuditError::InvalidTraceDomainLogSize {
                        log_size: entry.trace_domain_log_size,
                    },
                )?
                .coset;
            let randomized_log_degree = metadata
                .witness_randomization
                .private_column_degree_bounds
                .iter()
                .find(|bound| bound.range == entry.range)
                .map(|bound| bound.log_degree_bound)
                .ok_or(
                    ZkWitnessRandomizationVerifierAuditError::PrivateColumnDegreeBoundsMismatch,
                )?;
            let randomizer_dimension = 1u64.checked_shl(entry.trace_domain_log_size).ok_or(
                ZkWitnessRandomizationVerifierAuditError::InvalidTraceDomainLogSize {
                    log_size: entry.trace_domain_log_size,
                },
            )?;
            let semantic_trace_domains = entry
                .semantic_trace_domain_log_sizes
                .iter()
                .map(|&log_size| {
                    CanonicCoset::try_new(log_size)
                        .map(|domain| {
                            ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(domain.coset))
                        })
                        .map_err(|_| {
                            ZkWitnessRandomizationVerifierAuditError::InvalidTraceDomainLogSize {
                                log_size,
                            }
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ZkRandomizerSpaceEntry {
                range: entry.range,
                trace_domain: ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(trace_domain)),
                semantic_trace_domains,
                randomized_log_degree,
                randomizer_dimension,
            })
        })
        .collect::<Result<Vec<_>, ZkWitnessRandomizationVerifierAuditError>>()?;
    if canonical_zk_randomizer_space_hash(
        audit.private_column_scope.hash,
        &randomizer_space_entries,
    ) != metadata.witness_randomization.randomizer_space_hash
    {
        return Err(ZkWitnessRandomizationVerifierAuditError::RandomizerSpaceHashMismatch);
    }
    let sampled_metadata = if let Some((opening_geometry, log_blowup_factor)) = opening_geometry {
        build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
            &audit.privacy_map,
            &audit.private_column_scope,
            &metadata.witness_randomization.private_column_degree_bounds,
            sampled_points,
            fri_query_positions,
            lifting_log_size,
            opening_geometry,
            log_blowup_factor,
        )
    } else {
        build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope(
            &audit.privacy_map,
            &audit.private_column_scope,
            sampled_points,
            fri_query_positions,
            lifting_log_size,
        )
    }
    .map_err(ZkWitnessRandomizationVerifierAuditError::SampleMetadata)?;
    validate_zk_query_closure_for_witness_randomization(
        &audit.privacy_map,
        metadata,
        &sampled_metadata.closure,
    )
    .map_err(ZkWitnessRandomizationVerifierAuditError::QueryClosure)?;
    validate_zk_randomizer_rank_profile_for_witness_randomization(
        &audit.privacy_map,
        metadata,
        &sampled_metadata.closure,
        &sampled_metadata.rank_profile,
    )
    .map_err(ZkWitnessRandomizationVerifierAuditError::RandomizerRank)
}

pub fn validate_zk_witness_metadata(
    metadata: &ZkPublicMetadata,
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Result<(), ZkMetadataValidationError> {
    if metadata.version != ZkProofVersion::V1 {
        return Err(ZkMetadataValidationError::UnsupportedProofVersion {
            actual: metadata.version.0,
        });
    }
    if metadata.degree_profile.fri_first_layer_log_size != lifting_log_size {
        return Err(ZkMetadataValidationError::FriFirstLayerLogSizeMismatch {
            expected: lifting_log_size,
            actual: metadata.degree_profile.fri_first_layer_log_size,
        });
    }
    if metadata.quotient_integration.fri_first_layer_log_size != lifting_log_size {
        return Err(
            ZkMetadataValidationError::QuotientFirstLayerLogSizeMismatch {
                expected: lifting_log_size,
                actual: metadata.quotient_integration.fri_first_layer_log_size,
            },
        );
    }
    let expected_h_batch = expected_zk_fri_batch_degree_bound(lifting_log_size, log_blowup_factor)
        .ok_or(ZkMetadataValidationError::InvalidFriBatchDegreeBound)?;
    if metadata.degree_profile.h_batch != expected_h_batch {
        return Err(ZkMetadataValidationError::FriBatchDegreeMismatch {
            expected: expected_h_batch,
            actual: metadata.degree_profile.h_batch,
        });
    }
    if metadata.quotient_integration.h_batch != expected_h_batch {
        return Err(ZkMetadataValidationError::QuotientFriBatchDegreeMismatch {
            expected: expected_h_batch,
            actual: metadata.quotient_integration.h_batch,
        });
    }
    if metadata.degree_profile.h_witness != metadata.witness_randomization.h_witness {
        return Err(
            ZkMetadataValidationError::WitnessRandomizationDegreeMismatch {
                expected: metadata.degree_profile.h_witness,
                actual: metadata.witness_randomization.h_witness,
            },
        );
    }
    if metadata.witness_randomization.h_witness == 0 {
        return Err(ZkMetadataValidationError::EmptyWitnessRandomizer);
    }
    if metadata
        .witness_randomization
        .private_column_degree_bounds
        .is_empty()
    {
        return Err(ZkMetadataValidationError::MissingPrivateColumnDegreeBounds);
    }
    if metadata
        .quotient_integration
        .quotient_degree_bounds
        .is_empty()
    {
        return Err(ZkMetadataValidationError::MissingQuotientDegreeBounds);
    }
    validate_zk_column_degree_bound_ranges(
        &metadata.witness_randomization.private_column_degree_bounds,
    )?;
    validate_zk_column_degree_bound_ranges(&metadata.quotient_integration.quotient_degree_bounds)?;
    if metadata
        .witness_randomization
        .randomizer_space_hash
        .iter()
        .all(|&byte| byte == 0)
    {
        return Err(ZkMetadataValidationError::EmptyRandomizerSpaceHash);
    }
    if metadata
        .witness_randomization
        .private_column_scope_hash
        .iter()
        .all(|&byte| byte == 0)
    {
        return Err(ZkMetadataValidationError::EmptyPrivateColumnScopeHash);
    }
    if metadata
        .quotient_integration
        .split_derivation_hash
        .iter()
        .all(|&byte| byte == 0)
    {
        return Err(ZkMetadataValidationError::EmptySplitDerivationHash);
    }
    for bound in &metadata.witness_randomization.private_column_degree_bounds {
        1u64.checked_shl(bound.log_degree_bound).ok_or(
            ZkMetadataValidationError::WitnessRandomizerDimensionTooLarge {
                dimension: metadata.witness_randomization.h_witness,
            },
        )?;
    }

    Ok(())
}

pub fn validate_zk_column_degree_bound_ranges(
    bounds: &[ZkColumnDegreeBound],
) -> Result<(), ZkMetadataValidationError> {
    for bound in bounds {
        if bound.range.column_start >= bound.range.column_end {
            return Err(ZkMetadataValidationError::InvalidColumnDegreeBoundRange {
                range: bound.range,
            });
        }
    }

    Ok(())
}

#[must_use]
pub fn expected_zk_fri_batch_degree_bound(
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Option<u64> {
    lifting_log_size
        .checked_sub(log_blowup_factor)
        .and_then(|log_degree_bound| 1u64.checked_shl(log_degree_bound))
}

pub fn validate_zk_public_only_metadata(
    metadata: &ZkPublicMetadata,
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Result<(), ZkMetadataValidationError> {
    if metadata.version != ZkProofVersion::V1 {
        return Err(ZkMetadataValidationError::UnsupportedProofVersion {
            actual: metadata.version.0,
        });
    }

    let expected_h_batch = expected_zk_fri_batch_degree_bound(lifting_log_size, log_blowup_factor)
        .ok_or(ZkMetadataValidationError::InvalidFriBatchDegreeBound)?;

    if metadata.degree_profile.fri_first_layer_log_size != lifting_log_size {
        return Err(ZkMetadataValidationError::FriFirstLayerLogSizeMismatch {
            expected: lifting_log_size,
            actual: metadata.degree_profile.fri_first_layer_log_size,
        });
    }
    if metadata.quotient_integration.fri_first_layer_log_size != lifting_log_size {
        return Err(
            ZkMetadataValidationError::QuotientFirstLayerLogSizeMismatch {
                expected: lifting_log_size,
                actual: metadata.quotient_integration.fri_first_layer_log_size,
            },
        );
    }
    if metadata.degree_profile.h_batch != expected_h_batch {
        return Err(ZkMetadataValidationError::FriBatchDegreeMismatch {
            expected: expected_h_batch,
            actual: metadata.degree_profile.h_batch,
        });
    }
    if metadata.quotient_integration.h_batch != expected_h_batch {
        return Err(ZkMetadataValidationError::QuotientFriBatchDegreeMismatch {
            expected: expected_h_batch,
            actual: metadata.quotient_integration.h_batch,
        });
    }
    if metadata.degree_profile.h_witness != 0 || metadata.witness_randomization.h_witness != 0 {
        return Err(ZkMetadataValidationError::UnexpectedWitnessRandomizationForPhase1);
    }
    if !metadata
        .witness_randomization
        .private_column_degree_bounds
        .is_empty()
    {
        return Err(ZkMetadataValidationError::UnexpectedPrivateColumnDegreeBoundsForPhase1);
    }
    if !metadata
        .quotient_integration
        .quotient_degree_bounds
        .is_empty()
    {
        return Err(ZkMetadataValidationError::UnexpectedQuotientDegreeBoundsForPhase1);
    }
    if metadata
        .witness_randomization
        .randomizer_space_hash
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(ZkMetadataValidationError::UnexpectedRandomizerSpaceHashForPhase1);
    }
    if metadata
        .witness_randomization
        .private_column_scope_hash
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(ZkMetadataValidationError::UnexpectedPrivateColumnScopeHashForPhase1);
    }
    if metadata
        .quotient_integration
        .split_derivation_hash
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(ZkMetadataValidationError::UnexpectedSplitDerivationHashForPhase1);
    }

    Ok(())
}

const ZK_PUBLIC_METADATA_TRANSCRIPT_DOMAIN: u32 = 0x5a4b_0001;
const ZK_COLUMN_BOUNDS_TRANSCRIPT_DOMAIN: u32 = 0x5a4b_0002;
const ZK_QUOTIENT_SPLIT_MASK_PROFILE_TRANSCRIPT_DOMAIN: u32 = 0x5a4b_0003;

pub fn mix_zk_public_metadata<C: Channel>(
    channel: &mut C,
    metadata: &ZkPublicMetadata,
    column_degree_bounds: &[ZkColumnDegreeBound],
) {
    channel.mix_u32s(&[
        ZK_PUBLIC_METADATA_TRANSCRIPT_DOMAIN,
        metadata.version.0,
        metadata.degree_profile.trace_domain_log_size,
        metadata.degree_profile.fri_first_layer_log_size,
    ]);
    channel.mix_u64(metadata.degree_profile.h_witness);
    channel.mix_u64(metadata.degree_profile.h_batch);
    mix_hash_bytes(channel, &metadata.privacy_map_hash.0);
    mix_hash_bytes(channel, &metadata.public_statement_hash.0);
    mix_hash_bytes(channel, &metadata.logup_statistical_security_budget_hash);

    channel.mix_u64(metadata.witness_randomization.h_witness);
    mix_hash_bytes(
        channel,
        &metadata.witness_randomization.randomizer_space_hash,
    );
    mix_hash_bytes(
        channel,
        &metadata.witness_randomization.private_column_scope_hash,
    );
    mix_column_degree_bounds(
        channel,
        &metadata.witness_randomization.private_column_degree_bounds,
    );

    channel.mix_u64(metadata.quotient_integration.h_batch);
    channel.mix_u32s(&[metadata.quotient_integration.fri_first_layer_log_size]);
    mix_hash_bytes(
        channel,
        &metadata.quotient_integration.split_derivation_hash,
    );
    mix_column_degree_bounds(
        channel,
        &metadata.quotient_integration.quotient_degree_bounds,
    );

    mix_column_degree_bounds(channel, column_degree_bounds);
}

pub fn mix_zk_quotient_split_mask_profile<C: Channel>(
    channel: &mut C,
    profile: ZkQuotientSplitMaskProfile,
) -> Result<(), ZkQuotientSplitMaskProfileValidationError> {
    validate_zk_quotient_split_mask_profile(profile)?;

    channel.mix_u32s(&[
        ZK_QUOTIENT_SPLIT_MASK_PROFILE_TRANSCRIPT_DOMAIN,
        profile.split_index,
        profile.split_identity_log_degree_bound,
        profile.split_mask_log_degree_bound,
        profile.left_masked_log_degree_bound,
        profile.right_masked_log_degree_bound,
    ]);
    channel.mix_u64(profile.h_split);
    mix_hash_bytes(
        channel,
        &canonical_zk_quotient_split_mask_profile_hash(profile),
    );
    mix_column_range(channel, profile.left_range);
    mix_column_range(channel, profile.right_range);

    Ok(())
}

fn mix_hash_bytes<C: Channel>(channel: &mut C, bytes: &[u8; 32]) {
    let words: [u32; 8] = core::array::from_fn(|index| {
        let offset = index * 4;
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    });
    channel.mix_u32s(&words);
}

fn mix_column_degree_bounds<C: Channel>(channel: &mut C, bounds: &[ZkColumnDegreeBound]) {
    channel.mix_u32s(&[ZK_COLUMN_BOUNDS_TRANSCRIPT_DOMAIN]);
    channel.mix_u64(bounds.len() as u64);
    for bound in bounds {
        mix_column_range(channel, bound.range);
        channel.mix_u32s(&[bound.log_degree_bound]);
    }
}

fn mix_column_range<C: Channel>(channel: &mut C, range: ZkColumnRange) {
    channel.mix_u64(range.tree_index as u64);
    channel.mix_u64(range.column_start as u64);
    channel.mix_u64(range.column_end as u64);
}

/// Public OODS exclusion policy for the ZK path.
///
/// Every forbidden coset is interpreted through its STWO circle vanishing
/// polynomial. This is intentionally verifier-side data: prover and verifier
/// must derive the same accept/reject decision from the Fiat-Shamir point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkOodsExclusionSet {
    pub forbidden_cosets: Vec<Coset>,
    pub reject_line_degeneracy: bool,
}

impl ZkOodsExclusionSet {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            forbidden_cosets: Vec::new(),
            reject_line_degeneracy: true,
        }
    }

    #[must_use]
    pub fn accepts(&self, point: CirclePoint<SecureField>) -> bool {
        if self.reject_line_degeneracy && point.y == point.y.complex_conjugate() {
            return false;
        }

        self.forbidden_cosets
            .iter()
            .all(|&coset| !coset_vanishing(coset, point).is_zero())
    }
}

fn push_full_circle_domain_cosets(forbidden_cosets: &mut Vec<Coset>, half_coset: Coset) {
    forbidden_cosets.push(half_coset);
    forbidden_cosets.push(half_coset.conjugate());
}

/// Returns the public OODS exclusion set shared by the explicit ZK prover and
/// verifier paths.
///
/// This covers the trace domain and first FRI layer commitment domain. More
/// specialized translated/query-domain exclusions remain separate degree-gated
/// work; both prover and verifier must use this helper so the Fiat-Shamir
/// rejection rule cannot diverge.
#[must_use]
pub fn zk_oods_exclusion_set(
    trace_domain_log_size: u32,
    fri_first_layer_log_size: u32,
) -> Result<ZkOodsExclusionSet, crate::core::poly::circle::InvalidCanonicCosetLogSize> {
    let mut forbidden_cosets = Vec::new();
    push_full_circle_domain_cosets(
        &mut forbidden_cosets,
        CanonicCoset::try_new(trace_domain_log_size)?.coset,
    );
    push_full_circle_domain_cosets(
        &mut forbidden_cosets,
        CanonicCoset::try_new(fri_first_layer_log_size)?.coset,
    );

    Ok(ZkOodsExclusionSet {
        forbidden_cosets,
        reject_line_degeneracy: true,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkOodsSamplingError {
    ExhaustedAttempts { attempts: usize },
}

/// Draws an OODS point from the Fiat-Shamir channel with deterministic public
/// rejection. This must be used by both prover and verifier in the ZK path.
pub fn draw_zk_oods_point<C: Channel>(
    channel: &mut C,
    exclusion_set: &ZkOodsExclusionSet,
    max_attempts: usize,
) -> Result<CirclePoint<SecureField>, ZkOodsSamplingError> {
    for _ in 0..max_attempts {
        let point = CirclePoint::<SecureField>::get_random_point(channel);
        if exclusion_set.accepts(point) {
            return Ok(point);
        }
    }

    Err(ZkOodsSamplingError::ExhaustedAttempts {
        attempts: max_attempts,
    })
}

/// Draws a deterministic ZK OODS point together with a public preimage under
/// circle doubling.
///
/// The active private composition split commits the masked right side at one
/// lower degree than the masked left side. PCS sampling folds lower-degree
/// columns one extra time, so right-side composition columns are requested at a
/// known preimage to make their opened evaluation point match the left side.
/// Both the semantic OODS point and the preimage are rejected against the
/// public exclusion set so prover and verifier consume the Fiat-Shamir channel
/// identically.
pub fn draw_zk_oods_point_with_preimage<C: Channel>(
    channel: &mut C,
    exclusion_set: &ZkOodsExclusionSet,
    max_attempts: usize,
) -> Result<(CirclePoint<SecureField>, CirclePoint<SecureField>), ZkOodsSamplingError> {
    for _ in 0..max_attempts {
        let preimage = CirclePoint::<SecureField>::get_random_point(channel);
        let point = preimage.double();
        if exclusion_set.accepts(preimage) && exclusion_set.accepts(point) {
            return Ok((point, preimage));
        }
    }

    Err(ZkOodsSamplingError::ExhaustedAttempts {
        attempts: max_attempts,
    })
}

#[derive(Clone, Debug)]
pub struct ZkOodsSamplePointPlan {
    pub semantic_oods_point: CirclePoint<SecureField>,
    pub semantic_sample_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    pub pcs_sample_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    pub max_delta: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkOodsSamplePointPlanError {
    TreeCountMismatch {
        expected: usize,
        actual: usize,
    },
    ColumnCountMismatch {
        tree_index: usize,
        expected: usize,
        actual: usize,
    },
    ColumnLogSizeAboveLifting {
        tree_index: usize,
        column_index: usize,
        column_log_size: u32,
        lifting_log_size: u32,
    },
    SamplePointLiftDomainOverflow {
        max_log_degree_bound: u32,
        delta: u32,
    },
    InvalidSamplePointLiftDomain {
        log_size: u32,
    },
    SamplePointLiftDeltaUnderflow {
        max_delta: u32,
        delta: u32,
    },
    ForbiddenDeepestPoint,
    ForbiddenSemanticOodsPoint,
    ForbiddenSemanticPoint {
        tree_index: usize,
        column_index: usize,
        sample_index: usize,
    },
    ForbiddenPcsPoint {
        tree_index: usize,
        column_index: usize,
        sample_index: usize,
    },
    LiftRelationMismatch {
        tree_index: usize,
        column_index: usize,
        sample_index: usize,
    },
}

impl ZkOodsSamplePointPlanError {
    #[must_use]
    pub const fn is_sampling_rejection(self) -> bool {
        matches!(
            self,
            Self::ForbiddenDeepestPoint
                | Self::ForbiddenSemanticOodsPoint
                | Self::ForbiddenSemanticPoint { .. }
                | Self::ForbiddenPcsPoint { .. }
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkOodsSamplePointPlanningError {
    Sampling(ZkOodsSamplingError),
    Plan(ZkOodsSamplePointPlanError),
}

/// Derives semantic AIR sample points and PCS preimage sample points for mixed
/// committed column log sizes in the private-witness ZK path.
///
/// The channel supplies a deepest public point `u`. For each committed column
/// with `delta_i = lifting_log_size - committed_log_size_i`, the semantic AIR
/// point is built over `z = double^max_delta(u)`, while the PCS point is built
/// over `double^(max_delta - delta_i)(u)` and the deeper canonical step. This
/// enforces `double^delta_i(pcs_point_i) == semantic_point_i`.
pub fn plan_zk_oods_sample_points(
    deepest_point: CirclePoint<SecureField>,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    max_log_degree_bound: u32,
    exclusion_set: &ZkOodsExclusionSet,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanError> {
    let semantic_step_log_sizes = TreeVec(
        mask_offsets
            .iter()
            .map(|tree| vec![max_log_degree_bound; tree.len()])
            .collect(),
    );
    plan_zk_oods_sample_points_with_semantic_step_log_sizes(
        deepest_point,
        mask_offsets,
        &semantic_step_log_sizes,
        committed_column_log_sizes,
        lifting_log_size,
        exclusion_set,
    )
}

/// Derives semantic AIR sample points and PCS preimage sample points using an
/// explicit semantic offset-step domain for each column.
///
/// For a column with committed log size `c`, `delta = lifting_log_size - c`.
/// The returned PCS point `p` satisfies `double^delta(p) == semantic_point`.
/// The semantic point is translated from the semantic OODS point by the
/// column's own AIR offset-step domain; this is required for AIRs whose
/// constraint degree expansion is greater than one.
pub fn plan_zk_oods_sample_points_with_semantic_step_log_sizes(
    deepest_point: CirclePoint<SecureField>,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    semantic_step_log_sizes: &TreeVec<ColumnVec<u32>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    exclusion_set: &ZkOodsExclusionSet,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanError> {
    let semantic_base_doublings = TreeVec(
        mask_offsets
            .iter()
            .map(|tree| vec![0; tree.len()])
            .collect(),
    );
    plan_zk_oods_sample_points_with_semantic_domains(
        deepest_point,
        mask_offsets,
        semantic_step_log_sizes,
        &semantic_base_doublings,
        committed_column_log_sizes,
        lifting_log_size,
        exclusion_set,
    )
}

pub fn plan_zk_oods_sample_points_with_semantic_domains(
    deepest_point: CirclePoint<SecureField>,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    semantic_step_log_sizes: &TreeVec<ColumnVec<u32>>,
    semantic_base_doublings: &TreeVec<ColumnVec<u32>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    exclusion_set: &ZkOodsExclusionSet,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanError> {
    let semantic_sample_step_log_sizes = TreeVec(
        mask_offsets
            .iter()
            .zip(semantic_step_log_sizes.iter())
            .map(|(offset_tree, step_tree)| {
                offset_tree
                    .iter()
                    .zip(step_tree.iter())
                    .map(|(column_offsets, &step_log_size)| {
                        vec![step_log_size; column_offsets.len()]
                    })
                    .collect()
            })
            .collect(),
    );
    let semantic_sample_base_doublings = TreeVec(
        mask_offsets
            .iter()
            .zip(semantic_base_doublings.iter())
            .map(|(offset_tree, base_tree)| {
                offset_tree
                    .iter()
                    .zip(base_tree.iter())
                    .map(|(column_offsets, &base_doubling)| {
                        vec![base_doubling; column_offsets.len()]
                    })
                    .collect()
            })
            .collect(),
    );

    plan_zk_oods_sample_points_with_semantic_sample_domains(
        deepest_point,
        mask_offsets,
        &semantic_sample_step_log_sizes,
        &semantic_sample_base_doublings,
        committed_column_log_sizes,
        lifting_log_size,
        exclusion_set,
    )
}

pub fn plan_zk_oods_sample_points_with_semantic_sample_domains(
    deepest_point: CirclePoint<SecureField>,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    semantic_sample_step_log_sizes: &TreeVec<ColumnVec<Vec<u32>>>,
    semantic_sample_base_doublings: &TreeVec<ColumnVec<Vec<u32>>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    exclusion_set: &ZkOodsExclusionSet,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanError> {
    if mask_offsets.len() != committed_column_log_sizes.len() {
        return Err(ZkOodsSamplePointPlanError::TreeCountMismatch {
            expected: committed_column_log_sizes.len(),
            actual: mask_offsets.len(),
        });
    }
    if mask_offsets.len() != semantic_sample_step_log_sizes.len() {
        return Err(ZkOodsSamplePointPlanError::TreeCountMismatch {
            expected: semantic_sample_step_log_sizes.len(),
            actual: mask_offsets.len(),
        });
    }
    if mask_offsets.len() != semantic_sample_base_doublings.len() {
        return Err(ZkOodsSamplePointPlanError::TreeCountMismatch {
            expected: semantic_sample_base_doublings.len(),
            actual: mask_offsets.len(),
        });
    }
    CanonicCoset::try_new(lifting_log_size).map_err(|_| {
        ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain {
            log_size: lifting_log_size,
        }
    })?;

    let mut max_delta = 0;
    let mut column_deltas = TreeVec(Vec::with_capacity(mask_offsets.len()));
    for (tree_index, (((offset_tree, semantic_step_tree), semantic_base_tree), log_size_tree)) in
        mask_offsets
            .iter()
            .zip(semantic_sample_step_log_sizes.iter())
            .zip(semantic_sample_base_doublings.iter())
            .zip(committed_column_log_sizes.iter())
            .enumerate()
    {
        if offset_tree.len() != log_size_tree.len() {
            return Err(ZkOodsSamplePointPlanError::ColumnCountMismatch {
                tree_index,
                expected: log_size_tree.len(),
                actual: offset_tree.len(),
            });
        }
        if offset_tree.len() != semantic_step_tree.len() {
            return Err(ZkOodsSamplePointPlanError::ColumnCountMismatch {
                tree_index,
                expected: semantic_step_tree.len(),
                actual: offset_tree.len(),
            });
        }
        if offset_tree.len() != semantic_base_tree.len() {
            return Err(ZkOodsSamplePointPlanError::ColumnCountMismatch {
                tree_index,
                expected: semantic_base_tree.len(),
                actual: offset_tree.len(),
            });
        }

        let mut tree_deltas = Vec::with_capacity(offset_tree.len());
        for (
            column_index,
            (((column_offsets, column_step_log_sizes), column_base_doublings), &column_log_size),
        ) in offset_tree
            .iter()
            .zip(semantic_step_tree.iter())
            .zip(semantic_base_tree.iter())
            .zip(log_size_tree.iter())
            .enumerate()
        {
            if column_offsets.len() != column_step_log_sizes.len() {
                return Err(ZkOodsSamplePointPlanError::ColumnCountMismatch {
                    tree_index,
                    expected: column_offsets.len(),
                    actual: column_step_log_sizes.len(),
                });
            }
            if column_offsets.len() != column_base_doublings.len() {
                return Err(ZkOodsSamplePointPlanError::ColumnCountMismatch {
                    tree_index,
                    expected: column_offsets.len(),
                    actual: column_base_doublings.len(),
                });
            }
            let delta = lifting_log_size.checked_sub(column_log_size).ok_or(
                ZkOodsSamplePointPlanError::ColumnLogSizeAboveLifting {
                    tree_index,
                    column_index,
                    column_log_size,
                    lifting_log_size,
                },
            )?;
            // A large lift delta is not rejected by a fixed protocol constant.
            // Algebraic validity is checked dynamically below by ensuring that
            // all derived cosets exist, all generated points pass the exclusion
            // policy, and every PCS point doubles back to its semantic point.
            // Larger deltas reduce the challenge support seen after repeated
            // doubling, so callers must include this geometry in the soundness
            // budget rather than hiding it behind an example-specific cap.
            for (&semantic_step_log_size, &semantic_base_doubling) in column_step_log_sizes
                .iter()
                .zip(column_base_doublings.iter())
            {
                CanonicCoset::try_new(semantic_step_log_size).map_err(|_| {
                    ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain {
                        log_size: semantic_step_log_size,
                    }
                })?;
                let pcs_step_log_size = semantic_step_log_size.checked_add(delta).ok_or(
                    ZkOodsSamplePointPlanError::SamplePointLiftDomainOverflow {
                        max_log_degree_bound: semantic_step_log_size,
                        delta,
                    },
                )?;
                CanonicCoset::try_new(pcs_step_log_size).map_err(|_| {
                    ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain {
                        log_size: pcs_step_log_size,
                    }
                })?;
                max_delta = max_delta.max(delta.saturating_sub(semantic_base_doubling));
            }
            tree_deltas.push(delta);
        }
        column_deltas.push(tree_deltas);
    }

    if max_delta > 0 {
        CanonicCoset::try_new(max_delta).map_err(|_| {
            ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain {
                log_size: max_delta,
            }
        })?;
    }
    for ((offset_tree, semantic_base_tree), delta_tree) in mask_offsets
        .iter()
        .zip(semantic_sample_base_doublings.iter())
        .zip(column_deltas.iter())
    {
        for ((_, column_base_doublings), &delta) in offset_tree
            .iter()
            .zip(semantic_base_tree.iter())
            .zip(delta_tree.iter())
        {
            for &semantic_base_doubling in column_base_doublings {
                let semantic_base_doubles = max_delta.checked_add(semantic_base_doubling).ok_or(
                    ZkOodsSamplePointPlanError::SamplePointLiftDomainOverflow {
                        max_log_degree_bound: semantic_base_doubling,
                        delta: max_delta,
                    },
                )?;
                if semantic_base_doubles > lifting_log_size {
                    return Err(ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain {
                        log_size: semantic_base_doubles,
                    });
                }
                if semantic_base_doubles < delta {
                    return Err(ZkOodsSamplePointPlanError::SamplePointLiftDeltaUnderflow {
                        max_delta: semantic_base_doubles,
                        delta,
                    });
                }
            }
        }
    }

    if !exclusion_set.accepts(deepest_point) {
        return Err(ZkOodsSamplePointPlanError::ForbiddenDeepestPoint);
    }
    let semantic_oods_point = deepest_point.repeated_double(max_delta);
    if !exclusion_set.accepts(semantic_oods_point) {
        return Err(ZkOodsSamplePointPlanError::ForbiddenSemanticOodsPoint);
    }

    let mut semantic_sample_points = TreeVec(Vec::with_capacity(mask_offsets.len()));
    let mut pcs_sample_points = TreeVec(Vec::with_capacity(mask_offsets.len()));

    for (tree_index, (((offset_tree, semantic_step_tree), semantic_base_tree), delta_tree)) in
        mask_offsets
            .iter()
            .zip(semantic_sample_step_log_sizes.iter())
            .zip(semantic_sample_base_doublings.iter())
            .zip(column_deltas.iter())
            .enumerate()
    {
        let mut semantic_tree = Vec::with_capacity(offset_tree.len());
        let mut pcs_tree = Vec::with_capacity(offset_tree.len());
        for (
            column_index,
            (((column_offsets, column_step_log_sizes), column_base_doublings), &delta),
        ) in offset_tree
            .iter()
            .zip(semantic_step_tree.iter())
            .zip(semantic_base_tree.iter())
            .zip(delta_tree.iter())
            .enumerate()
        {
            if column_offsets.is_empty() {
                semantic_tree.push(Vec::new());
                pcs_tree.push(Vec::new());
                continue;
            }

            let mut semantic_column = Vec::with_capacity(column_offsets.len());
            let mut pcs_column = Vec::with_capacity(column_offsets.len());
            for (sample_index, ((&offset, &semantic_step_log_size), &semantic_base_doubling)) in
                column_offsets
                    .iter()
                    .zip(column_step_log_sizes.iter())
                    .zip(column_base_doublings.iter())
                    .enumerate()
            {
                let semantic_base_doubles = max_delta.checked_add(semantic_base_doubling).ok_or(
                    ZkOodsSamplePointPlanError::SamplePointLiftDomainOverflow {
                        max_log_degree_bound: semantic_base_doubling,
                        delta: max_delta,
                    },
                )?;
                let pcs_base_doubles = semantic_base_doubles.checked_sub(delta).ok_or(
                    ZkOodsSamplePointPlanError::SamplePointLiftDeltaUnderflow {
                        max_delta: semantic_base_doubles,
                        delta,
                    },
                )?;
                let semantic_base = deepest_point.repeated_double(semantic_base_doubles);
                let pcs_base = deepest_point.repeated_double(pcs_base_doubles);
                let semantic_step = CanonicCoset::try_new(semantic_step_log_size)
                    .map_err(
                        |_| ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain {
                            log_size: semantic_step_log_size,
                        },
                    )?
                    .step();
                let pcs_step_log_size = semantic_step_log_size.checked_add(delta).ok_or(
                    ZkOodsSamplePointPlanError::SamplePointLiftDomainOverflow {
                        max_log_degree_bound: semantic_step_log_size,
                        delta,
                    },
                )?;
                let pcs_step = CanonicCoset::try_new(pcs_step_log_size)
                    .map_err(
                        |_| ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain {
                            log_size: pcs_step_log_size,
                        },
                    )?
                    .step();
                let semantic_point = semantic_base + semantic_step.mul_signed(offset).into_ef();
                let pcs_point = pcs_base + pcs_step.mul_signed(offset).into_ef();

                if !exclusion_set.accepts(semantic_point) {
                    return Err(ZkOodsSamplePointPlanError::ForbiddenSemanticPoint {
                        tree_index,
                        column_index,
                        sample_index,
                    });
                }
                if !exclusion_set.accepts(pcs_point) {
                    return Err(ZkOodsSamplePointPlanError::ForbiddenPcsPoint {
                        tree_index,
                        column_index,
                        sample_index,
                    });
                }
                if pcs_point.repeated_double(delta) != semantic_point {
                    return Err(ZkOodsSamplePointPlanError::LiftRelationMismatch {
                        tree_index,
                        column_index,
                        sample_index,
                    });
                }

                semantic_column.push(semantic_point);
                pcs_column.push(pcs_point);
            }
            semantic_tree.push(semantic_column);
            pcs_tree.push(pcs_column);
        }
        semantic_sample_points.push(semantic_tree);
        pcs_sample_points.push(pcs_tree);
    }

    Ok(ZkOodsSamplePointPlan {
        semantic_oods_point,
        semantic_sample_points,
        pcs_sample_points,
        max_delta,
    })
}

pub fn draw_zk_oods_sample_point_plan<C: Channel>(
    channel: &mut C,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    max_log_degree_bound: u32,
    exclusion_set: &ZkOodsExclusionSet,
    max_attempts: usize,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanningError> {
    for _ in 0..max_attempts {
        let deepest_point = CirclePoint::<SecureField>::get_random_point(channel);
        match plan_zk_oods_sample_points(
            deepest_point,
            mask_offsets,
            committed_column_log_sizes,
            lifting_log_size,
            max_log_degree_bound,
            exclusion_set,
        ) {
            Ok(plan) => return Ok(plan),
            Err(error) if error.is_sampling_rejection() => {}
            Err(error) => return Err(ZkOodsSamplePointPlanningError::Plan(error)),
        }
    }

    Err(ZkOodsSamplePointPlanningError::Sampling(
        ZkOodsSamplingError::ExhaustedAttempts {
            attempts: max_attempts,
        },
    ))
}

pub fn draw_zk_oods_sample_point_plan_with_semantic_step_log_sizes<C: Channel>(
    channel: &mut C,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    semantic_step_log_sizes: &TreeVec<ColumnVec<u32>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    exclusion_set: &ZkOodsExclusionSet,
    max_attempts: usize,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanningError> {
    let semantic_base_doublings = TreeVec(
        mask_offsets
            .iter()
            .map(|tree| vec![0; tree.len()])
            .collect(),
    );
    draw_zk_oods_sample_point_plan_with_semantic_domains(
        channel,
        mask_offsets,
        semantic_step_log_sizes,
        &semantic_base_doublings,
        committed_column_log_sizes,
        lifting_log_size,
        exclusion_set,
        max_attempts,
    )
}

pub fn draw_zk_oods_sample_point_plan_with_semantic_domains<C: Channel>(
    channel: &mut C,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    semantic_step_log_sizes: &TreeVec<ColumnVec<u32>>,
    semantic_base_doublings: &TreeVec<ColumnVec<u32>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    exclusion_set: &ZkOodsExclusionSet,
    max_attempts: usize,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanningError> {
    for _ in 0..max_attempts {
        let deepest_point = CirclePoint::<SecureField>::get_random_point(channel);
        match plan_zk_oods_sample_points_with_semantic_domains(
            deepest_point,
            mask_offsets,
            semantic_step_log_sizes,
            semantic_base_doublings,
            committed_column_log_sizes,
            lifting_log_size,
            exclusion_set,
        ) {
            Ok(plan) => return Ok(plan),
            Err(error) if error.is_sampling_rejection() => {}
            Err(error) => return Err(ZkOodsSamplePointPlanningError::Plan(error)),
        }
    }

    Err(ZkOodsSamplePointPlanningError::Sampling(
        ZkOodsSamplingError::ExhaustedAttempts {
            attempts: max_attempts,
        },
    ))
}

pub fn draw_zk_oods_sample_point_plan_with_semantic_sample_domains<C: Channel>(
    channel: &mut C,
    mask_offsets: &TreeVec<ColumnVec<Vec<isize>>>,
    semantic_sample_step_log_sizes: &TreeVec<ColumnVec<Vec<u32>>>,
    semantic_sample_base_doublings: &TreeVec<ColumnVec<Vec<u32>>>,
    committed_column_log_sizes: &TreeVec<ColumnVec<u32>>,
    lifting_log_size: u32,
    exclusion_set: &ZkOodsExclusionSet,
    max_attempts: usize,
) -> Result<ZkOodsSamplePointPlan, ZkOodsSamplePointPlanningError> {
    for _ in 0..max_attempts {
        let deepest_point = CirclePoint::<SecureField>::get_random_point(channel);
        match plan_zk_oods_sample_points_with_semantic_sample_domains(
            deepest_point,
            mask_offsets,
            semantic_sample_step_log_sizes,
            semantic_sample_base_doublings,
            committed_column_log_sizes,
            lifting_log_size,
            exclusion_set,
        ) {
            Ok(plan) => return Ok(plan),
            Err(error) if error.is_sampling_rejection() => {}
            Err(error) => return Err(ZkOodsSamplePointPlanningError::Plan(error)),
        }
    }

    Err(ZkOodsSamplePointPlanningError::Sampling(
        ZkOodsSamplingError::ExhaustedAttempts {
            attempts: max_attempts,
        },
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkOodsSamplePointValidationError {
    ForbiddenSamplePoint {
        tree_index: usize,
        column_index: usize,
        sample_index: usize,
    },
}

/// Validates every generated ZK sample point against the public OODS exclusion
/// policy. This extends deterministic rejection from the sampled OODS point to
/// translated/shifted component mask points derived from it.
pub fn validate_zk_sample_points_outside_exclusion_set(
    sample_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    exclusion_set: &ZkOodsExclusionSet,
) -> Result<(), ZkOodsSamplePointValidationError> {
    for (tree_index, tree_points) in sample_points.iter().enumerate() {
        for (column_index, column_points) in tree_points.iter().enumerate() {
            for (sample_index, &point) in column_points.iter().enumerate() {
                if !exclusion_set.accepts(point) {
                    return Err(ZkOodsSamplePointValidationError::ForbiddenSamplePoint {
                        tree_index,
                        column_index,
                        sample_index,
                    });
                }
            }
        }
    }

    Ok(())
}

/// Canonical public encoding of `R(query)` values.
///
/// Each entry is one `SecureField` value encoded as four base-field
/// coordinates in the canonical order returned by `SecureField::to_m31_array`.
/// Entries are ordered by deduplicated FRI query position order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkFriBatchMaskQueryValues {
    pub queries: Vec<[BaseField; 4]>,
}

impl ZkFriBatchMaskQueryValues {
    #[must_use]
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    #[must_use]
    pub fn to_secure_values(&self) -> Vec<SecureField> {
        self.queries
            .iter()
            .map(|coordinates| SecureField::from_m31_array(*coordinates))
            .collect()
    }

    #[must_use]
    pub fn into_merkle_query_columns(self) -> Vec<Vec<BaseField>> {
        let mut columns = vec![Vec::with_capacity(self.queries.len()); 4];
        for coordinates in self.queries {
            for (column, value) in columns.iter_mut().zip(coordinates) {
                column.push(value);
            }
        }
        columns
    }
}

impl SizeEstimate for ZkFriBatchMaskQueryValues {
    fn size_estimate(&self) -> usize {
        self.queries.len() * mem::size_of::<[BaseField; 4]>()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZkFriBatchMaskVerificationError {
    QueryCountMismatch { expected: usize, actual: usize },
    DomainLogSizeMismatch { expected: u32, actual: u32 },
    QueryDomainTooSmall { log_size: u32 },
    QueryDomainTooLarge { log_size: u32 },
    QueryPositionOutOfDomain { position: usize, domain_size: usize },
    Merkle(MerkleVerificationError),
}

impl From<MerkleVerificationError> for ZkFriBatchMaskVerificationError {
    fn from(error: MerkleVerificationError) -> Self {
        Self::Merkle(error)
    }
}

impl<H: MerkleHasherLifted> SizeEstimate for ZkFriBatchMaskProof<H> {
    fn size_estimate(&self) -> usize {
        self.commitment.size_estimate()
            + mem::size_of_val(&self.log_size)
            + self.fri_proof.size_estimate()
            + self.decommitment.size_estimate()
            + self.queried_values.size_estimate()
            + self.fri_queried_values.size_estimate()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkFriBatchMaskQueryPositionsError {
    InvalidFriConfig,
    QueryDomainTooLarge { log_size: u32 },
    QueryPositionOutOfDomain { position: usize, domain_size: usize },
    InsufficientPositions { required: usize, available: usize },
}

/// Separate public oracle proof for the Protocol 2 FRI batch mask polynomial
/// `R`.
///
/// This ZK-only proof format is still guarded by explicit metadata validation.
/// Adding `fri_proof` and `fri_queried_values` is an intentional incompatible
/// V1 branch change: older masked proofs must fail closed rather than verify
/// without the low-degree proof for `R`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkFriBatchMaskProof<H: MerkleHasherLifted> {
    pub commitment: H::Hash,
    pub log_size: u32,
    pub fri_proof: FriProof<H>,
    pub decommitment: MerkleDecommitmentLifted<H>,
    pub queried_values: ZkFriBatchMaskQueryValues,
    pub fri_queried_values: ZkFriBatchMaskQueryValues,
}

impl<H: MerkleHasherLifted> ZkFriBatchMaskProof<H> {
    pub fn verify_openings(
        self,
        query_positions: &[usize],
        expected_log_size: u32,
    ) -> Result<Vec<SecureField>, ZkFriBatchMaskVerificationError> {
        if self.log_size != expected_log_size {
            return Err(ZkFriBatchMaskVerificationError::DomainLogSizeMismatch {
                expected: expected_log_size,
                actual: self.log_size,
            });
        }
        if self.log_size == 0 && !query_positions.is_empty() {
            return Err(ZkFriBatchMaskVerificationError::QueryDomainTooSmall {
                log_size: self.log_size,
            });
        }
        let domain_size = 1usize.checked_shl(self.log_size).ok_or(
            ZkFriBatchMaskVerificationError::QueryDomainTooLarge {
                log_size: self.log_size,
            },
        )?;
        if let Some(&position) = query_positions
            .iter()
            .find(|&&position| position >= domain_size)
        {
            return Err(ZkFriBatchMaskVerificationError::QueryPositionOutOfDomain {
                position,
                domain_size,
            });
        }
        if self.queried_values.len() != query_positions.len() {
            return Err(ZkFriBatchMaskVerificationError::QueryCountMismatch {
                expected: query_positions.len(),
                actual: self.queried_values.len(),
            });
        }

        let secure_values = self.queried_values.to_secure_values();
        let verifier =
            MerkleVerifierLifted::new(self.commitment, vec![self.log_size; 4], Some(self.log_size));
        verifier.verify(
            query_positions,
            self.queried_values.into_merkle_query_columns(),
            self.decommitment,
        )?;

        Ok(secure_values)
    }
}

#[must_use]
pub fn zk_fri_batch_mask_fri_config(mut config: FriConfig) -> FriConfig {
    config.fold_step = 1;
    config
}

#[must_use]
pub fn zk_fri_batch_mask_query_positions<C: Channel>(
    channel: &mut C,
    log_domain_size: u32,
    n_queries: usize,
    h_batch_query_positions: &[usize],
    h_batch_fri_config: FriConfig,
) -> Result<Vec<usize>, ZkFriBatchMaskQueryPositionsError> {
    if log_domain_size >= usize::BITS {
        return Err(ZkFriBatchMaskQueryPositionsError::QueryDomainTooLarge {
            log_size: log_domain_size,
        });
    }

    let domain_size = 1usize << log_domain_size;
    if n_queries == 0 {
        return Ok(Vec::new());
    }
    if log_domain_size == 0 {
        return Err(ZkFriBatchMaskQueryPositionsError::InsufficientPositions {
            required: n_queries,
            available: 0,
        });
    }

    let Some(mut current_log_degree) =
        log_domain_size.checked_sub(h_batch_fri_config.log_blowup_factor)
    else {
        return Err(ZkFriBatchMaskQueryPositionsError::InvalidFriConfig);
    };
    if h_batch_fri_config.fold_step == 0
        || current_log_degree <= h_batch_fri_config.log_last_layer_degree_bound
    {
        return Err(ZkFriBatchMaskQueryPositionsError::InvalidFriConfig);
    }

    let query_mask = domain_size - 1;
    let mut h_batch_positions = BTreeSet::new();
    for &position in h_batch_query_positions {
        if position >= domain_size {
            return Err(
                ZkFriBatchMaskQueryPositionsError::QueryPositionOutOfDomain {
                    position,
                    domain_size,
                },
            );
        }
        h_batch_positions.insert(position);
    }

    let mut forbidden_pair_ranges = Vec::new();
    let mut layer_queries: Vec<usize> = h_batch_positions.into_iter().collect();
    let mut layer_log_size = log_domain_size;
    let mut cumulative_folds = 0;
    let max_raw_openings = n_queries.saturating_mul(2);

    while current_log_degree > h_batch_fri_config.log_last_layer_degree_bound {
        let fold_step = if cumulative_folds == 0 {
            h_batch_fri_config.fold_step
        } else {
            (current_log_degree - h_batch_fri_config.log_last_layer_degree_bound)
                .min(h_batch_fri_config.fold_step)
        };

        if fold_step == 0
            || fold_step > layer_log_size
            || fold_step > current_log_degree
            || current_log_degree - fold_step < h_batch_fri_config.log_last_layer_degree_bound
        {
            return Err(ZkFriBatchMaskQueryPositionsError::InvalidFriConfig);
        }

        let fold_size = 1usize << fold_step;
        let original_block_size = 1usize << cumulative_folds;
        for &position in &layer_queries {
            let closure_start = (position >> fold_step) << fold_step;
            for layer_position in closure_start..closure_start + fold_size {
                let original_start = layer_position << cumulative_folds;
                if cumulative_folds == 0 {
                    forbid_pair_containing_position(&mut forbidden_pair_ranges, original_start);
                } else if original_block_size <= max_raw_openings {
                    forbid_pairs_inside_block(
                        &mut forbidden_pair_ranges,
                        original_start,
                        original_block_size,
                    );
                }
            }
        }

        layer_queries = layer_queries
            .into_iter()
            .map(|query| query >> fold_step)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        layer_log_size -= fold_step;
        current_log_degree -= fold_step;
        cumulative_folds += fold_step;
    }

    let last_layer_block_size = 1usize << cumulative_folds;
    if last_layer_block_size <= max_raw_openings {
        // The H-batch last-layer polynomial is public over the whole last-layer
        // domain, not only on the sampled query path. If a final folded block is
        // small enough to be reconstructed from the R raw openings, any R pair
        // inside such a block could leak the corresponding folded raw quotient.
        return Err(ZkFriBatchMaskQueryPositionsError::InsufficientPositions {
            required: n_queries,
            available: 0,
        });
    }

    let forbidden_pair_ranges = merge_pair_ranges(forbidden_pair_ranges);
    let pair_count = domain_size >> 1;
    let forbidden_pairs = forbidden_pair_ranges
        .iter()
        .map(|(start, end)| end - start)
        .sum::<usize>();
    let available = pair_count.saturating_sub(forbidden_pairs) * 2;
    if available < n_queries {
        return Err(ZkFriBatchMaskQueryPositionsError::InsufficientPositions {
            required: n_queries,
            available,
        });
    }

    let mut positions = BTreeSet::new();
    while positions.len() < n_queries {
        for word in channel.draw_u32s() {
            let position = (word as usize) & query_mask;
            if pair_is_forbidden(&forbidden_pair_ranges, position >> 1) {
                continue;
            }
            positions.insert(position);
            if positions.len() == n_queries {
                break;
            }
        }
    }

    Ok(positions.into_iter().collect())
}

fn forbid_pair_containing_position(ranges: &mut Vec<(usize, usize)>, position: usize) {
    let pair_index = position >> 1;
    ranges.push((pair_index, pair_index + 1));
}

fn forbid_pairs_inside_block(ranges: &mut Vec<(usize, usize)>, start: usize, len: usize) {
    if len == 0 {
        return;
    }
    ranges.push((start >> 1, (start + len) >> 1));
}

fn merge_pair_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges {
        if start == end {
            continue;
        }
        match merged.last_mut() {
            Some((_, last_end)) if start <= *last_end => {
                *last_end = (*last_end).max(end);
            }
            _ => merged.push((start, end)),
        }
    }
    merged
}

fn pair_is_forbidden(ranges: &[(usize, usize)], pair_index: usize) -> bool {
    ranges
        .iter()
        .any(|(start, end)| *start <= pair_index && pair_index < *end)
}

/// ZK PCS proof. The embedded randomized PCS proof is incomplete without
/// `fri_batch_mask`; this type intentionally does not implement `Deref` or
/// conversion into the original proof type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkCommitmentSchemeProof<H: MerkleHasherLifted> {
    pub version: ZkProofVersion,
    pub randomized_pcs_proof: CommitmentSchemeProof<H>,
    pub fri_batch_mask: ZkFriBatchMaskProof<H>,
    pub public_metadata: ZkPublicMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkCommitmentSchemeProofAux<H: MerkleHasherLifted> {
    pub randomized_pcs_aux: CommitmentSchemeProofAux<H>,
    pub fri_batch_mask_fri_aux: FriProofAux<H>,
    pub fri_batch_mask_decommitment_aux: MerkleDecommitmentLiftedAux<H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedZkCommitmentSchemeProof<H: MerkleHasherLifted> {
    pub proof: ZkCommitmentSchemeProof<H>,
    pub aux: ZkCommitmentSchemeProofAux<H>,
}

fn zk_column_range_size_estimate() -> usize {
    3 * mem::size_of::<usize>()
}

fn zk_column_degree_bounds_size_estimate(bounds: &[ZkColumnDegreeBound]) -> usize {
    bounds.len() * (zk_column_range_size_estimate() + mem::size_of::<u32>())
}

fn zk_public_metadata_size_estimate(metadata: &ZkPublicMetadata) -> usize {
    mem::size_of_val(&metadata.version)
        + mem::size_of_val(&metadata.privacy_map_hash)
        + mem::size_of_val(&metadata.public_statement_hash)
        + mem::size_of_val(&metadata.logup_statistical_security_budget_hash)
        + mem::size_of_val(&metadata.degree_profile)
        + mem::size_of_val(&metadata.witness_randomization.h_witness)
        + mem::size_of_val(&metadata.witness_randomization.randomizer_space_hash)
        + mem::size_of_val(&metadata.witness_randomization.private_column_scope_hash)
        + zk_column_degree_bounds_size_estimate(
            &metadata.witness_randomization.private_column_degree_bounds,
        )
        + mem::size_of_val(&metadata.quotient_integration.h_batch)
        + mem::size_of_val(&metadata.quotient_integration.fri_first_layer_log_size)
        + mem::size_of_val(&metadata.quotient_integration.split_derivation_hash)
        + zk_column_degree_bounds_size_estimate(
            &metadata.quotient_integration.quotient_degree_bounds,
        )
}

impl<H: MerkleHasherLifted> SizeEstimate for ZkCommitmentSchemeProof<H> {
    fn size_estimate(&self) -> usize {
        mem::size_of_val(&self.version)
            + self.randomized_pcs_proof.size_estimate()
            + self.fri_batch_mask.size_estimate()
            + zk_public_metadata_size_estimate(&self.public_metadata)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkStarkProof<H: MerkleHasherLifted>(pub ZkCommitmentSchemeProof<H>);

impl<H: MerkleHasherLifted> ZkStarkProof<H> {
    /// Returns the estimate size (in bytes) of the ZK proof.
    pub fn size_estimate(&self) -> usize {
        self.0.size_estimate()
    }

    /// Extracts the randomized composition trace Out-Of-Domain-Sample
    /// evaluation from the ZK PCS sampled values.
    pub(crate) fn extract_composition_oods_eval(
        &self,
        oods_point: CirclePoint<SecureField>,
        composition_log_degree_bound: u32,
    ) -> Option<SecureField> {
        let [.., left_and_right_composition_mask] = &self.0.randomized_pcs_proof.sampled_values[..]
        else {
            return None;
        };
        let left_and_right_coordinate_evals: [SecureField; 2 * SECURE_EXTENSION_DEGREE] =
            left_and_right_composition_mask
                .iter()
                .map(|columns| {
                    let &[eval] = &columns[..] else {
                        return None;
                    };
                    Some(eval)
                })
                .collect::<Option<Vec<_>>>()?
                .try_into()
                .ok()?;

        let (left_coordinate_evals, right_coordinate_evals) =
            left_and_right_coordinate_evals.split_at(SECURE_EXTENSION_DEGREE);

        let left_eval = SecureField::from_partial_evals(left_coordinate_evals.try_into().ok()?);
        let right_eval = SecureField::from_partial_evals(right_coordinate_evals.try_into().ok()?);
        let split_factor_log = composition_log_degree_bound.checked_sub(2)?;
        let value = left_eval + oods_point.repeated_double(split_factor_log).x * right_eval;
        Some(value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedZkStarkProof<H: MerkleHasherLifted> {
    pub proof: ZkStarkProof<H>,
    pub aux: ZkCommitmentSchemeProofAux<H>,
}

/// Non-overlapping metric identifiers required by the ZK integration plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkPerformanceMetricId {
    WitnessRandomizerGeneration,
    VanishingRandomizerConstruction,
    PrivateCommitmentEvaluation,
    PrivateCommitmentTreeConstruction,
    CompositionConstraintEvaluation,
    CompositionQuotientSplit,
    OodsValueProduction,
    FriBatchMaskSampling,
    FriBatchMaskEvaluation,
    FriBatchMaskCommitment,
    FriBatchMaskOpening,
    FriBatchMaskVerification,
    FriBatchMaskAnswerAddition,
    RawFriQuotientComputation,
    FriBatchMaskLayerAddition,
    FriCommit,
    FriFold,
    FriQuerySampling,
    FriProverDecommitment,
    FriVerifierDecommitment,
    ProofBytes,
    PeakMemoryBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkPerformanceMetricUnit {
    Nanoseconds,
    Bytes,
    Count,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPerformanceMetric {
    pub id: ZkPerformanceMetricId,
    pub value: u128,
    pub unit: ZkPerformanceMetricUnit,
}

impl ZkPerformanceMetric {
    #[must_use]
    pub const fn new(
        id: ZkPerformanceMetricId,
        value: u128,
        unit: ZkPerformanceMetricUnit,
    ) -> Self {
        Self { id, value, unit }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::channel::Blake2sChannel;
    use crate::core::poly::circle::CanonicCoset;

    fn zero_hash() -> [u8; 32] {
        [0; 32]
    }

    fn nonzero_hash() -> [u8; 32] {
        [7; 32]
    }

    fn public_only_metadata(lifting_log_size: u32, log_blowup_factor: u32) -> ZkPublicMetadata {
        let h_batch =
            expected_zk_fri_batch_degree_bound(lifting_log_size, log_blowup_factor).unwrap();

        ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash: ZkPrivacyMapHash(nonzero_hash()),
            public_statement_hash: ZkPublicStatementHash(nonzero_hash()),
            logup_statistical_security_budget_hash: [0x5a; 32],
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: lifting_log_size - log_blowup_factor,
                h_witness: 0,
                h_batch,
                fri_first_layer_log_size: lifting_log_size,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness: 0,
                randomizer_space_hash: zero_hash(),
                private_column_scope_hash: zero_hash(),
                private_column_degree_bounds: Vec::new(),
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch,
                fri_first_layer_log_size: lifting_log_size,
                split_derivation_hash: zero_hash(),
                quotient_degree_bounds: Vec::new(),
            },
        }
    }

    fn verification_config(metadata: ZkPublicMetadata) -> ZkVerificationConfig {
        let column_degree_bounds = expected_zk_column_degree_bounds(&metadata);
        let quotient_split_mask_profile = if metadata
            .witness_randomization
            .private_column_degree_bounds
            .is_empty()
            || metadata
                .quotient_integration
                .quotient_degree_bounds
                .is_empty()
        {
            None
        } else {
            metadata
                .quotient_integration
                .quotient_degree_bounds
                .first()
                .and_then(|bound| {
                    stwo_composition_quotient_split_mask_profile(
                        bound.range.tree_index,
                        bound.log_degree_bound + 1,
                        bound.log_degree_bound,
                        1u64 << bound.log_degree_bound,
                        bound.log_degree_bound + 1,
                        bound.log_degree_bound,
                    )
                    .ok()
                })
        };
        ZkVerificationConfig {
            metadata,
            column_degree_bounds,
            quotient_split_mask_profile,
            logup_statistical_security_budgets: Vec::new(),
        }
    }

    #[test]
    fn public_only_metadata_accepts_public_only_profile() {
        let metadata = public_only_metadata(16, 1);
        assert_eq!(validate_zk_public_only_metadata(&metadata, 16, 1), Ok(()));
    }

    #[test]
    fn public_only_metadata_rejects_unsupported_version() {
        let mut metadata = public_only_metadata(16, 1);
        metadata.version = ZkProofVersion(2);

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::UnsupportedProofVersion { actual: 2 })
        );
    }

    #[test]
    fn public_only_metadata_rejects_wrong_first_layer_log_size() {
        let mut metadata = public_only_metadata(16, 1);
        metadata.degree_profile.fri_first_layer_log_size = 15;

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::FriFirstLayerLogSizeMismatch {
                expected: 16,
                actual: 15,
            })
        );
    }

    #[test]
    fn public_only_metadata_rejects_wrong_h_batch() {
        let mut metadata = public_only_metadata(16, 1);
        metadata.degree_profile.h_batch += 1;

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::FriBatchDegreeMismatch {
                expected: 1 << 15,
                actual: (1 << 15) + 1,
            })
        );
    }

    #[test]
    fn public_only_metadata_rejects_witness_randomization_fields() {
        let mut metadata = public_only_metadata(16, 1);
        metadata.witness_randomization.h_witness = 1;

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::UnexpectedWitnessRandomizationForPhase1)
        );

        let mut metadata = public_only_metadata(16, 1);
        metadata.witness_randomization.randomizer_space_hash = nonzero_hash();

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::UnexpectedRandomizerSpaceHashForPhase1)
        );

        let mut metadata = public_only_metadata(16, 1);
        metadata.witness_randomization.private_column_scope_hash = nonzero_hash();

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::UnexpectedPrivateColumnScopeHashForPhase1)
        );
    }

    #[test]
    fn public_only_metadata_rejects_private_and_quotient_degree_bounds() {
        let degree_bound = ZkColumnDegreeBound {
            range: ZkColumnRange::new(0, 0, 1),
            log_degree_bound: 15,
        };

        let mut metadata = public_only_metadata(16, 1);
        metadata
            .witness_randomization
            .private_column_degree_bounds
            .push(degree_bound);

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::UnexpectedPrivateColumnDegreeBoundsForPhase1)
        );

        let mut metadata = public_only_metadata(16, 1);
        metadata
            .quotient_integration
            .quotient_degree_bounds
            .push(degree_bound);

        assert_eq!(
            validate_zk_public_only_metadata(&metadata, 16, 1),
            Err(ZkMetadataValidationError::UnexpectedQuotientDegreeBoundsForPhase1)
        );
    }

    #[test]
    fn metadata_private_witness_detection_covers_all_private_fields() {
        assert!(!zk_metadata_has_private_witness_randomization(
            &public_only_metadata(16, 1)
        ));

        let mut metadata = public_only_metadata(16, 1);
        metadata.degree_profile.h_witness = 1;
        assert!(zk_metadata_has_private_witness_randomization(&metadata));

        let mut metadata = public_only_metadata(16, 1);
        metadata.witness_randomization.h_witness = 1;
        assert!(zk_metadata_has_private_witness_randomization(&metadata));

        let mut metadata = public_only_metadata(16, 1);
        metadata
            .witness_randomization
            .private_column_degree_bounds
            .push(ZkColumnDegreeBound {
                range: ZkColumnRange::new(0, 0, 1),
                log_degree_bound: 15,
            });
        assert!(zk_metadata_has_private_witness_randomization(&metadata));

        let mut metadata = public_only_metadata(16, 1);
        metadata.witness_randomization.randomizer_space_hash = nonzero_hash();
        assert!(zk_metadata_has_private_witness_randomization(&metadata));

        let mut metadata = public_only_metadata(16, 1);
        metadata.witness_randomization.private_column_scope_hash = nonzero_hash();
        assert!(zk_metadata_has_private_witness_randomization(&metadata));
    }

    #[test]
    fn metadata_private_stark_activation_detection_covers_private_fields() {
        assert!(!zk_metadata_requires_private_stark_activation(
            &public_only_metadata(16, 1)
        ));

        let mut metadata = public_only_metadata(16, 1);
        metadata.witness_randomization.h_witness = 1;
        assert!(zk_metadata_requires_private_stark_activation(&metadata));

        let mut metadata = public_only_metadata(16, 1);
        metadata
            .quotient_integration
            .quotient_degree_bounds
            .push(ZkColumnDegreeBound {
                range: ZkColumnRange::new(0, 0, 1),
                log_degree_bound: 15,
            });
        assert!(zk_metadata_requires_private_stark_activation(&metadata));

        let mut metadata = public_only_metadata(16, 1);
        metadata.quotient_integration.split_derivation_hash = nonzero_hash();
        assert!(zk_metadata_requires_private_stark_activation(&metadata));
    }

    #[test]
    fn trace_domain_log_size_is_derived_from_base_column_bounds() {
        let base_column_bounds = TreeVec(vec![vec![3], vec![5, 4]]);

        assert_eq!(
            zk_trace_domain_log_size_from_column_bounds(&base_column_bounds),
            Some(5)
        );
        assert_eq!(
            zk_trace_domain_log_size_from_column_bounds(&TreeVec(vec![vec![], vec![]])),
            None
        );
    }

    fn witness_metadata(lifting_log_size: u32, log_blowup_factor: u32) -> ZkPublicMetadata {
        let h_batch =
            expected_zk_fri_batch_degree_bound(lifting_log_size, log_blowup_factor).unwrap();

        ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash: ZkPrivacyMapHash(nonzero_hash()),
            public_statement_hash: ZkPublicStatementHash(nonzero_hash()),
            logup_statistical_security_budget_hash: [0x5a; 32],
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: lifting_log_size - log_blowup_factor,
                h_witness: 1 << (lifting_log_size - log_blowup_factor),
                h_batch,
                fri_first_layer_log_size: lifting_log_size,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness: 1 << (lifting_log_size - log_blowup_factor),
                randomizer_space_hash: nonzero_hash(),
                private_column_scope_hash: nonzero_hash(),
                private_column_degree_bounds: vec![ZkColumnDegreeBound {
                    range: ZkColumnRange::new(1, 0, 1),
                    log_degree_bound: lifting_log_size,
                }],
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch,
                fri_first_layer_log_size: lifting_log_size,
                split_derivation_hash: canonical_zk_split_derivation_hash(1),
                quotient_degree_bounds: vec![ZkColumnDegreeBound {
                    range: ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE),
                    log_degree_bound: lifting_log_size - log_blowup_factor,
                }],
            },
        }
    }

    fn set_witness_trace_domain(metadata: &mut ZkPublicMetadata, trace_domain_log_size: u32) {
        let h_witness = 1u64 << trace_domain_log_size;
        metadata.degree_profile.trace_domain_log_size = trace_domain_log_size;
        metadata.degree_profile.h_witness = h_witness;
        metadata.witness_randomization.h_witness = h_witness;
    }

    #[test]
    fn witness_metadata_rejects_invalid_degree_bound_ranges() {
        let mut metadata = witness_metadata(6, 1);
        metadata.witness_randomization.private_column_degree_bounds[0].range =
            ZkColumnRange::new(1, 0, 0);

        assert_eq!(
            validate_zk_witness_metadata(&metadata, 6, 1),
            Err(ZkMetadataValidationError::InvalidColumnDegreeBoundRange {
                range: ZkColumnRange::new(1, 0, 0),
            })
        );

        let mut metadata = witness_metadata(6, 1);
        metadata.quotient_integration.quotient_degree_bounds[0].range = ZkColumnRange {
            tree_index: 2,
            column_start: 1,
            column_end: 0,
        };

        assert_eq!(
            validate_zk_witness_metadata(&metadata, 6, 1),
            Err(ZkMetadataValidationError::InvalidColumnDegreeBoundRange {
                range: ZkColumnRange {
                    tree_index: 2,
                    column_start: 1,
                    column_end: 0,
                },
            })
        );
    }

    #[test]
    fn zk_column_degree_bounds_overlay_increases_selected_columns() {
        let base_bounds = TreeVec(vec![vec![4, 4], vec![5]]);
        let zk_bounds = vec![
            ZkColumnDegreeBound {
                range: ZkColumnRange::new(0, 1, 2),
                log_degree_bound: 6,
            },
            ZkColumnDegreeBound {
                range: ZkColumnRange::new(1, 0, 1),
                log_degree_bound: 7,
            },
        ];

        let overlay = apply_zk_column_degree_bounds(base_bounds, &zk_bounds).unwrap();
        assert_eq!(overlay.0, vec![vec![4, 6], vec![7]]);
    }

    #[test]
    fn zk_column_degree_bounds_reject_invalid_ranges() {
        let base_bounds = TreeVec(vec![vec![4]]);

        assert_eq!(
            apply_zk_column_degree_bounds(
                base_bounds.clone(),
                &[ZkColumnDegreeBound {
                    range: ZkColumnRange::new(0, 0, 0),
                    log_degree_bound: 5,
                }],
            )
            .unwrap_err(),
            ZkColumnDegreeBoundApplicationError::InvalidRange {
                range: ZkColumnRange::new(0, 0, 0),
            }
        );

        assert_eq!(
            apply_zk_column_degree_bounds(
                base_bounds.clone(),
                &[ZkColumnDegreeBound {
                    range: ZkColumnRange {
                        tree_index: 0,
                        column_start: 1,
                        column_end: 0,
                    },
                    log_degree_bound: 5,
                }],
            )
            .unwrap_err(),
            ZkColumnDegreeBoundApplicationError::InvalidRange {
                range: ZkColumnRange {
                    tree_index: 0,
                    column_start: 1,
                    column_end: 0,
                },
            }
        );

        assert_eq!(
            apply_zk_column_degree_bounds(
                base_bounds.clone(),
                &[ZkColumnDegreeBound {
                    range: ZkColumnRange::new(1, 0, 1),
                    log_degree_bound: 5,
                }],
            )
            .unwrap_err(),
            ZkColumnDegreeBoundApplicationError::MissingTree {
                range: ZkColumnRange::new(1, 0, 1),
            }
        );

        assert_eq!(
            apply_zk_column_degree_bounds(
                base_bounds,
                &[ZkColumnDegreeBound {
                    range: ZkColumnRange::new(0, 0, 2),
                    log_degree_bound: 5,
                }],
            )
            .unwrap_err(),
            ZkColumnDegreeBoundApplicationError::RangeOutOfBounds {
                range: ZkColumnRange::new(0, 0, 2),
                column_count: 1,
            }
        );
    }

    #[test]
    fn zk_column_degree_bounds_reject_overlap_and_shrinking() {
        let base_bounds = TreeVec(vec![vec![4, 4]]);

        assert_eq!(
            apply_zk_column_degree_bounds(
                base_bounds.clone(),
                &[
                    ZkColumnDegreeBound {
                        range: ZkColumnRange::new(0, 0, 1),
                        log_degree_bound: 5,
                    },
                    ZkColumnDegreeBound {
                        range: ZkColumnRange::new(0, 0, 1),
                        log_degree_bound: 6,
                    },
                ],
            )
            .unwrap_err(),
            ZkColumnDegreeBoundApplicationError::DuplicateColumn {
                tree_index: 0,
                column_index: 0,
            }
        );

        assert_eq!(
            apply_zk_column_degree_bounds(
                base_bounds,
                &[ZkColumnDegreeBound {
                    range: ZkColumnRange::new(0, 1, 2),
                    log_degree_bound: 3,
                }],
            )
            .unwrap_err(),
            ZkColumnDegreeBoundApplicationError::DegreeBoundShrinksColumn {
                range: ZkColumnRange::new(0, 1, 2),
                column_index: 1,
                base_log_degree_bound: 4,
                zk_log_degree_bound: 3,
            }
        );
    }

    #[test]
    fn zk_committed_column_log_sizes_accept_expected_blowup() {
        assert_eq!(
            validate_zk_committed_column_log_sizes(
                &TreeVec(vec![vec![6, 7], vec![8]]),
                &TreeVec(vec![vec![5, 6], vec![7]]),
                1,
            ),
            Ok(())
        );
    }

    #[test]
    fn zk_committed_column_log_sizes_reject_shape_mismatch() {
        assert_eq!(
            validate_zk_committed_column_log_sizes(
                &TreeVec(vec![vec![6]]),
                &TreeVec(vec![vec![5], vec![7]]),
                1,
            ),
            Err(ZkCommittedColumnLogSizeValidationError::TreeCountMismatch {
                expected: 2,
                actual: 1,
            })
        );

        assert_eq!(
            validate_zk_committed_column_log_sizes(
                &TreeVec(vec![vec![6]]),
                &TreeVec(vec![vec![5, 6]]),
                1,
            ),
            Err(
                ZkCommittedColumnLogSizeValidationError::ColumnCountMismatch {
                    tree_index: 0,
                    expected: 2,
                    actual: 1,
                },
            )
        );
    }

    #[test]
    fn zk_committed_column_log_sizes_reject_degree_mismatch() {
        assert_eq!(
            validate_zk_committed_column_log_sizes(
                &TreeVec(vec![vec![6, 9]]),
                &TreeVec(vec![vec![5, 6]]),
                1,
            ),
            Err(ZkCommittedColumnLogSizeValidationError::LogSizeMismatch {
                tree_index: 0,
                column_index: 1,
                expected_log_size: 7,
                actual_log_size: 9,
            })
        );
    }

    #[test]
    fn zk_composition_column_log_sizes_accept_expected_split_bound() {
        assert_eq!(
            validate_zk_composition_column_log_sizes(&[6; 2 * SECURE_EXTENSION_DEGREE], 8, 5, 1),
            Ok(())
        );
    }

    #[test]
    fn zk_composition_column_log_sizes_reject_wrong_split_bound() {
        assert_eq!(
            validate_zk_composition_column_log_sizes(&[7; 2 * SECURE_EXTENSION_DEGREE], 8, 5, 1),
            Err(ZkCompositionColumnLogSizeValidationError::LogSizeMismatch {
                column_index: 0,
                expected_log_size: 6,
                actual_log_size: 7,
            })
        );
        assert_eq!(
            validate_zk_composition_column_log_sizes(&[6; SECURE_EXTENSION_DEGREE], 8, 5, 1),
            Err(
                ZkCompositionColumnLogSizeValidationError::ColumnCountMismatch {
                    expected: 8,
                    actual: 4,
                },
            )
        );
    }

    #[test]
    fn zk_sampled_values_shape_accepts_exact_nested_shape() {
        let point = CirclePoint::<SecureField>::zero();

        assert_eq!(
            validate_zk_sampled_values_shape(
                &TreeVec(vec![vec![vec![point, point]], vec![vec![point]]]),
                &TreeVec(vec![
                    vec![vec![SecureField::zero(), SecureField::one()]],
                    vec![vec![SecureField::zero()]],
                ]),
            ),
            Ok(())
        );
    }

    #[test]
    fn zk_sampled_values_shape_rejects_mismatches() {
        let point = CirclePoint::<SecureField>::zero();

        assert_eq!(
            validate_zk_sampled_values_shape(
                &TreeVec(vec![vec![vec![point]]]),
                &TreeVec(vec![vec![vec![SecureField::zero()]], vec![]]),
            ),
            Err(ZkSampledValuesShapeValidationError::TreeCountMismatch {
                expected: 1,
                actual: 2,
            })
        );

        assert_eq!(
            validate_zk_sampled_values_shape(
                &TreeVec(vec![vec![vec![point], vec![point]]]),
                &TreeVec(vec![vec![vec![SecureField::zero()]]]),
            ),
            Err(ZkSampledValuesShapeValidationError::ColumnCountMismatch {
                tree_index: 0,
                expected: 2,
                actual: 1,
            })
        );

        assert_eq!(
            validate_zk_sampled_values_shape(
                &TreeVec(vec![vec![vec![point, point]]]),
                &TreeVec(vec![vec![vec![SecureField::zero()]]]),
            ),
            Err(ZkSampledValuesShapeValidationError::SampleCountMismatch {
                tree_index: 0,
                column_index: 0,
                expected: 2,
                actual: 1,
            })
        );
    }

    fn secure_circle_point(point: CirclePoint<BaseField>) -> CirclePoint<SecureField> {
        CirclePoint {
            x: point.x.into(),
            y: point.y.into(),
        }
    }

    #[test]
    fn zk_oods_sample_points_accept_safe_points() {
        let point = CirclePoint::<SecureField>::get_point(5);
        let exclusion_set = ZkOodsExclusionSet {
            forbidden_cosets: vec![CanonicCoset::new(4).coset],
            reject_line_degeneracy: true,
        };

        assert_eq!(
            validate_zk_sample_points_outside_exclusion_set(
                &TreeVec(vec![vec![vec![point]]]),
                &exclusion_set,
            ),
            Ok(())
        );
    }

    #[test]
    fn zk_oods_point_with_preimage_binds_public_double_relation() {
        let exclusion_set = ZkOodsExclusionSet::empty();
        let mut channel = Blake2sChannel::default();
        let (point, preimage) =
            draw_zk_oods_point_with_preimage(&mut channel, &exclusion_set, 64).unwrap();

        assert_eq!(point, preimage.double());
        assert!(exclusion_set.accepts(point));
        assert!(exclusion_set.accepts(preimage));
    }

    #[test]
    fn zk_oods_sample_point_plan_handles_mixed_lifts_and_offsets() {
        let deepest_point = CirclePoint::<SecureField>::get_point(17);
        let mask_offsets = TreeVec(vec![vec![vec![0, 2], vec![-1]], vec![vec![0]]]);
        let committed_log_sizes = TreeVec(vec![vec![7, 8], vec![9]]);
        let plan = plan_zk_oods_sample_points(
            deepest_point,
            &mask_offsets,
            &committed_log_sizes,
            9,
            6,
            &ZkOodsExclusionSet {
                forbidden_cosets: vec![],
                reject_line_degeneracy: false,
            },
        )
        .unwrap();

        assert_eq!(plan.max_delta, 2);
        assert_eq!(plan.semantic_oods_point, deepest_point.repeated_double(2));
        for (tree_index, tree_points) in plan.pcs_sample_points.iter().enumerate() {
            for (column_index, column_points) in tree_points.iter().enumerate() {
                let delta = 9 - committed_log_sizes[tree_index][column_index];
                for (sample_index, &point) in column_points.iter().enumerate() {
                    assert_eq!(
                        point.repeated_double(delta),
                        plan.semantic_sample_points[tree_index][column_index][sample_index]
                    );
                }
            }
        }
    }

    #[test]
    fn zk_oods_sample_point_plan_uses_column_semantic_step_domains() {
        let deepest_point = CirclePoint::<SecureField>::get_point(19);
        let mask_offsets = TreeVec(vec![vec![vec![3]]]);
        let semantic_step_log_sizes = TreeVec(vec![vec![5]]);
        let committed_log_sizes = TreeVec(vec![vec![7]]);
        let plan = plan_zk_oods_sample_points_with_semantic_step_log_sizes(
            deepest_point,
            &mask_offsets,
            &semantic_step_log_sizes,
            &committed_log_sizes,
            9,
            &ZkOodsExclusionSet {
                forbidden_cosets: vec![],
                reject_line_degeneracy: false,
            },
        )
        .unwrap();

        let semantic_oods_point = deepest_point.repeated_double(2);
        let expected_semantic_point =
            semantic_oods_point + CanonicCoset::new(5).step().mul_signed(3).into_ef();

        assert_eq!(plan.semantic_oods_point, semantic_oods_point);
        assert_eq!(
            plan.semantic_sample_points[0][0][0],
            expected_semantic_point
        );
        assert_eq!(
            plan.pcs_sample_points[0][0][0].repeated_double(2),
            expected_semantic_point
        );
    }

    #[test]
    fn zk_oods_sample_point_plan_rejects_forbidden_deepest_point() {
        assert!(matches!(
            plan_zk_oods_sample_points(
                CirclePoint::<SecureField>::zero(),
                &TreeVec(vec![vec![vec![0]]]),
                &TreeVec(vec![vec![4]]),
                4,
                3,
                &ZkOodsExclusionSet::empty(),
            ),
            Err(ZkOodsSamplePointPlanError::ForbiddenDeepestPoint)
        ));
    }

    #[test]
    fn zk_oods_sample_point_plan_rejects_invalid_geometry() {
        assert!(matches!(
            plan_zk_oods_sample_points(
                CirclePoint::<SecureField>::get_point(17),
                &TreeVec(vec![vec![vec![0]]]),
                &TreeVec(vec![vec![16]]),
                15,
                6,
                &ZkOodsExclusionSet {
                    forbidden_cosets: vec![],
                    reject_line_degeneracy: false,
                },
            ),
            Err(ZkOodsSamplePointPlanError::ColumnLogSizeAboveLifting {
                tree_index: 0,
                column_index: 0,
                column_log_size: 16,
                lifting_log_size: 15,
            })
        ));

        assert!(matches!(
            plan_zk_oods_sample_points(
                CirclePoint::<SecureField>::get_point(17),
                &TreeVec(vec![vec![vec![0]]]),
                &TreeVec(vec![vec![1]]),
                u32::MAX,
                6,
                &ZkOodsExclusionSet {
                    forbidden_cosets: vec![],
                    reject_line_degeneracy: false,
                },
            ),
            Err(ZkOodsSamplePointPlanError::InvalidSamplePointLiftDomain { log_size: u32::MAX })
        ));

        assert_eq!(
            plan_zk_oods_sample_points(
                CirclePoint::<SecureField>::get_point(17),
                &TreeVec(vec![vec![vec![0]]]),
                &TreeVec(vec![vec![1]]),
                10,
                6,
                &ZkOodsExclusionSet {
                    forbidden_cosets: vec![],
                    reject_line_degeneracy: false,
                },
            )
            .unwrap()
            .max_delta,
            9
        );

        for (committed_log_size, lifting_log_size, expected_delta) in [(1, 11, 10), (1, 13, 12)] {
            let plan = plan_zk_oods_sample_points(
                CirclePoint::<SecureField>::get_point(17),
                &TreeVec(vec![vec![vec![0, 1]]]),
                &TreeVec(vec![vec![committed_log_size]]),
                lifting_log_size,
                6,
                &ZkOodsExclusionSet {
                    forbidden_cosets: vec![],
                    reject_line_degeneracy: false,
                },
            )
            .unwrap();

            assert_eq!(plan.max_delta, expected_delta);
            for (semantic_point, pcs_point) in plan.semantic_sample_points[0][0]
                .iter()
                .zip(plan.pcs_sample_points[0][0].iter())
            {
                assert_eq!(pcs_point.repeated_double(expected_delta), *semantic_point);
            }
        }
    }

    #[test]
    fn zk_oods_sample_point_plan_is_deterministic() {
        let deepest_point = CirclePoint::<SecureField>::get_point(17);
        let mask_offsets = TreeVec(vec![vec![vec![0, 1]], vec![vec![0]]]);
        let committed_log_sizes = TreeVec(vec![vec![8], vec![9]]);
        let exclusion_set = ZkOodsExclusionSet {
            forbidden_cosets: vec![],
            reject_line_degeneracy: false,
        };

        let first = plan_zk_oods_sample_points(
            deepest_point,
            &mask_offsets,
            &committed_log_sizes,
            9,
            6,
            &exclusion_set,
        )
        .unwrap();
        let second = plan_zk_oods_sample_points(
            deepest_point,
            &mask_offsets,
            &committed_log_sizes,
            9,
            6,
            &exclusion_set,
        )
        .unwrap();

        assert_eq!(first.semantic_oods_point, second.semantic_oods_point);
        assert_eq!(first.max_delta, second.max_delta);
        assert_eq!(
            first.semantic_sample_points.0,
            second.semantic_sample_points.0
        );
        assert_eq!(first.pcs_sample_points.0, second.pcs_sample_points.0);
    }

    #[test]
    fn zk_oods_sample_points_reject_forbidden_coset_and_degeneracy() {
        let coset = CanonicCoset::new(4).coset;
        let forbidden_point = secure_circle_point(coset.at(0));
        let forbidden_coset_exclusion_set = ZkOodsExclusionSet {
            forbidden_cosets: vec![coset],
            reject_line_degeneracy: false,
        };

        assert_eq!(
            validate_zk_sample_points_outside_exclusion_set(
                &TreeVec(vec![vec![vec![forbidden_point]]]),
                &forbidden_coset_exclusion_set,
            ),
            Err(ZkOodsSamplePointValidationError::ForbiddenSamplePoint {
                tree_index: 0,
                column_index: 0,
                sample_index: 0,
            })
        );

        let degenerate_point = CirclePoint::<SecureField>::zero();
        assert_eq!(
            validate_zk_sample_points_outside_exclusion_set(
                &TreeVec(vec![vec![vec![degenerate_point]]]),
                &ZkOodsExclusionSet::empty(),
            ),
            Err(ZkOodsSamplePointValidationError::ForbiddenSamplePoint {
                tree_index: 0,
                column_index: 0,
                sample_index: 0,
            })
        );
    }

    #[test]
    fn zk_stark_degree_profile_uses_normal_public_only_bounds() {
        let metadata = public_only_metadata(6, 1);
        let verifier_config = verification_config(metadata);
        let profile = derive_zk_stark_degree_bound_profile(
            TreeVec(vec![vec![5], vec![5]]),
            6,
            1,
            &verifier_config,
            6,
            1,
        )
        .unwrap();

        assert_eq!(profile.column_log_degree_bounds.0, vec![vec![5], vec![5]]);
        assert_eq!(profile.trace_log_degree_bound, 5);
        assert_eq!(profile.split_composition_log_degree_bound, 5);
        assert_eq!(profile.composition_log_degree_bound, 6);
        assert_eq!(profile.fri_first_layer_log_size, 6);
    }

    #[test]
    fn zk_stark_degree_profile_applies_private_and_quotient_bounds() {
        let mut metadata = witness_metadata(7, 1);
        set_witness_trace_domain(&mut metadata, 5);
        metadata.witness_randomization.private_column_degree_bounds[0].log_degree_bound = 6;
        metadata.quotient_integration.quotient_degree_bounds[0].log_degree_bound = 5;
        let verifier_config = verification_config(metadata);
        let profile = derive_zk_stark_degree_bound_profile(
            TreeVec(vec![vec![5], vec![5]]),
            6,
            1,
            &verifier_config,
            7,
            1,
        )
        .unwrap();

        assert_eq!(profile.column_log_degree_bounds.0, vec![vec![5], vec![6]]);
        assert_eq!(profile.trace_log_degree_bound, 6);
        assert_eq!(profile.split_composition_log_degree_bound, 5);
        assert_eq!(profile.composition_log_degree_bound, 6);
        assert_eq!(profile.fri_first_layer_log_size, 7);
    }

    #[test]
    fn zk_stark_degree_profile_rejects_unsplit_private_quotient_width() {
        let mut metadata = witness_metadata(7, 1);
        set_witness_trace_domain(&mut metadata, 5);
        metadata.witness_randomization.private_column_degree_bounds[0].log_degree_bound = 6;
        metadata.quotient_integration.split_derivation_hash = canonical_zk_split_derivation_hash(0);
        metadata.quotient_integration.quotient_degree_bounds[0].log_degree_bound = 5;
        metadata.quotient_integration.quotient_degree_bounds[0].range =
            ZkColumnRange::new(2, 0, SECURE_EXTENSION_DEGREE);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                5,
                0,
                &verifier_config,
                7,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::VerificationConfig(
                ZkVerificationConfigValidationError::QuotientSplitMaskProfile(
                    ZkQuotientSplitMaskProfileBindingError::UnexpectedQuotientSplitDegreeBounds {
                        expected_range: ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE),
                        actual_range: Some(ZkColumnRange::new(2, 0, SECURE_EXTENSION_DEGREE)),
                        actual_count: 1,
                    }
                )
            )
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_wider_private_quotient_split_width() {
        let mut metadata = witness_metadata(8, 1);
        set_witness_trace_domain(&mut metadata, 6);
        metadata.witness_randomization.private_column_degree_bounds[0].log_degree_bound = 7;
        metadata.quotient_integration.split_derivation_hash = canonical_zk_split_derivation_hash(2);
        metadata.quotient_integration.quotient_degree_bounds[0].log_degree_bound = 6;
        metadata.quotient_integration.quotient_degree_bounds[0].range =
            ZkColumnRange::new(2, 0, 4 * SECURE_EXTENSION_DEGREE);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                2,
                &verifier_config,
                8,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::VerificationConfig(
                ZkVerificationConfigValidationError::QuotientSplitMaskProfile(
                    ZkQuotientSplitMaskProfileBindingError::UnexpectedQuotientSplitDegreeBounds {
                        expected_range: ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE),
                        actual_range: Some(ZkColumnRange::new(2, 0, 4 * SECURE_EXTENSION_DEGREE)),
                        actual_count: 1,
                    }
                )
            )
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_wrong_split_derivation_hash() {
        let mut metadata = witness_metadata(7, 1);
        set_witness_trace_domain(&mut metadata, 5);
        metadata.witness_randomization.private_column_degree_bounds[0].log_degree_bound = 6;
        metadata.quotient_integration.quotient_degree_bounds[0].log_degree_bound = 5;
        metadata.quotient_integration.split_derivation_hash = canonical_zk_split_derivation_hash(2);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                1,
                &verifier_config,
                7,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::SplitDerivationHashMismatch {
                expected: canonical_zk_split_derivation_hash(1),
                actual: canonical_zk_split_derivation_hash(2),
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_shrinking_quotient_bound() {
        let mut metadata = witness_metadata(6, 1);
        metadata.quotient_integration.quotient_degree_bounds[0].log_degree_bound = 4;
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::QuotientDegreeBoundShrinksComposition {
                normal_split_composition_log_degree_bound: 5,
                zk_split_composition_log_degree_bound: 4,
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_invalid_quotient_range() {
        let mut metadata = witness_metadata(6, 1);
        metadata.quotient_integration.quotient_degree_bounds[0].range = ZkColumnRange::new(2, 0, 0);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::VerificationConfig(
                ZkVerificationConfigValidationError::Metadata(
                    ZkMetadataValidationError::InvalidColumnDegreeBoundRange {
                        range: ZkColumnRange::new(2, 0, 0),
                    },
                ),
            )
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_small_fri_layer() {
        let mut metadata = witness_metadata(5, 1);
        set_witness_trace_domain(&mut metadata, 3);
        metadata.witness_randomization.private_column_degree_bounds[0].log_degree_bound = 4;
        metadata.quotient_integration.quotient_degree_bounds[0].log_degree_bound = 5;
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![4], vec![4]]),
                5,
                1,
                &verifier_config,
                5,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::FriFirstLayerTooSmall {
                required: 7,
                actual: 5,
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_public_trace_bound_above_fri_layer() {
        let metadata = public_only_metadata(6, 1);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![7], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::FriFirstLayerBelowCommittedTraceSize {
                required: 8,
                actual: 6,
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_private_trace_bound_above_fri_layer() {
        let metadata = witness_metadata(6, 1);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![7], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::FriFirstLayerBelowCommittedTraceSize {
                required: 8,
                actual: 6,
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_public_committed_trace_above_fri_layer() {
        let metadata = public_only_metadata(6, 1);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![6], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::FriFirstLayerBelowCommittedTraceSize {
                required: 7,
                actual: 6,
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_private_committed_trace_above_fri_layer() {
        let metadata = witness_metadata(6, 1);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![6], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::FriFirstLayerBelowCommittedTraceSize {
                required: 7,
                actual: 6,
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_requires_quotient_bounds_for_private_witness() {
        let mut metadata = witness_metadata(6, 1);
        metadata.quotient_integration.quotient_degree_bounds.clear();
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::Metadata(
                ZkMetadataValidationError::MissingQuotientDegreeBounds,
            )
        );
    }

    #[test]
    fn zk_stark_degree_profile_expands_to_larger_quotient_bound() {
        let mut metadata = witness_metadata(8, 1);
        set_witness_trace_domain(&mut metadata, 6);
        metadata.witness_randomization.private_column_degree_bounds[0].log_degree_bound = 7;
        metadata.quotient_integration.quotient_degree_bounds[0].log_degree_bound = 6;
        let verifier_config = verification_config(metadata);
        let profile = derive_zk_stark_degree_bound_profile(
            TreeVec(vec![vec![5], vec![5]]),
            6,
            1,
            &verifier_config,
            8,
            1,
        )
        .unwrap();

        assert_eq!(profile.column_log_degree_bounds.0, vec![vec![5], vec![7]]);
        assert_eq!(profile.trace_log_degree_bound, 7);
        assert_eq!(profile.split_composition_log_degree_bound, 6);
        assert_eq!(profile.composition_log_degree_bound, 7);
        assert_eq!(profile.fri_first_layer_log_size, 8);
    }

    #[test]
    fn zk_stark_degree_profile_rejects_wrong_quotient_tree() {
        let mut metadata = witness_metadata(6, 1);
        metadata.quotient_integration.quotient_degree_bounds[0].range =
            ZkColumnRange::new(3, 0, 2 * SECURE_EXTENSION_DEGREE);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::UnexpectedQuotientDegreeBoundRange {
                expected_range: ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE),
                actual_range: ZkColumnRange::new(3, 0, 2 * SECURE_EXTENSION_DEGREE),
            }
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_noncanonical_quotient_bounds() {
        let mut metadata = witness_metadata(6, 1);
        metadata
            .quotient_integration
            .quotient_degree_bounds
            .push(metadata.quotient_integration.quotient_degree_bounds[0]);
        let verifier_config = verification_config(metadata);

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::VerificationConfig(
                ZkVerificationConfigValidationError::QuotientSplitMaskProfile(
                    ZkQuotientSplitMaskProfileBindingError::UnexpectedQuotientSplitDegreeBounds {
                        expected_range: ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE),
                        actual_range: Some(ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE)),
                        actual_count: 2,
                    }
                )
            )
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_verifier_config_bound_mismatch() {
        let metadata = public_only_metadata(6, 1);
        let verifier_config = ZkVerificationConfig {
            metadata,
            column_degree_bounds: vec![ZkColumnDegreeBound {
                range: ZkColumnRange::new(0, 0, 1),
                log_degree_bound: 5,
            }],
            quotient_split_mask_profile: None,
            logup_statistical_security_budgets: Vec::new(),
        };

        assert_eq!(
            derive_zk_stark_degree_bound_profile(
                TreeVec(vec![vec![5], vec![5]]),
                6,
                1,
                &verifier_config,
                6,
                1,
            )
            .unwrap_err(),
            ZkStarkDegreeBoundProfileError::VerificationConfig(
                ZkVerificationConfigValidationError::ColumnDegreeBoundsMismatch,
            )
        );
    }

    #[test]
    fn verifier_config_rejects_proof_metadata_mismatch() {
        let metadata = public_only_metadata(16, 1);
        let mut proof_metadata = metadata.clone();
        proof_metadata.public_statement_hash = ZkPublicStatementHash([9; 32]);
        let verifier_config = ZkVerificationConfig {
            metadata,
            column_degree_bounds: Vec::new(),
            quotient_split_mask_profile: None,
            logup_statistical_security_budgets: Vec::new(),
        };

        assert_eq!(
            validate_zk_public_only_metadata_against_verifier_config(
                &proof_metadata,
                &verifier_config,
                16,
                1,
            ),
            Err(ZkVerificationConfigValidationError::PublicMetadataMismatch)
        );
    }

    #[test]
    fn verifier_config_requires_private_quotient_split_mask_profile() {
        let metadata = witness_metadata(6, 1);
        let verifier_config = ZkVerificationConfig {
            metadata: metadata.clone(),
            column_degree_bounds: expected_zk_column_degree_bounds(&metadata),
            quotient_split_mask_profile: None,
            logup_statistical_security_budgets: Vec::new(),
        };

        assert_eq!(
            validate_zk_public_metadata_against_verifier_config(&metadata, &verifier_config),
            Err(ZkVerificationConfigValidationError::MissingQuotientSplitMaskProfile)
        );
    }

    #[test]
    fn verifier_config_rejects_wrong_private_quotient_split_mask_profile() {
        let metadata = witness_metadata(6, 1);
        let mut verifier_config = verification_config(metadata.clone());
        let expected = verifier_config
            .quotient_split_mask_profile
            .expect("witness verifier config must derive split mask profile");
        let mut actual = expected;
        actual.left_range.tree_index += 1;
        actual.right_range.tree_index += 1;
        verifier_config.quotient_split_mask_profile = Some(actual);

        assert_eq!(
            validate_zk_public_metadata_against_verifier_config(&metadata, &verifier_config),
            Err(
                ZkVerificationConfigValidationError::QuotientSplitMaskProfile(
                    ZkQuotientSplitMaskProfileBindingError::ProfileMismatch { expected, actual }
                )
            )
        );
    }

    #[test]
    fn verifier_config_rejects_larger_private_quotient_split_right_bound() {
        let metadata = witness_metadata(6, 1);
        let mut verifier_config = verification_config(metadata.clone());
        let expected = verifier_config
            .quotient_split_mask_profile
            .expect("witness verifier config must derive split mask profile");
        let mut actual = expected;
        actual.right_masked_log_degree_bound += 1;
        verifier_config.quotient_split_mask_profile = Some(actual);

        assert_eq!(
            validate_zk_public_metadata_against_verifier_config(&metadata, &verifier_config),
            Err(
                ZkVerificationConfigValidationError::QuotientSplitMaskProfile(
                    ZkQuotientSplitMaskProfileBindingError::ProfileMismatch { expected, actual }
                )
            )
        );
    }

    #[test]
    fn verifier_config_rejects_undercovered_private_quotient_split_entropy() {
        let metadata = witness_metadata(6, 1);
        let mut verifier_config = verification_config(metadata.clone());
        let mut actual = verifier_config
            .quotient_split_mask_profile
            .expect("witness verifier config must derive split mask profile");
        actual.h_split -= 1;
        verifier_config.quotient_split_mask_profile = Some(actual);

        assert_eq!(
            validate_zk_public_metadata_against_verifier_config(&metadata, &verifier_config),
            Err(
                ZkVerificationConfigValidationError::QuotientSplitMaskProfile(
                    ZkQuotientSplitMaskProfileBindingError::SplitMaskEntropyBelowFullDimension {
                        required: actual.h_split + 1,
                        actual: actual.h_split,
                    }
                )
            )
        );
    }

    #[test]
    fn verifier_config_rejects_public_only_quotient_split_mask_profile() {
        let metadata = public_only_metadata(16, 1);
        let verifier_config = ZkVerificationConfig {
            metadata: metadata.clone(),
            column_degree_bounds: Vec::new(),
            quotient_split_mask_profile: Some(
                stwo_composition_quotient_split_mask_profile(1, 6, 5, 32, 6, 5).unwrap(),
            ),
            logup_statistical_security_budgets: Vec::new(),
        };

        assert_eq!(
            validate_zk_public_metadata_against_verifier_config(&metadata, &verifier_config),
            Err(
                ZkVerificationConfigValidationError::UnexpectedQuotientSplitMaskProfileForPublicOnly
            )
        );
    }

    #[test]
    fn verifier_config_rejects_public_only_column_degree_bounds() {
        let metadata = public_only_metadata(16, 1);
        let verifier_config = ZkVerificationConfig {
            metadata: metadata.clone(),
            column_degree_bounds: vec![ZkColumnDegreeBound {
                range: ZkColumnRange::new(0, 0, 1),
                log_degree_bound: 15,
            }],
            quotient_split_mask_profile: None,
            logup_statistical_security_budgets: Vec::new(),
        };

        assert_eq!(
            validate_zk_public_only_metadata_against_verifier_config(
                &metadata,
                &verifier_config,
                16,
                1,
            ),
            Err(ZkVerificationConfigValidationError::ColumnDegreeBoundsMismatch)
        );
    }

    #[test]
    fn base_field_matrix_rank_computes_row_rank() {
        let rows = vec![
            vec![BaseField::from(1), BaseField::from(2), BaseField::from(3)],
            vec![BaseField::from(2), BaseField::from(4), BaseField::from(6)],
            vec![BaseField::from(0), BaseField::from(1), BaseField::from(1)],
        ];

        assert_eq!(base_field_matrix_rank(rows), 2);
    }

    #[test]
    fn private_column_scope_rejects_non_singleton_private_range() {
        let range = ZkColumnRange::new(0, 0, 2);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(nonzero_hash()),
        };
        let mut scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: zero_hash(),
            entries: vec![ZkPrivateColumnScopeEntry {
                range,
                usage: ZkPrivateColumnUsage::OrdinaryWitness,
                trace_domain_log_size: 3,
                semantic_trace_domain_log_sizes: vec![3],
            }],
        };
        let scope_hash = canonical_zk_private_column_scope_hash(&scope);
        scope.hash = scope_hash;

        assert_eq!(
            validate_zk_private_column_scope_for_witness_randomization(
                &privacy_map,
                scope_hash,
                &scope,
            ),
            Err(ZkPrivateColumnScopeValidationError::NonSingletonPrivateRange { range })
        );
    }

    #[test]
    fn private_column_scope_accepts_reviewed_logup_private_column() {
        let range = ZkColumnRange::new(0, 0, 1);
        let mut privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(zero_hash()),
        };
        privacy_map.hash = canonical_zk_privacy_map_hash(&privacy_map);
        let mut scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: zero_hash(),
            entries: vec![ZkPrivateColumnScopeEntry {
                range,
                usage: ZkPrivateColumnUsage::LogUp,
                trace_domain_log_size: 3,
                semantic_trace_domain_log_sizes: vec![3],
            }],
        };
        let scope_hash = canonical_zk_private_column_scope_hash(&scope);
        scope.hash = scope_hash;

        assert!(validate_zk_private_column_scope_for_witness_randomization(
            &privacy_map,
            scope_hash,
            &scope,
        )
        .is_ok());
    }

    #[test]
    fn randomizer_matrix_builder_constructs_rank_profile() {
        let range = ZkColumnRange::new(0, 0, 1);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(nonzero_hash()),
        };
        let trace_domain = CanonicCoset::new(3).coset;
        let query_point = CirclePoint::<SecureField>::get_point(9834759221);
        let build = build_zk_randomizer_matrices_for_witness_randomization(
            trace_domain,
            &privacy_map,
            2,
            &[ZkRandomizerQueryFunctional {
                range,
                kind: ZkQueryClosureKind::OodsExtension,
                domain_id: 0,
                point_or_position: [1, 0, 0, 0, 0, 0, 0, 0],
                point: query_point,
                coordinate_index: 0,
            }],
        )
        .unwrap();

        assert_eq!(build.closure.query_count_for_range(range), 1);
        assert_eq!(build.matrices.len(), 1);
        assert_eq!(build.matrices[0].query_count, 1);
        assert_eq!(build.matrices[0].randomizer_dimension, 2);
        assert_eq!(build.rank_profile.entries.len(), 1);
        assert_eq!(build.rank_profile.entries[0].range, range);
        assert_eq!(build.rank_profile.entries[0].query_count, 1);
        assert_eq!(build.rank_profile.entries[0].randomizer_dimension, 2);
        assert_eq!(build.rank_profile.entries[0].rank, 1);
    }

    #[test]
    fn randomizer_matrix_builder_rejects_invalid_coordinate_index() {
        let range = ZkColumnRange::new(0, 0, 1);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(nonzero_hash()),
        };
        let query_point = CirclePoint::<SecureField>::get_point(9834759221);
        let functional = ZkRandomizerQueryFunctional {
            range,
            kind: ZkQueryClosureKind::FriPosition,
            domain_id: 4,
            point_or_position: encode_zk_query_position(1),
            point: query_point,
            coordinate_index: 1,
        };

        assert_eq!(
            build_zk_randomizer_matrices_for_witness_randomization(
                CanonicCoset::new(3).coset,
                &privacy_map,
                2,
                &[functional],
            ),
            Err(ZkRandomizerMatrixBuildError::InvalidCoordinateIndex {
                functional,
                max_exclusive: 1,
            })
        );
    }

    #[test]
    fn randomizer_rank_profile_rejects_duplicate_private_range() {
        let range = ZkColumnRange::new(0, 0, 1);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(nonzero_hash()),
        };
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash: privacy_map.hash,
            public_statement_hash: ZkPublicStatementHash(nonzero_hash()),
            logup_statistical_security_budget_hash: [0x5a; 32],
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: 3,
                h_witness: 2,
                h_batch: 0,
                fri_first_layer_log_size: 4,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness: 2,
                randomizer_space_hash: nonzero_hash(),
                private_column_scope_hash: nonzero_hash(),
                private_column_degree_bounds: Vec::new(),
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch: 0,
                fri_first_layer_log_size: 4,
                split_derivation_hash: nonzero_hash(),
                quotient_degree_bounds: Vec::new(),
            },
        };
        let closure = ZkQueryClosure {
            entries: vec![ZkQueryClosureEntry::new(
                range,
                ZkQueryClosureKind::FriPosition,
                4,
                encode_zk_query_position(1),
                0,
            )],
        };
        let profile = ZkRandomizerRankProfile {
            entries: vec![
                ZkRandomizerRankEntry {
                    range,
                    query_count: 1,
                    randomizer_dimension: 2,
                    rank: 1,
                },
                ZkRandomizerRankEntry {
                    range,
                    query_count: 2,
                    randomizer_dimension: 2,
                    rank: 2,
                },
            ],
        };

        assert_eq!(
            validate_zk_randomizer_rank_profile_for_witness_randomization(
                &privacy_map,
                &metadata,
                &closure,
                &profile,
            ),
            Err(ZkRandomizerRankValidationError::DuplicatePrivateColumn { range })
        );
    }

    fn verifier_witness_audit_fixture(
        h_witness: u64,
    ) -> (
        ZkVerificationConfig,
        ZkWitnessRandomizationVerifierAudit,
        TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        Vec<usize>,
    ) {
        let range = ZkColumnRange::new(0, 0, 1);
        let trace_domain = CanonicCoset::new(3).coset;
        let randomized_log_degree = 4;
        let privacy_map_hash = ZkPrivacyMapHash(nonzero_hash());
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: privacy_map_hash,
        };
        let mut private_column_scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: [0; 32],
            entries: vec![ZkPrivateColumnScopeEntry {
                range,
                usage: ZkPrivateColumnUsage::OrdinaryWitness,
                trace_domain_log_size: trace_domain.log_size(),
                semantic_trace_domain_log_sizes: vec![trace_domain.log_size()],
            }],
        };
        private_column_scope.hash = canonical_zk_private_column_scope_hash(&private_column_scope);
        let private_degree_bound = ZkColumnDegreeBound {
            range,
            log_degree_bound: randomized_log_degree,
        };
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash,
            public_statement_hash: ZkPublicStatementHash(nonzero_hash()),
            logup_statistical_security_budget_hash: [0x5a; 32],
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: trace_domain.log_size(),
                h_witness,
                h_batch: 0,
                fri_first_layer_log_size: 4,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness,
                randomizer_space_hash: canonical_zk_randomizer_space_hash(
                    private_column_scope.hash,
                    &[ZkRandomizerSpaceEntry {
                        range,
                        trace_domain: ZkCircleCosetEncoding::from(zk_trace_domain_half_coset(
                            trace_domain,
                        )),
                        semantic_trace_domains: vec![ZkCircleCosetEncoding::from(
                            zk_trace_domain_half_coset(trace_domain),
                        )],
                        randomized_log_degree,
                        randomizer_dimension: 1 << trace_domain.log_size(),
                    }],
                ),
                private_column_scope_hash: private_column_scope.hash,
                private_column_degree_bounds: vec![private_degree_bound],
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch: 0,
                fri_first_layer_log_size: 4,
                split_derivation_hash: nonzero_hash(),
                quotient_degree_bounds: Vec::new(),
            },
        };
        let verifier_config = ZkVerificationConfig {
            metadata,
            column_degree_bounds: vec![private_degree_bound],
            quotient_split_mask_profile: None,
            logup_statistical_security_budgets: Vec::new(),
        };
        let audit = ZkWitnessRandomizationVerifierAudit {
            privacy_map,
            private_column_scope,
        };
        (
            verifier_config,
            audit,
            TreeVec(vec![vec![vec![CirclePoint::<SecureField>::get_point(
                9834759221,
            )]]]),
            vec![1],
        )
    }

    #[test]
    fn verifier_witness_randomization_audit_accepts_derived_rank() {
        let (verifier_config, audit, sampled_points, fri_query_positions) =
            verifier_witness_audit_fixture(8);

        assert_eq!(
            validate_zk_witness_randomization_audit_for_verifier(
                &verifier_config,
                Some(&audit),
                &sampled_points,
                &fri_query_positions,
                4,
            ),
            Ok(())
        );
    }

    #[test]
    fn verifier_witness_randomization_audit_with_geometry_accepts_same_height() {
        let (verifier_config, audit, sampled_points, fri_query_positions) =
            verifier_witness_audit_fixture(8);
        let range = ZkColumnRange::new(0, 0, 1);
        let opening_geometry = vec![ZkPrivateColumnOpeningGeometry {
            range,
            tree_height: 4,
            committed_column_log_size: 4,
        }];

        assert_eq!(
            validate_zk_witness_randomization_audit_for_verifier_with_opening_geometry(
                &verifier_config,
                Some(&audit),
                &sampled_points,
                &fri_query_positions,
                4,
                &opening_geometry,
                0,
            ),
            Ok(())
        );
    }

    #[test]
    fn verifier_witness_randomization_audit_rejects_missing_audit() {
        let (verifier_config, _audit, sampled_points, fri_query_positions) =
            verifier_witness_audit_fixture(8);

        assert_eq!(
            validate_zk_witness_randomization_audit_for_verifier(
                &verifier_config,
                None,
                &sampled_points,
                &fri_query_positions,
                4,
            ),
            Err(ZkWitnessRandomizationVerifierAuditError::MissingAudit)
        );
    }

    #[test]
    fn verifier_witness_randomization_audit_rejects_randomizer_space_hash_mismatch() {
        let (mut verifier_config, audit, sampled_points, fri_query_positions) =
            verifier_witness_audit_fixture(8);
        verifier_config
            .metadata
            .witness_randomization
            .randomizer_space_hash[0] ^= 1;

        assert_eq!(
            validate_zk_witness_randomization_audit_for_verifier(
                &verifier_config,
                Some(&audit),
                &sampled_points,
                &fri_query_positions,
                4,
            ),
            Err(ZkWitnessRandomizationVerifierAuditError::RandomizerSpaceHashMismatch)
        );
    }

    #[test]
    fn verifier_witness_randomization_audit_rejects_insufficient_dynamic_rank_space() {
        let (verifier_config, audit, sampled_points, fri_query_positions) =
            verifier_witness_audit_fixture(4);
        let range = ZkColumnRange::new(0, 0, 1);

        assert_eq!(
            validate_zk_witness_randomization_audit_for_verifier(
                &verifier_config,
                Some(&audit),
                &sampled_points,
                &fri_query_positions,
                4,
            ),
            Err(ZkWitnessRandomizationVerifierAuditError::QueryClosure(
                ZkQueryClosureValidationError::InsufficientRandomizerDimension {
                    range,
                    required: 5,
                    actual: 4,
                },
            ))
        );
    }

    fn geometry_sample_metadata_fixture(
        bound_log_degree: u32,
        committed_column_log_size: u32,
    ) -> (
        ZkPrivacyMap,
        ZkPrivateColumnScope,
        Vec<ZkColumnDegreeBound>,
        TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        Vec<usize>,
        Vec<ZkPrivateColumnOpeningGeometry>,
    ) {
        let range = ZkColumnRange::new(0, 0, 1);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(nonzero_hash()),
        };
        let private_column_scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: nonzero_hash(),
            entries: vec![ZkPrivateColumnScopeEntry {
                range,
                usage: ZkPrivateColumnUsage::OrdinaryWitness,
                trace_domain_log_size: 3,
                semantic_trace_domain_log_sizes: vec![3],
            }],
        };
        let degree_bounds = vec![ZkColumnDegreeBound {
            range,
            log_degree_bound: bound_log_degree,
        }];
        let sampled_point = CirclePoint::<SecureField>::get_point(9834759221);
        let sampled_points = TreeVec(vec![vec![vec![sampled_point]]]);
        let fri_query_positions = vec![7];
        let opening_geometry = vec![ZkPrivateColumnOpeningGeometry {
            range,
            tree_height: committed_column_log_size,
            committed_column_log_size,
        }];

        (
            privacy_map,
            private_column_scope,
            degree_bounds,
            sampled_points,
            fri_query_positions,
            opening_geometry,
        )
    }

    #[test]
    fn geometry_sample_metadata_builder_projects_openings_to_committed_domain() {
        let (
            privacy_map,
            private_column_scope,
            degree_bounds,
            sampled_points,
            fri_query_positions,
            opening_geometry,
        ) = geometry_sample_metadata_fixture(4, 5);
        let range = ZkColumnRange::new(0, 0, 1);
        let lifting_log_size = 6;
        let build =
            build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
                &privacy_map,
                &private_column_scope,
                &degree_bounds,
                &sampled_points,
                &fri_query_positions,
                lifting_log_size,
                &opening_geometry,
                1,
            )
            .unwrap();
        let opened_point = sampled_points[0][0][0].repeated_double(1);
        let projected_fri_position = prepare_preprocessed_query_positions(
            &fri_query_positions,
            lifting_log_size,
            opening_geometry[0].tree_height,
        )[0];

        assert!(build.closure.entries.contains(&ZkQueryClosureEntry::new(
            range,
            ZkQueryClosureKind::OodsExtension,
            5,
            encode_zk_query_point(opened_point),
            0,
        )));
        assert!(build.closure.entries.contains(&ZkQueryClosureEntry::new(
            range,
            ZkQueryClosureKind::FriPosition,
            5,
            encode_zk_query_position(projected_fri_position),
            0,
        )));
    }

    #[test]
    fn geometry_sample_metadata_differs_from_legacy_raw_point_builder() {
        let (
            privacy_map,
            private_column_scope,
            degree_bounds,
            sampled_points,
            fri_query_positions,
            opening_geometry,
        ) = geometry_sample_metadata_fixture(4, 5);
        let lifting_log_size = 6;
        let legacy = build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope(
            &privacy_map,
            &private_column_scope,
            &sampled_points,
            &fri_query_positions,
            lifting_log_size,
        )
        .unwrap();
        let geometry =
            build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
                &privacy_map,
                &private_column_scope,
                &degree_bounds,
                &sampled_points,
                &fri_query_positions,
                lifting_log_size,
                &opening_geometry,
                1,
            )
            .unwrap();

        assert_ne!(legacy.closure, geometry.closure);
    }

    #[test]
    fn geometry_sample_metadata_rejects_product_degree_too_small() {
        let (
            privacy_map,
            private_column_scope,
            degree_bounds,
            sampled_points,
            fri_query_positions,
            opening_geometry,
        ) = geometry_sample_metadata_fixture(3, 4);
        let range = ZkColumnRange::new(0, 0, 1);

        assert_eq!(
            build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
                &privacy_map,
                &private_column_scope,
                &degree_bounds,
                &sampled_points,
                &fri_query_positions,
                6,
                &opening_geometry,
                1,
            ),
            Err(ZkSampleMetadataBuildError::PrivateColumnDegreeBoundTooSmall {
                range,
                required: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn geometry_sample_metadata_rejects_committed_log_size_mismatch() {
        let (
            privacy_map,
            private_column_scope,
            degree_bounds,
            sampled_points,
            fri_query_positions,
            mut opening_geometry,
        ) = geometry_sample_metadata_fixture(4, 5);
        opening_geometry[0].committed_column_log_size = 4;
        let range = ZkColumnRange::new(0, 0, 1);

        assert_eq!(
            build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
                &privacy_map,
                &private_column_scope,
                &degree_bounds,
                &sampled_points,
                &fri_query_positions,
                6,
                &opening_geometry,
                1,
            ),
            Err(ZkSampleMetadataBuildError::CommittedColumnLogSizeMismatch {
                range,
                expected: 5,
                actual: 4,
            })
        );
    }

    #[test]
    fn geometry_sample_metadata_rejects_duplicate_degree_bounds() {
        let (
            privacy_map,
            private_column_scope,
            mut degree_bounds,
            sampled_points,
            fri_query_positions,
            opening_geometry,
        ) = geometry_sample_metadata_fixture(4, 5);
        degree_bounds.push(degree_bounds[0]);
        let range = ZkColumnRange::new(0, 0, 1);

        assert_eq!(
            build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
                &privacy_map,
                &private_column_scope,
                &degree_bounds,
                &sampled_points,
                &fri_query_positions,
                6,
                &opening_geometry,
                1,
            ),
            Err(ZkSampleMetadataBuildError::DuplicatePrivateColumnDegreeBound { range })
        );
    }

    #[test]
    fn geometry_sample_metadata_rejects_missing_degree_bounds() {
        let (
            privacy_map,
            private_column_scope,
            _degree_bounds,
            sampled_points,
            fri_query_positions,
            opening_geometry,
        ) = geometry_sample_metadata_fixture(4, 5);
        let range = ZkColumnRange::new(0, 0, 1);

        assert_eq!(
            build_zk_randomizer_matrices_from_stwo_sample_metadata_with_private_scope_and_opening_geometry(
                &privacy_map,
                &private_column_scope,
                &[],
                &sampled_points,
                &fri_query_positions,
                6,
                &opening_geometry,
                1,
            ),
            Err(ZkSampleMetadataBuildError::MissingPrivateColumnDegreeBound { range })
        );
    }

    #[test]
    fn stwo_sample_metadata_builder_adds_sampled_points_and_fri_queries() {
        let range = ZkColumnRange::new(0, 0, 1);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(nonzero_hash()),
        };
        let trace_domain = CanonicCoset::new(3).coset;
        let sampled_point = CirclePoint::<SecureField>::get_point(9834759221);
        let sampled_points = TreeVec(vec![vec![vec![sampled_point]]]);
        let build = build_zk_randomizer_matrices_from_stwo_sample_metadata(
            trace_domain,
            &privacy_map,
            8,
            &sampled_points,
            &[1],
            4,
        )
        .unwrap();

        assert_eq!(build.closure.query_count_for_range(range), 5);
        assert_eq!(build.matrices.len(), 1);
        assert_eq!(build.matrices[0].query_count, 5);
        assert_eq!(build.matrices[0].randomizer_dimension, 8);
        assert_eq!(build.rank_profile.entries[0].query_count, 5);
        assert_eq!(build.rank_profile.entries[0].randomizer_dimension, 8);
    }

    #[test]
    fn stwo_sample_metadata_builder_rejects_range_spanning_multiple_columns() {
        let range = ZkColumnRange::new(0, 0, 2);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range],
            hash: ZkPrivacyMapHash(nonzero_hash()),
        };
        let sampled_points = TreeVec(vec![vec![Vec::new(), Vec::new()]]);

        assert_eq!(
            build_zk_randomizer_matrices_from_stwo_sample_metadata(
                CanonicCoset::new(3).coset,
                &privacy_map,
                8,
                &sampled_points,
                &[],
                4,
            ),
            Err(ZkSampleMetadataBuildError::NonSingletonPrivateRange { range })
        );
    }

    #[test]
    fn private_column_scope_hash_is_canonical_and_binds_usage() {
        let range0 = ZkColumnRange::new(0, 0, 1);
        let range1 = ZkColumnRange::new(0, 1, 2);
        let scope = ZkPrivateColumnScope {
            version: ZkProofVersion::V1,
            hash: zero_hash(),
            entries: vec![
                ZkPrivateColumnScopeEntry {
                    range: range1,
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                    trace_domain_log_size: 4,
                    semantic_trace_domain_log_sizes: vec![4],
                },
                ZkPrivateColumnScopeEntry {
                    range: range0,
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                    trace_domain_log_size: 3,
                    semantic_trace_domain_log_sizes: vec![3],
                },
            ],
        };
        let mut canonical_scope = scope.clone();
        canonical_scope.canonicalize();
        let mut changed_scope = canonical_scope.clone();
        changed_scope.entries[0].usage = ZkPrivateColumnUsage::Lookup;
        let mut changed_domain_scope = canonical_scope.clone();
        changed_domain_scope.entries[0].trace_domain_log_size += 1;

        assert_eq!(
            canonical_zk_private_column_scope_hash(&scope),
            canonical_zk_private_column_scope_hash(&canonical_scope)
        );
        assert_ne!(
            canonical_zk_private_column_scope_hash(&canonical_scope),
            canonical_zk_private_column_scope_hash(&changed_scope)
        );
        assert_ne!(
            canonical_zk_private_column_scope_hash(&canonical_scope),
            canonical_zk_private_column_scope_hash(&changed_domain_scope)
        );
    }

    #[test]
    fn privacy_map_hash_is_canonical_and_binds_ranges() {
        let range0 = ZkColumnRange::new(0, 0, 1);
        let range1 = ZkColumnRange::new(0, 1, 2);
        let privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range1, range0],
            hash: ZkPrivacyMapHash(zero_hash()),
        };
        let canonical_privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range0, range1],
            hash: ZkPrivacyMapHash(zero_hash()),
        };
        let changed_privacy_map = ZkPrivacyMap {
            version: ZkProofVersion::V1,
            private_columns: vec![range0],
            hash: ZkPrivacyMapHash(zero_hash()),
        };

        assert_eq!(
            canonical_zk_privacy_map_hash(&privacy_map),
            canonical_zk_privacy_map_hash(&canonical_privacy_map)
        );
        assert_ne!(
            canonical_zk_privacy_map_hash(&canonical_privacy_map),
            canonical_zk_privacy_map_hash(&changed_privacy_map)
        );
    }

    #[test]
    fn randomizer_space_hash_binds_per_column_metadata() {
        let scope_hash = nonzero_hash();
        let range0 = ZkColumnRange::new(0, 0, 1);
        let range1 = ZkColumnRange::new(0, 1, 2);
        let trace_domain = ZkCircleCosetEncoding::from(CanonicCoset::new(3).coset);
        let entries = vec![
            ZkRandomizerSpaceEntry {
                range: range1,
                trace_domain,
                semantic_trace_domains: vec![trace_domain],
                randomized_log_degree: 5,
                randomizer_dimension: 8,
            },
            ZkRandomizerSpaceEntry {
                range: range0,
                trace_domain,
                semantic_trace_domains: vec![trace_domain],
                randomized_log_degree: 5,
                randomizer_dimension: 8,
            },
        ];
        let mut canonical_entries = entries.clone();
        canonical_entries.sort_unstable();
        let mut changed_entries = canonical_entries.clone();
        changed_entries[0].randomizer_dimension += 1;

        assert_eq!(
            canonical_zk_randomizer_space_hash(scope_hash, &entries),
            canonical_zk_randomizer_space_hash(scope_hash, &canonical_entries)
        );
        assert_ne!(
            canonical_zk_randomizer_space_hash(scope_hash, &canonical_entries),
            canonical_zk_randomizer_space_hash(scope_hash, &changed_entries)
        );
    }

    #[test]
    fn split_derivation_hash_binds_split_parameter() {
        assert_ne!(
            canonical_zk_split_derivation_hash(1),
            canonical_zk_split_derivation_hash(2)
        );
    }

    fn quotient_split_mask_profile() -> ZkQuotientSplitMaskProfile {
        ZkQuotientSplitMaskProfile {
            split_index: 0,
            split_identity_log_degree_bound: 6,
            split_mask_log_degree_bound: 5,
            h_split: 32,
            left_range: ZkColumnRange::new(1, 0, SECURE_EXTENSION_DEGREE),
            right_range: ZkColumnRange::new(
                1,
                SECURE_EXTENSION_DEGREE,
                2 * SECURE_EXTENSION_DEGREE,
            ),
            left_masked_log_degree_bound: 6,
            right_masked_log_degree_bound: 5,
        }
    }

    #[test]
    fn quotient_split_mask_profile_accepts_stwo_cancellation_bounds() {
        assert_eq!(
            validate_zk_quotient_split_mask_profile(quotient_split_mask_profile()),
            Ok(())
        );
    }

    #[test]
    fn stwo_composition_quotient_split_mask_profile_binds_split_geometry() {
        let profile = stwo_composition_quotient_split_mask_profile(3, 6, 5, 32, 6, 5).unwrap();

        assert_eq!(profile.split_index, 0);
        assert_eq!(
            profile.left_range,
            ZkColumnRange::new(3, 0, SECURE_EXTENSION_DEGREE)
        );
        assert_eq!(
            profile.right_range,
            ZkColumnRange::new(3, SECURE_EXTENSION_DEGREE, 2 * SECURE_EXTENSION_DEGREE)
        );
    }

    #[test]
    fn quotient_split_mask_profile_hash_binds_geometry_and_bounds() {
        let profile = quotient_split_mask_profile();
        let mut changed_tree = profile;
        changed_tree.left_range = ZkColumnRange::new(2, 0, SECURE_EXTENSION_DEGREE);
        changed_tree.right_range =
            ZkColumnRange::new(2, SECURE_EXTENSION_DEGREE, 2 * SECURE_EXTENSION_DEGREE);
        let mut changed_h_split = profile;
        changed_h_split.h_split -= 1;
        let mut changed_identity_bound = profile;
        changed_identity_bound.split_identity_log_degree_bound += 1;

        assert_ne!(
            canonical_zk_quotient_split_mask_profile_hash(profile),
            canonical_zk_quotient_split_mask_profile_hash(changed_tree)
        );
        assert_ne!(
            canonical_zk_quotient_split_mask_profile_hash(profile),
            canonical_zk_quotient_split_mask_profile_hash(changed_h_split)
        );
        assert_ne!(
            canonical_zk_quotient_split_mask_profile_hash(profile),
            canonical_zk_quotient_split_mask_profile_hash(changed_identity_bound)
        );
    }

    #[test]
    fn quotient_split_mask_profile_transcript_mixing_binds_profile_hash() {
        let profile = quotient_split_mask_profile();
        let mut changed = profile;
        changed.right_masked_log_degree_bound += 1;
        let mut channel = Blake2sChannel::default();
        let mut changed_channel = Blake2sChannel::default();

        mix_zk_quotient_split_mask_profile(&mut channel, profile).unwrap();
        mix_zk_quotient_split_mask_profile(&mut changed_channel, changed).unwrap();

        assert_ne!(
            channel.draw_secure_felt(),
            changed_channel.draw_secure_felt()
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_empty_entropy() {
        let mut profile = quotient_split_mask_profile();
        profile.h_split = 0;

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(ZkQuotientSplitMaskProfileValidationError::EmptySplitMask)
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_empty_range() {
        let mut profile = quotient_split_mask_profile();
        profile.left_range = ZkColumnRange::new(1, 0, 0);

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(ZkQuotientSplitMaskProfileValidationError::EmptySplitRange {
                range: profile.left_range,
            })
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_tree_mismatch() {
        let mut profile = quotient_split_mask_profile();
        profile.right_range =
            ZkColumnRange::new(2, SECURE_EXTENSION_DEGREE, 2 * SECURE_EXTENSION_DEGREE);

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::SplitRangeTreeMismatch {
                    left_range: profile.left_range,
                    right_range: profile.right_range,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_width_mismatch() {
        let mut profile = quotient_split_mask_profile();
        profile.right_range = ZkColumnRange::new(1, SECURE_EXTENSION_DEGREE, 9);

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::SplitRangeWidthMismatch {
                    left_range: profile.left_range,
                    right_range: profile.right_range,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_noncontiguous_ranges() {
        let mut profile = quotient_split_mask_profile();
        profile.right_range = ZkColumnRange::new(1, SECURE_EXTENSION_DEGREE + 1, 9);

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::NonContiguousSplitRanges {
                    left_range: profile.left_range,
                    right_range: profile.right_range,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_split_identity_underflow() {
        let mut profile = quotient_split_mask_profile();
        profile.split_identity_log_degree_bound = 1;

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::SplitIdentityUnderflow {
                    split_identity_log_degree_bound: 1,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_mask_above_split_component_bound() {
        let mut profile = quotient_split_mask_profile();
        profile.split_mask_log_degree_bound = profile.split_identity_log_degree_bound;

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::SplitMaskDegreeExceedsSplitComponent {
                    split_mask_log_degree_bound: 6,
                    split_component_log_degree_bound: 5,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_mask_dimension_overflow() {
        let mut profile = quotient_split_mask_profile();
        profile.split_identity_log_degree_bound = 66;
        profile.split_mask_log_degree_bound = 64;
        profile.left_masked_log_degree_bound = 66;
        profile.right_masked_log_degree_bound = 65;

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::SplitMaskDimensionTooLarge {
                    split_mask_log_degree_bound: 64,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_entropy_above_mask_dimension() {
        let mut profile = quotient_split_mask_profile();
        profile.h_split = 33;

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::SplitMaskEntropyTooLarge {
                    h_split: 33,
                    max: 32,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_profile_rejects_bounds_below_cancellation_requirements() {
        let mut profile = quotient_split_mask_profile();
        profile.left_masked_log_degree_bound = 5;

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::LeftMaskedBoundBelowCancellationProduct {
                    required: 6,
                    actual: 5,
                }
            )
        );

        let mut profile = quotient_split_mask_profile();
        profile.right_masked_log_degree_bound = 4;

        assert_eq!(
            validate_zk_quotient_split_mask_profile(profile),
            Err(
                ZkQuotientSplitMaskProfileValidationError::RightMaskedBoundBelowOriginalSplit {
                    required: 5,
                    actual: 4,
                }
            )
        );
    }

    #[test]
    fn quotient_split_mask_query_budget_accepts_conservative_openings() {
        assert_eq!(
            validate_zk_quotient_split_mask_query_budget(quotient_split_mask_profile(), 15),
            Ok(())
        );
    }

    #[test]
    fn quotient_split_mask_query_budget_rejects_insufficient_mask_dimension() {
        assert_eq!(
            validate_zk_quotient_split_mask_query_budget(quotient_split_mask_profile(), 16),
            Err(
                ZkQuotientSplitMaskQueryBudgetError::InsufficientMaskDimension {
                    required: 33,
                    actual: 32,
                }
            )
        );
    }

    #[test]
    fn fri_batch_mask_query_positions_avoid_small_h_batch_fri_opened_blocks() {
        let mut channel = Blake2sChannel::default();
        let positions =
            zk_fri_batch_mask_query_positions(&mut channel, 8, 8, &[0], FriConfig::new(0, 1, 8, 1))
                .unwrap();

        assert_eq!(positions.len(), 8);
        assert!(positions.iter().all(|position| *position >= 32));
    }

    #[test]
    fn fri_batch_mask_query_positions_reject_insufficient_safe_domain() {
        let mut channel = Blake2sChannel::default();

        assert_eq!(
            zk_fri_batch_mask_query_positions(
                &mut channel,
                4,
                16,
                &[0, 15],
                FriConfig::new(0, 1, 16, 1),
            ),
            Err(ZkFriBatchMaskQueryPositionsError::InsufficientPositions {
                required: 16,
                available: 0,
            })
        );
    }

    struct TestPrivacyProvider {
        complete: bool,
        dependencies: Vec<ZkPrivacyDependency>,
        logup_claim_metadata_complete: bool,
        logup_claim_manifest: Vec<ZkLogupClaimManifestEntry>,
        logup_claim_policies: Vec<ZkLogupClaimPolicy>,
        logup_statistical_aggregate_groups: Vec<ZkLogupStatisticalAggregateGroup>,
    }

    fn test_logup_claim_policy() -> ZkLogupClaimPolicy {
        ZkLogupClaimPolicy {
            interaction_index: 0,
            claim_index: 0,
            visibility: ZkLogupClaimVisibility::SemanticallyPublic,
            semantic_domain: b"test.logup.claim.public.v1".to_vec(),
            semantic_statement: b"test fixture declares this scalar public".to_vec(),
        }
    }

    fn test_statistical_logup_claim_policy() -> ZkLogupClaimPolicy {
        ZkLogupClaimPolicy {
            interaction_index: 0,
            claim_index: 0,
            visibility: ZkLogupClaimVisibility::StatisticalAggregate { aggregate_id: 7 },
            semantic_domain: vec![],
            semantic_statement: vec![],
        }
    }

    fn test_logup_statistical_aggregate_group(
        min_statistical_security_bits: u32,
    ) -> ZkLogupStatisticalAggregateGroup {
        ZkLogupStatisticalAggregateGroup {
            aggregate_id: 7,
            target: ZkLogupStatisticalAggregateTarget::Zero,
            relation_domain: b"test.logup.aggregate.relation.v1".to_vec(),
            relation_statement: b"private send receive aggregate sums to zero".to_vec(),
            private_lookup_term_count_bound: 16,
            lookup_challenge_count: 2,
            expected_proof_volume: 1024,
            min_statistical_security_bits,
            safety_margin_bits: 8,
        }
    }

    fn complete_test_privacy_provider(
        dependencies: Vec<ZkPrivacyDependency>,
    ) -> TestPrivacyProvider {
        TestPrivacyProvider {
            complete: true,
            dependencies,
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![test_logup_claim_policy()],
            logup_statistical_aggregate_groups: vec![],
        }
    }

    impl ZkAirPrivacyProvider for TestPrivacyProvider {
        fn air_id(&self) -> ZkAirId {
            ZkAirId(b"test.air.v1".to_vec())
        }

        fn component_column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
            TreeVec::new(vec![vec![2], vec![2, 2], vec![2]])
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            3
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
            vec![ZkColumnRange::new(0, 0, 1)]
        }

        fn private_roots(&self) -> Vec<ZkPrivateRoot> {
            vec![ZkPrivateRoot {
                range: ZkColumnRange::new(1, 0, 2),
                usage: ZkPrivateColumnUsage::OrdinaryWitness,
                reason: ZkPrivacyReason::Witness,
            }]
        }

        fn dependency_edges(&self) -> Vec<ZkPrivacyDependency> {
            self.dependencies.clone()
        }

        fn dependency_metadata_completeness(&self) -> ZkDependencyMetadataCompleteness {
            if self.complete {
                ZkDependencyMetadataCompleteness::CompleteTraceAndInteractionClosure
            } else {
                ZkDependencyMetadataCompleteness::Incomplete
            }
        }

        fn logup_claim_policies(&self) -> Vec<ZkLogupClaimPolicy> {
            self.logup_claim_policies.clone()
        }

        fn logup_claim_manifest(&self) -> Vec<ZkLogupClaimManifestEntry> {
            self.logup_claim_manifest.clone()
        }

        fn logup_statistical_aggregate_groups(&self) -> Vec<ZkLogupStatisticalAggregateGroup> {
            self.logup_statistical_aggregate_groups.clone()
        }

        fn logup_claim_metadata_completeness(&self) -> ZkLogupClaimMetadataCompleteness {
            if self.logup_claim_metadata_complete {
                ZkLogupClaimMetadataCompleteness::Complete
            } else {
                ZkLogupClaimMetadataCompleteness::Incomplete
            }
        }

        fn application_domain(&self) -> &[u8] {
            b"test.zk.domain"
        }

        fn application_statement(&self) -> Vec<u8> {
            b"test-public-statement".to_vec()
        }
    }

    #[test]
    fn privacy_provider_closure_marks_logup_interaction_private() {
        let provider = complete_test_privacy_provider(vec![ZkPrivacyDependency {
            from: ZkColumnRange::new(1, 0, 2),
            to: ZkColumnRange::new(2, 0, 1),
            kind: ZkDependencyKind::LogUpRunningSum,
        }]);

        let metadata = build_zk_air_metadata_from_privacy_provider(
            &provider,
            1,
            ZkPrivacyInferenceMode::FailClosed,
        )
        .unwrap();

        assert!(metadata.private_column_scope.entries.iter().any(|entry| {
            entry.range == ZkColumnRange::new(1, 0, 1)
                && entry.usage == ZkPrivateColumnUsage::OrdinaryWitness
        }));
        assert!(metadata.private_column_scope.entries.iter().any(|entry| {
            entry.range == ZkColumnRange::new(1, 1, 2)
                && entry.usage == ZkPrivateColumnUsage::OrdinaryWitness
        }));
        assert!(metadata.private_column_scope.entries.iter().any(|entry| {
            entry.range == ZkColumnRange::new(2, 0, 1) && entry.usage == ZkPrivateColumnUsage::LogUp
        }));
        assert_eq!(
            metadata.logup_claim_metadata_completeness,
            ZkLogupClaimMetadataCompleteness::Complete
        );
        assert_eq!(
            metadata.logup_claim_manifest,
            vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }]
        );
        assert_eq!(
            metadata.logup_claim_policies,
            vec![test_logup_claim_policy()]
        );
    }

    #[test]
    fn privacy_provider_fail_closed_rejects_incomplete_dependency_metadata() {
        let provider = TestPrivacyProvider {
            complete: false,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![test_logup_claim_policy()],
            logup_statistical_aggregate_groups: vec![],
        };

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::IncompleteDependencyMetadata)
        ));
    }

    #[test]
    fn privacy_provider_relaxed_modes_still_reject_incomplete_dependency_metadata() {
        let provider = TestPrivacyProvider {
            complete: false,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![test_logup_claim_policy()],
            logup_statistical_aggregate_groups: vec![],
        };

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::UseDeclaredDependenciesOnly
            ),
            Err(ZkAirMetadataBuildError::IncompleteDependencyMetadata)
        ));
        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::MarkAllInteractionDerivedPrivate
            ),
            Err(ZkAirMetadataBuildError::IncompleteDependencyMetadata)
        ));
    }

    #[test]
    fn privacy_provider_rejects_invalid_dependency_target_even_if_source_is_public() {
        let provider = complete_test_privacy_provider(vec![ZkPrivacyDependency {
            from: ZkColumnRange::new(0, 0, 1),
            to: ZkColumnRange::new(9, 0, 1),
            kind: ZkDependencyKind::AirConstraint,
        }]);

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::InvalidPrivacyDependencyRange {
                range: ZkColumnRange {
                    tree_index: 9,
                    column_start: 0,
                    column_end: 1,
                },
            })
        ));
    }

    #[test]
    fn privacy_provider_rejects_unclassified_air_columns() {
        struct PartialProvider;

        impl ZkAirPrivacyProvider for PartialProvider {
            fn air_id(&self) -> ZkAirId {
                ZkAirId(b"partial.test.air.v1".to_vec())
            }

            fn component_column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
                TreeVec::new(vec![vec![2], vec![2, 2]])
            }

            fn max_constraint_log_degree_bound(&self) -> u32 {
                3
            }

            fn trace_tree_scopes(&self) -> Vec<ZkTraceTreeScope> {
                vec![
                    ZkTraceTreeScope::Preprocessed,
                    ZkTraceTreeScope::OriginalTrace,
                ]
            }

            fn public_roots(&self) -> Vec<ZkColumnRange> {
                vec![ZkColumnRange::new(0, 0, 1)]
            }

            fn private_roots(&self) -> Vec<ZkPrivateRoot> {
                vec![ZkPrivateRoot {
                    range: ZkColumnRange::new(1, 0, 1),
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                    reason: ZkPrivacyReason::Witness,
                }]
            }

            fn dependency_edges(&self) -> Vec<ZkPrivacyDependency> {
                vec![]
            }

            fn dependency_metadata_completeness(&self) -> ZkDependencyMetadataCompleteness {
                ZkDependencyMetadataCompleteness::CompleteTraceAndInteractionClosure
            }

            fn application_domain(&self) -> &[u8] {
                b"partial.test.zk.domain"
            }

            fn application_statement(&self) -> Vec<u8> {
                b"partial-test-public-statement".to_vec()
            }
        }

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &PartialProvider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::UnclassifiedAirColumn {
                range: ZkColumnRange {
                    tree_index: 1,
                    column_start: 1,
                    column_end: 2,
                },
            })
        ));
    }

    #[test]
    fn privacy_provider_rejects_missing_private_interaction_closure() {
        let provider = complete_test_privacy_provider(vec![]);

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::MissingPrivateInteractionColumn {
                range: ZkColumnRange {
                    tree_index: 2,
                    column_start: 0,
                    column_end: 1,
                },
            })
        ));
    }

    #[test]
    fn privacy_provider_rejects_private_logup_without_claim_metadata() {
        let provider = TestPrivacyProvider {
            complete: true,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: false,
            logup_claim_manifest: vec![],
            logup_claim_policies: vec![],
            logup_statistical_aggregate_groups: vec![],
        };

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::IncompleteLogupClaimMetadata)
        ));
    }

    #[test]
    fn privacy_provider_rejects_private_logup_without_claim_for_interaction() {
        let provider = TestPrivacyProvider {
            complete: true,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![],
            logup_claim_policies: vec![],
            logup_statistical_aggregate_groups: vec![],
        };

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::MissingPrivateLogupClaimManifest {
                interaction_index: 0,
            })
        ));
    }

    #[test]
    fn privacy_provider_rejects_private_unsupported_logup_claim() {
        let mut policy = test_logup_claim_policy();
        policy.visibility = ZkLogupClaimVisibility::PrivateUnsupported;
        let provider = TestPrivacyProvider {
            complete: true,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![policy],
            logup_statistical_aggregate_groups: vec![],
        };

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(
                ZkAirMetadataBuildError::UnsupportedPrivateLogupClaimPolicy {
                    interaction_index: 0,
                    claim_index: 0,
                }
            )
        ));
    }

    #[test]
    fn privacy_provider_rejects_wrong_logup_claim_index() {
        let mut policy = test_logup_claim_policy();
        policy.claim_index = 999;
        let provider = TestPrivacyProvider {
            complete: true,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![policy],
            logup_statistical_aggregate_groups: vec![],
        };

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::InvalidLogupClaimPolicy {
                interaction_index: 0,
                claim_index: 999,
            })
        ));
    }

    #[test]
    fn privacy_provider_rejects_stale_logup_claim_policy_without_private_logup() {
        struct PublicOnlyProvider;

        impl ZkAirPrivacyProvider for PublicOnlyProvider {
            fn air_id(&self) -> ZkAirId {
                ZkAirId(b"public-only.test.air.v1".to_vec())
            }

            fn component_column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
                TreeVec::new(vec![vec![2]])
            }

            fn max_constraint_log_degree_bound(&self) -> u32 {
                3
            }

            fn trace_tree_scopes(&self) -> Vec<ZkTraceTreeScope> {
                vec![ZkTraceTreeScope::Preprocessed]
            }

            fn public_roots(&self) -> Vec<ZkColumnRange> {
                vec![ZkColumnRange::new(0, 0, 1)]
            }

            fn private_roots(&self) -> Vec<ZkPrivateRoot> {
                vec![]
            }

            fn dependency_edges(&self) -> Vec<ZkPrivacyDependency> {
                vec![]
            }

            fn logup_claim_policies(&self) -> Vec<ZkLogupClaimPolicy> {
                vec![test_logup_claim_policy()]
            }

            fn logup_claim_metadata_completeness(&self) -> ZkLogupClaimMetadataCompleteness {
                ZkLogupClaimMetadataCompleteness::Complete
            }

            fn application_domain(&self) -> &[u8] {
                b"public-only.test.zk.domain"
            }

            fn application_statement(&self) -> Vec<u8> {
                b"public-only-test".to_vec()
            }
        }

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &PublicOnlyProvider,
                1,
                ZkPrivacyInferenceMode::FailClosed
            ),
            Err(ZkAirMetadataBuildError::InvalidLogupClaimPolicy {
                interaction_index: 0,
                claim_index: 0,
            })
        ));
    }

    #[test]
    fn privacy_provider_binds_logup_claim_policy_to_public_statement() {
        let provider = complete_test_privacy_provider(vec![ZkPrivacyDependency {
            from: ZkColumnRange::new(1, 0, 2),
            to: ZkColumnRange::new(2, 0, 1),
            kind: ZkDependencyKind::LogUpRunningSum,
        }]);
        let mut changed_policy = test_logup_claim_policy();
        changed_policy.semantic_statement = b"changed semantic public rationale".to_vec();
        let changed_provider = TestPrivacyProvider {
            complete: true,
            dependencies: provider.dependencies.clone(),
            logup_claim_metadata_complete: true,
            logup_claim_manifest: provider.logup_claim_manifest.clone(),
            logup_claim_policies: vec![changed_policy],
            logup_statistical_aggregate_groups: vec![],
        };

        let metadata = build_zk_air_metadata_from_privacy_provider(
            &provider,
            1,
            ZkPrivacyInferenceMode::FailClosed,
        )
        .unwrap();
        let changed_metadata = build_zk_air_metadata_from_privacy_provider(
            &changed_provider,
            1,
            ZkPrivacyInferenceMode::FailClosed,
        )
        .unwrap();

        assert_ne!(
            metadata.public_statement_hash,
            changed_metadata.public_statement_hash
        );
    }

    #[test]
    fn privacy_provider_accepts_statistical_logup_aggregate_metadata() {
        let provider = TestPrivacyProvider {
            complete: true,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![test_statistical_logup_claim_policy()],
            logup_statistical_aggregate_groups: vec![test_logup_statistical_aggregate_group(100)],
        };

        let metadata = build_zk_air_metadata_from_privacy_provider(
            &provider,
            1,
            ZkPrivacyInferenceMode::FailClosed,
        )
        .unwrap();
        assert_eq!(
            metadata.logup_statistical_aggregate_groups,
            vec![test_logup_statistical_aggregate_group(100)]
        );
        assert_eq!(
            metadata.logup_statistical_security_budgets,
            vec![ZkLogupStatisticalSecurityBudget {
                aggregate_id: 7,
                extension_field_bits: ZK_QM31_STATISTICAL_EXTENSION_FIELD_BITS,
                private_lookup_term_count_bound: 16,
                lookup_challenge_count: 2,
                expected_proof_volume: 1024,
                safety_margin_bits: 8,
                computed_security_bits: 101,
                min_statistical_security_bits: 100,
            }]
        );

        let artifacts = build_zk_config_from_air_privacy_provider(
            &provider,
            1,
            ZkPrivacyInferenceMode::FailClosed,
        )
        .unwrap();
        assert_eq!(
            artifacts.verifier_config.logup_statistical_security_budgets,
            metadata.logup_statistical_security_budgets
        );
        assert_eq!(
            validate_zk_public_metadata_against_verifier_config(
                &artifacts.metadata,
                &artifacts.verifier_config,
            ),
            Ok(())
        );
    }

    #[test]
    fn privacy_provider_rejects_underbudget_statistical_logup_aggregate() {
        let provider = TestPrivacyProvider {
            complete: true,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![test_statistical_logup_claim_policy()],
            logup_statistical_aggregate_groups: vec![test_logup_statistical_aggregate_group(102)],
        };

        assert!(matches!(
            build_zk_air_metadata_from_privacy_provider(
                &provider,
                1,
                ZkPrivacyInferenceMode::FailClosed,
            ),
            Err(
                ZkAirMetadataBuildError::InsufficientLogupStatisticalSecurity {
                    aggregate_id: 7,
                    computed_bits: 101,
                    min_bits: 102,
                }
            )
        ));
    }

    #[test]
    fn privacy_provider_binds_statistical_logup_aggregate_to_public_statement() {
        let provider = TestPrivacyProvider {
            complete: true,
            dependencies: vec![ZkPrivacyDependency {
                from: ZkColumnRange::new(1, 0, 2),
                to: ZkColumnRange::new(2, 0, 1),
                kind: ZkDependencyKind::LogUpRunningSum,
            }],
            logup_claim_metadata_complete: true,
            logup_claim_manifest: vec![ZkLogupClaimManifestEntry {
                interaction_index: 0,
                claim_count: 1,
            }],
            logup_claim_policies: vec![test_statistical_logup_claim_policy()],
            logup_statistical_aggregate_groups: vec![test_logup_statistical_aggregate_group(100)],
        };
        let mut changed_group = test_logup_statistical_aggregate_group(100);
        changed_group.private_lookup_term_count_bound += 1;
        let changed_provider = TestPrivacyProvider {
            complete: true,
            dependencies: provider.dependencies.clone(),
            logup_claim_metadata_complete: true,
            logup_claim_manifest: provider.logup_claim_manifest.clone(),
            logup_claim_policies: provider.logup_claim_policies.clone(),
            logup_statistical_aggregate_groups: vec![changed_group],
        };

        let metadata = build_zk_air_metadata_from_privacy_provider(
            &provider,
            1,
            ZkPrivacyInferenceMode::FailClosed,
        )
        .unwrap();
        let changed_metadata = build_zk_air_metadata_from_privacy_provider(
            &changed_provider,
            1,
            ZkPrivacyInferenceMode::FailClosed,
        )
        .unwrap();

        assert_ne!(
            metadata.public_statement_hash,
            changed_metadata.public_statement_hash
        );
    }
}
