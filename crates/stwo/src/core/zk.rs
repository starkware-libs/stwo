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
use crate::core::pcs::utils::TreeVec;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
use crate::core::poly::utils::get_folding_alphas;
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
        matches!(self, Self::OrdinaryWitness)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkPrivateColumnScopeEntry {
    pub range: ZkColumnRange,
    pub usage: ZkPrivateColumnUsage,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ZkRandomizerSpaceEntry {
    pub range: ZkColumnRange,
    pub trace_domain: ZkCircleCosetEncoding,
    pub randomized_log_degree: u32,
    pub randomizer_dimension: u64,
}

const ZK_PRIVATE_COLUMN_SCOPE_HASH_DOMAIN: &[u8] = b"stwo.zk.private-column-scope.v1";
const ZK_RANDOMIZER_SPACE_HASH_DOMAIN: &[u8] = b"stwo.zk.randomizer-space.v1";
const ZK_SPLIT_DERIVATION_HASH_DOMAIN: &[u8] = b"stwo.zk.split-derivation.v1";
const ZK_RANDOMIZER_BASIS_ID: &[u8] = b"circle-fft-bit-reversed";
const ZK_RANDOMIZER_CONSTRUCTION_ID: &[u8] = b"eval-coset-vanishing-times-r-interpolate";
const ZK_RANDOMIZER_RANK_MATRIX_ID: &[u8] = b"base-field-functional-matrix-v1";
const ZK_SPLIT_IDENTITY: &[u8] = b"p(z)=left(z)+pi^(L-2)(z.x)*right(z)";
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

fn blake2s_hash(bytes: &[u8]) -> [u8; 32] {
    Blake2sHasher::hash(bytes).into()
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
        push_u32(&mut bytes, entry.randomized_log_degree);
        push_u64(&mut bytes, entry.randomizer_dimension);
        push_tag(&mut bytes, ZK_RANDOMIZER_BASIS_ID);
        push_tag(&mut bytes, ZK_RANDOMIZER_CONSTRUCTION_ID);
        push_tag(&mut bytes, ZK_RANDOMIZER_RANK_MATRIX_ID);
    }
    blake2s_hash(&bytes)
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
            let expected_domain_id = metadata.degree_profile.fri_first_layer_log_size as u64;
            if entry.domain_id != expected_domain_id {
                return Err(ZkQueryClosureValidationError::QueryDomainMismatch {
                    entry,
                    expected: expected_domain_id,
                    actual: entry.domain_id,
                });
            }
            if expected_domain_id >= u64::BITS as u64 {
                return Err(ZkQueryClosureValidationError::QueryDomainTooLarge {
                    entry,
                    log_size: expected_domain_id,
                });
            }
            let domain_size = 1u64 << expected_domain_id;
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

        let expected_dimension = metadata.witness_randomization.h_witness;
        if entry.randomizer_dimension != expected_dimension {
            return Err(
                ZkRandomizerRankValidationError::RandomizerDimensionMismatch {
                    range,
                    expected: expected_dimension,
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
    NonSingletonPrivateRange { range: ZkColumnRange },
    MissingTree { tree_index: usize },
    MissingColumn { range: ZkColumnRange },
    QueryPositionOutOfDomain { position: usize, domain_size: usize },
    LiftingDomainTooLarge { log_size: u32 },
    RandomizerMatrix(ZkRandomizerMatrixBuildError),
}

pub fn build_zk_randomizer_matrices_from_stwo_sample_metadata(
    trace_domain: Coset,
    privacy_map: &ZkPrivacyMap,
    randomizer_dimension: u64,
    sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    fri_query_positions: &[usize],
    lifting_log_size: u32,
) -> Result<ZkRandomizerMatrixBuild, ZkSampleMetadataBuildError> {
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
            for coordinate_index in 0..4 {
                functionals.push(ZkRandomizerQueryFunctional {
                    range,
                    kind: ZkQueryClosureKind::OodsExtension,
                    domain_id: range.tree_index as u64,
                    point_or_position: point_encoding,
                    point,
                    coordinate_index,
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

    build_zk_randomizer_matrices_for_witness_randomization(
        trace_domain,
        privacy_map,
        randomizer_dimension,
        &functionals,
    )
    .map_err(ZkSampleMetadataBuildError::RandomizerMatrix)
}

pub fn build_zk_randomizer_matrices_for_witness_randomization(
    trace_domain: Coset,
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
                    trace_domain,
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
    trace_domain: Coset,
    randomizer_dimension: u64,
    ambient_log_dimension: usize,
    functional: ZkRandomizerQueryFunctional,
) -> Vec<BaseField> {
    let trace_domain = zk_trace_domain_half_coset(trace_domain);
    let vanishing = coset_vanishing(trace_domain, functional.point)
        * coset_vanishing(trace_domain.conjugate(), functional.point);
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

/// Public metadata echoed by a ZK proof and compared against verifier-owned
/// configuration before affected Fiat-Shamir challenges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPublicMetadata {
    pub version: ZkProofVersion,
    pub privacy_map_hash: ZkPrivacyMapHash,
    pub public_statement_hash: ZkPublicStatementHash,
    pub degree_profile: ZkDegreeProfile,
    pub witness_randomization: ZkWitnessRandomizationProfile,
    pub quotient_integration: ZkQuotientIntegrationProfile,
}

/// Verifier-owned ZK configuration. Verification trusts this configuration,
/// not the proof's echoed metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkVerificationConfig {
    pub metadata: ZkPublicMetadata,
    pub column_degree_bounds: Vec<ZkColumnDegreeBound>,
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

#[derive(Clone, Debug)]
pub struct ZkStarkDegreeBoundProfile {
    pub column_log_degree_bounds: TreeVec<ColumnVec<u32>>,
    pub trace_log_degree_bound: u32,
    pub composition_log_degree_bound: u32,
    pub split_composition_log_degree_bound: u32,
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
    QuotientDegreeBoundShrinksComposition {
        normal_split_composition_log_degree_bound: u32,
        zk_split_composition_log_degree_bound: u32,
    },
    FriFirstLayerTooSmall {
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
    validate_zk_quotient_degree_bounds_for_stark_profile(
        &metadata.quotient_integration.quotient_degree_bounds,
        expected_quotient_range,
    )?;

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
    if metadata.degree_profile.fri_first_layer_log_size
        != metadata.quotient_integration.fri_first_layer_log_size
    {
        return Err(ZkStarkDegreeBoundProfileError::FriFirstLayerMismatch {
            degree_profile: metadata.degree_profile.fri_first_layer_log_size,
            quotient_integration: metadata.quotient_integration.fri_first_layer_log_size,
        });
    }
    let required_fri_first_layer_log_size = zk_split_composition_log_degree_bound
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
        composition_log_degree_bound: zk_split_composition_log_degree_bound
            .checked_add(composition_log_split)
            .ok_or(ZkStarkDegreeBoundProfileError::CompositionDegreeOverflow {
                split_composition_log_degree_bound: zk_split_composition_log_degree_bound,
                composition_log_split,
            })?,
        split_composition_log_degree_bound: zk_split_composition_log_degree_bound,
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
    let randomizer_coefficient_count = metadata
        .witness_randomization
        .h_witness
        .checked_next_power_of_two()
        .ok_or(
            ZkMetadataValidationError::WitnessRandomizerDimensionTooLarge {
                dimension: metadata.witness_randomization.h_witness,
            },
        )?;
    let trace_domain_size = 1u64
        .checked_shl(metadata.degree_profile.trace_domain_log_size)
        .ok_or(
            ZkMetadataValidationError::WitnessRandomizerDimensionTooLarge {
                dimension: metadata.witness_randomization.h_witness,
            },
        )?;
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
        if bound.log_degree_bound <= metadata.degree_profile.trace_domain_log_size {
            return Err(
                ZkMetadataValidationError::WitnessRandomizedDomainNotLarger {
                    trace_domain_log_size: metadata.degree_profile.trace_domain_log_size,
                    randomized_log_degree: bound.log_degree_bound,
                },
            );
        }
        let randomized_domain_size = 1u64.checked_shl(bound.log_degree_bound).ok_or(
            ZkMetadataValidationError::WitnessRandomizerDimensionTooLarge {
                dimension: metadata.witness_randomization.h_witness,
            },
        )?;
        if trace_domain_size
            .checked_add(randomizer_coefficient_count)
            .is_none_or(|required| required > randomized_domain_size)
        {
            return Err(ZkMetadataValidationError::WitnessRandomizedDomainTooSmall {
                trace_domain_size,
                randomizer_coefficient_count,
                randomized_domain_size,
            });
        }
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
        channel.mix_u64(bound.range.tree_index as u64);
        channel.mix_u64(bound.range.column_start as u64);
        channel.mix_u64(bound.range.column_end as u64);
        channel.mix_u32s(&[bound.log_degree_bound]);
    }
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkStarkProof<H: MerkleHasherLifted>(pub ZkCommitmentSchemeProof<H>);

impl<H: MerkleHasherLifted> ZkStarkProof<H> {
    /// Extracts the randomized composition trace Out-Of-Domain-Sample
    /// evaluation from the ZK PCS sampled values.
    pub(crate) fn extract_composition_oods_eval(
        &self,
        oods_point: CirclePoint<SecureField>,
        max_log_degree_bound: u32,
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
        let value = left_eval + oods_point.repeated_double(max_log_degree_bound - 1).x * right_eval;
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
        ZkVerificationConfig {
            metadata,
            column_degree_bounds,
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

    fn witness_metadata(lifting_log_size: u32, log_blowup_factor: u32) -> ZkPublicMetadata {
        let h_batch =
            expected_zk_fri_batch_degree_bound(lifting_log_size, log_blowup_factor).unwrap();

        ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash: ZkPrivacyMapHash(nonzero_hash()),
            public_statement_hash: ZkPublicStatementHash(nonzero_hash()),
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
                split_derivation_hash: nonzero_hash(),
                quotient_degree_bounds: vec![ZkColumnDegreeBound {
                    range: ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE),
                    log_degree_bound: lifting_log_size - log_blowup_factor,
                }],
            },
        }
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
        let metadata = witness_metadata(6, 1);
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

        assert_eq!(profile.column_log_degree_bounds.0, vec![vec![5], vec![6]]);
        assert_eq!(profile.trace_log_degree_bound, 6);
        assert_eq!(profile.split_composition_log_degree_bound, 5);
        assert_eq!(profile.composition_log_degree_bound, 6);
        assert_eq!(profile.fri_first_layer_log_size, 6);
    }

    #[test]
    fn zk_stark_degree_profile_accepts_unsplit_quotient_width() {
        let mut metadata = witness_metadata(6, 1);
        metadata.quotient_integration.quotient_degree_bounds[0].range =
            ZkColumnRange::new(2, 0, SECURE_EXTENSION_DEGREE);
        let verifier_config = verification_config(metadata);
        let profile = derive_zk_stark_degree_bound_profile(
            TreeVec(vec![vec![5], vec![5]]),
            5,
            0,
            &verifier_config,
            6,
            1,
        )
        .unwrap();

        assert_eq!(profile.column_log_degree_bounds.0, vec![vec![5], vec![6]]);
        assert_eq!(profile.trace_log_degree_bound, 6);
        assert_eq!(profile.split_composition_log_degree_bound, 5);
        assert_eq!(profile.composition_log_degree_bound, 5);
        assert_eq!(profile.fri_first_layer_log_size, 6);
    }

    #[test]
    fn zk_stark_degree_profile_accepts_wider_quotient_split_width() {
        let mut metadata = witness_metadata(7, 1);
        metadata.quotient_integration.quotient_degree_bounds[0].range =
            ZkColumnRange::new(2, 0, 4 * SECURE_EXTENSION_DEGREE);
        let verifier_config = verification_config(metadata);
        let profile = derive_zk_stark_degree_bound_profile(
            TreeVec(vec![vec![5], vec![5]]),
            6,
            2,
            &verifier_config,
            7,
            1,
        )
        .unwrap();

        assert_eq!(profile.column_log_degree_bounds.0, vec![vec![5], vec![7]]);
        assert_eq!(profile.trace_log_degree_bound, 7);
        assert_eq!(profile.split_composition_log_degree_bound, 6);
        assert_eq!(profile.composition_log_degree_bound, 8);
        assert_eq!(profile.fri_first_layer_log_size, 7);
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
            ZkStarkDegreeBoundProfileError::Metadata(
                ZkMetadataValidationError::InvalidColumnDegreeBoundRange {
                    range: ZkColumnRange::new(2, 0, 0),
                },
            )
        );
    }

    #[test]
    fn zk_stark_degree_profile_rejects_small_fri_layer() {
        let mut metadata = witness_metadata(5, 1);
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
                required: 6,
                actual: 5,
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
        let metadata = witness_metadata(7, 1);
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

        assert_eq!(profile.column_log_degree_bounds.0, vec![vec![5], vec![7]]);
        assert_eq!(profile.trace_log_degree_bound, 7);
        assert_eq!(profile.split_composition_log_degree_bound, 6);
        assert_eq!(profile.composition_log_degree_bound, 7);
        assert_eq!(profile.fri_first_layer_log_size, 7);
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
            ZkStarkDegreeBoundProfileError::NonCanonicalQuotientDegreeBounds {
                expected_range: ZkColumnRange::new(2, 0, 2 * SECURE_EXTENSION_DEGREE),
                actual_count: 2,
            }
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
    fn verifier_config_rejects_public_only_column_degree_bounds() {
        let metadata = public_only_metadata(16, 1);
        let verifier_config = ZkVerificationConfig {
            metadata: metadata.clone(),
            column_degree_bounds: vec![ZkColumnDegreeBound {
                range: ZkColumnRange::new(0, 0, 1),
                log_degree_bound: 15,
            }],
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
                },
                ZkPrivateColumnScopeEntry {
                    range: range0,
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                },
            ],
        };
        let mut canonical_scope = scope.clone();
        canonical_scope.canonicalize();
        let mut changed_scope = canonical_scope.clone();
        changed_scope.entries[0].usage = ZkPrivateColumnUsage::Lookup;

        assert_eq!(
            canonical_zk_private_column_scope_hash(&scope),
            canonical_zk_private_column_scope_hash(&canonical_scope)
        );
        assert_ne!(
            canonical_zk_private_column_scope_hash(&canonical_scope),
            canonical_zk_private_column_scope_hash(&changed_scope)
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
                randomized_log_degree: 5,
                randomizer_dimension: 8,
            },
            ZkRandomizerSpaceEntry {
                range: range0,
                trace_domain,
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
}
