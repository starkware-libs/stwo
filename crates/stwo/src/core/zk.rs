use serde::{Deserialize, Serialize};
use std_shims::{vec, Vec};

use crate::core::channel::Channel;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::constraints::coset_vanishing;
use crate::core::fields::ComplexConjugate;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{CommitmentSchemeProof, CommitmentSchemeProofAux};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux, MerkleVerificationError,
    MerkleVerifierLifted,
};
use num_traits::Zero;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
}

/// Verifier-owned privacy map. Proof metadata may echo its hash, but must not
/// be trusted as the source of privacy policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPrivacyMap {
    pub version: ZkProofVersion,
    pub private_columns: Vec<ZkColumnRange>,
    pub hash: ZkPrivacyMapHash,
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
    InvalidFriBatchDegreeBound,
    FriFirstLayerLogSizeMismatch { expected: u32, actual: u32 },
    QuotientFirstLayerLogSizeMismatch { expected: u32, actual: u32 },
    FriBatchDegreeMismatch { expected: u64, actual: u64 },
    QuotientFriBatchDegreeMismatch { expected: u64, actual: u64 },
    UnexpectedWitnessRandomizationForPhase1,
    UnexpectedPrivateColumnDegreeBoundsForPhase1,
    UnexpectedQuotientDegreeBoundsForPhase1,
    UnexpectedRandomizerSpaceHashForPhase1,
    UnexpectedSplitDerivationHashForPhase1,
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

pub fn validate_zk_phase1_metadata(
    metadata: &ZkPublicMetadata,
    lifting_log_size: u32,
    log_blowup_factor: u32,
) -> Result<(), ZkMetadataValidationError> {
    let expected_h_batch = expected_zk_fri_batch_degree_bound(
        lifting_log_size,
        log_blowup_factor,
    )
    .ok_or(ZkMetadataValidationError::InvalidFriBatchDegreeBound)?;

    if metadata.degree_profile.fri_first_layer_log_size != lifting_log_size {
        return Err(ZkMetadataValidationError::FriFirstLayerLogSizeMismatch {
            expected: lifting_log_size,
            actual: metadata.degree_profile.fri_first_layer_log_size,
        });
    }
    if metadata.quotient_integration.fri_first_layer_log_size != lifting_log_size {
        return Err(ZkMetadataValidationError::QuotientFirstLayerLogSizeMismatch {
            expected: lifting_log_size,
            actual: metadata.quotient_integration.fri_first_layer_log_size,
        });
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
        return Err(
            ZkMetadataValidationError::UnexpectedPrivateColumnDegreeBoundsForPhase1,
        );
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
    mix_column_degree_bounds(
        channel,
        &metadata.witness_randomization.private_column_degree_bounds,
    );

    channel.mix_u64(metadata.quotient_integration.h_batch);
    channel.mix_u32s(&[metadata
        .quotient_integration
        .fri_first_layer_log_size]);
    mix_hash_bytes(channel, &metadata.quotient_integration.split_derivation_hash);
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

/// Separate public oracle proof for the Protocol 2 FRI batch mask polynomial
/// `R`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkFriBatchMaskProof<H: MerkleHasherLifted> {
    pub commitment: H::Hash,
    pub log_size: u32,
    pub decommitment: MerkleDecommitmentLifted<H>,
    pub queried_values: ZkFriBatchMaskQueryValues,
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
        let domain_size = 1usize
            .checked_shl(self.log_size)
            .ok_or(ZkFriBatchMaskVerificationError::QueryDomainTooLarge {
                log_size: self.log_size,
            })?;
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
    pub fri_batch_mask_decommitment_aux: MerkleDecommitmentLiftedAux<H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedZkCommitmentSchemeProof<H: MerkleHasherLifted> {
    pub proof: ZkCommitmentSchemeProof<H>,
    pub aux: ZkCommitmentSchemeProofAux<H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZkStarkProof<H: MerkleHasherLifted>(pub ZkCommitmentSchemeProof<H>);

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
