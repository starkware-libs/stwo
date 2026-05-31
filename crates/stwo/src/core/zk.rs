use serde::{Deserialize, Serialize};
use std_shims::{vec, Vec};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{CommitmentSchemeProof, CommitmentSchemeProofAux};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::{
    MerkleDecommitmentLifted, MerkleDecommitmentLiftedAux, MerkleVerificationError,
    MerkleVerifierLifted,
};

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

/// Public metadata echoed by a ZK proof and compared against verifier-owned
/// configuration before affected Fiat-Shamir challenges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkPublicMetadata {
    pub version: ZkProofVersion,
    pub privacy_map_hash: ZkPrivacyMapHash,
    pub public_statement_hash: ZkPublicStatementHash,
    pub degree_profile: ZkDegreeProfile,
}

/// Verifier-owned ZK configuration. Verification trusts this configuration,
/// not the proof's echoed metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkVerificationConfig {
    pub metadata: ZkPublicMetadata,
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
    ) -> Result<Vec<SecureField>, ZkFriBatchMaskVerificationError> {
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
