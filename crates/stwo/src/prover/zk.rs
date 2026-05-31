use itertools::Itertools;
use rand::{CryptoRng, RngCore};
use std_shims::Vec;

use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::{BaseField, P as M31_MODULUS};
use crate::core::fields::qm31::SecureField;
use crate::core::fri::FriProof;
use crate::core::pcs::utils::TreeVec;
use crate::core::poly::circle::CircleDomain;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLiftedAux;
use crate::core::zk::{
    build_zk_randomizer_matrices_from_stwo_sample_metadata,
    validate_zk_private_column_scope_for_witness_randomization, validate_zk_public_only_metadata,
    validate_zk_query_closure_for_witness_randomization,
    validate_zk_randomizer_rank_profile_for_witness_randomization, ZkColumnDegreeBound,
    ZkFriBatchMaskProof, ZkFriBatchMaskQueryValues, ZkMetadataValidationError, ZkPrivacyMap,
    ZkPrivateColumnScope, ZkPrivateColumnScopeValidationError, ZkProofVersion, ZkPublicMetadata,
    ZkQueryClosure, ZkQueryClosureValidationError, ZkRandomizerRankProfile,
    ZkRandomizerRankValidationError, ZkSampleMetadataBuildError,
};
use crate::core::ColumnVec;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::poly::circle::{
    CircleCoefficients, PolyOps, SecureCirclePoly, SecureEvaluation,
};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkDerivationGate {
    StwoSplitQueryExpansion,
    CircleRandomizerSpace,
    OodsDomainExclusion,
    ZkAwareDegreeMetadata,
    FriBatchMaskDegree,
    PrivateLookupPermutationExclusion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkDerivationReview {
    pub gate: ZkDerivationGate,
    pub review_hash: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkProvingConfig {
    pub metadata: ZkPublicMetadata,
    pub privacy_map: ZkPrivacyMap,
    pub private_column_scope: Option<ZkPrivateColumnScope>,
    pub query_closure: Option<ZkQueryClosure>,
    pub randomizer_rank_profile: Option<ZkRandomizerRankProfile>,
    pub derived_randomizer_metadata: Option<ZkDerivedRandomizerMetadata>,
    pub column_degree_bounds: Vec<ZkColumnDegreeBound>,
    pub derivation_reviews: Vec<ZkDerivationReview>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkDerivedRandomizerMetadata {
    trace_domain_log_size: u32,
    fri_first_layer_log_size: u32,
    query_closure: ZkQueryClosure,
    randomizer_rank_profile: ZkRandomizerRankProfile,
}

impl ZkDerivedRandomizerMetadata {
    #[must_use]
    pub fn trace_domain_log_size(&self) -> u32 {
        self.trace_domain_log_size
    }

    #[must_use]
    pub fn fri_first_layer_log_size(&self) -> u32 {
        self.fri_first_layer_log_size
    }

    #[must_use]
    pub fn query_closure(&self) -> &ZkQueryClosure {
        &self.query_closure
    }

    #[must_use]
    pub fn randomizer_rank_profile(&self) -> &ZkRandomizerRankProfile {
        &self.randomizer_rank_profile
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkProvingConfigError {
    UnsupportedProofVersion { actual: u32 },
    MissingDerivationReview(ZkDerivationGate),
    EmptyReviewHash(ZkDerivationGate),
    MissingPrivateColumnScope,
    MissingQueryClosure,
    MissingRandomizerRankProfile,
    MissingDerivedRandomizerMetadata,
    DerivedRandomizerMetadataMismatch,
    MissingPrivateColumnDegreeBounds,
    MissingQuotientDegreeBounds,
    TraceDomainLogSizeMismatch { expected: u32, actual: u32 },
    FriFirstLayerLogSizeMismatch { expected: u32, actual: u32 },
    EmptyRandomizerSpaceHash,
    EmptyPrivateColumnScopeHash,
    EmptySplitDerivationHash,
    DegreeProfileMismatch,
    PrivateColumnDegreeBoundsMismatch,
    PrivacyMapVersionMismatch,
    PrivacyMapHashMismatch,
    PrivateColumnScope(ZkPrivateColumnScopeValidationError),
    QueryClosure(ZkQueryClosureValidationError),
    RandomizerRank(ZkRandomizerRankValidationError),
    SampleMetadata(ZkSampleMetadataBuildError),
    ColumnDegreeBoundsMismatch,
    InsufficientFriBatchMaskQueryDomain,
    UnexpectedPrivateColumnsForPhase1,
    UnexpectedColumnDegreeBoundsForPhase1,
    Phase1MetadataMismatch(ZkMetadataValidationError),
    Phase2And3ActivationBlocked,
}

impl ZkProvingConfig {
    pub fn derive_randomizer_metadata_from_stwo_samples(
        &mut self,
        trace_domain: Coset,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        fri_query_positions: &[usize],
        lifting_log_size: u32,
    ) -> Result<(), ZkProvingConfigError> {
        let expected_trace_log_size = self.metadata.degree_profile.trace_domain_log_size;
        let actual_trace_log_size = trace_domain.log_size();
        if actual_trace_log_size != expected_trace_log_size {
            return Err(ZkProvingConfigError::TraceDomainLogSizeMismatch {
                expected: expected_trace_log_size,
                actual: actual_trace_log_size,
            });
        }
        let expected_fri_log_size = self.metadata.degree_profile.fri_first_layer_log_size;
        if lifting_log_size != expected_fri_log_size {
            return Err(ZkProvingConfigError::FriFirstLayerLogSizeMismatch {
                expected: expected_fri_log_size,
                actual: lifting_log_size,
            });
        }

        let build = build_zk_randomizer_matrices_from_stwo_sample_metadata(
            trace_domain,
            &self.privacy_map,
            self.metadata.witness_randomization.h_witness,
            sampled_points,
            fri_query_positions,
            lifting_log_size,
        )
        .map_err(ZkProvingConfigError::SampleMetadata)?;

        validate_zk_query_closure_for_witness_randomization(
            &self.privacy_map,
            &self.metadata,
            &build.closure,
        )
        .map_err(ZkProvingConfigError::QueryClosure)?;
        validate_zk_randomizer_rank_profile_for_witness_randomization(
            &self.privacy_map,
            &self.metadata,
            &build.closure,
            &build.rank_profile,
        )
        .map_err(ZkProvingConfigError::RandomizerRank)?;

        self.query_closure = Some(build.closure);
        self.randomizer_rank_profile = Some(build.rank_profile);
        self.derived_randomizer_metadata = Some(ZkDerivedRandomizerMetadata {
            trace_domain_log_size: expected_trace_log_size,
            fri_first_layer_log_size: expected_fri_log_size,
            query_closure: self.query_closure.clone().expect("query closure just set"),
            randomizer_rank_profile: self
                .randomizer_rank_profile
                .clone()
                .expect("rank profile just set"),
        });

        Ok(())
    }

    fn validate_privacy_map_binding(&self) -> Result<(), ZkProvingConfigError> {
        if self.metadata.version != self.privacy_map.version {
            return Err(ZkProvingConfigError::PrivacyMapVersionMismatch);
        }
        if self.metadata.privacy_map_hash != self.privacy_map.hash {
            return Err(ZkProvingConfigError::PrivacyMapHashMismatch);
        }

        Ok(())
    }

    pub fn validate_for_fri_batch_mask_only(
        &self,
        lifting_log_size: u32,
        log_blowup_factor: u32,
    ) -> Result<(), ZkProvingConfigError> {
        self.validate_privacy_map_binding()?;
        validate_zk_public_only_metadata(&self.metadata, lifting_log_size, log_blowup_factor)
            .map_err(ZkProvingConfigError::Phase1MetadataMismatch)?;

        if !self.privacy_map.private_columns.is_empty() {
            return Err(ZkProvingConfigError::UnexpectedPrivateColumnsForPhase1);
        }
        if !self.column_degree_bounds.is_empty() {
            return Err(ZkProvingConfigError::UnexpectedColumnDegreeBoundsForPhase1);
        }

        Ok(())
    }

    fn validate_derivation_reviews_present(&self) -> Result<(), ZkProvingConfigError> {
        for gate in [
            ZkDerivationGate::StwoSplitQueryExpansion,
            ZkDerivationGate::CircleRandomizerSpace,
            ZkDerivationGate::OodsDomainExclusion,
            ZkDerivationGate::ZkAwareDegreeMetadata,
            ZkDerivationGate::FriBatchMaskDegree,
            ZkDerivationGate::PrivateLookupPermutationExclusion,
        ] {
            let Some(review) = self
                .derivation_reviews
                .iter()
                .find(|review| review.gate == gate)
            else {
                return Err(ZkProvingConfigError::MissingDerivationReview(gate));
            };
            if review.review_hash.iter().all(|&byte| byte == 0) {
                return Err(ZkProvingConfigError::EmptyReviewHash(gate));
            }
        }

        Ok(())
    }

    pub fn validate_for_witness_randomization(&self) -> Result<(), ZkProvingConfigError> {
        self.validate_for_witness_and_quotient_integration()
    }

    pub fn validate_for_witness_and_quotient_integration(
        &self,
    ) -> Result<(), ZkProvingConfigError> {
        if self.metadata.version != ZkProofVersion::V1 {
            return Err(ZkProvingConfigError::UnsupportedProofVersion {
                actual: self.metadata.version.0,
            });
        }
        self.validate_privacy_map_binding()?;
        self.validate_derivation_reviews_present()?;

        let metadata = &self.metadata;
        if metadata
            .witness_randomization
            .randomizer_space_hash
            .iter()
            .all(|&byte| byte == 0)
        {
            return Err(ZkProvingConfigError::EmptyRandomizerSpaceHash);
        }
        if metadata
            .witness_randomization
            .private_column_scope_hash
            .iter()
            .all(|&byte| byte == 0)
        {
            return Err(ZkProvingConfigError::EmptyPrivateColumnScopeHash);
        }
        let private_column_scope = self
            .private_column_scope
            .as_ref()
            .ok_or(ZkProvingConfigError::MissingPrivateColumnScope)?;
        validate_zk_private_column_scope_for_witness_randomization(
            &self.privacy_map,
            metadata.witness_randomization.private_column_scope_hash,
            private_column_scope,
        )
        .map_err(ZkProvingConfigError::PrivateColumnScope)?;
        if metadata
            .quotient_integration
            .split_derivation_hash
            .iter()
            .all(|&byte| byte == 0)
        {
            return Err(ZkProvingConfigError::EmptySplitDerivationHash);
        }
        if metadata
            .witness_randomization
            .private_column_degree_bounds
            .is_empty()
        {
            return Err(ZkProvingConfigError::MissingPrivateColumnDegreeBounds);
        }
        if metadata
            .quotient_integration
            .quotient_degree_bounds
            .is_empty()
        {
            return Err(ZkProvingConfigError::MissingQuotientDegreeBounds);
        }
        if metadata.degree_profile.h_witness != metadata.witness_randomization.h_witness
            || metadata.degree_profile.h_batch != metadata.quotient_integration.h_batch
            || metadata.degree_profile.fri_first_layer_log_size
                != metadata.quotient_integration.fri_first_layer_log_size
        {
            return Err(ZkProvingConfigError::DegreeProfileMismatch);
        }
        let mut expected_private_ranges = self.privacy_map.private_columns.clone();
        expected_private_ranges.sort_unstable();
        let mut actual_private_ranges = metadata
            .witness_randomization
            .private_column_degree_bounds
            .iter()
            .map(|bound| bound.range)
            .collect::<Vec<_>>();
        actual_private_ranges.sort_unstable();
        if actual_private_ranges != expected_private_ranges {
            return Err(ZkProvingConfigError::PrivateColumnDegreeBoundsMismatch);
        }
        let mut expected_column_degree_bounds = metadata
            .witness_randomization
            .private_column_degree_bounds
            .clone();
        expected_column_degree_bounds
            .extend(metadata.quotient_integration.quotient_degree_bounds.clone());
        if self.column_degree_bounds != expected_column_degree_bounds {
            return Err(ZkProvingConfigError::ColumnDegreeBoundsMismatch);
        }
        let query_closure = self
            .query_closure
            .as_ref()
            .ok_or(ZkProvingConfigError::MissingQueryClosure)?;
        validate_zk_query_closure_for_witness_randomization(
            &self.privacy_map,
            metadata,
            query_closure,
        )
        .map_err(ZkProvingConfigError::QueryClosure)?;
        let randomizer_rank_profile = self
            .randomizer_rank_profile
            .as_ref()
            .ok_or(ZkProvingConfigError::MissingRandomizerRankProfile)?;
        validate_zk_randomizer_rank_profile_for_witness_randomization(
            &self.privacy_map,
            metadata,
            query_closure,
            randomizer_rank_profile,
        )
        .map_err(ZkProvingConfigError::RandomizerRank)?;
        let derived_randomizer_metadata = self
            .derived_randomizer_metadata
            .as_ref()
            .ok_or(ZkProvingConfigError::MissingDerivedRandomizerMetadata)?;
        if derived_randomizer_metadata.trace_domain_log_size()
            != metadata.degree_profile.trace_domain_log_size
        {
            return Err(ZkProvingConfigError::TraceDomainLogSizeMismatch {
                expected: derived_randomizer_metadata.trace_domain_log_size(),
                actual: metadata.degree_profile.trace_domain_log_size,
            });
        }
        if derived_randomizer_metadata.fri_first_layer_log_size()
            != metadata.degree_profile.fri_first_layer_log_size
        {
            return Err(ZkProvingConfigError::FriFirstLayerLogSizeMismatch {
                expected: derived_randomizer_metadata.fri_first_layer_log_size(),
                actual: metadata.degree_profile.fri_first_layer_log_size,
            });
        }
        if derived_randomizer_metadata.query_closure() != query_closure
            || derived_randomizer_metadata.randomizer_rank_profile() != randomizer_rank_profile
        {
            return Err(ZkProvingConfigError::DerivedRandomizerMetadataMismatch);
        }

        Err(ZkProvingConfigError::Phase2And3ActivationBlocked)
    }
}

/// Forms the Protocol 2 FRI input `H_batch = raw_quotient + R`.
///
/// This is intentionally explicit so `R` cannot be accidentally folded into raw
/// quotient batching before the Fiat-Shamir batching challenge.
#[must_use]
pub fn add_fri_batch_mask<B>(
    mut raw_quotient: SecureEvaluation<B, BitReversedOrder>,
    fri_batch_mask: &SecureEvaluation<B, BitReversedOrder>,
) -> SecureEvaluation<B, BitReversedOrder>
where
    B: ColumnOps<BaseField>,
{
    assert_eq!(
        raw_quotient.domain.log_size(),
        fri_batch_mask.domain.log_size(),
        "raw quotient and FRI batch mask must share a domain"
    );
    assert_eq!(
        raw_quotient.values.len(),
        fri_batch_mask.values.len(),
        "raw quotient and FRI batch mask must share a length"
    );

    for index in 0..raw_quotient.values.len() {
        let value = raw_quotient.values.at(index) + fri_batch_mask.values.at(index);
        raw_quotient.values.set(index, value);
    }

    raw_quotient
}

/// Samples the Protocol 2 FRI batch mask polynomial `R` from prover-private
/// randomness and evaluates it on the first FRI layer domain.
///
/// The RNG must be private to the prover and must not be derived from the
/// Fiat-Shamir channel. The verifier sees only the commitment and authenticated
/// query openings.
pub fn sample_fri_batch_mask_evaluation<B, R>(
    domain: CircleDomain,
    log_degree_bound: u32,
    twiddles: &TwiddleTree<B>,
    rng: &mut R,
) -> SecureEvaluation<B, BitReversedOrder>
where
    B: PolyOps,
    R: RngCore + CryptoRng + ?Sized,
{
    let coefficient_count = 1usize
        .checked_shl(log_degree_bound)
        .expect("FRI batch mask degree bound must fit usize");
    let coordinate_polys = core::array::from_fn(|_| {
        let coeffs: Col<B, BaseField> = (0..coefficient_count)
            .map(|_| sample_base_field(rng))
            .collect();
        CircleCoefficients::<B>::new(coeffs)
    });
    SecureCirclePoly(coordinate_polys).evaluate_with_twiddles(domain, twiddles)
}

fn sample_base_field<R: RngCore + ?Sized>(rng: &mut R) -> BaseField {
    loop {
        let candidate = rng.next_u32() & 0x7fff_ffff;
        if candidate < M31_MODULUS {
            return BaseField::from_u32_unchecked(candidate);
        }
    }
}

/// Prover-side oracle for the Protocol 2 FRI batch mask polynomial `R`.
///
/// This helper commits only the `R` oracle. The evaluation must come from
/// [`sample_fri_batch_mask_evaluation`] or an equivalent prover-private,
/// bounded polynomial sampler.
pub struct ZkFriBatchMaskOracleProver<B, H>
where
    B: ColumnOps<BaseField> + MerkleOpsLifted<H>,
    H: MerkleHasherLifted,
{
    evaluation: SecureEvaluation<B, BitReversedOrder>,
    commitment: MerkleProverLifted<B, H>,
}

impl<B, H> ZkFriBatchMaskOracleProver<B, H>
where
    B: ColumnOps<BaseField> + MerkleOpsLifted<H>,
    H: MerkleHasherLifted,
{
    #[must_use]
    pub fn new(evaluation: SecureEvaluation<B, BitReversedOrder>) -> Self {
        let commitment = MerkleProverLifted::commit(
            evaluation.values.columns.iter().collect_vec(),
            evaluation.domain.log_size(),
            0,
        );
        Self {
            evaluation,
            commitment,
        }
    }

    #[must_use]
    pub fn root(&self) -> H::Hash {
        self.commitment.root()
    }

    #[must_use]
    pub fn log_size(&self) -> u32 {
        self.evaluation.domain.log_size()
    }

    #[must_use]
    pub fn evaluation(&self) -> &SecureEvaluation<B, BitReversedOrder> {
        &self.evaluation
    }

    #[must_use]
    pub fn query_values(&self, query_positions: &[usize]) -> ZkFriBatchMaskQueryValues {
        let queries = query_positions
            .iter()
            .map(|&position| self.evaluation.values.at(position).to_m31_array())
            .collect();
        ZkFriBatchMaskQueryValues { queries }
    }

    pub fn decommit(
        self,
        query_positions: &[usize],
        fri_query_positions: &[usize],
        fri_proof: FriProof<H>,
    ) -> (ZkFriBatchMaskProof<H>, MerkleDecommitmentLiftedAux<H>) {
        let fri_queried_values = self.query_values(fri_query_positions);
        let log_size = self.log_size();
        let commitment = self.root();
        let (queried_columns, extended_decommitment) = self.commitment.decommit(
            query_positions,
            self.evaluation.values.columns.iter().collect_vec(),
        );
        assert_eq!(
            queried_columns.len(),
            4,
            "FRI batch mask oracle must decommit four QM31 coordinate columns"
        );

        let query_count = queried_columns[0].len();
        assert!(
            queried_columns
                .iter()
                .all(|column| column.len() == query_count),
            "FRI batch mask coordinate columns must have matching query counts"
        );

        let queries = (0..query_count)
            .map(|query_index| {
                [
                    queried_columns[0][query_index],
                    queried_columns[1][query_index],
                    queried_columns[2][query_index],
                    queried_columns[3][query_index],
                ]
            })
            .collect();

        (
            ZkFriBatchMaskProof {
                commitment,
                log_size,
                fri_proof,
                decommitment: extended_decommitment.decommitment,
                queried_values: ZkFriBatchMaskQueryValues { queries },
                fri_queried_values,
            },
            extended_decommitment.aux,
        )
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    use super::*;
    use crate::core::channel::{Blake2sChannel, MerkleChannel};
    use crate::core::fri::FriConfig;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::queries::Queries;
    use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::zk::{
        ZkColumnRange, ZkDegreeProfile, ZkFriBatchMaskVerificationError, ZkPrivacyMapHash,
        ZkPrivateColumnScope, ZkPrivateColumnScopeEntry, ZkPrivateColumnScopeValidationError,
        ZkPrivateColumnUsage, ZkProofVersion, ZkPublicStatementHash, ZkQueryClosureKind,
        ZkQueryClosureValidationError, ZkQuotientIntegrationProfile,
        ZkRandomizerRankValidationError, ZkSampleMetadataBuildError, ZkWitnessRandomizationProfile,
    };
    use crate::prover::backend::CpuBackend;
    use crate::prover::fri::FriProver;

    type TestFriMaskHasher = <Blake2sMerkleChannel as MerkleChannel>::H;

    struct DeterministicTestCryptoRng(SmallRng);

    impl DeterministicTestCryptoRng {
        fn seed_from_u64(seed: u64) -> Self {
            Self(SmallRng::seed_from_u64(seed))
        }
    }

    impl RngCore for DeterministicTestCryptoRng {
        fn next_u32(&mut self) -> u32 {
            self.0.next_u32()
        }

        fn next_u64(&mut self) -> u64 {
            self.0.next_u64()
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            self.0.fill_bytes(dest);
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
            self.0.try_fill_bytes(dest)
        }
    }

    impl CryptoRng for DeterministicTestCryptoRng {}

    fn zero_hash() -> [u8; 32] {
        [0; 32]
    }

    fn hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn degree_bound(tree_index: usize) -> ZkColumnDegreeBound {
        ZkColumnDegreeBound {
            range: ZkColumnRange::new(tree_index, 0, 1),
            log_degree_bound: 15,
        }
    }

    fn witness_and_quotient_trace_domain() -> Coset {
        CanonicCoset::new(3).coset
    }

    fn witness_and_quotient_sampled_points() -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec(vec![vec![vec![CirclePoint::<SecureField>::get_point(
            9_834_759_221,
        )]]])
    }

    fn witness_and_quotient_fri_query_positions() -> Vec<usize> {
        vec![1]
    }

    fn witness_and_quotient_lifting_log_size() -> u32 {
        4
    }

    fn fri_batch_mask_only_config() -> ZkProvingConfig {
        let privacy_map_hash = ZkPrivacyMapHash(hash(1));
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash,
            public_statement_hash: ZkPublicStatementHash(hash(2)),
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: 15,
                h_witness: 0,
                h_batch: 1 << 15,
                fri_first_layer_log_size: 16,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness: 0,
                randomizer_space_hash: zero_hash(),
                private_column_scope_hash: zero_hash(),
                private_column_degree_bounds: Vec::new(),
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch: 1 << 15,
                fri_first_layer_log_size: 16,
                split_derivation_hash: zero_hash(),
                quotient_degree_bounds: Vec::new(),
            },
        };

        ZkProvingConfig {
            metadata,
            privacy_map: ZkPrivacyMap {
                version: ZkProofVersion::V1,
                private_columns: Vec::new(),
                hash: privacy_map_hash,
            },
            private_column_scope: None,
            query_closure: None,
            randomizer_rank_profile: None,
            derived_randomizer_metadata: None,
            column_degree_bounds: Vec::new(),
            derivation_reviews: Vec::new(),
        }
    }

    fn witness_and_quotient_config() -> ZkProvingConfig {
        let privacy_map_hash = ZkPrivacyMapHash(hash(3));
        let private_bound = degree_bound(0);
        let quotient_bound = degree_bound(1);
        let private_range = private_bound.range;
        let private_column_scope_hash = hash(7);
        let metadata = ZkPublicMetadata {
            version: ZkProofVersion::V1,
            privacy_map_hash,
            public_statement_hash: ZkPublicStatementHash(hash(4)),
            degree_profile: ZkDegreeProfile {
                trace_domain_log_size: 3,
                h_witness: 32,
                h_batch: 1 << 15,
                fri_first_layer_log_size: 4,
            },
            witness_randomization: ZkWitnessRandomizationProfile {
                h_witness: 32,
                randomizer_space_hash: hash(5),
                private_column_scope_hash,
                private_column_degree_bounds: vec![private_bound],
            },
            quotient_integration: ZkQuotientIntegrationProfile {
                h_batch: 1 << 15,
                fri_first_layer_log_size: 4,
                split_derivation_hash: hash(6),
                quotient_degree_bounds: vec![quotient_bound],
            },
        };

        let mut config = ZkProvingConfig {
            metadata,
            privacy_map: ZkPrivacyMap {
                version: ZkProofVersion::V1,
                private_columns: vec![private_range],
                hash: privacy_map_hash,
            },
            private_column_scope: Some(ZkPrivateColumnScope {
                version: ZkProofVersion::V1,
                hash: private_column_scope_hash,
                entries: vec![ZkPrivateColumnScopeEntry {
                    range: private_range,
                    usage: ZkPrivateColumnUsage::OrdinaryWitness,
                }],
            }),
            query_closure: None,
            randomizer_rank_profile: None,
            derived_randomizer_metadata: None,
            column_degree_bounds: vec![private_bound, quotient_bound],
            derivation_reviews: vec![
                ZkDerivationReview {
                    gate: ZkDerivationGate::StwoSplitQueryExpansion,
                    review_hash: hash(10),
                },
                ZkDerivationReview {
                    gate: ZkDerivationGate::CircleRandomizerSpace,
                    review_hash: hash(11),
                },
                ZkDerivationReview {
                    gate: ZkDerivationGate::OodsDomainExclusion,
                    review_hash: hash(12),
                },
                ZkDerivationReview {
                    gate: ZkDerivationGate::ZkAwareDegreeMetadata,
                    review_hash: hash(13),
                },
                ZkDerivationReview {
                    gate: ZkDerivationGate::FriBatchMaskDegree,
                    review_hash: hash(14),
                },
                ZkDerivationReview {
                    gate: ZkDerivationGate::PrivateLookupPermutationExclusion,
                    review_hash: hash(15),
                },
            ],
        };
        config
            .derive_randomizer_metadata_from_stwo_samples(
                witness_and_quotient_trace_domain(),
                &witness_and_quotient_sampled_points(),
                &witness_and_quotient_fri_query_positions(),
                witness_and_quotient_lifting_log_size(),
            )
            .expect("test fixture metadata must derive from STWO samples");
        config
    }

    fn fri_batch_mask_proof(
        seed: u64,
    ) -> (ZkFriBatchMaskProof<TestFriMaskHasher>, Vec<usize>, u32) {
        let log_size = 4;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
        let mut rng = DeterministicTestCryptoRng::seed_from_u64(seed);
        let mask = sample_fri_batch_mask_evaluation::<CpuBackend, _>(
            domain,
            log_size - 1,
            &twiddles,
            &mut rng,
        );
        let queries = vec![0, 3, 7];
        let fri_queries = vec![1, 5, 9];
        let oracle = ZkFriBatchMaskOracleProver::<CpuBackend, TestFriMaskHasher>::new(mask);
        let fri_config = FriConfig::new(0, 1, queries.len(), 1);
        let mut channel = Blake2sChannel::default();
        let fri_prover = FriProver::<CpuBackend, Blake2sMerkleChannel>::commit(
            &mut channel,
            fri_config,
            oracle.evaluation(),
            &twiddles,
        );
        let fri_proof = fri_prover.decommit_on_queries(&Queries::new(&fri_queries, log_size));

        (
            oracle.decommit(&queries, &fri_queries, fri_proof.proof).0,
            queries,
            log_size,
        )
    }

    #[test]
    fn witness_and_quotient_config_derives_randomizer_metadata_from_stwo_samples() {
        let config = witness_and_quotient_config();
        let query_closure = config
            .query_closure
            .as_ref()
            .expect("query closure must be derived from STWO samples");
        let rank_profile = config
            .randomizer_rank_profile
            .as_ref()
            .expect("rank profile must be derived from STWO samples");
        let derived_query_count = u64::try_from(query_closure.entries.len())
            .expect("query closure length must fit randomizer rank metadata");

        assert_eq!(query_closure.entries.len(), 5);
        assert_eq!(rank_profile.entries.len(), 1);
        assert_eq!(rank_profile.entries[0].query_count, derived_query_count);
        assert_eq!(rank_profile.entries[0].randomizer_dimension, 32);
        assert_eq!(rank_profile.entries[0].rank, derived_query_count);
        assert!(config.derived_randomizer_metadata.is_some());
    }

    #[test]
    fn deriving_randomizer_metadata_rejects_out_of_domain_fri_query() {
        let mut config = witness_and_quotient_config();
        let lifting_log_size = witness_and_quotient_lifting_log_size();
        let invalid_position = 1usize << lifting_log_size;

        let err = config
            .derive_randomizer_metadata_from_stwo_samples(
                witness_and_quotient_trace_domain(),
                &witness_and_quotient_sampled_points(),
                &[invalid_position],
                lifting_log_size,
            )
            .unwrap_err();

        assert_eq!(
            err,
            ZkProvingConfigError::SampleMetadata(
                ZkSampleMetadataBuildError::QueryPositionOutOfDomain {
                    position: invalid_position,
                    domain_size: invalid_position,
                }
            )
        );
    }

    #[test]
    fn deriving_randomizer_metadata_rejects_trace_domain_mismatch() {
        let mut config = witness_and_quotient_config();
        let wrong_trace_domain = CanonicCoset::new(4).coset;

        assert_eq!(
            config.derive_randomizer_metadata_from_stwo_samples(
                wrong_trace_domain,
                &witness_and_quotient_sampled_points(),
                &witness_and_quotient_fri_query_positions(),
                witness_and_quotient_lifting_log_size(),
            ),
            Err(ZkProvingConfigError::TraceDomainLogSizeMismatch {
                expected: 3,
                actual: 4,
            })
        );
    }

    #[test]
    fn deriving_randomizer_metadata_rejects_fri_domain_mismatch() {
        let mut config = witness_and_quotient_config();

        assert_eq!(
            config.derive_randomizer_metadata_from_stwo_samples(
                witness_and_quotient_trace_domain(),
                &witness_and_quotient_sampled_points(),
                &witness_and_quotient_fri_query_positions(),
                witness_and_quotient_lifting_log_size() + 1,
            ),
            Err(ZkProvingConfigError::FriFirstLayerLogSizeMismatch {
                expected: 4,
                actual: 5,
            })
        );
    }

    #[test]
    fn witness_and_quotient_rejects_manual_randomizer_metadata_without_derivation() {
        let mut config = witness_and_quotient_config();
        config.derived_randomizer_metadata = None;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::MissingDerivedRandomizerMetadata)
        );
    }

    #[test]
    fn witness_and_quotient_rejects_out_of_domain_manual_fri_closure() {
        let mut config = witness_and_quotient_config();
        let entry = config
            .query_closure
            .as_mut()
            .unwrap()
            .entries
            .iter_mut()
            .find(|entry| entry.kind == ZkQueryClosureKind::FriPosition)
            .expect("test fixture must contain a FRI query closure entry");
        entry.point_or_position = [
            1 << witness_and_quotient_lifting_log_size(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let invalid_entry = *entry;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::QueryClosure(
                ZkQueryClosureValidationError::QueryPositionOutOfDomain {
                    entry: invalid_entry,
                    position: 1 << witness_and_quotient_lifting_log_size(),
                    domain_size: 1 << witness_and_quotient_lifting_log_size(),
                }
            ))
        );
    }

    #[test]
    fn fri_batch_mask_only_config_accepts_public_only_metadata() {
        assert_eq!(
            fri_batch_mask_only_config().validate_for_fri_batch_mask_only(16, 1),
            Ok(())
        );
    }

    #[test]
    fn fri_batch_mask_only_config_rejects_private_columns() {
        let mut config = fri_batch_mask_only_config();
        config
            .privacy_map
            .private_columns
            .push(ZkColumnRange::new(0, 0, 1));

        assert_eq!(
            config.validate_for_fri_batch_mask_only(16, 1),
            Err(ZkProvingConfigError::UnexpectedPrivateColumnsForPhase1)
        );
    }

    #[test]
    fn fri_batch_mask_only_config_rejects_column_degree_bounds() {
        let mut config = fri_batch_mask_only_config();
        config.column_degree_bounds.push(degree_bound(0));

        assert_eq!(
            config.validate_for_fri_batch_mask_only(16, 1),
            Err(ZkProvingConfigError::UnexpectedColumnDegreeBoundsForPhase1)
        );
    }

    #[test]
    fn config_rejects_privacy_map_mismatch() {
        let mut config = fri_batch_mask_only_config();
        config.privacy_map.hash = ZkPrivacyMapHash(hash(99));

        assert_eq!(
            config.validate_for_fri_batch_mask_only(16, 1),
            Err(ZkProvingConfigError::PrivacyMapHashMismatch)
        );
    }

    #[test]
    fn witness_randomization_validation_remains_fail_closed() {
        assert_eq!(
            witness_and_quotient_config().validate_for_witness_randomization(),
            Err(ZkProvingConfigError::Phase2And3ActivationBlocked)
        );
    }

    #[test]
    fn witness_and_quotient_remain_fail_closed_after_reviews_are_present() {
        assert_eq!(
            witness_and_quotient_config().validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::Phase2And3ActivationBlocked)
        );
    }

    #[test]
    fn witness_and_quotient_reject_missing_review_before_fail_closed_terminal() {
        let mut config = witness_and_quotient_config();
        config.derivation_reviews.pop();

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::MissingDerivationReview(
                ZkDerivationGate::PrivateLookupPermutationExclusion
            ))
        );
    }

    #[test]
    fn witness_and_quotient_reject_empty_private_column_scope_hash() {
        let mut config = witness_and_quotient_config();
        config
            .metadata
            .witness_randomization
            .private_column_scope_hash = zero_hash();

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::EmptyPrivateColumnScopeHash)
        );
    }

    #[test]
    fn witness_and_quotient_reject_missing_private_column_scope() {
        let mut config = witness_and_quotient_config();
        config.private_column_scope = None;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::MissingPrivateColumnScope)
        );
    }

    #[test]
    fn witness_and_quotient_reject_ineligible_private_column_scope() {
        let mut config = witness_and_quotient_config();
        config.private_column_scope.as_mut().unwrap().entries[0].usage =
            ZkPrivateColumnUsage::Lookup;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::PrivateColumnScope(
                ZkPrivateColumnScopeValidationError::IneligiblePrivateColumn {
                    range: ZkColumnRange::new(0, 0, 1),
                    usage: ZkPrivateColumnUsage::Lookup,
                }
            ))
        );
    }

    #[test]
    fn witness_and_quotient_reject_private_degree_bounds_mismatch() {
        let mut config = witness_and_quotient_config();
        config
            .metadata
            .witness_randomization
            .private_column_degree_bounds[0]
            .range = ZkColumnRange::new(99, 0, 1);

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::PrivateColumnDegreeBoundsMismatch)
        );
    }

    #[test]
    fn witness_and_quotient_reject_trace_domain_metadata_mutation_after_derivation() {
        let mut config = witness_and_quotient_config();
        config.metadata.degree_profile.trace_domain_log_size = 4;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::TraceDomainLogSizeMismatch {
                expected: 3,
                actual: 4,
            })
        );
    }

    #[test]
    fn witness_and_quotient_reject_insufficient_query_closure_dimension() {
        let mut config = witness_and_quotient_config();
        config.metadata.witness_randomization.h_witness = 0;
        config.metadata.degree_profile.h_witness = 0;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::QueryClosure(
                ZkQueryClosureValidationError::InsufficientRandomizerDimension {
                    range: ZkColumnRange::new(0, 0, 1),
                    required: 5,
                    actual: 0,
                }
            ))
        );
    }

    #[test]
    fn witness_and_quotient_reject_noncanonical_query_closure() {
        let mut config = witness_and_quotient_config();
        let entry = config.query_closure.as_ref().unwrap().entries[0];
        config.query_closure.as_mut().unwrap().entries.push(entry);

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::QueryClosure(
                ZkQueryClosureValidationError::NonCanonicalEntries
            ))
        );
    }

    #[test]
    fn witness_and_quotient_reject_missing_randomizer_rank_profile() {
        let mut config = witness_and_quotient_config();
        config.randomizer_rank_profile = None;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::MissingRandomizerRankProfile)
        );
    }

    #[test]
    fn witness_and_quotient_reject_rank_deficient_randomizer_profile() {
        let mut config = witness_and_quotient_config();
        config.randomizer_rank_profile.as_mut().unwrap().entries[0].rank = 0;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::RandomizerRank(
                ZkRandomizerRankValidationError::RankDeficient {
                    range: ZkColumnRange::new(0, 0, 1),
                    required: 5,
                    actual: 0,
                }
            ))
        );
    }

    #[test]
    fn witness_and_quotient_reject_randomizer_query_count_mismatch() {
        let mut config = witness_and_quotient_config();
        config.randomizer_rank_profile.as_mut().unwrap().entries[0].query_count = 2;

        assert_eq!(
            config.validate_for_witness_and_quotient_integration(),
            Err(ZkProvingConfigError::RandomizerRank(
                ZkRandomizerRankValidationError::QueryCountMismatch {
                    range: ZkColumnRange::new(0, 0, 1),
                    expected: 5,
                    actual: 2,
                }
            ))
        );
    }

    #[test]
    fn fri_batch_mask_rejects_tampered_query_value() {
        let (mut proof, queries, log_size) = fri_batch_mask_proof(21);
        proof.queried_values.queries[0][0] = BaseField::from_u32_unchecked(123456);

        assert!(matches!(
            proof.verify_openings(&queries, log_size),
            Err(ZkFriBatchMaskVerificationError::Merkle(_))
        ));
    }

    #[test]
    fn fri_batch_mask_rejects_tampered_commitment() {
        let (mut proof, queries, log_size) = fri_batch_mask_proof(22);
        let (other_proof, ..) = fri_batch_mask_proof(23);
        proof.commitment = other_proof.commitment;

        assert!(matches!(
            proof.verify_openings(&queries, log_size),
            Err(ZkFriBatchMaskVerificationError::Merkle(_))
        ));
    }

    #[test]
    fn fri_batch_mask_rejects_tampered_query_position() {
        let (proof, mut queries, log_size) = fri_batch_mask_proof(24);
        queries[1] += 1;

        assert!(matches!(
            proof.verify_openings(&queries, log_size),
            Err(ZkFriBatchMaskVerificationError::Merkle(_))
        ));
    }
}
