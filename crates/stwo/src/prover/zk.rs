use itertools::Itertools;
use rand::{CryptoRng, RngCore};
use std_shims::Vec;

use crate::core::fields::m31::P as M31_MODULUS;
use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CircleDomain;
use crate::core::zk::{
    validate_zk_phase1_metadata, ZkColumnDegreeBound, ZkFriBatchMaskProof,
    ZkFriBatchMaskQueryValues, ZkMetadataValidationError, ZkPrivacyMap, ZkPublicMetadata,
};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLiftedAux;
use crate::prover::backend::{Col, ColumnOps};
use crate::prover::poly::circle::{CircleCoefficients, PolyOps, SecureCirclePoly, SecureEvaluation};
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
    pub column_degree_bounds: Vec<ZkColumnDegreeBound>,
    pub derivation_reviews: Vec<ZkDerivationReview>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkProvingConfigError {
    MissingDerivationReview(ZkDerivationGate),
    EmptyReviewHash(ZkDerivationGate),
    MissingPrivateColumnDegreeBounds,
    MissingQuotientDegreeBounds,
    EmptyRandomizerSpaceHash,
    EmptySplitDerivationHash,
    DegreeProfileMismatch,
    PrivacyMapVersionMismatch,
    PrivacyMapHashMismatch,
    ColumnDegreeBoundsMismatch,
    UnexpectedPrivateColumnsForPhase1,
    UnexpectedColumnDegreeBoundsForPhase1,
    Phase1MetadataMismatch(ZkMetadataValidationError),
    Phase2And3ActivationBlocked,
}

impl ZkProvingConfig {
    fn validate_privacy_map_binding(&self) -> Result<(), ZkProvingConfigError> {
        if self.metadata.version != self.privacy_map.version {
            return Err(ZkProvingConfigError::PrivacyMapVersionMismatch);
        }
        if self.metadata.privacy_map_hash != self.privacy_map.hash {
            return Err(ZkProvingConfigError::PrivacyMapHashMismatch);
        }

        Ok(())
    }

    pub fn validate_for_phase_1_fri_batch_mask_only(
        &self,
        lifting_log_size: u32,
        log_blowup_factor: u32,
    ) -> Result<(), ZkProvingConfigError> {
        self.validate_privacy_map_binding()?;
        validate_zk_phase1_metadata(&self.metadata, lifting_log_size, log_blowup_factor)
            .map_err(ZkProvingConfigError::Phase1MetadataMismatch)?;

        if !self.privacy_map.private_columns.is_empty() {
            return Err(ZkProvingConfigError::UnexpectedPrivateColumnsForPhase1);
        }
        if !self.column_degree_bounds.is_empty() {
            return Err(ZkProvingConfigError::UnexpectedColumnDegreeBoundsForPhase1);
        }

        Ok(())
    }

    pub fn validate_for_witness_randomization(&self) -> Result<(), ZkProvingConfigError> {
        self.validate_privacy_map_binding()?;

        for gate in [
            ZkDerivationGate::StwoSplitQueryExpansion,
            ZkDerivationGate::CircleRandomizerSpace,
            ZkDerivationGate::OodsDomainExclusion,
            ZkDerivationGate::ZkAwareDegreeMetadata,
            ZkDerivationGate::FriBatchMaskDegree,
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

    pub fn validate_for_phase_2_and_3(&self) -> Result<(), ZkProvingConfigError> {
        self.validate_for_witness_randomization()?;

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
        let mut expected_column_degree_bounds = metadata
            .witness_randomization
            .private_column_degree_bounds
            .clone();
        expected_column_degree_bounds
            .extend(metadata.quotient_integration.quotient_degree_bounds.clone());
        if self.column_degree_bounds != expected_column_degree_bounds {
            return Err(ZkProvingConfigError::ColumnDegreeBoundsMismatch);
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

    pub fn decommit(
        self,
        query_positions: &[usize],
    ) -> (
        ZkFriBatchMaskProof<H>,
        MerkleDecommitmentLiftedAux<H>,
    ) {
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
                decommitment: extended_decommitment.decommitment,
                queried_values: ZkFriBatchMaskQueryValues { queries },
            },
            extended_decommitment.aux,
        )
    }
}
