use itertools::Itertools;
use std_shims::Vec;

use crate::core::fields::m31::BaseField;
use crate::core::zk::{
    ZkColumnDegreeBound, ZkFriBatchMaskProof, ZkFriBatchMaskQueryValues, ZkPrivacyMap,
    ZkPublicMetadata,
};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLiftedAux;
use crate::prover::backend::ColumnOps;
use crate::prover::poly::circle::SecureEvaluation;
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
}

impl ZkProvingConfig {
    pub fn validate_for_witness_randomization(&self) -> Result<(), ZkProvingConfigError> {
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

        Ok(())
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

/// Prover-side oracle for the Protocol 2 FRI batch mask polynomial `R`.
///
/// This helper commits only the `R` oracle. It does not alter the default PCS
/// proof path and does not sample `R`.
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
