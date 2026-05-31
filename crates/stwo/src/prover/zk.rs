use itertools::Itertools;

use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use crate::core::vcs_lifted::verifier::MerkleDecommitmentLiftedAux;
use crate::core::zk::{ZkFriBatchMaskProof, ZkFriBatchMaskQueryValues};
use crate::prover::backend::ColumnOps;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;
use crate::prover::vcs_lifted::prover::MerkleProverLifted;

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
