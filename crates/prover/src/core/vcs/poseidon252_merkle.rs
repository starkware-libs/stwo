use itertools::Itertools;
use num_traits::Zero;
use starknet_crypto::poseidon_hash;
use starknet_ff::FieldElement as FieldElement252;

use super::ops::{MerkleHasher, MerkleOps};
use crate::core::backend::CpuBackend;
use crate::core::fields::m31::BaseField;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Poseidon252MerkleHasher;
impl MerkleHasher for Poseidon252MerkleHasher {
    type Hash = FieldElement252;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut state = FieldElement252::default();
        if let Some((left, right)) = children_hashes {
            state = poseidon_hash(left, right);
        }
        let rem = 7 - ((column_values.len() + 7) % 8);
        let padded_values = column_values
            .iter()
            .copied()
            .chain(std::iter::repeat(BaseField::zero()).take(rem));
        for chunk in padded_values.array_chunks::<8>() {
            let mut word = FieldElement252::default();
            for x in chunk {
                word = word * FieldElement252::from(2u64.pow(31)) + FieldElement252::from(x.0);
            }
            state = poseidon_hash(state, word);
        }
        state
    }
}

impl MerkleOps<Poseidon252MerkleHasher> for CpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<FieldElement252>>,
        columns: &[&Vec<BaseField>],
    ) -> Vec<FieldElement252> {
        (0..(1 << log_size))
            .map(|i| {
                Poseidon252MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column[i]).collect_vec(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use starknet_ff::FieldElement as FieldElement252;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher;
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Poseidon252MerkleHasher>();

        verifier.verify(queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        decommitment.hash_witness[20] = FieldElement252::default();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        values[3][6] = BaseField::zero();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        values[3].push(BaseField::zero());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        values[3].pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        decommitment.hash_witness.push(FieldElement252::default());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }
}
