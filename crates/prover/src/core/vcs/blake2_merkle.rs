use num_traits::Zero;

use super::blake2_hash::Blake2sHash;
use super::blake2s_ref::compress;
use super::ops::MerkleHasher;
use crate::core::fields::m31::BaseField;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Blake2sMerkleHasher;
impl MerkleHasher for Blake2sMerkleHasher {
    type Hash = Blake2sHash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut state = [0; 8];
        if let Some((left, right)) = children_hashes {
            state = compress(
                state,
                unsafe { std::mem::transmute([left, right]) },
                0,
                0,
                0,
                0,
            );
        }
        let rem = 15 - ((column_values.len() + 15) % 16);
        let padded_values = column_values
            .iter()
            .copied()
            .chain(std::iter::repeat(BaseField::zero()).take(rem));
        for chunk in padded_values.array_chunks::<16>() {
            state = compress(state, unsafe { std::mem::transmute(chunk) }, 0, 0, 0, 0);
        }
        state.map(|x| x.to_le_bytes()).flatten().into()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::blake2_merkle::{Blake2sHash, Blake2sMerkleHasher};
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();

        verifier.verify(queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness[4] = Blake2sHash::default();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        values[3][2] = BaseField::zero();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness.push(Blake2sHash::default());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        values[3].push(BaseField::zero());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        values[3].pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::ColumnValuesTooShort
        );
    }
}
