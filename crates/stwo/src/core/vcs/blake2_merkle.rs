use blake2::{Blake2s256, Digest};
use serde::{Deserialize, Serialize};

use super::blake2_hash::{reduce_to_m31, Blake2sHash};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::MerkleHasher;

pub const LEAF_PREFIX: [u8; 64] = [
    b'l', b'e', b'a', b'f', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];
pub const NODE_PREFIX: [u8; 64] = [
    b'n', b'o', b'd', b'e', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];

pub type Blake2sMerkleHasher = Blake2sMerkleHasherGeneric<false>;
/// Same as [Blake2sMerkleHasher], expect that the hash output is taken modulo M31::P.
pub type Blake2sM31MerkleHasher = Blake2sMerkleHasherGeneric<true>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Blake2sMerkleHasherGeneric<const IS_M31_OUTPUT: bool>;
impl<const IS_M31_OUTPUT: bool> MerkleHasher for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT> {
    type Hash = Blake2sHash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut hasher = Blake2s256::new();

        // TODO(Ilya): Avoid computing the hash of the prefix in runtime.
        if let Some((left_child, right_child)) = children_hashes {
            hasher.update(NODE_PREFIX);
            hasher.update(left_child);
            hasher.update(right_child);
        } else {
            hasher.update(LEAF_PREFIX);
        }

        for value in column_values {
            hasher.update(value.0.to_le_bytes());
        }

        let mut r: [u8; 32] = hasher.finalize().into();
        if IS_M31_OUTPUT {
            r = reduce_to_m31(r);
        }

        Blake2sHash(r)
    }
}

pub type Blake2sMerkleChannel = Blake2sMerkleChannelGeneric<false>;
/// Same as [Blake2sMerkleChannel], expect that the hash output is taken modulo M31::P.
pub type Blake2sM31MerkleChannel = Blake2sMerkleChannelGeneric<true>;

#[derive(Default)]
pub struct Blake2sMerkleChannelGeneric<const IS_M31_OUTPUT: bool>;

#[cfg(all(test, feature = "prover"))]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::blake2_merkle::{Blake2sHash, Blake2sMerkleHasher};
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness[4] = Blake2sHash::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        decommitment.hash_witness.push(Blake2sHash::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        values.insert(3, BaseField::zero());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooManyQueriedValues
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        values.remove(3);

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooFewQueriedValues
        );
    }
}
