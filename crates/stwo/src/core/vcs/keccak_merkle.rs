use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

use super::keccak_hash::{reduce_to_m31, Keccak256Hash};
use crate::core::channel::{Keccak256ChannelGeneric, MerkleChannel};
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

pub type Keccak256MerkleHasher = Keccak256MerkleHasherGeneric<false>;
/// Same as [Keccak256MerkleHasher], expect that the hash output is taken modulo M31::P.
pub type Keccak256M31MerkleHasher = Keccak256MerkleHasherGeneric<true>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Keccak256MerkleHasherGeneric<const IS_M31_OUTPUT: bool>;
impl<const IS_M31_OUTPUT: bool> MerkleHasher for Keccak256MerkleHasherGeneric<IS_M31_OUTPUT> {
    type Hash = Keccak256Hash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut hasher = Keccak256::new();

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

        Keccak256Hash(r)
    }
}

pub type Keccak256MerkleChannel = Keccak256MerkleChannelGeneric<false>;
/// Same as [Keccak256MerkleChannel], expect that the hash output is taken modulo M31::P.
pub type Keccak256M31MerkleChannel = Keccak256MerkleChannelGeneric<true>;

#[derive(Default)]
pub struct Keccak256MerkleChannelGeneric<const IS_M31_OUTPUT: bool>;

impl<const IS_M31_OUTPUT: bool> MerkleChannel for Keccak256MerkleChannelGeneric<IS_M31_OUTPUT> {
    type C = Keccak256ChannelGeneric<IS_M31_OUTPUT>;
    type H = Keccak256MerkleHasherGeneric<IS_M31_OUTPUT>;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.update_digest(
            super::keccak_hash::Keccak256HasherGeneric::<IS_M31_OUTPUT>::concat_and_hash(
                &channel.digest(),
                &root,
            ),
        );
    }
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::keccak_hash::Keccak256Hash;
    use crate::core::vcs::keccak_merkle::Keccak256MerkleHasher;
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Keccak256MerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Keccak256MerkleHasher>();
        decommitment.hash_witness[4] = Keccak256Hash::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Keccak256MerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Keccak256MerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Keccak256MerkleHasher>();
        decommitment.hash_witness.push(Keccak256Hash::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_column_values_too_long() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Keccak256MerkleHasher>();
        values.insert(3, BaseField::zero());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooManyQueriedValues
        );
    }

    #[test]
    fn test_merkle_column_values_too_short() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Keccak256MerkleHasher>();
        values.remove(3);

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooFewQueriedValues
        );
    }
}

