use blake2::{Blake2s256, Digest};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
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

impl MerkleHasherLifted for Blake2sHasher {
    type Hash = Blake2sHash;

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        let mut hasher = Blake2s256::new();
        let (left_child, right_child) = children_hashes;
        // TODO(Ilya): Avoid computing the hash of the prefix in runtime.
        hasher.update(NODE_PREFIX);
        hasher.update(left_child);
        hasher.update(right_child);

        Blake2sHash(hasher.finalize().into())
    }
    fn update_leaf(&mut self, column_value: BaseField) {
        self.update(column_value.0.to_le_bytes().as_slice());
    }
    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

impl MerkleHasher for Blake2sHasher {
    type Hash = Blake2sHash;

    fn hash_node(
        _children_hashes: Option<(Self::Hash, Self::Hash)>,
        _column_values: &[BaseField],
    ) -> Self::Hash {
        unimplemented!()
    }
}

// #[derive(Default)]
// pub struct Blake2sMerkleChannel;

// impl MerkleChannel for Blake2sMerkleChannel {
//     type C = Blake2sChannel;
//     type H = Blake2sHasher;

//     fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
//         channel.update_digest(crate::core::vcs::blake2_hash::Blake2sHasher::concat_and_hash(
//             &channel.digest(),
//             &root,
//         ));
//     }
// }

#[cfg(all(test, feature = "prover"))]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs_lifted::blake2_merkle::{Blake2sHash, Blake2sHasher};
    use crate::core::vcs_lifted::test_utils::prepare_merkle;
    use crate::core::vcs_lifted::verifier::MerkleVerificationError;

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Blake2sHasher>();

        verifier.verify(queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sHasher>();
        decommitment.hash_witness[4] = Blake2sHash::default();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<Blake2sHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<Blake2sHasher>();
        decommitment.hash_witness.push(Blake2sHash::default());

        assert_eq!(
            verifier.verify(queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }
}
