use blake2::{Blake2s256, Digest};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};

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

/// TODO(Leo): Document
pub type Blake2sMerkleHasher = Blake2sHasher;
impl MerkleHasherLifted for Blake2sMerkleHasher {
    type Hash = Blake2sHash;

    fn default_with_prefix() -> Self {
        let mut hasher = Blake2sHasher::new();
        hasher.update(&LEAF_PREFIX);
        hasher
    }

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        let mut hasher = Blake2s256::new();
        let (left_child, right_child) = children_hashes;
        // TODO(Ilya): Avoid computing the hash of the prefix in runtime.
        hasher.update(NODE_PREFIX);
        hasher.update(left_child);
        hasher.update(right_child);

        Blake2sHash(hasher.finalize().into())
    }

    /// TODO(Leo). The prover only uses this in the CpuBackend. It used by the verifier to verify
    /// the decommit.
    fn update_leaves(&mut self, column_values: &[BaseField]) {
        column_values
            .iter()
            .for_each(|x| self.update(&x.0.to_le_bytes()));
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

// TODO(Leo): need to implement MerkleChannel that has as associated type H the above
// `Blake2sMerkleHasher`. But for this we need to modify MerkleChannel's H trait bound to implement
// MerkleHasherLifted instead of MerkleHasher.

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
    use crate::core::vcs_lifted::blake2_merkle::{Blake2sHash, Blake2sMerkleHasher};
    use crate::core::vcs_lifted::test_utils::prepare_merkle;
    use crate::core::vcs_lifted::verifier::MerkleVerificationError;

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
        values[6] = BaseField::zero();

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
}
