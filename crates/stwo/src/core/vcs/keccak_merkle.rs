use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};

use super::keccak_hash::KeccakHash;
use crate::core::channel::{KeccakChannel, MerkleChannel};
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct KeccakMerkleHasher;

impl MerkleHasher for KeccakMerkleHasher {
    type Hash = KeccakHash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut hasher = Keccak256::new();

        // Use same prefix structure as Blake2s for compatibility
        if let Some((left_child, right_child)) = children_hashes {
            // println!("Node hashing with left: 0x{} right: 0x{}", hex::encode(left_child.0), hex::encode(right_child.0));
            hasher.update(NODE_PREFIX);
            hasher.update(left_child.as_ref());
            hasher.update(right_child.as_ref());
        } else {
            hasher.update(LEAF_PREFIX);
        }

        for value in column_values {
            // println!("Hashing column value: {:?}", value.0.to_le_bytes());
            hasher.update(value.0.to_le_bytes());
        }

        let result = KeccakHash(hasher.finalize().into());
        // if children_hashes.is_some() {
        //     println!("  -> Node hash: 0x{}", hex::encode(result.0));
        // } else {
        //     println!("  -> Leaf hash: 0x{}", hex::encode(result.0));
        // }
        result
    }
}

#[derive(Default)]
pub struct KeccakMerkleChannel;

impl MerkleChannel for KeccakMerkleChannel {
    type C = KeccakChannel;
    type H = KeccakMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.update_digest(super::keccak_hash::KeccakHasher::concat_and_hash(
            &channel.digest(),
            &root,
        ));
    }
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::keccak_hash::KeccakHash;
    use crate::core::vcs::keccak_merkle::KeccakMerkleHasher;
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;
    use crate::core::vcs::MerkleHasher;

    #[test]
    fn test_merkle_success_keccak() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<KeccakMerkleHasher>();

        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness_keccak() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<KeccakMerkleHasher>();
        decommitment.hash_witness[4] = KeccakHash::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short_keccak() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<KeccakMerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long_keccak() {
        let (queries, mut decommitment, values, verifier) = prepare_merkle::<KeccakMerkleHasher>();
        decommitment.hash_witness.push(KeccakHash::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_invalid_value_keccak() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<KeccakMerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_column_values_too_long_keccak() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<KeccakMerkleHasher>();
        values.insert(3, BaseField::zero());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooManyQueriedValues
        );
    }

    #[test]
    fn test_merkle_column_values_too_short_keccak() {
        let (queries, decommitment, mut values, verifier) = prepare_merkle::<KeccakMerkleHasher>();
        values.remove(3);

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooFewQueriedValues
        );
    }

    #[test]
    fn test_hash_comparison_with_blake2s() {
        use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;

        // Create test data
        let values = vec![BaseField::from(1), BaseField::from(2), BaseField::from(3)];

        // Hash with both implementations
        let keccak_hash = KeccakMerkleHasher::hash_node(None, &values);
        let blake2s_hash = Blake2sMerkleHasher::hash_node(None, &values);

        // Hashes should be different (different algorithms)
        assert_ne!(keccak_hash.0.to_vec(), blake2s_hash.0.to_vec());

        // But both should be non-zero
        assert_ne!(keccak_hash.0, [0u8; 32]);
        assert_ne!(blake2s_hash.0, [0u8; 32]);
    }

    #[test]
    fn test_leaf_vs_node_hashing() {
        let values = vec![BaseField::from(42), BaseField::from(45)];

        // Hash as leaf (no children)
        let leaf_hash = KeccakMerkleHasher::hash_node(None, &values);

        // Hash as node (with dummy children)
        let dummy_child = leaf_hash;
        let node_hash = KeccakMerkleHasher::hash_node(Some((dummy_child, dummy_child)), &values);

        // Should produce different hashes due to different prefixes
        assert_ne!(leaf_hash, node_hash);
    }

    #[test]
    fn test_deterministic_hashing() {
        let values = vec![BaseField::from(123), BaseField::from(456)];

        let hash1 = KeccakMerkleHasher::hash_node(None, &values);
        let hash2 = KeccakMerkleHasher::hash_node(None, &values);

        // Should be deterministic
        assert_eq!(hash1, hash2);
    }


}
