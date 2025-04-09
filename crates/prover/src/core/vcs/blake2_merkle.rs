use blake2::{Blake2s256, Digest};
use serde::{Deserialize, Serialize};

use super::blake2_hash::Blake2sHash;
use super::ops::MerkleHasher;
use crate::core::channel::{Blake2sChannel, MerkleChannel};
use crate::core::fields::m31::BaseField;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Blake2sMerkleHasher;
impl MerkleHasher for Blake2sMerkleHasher {
    type Hash = Blake2sHash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let mut hasher = Blake2s256::new();

        if let Some((left_child, right_child)) = children_hashes {
            hasher.update(left_child);
            hasher.update(right_child);
        }

        for value in column_values {
            hasher.update(value.0.to_le_bytes());
        }

        Blake2sHash(hasher.finalize().into())
    }
}

#[derive(Default)]
pub struct Blake2sMerkleChannel;

impl MerkleChannel for Blake2sMerkleChannel {
    type C = Blake2sChannel;
    type H = Blake2sMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.update_digest(super::blake2_hash::Blake2sHasher::concat_and_hash(
            &channel.digest(),
            &root,
        ));
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::Blake2sMerkleChannel;
    use crate::core::channel::{Blake2sChannel, MerkleChannel};
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

    #[test]
    fn test_merkle_channel() {
        let mut channel = Blake2sChannel::default();
        let (_queries, _decommitment, _values, verifier) = prepare_merkle::<Blake2sMerkleHasher>();
        Blake2sMerkleChannel::mix_root(&mut channel, verifier.root);
        assert_eq!(channel.channel_time.n_challenges, 1);
    }
}
