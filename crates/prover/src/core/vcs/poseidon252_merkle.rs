use num_traits::Zero;
use serde::{Deserialize, Serialize};
use starknet_crypto::{poseidon_hash, poseidon_hash_many};
use starknet_ff::FieldElement as FieldElement252;

use super::ops::MerkleHasher;
use crate::core::channel::{MerkleChannel, Poseidon252Channel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::hash::Hash;

const ELEMENTS_IN_BLOCK: usize = 8;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Poseidon252MerkleHasher;
impl MerkleHasher for Poseidon252MerkleHasher {
    type Hash = FieldElement252;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let n_column_blocks = column_values.len().div_ceil(ELEMENTS_IN_BLOCK);
        let values_len = 2 + n_column_blocks;
        let mut values = Vec::with_capacity(values_len);

        if let Some((left, right)) = children_hashes {
            values.push(left);
            values.push(right);
        }

        let padding_length = ELEMENTS_IN_BLOCK * n_column_blocks - column_values.len();
        let padded_values = column_values
            .iter()
            .copied()
            .chain(std::iter::repeat(BaseField::zero()).take(padding_length));
        for chunk in padded_values.array_chunks::<ELEMENTS_IN_BLOCK>() {
            let mut word = FieldElement252::default();
            for x in chunk {
                word = word * FieldElement252::from(2u64.pow(31)) + FieldElement252::from(x.0);
            }
            values.push(word);
        }
        poseidon_hash_many(&values)
    }
}

impl Hash for FieldElement252 {}

#[derive(Default)]
pub struct Poseidon252MerkleChannel;

impl MerkleChannel for Poseidon252MerkleChannel {
    type C = Poseidon252Channel;
    type H = Poseidon252MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.update_digest(poseidon_hash(channel.digest(), root));
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;
    use starknet_ff::FieldElement as FieldElement252;

    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::ops::MerkleHasher;
    use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher;
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;
    use crate::m31;

    #[test]
    fn test_vector() {
        assert_eq!(
            Poseidon252MerkleHasher::hash_node(None, &[m31!(0), m31!(1)]),
            FieldElement252::from_dec_str(
                "2552053700073128806553921687214114320458351061521275103654266875084493044716"
            )
            .unwrap()
        );

        assert_eq!(
            Poseidon252MerkleHasher::hash_node(
                Some((FieldElement252::from(1u32), FieldElement252::from(2u32))),
                &[m31!(3)]
            ),
            FieldElement252::from_dec_str(
                "159358216886023795422515519110998391754567506678525778721401012606792642769"
            )
            .unwrap()
        );
    }

    #[test]
    fn test_merkle_success() {
        let (queries, decommitment, values, verifier) = prepare_merkle::<Poseidon252MerkleHasher>();
        verifier.verify(&queries, values, decommitment).unwrap();
    }

    #[test]
    fn test_merkle_invalid_witness() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        decommitment.hash_witness[4] = FieldElement252::default();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_invalid_value() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        values[6] = BaseField::zero();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::RootMismatch
        );
    }

    #[test]
    fn test_merkle_witness_too_short() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        decommitment.hash_witness.pop();

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooShort
        );
    }

    #[test]
    fn test_merkle_witness_too_long() {
        let (queries, mut decommitment, values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        decommitment.hash_witness.push(FieldElement252::default());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::WitnessTooLong
        );
    }

    #[test]
    fn test_merkle_values_too_long() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        values.insert(3, BaseField::zero());

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooManyQueriedValues
        );
    }

    #[test]
    fn test_merkle_values_too_short() {
        let (queries, decommitment, mut values, verifier) =
            prepare_merkle::<Poseidon252MerkleHasher>();
        values.remove(3);

        assert_eq!(
            verifier.verify(&queries, values, decommitment).unwrap_err(),
            MerkleVerificationError::TooFewQueriedValues
        );
    }
}
