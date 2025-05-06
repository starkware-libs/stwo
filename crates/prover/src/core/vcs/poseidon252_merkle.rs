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
            .chain(std::iter::repeat_n(BaseField::zero(), padding_length));
        for chunk in padded_values.array_chunks::<ELEMENTS_IN_BLOCK>() {
            let word = chunk.map(|x| x.0);
            values.push(construct_felt_252(&word));
        }
        poseidon_hash_many(&values)
    }
}

fn construct_felt_252(word: &[u32; 8]) -> FieldElement252 {
    let mut felt = [0; 32];
    let bytes = word.map(|x| x.to_be_bytes());

    // First limb.
    let num = bytes[0];
    felt[1] |= num[0] << 1 | num[1] >> 7;
    felt[2] |= num[1] << 1 | num[2] >> 7;
    felt[3] |= num[2] << 1 | num[3] >> 7;
    felt[4] |= num[3] << 1;

    // Second limb.
    let num = bytes[1];
    felt[4] |= num[0] >> 6;
    felt[5] |= num[0] << 2 | num[1] >> 6;
    felt[6] |= num[1] << 2 | num[2] >> 6;
    felt[7] |= num[2] << 2 | num[3] >> 6;
    felt[8] |= num[3] << 2;

    // Third limb.
    let num = bytes[2];
    felt[8] |= num[0] >> 5;
    felt[9] |= num[0] << 3 | num[1] >> 5;
    felt[10] |= num[1] << 3 | num[2] >> 5;
    felt[11] |= num[2] << 3 | num[3] >> 5;
    felt[12] |= num[3] << 3;

    // Fourth limb.
    let num = bytes[3];
    felt[12] |= num[0] >> 4;
    felt[13] |= num[0] << 4 | num[1] >> 4;
    felt[14] |= num[1] << 4 | num[2] >> 4;
    felt[15] |= num[2] << 4 | num[3] >> 4;
    felt[16] |= num[3] << 4;

    // Fifth limb.
    let num = bytes[4];
    felt[16] |= num[0] >> 3;
    felt[17] |= num[0] << 5 | num[1] >> 3;
    felt[18] |= num[1] << 5 | num[2] >> 3;
    felt[19] |= num[2] << 5 | num[3] >> 3;
    felt[20] |= num[3] << 5;

    // Sixth limb.
    let num = bytes[5];
    felt[20] |= num[0] >> 2;
    felt[21] |= num[0] << 6 | num[1] >> 2;
    felt[22] |= num[1] << 6 | num[2] >> 2;
    felt[23] |= num[2] << 6 | num[3] >> 2;
    felt[24] |= num[3] << 6;

    // Seventh limb.
    let num = bytes[6];
    felt[24] |= num[0] >> 1;
    felt[25] |= num[0] << 7 | num[1] >> 1;
    felt[26] |= num[1] << 7 | num[2] >> 1;
    felt[27] |= num[2] << 7 | num[3] >> 1;
    felt[28] |= num[3] << 7;

    // Eighth limb.
    let num = bytes[7];
    felt[28] |= num[0];
    felt[29] |= num[1];
    felt[30] |= num[2];
    felt[31] |= num[3];

    FieldElement252::from_bytes_be(&felt).unwrap()
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
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use starknet_ff::FieldElement as FieldElement252;

    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::ops::MerkleHasher;
    use crate::core::vcs::poseidon252_merkle::{construct_felt_252, Poseidon252MerkleHasher};
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

    #[test]
    fn test_construct_word() {
        let mut rng = SmallRng::seed_from_u64(1638);
        let random_values = (0..8 * 1000)
            .map(|_| rng.gen::<M31>().0)
            .array_chunks::<8>()
            .collect_vec();
        let expected = random_values
            .iter()
            .map(|&word| {
                let mut felt = FieldElement252::default();
                for x in word {
                    felt = felt * FieldElement252::from(2u64.pow(31)) + FieldElement252::from(x);
                }
                felt
            })
            .collect_vec();

        let result = random_values.iter().map(construct_felt_252).collect_vec();

        assert_eq!(expected, result);
    }
}
