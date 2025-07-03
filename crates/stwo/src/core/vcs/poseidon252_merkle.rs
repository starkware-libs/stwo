#![allow(unused)]
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use starknet_crypto::{poseidon_hash, poseidon_hash_many};
use starknet_ff::FieldElement as FieldElement252;

use crate::core::channel::{MerkleChannel, Poseidon252Channel};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::vcs::hash::Hash;
use crate::core::vcs::MerkleHasher;

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
            values.push(construct_felt252_from_m31s(&chunk));
        }
        poseidon_hash_many(&values)
    }
}

fn construct_felt252_from_m31s(word: &[M31; 8]) -> FieldElement252 {
    // Felt = Felt << 31 + limb.
    let append_m31 = |felt: &mut [u128; 2], limb: M31| {
        *felt = [
            felt[0] << 31 | limb.0 as u128,
            felt[0] >> (128 - 31) | felt[1] << 31,
        ];
    };

    let mut felt_as_u256 = [0u128; 2];
    for limb in word {
        append_m31(&mut felt_as_u256, *limb);
    }

    let felt_bytes = [felt_as_u256[1].to_be_bytes(), felt_as_u256[0].to_be_bytes()];
    let felt_bytes = unsafe { std::mem::transmute::<[[u8; 16]; 2], [u8; 32]>(felt_bytes) };
    FieldElement252::from_bytes_be(&felt_bytes).unwrap()
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

#[cfg(all(test, feature = "prover"))]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use num_traits::Zero;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use starknet_ff::FieldElement as FieldElement252;

    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::vcs::poseidon252_merkle::{
        construct_felt252_from_m31s, Poseidon252MerkleHasher,
    };
    use crate::core::vcs::test_utils::prepare_merkle;
    use crate::core::vcs::verifier::MerkleVerificationError;
    use crate::core::vcs::MerkleHasher;
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
            .map(|_| rng.gen::<M31>())
            .array_chunks::<8>()
            .collect_vec();
        let expected = random_values
            .iter()
            .map(|&word| {
                let mut felt = FieldElement252::default();
                for x in word {
                    felt = felt * FieldElement252::from(2u64.pow(31)) + FieldElement252::from(x.0);
                }
                felt
            })
            .collect_vec();

        let result = random_values
            .iter()
            .map(construct_felt252_from_m31s)
            .collect_vec();

        assert_eq!(expected, result);
    }
}
