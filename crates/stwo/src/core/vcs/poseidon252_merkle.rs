#![allow(unused)]
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use starknet_crypto::{poseidon_hash, poseidon_hash_many};
use starknet_ff::FieldElement as FieldElement252;
use std_shims::Vec;

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

        for chunk in column_values.chunks(ELEMENTS_IN_BLOCK) {
            values.push(construct_felt252_from_m31s(chunk));
        }
        poseidon_hash_many(&values)
    }
}

// Performs felt252 = felt252 << 31 + limb.
const fn append_m31(felt: &mut [u128; 2], limb: M31) {
    // Felt = Felt << 31 + limb.
    *felt = [
        felt[0] << 31 | limb.0 as u128,
        felt[1] << 31 | felt[0] >> (128 - 31),
    ];
}

/// Constructs a felt252 from a slice of m31s.
/// The maximum assumed word size is 8 limbs, which is usually the case except for the remainder.
fn construct_felt252_from_m31s(word: &[M31]) -> FieldElement252 {
    let mut felt_as_u256 = [0u128; 2];
    for limb in word {
        append_m31(&mut felt_as_u256, *limb);
    }

    // If this is the remainder, store its length in bits 248, 249 and 250.
    // Note, you can also look at these 3 bits as word length modulo 8.
    if word.len() < ELEMENTS_IN_BLOCK {
        felt_as_u256[1] += (word.len() as u128) << (248 - 128);
    }

    let felt_bytes = [felt_as_u256[1].to_be_bytes(), felt_as_u256[0].to_be_bytes()];
    let felt_bytes = unsafe { core::mem::transmute::<[[u8; 16]; 2], [u8; 32]>(felt_bytes) };
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
        construct_felt252_from_m31s, Poseidon252MerkleHasher, ELEMENTS_IN_BLOCK,
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
                "2978883932528585652864046122079599882777358126302490183268546077323303473078"
            )
            .unwrap()
        );

        assert_eq!(
            Poseidon252MerkleHasher::hash_node(
                Some((FieldElement252::from(1u32), FieldElement252::from(2u32))),
                &[m31!(3)]
            ),
            FieldElement252::from_dec_str(
                "3286095315900630438551061262740794783852190427874264245042874292062185873630"
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
        let random_values = (0..8 * 1000 + 5).map(|_| rng.gen::<M31>()).collect_vec();

        let expected = random_values
            .chunks(ELEMENTS_IN_BLOCK)
            .map(|word| {
                let mut felt = FieldElement252::default();
                for x in word {
                    felt = felt * FieldElement252::from(2u64.pow(31)) + FieldElement252::from(x.0);
                }
                if word.len() < ELEMENTS_IN_BLOCK {
                    // felt = felt + word.len() << 248;
                    let shift_124 = FieldElement252::from(2u128.pow(124));
                    let shift_248 = shift_124 * shift_124;
                    felt += FieldElement252::from(word.len()) * shift_248;
                }
                felt
            })
            .collect_vec();

        let result = random_values
            .chunks(ELEMENTS_IN_BLOCK)
            .map(construct_felt252_from_m31s)
            .collect_vec();

        assert_eq!(expected, result);
    }
}
