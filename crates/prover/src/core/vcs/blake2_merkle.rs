use std::mem::transmute;

use serde::{Deserialize, Serialize};

use super::blake2_hash::Blake2sHash;
use super::blake2s_ref::{compress, IV};
use super::ops::MerkleHasher;
use crate::core::channel::{Blake2sChannel, MerkleChannel};
use crate::core::fields::m31::BaseField;

const BLOCK_BYTES: u64 = 64;
const BYTES_PER_FELT: u64 = 4;
const FELTS_PER_HASH: usize = 16;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Blake2sMerkleHasher;
impl MerkleHasher for Blake2sMerkleHasher {
    type Hash = Blake2sHash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        hash_node(children_hashes, column_values)
    }
}

fn hash_node(
    children_hashes: Option<(Blake2sHash, Blake2sHash)>,
    column_values: &[BaseField],
) -> Blake2sHash {
    let children_hashes: Option<[u32; 16]> = children_hashes
        .map(|(left, right)| unsafe { transmute::<[Blake2sHash; 2], [u32; 16]>([left, right]) });
    let column_values: &[u32] = unsafe { transmute(column_values) };

    let state = hash_node_native_types(children_hashes, column_values);

    Blake2sHash(unsafe { transmute::<[u32; 8], [u8; 32]>(state) })
}

fn hash_node_native_types(children_hashes: Option<[u32; 16]>, column_values: &[u32]) -> [u32; 8] {
    let mut state = IV;
    // No columns in the layer.
    if column_values.is_empty() {
        let node = children_hashes.unwrap_or_default();
        return compress_finalize(state, node, BLOCK_BYTES);
    }

    // Columns in the layer.
    let mut t: u64 = 0;
    if let Some(node) = children_hashes {
        t += BLOCK_BYTES;
        state = compress_unfinalized(state, node, t);
    }

    let last_block_offset = (column_values.len() / FELTS_PER_HASH) * FELTS_PER_HASH;
    let (column_values, rem) = column_values.split_at(last_block_offset);
    for &chunk in column_values.array_chunks::<FELTS_PER_HASH>() {
        t += BLOCK_BYTES;
        state = compress_unfinalized(state, chunk, t);
    }

    t += rem.len() as u64 * BYTES_PER_FELT;
    let mut last_block = [0; FELTS_PER_HASH];
    last_block[..rem.len()].copy_from_slice(rem);
    compress_finalize(state, last_block, t)
}

const fn compress_unfinalized(state: [u32; 8], chunk: [u32; 16], t: u64) -> [u32; 8] {
    compress(state, chunk, t as u32, (t >> 32) as u32, 0, 0)
}

const fn compress_finalize(state: [u32; 8], last_block: [u32; 16], t: u64) -> [u32; 8] {
    compress(state, last_block, t as u32, (t >> 32) as u32, 0xFFFFFFFF, 0)
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
