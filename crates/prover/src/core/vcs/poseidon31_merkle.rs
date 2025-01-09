use num_traits::Zero;
use serde::{Deserialize, Serialize};

use crate::core::channel::{MerkleChannel, Poseidon31Channel};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::vcs::ops::MerkleHasher;
use crate::core::vcs::poseidon31_hash::Poseidon31Hash;
use crate::core::vcs::poseidon31_ref::{poseidon2_permute, Poseidon31CRH};

const ELEMENTS_IN_BLOCK: usize = 8;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Poseidon31MerkleHasher;
impl MerkleHasher for Poseidon31MerkleHasher {
    type Hash = Poseidon31Hash;

    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash {
        let zero = M31::zero();

        let hash_tree = if children_hashes.is_some() {
            let (left, right) = children_hashes.unwrap();
            let mut res = [zero; 16];
            for i in 0..ELEMENTS_IN_BLOCK {
                res[i] = left.0[i];
                res[i + ELEMENTS_IN_BLOCK] = right.0[i];
            }
            Some(Poseidon31Hash(Poseidon31CRH::compress(&res)))
        } else {
            None
        };

        let hash_column = if !column_values.is_empty() {
            let len = column_values.len();
            let num_chunk = len.div_ceil(8);

            let mut digest = if num_chunk == 1 {
                let mut res = [zero; 8];
                res[..len].copy_from_slice(column_values);
                res
            } else {
                let mut res = [zero; 16];
                for i in 0..16 {
                    res[i] = column_values[i];
                }
                Poseidon31CRH::compress(&res)
            };

            for chunk in column_values.chunks(ELEMENTS_IN_BLOCK).skip(2) {
                let mut state = [zero; 16];
                state[..8].copy_from_slice(&digest);
                state[8..16].copy_from_slice(chunk);
                digest = Poseidon31CRH::compress(&state);
            }

            let remain = len % ELEMENTS_IN_BLOCK;
            if remain != 0 {
                let mut state = [zero; 16];
                state[..8].copy_from_slice(&digest);
                state[8..8 + remain].copy_from_slice(&column_values[len - remain..]);
                digest = Poseidon31CRH::compress(&state);
            }

            Some(Poseidon31Hash(digest))
        } else {
            None
        };

        match (hash_tree, hash_column) {
            (Some(hash_tree), Some(hash_column)) => {
                let mut state = [zero; 16];
                state[..8].copy_from_slice(&hash_tree.0);
                state[8..].copy_from_slice(&hash_column.0);
                Poseidon31Hash(Poseidon31CRH::compress(&state))
            }
            (Some(hash_tree), None) => hash_tree,
            (None, Some(hash_column)) => hash_column,
            _ => {
                unreachable!()
            }
        }
    }
}

#[derive(Default)]
pub struct Poseidon31MerkleChannel;

impl MerkleChannel for Poseidon31MerkleChannel {
    type C = Poseidon31Channel;
    type H = Poseidon31MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        let channel_digest = channel.digest();
        let mut state = [
            root.0[0],
            root.0[1],
            root.0[2],
            root.0[3],
            root.0[4],
            root.0[5],
            root.0[6],
            root.0[7],
            channel_digest[0],
            channel_digest[1],
            channel_digest[2],
            channel_digest[3],
            channel_digest[4],
            channel_digest[5],
            channel_digest[6],
            channel_digest[7],
        ];
        poseidon2_permute(&mut state);

        let new_digest = state.last_chunk::<8>().unwrap();
        channel.update_digest(*new_digest);
    }
}
