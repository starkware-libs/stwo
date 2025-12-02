use itertools::Itertools;
use serde::{Deserialize, Serialize};
use starknet_crypto::{poseidon_hash, poseidon_permute_comp};
use starknet_ff::FieldElement as FieldElement252;
use std_shims::Vec;

use crate::core::channel::{MerkleChannel, Poseidon252Channel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::poseidon252_merkle::{construct_felt252_from_m31s, ELEMENTS_IN_BLOCK};
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;

pub const ELEMENTS_IN_BUFFER: usize = 2 * ELEMENTS_IN_BLOCK;

/// A stateful Poseidon hasher.
///
/// Note that in the case of Blake2s hash we import an external library that implements
/// a stateful hasher, while here we need to manually implement one.
#[derive(Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Poseidon252MerkleHasher {
    state: [FieldElement252; 3],
    // The buffer can hold at most 15 (i.e. `ELEMENTS_IN_BUFFER - 1`) M31 elements.
    buffer: Vec<BaseField>,
}

impl MerkleHasherLifted for Poseidon252MerkleHasher {
    type Hash = FieldElement252;

    fn default_with_initial_state() -> Self {
        Self::default()
    }

    fn hash_children((left, right): (Self::Hash, Self::Hash)) -> Self::Hash {
        poseidon_hash(left, right)
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        let chunks = self
            .buffer
            .iter()
            .chain(column_values)
            .chunks(ELEMENTS_IN_BUFFER);

        // The rest is collected in `remainder` and stored in the hasher's buffer.
        let mut remainder = Vec::new();
        for chunk in &chunks {
            let mut chunk: Vec<_> = chunk.into_iter().copied().collect();
            // If we take this branch we are in the last iteration.
            if chunk.len() < ELEMENTS_IN_BUFFER {
                remainder.extend(chunk);
                break;
            }
            let second = chunk.split_off(ELEMENTS_IN_BLOCK);
            poseidon_update(
                &[
                    construct_felt252_from_m31s(&chunk),
                    construct_felt252_from_m31s(&second),
                ],
                &mut self.state,
            );
        }
        self.buffer = remainder;
    }

    fn finalize(self) -> Self::Hash {
        let remainder: Vec<FieldElement252> = self
            .buffer
            .chunks(ELEMENTS_IN_BLOCK)
            .map(construct_felt252_from_m31s)
            .collect();
        let state = poseidon_finalize(&remainder, self.state);
        state[0]
    }
}

pub fn poseidon_update(values: &[FieldElement252], state: &mut [FieldElement252; 3]) {
    let mut iter = values.chunks_exact(2);
    for msg in iter.by_ref() {
        state[0] += msg[0];
        state[1] += msg[1];
        poseidon_permute_comp(state);
    }
}

pub fn poseidon_finalize(
    values: &[FieldElement252],
    mut state: [FieldElement252; 3],
) -> [FieldElement252; 3] {
    let mut iter = values.chunks_exact(2);
    for msg in iter.by_ref() {
        state[0] += msg[0];
        state[1] += msg[1];
        poseidon_permute_comp(&mut state);
    }
    let r = iter.remainder();
    if r.len() == 1 {
        state[0] += r[0];
    }
    state[r.len()] += FieldElement252::ONE;
    poseidon_permute_comp(&mut state);
    state
}

#[derive(Default)]
pub struct Poseidon252MerkleChannel;

impl MerkleChannel for Poseidon252MerkleChannel {
    type C = Poseidon252Channel;
    type H = Poseidon252MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasherLifted>::Hash) {
        channel.update_digest(poseidon_hash(channel.digest(), root));
    }
}
