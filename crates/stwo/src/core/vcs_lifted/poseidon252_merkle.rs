use serde::{Deserialize, Serialize};
use starknet_ff::FieldElement as FieldElement252;

use crate::core::channel::{MerkleChannel, Poseidon252Channel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct Poseidon252MerkleHasher;

#[allow(unused)]
impl MerkleHasherLifted for Poseidon252MerkleHasher {
    type Hash = FieldElement252;

    fn default_with_initial_state() -> Self {
        unimplemented!()
    }

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        unimplemented!()
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        unimplemented!()
    }

    fn finalize(self) -> Self::Hash {
        unimplemented!()
    }
}

#[derive(Default)]
pub struct Poseidon252MerkleChannel;

#[allow(unused)]
impl MerkleChannel for Poseidon252MerkleChannel {
    type C = Poseidon252Channel;
    type H = Poseidon252MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasherLifted>::Hash) {
        unimplemented!()
    }
}
