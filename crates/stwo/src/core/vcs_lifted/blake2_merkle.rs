use super::merkle_hasher::MerkleHasherLifted;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};

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

pub type Blake2sMerkleHasher = Blake2sHasher;

impl MerkleHasherLifted for Blake2sMerkleHasher {
    type Hash = Blake2sHash;

    fn default_with_initial_state() -> Self {
        let mut hasher = Self::default();
        // TODO(Leo): check if domain separation is necessary in lifted Merkle.
        hasher.update(&LEAF_PREFIX);
        hasher
    }

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        let mut hasher = Self::default();
        let (left_child, right_child) = children_hashes;
        // TODO(Ilya): Avoid computing the hash of the prefix in runtime.
        hasher.update(&NODE_PREFIX);
        hasher.update(&left_child.0);
        hasher.update(&right_child.0);

        hasher.finalize()
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        column_values
            .iter()
            .for_each(|x| self.update(&x.0.to_le_bytes()));
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}
