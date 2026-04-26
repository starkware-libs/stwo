use serde::{Deserialize, Serialize};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::channel::{Keccak256Channel, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::keccak256_hash::{Keccak256Hash, Keccak256Hasher};

pub type Keccak256MerkleHasher = Keccak256Hasher;

impl MerkleHasherLifted for Keccak256Hasher {
    type Hash = Keccak256Hash;

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        let mut hasher = Self::default();
        let (left_child, right_child) = children_hashes;
        hasher.update(&left_child.0);
        hasher.update(&right_child.0);
        hasher.finalize()
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        column_values
            .iter()
            .for_each(|x| self.update(&x.0.to_be_bytes()));
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

#[derive(Default)]
pub struct Keccak256MerkleChannel;

impl MerkleChannel for Keccak256MerkleChannel {
    type C = Keccak256Channel;
    type H = Keccak256MerkleHasher;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasherLifted>::Hash) {
        channel.update_digest(Keccak256Hasher::concat_and_hash(&channel.digest(), &root));
    }
}

/// Dummy implementations of `Serialize` and `Deserialize` for `Keccak256Hasher` (we cannot derive
/// them because its inner field is from an external crate and doesn't implement these traits).
/// Note: remove this code when possible.
impl Serialize for Keccak256Hasher {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let ser = serializer.serialize_struct("Keccak256Hasher", 1)?;
        serde::ser::SerializeStruct::end(ser)
    }
}

impl<'de> Deserialize<'de> for Keccak256Hasher {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::default())
    }
}
