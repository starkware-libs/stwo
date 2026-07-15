use serde::{Deserialize, Serialize};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::channel::{Blake2sChannel, Blake2sM31Channel, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasherGeneric};

pub type Blake2sMerkleHasher = Blake2sHasherGeneric<false>;

impl MerkleHasherLifted for Blake2sMerkleHasher {
    type Hash = Blake2sHash;

    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash {
        let mut hasher = Self::default();
        let (left_child, right_child) = children_hashes;
        hasher.update(&left_child.0);
        hasher.update(&right_child.0);

        hasher.finalize()
    }

    fn update_leaf(&mut self, column_values: &[BaseField]) {
        column_values.iter().for_each(|x| self.update(&x.0.to_le_bytes()));
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

/// A [`MerkleChannel`] that commits with the **standard** Blake2s Merkle hasher and runs
/// Fiat-Shamir through the **standard** Blake2s channel.
#[derive(Default)]
pub struct Blake2sMerkleChannel;

impl MerkleChannel for Blake2sMerkleChannel {
    type C = Blake2sChannel;
    type H = Blake2sMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: Blake2sHash) {
        channel.update_digest(Blake2sHasherGeneric::<false>::concat_and_hash(
            &channel.digest(),
            &root,
        ));
    }
}

/// A [`MerkleChannel`] that commits with the **standard** (non-`M31`-reduced) Blake2s Merkle
/// hasher, while running Fiat-Shamir through the **`M31`** Blake2s channel.
///
/// All challenge and query derivation in the reduced field.
/// This avoids the standard channel's rejection sampling and raw-word query draws
/// (`Channel::draw_u32s`).
#[derive(Default)]
pub struct Blake2sM31MerkleChannel;

impl MerkleChannel for Blake2sM31MerkleChannel {
    type C = Blake2sM31Channel;
    type H = Blake2sMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: Blake2sHash) {
        // Mix the *unreduced* Merkle root into the channel.
        channel
            .update_digest(Blake2sHasherGeneric::<true>::concat_and_hash(&channel.digest(), &root));
    }
}

/// Dummy implementations of `Serialize` and `Deserialize` for `Blake2sMerkleHasher` (we
/// cannot simply derive them because its inner field doesn't implement these traits and is from an
/// external crate).
/// Note: remove this code when possible.
impl Serialize for Blake2sMerkleHasher {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let ser = serializer.serialize_struct("Blake2sMerkleHasher", 1)?;
        serde::ser::SerializeStruct::end(ser)
    }
}

impl<'de> Deserialize<'de> for Blake2sMerkleHasher {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::default())
    }
}
