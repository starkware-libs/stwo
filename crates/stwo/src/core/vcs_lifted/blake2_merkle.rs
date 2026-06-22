use serde::{Deserialize, Serialize};

use super::merkle_hasher::MerkleHasherLifted;
use crate::core::channel::{Blake2sChannelGeneric, Blake2sM31Channel, MerkleChannel};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::{reduce_to_m31, Blake2sHash, Blake2sHasherGeneric};

pub type Blake2sMerkleHasherGeneric<const IS_M31_OUTPUT: bool> =
    Blake2sHasherGeneric<IS_M31_OUTPUT>;

pub type Blake2sMerkleHasher = Blake2sMerkleHasherGeneric<false>;
/// Same as [Blake2sMerkleHasher], except that the hash output is taken modulo M31::P.
pub type Blake2sM31MerkleHasher = Blake2sMerkleHasherGeneric<true>;

impl<const IS_M31_OUTPUT: bool> MerkleHasherLifted for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT> {
    type Hash = Blake2sHash;

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
            .for_each(|x| self.update(&x.0.to_le_bytes()));
    }

    fn finalize(self) -> Self::Hash {
        self.finalize()
    }
}

pub type Blake2sMerkleChannel = Blake2sMerkleChannelGeneric<false>;
/// Same as [Blake2sMerkleChannel], expect that the hash output is taken modulo M31::P.
pub type Blake2sM31MerkleChannel = Blake2sMerkleChannelGeneric<true>;

#[derive(Default)]
pub struct Blake2sMerkleChannelGeneric<const IS_M31_OUTPUT: bool>;

impl<const IS_M31_OUTPUT: bool> MerkleChannel for Blake2sMerkleChannelGeneric<IS_M31_OUTPUT> {
    type C = Blake2sChannelGeneric<IS_M31_OUTPUT>;
    type H = Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasherLifted>::Hash) {
        use crate::core::vcs::blake2_hash::Blake2sHasherGeneric;
        channel.update_digest(Blake2sHasherGeneric::<IS_M31_OUTPUT>::concat_and_hash(
            &channel.digest(),
            &root,
        ));
    }
}

/// A [`MerkleChannel`] that commits with the **standard** (non-`M31`-reduced) Blake2s Merkle
/// hasher, while running Fiat-Shamir through the **`M31`** Blake2s channel.
///
/// Standard Blake2s gives lossless commitments (each 32-bit word kept in full as two 16-bit
/// limbs), while the `M31` channel keeps all challenge and query derivation in the reduced field.
/// This avoids the standard channel's rejection sampling and raw-word query draws
/// (`Channel::draw_u32s`), so the in-circuit Fiat-Shamir logic stays in `M31` exactly as before.
#[derive(Default)]
pub struct MerkleChannelForCircuit;

impl MerkleChannel for MerkleChannelForCircuit {
    type C = Blake2sM31Channel;
    type H = Blake2sMerkleHasher;

    fn mix_root(channel: &mut Self::C, root: Blake2sHash) {
        channel.update_digest(Blake2sHasherGeneric::<true>::concat_and_hash(
            &channel.digest(),
            &root,
        ));
    }
}

/// Dummy implementations of `Serialize` and `Deserialize` for `Blake2sMerkleHasherGeneric` (we
/// cannot simply derive them because its inner field doesn't implement these traits and is from an
/// external crate).
/// Note: remove this code when possible.
impl<const IS_M31_OUTPUT: bool> Serialize for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let ser = serializer.serialize_struct("Blake2sMerkleHasherGeneric", 1)?;
        serde::ser::SerializeStruct::end(ser)
    }
}

impl<'de, const IS_M31_OUTPUT: bool> Deserialize<'de>
    for Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>
{
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::default())
    }
}
