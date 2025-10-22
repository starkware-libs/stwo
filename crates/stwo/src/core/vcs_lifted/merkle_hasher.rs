use core::fmt::Debug;

use crate::core::fields::m31::BaseField;
use crate::core::vcs::hash::Hash;

/// TODO(Leo): document!
/// An interface for an hasher that only operates on types `Self::Hash` or
/// `BaseField`, as opposed to e.g. bytes in the case of Blake2s or elements of other fields
/// in the case of Poseidon252.
pub trait MerkleHasherLifted: Debug + Default + Clone {
    type Hash: Hash;

    fn default_with_prefix() -> Self;

    /// Hashes a single Merkle node. See [MerkleHasher] for more details.
    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash;

    fn update_leaves(&mut self, column_values: &[BaseField]);

    fn finalize(self) -> Self::Hash;
}
