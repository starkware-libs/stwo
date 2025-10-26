use core::fmt::Debug;

use crate::core::fields::m31::BaseField;
use crate::core::vcs::hash::Hash;

/// An interface for a hasher that only operates on types `Self::Hash` or
/// `BaseField`, as opposed to the underlying hasher's data format (e.g. bytes in the case of
/// Blake2s or elements of other fields in the case of Poseidon252).
pub trait MerkleHasherLifted: Debug + Default + Clone {
    type Hash: Hash;

    /// Constructs an hasher with a state that is already updated with a prefix.
    fn default_with_initial_state() -> Self;

    /// Hashes an inner Merkle node.
    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash;

    /// Converts each `BaseField` elements into the underlying hasher's data format,
    /// and updates the hasher's state.
    fn update_leaf(&mut self, column_values: &[BaseField]);

    /// Finalizes the underlying hasher.
    fn finalize(self) -> Self::Hash;
}
