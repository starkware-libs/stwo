use core::fmt::Debug;

use crate::core::fields::m31::BaseField;
use crate::core::vcs::hash::Hash;

/// A Merkle node hash is a hash of: `[left_child_hash, right_child_hash], column0_value,
/// column1_value, ...` where `[]` denotes optional values.
///
/// The largest Merkle layer has no left and right child hashes. The rest of the layers have
/// children hashes. At each layer, the tree may have multiple columns of the same length as the
/// layer. Each node in that layer contains one value from each column.
pub trait MerkleHasherLifted: Debug + Default + Clone {
    type Hash: Hash;

    /// Hashes a single Merkle node. See [MerkleHasher] for more details.
    fn hash_children(children_hashes: (Self::Hash, Self::Hash)) -> Self::Hash;

    fn update_leaf(&mut self, column_value: BaseField);

    fn finalize(self) -> Self::Hash;

    fn finalize_leaf_slice(mut self, columns_values: &[BaseField]) -> Self::Hash {
        for value in columns_values.iter() {
            self.update_leaf(*value);
        }
        self.finalize()
    }
}
