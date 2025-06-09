use std::fmt::Debug;

use crate::core::fields::m31::BaseField;
use crate::core::vcs::hash::Hash;

/// A Merkle node hash is a hash of: `[left_child_hash, right_child_hash], column0_value,
/// column1_value, ...` where `[]` denotes optional values.
///
/// The largest Merkle layer has no left and right child hashes. The rest of the layers have
/// children hashes. At each layer, the tree may have multiple columns of the same length as the
/// layer. Each node in that layer contains one value from each column.
pub trait MerkleHasher: Debug + Default + Clone {
    type Hash: Hash;
    /// Hashes a single Merkle node. See [MerkleHasher] for more details.
    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash;
}
