use super::hasher::Hasher;
use crate::core::backend::{Col, ColumnOps};
use crate::core::fields::m31::BaseField;

pub trait MerkleHasher: Hasher {
    /// Hashes a single Merkle node.
    /// The node may or may not need to hash 2 hashes from the previous layer - depending if it is a
    /// leaf or not.
    /// In addition, the node may have extra column values that need to be hashed.
    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        column_values: &[BaseField],
    ) -> Self::Hash;
}

pub trait MerkleOps<H: MerkleHasher>: ColumnOps<BaseField> + ColumnOps<H::Hash> {
    /// Commits on an entire layer of the Merkle tree.
    /// The layer has 2^`log_size` nodes that need be hashed. The top most layer has 1 node,
    /// which is a hash of 2 children and some columns.
    /// `prev_layer` is the previous layer of the Merkle tree, if this is not the leaf layer..
    /// That layer is assumed to have 2^(`log_size`+1) nodes.
    /// `columns` are the extra columns that need to be hashed in each node.
    /// They are assumed to be on size 2^`log_size`.
    /// Return the next Merkle layer hashes.
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, H::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, H::Hash>;
}
