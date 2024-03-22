use super::hasher::Hasher;
use crate::core::backend::{Col, ColumnOps};
use crate::core::fields::m31::BaseField;

pub trait MerkleHasher: Hasher {
    fn hash_node(
        children_hashes: Option<(Self::Hash, Self::Hash)>,
        node_values: &[BaseField],
    ) -> Self::Hash;
}

pub trait MerkleOps<H: MerkleHasher>: ColumnOps<BaseField> + ColumnOps<H::Hash> {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, H::Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, H::Hash>;
}
