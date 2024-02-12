use super::hasher::Hasher;
use crate::core::fields::Field;

/// A Merkle proof of queried indices.
/// Used for storing a all the paths from the query leaves to the root.
/// A correctly generated decommitment should hold all the information needed to generate the root
/// of the tree, proving the queried values and the tree's structure.
// TODO(Ohad): write printing functions.
pub struct MixedDecommitment<F: Field, H: Hasher> {
    pub hashes: Vec<H::Hash>,
    pub witness_elements: Vec<F>,
    pub queried_values: Vec<F>,
}

#[allow(clippy::new_without_default)]
impl<F: Field, H: Hasher> MixedDecommitment<F, H> {
    pub fn new() -> Self {
        Self {
            hashes: vec![],
            witness_elements: vec![],
            queried_values: vec![],
        }
    }

    pub fn verify(&self, _root: H::Hash, _structre: usize, _queried_values: Vec<F>) {
        todo!()
    }
}
