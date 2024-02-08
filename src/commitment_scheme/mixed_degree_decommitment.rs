use super::hasher::Hasher;
use crate::core::fields::Field;

/// A Merkle proof of queried indices.
/// Used for storing a all the paths from the query leaves to the root.
/// A correctly generated decommitment should hold all the information needed to generate the root
/// of the tree, proving the queried values.
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
}

