use std::collections::BTreeSet;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
use super::merkle_multilayer::MerkleMultiLayer;
use super::mixed_degree_decommitment::MixedDecommitment;
use crate::core::fields::{Field, IntoSlice};

pub struct MixedDegreeMerkleTree<'a, F: Field + Sync, H: Hasher> {
    pub input: MerkleTreeInput<'a, F>,
    pub layers: Vec<MerkleMultiLayer<H>>,
}

pub struct MixedDegreeMerkleTreeConfig {
    pub multi_layer_sizes: Vec<usize>,
}

impl<'a, F: Field + Sync, H: Hasher> MixedDegreeMerkleTree<'a, F, H>
where
    F: IntoSlice<H::NativeType>,
{
    pub fn new(_input: MerkleTreeInput<'a, F>, _config: MixedDegreeMerkleTreeConfig) -> Self {
        todo!()
    }
    pub fn commit() -> H::Hash {
        todo!()
    }

    pub fn generate_decommitment(&self, _queries: BTreeSet<usize>) -> MixedDecommitment<F, H> {
        todo!()
    }

    pub fn root(&self) -> H::Hash {
        match &self.layers.last() {
            Some(top_layer) => top_layer
                .get_roots()
                .next()
                .expect("Empty top layer")
                .to_owned(),
            None => self
                .layers
                .first()
                .expect("Empty Tree")
                .get_roots()
                .next()
                .unwrap()
                .to_owned(),
        }
    }
}
