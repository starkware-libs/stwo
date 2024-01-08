use std::collections::BTreeSet;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
use super::merkle_multilayer::MerkleMultiLayer;
use super::mixed_degree_decommitment::MixedDecommitment;
use crate::core::fields::{Field, IntoSlice};

pub struct MixedDegreeMerkleTree<'a, F: Field, H: Hasher> {
    pub input: MerkleTreeInput<'a, F>,
    pub layers: Vec<MerkleMultiLayer<H>>,
}

pub struct MixedDegreeMerkleTreeConfig {
    pub multi_layer_sizes: Vec<usize>,
}

impl<'a, F: Field, H: Hasher> MixedDegreeMerkleTree<'a, F, H>
where
    F: IntoSlice<H::NativeType>,
    H::Hash: 'static
{
    pub fn new(_input: MerkleTreeInput<'a, F>, _config: MixedDegreeMerkleTreeConfig) -> Self {
        todo!()
    }
    pub fn commit() -> H::Hash {
        todo!()
    }

    pub fn decommit(&self, _queries: BTreeSet<usize>) -> MixedDecommitment<F, H> {
        todo!()
    }

    pub fn root(&self) -> H::Hash {
        match &self.layers.last() {
            Some(top_layer) => {
                let mut roots = top_layer.get_roots();
                assert_eq!(roots.len(), 1, "Top layer should have exactly one root");
                *roots.next().unwrap()
            }
            None => panic!("Empty tree!"),
        }
    }
}
