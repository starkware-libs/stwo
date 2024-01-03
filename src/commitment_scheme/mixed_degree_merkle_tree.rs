use std::collections::BTreeSet;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
use super::merkle_multilayer::MerkleMultiLayer;
use super::mixed_degree_decommitment::MixedDecommitment;
use crate::core::fields::{Field, IntoSlice};

pub struct MixedDegreeMerkleTree<'a, F: Field + Sync, H: Hasher> {
    pub input: MerkleTreeInput<'a, F>,
    pub lower_layers: MerkleMultiLayer<H>,
    pub top_layers: Option<MerkleMultiLayer<H>>,
}

impl<'a, F: Field + Sync, H: Hasher> MixedDegreeMerkleTree<'a, F, H>
where
    F: IntoSlice<H::NativeType>,
{
    pub fn commit(_input: MerkleTreeInput<'a, F>, _lower_layers_max_height: usize) -> Self {
        todo!()
    }

    pub fn generate_decommitment(&self, _queries: BTreeSet<usize>) -> MixedDecommitment<F, H> {
        todo!()
    }

    pub fn root(&self) -> H::Hash {
        match &self.top_layers {
            Some(top_layers) => top_layers
                .get_roots()
                .next()
                .expect("Empty top layer")
                .to_owned(),
            None => self
                .lower_layers
                .get_roots()
                .next()
                .expect("Empty Tree")
                .to_owned(),
        }
    }
}
