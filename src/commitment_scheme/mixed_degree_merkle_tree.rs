use std::collections::BTreeSet;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
use super::merkle_multilayer::MerkleMultiLayer;
use super::mixed_degree_decommitment::MixedDecommitment;
use crate::commitment_scheme::merkle_multilayer::MerkleMultiLayerConfig;
use crate::core::fields::{Field, IntoSlice};

pub struct MixedDegreeMerkleTree<'a, F: Field, H: Hasher> {
    pub input: MerkleTreeInput<'a, F>,
    pub layers: Vec<MerkleMultiLayer<H>>,
}

/// Sets the heights of the different layers in the tree.
/// Ordered Lower layers first. e.g [3, 1,2] means the bottom MultiLayer is of height 3, and the
/// top of the tree is of height 1.
pub struct MixedDegreeMerkleTreeConfig {
    pub multi_layer_sizes: Vec<usize>,
}

impl<'a, F: Field, H: Hasher> MixedDegreeMerkleTree<'a, F, H>
where
    F: IntoSlice<H::NativeType>,
{
    pub fn new(input: MerkleTreeInput<'a, F>, config: MixedDegreeMerkleTreeConfig) -> Self {
        let tree_height = input.max_injected_depth();
        let config_tree_height = config.multi_layer_sizes.iter().sum::<usize>();
        assert_eq!(tree_height, config_tree_height);

        let mut layers = Vec::<MerkleMultiLayer<H>>::new();
        let mut current_depth = tree_height;
        for layer_height in config.multi_layer_sizes.into_iter() {
            let layer_config =
                MerkleMultiLayerConfig::new(layer_height, 1 << (current_depth - layer_height));
            layers.push(MerkleMultiLayer::<H>::new(layer_config));
            current_depth -= layer_height;
        }

        MixedDegreeMerkleTree { input, layers }
    }

    pub fn commit(&self) -> H::Hash {
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

#[cfg(test)]
mod tests {
    use super::{MixedDegreeMerkleTree, MixedDegreeMerkleTreeConfig};
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::core::fields::m31::M31;

    #[test]
    fn new_mixed_degree_merkle_tree_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 4096];
        input.insert_column(12, &column);

        let tree = MixedDegreeMerkleTree::<M31, Blake2sHasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [5, 4, 3].to_vec(),
            },
        );

        assert_eq!(tree.layers[0].config.n_sub_trees, 1 << 7);
        assert_eq!(tree.layers[0].config.sub_tree_height, 5);
        assert_eq!(tree.layers[1].config.n_sub_trees, 1 << 3);
        assert_eq!(tree.layers[1].config.sub_tree_height, 4);
        assert_eq!(tree.layers[2].config.n_sub_trees, 1);
        assert_eq!(tree.layers[2].config.sub_tree_height, 3);
    }

    #[test]
    #[should_panic]
    fn new_mixed_degree_merkle_tree_bad_config_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 4096];
        input.insert_column(12, &column);

        // This should panic because the sum of the layer heights is not equal to the tree height
        // deferred by the input.
        MixedDegreeMerkleTree::<M31, Blake2sHasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [5, 4, 2].to_vec(),
            },
        );
    }
}
