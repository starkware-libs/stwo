use std::iter::Peekable;

use itertools::Itertools;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
use super::merkle_multilayer::MerkleMultiLayer;
use super::mixed_degree_decommitment::{DecommitmentNode, MixedDecommitment};
use crate::commitment_scheme::merkle_multilayer::MerkleMultiLayerConfig;
use crate::core::fields::{Field, IntoSlice};

/// A mixed degree merkle tree.
///
/// # Example
///
/// ```rust
/// use prover_research::commitment_scheme::merkle_input::MerkleTreeInput;
/// use prover_research::commitment_scheme::mixed_degree_merkle_tree::*;
/// use prover_research::commitment_scheme::blake3_hash::Blake3Hasher;
/// use prover_research::core::fields::m31::M31;
///
/// let mut input = MerkleTreeInput::<M31>::new();
/// let column = vec![M31::from_u32_unchecked(0); 1024];
/// input.insert_column(7, &column);
///
///
/// let mut tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(input,MixedDegreeMerkleTreeConfig {multi_layer_sizes: [5,2].to_vec(),});
/// let root = tree.commit();
pub struct MixedDegreeMerkleTree<'a, F: Field, H: Hasher> {
    input: MerkleTreeInput<'a, F>,
    pub multi_layers: Vec<MerkleMultiLayer<H>>,
}

/// Sets the heights of the multi layers in the tree in ascending order.
pub struct MixedDegreeMerkleTreeConfig {
    pub multi_layer_sizes: Vec<usize>,
}

impl<'a, F: Field, H: Hasher> MixedDegreeMerkleTree<'a, F, H>
where
    F: IntoSlice<H::NativeType>,
{
    pub fn new(input: MerkleTreeInput<'a, F>, config: MixedDegreeMerkleTreeConfig) -> Self {
        let tree_height = input.max_injected_depth();
        Self::validate_config(&config, tree_height);

        let mut layers = Vec::<MerkleMultiLayer<H>>::new();
        let mut current_depth = tree_height;
        for layer_height in config.multi_layer_sizes.into_iter() {
            let layer_config =
                MerkleMultiLayerConfig::new(layer_height, 1 << (current_depth - layer_height));
            layers.push(MerkleMultiLayer::<H>::new(layer_config));
            current_depth -= layer_height;
        }

        MixedDegreeMerkleTree {
            input,
            multi_layers: layers,
        }
    }

    pub fn height(&self) -> usize {
        self.input.max_injected_depth()
    }

    pub fn commit(&mut self) -> H::Hash {
        let tree_height = self.height();
        let mut curr_layer = self.height() - self.multi_layer_height(0);
        // Bottom layer.
        let bottom_multi_layer_input = self.input.split(curr_layer + 1);
        self.multi_layers[0].commit_layer::<F, false>(&bottom_multi_layer_input, &[]);

        // Rest of the tree.
        let mut rebuilt_input = bottom_multi_layer_input;
        for i in 1..self.multi_layers.len() {
            // TODO(Ohad): implement Hash oracle and avoid these copies.
            let prev_hashes = self.multi_layers[i - 1]
                .get_roots()
                .copied()
                .collect::<Vec<H::Hash>>();
            debug_assert_eq!(prev_hashes.len(), 1 << (curr_layer));
            curr_layer -= self.multi_layer_height(i);
            let layer_input = self.input.split(curr_layer + 1);
            self.multi_layers[i].commit_layer::<F, true>(&layer_input, &prev_hashes);
            rebuilt_input.prepend(layer_input);
        }

        let mut top_layer_roots = self.multi_layers.last().unwrap().get_roots();
        let root = top_layer_roots
            .next()
            .expect("Top layer should have exactly one root")
            .to_owned();
        debug_assert_eq!(top_layer_roots.count(), 0);
        debug_assert_eq!(rebuilt_input.max_injected_depth(), tree_height);
        self.input = rebuilt_input;
        root
    }

    // queries should be a query struct that supports queries at multiple layers/
    pub fn decommit(&self, mut _queries: Vec<Vec<usize>>) -> MixedDecommitment<F, H> {
        let mut proof_layers = Vec::<Vec<DecommitmentNode<F, H>>>::new();

        let mut parent_indices_of_previous_layer = vec![];
        for i in (1..=self.height()).rev() {
            let current_layer_bag_queries = _queries.remove(i - 1);
            proof_layers.push(self.decommit_layer(
                i,
                parent_indices_of_previous_layer.iter().copied().peekable(),
                current_layer_bag_queries.iter().copied().peekable(),
            ));
            parent_indices_of_previous_layer = Self::prepare_parent_indices(
                parent_indices_of_previous_layer,
                current_layer_bag_queries,
            );
        }
        MixedDecommitment::new(proof_layers)
    }

    // TODO(Ohad): discuss with the team, maybe queries should be btree-set again.
    fn prepare_parent_indices(
        previous_layer_parents: Vec<usize>,
        current_layer_bag_queries: Vec<usize>,
    ) -> Vec<usize> {
        let mut next_layer_parents = previous_layer_parents.iter().map(|q| *q / 2).collect_vec();
        next_layer_parents.extend(current_layer_bag_queries);
        next_layer_parents.sort();
        next_layer_parents.dedup();
        next_layer_parents
    }

    pub fn get_hash_at(&self, layer_depth: usize, position: usize) -> H::Hash {
        // Determine correct multilayer
        let mut depth_accumulator = layer_depth;
        for multi_layer in self.multi_layers.iter().rev() {
            let multi_layer_height = multi_layer.config.sub_tree_height;
            if multi_layer_height > depth_accumulator {
                return multi_layer.get_hash_value(depth_accumulator, position);
            }
            depth_accumulator -= multi_layer_height;
        }
        panic!()
    }

    // Takes iterators to the parent indices of the previous layer and the current layer's queried
    // node indices. Advances on both simultaneously to produce the current layer's proof in
    // ascending order (left -> right).
    fn decommit_layer(
        &self,
        layer_depth: usize,
        mut parent_indices_of_previous_layer: Peekable<impl Iterator<Item = usize>>,
        mut current_layer_node_queries: Peekable<impl Iterator<Item = usize>>,
    ) -> Vec<DecommitmentNode<F, H>> {
        let mut proof_layer = Vec::<DecommitmentNode<F, H>>::new();

        while let Some(query) = parent_indices_of_previous_layer.next() {
            // Handle bag queries.
            self.add_predecessing_node_query_nodes(
                &mut current_layer_node_queries,
                query / 2,
                layer_depth,
                &mut proof_layer,
            );

            // Handle parent query.
            if let Some(parent_node) =
                self.get_query_node(query, &mut parent_indices_of_previous_layer, layer_depth)
            {
                proof_layer.push(parent_node);
            }
        }

        // Consumes remaining node queries.
        self.add_predecessing_node_query_nodes(
            &mut current_layer_node_queries,
            usize::MAX,
            layer_depth,
            &mut proof_layer,
        );
        proof_layer
    }

    fn get_node(
        &self,
        layer_depth: usize,
        node_index: usize,
        include_left_hash: bool,
        include_right_hash: bool,
    ) -> Option<DecommitmentNode<F, H>> {
        let right_hash = if include_right_hash {
            Some(self.get_hash_at(layer_depth, node_index * 2 + 1))
        } else {
            None
        };
        let left_hash = if include_left_hash {
            Some(self.get_hash_at(layer_depth, node_index * 2))
        } else {
            None
        };
        let injected_elements = self.input.get_injected_elements(layer_depth, node_index);

        if right_hash.is_none() && left_hash.is_none() && injected_elements.is_empty() {
            return None;
        }
        Some(DecommitmentNode {
            right_hash,
            left_hash,
            injected_elements,
            position_in_layer: node_index,
        })
    }

    // Consumes node queries until the next parent query is reached.
    fn add_predecessing_node_query_nodes(
        &self,
        node_query_iterator: &mut Peekable<impl Iterator<Item = usize>>,
        current_parent_query_node_index: usize,
        layer_depth: usize,
        proof_layer: &mut Vec<DecommitmentNode<F, H>>,
    ) {
        loop {
            let next_node_query = node_query_iterator.peek();
            if next_node_query.is_none()
                || *next_node_query.unwrap() >= current_parent_query_node_index
            {
                break;
            }
            // This is where 'true' leaf queries are handled, if the node is a leaf, don't include
            // hash values.
            if let Some(node_bag) = if layer_depth == self.height() {
                self.get_node(layer_depth, *next_node_query.unwrap(), false, false)
            } else {
                self.get_node(layer_depth, *next_node_query.unwrap(), true, true)
            } {
                proof_layer.push(node_bag);
            }
            node_query_iterator.next();
        }
    }

    fn get_query_node(
        &self,
        query: usize,
        parent_query_iterator: &mut Peekable<impl Iterator<Item = usize>>,
        layer_depth: usize,
    ) -> Option<DecommitmentNode<F, H>> {
        match parent_query_iterator.peek() {
            // If both children are in the layer, only injected elements are needed
            // to calculate the parent.
            Some(next_q) if *next_q == query ^ 1 => {
                parent_query_iterator.next();
                None
            }
            _ => {
                if query % 2 == 0 {
                    self.get_node(layer_depth, query / 2, false, true)
                } else {
                    self.get_node(layer_depth, query / 2, true, false)
                }
            }
        }
    }

    pub fn root(&self) -> H::Hash {
        match &self.multi_layers.last() {
            Some(top_layer) => {
                let mut roots = top_layer.get_roots();
                assert_eq!(roots.len(), 1, "Top layer should have exactly one root");
                *roots.next().unwrap()
            }
            None => panic!("Empty tree!"),
        }
    }

    fn validate_config(config: &MixedDegreeMerkleTreeConfig, tree_height: usize) {
        let config_tree_height = config.multi_layer_sizes.iter().sum::<usize>();
        assert_eq!(
            config.multi_layer_sizes.iter().sum::<usize>(),
            tree_height,
            "Sum of the layer heights {} does not match merkle input size {}.",
            config_tree_height,
            tree_height
        );
    }

    fn multi_layer_height(&self, layer_index: usize) -> usize {
        assert!(layer_index < self.multi_layers.len());
        self.multi_layers[layer_index].config.sub_tree_height
    }
}

#[cfg(test)]
mod tests {
    use super::{MixedDegreeMerkleTree, MixedDegreeMerkleTreeConfig};
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::core::fields::m31::M31;

    #[test]
    fn new_mixed_degree_merkle_tree_test() {
        let mut input = super::MerkleTreeInput::<M31>::new();
        let column = vec![M31::from_u32_unchecked(0); 1 << 12];
        input.insert_column(12, &column);

        let multi_layer_sizes = [5, 4, 3].to_vec();
        let tree = MixedDegreeMerkleTree::<M31, Blake2sHasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: multi_layer_sizes.clone(),
            },
        );

        let mut remaining_height = multi_layer_sizes.iter().sum::<usize>();
        multi_layer_sizes
            .iter()
            .enumerate()
            .for_each(|(i, layer_height)| {
                assert_eq!(tree.multi_layers[i].config.sub_tree_height, *layer_height);
                assert_eq!(
                    tree.multi_layers[i].config.n_sub_trees,
                    1 << (remaining_height - layer_height)
                );
                remaining_height -= layer_height;
            });
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

    fn hash_symmetric_path<H: Hasher>(
        initial_value: &[H::NativeType],
        path_length: usize,
    ) -> H::Hash {
        (1..path_length).fold(H::hash(initial_value), |curr_hash, _| {
            H::concat_and_hash(&curr_hash, &curr_hash)
        })
    }

    #[test]
    fn commit_test() {
        const TREE_HEIGHT: usize = 8;
        const INJECT_DEPTH: usize = 3;
        let mut input = super::MerkleTreeInput::<M31>::new();
        let base_column = vec![M31::from_u32_unchecked(0); 1 << (TREE_HEIGHT)];
        let injected_column = vec![M31::from_u32_unchecked(1); 1 << (INJECT_DEPTH - 1)];
        input.insert_column(TREE_HEIGHT + 1, &base_column);
        input.insert_column(INJECT_DEPTH, &injected_column);
        let mut tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [5, 2, 2].to_vec(),
            },
        );
        let expected_hash_at_injected_depth = hash_symmetric_path::<Blake3Hasher>(
            0_u32.to_le_bytes().as_ref(),
            TREE_HEIGHT + 1 - INJECT_DEPTH,
        );
        let mut sack_at_injected_depth = expected_hash_at_injected_depth.as_ref().to_vec();
        sack_at_injected_depth.extend(expected_hash_at_injected_depth.as_ref().to_vec());
        sack_at_injected_depth.extend(1u32.to_le_bytes());
        let expected_result =
            hash_symmetric_path::<Blake3Hasher>(sack_at_injected_depth.as_ref(), INJECT_DEPTH);

        let root = tree.commit();
        assert_eq!(root, expected_result);
    }

    #[test]
    fn get_hash_at_test() {
        const TREE_HEIGHT: usize = 3;
        let mut input = super::MerkleTreeInput::<M31>::new();
        let base_column = (0..4).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        input.insert_column(TREE_HEIGHT, &base_column);
        let mut tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [2, 1].to_vec(),
            },
        );
        let root = tree.commit();
        assert_eq!(root, tree.get_hash_at(0, 0));

        let mut hasher = Blake3Hasher::new();
        hasher.update(&0_u32.to_le_bytes());
        // hasher.update(&1_u32.to_le_bytes());
        let expected_hash_at_2_0 = hasher.finalize_reset();
        let hash_at_2_0 = tree.get_hash_at(2, 0);
        assert_eq!(hash_at_2_0, expected_hash_at_2_0);

        hasher.update(&2_u32.to_le_bytes());
        let expected_hash_at_2_2 = hasher.finalize_reset();
        let hash_at_2_2 = tree.get_hash_at(2, 2);
        assert_eq!(hash_at_2_2, expected_hash_at_2_2);
        hasher.update(&3_u32.to_le_bytes());
        let expected_hash_at_2_3 = hasher.finalize_reset();
        let hash_at_2_3 = tree.get_hash_at(2, 3);
        assert_eq!(hash_at_2_3, expected_hash_at_2_3);

        let expected_parent_of_2_2_and_2_3 =
            Blake3Hasher::concat_and_hash(&expected_hash_at_2_2, &expected_hash_at_2_3);
        let parent_of_2_2_and_2_3 = tree.get_hash_at(1, 1);
        assert_eq!(parent_of_2_2_and_2_3, expected_parent_of_2_2_and_2_3);
    }

    #[test]
    #[should_panic]
    fn get_hash_at_invalid_layer_test() {
        const TREE_HEIGHT: usize = 3;
        let mut input = super::MerkleTreeInput::<M31>::new();
        let base_column = (0..4).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        input.insert_column(TREE_HEIGHT, &base_column);
        let tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [2, 1].to_vec(),
            },
        );
        tree.get_hash_at(4, 0);
    }

    #[test]
    fn decommit_test() {
        const TREE_HEIGHT: usize = 4;
        let mut input = super::MerkleTreeInput::<M31>::new();
        let base_column = (0..8).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let injected_column = (0..4)
            .rev()
            .map(M31::from_u32_unchecked)
            .collect::<Vec<M31>>();
        input.insert_column(TREE_HEIGHT, &base_column);
        input.insert_column(TREE_HEIGHT - 1, &injected_column);
        let mut tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [4].to_vec(),
            },
        );
        tree.commit();
        let decommitment = tree.decommit(vec![vec![0], vec![], vec![3], vec![2]]);
        println!("{}", decommitment);
    }
}
