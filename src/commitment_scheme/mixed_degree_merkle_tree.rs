use itertools::Itertools;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
use super::merkle_multilayer::MerkleMultiLayer;
use super::mixed_degree_decommitment::{DebugInfo, DecommitmentNode, MixedDecommitment};
use crate::commitment_scheme::merkle_multilayer::MerkleMultiLayerConfig;
use crate::commitment_scheme::utils::get_column_chunk;
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

    // Queries should be a query struct that supports queries at multiple layers.
    pub fn decommit(&self, _queries: Vec<Vec<usize>>) -> MixedDecommitment<F, H> {
        todo!()
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

    // TODO(Ohad): remove '_'.
    fn _get_node(
        &self,
        layer_depth: usize,
        node_index: usize,
        include_left_hash: bool,
        include_right_hash: bool,
    ) -> Option<DecommitmentNode<F, H>> {
        let injected_elements = self.input.get_injected_elements(layer_depth, node_index);
        if !include_left_hash && !include_right_hash && injected_elements.is_empty() {
            return None;
        }

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

        #[cfg(not(debug_assertions))]
        return Some(DecommitmentNode {
            right_hash,
            left_hash,
            witness_elements: injected_elements,
            d: DebugInfo {
                _phantom: std::marker::PhantomData,
            },
        });

        // TODO(Ohad): this currently does not make sense, change this function to correctly deal
        // with witness/queried values.
        #[cfg(debug_assertions)]
        return Some(DecommitmentNode {
            right_hash,
            left_hash,
            witness_elements: vec![],
            d: DebugInfo {
                position_in_layer: node_index,
                queried_values: injected_elements,
            },
        });
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

    // Returns the felt witnesses and queried elements for the given node indices in the specified
    // layer. Assumes that the queries & node indices are sorted in ascending order.
    #[allow(dead_code)]
    fn layer_felt_witnesses_and_queried_elements(
        &self,
        layer_depth: usize,
        queries: impl Iterator<Item = &'a Vec<usize>>,
        node_indices: impl ExactSizeIterator<Item = usize>,
    ) -> Vec<(Vec<F>, Vec<F>)> {
        let mut witnesses_and_queried_values_by_node = vec![(vec![], vec![]); node_indices.len()];
        let mut column_query_iterators = queries
            .map(|column_queries| column_queries.iter().peekable())
            .collect_vec();

        // For every node --> For every column --> For every column chunk --> Append
        // queried/witness elements according to that layer's queries.
        for (node_index, (witness_elements, queried_elements)) in
            node_indices.zip(witnesses_and_queried_values_by_node.iter_mut())
        {
            for (column, column_queries) in self
                .input
                .get_columns(layer_depth)
                .iter()
                .zip(column_query_iterators.iter_mut())
            {
                let column_chunk = get_column_chunk(column, node_index, 1 << (layer_depth - 1));
                let column_chunk_start_index = column_chunk.len() * node_index;
                for (i, &felt) in column_chunk.iter().enumerate() {
                    match column_queries.next_if(|&&q| q == i + column_chunk_start_index) {
                        Some(_) => queried_elements.push(felt),
                        None => witness_elements.push(felt),
                    }
                }
            }
        }

        witnesses_and_queried_values_by_node
    }

    #[allow(dead_code)]
    fn sibling_hash(&self, query: usize, layer_depth: usize) -> (Option<H::Hash>, Option<H::Hash>) {
        if query % 2 == 0 {
            (None, Some(self.get_hash_at(layer_depth, query ^ 1)))
        } else {
            (Some(self.get_hash_at(layer_depth, query ^ 1)), None)
        }
    }

    #[allow(dead_code)]
    fn both_hash_siblings(&self, node_index: usize, layer_depth: usize) -> (H::Hash, H::Hash) {
        (
            self.get_hash_at(layer_depth, node_index * 2),
            self.get_hash_at(layer_depth, node_index * 2 + 1),
        )
    }
}

// Translates queries of the form <column, entry_index> to the form <layer, node_index>
// Input queries are per column, i.e queries[0] is a vector of queries for the first column that was
// inserted to the tree's input in that layer.
#[allow(dead_code)]
fn queried_node_indices_in_layer<'a>(
    queries: impl Iterator<Item = &'a Vec<usize>>,
    input: &MerkleTreeInput<'_, impl Field>,
    layer_depth: usize,
) -> Vec<usize> {
    let column_log_lengths = input
        .get_columns(layer_depth)
        .iter()
        .map(|c| c.len().ilog2() as usize);
    let mut node_queries = queries
        .into_iter()
        .zip(column_log_lengths)
        .flat_map(|(column_queries, log_column_length)| {
            let log_n_bags_in_layer = layer_depth - 1;
            let log_n_elements_in_bag = log_column_length - log_n_bags_in_layer;
            column_queries
                .iter()
                .map(move |q| q >> log_n_elements_in_bag)
        })
        .collect::<Vec<_>>();
    node_queries.sort();
    node_queries.dedup();
    node_queries
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::{
        queried_node_indices_in_layer, MixedDegreeMerkleTree, MixedDegreeMerkleTreeConfig,
    };
    use crate::commitment_scheme::blake2_hash::Blake2sHasher;
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_input::MerkleTreeInput;
    use crate::core::fields::m31::M31;
    use crate::core::fields::Field;
    use crate::m31;

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

    // TODO(Ohad): remove after test sub-routine is used.
    fn translate_queries<F: Field>(
        mut queries: Vec<Vec<usize>>,
        input: &MerkleTreeInput<'_, F>,
    ) -> Vec<Vec<usize>> {
        (1..=input.max_injected_depth())
            .rev()
            .map(|i| {
                let n_columns_injected_at_depth = input.get_columns(i).len();
                let column_queries_at_depth = queries
                    .drain(..n_columns_injected_at_depth)
                    .collect::<Vec<_>>();
                super::queried_node_indices_in_layer(column_queries_at_depth.iter(), input, i)
            })
            .collect::<Vec<Vec<usize>>>()
    }

    #[test]
    fn translate_queries_test() {
        let col_length_8 = [m31!(0); 8];
        let col_length_4 = [m31!(0); 4];
        let mut merkle_input = MerkleTreeInput::<M31>::new();

        // Column Length 8 -> depth 4
        // Column Length 8 -> depth 3
        // Column Length 4 -> depth 3
        merkle_input.insert_column(4, &col_length_8);
        merkle_input.insert_column(3, &col_length_8);
        merkle_input.insert_column(3, &col_length_4);

        let first_column_queries = [0, 7];
        let second_column_queries = [3, 7];
        let third_column_queries = [1, 2];

        let expeted_queries_at_depth_4 = [0, 7];
        let expeted_queries_at_depth_3 = [1, 2, 3]; // [1,3] U [1,2]

        let translated_queries = translate_queries(
            vec![
                first_column_queries.to_vec(),
                second_column_queries.to_vec(),
                third_column_queries.to_vec(),
            ],
            &merkle_input,
        );

        assert_eq!(translated_queries[0], expeted_queries_at_depth_4);
        assert_eq!(translated_queries[1], expeted_queries_at_depth_3);
        assert_eq!(translated_queries[2], vec![]);
        assert_eq!(translated_queries[3], vec![]);
    }

    #[test]
    fn build_node_felt_witness_test() {
        let col_length_16 = (0..16).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let col_length_8 = (0..8).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let col_length_4 = (0..4).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let mut merkle_input = MerkleTreeInput::<M31>::new();

        // Column Length 8 -> depth 4
        // Column Length 8 -> depth 3
        // Column Length 4 -> depth 3
        merkle_input.insert_column(4, &col_length_16);
        merkle_input.insert_column(4, &col_length_8);
        merkle_input.insert_column(3, &col_length_8);
        merkle_input.insert_column(3, &col_length_4);
        let tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
            merkle_input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [4].to_vec(),
            },
        );

        let zero_column_queries = vec![0, 15];
        let first_column_queries = vec![0, 7];
        let second_column_queries = vec![3, 7];
        let third_column_queries = vec![1, 2];
        let queries = vec![
            zero_column_queries,
            first_column_queries,
            second_column_queries,
            third_column_queries,
        ];

        let node_indices = queried_node_indices_in_layer(queries.iter().take(2), &tree.input, 4);
        let w4 = tree.layer_felt_witnesses_and_queried_elements(
            4,
            queries[..2].iter(),
            node_indices.iter().copied(),
        );
        let node_indices = queried_node_indices_in_layer(queries.iter().skip(2), &tree.input, 3);
        let w3 = tree.layer_felt_witnesses_and_queried_elements(
            4,
            queries[2..4].iter(),
            node_indices.iter().copied(),
        );

        assert_eq!(
            format!("{:?}", w4),
            "[([M31(1)], [M31(0), M31(0)]), ([M31(14)], [M31(15), M31(7)])]"
        );
        assert_eq!(
            format!("{:?}", w3),
            "[([M31(2)], [M31(3), M31(1)]), ([M31(4), M31(5)], [M31(2)]), ([M31(6), M31(3)], [M31(7)])]"
        );
    }
}
