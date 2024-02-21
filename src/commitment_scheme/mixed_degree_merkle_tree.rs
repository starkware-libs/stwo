use std::iter::Peekable;

use itertools::Itertools;
use merging_iterator::MergeIter;

use super::hasher::Hasher;
use super::merkle_input::{MerkleTreeConfig, MerkleTreeInput};
use super::merkle_multilayer::MerkleMultiLayer;
use super::mixed_degree_decommitment::MixedDecommitment;
use crate::commitment_scheme::merkle_multilayer::MerkleMultiLayerConfig;
use crate::commitment_scheme::utils::get_column_chunk;
use crate::core::fields::{Field, IntoSlice};

/// A mixed degree merkle tree.
///
/// # Example
///
/// ```rust
/// use stwo::commitment_scheme::merkle_input::MerkleTreeInput;
/// use stwo::commitment_scheme::mixed_degree_merkle_tree::*;
/// use stwo::commitment_scheme::blake3_hash::Blake3Hasher;
/// use stwo::core::fields::m31::M31;
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
        let mut curr_layer = self.height() - self.multi_layer_height(0);
        // Bottom layer.
        self.multi_layers[0].commit_layer::<F, false>(&self.input, [].iter());
        // Rest of the tree.
        for i in 1..self.multi_layers.len() {
            // TODO(Ohad): implement Hash oracle and avoid these copies.
            let multi_layer_height = self.multi_layer_height(i);
            let (prev_multi_layers, next_multi_layers) = self.multi_layers.split_at_mut(i);
            let prev_hashes = prev_multi_layers[i - 1].get_roots();
            debug_assert_eq!(prev_hashes.len(), 1 << (curr_layer));
            curr_layer -= multi_layer_height;
            next_multi_layers[0].commit_layer::<F, true>(&self.input, prev_hashes);
        }
        let mut top_layer_roots = self.multi_layers.last().unwrap().get_roots();
        let root = top_layer_roots
            .next()
            .expect("Top layer should have exactly one root")
            .to_owned();
        debug_assert_eq!(top_layer_roots.count(), 0);
        root
    }

    /// Generates a mixed degree merkle decommitment.
    ///
    /// # Arguments
    ///
    /// * 'queries' - A sequence of queries to each of the columns. Expected to be ordered by the
    ///   order in which the columns were inserted to the tree.
    ///
    /// # Example
    ///
    /// ```rust
    /// use stwo::commitment_scheme::blake3_hash::Blake3Hasher;
    /// use stwo::commitment_scheme::merkle_input::MerkleTreeInput;
    /// use stwo::commitment_scheme::mixed_degree_merkle_tree::{
    ///     MixedDegreeMerkleTree, MixedDegreeMerkleTreeConfig,
    /// };
    /// use stwo::core::fields::m31::M31;
    ///
    /// let mut input = MerkleTreeInput::<M31>::new();
    /// let column_0 = vec![M31::from_u32_unchecked(0); 1024];
    /// let column_1 = vec![M31::from_u32_unchecked(0); 512];
    /// input.insert_column(7, &column_0);
    /// input.insert_column(6, &column_1);
    /// let mut tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
    ///     input,
    ///     MixedDegreeMerkleTreeConfig {
    ///         multi_layer_sizes: [5, 2].to_vec(),
    ///     },
    /// );
    /// let root = tree.commit();
    ///
    /// let queries = vec![vec![0], vec![300, 511]];
    /// let decommitment = tree.decommit(queries.as_ref());
    /// ```
    // TODO(Ohad): introduce a proper query struct, then deprecate 'drain' usage and accepting vecs.
    pub fn decommit(&self, queries: &[Vec<usize>]) -> MixedDecommitment<F, H> {
        assert_eq!(
            queries.len(),
            self.input.n_injected_columns(),
            "Number of query vectors does not match number of injected columns."
        );
        let mut decommitment = MixedDecommitment::<F, H>::new();
        let queries_to_layers = self.input.configuration().sort_queries_by_layer(queries);

        // Decommitment layers are built from the bottom up, excluding the root.
        let mut ancestor_indices = vec![];
        (1..=self.input.max_injected_depth()).rev().for_each(|i| {
            ancestor_indices = self.decommit_single_layer(
                i,
                &queries_to_layers[i - 1],
                ancestor_indices.iter().copied().peekable(),
                &mut decommitment,
            );
        });
        decommitment
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

    // Generates the witness of a single layer and adds it to the decommitment.
    // 'previous_layer_indices' - node indices that are part of the witness for a query below .
    // 'queries_to_layer'- queries to columns at this layer.
    fn decommit_single_layer(
        &self,
        layer_depth: usize,
        queries_to_layer: &[Vec<usize>],
        mut previous_layers_indices: Peekable<impl ExactSizeIterator<Item = usize> + Clone>,
        decommitment: &mut MixedDecommitment<F, H>,
    ) -> Vec<usize> {
        let directly_queried_node_indices = queried_nodes_in_layer(
            queries_to_layer.iter(),
            &self.input.configuration(),
            layer_depth,
        );
        let mut index_value_iterator = directly_queried_node_indices
            .iter()
            .copied()
            .zip(self.layer_felt_witnesses_and_queried_elements(
                layer_depth,
                queries_to_layer.iter(),
                directly_queried_node_indices.iter().copied(),
            ))
            .peekable();
        let mut node_indices = MergeIter::new(
            directly_queried_node_indices.iter().copied(),
            previous_layers_indices.clone().map(|q| q / 2),
        )
        .collect_vec();
        node_indices.dedup();

        for &node_index in node_indices.iter() {
            match previous_layers_indices.next_if(|&q| q / 2 == node_index) {
                None if layer_depth < self.height() => {
                    // If the node is not a direct query, include both hashes.
                    let (l_hash, r_hash) = self.child_hashes(node_index, layer_depth);
                    decommitment.hashes.push(l_hash);
                    decommitment.hashes.push(r_hash);
                }
                Some(q)
                    if previous_layers_indices
                        .next_if(|&next_q| next_q ^ 1 == q)
                        .is_none() =>
                {
                    decommitment.hashes.push(self.sibling_hash(q, layer_depth));
                }
                _ => {}
            }

            if let Some((_, (witness, queried))) =
                index_value_iterator.next_if(|(n, _)| *n == node_index)
            {
                decommitment.witness_elements.extend(witness);
                decommitment.queried_values.extend(queried);
            } else {
                let injected_elements = self.input.get_injected_elements(layer_depth, node_index);
                decommitment.witness_elements.extend(injected_elements);
            }
        }
        node_indices
    }

    // Returns the felt witnesses and queried elements for the given node indices in the specified
    // layer. Assumes that the queries & node indices are sorted in ascending order.
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

    fn sibling_hash(&self, query: usize, layer_depth: usize) -> H::Hash {
        self.get_hash_at(layer_depth, query ^ 1)
    }

    fn child_hashes(&self, node_index: usize, layer_depth: usize) -> (H::Hash, H::Hash) {
        (
            self.get_hash_at(layer_depth, node_index * 2),
            self.get_hash_at(layer_depth, node_index * 2 + 1),
        )
    }
}

/// Translates queries of the form <column, entry_index> to the form <layer, node_index>
/// Input queries are per column, i.e queries[0] is a vector of queries for the first column that
/// was inserted to the tree's input in that layer.
pub fn queried_nodes_in_layer<'a>(
    queries: impl Iterator<Item = &'a Vec<usize>>,
    config: &MerkleTreeConfig,
    layer_depth: usize,
) -> Vec<usize> {
    let columns_lengths = config.column_lengths_at_depth(layer_depth);
    let column_log_lengths = columns_lengths.iter().map(|c_len| c_len.ilog2() as usize);
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

    use itertools::Itertools;

    use super::{queried_nodes_in_layer, MixedDegreeMerkleTree, MixedDegreeMerkleTreeConfig};
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
                super::queried_nodes_in_layer(
                    column_queries_at_depth.iter(),
                    &input.configuration(),
                    i,
                )
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

        let node_indices =
            queried_nodes_in_layer(queries.iter().take(2), &tree.input.configuration(), 4);
        let w4 = tree.layer_felt_witnesses_and_queried_elements(
            4,
            queries[..2].iter(),
            node_indices.iter().copied(),
        );
        let node_indices =
            queried_nodes_in_layer(queries.iter().skip(2), &tree.input.configuration(), 3);
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

    #[test]
    fn decommit_test() {
        let mut input = MerkleTreeInput::<M31>::new();
        let column_0 = (1600..1616).map(M31::from_u32_unchecked).collect_vec();
        let column_1 = (800..808).map(M31::from_u32_unchecked).collect_vec();
        let column_2 = (400..404).map(M31::from_u32_unchecked).collect_vec();
        let column_3 = (0..4096).map(M31::from_u32_unchecked).collect_vec();

        const TREE_HEIGHT: usize = 8;
        input.insert_column(TREE_HEIGHT, &column_3);
        input.insert_column(TREE_HEIGHT - 5, &column_1);
        input.insert_column(TREE_HEIGHT, &column_3);
        input.insert_column(TREE_HEIGHT - 4, &column_0);
        input.insert_column(TREE_HEIGHT - 6, &column_2);
        input.insert_column(TREE_HEIGHT - 4, &column_1);
        input.insert_column(TREE_HEIGHT, &column_3);
        let configuration = input.configuration();
        let mut tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: vec![8],
            },
        );
        let commitment = tree.commit();
        let queries: Vec<Vec<usize>> = vec![
            vec![2],
            vec![0],
            vec![],
            vec![3],
            vec![0, 1, 2, 3],
            vec![4, 7],
            vec![0, 1, 1000, 4095],
        ];

        let test_decommitment = tree.decommit(&queries);
        assert!(test_decommitment.verify(
            commitment,
            &configuration,
            &queries,
            test_decommitment.queried_values.iter().copied()
        ));
    }

    #[test]
    fn dummy_test() {
        const TREE_HEIGHT: usize = 4;
        let mut input = MerkleTreeInput::<M31>::new();
        let column_length_8 = (80..88).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let column_length_4 = (40..44).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        input.insert_column(TREE_HEIGHT, &column_length_8);
        input.insert_column(TREE_HEIGHT - 1, &column_length_4);
        input.insert_column(TREE_HEIGHT - 1, &column_length_8);
        let mut tree = MixedDegreeMerkleTree::<M31, Blake3Hasher>::new(
            input,
            MixedDegreeMerkleTreeConfig {
                multi_layer_sizes: [3, 1].to_vec(),
            },
        );
        let commitment = tree.commit();
        print!("{}", commitment);
    }
}
