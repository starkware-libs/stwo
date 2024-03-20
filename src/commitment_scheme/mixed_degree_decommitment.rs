use std::iter::Peekable;

use itertools::Itertools;
use merging_iterator::MergeIter;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeStructure;
use super::mixed_degree_merkle_tree::queried_nodes_in_layer;
use crate::core::fields::{Field, IntoSlice};

/// A Merkle proof of queried indices.
/// Used for storing a all the paths from the query leaves to the root.
/// A correctly generated decommitment should hold all the information needed to generate the root
/// of the tree, proving the queried values and the tree's structure.
// TODO(Ohad): write printing functions.
#[derive(Debug)]
pub struct MixedDecommitment<F: Field, H: Hasher> {
    pub hashes: Vec<H::Hash>,
    pub witness_elements: Vec<F>,

    // TODO(Ohad): remove these in non-debug builds.
    pub queried_values: Vec<F>,
    pub structure: MerkleTreeStructure,
}

#[allow(clippy::new_without_default)]
impl<F: Field, H: Hasher> MixedDecommitment<F, H> {
    pub fn new() -> Self {
        Self {
            hashes: vec![],
            witness_elements: vec![],
            queried_values: vec![],
            structure: MerkleTreeStructure::default(),
        }
    }

    pub fn verify(
        &self,
        root: H::Hash,
        queries: &[Vec<usize>],
        mut queried_values: impl Iterator<Item = F>,
    ) -> bool
    where
        F: IntoSlice<H::NativeType>,
    {
        let mut witness_hashes = self.hashes.iter();
        let sorted_queries_by_layer = self.structure.sort_queries_by_layer(queries);

        let mut next_layer_hashes = vec![];
        let mut ancestor_indices = vec![];
        let mut witness_elements = self.witness_elements.iter().copied();
        for i in (1..=self.structure.height()).rev() {
            (next_layer_hashes, ancestor_indices) = Self::verify_single_layer(
                i,
                &sorted_queries_by_layer[i - 1],
                &self.structure,
                ancestor_indices.iter().copied().peekable(),
                queried_values.by_ref(),
                &mut witness_elements,
                &mut witness_hashes,
                next_layer_hashes.into_iter(),
            );
        }
        debug_assert_eq!(next_layer_hashes.len(), 1);
        next_layer_hashes[0] == root
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_single_layer<'a>(
        layer_depth: usize,
        queries_to_layer: &[Vec<usize>],
        structure: &MerkleTreeStructure,
        mut previous_layers_indices: Peekable<impl ExactSizeIterator<Item = usize> + Clone>,
        mut queried_values: impl Iterator<Item = F>,
        mut witness_elements: impl Iterator<Item = F>,
        mut witness_hashes_iter: impl Iterator<Item = &'a H::Hash>,
        mut produced_hashes: impl Iterator<Item = H::Hash>,
    ) -> (Vec<H::Hash>, Vec<usize>)
    where
        F: IntoSlice<H::NativeType>,
    {
        let directly_queried_node_indices =
            queried_nodes_in_layer(queries_to_layer.iter(), structure, layer_depth);
        let mut node_indices = MergeIter::new(
            directly_queried_node_indices.iter().copied(),
            previous_layers_indices.clone().map(|q| q / 2),
        )
        .collect_vec();
        node_indices.dedup();

        // Instead of iterating over every query for every column in the layer, we advance the
        // specific column query-iterator only when it's in the current node.
        let mut column_query_iterators = queries_to_layer
            .iter()
            .map(|column_queries| column_queries.iter().peekable())
            .collect_vec();
        let mut next_layer_hashes = vec![];
        let mut hasher = H::new();
        for &node_index in &node_indices {
            // Push correct child hashes to the hasher.
            match previous_layers_indices.next_if(|hash_index| *hash_index / 2 == node_index) {
                None if layer_depth < structure.height() => {
                    hasher.update(witness_hashes_iter.next().unwrap().as_ref());
                    hasher.update(witness_hashes_iter.next().unwrap().as_ref());
                }
                Some(hash_index) => {
                    if previous_layers_indices
                        .next_if(|&next_h| next_h ^ 1 == hash_index)
                        .is_some()
                    {
                        hasher.update(produced_hashes.next().unwrap().as_ref());
                        hasher.update(produced_hashes.next().unwrap().as_ref());
                    } else {
                        let (left_hash, right_hash) = if hash_index % 2 == 0 {
                            (
                                produced_hashes.next().unwrap(),
                                *witness_hashes_iter.next().unwrap(),
                            )
                        } else {
                            (
                                *witness_hashes_iter.next().unwrap(),
                                produced_hashes.next().unwrap(),
                            )
                        };
                        hasher.update(left_hash.as_ref());
                        hasher.update(right_hash.as_ref());
                    }
                }
                _ => {}
            }

            // Chunk size - according to the column's length and the current depth, we calculate the
            // number of elements from that column 'injected' to the current node.
            for (chunk_size, column_queries) in structure
                .column_lengths_at_depth(layer_depth)
                .iter()
                .map(|&column_length| column_length >> (layer_depth - 1))
                .zip(&mut column_query_iterators)
            {
                for i in 0..chunk_size {
                    let column_chunk_start_index = chunk_size * node_index;
                    match column_queries.next_if(|&&q| q == i + column_chunk_start_index) {
                        Some(_) => hasher.update(F::into_slice(&[queried_values.next().unwrap()])),
                        None => hasher.update(F::into_slice(&[witness_elements.next().unwrap()])),
                    }
                }
            }
            next_layer_hashes.push(hasher.finalize_reset());
        }
        (next_layer_hashes, node_indices)
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_input::MerkleTreeInput;
    use crate::commitment_scheme::mixed_degree_merkle_tree::MixedDegreeMerkleTree;
    use crate::core::fields::m31::M31;

    #[test]
    fn verify_test() {
        const TREE_HEIGHT: usize = 4;
        let mut input = MerkleTreeInput::<M31>::new();
        let column_length_8 = (80..88).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let column_length_4 = (40..44).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        input.insert_column(TREE_HEIGHT, &column_length_8);
        input.insert_column(TREE_HEIGHT - 1, &column_length_4);
        input.insert_column(TREE_HEIGHT - 1, &column_length_8);
        let (tree, commitment) = MixedDegreeMerkleTree::<M31, Blake3Hasher>::commit(&input);
        let queries: Vec<Vec<usize>> = vec![vec![2], vec![0_usize], vec![4, 7]];

        let decommitment = tree.decommit(&input, &queries);
        assert!(decommitment.verify(
            commitment,
            &queries,
            decommitment.queried_values.iter().copied(),
        ));
    }

    #[test]
    #[should_panic]
    fn verify_proof_invalid_commitment_fails_test() {
        const TREE_HEIGHT: usize = 4;
        let mut input = MerkleTreeInput::<M31>::new();
        let column_length_8 = (80..88).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let column_length_4 = (40..44).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        input.insert_column(TREE_HEIGHT, &column_length_8);
        input.insert_column(TREE_HEIGHT - 1, &column_length_4);
        input.insert_column(TREE_HEIGHT - 1, &column_length_8);
        let (tree, _) = MixedDegreeMerkleTree::<M31, Blake3Hasher>::commit(&input);
        let false_commitment = Blake3Hasher::hash(b"false_commitment");

        let queries: Vec<Vec<usize>> = vec![vec![2], vec![0_usize], vec![4, 7]];
        let decommitment = tree.decommit(&input, &queries);

        assert!(decommitment.verify(
            false_commitment,
            &queries,
            decommitment.queried_values.iter().copied(),
        ));
    }

    #[test]
    #[should_panic]
    fn verify_amended_hash_witness_proof_fails_test() {
        const TREE_HEIGHT: usize = 4;
        let mut input = MerkleTreeInput::<M31>::new();
        let column_length_8 = (80..88).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        let column_length_4 = (40..44).map(M31::from_u32_unchecked).collect::<Vec<M31>>();
        input.insert_column(TREE_HEIGHT, &column_length_8);
        input.insert_column(TREE_HEIGHT - 1, &column_length_4);
        input.insert_column(TREE_HEIGHT - 1, &column_length_8);
        let (tree, commitment) = MixedDegreeMerkleTree::<M31, Blake3Hasher>::commit(&input);

        let queries: Vec<Vec<usize>> = vec![vec![2], vec![0_usize], vec![4, 7]];
        let mut decommitment = tree.decommit(&input, &queries);
        decommitment.hashes[0] = Blake3Hasher::hash(b"false_hash");

        assert!(decommitment.verify(
            commitment,
            &queries,
            decommitment.queried_values.iter().copied(),
        ));
    }
}
