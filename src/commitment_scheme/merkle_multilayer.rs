use std::fmt::{self, Display};
use std::iter::Peekable;

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
use super::mixed_degree_decommitment::{DecommitmentNode, MixedDecommitment, PositionInLayer};
use super::utils::{get_column_chunk, inject_and_hash_layer};
use crate::core::fields::{Field, IntoSlice};

/// A MerkleMultiLayer represents multiple sequential merkle-tree layers, as a SubTreeMajor array of
/// hash values. Each SubTree is a balanced binary tree of height `sub_trees_height`.
/// Intended to be used as a layer of L1/L2 cache-sized sub-trees and commited on serially, and
/// multithreaded within the multilayer.
// TODO(Ohad): Implement get_layer_view() and get_layer_mut() for subtrees.
// TODO(Ohad): Implement .commit(), .decommit() for MerkleMultiLayer.
// TODO(Ohad): Add as an attribute of the merkle tree.
// TODO(Ohad): Implement Iterator for MerkleMultiLayer.
pub struct MerkleMultiLayer<H: Hasher> {
    pub data: Vec<H::Hash>,
    config: MerkleMultiLayerConfig,
}

pub struct MerkleMultiLayerConfig {
    pub n_sub_trees: usize,
    pub sub_tree_height: usize,
    pub sub_tree_size: usize,
}

impl MerkleMultiLayerConfig {
    pub fn new(sub_tree_height: usize, n_sub_trees: usize) -> Self {
        let sub_tree_size = (1 << sub_tree_height) - 1;
        Self {
            n_sub_trees,
            sub_tree_height,
            sub_tree_size,
        }
    }
}

impl<H: Hasher> MerkleMultiLayer<H> {
    pub fn new(config: MerkleMultiLayerConfig) -> Self {
        // TODO(Ohad): investigate if this is the best way to initialize the vector. Consider unsafe
        // implementation.
        let data = vec![H::Hash::default(); config.sub_tree_size * config.n_sub_trees];
        Self { data, config }
    }

    /// Returns the roots of the sub-trees.
    pub fn get_roots(&self) -> impl ExactSizeIterator<Item = &H::Hash> {
        self.data
            .iter()
            .skip(self.config.sub_tree_size - 1)
            .step_by(self.config.sub_tree_size)
    }

    pub fn commit_layer<F: Field + Sync + IntoSlice<H::NativeType>, const IS_INTERMEDIATE: bool>(
        &mut self,
        input: &MerkleTreeInput<'_, F>,
        prev_hashes: &[H::Hash],
    ) {
        // TODO(Ohad): implement multithreading (rayon par iter).
        let tree_iter = self.data.chunks_mut(self.config.sub_tree_size);
        tree_iter.enumerate().for_each(|(i, tree_data)| {
            let prev_hashes = if IS_INTERMEDIATE {
                let sub_layer_size = 1 << self.config.sub_tree_height;
                &prev_hashes[i * sub_layer_size..(i + 1) * sub_layer_size]
            } else {
                &[]
            };
            hash_subtree::<F, H, IS_INTERMEDIATE>(tree_data, input, prev_hashes, &self.config, i);
        });
    }

    pub fn generate_decommitment<'a, F: Field>(
        &self,
        input: &MerkleTreeInput<'_, F>,
        multi_layer_depth: usize,
        mut queried_indices: Vec<usize>,
    ) -> MixedDecommitment<F, H>
    where
        H::Hash: 'a,
    {
        // MultiLayer::generate_decommitment gets the input for the entire tree, and needs to know
        // how deep it is in it.
        let tree_height = self.config.sub_tree_height + multi_layer_depth;
        let mut proof_layers = Vec::<Vec<DecommitmentNode<F, H>>>::new();
        let last_included_layer = std::cmp::max(1, multi_layer_depth);
        for i in (last_included_layer..tree_height).rev() {
            let proof_layer =
                self.decommit_intermediate_layer(i, queried_indices.iter().peekable(), input);
            proof_layers.push(proof_layer);
            queried_indices = get_parent_indices(queried_indices);
        }
        MixedDecommitment::<F, H>::new(proof_layers)
    }

    fn decommit_intermediate_layer<'a, F: Field>(
        &self,
        layer_depth: usize,
        mut current_queried_indices: Peekable<impl Iterator<Item = &'a usize>>,
        input: &MerkleTreeInput<'_, F>,
    ) -> Vec<DecommitmentNode<F, H>> {
        let mut proof_layer = Vec::<DecommitmentNode<F, H>>::new();
        while let Some(q) = current_queried_indices.next() {
            let sibling_index = *q ^ 1;
            let hash_witness = match current_queried_indices.peek() {
                // If both children are in the layer, only injected elements are needed
                // to calculate the parent.
                Some(next_q) if **next_q == sibling_index => {
                    current_queried_indices.next();
                    None
                }
                _ => Some(self.get_hash_value(layer_depth, sibling_index)),
            };
            let injected_elements = self.get_injected_elements(input, layer_depth, *q);
            if hash_witness.is_some() || !injected_elements.is_empty() {
                let position_in_layer = PositionInLayer::new_child(sibling_index);
                proof_layer.push(DecommitmentNode {
                    position_in_layer,
                    hash: hash_witness,
                    injected_elements,
                });
            }
        }
        proof_layer
    }

    fn get_injected_elements<F: Field>(
        &self,
        input: &MerkleTreeInput<'_, F>,
        depth: usize,
        query: usize,
    ) -> Vec<F> {
        // TODO(Ohad): Redefine tree height.
        let mut injected_elements = Vec::<F>::new();
        let tree_idx = self.get_containing_tree_idx(depth, query);
        let relative_sack_query = (query % (1 << depth)) / 2;
        for column in input.get_columns(depth).iter() {
            let col_chunk = get_column_chunk(column, tree_idx, self.config.n_sub_trees);
            let chunk = col_chunk
                .chunks(col_chunk.len() >> (depth - 1))
                .nth(relative_sack_query)
                .unwrap();
            injected_elements.extend(chunk);
        }
        injected_elements
    }

    // Consider adding safety checks and making public.
    fn get_hash_value(&self, layer: usize, node_idx: usize) -> H::Hash {
        let layer_len = 1 << layer;
        let tree_idx = node_idx / layer_len;
        let sub_tree_data = self
            .data
            .chunks(self.config.sub_tree_size)
            .nth(tree_idx)
            .unwrap();
        let layer_view = &sub_tree_data[sub_tree_data.len() - (layer_len * 2 - 1)
            ..sub_tree_data.len() - (layer_len * 2 - 1) + layer_len];
        layer_view[node_idx % layer_len]
    }
    fn get_containing_tree_idx(&self, depth: usize, node_idx: usize) -> usize {
        let n_nodes_in_layer = (1 << (depth - 1)) * self.config.n_sub_trees;
        node_idx / n_nodes_in_layer
    }
}

fn get_parent_indices(children: Vec<usize>) -> Vec<usize> {
    let mut parent_indices = children.into_iter().map(|c| c / 2).collect::<Vec<_>>();
    parent_indices.dedup();
    parent_indices
}

// Hashes a single sub-tree.
fn hash_subtree<F: Field, H: Hasher, const IS_INTERMEDIATE: bool>(
    sub_tree_data: &mut [H::Hash],
    input: &MerkleTreeInput<'_, F>,
    prev_hashes: &[H::Hash],
    config: &MerkleMultiLayerConfig,
    index_in_layer: usize,
) where
    F: IntoSlice<H::NativeType>,
{
    // First layer is special, as it is the only layer that might have inputs from the previous
    // MultiLayer, and does not need to look at the current sub_tree for previous hash values.
    let dst = sub_tree_data
        .split_at_mut(1 << (config.sub_tree_height - 1))
        .0;
    inject_and_hash_layer::<H, F, IS_INTERMEDIATE>(
        prev_hashes,
        dst,
        &input
            .get_columns(config.sub_tree_height)
            .iter()
            .map(|c| get_column_chunk(c, index_in_layer, config.n_sub_trees))
            .collect::<Vec<_>>()
            .iter(),
    );

    // Rest of the layers.
    let mut offset_idx = 0;
    for hashed_layer_idx in (1..(config.sub_tree_height)).rev() {
        let hashed_layer_len = 1 << hashed_layer_idx;
        let produced_layer_len = hashed_layer_len / 2;
        let (s1, s2) = sub_tree_data.split_at_mut(offset_idx + hashed_layer_len);
        let (prev_hashes, dst) = (&s1[offset_idx..], &mut s2[..produced_layer_len]);
        offset_idx += hashed_layer_len;

        inject_and_hash_layer::<H, F, true>(
            prev_hashes,
            dst,
            &input
                .get_columns(hashed_layer_idx)
                .iter()
                .map(|c| get_column_chunk(c, index_in_layer, config.n_sub_trees))
                .collect::<Vec<_>>()
                .iter(),
        );
    }
}

// TODO(Ohad): change according to the future implementation of get_layer_view() and
// get_root().
impl<H: Hasher> Display for MerkleMultiLayer<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data
            .chunks(self.config.sub_tree_size)
            .enumerate()
            .for_each(|(i, c)| {
                f.write_str(&std::format!("\nSubTree #[{}]:", i)).unwrap();
                for (i, h) in c.iter().enumerate() {
                    f.write_str(&std::format!("\nNode #[{}]: {}", i, h))
                        .unwrap();
                }
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_input::MerkleTreeInput;
    use crate::commitment_scheme::merkle_multilayer;
    use crate::commitment_scheme::mixed_degree_decommitment::PositionInLayer;
    use crate::core::fields::m31::M31;
    use crate::m31;

    #[test]
    pub fn multi_layer_init_test() {
        let (sub_trees_height, n_sub_trees) = (4, 4);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, n_sub_trees);
        let sub_tree_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        assert_eq!(
            sub_tree_layer.data.len(),
            ((1 << sub_trees_height) - 1) * n_sub_trees
        );
    }

    #[test]
    pub fn multi_layer_display_test() {
        let (sub_trees_height, n_sub_trees) = (8, 8);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, n_sub_trees);
        let multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        println!("{}", multi_layer);
    }

    #[test]
    pub fn get_roots_test() {
        let (sub_trees_height, n_sub_trees) = (4, 4);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, n_sub_trees);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        multi_layer
            .data
            .chunks_mut(multi_layer.config.sub_tree_size)
            .enumerate()
            .for_each(|(i, sub_tree)| {
                sub_tree[sub_tree.len() - 1] = Blake3Hasher::hash(&i.to_le_bytes());
            });

        let roots = multi_layer.get_roots();

        assert_eq!(roots.len(), n_sub_trees);
        roots
            .enumerate()
            .for_each(|(i, r)| assert_eq!(r, &Blake3Hasher::hash(&i.to_le_bytes())));
    }

    fn gen_example_column() -> Vec<M31> {
        let mut trace_column = std::iter::repeat(M31::from_u32_unchecked(1))
            .take(8)
            .collect::<Vec<M31>>();
        trace_column.extend(
            std::iter::repeat(M31::from_u32_unchecked(2))
                .take(8)
                .collect::<Vec<M31>>(),
        );
        trace_column
    }

    fn hash_symmetric_path<H: Hasher>(
        initial_value: &[H::NativeType],
        path_length: usize,
    ) -> H::Hash {
        (1..path_length).fold(H::hash(initial_value), |curr_hash, _| {
            H::concat_and_hash(&curr_hash, &curr_hash)
        })
    }

    fn assert_correct_roots<H: Hasher>(
        initial_value_0: &[H::NativeType],
        initial_value_1: &[H::NativeType],
        path_length: usize,
        roots: &[H::Hash],
    ) {
        let expected_root0 = hash_symmetric_path::<H>(initial_value_0, path_length);
        let expected_root1 = hash_symmetric_path::<H>(initial_value_1, path_length);
        assert_eq!(roots[0], expected_root0);
        assert_eq!(roots[1], expected_root1);
    }

    fn prepare_intermediate_initial_values() -> (Vec<u8>, Vec<u8>) {
        // Column will get spread to one value per leaf.
        let mut leaf_0_input: Vec<u8> = vec![];
        leaf_0_input.extend(Blake3Hasher::hash(b"a").as_ref());
        leaf_0_input.extend(Blake3Hasher::hash(b"a").as_ref());
        leaf_0_input.extend(&u32::to_le_bytes(1));
        let mut leaf_1_input: Vec<u8> = vec![];
        leaf_1_input.extend(Blake3Hasher::hash(b"b").as_ref());
        leaf_1_input.extend(Blake3Hasher::hash(b"b").as_ref());
        leaf_1_input.extend(&u32::to_le_bytes(2));
        (leaf_0_input, leaf_1_input)
    }

    #[test]
    pub fn hash_sub_tree_non_intermediate_test() {
        // trace_column: [M31;16] = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
        let trace_column = gen_example_column();
        let sub_trees_height = 4;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, 2);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);

        merkle_multilayer::hash_subtree::<M31, Blake3Hasher, false>(
            &mut multi_layer.data[..multi_layer.config.sub_tree_size],
            &input,
            &[],
            &multi_layer.config,
            0,
        );
        merkle_multilayer::hash_subtree::<M31, Blake3Hasher, false>(
            &mut multi_layer.data[multi_layer.config.sub_tree_size..],
            &input,
            &[],
            &multi_layer.config,
            1,
        );
        let roots = multi_layer.get_roots();

        assert_correct_roots::<Blake3Hasher>(
            &u32::to_le_bytes(1),
            &u32::to_le_bytes(2),
            sub_trees_height,
            &roots.copied().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn hash_sub_tree_intermediate_test() {
        // trace_column: [M31;16] = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
        let trace_column = gen_example_column();
        let mut prev_hash_values = vec![Blake3Hasher::hash(b"a"); 16];
        prev_hash_values.extend(vec![Blake3Hasher::hash(b"b"); 16]);
        let sub_trees_height = 4;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, 2);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        let (leaf_0_input, leaf_1_input) = prepare_intermediate_initial_values();

        merkle_multilayer::hash_subtree::<M31, Blake3Hasher, true>(
            &mut multi_layer.data[..multi_layer.config.sub_tree_size],
            &input,
            &prev_hash_values[..prev_hash_values.len() / 2],
            &multi_layer.config,
            0,
        );
        merkle_multilayer::hash_subtree::<M31, Blake3Hasher, true>(
            &mut multi_layer.data[multi_layer.config.sub_tree_size..],
            &input,
            &prev_hash_values[prev_hash_values.len() / 2..],
            &multi_layer.config,
            1,
        );
        let roots = multi_layer.get_roots();

        assert_correct_roots::<Blake3Hasher>(
            leaf_0_input.as_slice(),
            leaf_1_input.as_slice(),
            sub_trees_height,
            &roots.copied().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn commit_layer_non_intermediate_test() {
        // trace_column: [M31;16] = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
        let trace_column = gen_example_column();
        let sub_trees_height = 4;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, 2);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);

        multi_layer.commit_layer::<M31, false>(&input, &[]);
        let roots = multi_layer.get_roots();

        assert_correct_roots::<Blake3Hasher>(
            &u32::to_le_bytes(1),
            &u32::to_le_bytes(2),
            sub_trees_height,
            &roots.copied().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn commit_layer_intermediate_test() {
        // trace_column: [M31;16] = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
        let trace_column = gen_example_column();
        let mut prev_hash_values = vec![Blake3Hasher::hash(b"a"); 16];
        prev_hash_values.extend(vec![Blake3Hasher::hash(b"b"); 16]);
        let sub_trees_height = 4;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, 2);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        let (leaf_0_input, leaf_1_input) = prepare_intermediate_initial_values();

        multi_layer.commit_layer::<M31, true>(&input, &prev_hash_values);
        let roots = multi_layer.get_roots();

        assert_correct_roots::<Blake3Hasher>(
            leaf_0_input.as_slice(),
            leaf_1_input.as_slice(),
            sub_trees_height,
            &roots.copied().collect::<Vec<_>>(),
        )
    }

    // TODO(Ohad): Implement 'verify' for decommitment
    #[test]
    fn decommit_layer_test() {
        let trace_column = (0..16).map(M31::from_u32_unchecked).collect::<Vec<_>>();
        let trace_column_rev = trace_column.clone().into_iter().rev().collect::<Vec<M31>>();
        let sub_trees_height = 4;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        input.insert_column(sub_trees_height - 1, trace_column_rev.as_slice());
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, 2);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);
        let prev_hashes = [Blake3Hasher::hash(b"a"); 32];
        let queried_leaves = vec![0, 2, 14, 30];
        let queried_leaf_parents = super::get_parent_indices(queried_leaves);

        multi_layer.commit_layer::<M31, true>(&input, &prev_hashes);
        let decommitment =
            multi_layer.generate_decommitment::<M31>(&input, 0, queried_leaf_parents);

        assert_eq!(decommitment.decommitment_layers.len(), 3);

        // Layer #1. Siblings are mapped to the same parent node.
        assert_eq!(decommitment.decommitment_layers[0].len(), 3);

        // Parents of 0,1 should not contain a hash as they can both be calculated by the verifier.
        assert_eq!(decommitment.decommitment_layers[0][0].hash, None);
        assert_eq!(
            decommitment.decommitment_layers[0][0].injected_elements[0],
            m31!(15)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][0].injected_elements[1],
            m31!(14)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][0].position_in_layer,
            PositionInLayer::Right(1)
        );

        // Parents of 7 should contain the hash of it's sibling.
        let mut hasher = Blake3Hasher::new();
        hasher.update(Blake3Hasher::hash(b"a").as_ref());
        hasher.update(Blake3Hasher::hash(b"a").as_ref());
        hasher.update(6u32.to_le_bytes().as_ref());
        let expected_hash = hasher.finalize();
        assert_eq!(
            decommitment.decommitment_layers[0][1].hash,
            Some(expected_hash)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][1].injected_elements[0],
            m31!(9)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][1].injected_elements[1],
            m31!(8)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][1].position_in_layer,
            PositionInLayer::Left(6)
        );

        // Parents of 15 should contain the hash of it's sibling.
        let mut hasher = Blake3Hasher::new();
        hasher.update(Blake3Hasher::hash(b"a").as_ref());
        hasher.update(Blake3Hasher::hash(b"a").as_ref());
        hasher.update(14u32.to_le_bytes().as_ref());
        let expected_hash = hasher.finalize();
        assert_eq!(
            decommitment.decommitment_layers[0][2].hash,
            Some(expected_hash)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][2].injected_elements[0],
            m31!(1)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][2].injected_elements[1],
            m31!(0)
        );
        assert_eq!(
            decommitment.decommitment_layers[0][2].position_in_layer,
            PositionInLayer::Left(14)
        );

        // Layer #2
        assert_eq!(
            decommitment.decommitment_layers[1][0].position_in_layer,
            PositionInLayer::Right(1)
        );
        assert_eq!(
            decommitment.decommitment_layers[1][1].position_in_layer,
            PositionInLayer::Left(2)
        );
        assert_eq!(
            decommitment.decommitment_layers[1][2].position_in_layer,
            PositionInLayer::Left(6)
        );

        // Layer #3
        assert_eq!(
            decommitment.decommitment_layers[2][0].position_in_layer,
            PositionInLayer::Left(2)
        );
    }
}
