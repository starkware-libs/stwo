use std::fmt::{self, Display};

use super::hasher::Hasher;
use super::merkle_input::MerkleTreeInput;
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
}

// Hashes a single sub-tree.
// TODO(Ohad): Remove '_' after using it in the commit function.
fn _hash_subtree<F: Field, H: Hasher, const IS_INTERMEDIATE: bool>(
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
    use crate::core::fields::m31::M31;

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

    #[test]
    pub fn hash_sub_tree_non_intermediate_test() {
        // trace_column: [M31;16] = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
        let mut trace_column = std::iter::repeat(M31::from_u32_unchecked(1))
            .take(8)
            .collect::<Vec<M31>>();
        trace_column.extend(
            std::iter::repeat(M31::from_u32_unchecked(2))
                .take(8)
                .collect::<Vec<M31>>(),
        );
        let sub_trees_height = 4;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, 2);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);

        // Column will get spread to one value per leaf.
        let expected_root0 = (1..sub_trees_height)
            .fold(Blake3Hasher::hash(&u32::to_le_bytes(1)), |curr_hash, _i| {
                Blake3Hasher::concat_and_hash(&curr_hash, &curr_hash)
            });
        let expected_root1 = (1..sub_trees_height)
            .fold(Blake3Hasher::hash(&u32::to_le_bytes(2)), |curr_hash, _i| {
                Blake3Hasher::concat_and_hash(&curr_hash, &curr_hash)
            });

        merkle_multilayer::_hash_subtree::<M31, Blake3Hasher, false>(
            &mut multi_layer.data[..multi_layer.config.sub_tree_size],
            &input,
            &[],
            &multi_layer.config,
            0,
        );
        merkle_multilayer::_hash_subtree::<M31, Blake3Hasher, false>(
            &mut multi_layer.data[multi_layer.config.sub_tree_size..],
            &input,
            &[],
            &multi_layer.config,
            1,
        );
        let mut roots = multi_layer.get_roots();

        assert_eq!(
            hex::encode(roots.next().unwrap()),
            hex::encode(expected_root0)
        );
        assert_eq!(
            hex::encode(roots.next().unwrap()),
            hex::encode(expected_root1)
        );
    }

    #[test]
    fn hash_sub_tree_intermediate_test() {
        // trace_column: [M31;16] = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]
        let mut trace_column = std::iter::repeat(M31::from_u32_unchecked(1))
            .take(8)
            .collect::<Vec<M31>>();
        trace_column.extend(
            std::iter::repeat(M31::from_u32_unchecked(2))
                .take(8)
                .collect::<Vec<M31>>(),
        );
        let mut prev_hash_values = vec![Blake3Hasher::hash(b"a"); 16];
        prev_hash_values.extend(vec![Blake3Hasher::hash(b"b"); 16]);
        let sub_trees_height = 4;
        let mut input = MerkleTreeInput::new();
        input.insert_column(sub_trees_height, &trace_column);
        let config = super::MerkleMultiLayerConfig::new(sub_trees_height, 2);
        let mut multi_layer = super::MerkleMultiLayer::<Blake3Hasher>::new(config);

        // Column will get spread to one value per leaf.
        let mut leaf_0_input: Vec<u8> = vec![];
        leaf_0_input.extend(Blake3Hasher::hash(b"a").as_ref());
        leaf_0_input.extend(Blake3Hasher::hash(b"a").as_ref());
        leaf_0_input.extend(&u32::to_le_bytes(1));
        let mut leaf_1_input: Vec<u8> = vec![];
        leaf_1_input.extend(Blake3Hasher::hash(b"b").as_ref());
        leaf_1_input.extend(Blake3Hasher::hash(b"b").as_ref());
        leaf_1_input.extend(&u32::to_le_bytes(2));

        let expected_root0 = (1..sub_trees_height).fold(
            Blake3Hasher::hash(leaf_0_input.as_slice()),
            |curr_hash, _i| Blake3Hasher::concat_and_hash(&curr_hash, &curr_hash),
        );
        let expected_root1 = (1..sub_trees_height).fold(
            Blake3Hasher::hash(leaf_1_input.as_slice()),
            |curr_hash, _i| Blake3Hasher::concat_and_hash(&curr_hash, &curr_hash),
        );

        merkle_multilayer::_hash_subtree::<M31, Blake3Hasher, true>(
            &mut multi_layer.data[..multi_layer.config.sub_tree_size],
            &input,
            &prev_hash_values[..prev_hash_values.len() / 2],
            &multi_layer.config,
            0,
        );
        merkle_multilayer::_hash_subtree::<M31, Blake3Hasher, true>(
            &mut multi_layer.data[multi_layer.config.sub_tree_size..],
            &input,
            &prev_hash_values[prev_hash_values.len() / 2..],
            &multi_layer.config,
            1,
        );
        let mut roots = multi_layer.get_roots();
        assert_eq!(
            hex::encode(roots.next().unwrap()),
            hex::encode(expected_root0)
        );
        assert_eq!(
            hex::encode(roots.next().unwrap()),
            hex::encode(expected_root1)
        );
    }
}
