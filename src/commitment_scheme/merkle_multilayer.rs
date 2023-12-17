use std::fmt::{self, Display};

use super::hasher::Hasher;
use super::merkle_input::{LayerColumns, MerkleTreeInput};
use super::utils::{inject_column_chunks, inject_hash_in_pairs};
use crate::core::fields::{Field, IntoSlice};

/// A MerkleMultiLayer represents multiple sequential merkle-tree layers, as a SubTreeMajor array of
/// hash values. Each SubTree is a balanced binary tree of height `sub_trees_height`.
/// Intended to be used as a layer of L1/L2 cache-sized sub-trees and commited on serially, and
/// multithreaded within the multilayer.
// TODO(Ohad): Implement get_layer_view() and get_layer_mut() for subtrees.
// TODO(Ohad): Implement .commit(), .decommit(), .roots() for MerklePolyLayer.
// TODO(Ohad): Add as an attribute of the merkle tree.
pub struct MerkleMultiLayer<H: Hasher> {
    pub data: Vec<H::Hash>,
    config: MerkleMultiLayerConfig,
}

impl<H: Hasher> MerkleMultiLayer<H> {
    pub fn new(config: MerkleMultiLayerConfig) -> Self {
        // TODO(Ohad): investigate if this is the best way to initialize the vector. Consider unsafe
        // implementation.
        let data = vec![H::Hash::default(); config.sub_tree_size * config.n_sub_trees];
        Self { data, config }
    }

    /// Returns the roots of the sub-trees.
    pub fn get_roots(&self) -> Vec<H::Hash> {
        self.data
            .chunks(self.config.sub_tree_size)
            .map(|sub_tree| {
                sub_tree
                    .last()
                    .expect("Tried to extract roots but MerkleMultiLayer is empty!")
                    .to_owned()
            })
            .collect()
    }
}

// Commits on a single sub-tree contained in the MerkleMultiLayer.
// The sub-tree is commited on from the bottom up, and the root is placed in the last index of the
// sub-tree's data. TODO(Ohad): remove '_' after using the function.
fn _commit_subtree<F: Field, H: Hasher, const IS_INTERMEDIATE: bool>(
    sub_tree_data: &mut [H::Hash],
    input: &MerkleTreeInput<'_, F>,
    prev_hashes: Option<&[H::Hash]>,
    cfg: &MerkleMultiLayerConfig,
    index_in_layer: usize,
) where
    F: IntoSlice<H::NativeType>,
{
    // First layer is special, as it is the only layer that might have inputs from the previous
    // MultiLayer, and does not need to look at the current sub_tree for previous hash values.
    let hash_inputs = _prepare_hash_inputs::<F, H, IS_INTERMEDIATE>(
        prev_hashes,
        input.get_columns(cfg.sub_tree_height),
        cfg.sub_tree_height,
        index_in_layer,
        cfg.n_sub_trees,
    );
    let dst = sub_tree_data.split_at_mut(1 << (cfg.sub_tree_height - 1)).0;
    H::hash_many_multi_src_in_place(&hash_inputs, dst);

    // Rest of the layers.
    let mut offset_idx = 0;
    for hashed_layer_idx in (1..(cfg.sub_tree_height)).rev() {
        let hashed_layer_len = 1 << hashed_layer_idx;
        let produced_layer_len = hashed_layer_len / 2;
        let (s1, s2) = sub_tree_data.split_at_mut(offset_idx + hashed_layer_len);
        let (prev_hashes, dst) = (&s1[offset_idx..], &mut s2[..produced_layer_len]);
        offset_idx += hashed_layer_len;

        let hash_inputs = _prepare_hash_inputs::<F, H, true>(
            Some(prev_hashes),
            input.get_columns(hashed_layer_idx),
            hashed_layer_idx,
            index_in_layer,
            cfg.n_sub_trees,
        );
        H::hash_many_multi_src_in_place(&hash_inputs, dst);
    }
}

// TODO(Ohad): remove '_' after using the function.
fn _prepare_hash_inputs<'b, 'a: 'b, F: Field, H: Hasher, const IS_INTERMEDIATE: bool>(
    prev_hashes: Option<&'a [H::Hash]>,
    input_columns: Option<&'a LayerColumns<'_, F>>,
    hashed_layer_depth: usize,
    index_in_layer: usize,
    n_sub_trees: usize,
) -> Vec<Vec<&'b [<H as Hasher>::NativeType]>>
where
    F: IntoSlice<H::NativeType>,
{
    let mut hash_inputs = vec![Vec::new(); 1 << (hashed_layer_depth - 1)];
    if IS_INTERMEDIATE {
        let prev_hashes = unsafe { prev_hashes.unwrap_unchecked() };
        inject_hash_in_pairs::<'_, '_, H>(&mut hash_inputs, prev_hashes);
    }

    if let Some(columns) = input_columns {
        inject_column_chunks::<H, F>(columns, &mut hash_inputs, index_in_layer, n_sub_trees);
    }
    hash_inputs
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

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::commitment_scheme::blake3_hash::Blake3Hasher;
    use crate::commitment_scheme::hasher::Hasher;
    use crate::commitment_scheme::merkle_input::MerkleTreeInput;
    use crate::commitment_scheme::merkle_multilayer::_commit_subtree;
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
        let rand_arr: Vec<u8> = (0..n_sub_trees)
            .map(|_| rand::thread_rng().gen::<u8>())
            .take(n_sub_trees)
            .collect();
        for i in 0..n_sub_trees {
            multi_layer.data[(i + 1) * multi_layer.config.sub_tree_size - 1] =
                Blake3Hasher::hash(&rand_arr[i..i + 1]);
        }

        let roots = multi_layer.get_roots();

        roots
            .iter()
            .enumerate()
            .for_each(|(i, r)| assert_eq!(r, &Blake3Hasher::hash(&rand_arr[i..i + 1])));
        assert_eq!(roots.len(), n_sub_trees);
    }

    #[test]
    pub fn commit_test() {
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

        _commit_subtree::<M31, Blake3Hasher, false>(
            &mut multi_layer.data[..multi_layer.config.sub_tree_size],
            &input,
            None,
            &multi_layer.config,
            0,
        );
        _commit_subtree::<M31, Blake3Hasher, false>(
            &mut multi_layer.data[multi_layer.config.sub_tree_size..],
            &input,
            None,
            &multi_layer.config,
            1,
        );
        let roots = multi_layer.get_roots();

        assert_eq!(hex::encode(roots[0]), hex::encode(expected_root0));
        assert_eq!(hex::encode(roots[1]), hex::encode(expected_root1));
        assert_ne!(roots[0], roots[1]);
    }
}
